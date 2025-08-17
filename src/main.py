import os 
from pathlib import Path
import warnings

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.wandb import WandbLogger

# ------------------------------------------------------------------------------------
# beartype/jaxtyping로 타입 검사/주석을 깔끔히 쓰기 위한 import hook 설정
# ("src" 패키지 안에서만 유효)
# ------------------------------------------------------------------------------------
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    # ---- 프로젝트 내부 핵심 모듈들 ----
    from src.config import load_typed_root_config      # Hydra Dict → 내부 Typed Config 변환
    from src.dataset.data_module import DataModule     # PL LightningDataModule 구현
    from src.global_cfg import set_cfg                 # 전역 cfg 저장(다른 모듈에서 접근)
    from src.loss import get_losses                    # 손실 구성 생성
    from src.misc.LocalLogger import LocalLogger       # WandB 미사용 시 대체 로거
    from src.misc.step_tracker import StepTracker      # step을 DataLoader에 공유하기 위한 도우미
    from src.misc.wandb_tools import update_checkpoint_path  # ckpt 경로 정규화(artifact 지원)
    from src.model.decoder import get_decoder          # 디코더 팩토리
    from src.model.encoder import get_encoder          # 인코더(및 비주얼라이저) 팩토리
    from src.model.model_wrapper import ModelWrapper   # LightningModule 래퍼(학습/테스트 로직)

def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

# ------------------------------------------------------------------------------------
# Hydra 진입점: config_path="../config", config_name="main"
#   - 실행 시 "config/main.yaml"을 루트로 병합(overrides 포함)
#   - cfg_dict: OmegaConf DictConfig (중첩 구조)
# ------------------------------------------------------------------------------------
@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    # 1) DictConfig → 내부에서 쓰는 typed config로 변환
    cfg = load_typed_root_config(cfg_dict)
    # 2) 전역에도 원본 cfg_dict를 저장(다른 모듈에서 참조 가능)
    set_cfg(cfg_dict)

    # --------------------------------------------------------------------------------
    # 출력 폴더 결정
    #  - Hydra가 자동으로 run 디렉토리를 만들어줌(ex: outputs/2025-08-16/12-34-56)
    #  - cfg_dict.output_dir가 지정되면 그대로 사용(재실행/재개 용이)
    # --------------------------------------------------------------------------------
    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    print(cyan(f"Saving outputs to {output_dir}."))

    # 편의를 위해 상위에 latest-run 심볼릭 링크 갱신
    latest_run = output_dir.parents[1] / "latest-run"
    os.system(f"rm {latest_run}")
    os.system(f"ln -s {output_dir} {latest_run}")

    # --------------------------------------------------------------------------------
    # 로깅 설정 (WandB / Local)
    #  - cfg_dict.wandb.mode != "disabled"면 WandB 사용
    #  - wandb.id가 있으면 중단 후 재개(resume) 모드
    # --------------------------------------------------------------------------------
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        wandb_extra_kwargs = {}
        if cfg_dict.wandb.id is not None:
            wandb_extra_kwargs.update({'id': cfg_dict.wandb.id,
                                       'resume': "must"})
        logger = WandbLogger(
            entity=cfg_dict.wandb.entity,
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            **wandb_extra_kwargs,
        )
        # 학습 중 step별 lr 로깅
        callbacks.append(LearningRateMonitor("step", True))

        # rank==0에서만 코드 스냅샷 업로드
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()  # 터미널/파일 로깅 대체

    # --------------------------------------------------------------------------------
    # 체크포인트 콜백
    #  - monitor="info/global_step" 기준으로 최근 k개 저장
    #  - 경로: <output_dir>/checkpoints/...
    # --------------------------------------------------------------------------------
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            monitor="info/global_step",
            mode="max",  # 최근 step이 큰 것이 최신이므로 max
        )
    )
    # 파일명에 '/' 대신 '_' 쓰도록(일부 플랫폼 호환)
    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = '_'

    # --------------------------------------------------------------------------------
    # 체크포인트 로드 경로 정리
    #  - wandb artifact 경로/로컬 경로 모두 지원
    #  - 재학습/테스트 시 "어디서 로드할지" 최종 문자열 반환
    # --------------------------------------------------------------------------------
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # --------------------------------------------------------------------------------
    # DataLoader 프로세스들과 현재 step을 공유(예: curriculum/뷰 샘플링과 동기화)
    # --------------------------------------------------------------------------------
    step_tracker = StepTracker()

    # --------------------------------------------------------------------------------
    # PyTorch Lightning Trainer 구성
    #  - accelerator="gpu", devices="auto": 가용 GPU 자동 사용
    #  - 다GPU면 DDP, 1GPU면 단일프로세스
    #  - val_check_interval/test 관련 빈도/스텝 제한/gradient clip 등 설정 반영
    # --------------------------------------------------------------------------------
    trainer = Trainer(
        max_epochs=-1,                           # step 기반 제어(max_steps) 사용
        accelerator="gpu",
        logger=logger,
        devices="auto",
        num_nodes=cfg.trainer.num_nodes,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.mode == "test",  # test 모드일 때만 progress bar
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
    )
    # 랜덤시드 고정(멀티랭크 고려)
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    # --------------------------------------------------------------------------------
    # Encoder/Visualizer, Decoder, Losses, ModelWrapper 구성
    #  - encoder_visualizer: 테스트 시 결과(이미지/PLY 등) 저장을 담당
    # --------------------------------------------------------------------------------
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    model_kwargs = {
        "optimizer_cfg": cfg.optimizer,
        "test_cfg": cfg.test,
        "train_cfg": cfg.train,
        "encoder": encoder,
        "encoder_visualizer": encoder_visualizer,
        "decoder": get_decoder(cfg.model.decoder, cfg.dataset),
        "losses": get_losses(cfg.loss),
        "step_tracker": step_tracker,
    }

    # --------------------------------------------------------------------------------
    # 모델 로드 로직
    #  - mode=train & load 지정 & resume=False → weights만 로드(옵티마상태 제외)
    #  - 그 외 → 새 모델 생성 또는 ckpt에서 이어서
    # --------------------------------------------------------------------------------
    if cfg.mode == "train" and checkpoint_path is not None and not cfg.checkpointing.resume:
        model_wrapper = ModelWrapper.load_from_checkpoint(
            checkpoint_path, **model_kwargs, strict=True)
        print(cyan(f"Loaded weigths from {checkpoint_path}."))
    else:
        model_wrapper = ModelWrapper(**model_kwargs)

    # --------------------------------------------------------------------------------
    # DataModule 구성
    #  - cfg.dataset (예: config/dataset/re10k.yaml 병합 결과)
    #  - cfg.data_loader (배치/워커/셔플 등)
    #  - step_tracker/global_rank 전달
    #  - 내부에서 train/val/test DataLoader를 생성
    # --------------------------------------------------------------------------------
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    # --------------------------------------------------------------------------------
    # 모드 분기: 학습 vs 테스트
    #  - 학습: trainer.fit(...)
    #  - 테스트: trainer.test(...) → 이때 encoder_visualizer가 결과 저장(이미지/포인트/메타)
    # --------------------------------------------------------------------------------
    if cfg.mode == "train":
        trainer.fit(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=(checkpoint_path if cfg.checkpointing.resume else None)
        )
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )

# 표준 main 가드
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # matmul 정밀도 설정(성능/정밀도 트레이드오프)
    torch.set_float32_matmul_precision('high')

    train()
