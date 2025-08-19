from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from ...misc.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler



@dataclass
class ViewSamplerCozyroomCfg:
    name: Literal["cozyroom_eval"]
    

class ViewSamplerCozyroom(ViewSampler[ViewSamplerCozyroomCfg]):
    
    def __init__(
        self,
        cfg: ViewSamplerCozyroomCfg,
        stage: Stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__(cfg, stage, is_overfitting, cameras_are_circular, step_tracker)

    # 2. sample 메서드 시그니처를 device 인자를 포함하도록 수정
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[Int64[Tensor, " context_view"], Int64[Tensor, " target_view"]]:
        # ---- 디버깅: 진입 로그 ----
        N = int(extrinsics.shape[0])
        # print(f"[SAMPLE enter] scene={scene} N={N}", flush=True)

        if N < 2:
            raise ValueError("Not enough views")
        if N == 2:
            # 폴백: 1장을 컨텍스트, 1장을 타깃
            ctx = torch.tensor([0], dtype=torch.int64, device=device)
            tgt = torch.tensor([1], dtype=torch.int64, device=device)
            print(f"[SAMPLE fallback-2] ctx={ctx.tolist()} tgt={tgt.tolist()}", flush=True)
            return ctx, tgt

        # 타깃 선택 규칙:
        # - test에서 한 번만 뽑는다면 가운데 프레임을 타깃으로(안전)
        # - 여러 번 뽑고 싶으면 step 기반으로 순환
        if self.step_tracker is not None:
            step = int(self.step_tracker.step)
            # 1..N-2 범위에서 순환 (양 옆 이웃 확보용)
            t = (step % (N - 2)) + 1
        else:
            t = N // 2  # 가운데

        ctx_idx = [t - 1, t + 1]
        ctx = torch.tensor(ctx_idx, dtype=torch.int64, device=device)
        tgt = torch.tensor([t], dtype=torch.int64, device=device)

        # print(f"[SAMPLE ok] t={t} ctx={ctx.tolist()} tgt={tgt.tolist()}", flush=True)
        return ctx, tgt

    @property
    def num_context_views(self) -> int:
        return 2

    @property
    def num_target_views(self) -> int:
        return 1