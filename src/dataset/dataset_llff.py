# src/dataset/dataset_llff.py
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
import torchvision.transforms as tf
from torch.utils.data import IterableDataset
from ..geometry.projection import get_fov  # ★ FOV 필터에 사용

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler, ViewSamplerCfg
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim


@dataclass
class DatasetLLFFCfg(DatasetCfgCommon):
    name: Literal["cozyroom"]
    roots: list[Path]
    rf_subdir: str
    images_subdir: str
    poses_file: str
    hold_k: int
    
    # DatasetCfgCommon에서 필요한 필드들
    image_shape: list[int] | None
    background_color: list[float]
    cameras_are_circular: bool
    overfit_to_scene: str | None
    view_sampler: ViewSamplerCfg
    
    # LLFF 전용 필드들
    augment: bool
    skip_bad_shape: bool
    max_fov: float
    make_baseline_1: bool
    baseline_epsilon: float
    baseline_scale_bounds: bool
    near: float | int
    far: float | int
    
    # 추가 필드들
    shuffle_val: bool
    test_len: int
    test_chunk_interval: int
    test_times_per_scene: int

class DatasetLLFF(IterableDataset):
    cfg: DatasetLLFFCfg
    stage: Stage
    view_sampler: ViewSampler
    to_tensor: tf.ToTensor

    def __init__(self, cfg: DatasetLLFFCfg, stage: Stage, view_sampler: ViewSampler) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        # 루트/경로
        self.root = Path(self.cfg.roots[0])
        self.rf_dir = self.root / self.cfg.rf_subdir
        self.images_dir = self.rf_dir / self.cfg.images_subdir
        self.poses_path = self.rf_dir / self.cfg.poses_file

        assert self.images_dir.is_dir(), f"Missing images dir: {self.images_dir}"
        assert self.poses_path.is_file(), f"Missing poses file: {self.poses_path}"

        # 파일명 숫자 기준 정렬 (00000.png …)
        self.all_frames = sorted(self.images_dir.glob("*.png"), key=lambda p: int(p.stem))
        self.all_ids = [int(p.stem) for p in self.all_frames]

        # hold 분할
        if self.stage == "test":
            keep_ids = [i for i in self.all_ids if i % self.cfg.hold_k == 0]
        elif self.stage in ("val",):
            keep_ids = [i for i in self.all_ids if i % self.cfg.hold_k == 0]
        else:
            keep_ids = [i for i in self.all_ids if i % self.cfg.hold_k != 0]

        id2idx = {i: idx for idx, i in enumerate(self.all_ids)}
        self.frames = [self.all_frames[id2idx[i]] for i in keep_ids]
        self.ids = keep_ids

        # LLFF 포즈/내외부 파라미터 로드
        self.H_llff, self.W_llff, self.K_all, self.c2w_all = self._load_llff_poses_bounds(self.poses_path)
        sel = [id2idx[i] for i in keep_ids]
        self.K_all = self.K_all[sel]       # (M,3,3)
        self.c2w_all = self.c2w_all[sel]   # (M,4,4)

        # 리사이즈 목표
        self.resize_to = None
        if self.cfg.image_shape is not None:
            H, W = int(self.cfg.image_shape[0]), int(self.cfg.image_shape[1])
            self.resize_to = (H, W)

        # near/far: -1 → 기본값(0.1/1000)로 치환
        self.near = 0.1 if self.cfg.near == -1 else float(self.cfg.near)
        self.far  = 1000.0 if self.cfg.far == -1 else float(self.cfg.far)

    def __len__(self):
        # scene 하나 기준으로 프레임 수 반환(PL은 IterableDataset라 len을 엄격히 쓰진 않음)
        return len(self.ids)

    def __iter__(self):
        # 한 scene 안에서 샘플링
        scene_key = self.root.name  # 예: 'cozyroom'
        extrinsics = torch.from_numpy(self.c2w_all).float()     # (M,4,4) c2w
        intrinsics = torch.from_numpy(self.K_all).float()       # (M,3,3)

         # 뷰 인덱스 샘플링
        try:
            context_indices, target_indices = self.view_sampler.sample(
                scene_key, extrinsics, intrinsics
            )
        except ValueError as e:
            return  # 충분한 프레임이 없으면 skip

        # ★ FOV 필터: re10k와 동일한 기준 유지
        if self.cfg.max_fov is not None:
            fov_degrees = get_fov(intrinsics).rad2deg()
            if (fov_degrees > float(self.cfg.max_fov)).any():
                return

        # 이미지 로드 함수
        def _load_imgs(indices: torch.Tensor) -> torch.Tensor:
            imgs = []
            for idx in indices.tolist():
                p = self.frames[idx]
                im = cv2.imread(str(p), cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR->RGB
                if self.resize_to is not None:
                    H, W = self.resize_to
                    im = cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC)
                im = (im.astype(np.float32) / 255.0)                   # 0~1
                imgs.append(torch.from_numpy(im).permute(2, 0, 1))     # 3xHxW
            return torch.stack(imgs, dim=0)  # (V,3,H,W)

        context_images = _load_imgs(context_indices)
        target_images  = _load_imgs(target_indices)

        # 고정된 해상도 검증
        if self.cfg.skip_bad_shape:
            if self.resize_to is not None:
                H, W = self.resize_to
            else:
                H, W = context_images.shape[-2], context_images.shape[-1]
            if (context_images.shape[1:] != (3, H, W)) or (target_images.shape[1:] != (3, H, W)):
                return

        # ★ baseline=1 정규화(+ near/far 스케일) : re10k와 동작 일치
        nf_scale = 1.0
        if self.cfg.make_baseline_1 and context_indices.numel() == 2:
            a = extrinsics[context_indices[0], :3, 3]
            b = extrinsics[context_indices[1], :3, 3]
            scale = (a - b).norm()
            if scale < float(self.cfg.baseline_epsilon):
                return  # insufficient baseline → skip
            extrinsics[:, :3, 3] /= scale
            nf_scale = float(scale) if self.cfg.baseline_scale_bounds else 1.0

        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "near": torch.full((len(context_indices),), self.near / nf_scale, dtype=torch.float32),
                "far":  torch.full((len(context_indices),), self.far  / nf_scale, dtype=torch.float32),
                "index": context_indices,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "near": torch.full((len(target_indices),), self.near / nf_scale, dtype=torch.float32),
                "far":  torch.full((len(target_indices),), self.far  / nf_scale, dtype=torch.float32),
                "index": target_indices,
            },
            "scene": scene_key,
        }
        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)

        # re10k와 동일하게 crop/resize shim 적용
        return_iter = apply_crop_shim(example, tuple(self.cfg.image_shape) if self.cfg.image_shape else None)
        yield return_iter

    # ---------- LLFF poses_bounds.npy 파서 ----------
    def _load_llff_poses_bounds(self, path: Path):
        pb = np.load(str(path))  # (N, 17) 등: 앞 15개 pose, 마지막 2개 near/far
        poses = pb[:, :15].reshape(-1, 3, 5)     # (N,3,5)
        # bds = pb[:, -2:]  # [near, far] (원한다면 사용할 수 있음)

        H = poses[0, 0, -1]; W = poses[0, 1, -1]; focal = poses[0, 2, -1]
        c2w_3x4 = poses[:, :, :4]                # (N,3,4)
        c2w = np.tile(np.eye(4, dtype=np.float32)[None, ...], (c2w_3x4.shape[0], 1, 1))
        c2w[:, :3, :4] = c2w_3x4                 # (N,4,4)

        K = np.array([[focal, 0.0, W * 0.5],
                      [0.0,  focal, H * 0.5],
                      [0.0,  0.0,   1.0]], dtype=np.float32)
        Ks = np.tile(K[None, ...], (c2w.shape[0], 1, 1)).astype(np.float32)
        return int(H), int(W), Ks, c2w.astype(np.float32)
