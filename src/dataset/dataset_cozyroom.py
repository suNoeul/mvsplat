# src/dataset/dataset_cozyroom.py (수정 완료)

import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetCozyroomCfg(DatasetCfgCommon):
    name: Literal["cozyroom"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    test_times_per_scene: int
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True


class DatasetCozyroom(IterableDataset):
    cfg: DatasetCozyroomCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetCozyroomCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            root_chunks = sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)
        if self.stage == "test":
            self.chunks = self.chunks[:: cfg.test_chunk_interval]

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # DEBUG: 1. 메서드 진입 확인
        print("DEBUG: 1. __iter__ 메서드 진입 성공.")

        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.chunks = self.shuffle(self.chunks)

        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        # DEBUG: 2. 청크 루프 시작 확인
        print(f"DEBUG: 2. 총 {len(self.chunks)}개의 청크 파일 처리를 시작합니다.")

        for chunk_path in self.chunks:
            # DEBUG: 3. 청크 파일 로딩 확인
            print(f"DEBUG: 3. 청크 파일 로딩 시도: {chunk_path}")
            chunk = torch.load(chunk_path)
            print(f"DEBUG: 3. 청크 파일 로딩 성공.")

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                chunk = self.shuffle(chunk)

            times_per_scene = self.cfg.test_times_per_scene
            num_runs = int(times_per_scene * len(chunk))
            
            # DEBUG: 4. Scene/Example 루프 시작 확인
            print(f"DEBUG: 4. 청크 내 {len(chunk)}개 Scene에 대해 총 {num_runs}번 루프를 시작합니다.")

            for run_idx in range(num_runs):
                example = chunk[run_idx // times_per_scene]
                
                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                scene = example["key"]

                try:
                    # DEBUG: 5. View Sampler 호출 직전 확인 (가장 유력한 용의자)
                    print(f"DEBUG: 5. 루프 {run_idx+1}/{num_runs} - view_sampler.sample 호출 직전...")
                    context_indices, target_indices = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                    )
                    # DEBUG: 6. View Sampler 반환 성공 확인
                    print(f"DEBUG: 6. 루프 {run_idx+1}/{num_runs} - view_sampler.sample 반환 성공.")
                except ValueError:
                    continue

                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                context_images = [
                    example["images"][index.item()] for index in context_indices
                ]
                
                # DEBUG: 7. 이미지 변환 직전 확인
                print(f"DEBUG: 7. 루프 {run_idx+1}/{num_runs} - context 이미지 변환 시작...")
                context_images = self.convert_images(context_images)
                print(f"DEBUG: 7. 루프 {run_idx+1}/{num_runs} - context 이미지 변환 완료.")

                target_images = [
                    example["images"][index.item()] for index in target_indices
                ]

                print(f"DEBUG: 7. 루프 {run_idx+1}/{num_runs} - target 이미지 변환 시작...")
                target_images = self.convert_images(target_images)
                print(f"DEBUG: 7. 루프 {run_idx+1}/{num_runs} - target 이미지 변환 완료.")

                # ################# BUG FIX #################
                # 하드코딩된 이미지 크기 대신 config 파일의 image_shape 사용
                h, w = self.cfg.image_shape
                expected_shape = (3, h, w)
                context_image_invalid = context_images.shape[2:] != (h, w)
                target_image_invalid = target_images.shape[2:] != (h, w)
                # ###########################################

                if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
                    print(
                        f"Skipped bad example {example['key']}. Context shape was "
                        f"{context_images.shape} and target shape was "
                        f"{target_images.shape}. Expected (B, 3, H, W) with H={h}, W={w}"
                    )
                    continue

                context_extrinsics = extrinsics[context_indices]
                if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
                    a, b = context_extrinsics[:, :3, 3]
                    scale = (a - b).norm()
                    if scale < self.cfg.baseline_epsilon:
                        continue
                    extrinsics[:, :3, 3] /= scale
                else:
                    scale = 1

                nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
                
                # DEBUG: 8. 최종 데이터 생성 및 yield 직전 확인
                print(f"DEBUG: 8. 루프 {run_idx+1}/{num_runs} - 최종 데이터 생성 완료. yield 직전.")

                example = {
                    "context": {
                        "extrinsics": extrinsics[context_indices],
                        "intrinsics": intrinsics[context_indices],
                        "image": context_images,
                        "near": self.get_bound("near", len(context_indices)) / nf_scale,
                        "far": self.get_bound("far", len(context_indices)) / nf_scale,
                        "index": context_indices,
                    },
                    "target": {
                        "extrinsics": extrinsics[target_indices],
                        "intrinsics": intrinsics[target_indices],
                        "image": target_images,
                        "near": self.get_bound("near", len(target_indices)) / nf_scale,
                        "far": self.get_bound("far", len(target_indices)) / nf_scale,
                        "index": target_indices,
                    },
                    "scene": scene,
                }
                if self.stage == "train" and self.cfg.augment:
                    example = apply_augmentation_shim(example)
                yield apply_crop_shim(example, tuple(self.cfg.image_shape))

    # ... (이하 나머지 메서드는 동일) ...
    
    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],
        Float[Tensor, "batch 3 3"],
    ]:
        b, _ = poses.shape
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}
                assert not (set(merged_index.keys()) & set(index.keys()))
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        return (
            min(len(self.index.keys()) *
                self.cfg.test_times_per_scene, self.cfg.test_len)
            if self.stage == "test" and self.cfg.test_len > 0
            else len(self.index.keys()) * self.cfg.test_times_per_scene
        )