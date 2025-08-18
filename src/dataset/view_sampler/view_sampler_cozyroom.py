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
    ) -> tuple[
        Int64[Tensor, " context_view"],
        Int64[Tensor, " target_view"],
    ]:
        num_total_views = 34
        deblurred_indices = [i for i in range(num_total_views) if i % 8 != 0]

        try:
            target_idx_original = int(scene.split("_")[-1])
        except (ValueError, IndexError):
            raise ValueError(f"Scene name '{scene}' is not in the expected format 'cozyroom_view_###'")

        try:
            # .torch 파일 내의 인덱스 (0~28)
            target_idx_in_chunk = deblurred_indices.index(target_idx_original)
        except ValueError:
            raise ValueError(f"Target index {target_idx_original} not found in deblurred list.")

        # 인접 뷰 선택 (리스트 인덱스 기준)
        if target_idx_in_chunk == 0:
            context_indices_in_chunk = [1, 2]
        elif target_idx_in_chunk == len(deblurred_indices) - 1:
            context_indices_in_chunk = [target_idx_in_chunk - 2, target_idx_in_chunk - 1]
        else:
            context_indices_in_chunk = [target_idx_in_chunk - 1, target_idx_in_chunk + 1]

        # 3. 텐서를 생성할 때 device를 명시해줌
        return (
            torch.tensor(context_indices_in_chunk, dtype=torch.int64, device=device),
            torch.tensor([target_idx_in_chunk], dtype=torch.int64, device=device),
        )

    @property
    def num_context_views(self) -> int:
        return 2

    @property
    def num_target_views(self) -> int:
        return 1