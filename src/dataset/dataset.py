from dataclasses import dataclass
from typing import List, Optional
from .view_sampler import ViewSamplerCfg


@dataclass
class DatasetCfgCommon:
    image_shape: List[int]
    background_color: List[float]
    cameras_are_circular: bool
    overfit_to_scene: Optional[str]
    view_sampler: ViewSamplerCfg
