from torch.utils.data import Dataset

from ..misc.step_tracker import StepTracker
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from .dataset_llff import DatasetLLFF, DatasetLLFFCfg  # NEW    
from .types import Stage
from .view_sampler import get_view_sampler



DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
    "cozyroom": DatasetLLFF,
}


DatasetCfg = DatasetRE10kCfg | DatasetLLFFCfg

def get_dataset(
    cfg: DatasetCfg,
    stage: Stage,
    step_tracker: StepTracker | None,
) -> Dataset:
    # cfg.name이 DATASETS에 있는지 확인
    if cfg.name not in DATASETS:
        raise ValueError(f"Unknown dataset: {cfg.name}. Available: {list(DATASETS.keys())}")

    view_sampler = get_view_sampler(
        cfg.view_sampler,
        stage,
        cfg.overfit_to_scene is not None,
        cfg.cameras_are_circular,
        step_tracker,
    )
    return DATASETS[cfg.name](cfg, stage, view_sampler)
