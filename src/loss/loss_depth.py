from dataclasses import dataclass

import torch
from einops import reduce
from jaxtyping import Float
from torch import Tensor
from typing import Optional

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossDepthCfg:
    weight: float
    sigma_image: Optional[float]
    use_second_derivative: bool


@dataclass
class LossDepthCfgWrapper:
    depth: LossDepthCfg


class LossDepth(Loss[LossDepthCfg, LossDepthCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # Scale the depth between the near and far planes.
        near = batch["target"]["near"][..., None, None].log()
        far = batch["target"]["far"][..., None, None].log()
        depth = prediction.depth.min(far).max(near)
        depth = (depth - near) / (far - near)

        # Compute the difference between neighboring pixels in each direction.
        depth_dx = depth.diff(dim=-1)
        depth_dy = depth.diff(dim=-2)

        # If desired, compute a 2nd derivative.
        if self.cfg.use_second_derivative:
            depth_dx = depth_dx.diff(dim=-1)
            depth_dy = depth_dy.diff(dim=-2)

        # If desired, add bilateral filtering.
        if self.cfg.sigma_image is not None:
            color_gt = batch["target"]["image"]
            color_dx = reduce(color_gt.diff(dim=-1), "b v c h w -> b v h w", "max")
            color_dy = reduce(color_gt.diff(dim=-2), "b v c h w -> b v h w", "max")
            if self.cfg.use_second_derivative:
                color_dx = torch.max(color_dx[..., :, 1:], color_dx[..., :, :-1])
                color_dy = torch.max(color_dy[..., 1:, :], color_dy[..., :-1, :])
            depth_dx = depth_dx * torch.exp(torch.clamp(-color_dx * self.cfg.sigma_image, -10, 0))
            depth_dy = depth_dy * torch.exp(torch.clamp(-color_dy * self.cfg.sigma_image, -10, 0))

        return self.cfg.weight * (depth_dx.abs().mean() + depth_dy.abs().mean())
