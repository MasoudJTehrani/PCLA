# Vendored HardSimpleVFE from mmdet3d/models/voxel_encoders/voxel_encoder.py (Apache-2.0).
# Parameter-free: averages the points inside each voxel. Pure torch.
import torch
from torch import Tensor, nn

from ..tt_compat import TT_MODELS as MODELS


@MODELS.register_module()
class HardSimpleVFE(nn.Module):
    """Simple voxel feature encoder used in SECOND. Averages point values per voxel."""

    def __init__(self, num_features: int = 4) -> None:
        super(HardSimpleVFE, self).__init__()
        self.num_features = num_features

    def forward(self, features: Tensor, num_points: Tensor, coors: Tensor,
                *args, **kwargs) -> Tensor:
        points_mean = features[:, :, :self.num_features].sum(
            dim=1, keepdim=False) / num_points.type_as(features).view(-1, 1)
        return points_mean.contiguous()
