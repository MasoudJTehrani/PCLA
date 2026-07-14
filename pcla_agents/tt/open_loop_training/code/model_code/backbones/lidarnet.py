# PCLA tt port of lidarnet.py.
# Original subclassed mmdet3d's MVXTwoStageDetector and used the mmcv.ops Voxelization CUDA
# op. Here LidarNet is a plain BaseModule that builds its submodules through the local
# registry and voxelizes with spconv's PointToVoxel (ships precompiled, no nvcc). Submodule
# names (pts_voxel_encoder / pts_middle_encoder / pts_backbone / pts_neck) are preserved so
# the checkpoint keys (lidar_encoder.*) load unchanged.
import torch
import torch.nn as nn
import torch.nn.functional as F

from open_loop_training.code.tt_compat import (
    TT_MODELS, BaseModule, build_backbone, build_neck, build_middle_encoder,
    build_voxel_encoder, force_fp32, auto_fp16)
from open_loop_training.code.vendored.sparse_encoder import SparseEncoder
from open_loop_training.code.vendored.spconv_register import SparseConvTensor


# To avoid a strange FP16 inf-norm bug (kept from the original; forward is fp32).
@TT_MODELS.register_module()
class SparseEncoder_fp32(SparseEncoder):

    @force_fp32()
    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.sparse_shape, batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        return spatial_features


@TT_MODELS.register_module()
class LidarNet(BaseModule):
    def __init__(self,
                 bev_h=None,
                 bev_w=None,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 pts_backbone=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pts_voxel_cfg = dict(pts_voxel_layer)
        self.pts_voxel_encoder = build_voxel_encoder(pts_voxel_encoder)
        self.pts_middle_encoder = build_middle_encoder(pts_middle_encoder)
        self.pts_backbone = build_backbone(pts_backbone) if pts_backbone else None
        self.pts_neck = build_neck(pts_neck) if pts_neck else None
        # spconv voxel grid must line up with the middle encoder's sparse_shape (z, y, x);
        # derive the coords range from it so every voxel index stays inside the grid.
        self.sparse_shape = list(self.pts_middle_encoder.sparse_shape)  # [z, y, x]
        self._voxel_generators = {}  # keyed by (device, num_features)
        self.fp16_enabled = False

    @property
    def with_pts_backbone(self):
        return self.pts_backbone is not None

    @property
    def with_pts_neck(self):
        return self.pts_neck is not None

    def _get_voxel_generator(self, device, num_features):
        key = (str(device), num_features)
        gen = self._voxel_generators.get(key)
        if gen is None:
            from spconv.pytorch.utils import PointToVoxel
            cfg = self.pts_voxel_cfg
            vsize = list(cfg['voxel_size'])  # [x, y, z]
            pc_min = list(cfg['point_cloud_range'])[:3]  # [x_min, y_min, z_min]
            z_sh, y_sh, x_sh = self.sparse_shape  # sparse_shape is (z, y, x)
            coors_range = [
                pc_min[0], pc_min[1], pc_min[2],
                pc_min[0] + x_sh * vsize[0],
                pc_min[1] + y_sh * vsize[1],
                pc_min[2] + z_sh * vsize[2],
            ]
            max_voxels = cfg['max_voxels']
            max_v = max_voxels[0] if self.training else max_voxels[1]
            gen = PointToVoxel(
                vsize_xyz=vsize,
                coors_range_xyz=coors_range,
                num_point_features=num_features,
                max_num_voxels=int(max_v),
                max_num_points_per_voxel=int(cfg['max_num_points']),
                device=device)
            self._voxel_generators[key] = gen
        return gen

    @torch.no_grad()
    def voxelize(self, points):
        """points: [B, N, C] tensor or list of [N, C]. Returns hard-voxelization
        (voxels, num_points_per_voxel, coors) with coors as (batch_idx, z, y, x)."""
        if torch.is_tensor(points) and points.dim() == 3:
            points = [points[i] for i in range(points.shape[0])]
        elif torch.is_tensor(points):
            points = [points]

        voxels_list, coors_list, num_list = [], [], []
        for i, pc in enumerate(points):
            pc = pc.contiguous()
            gen = self._get_voxel_generator(pc.device, pc.shape[-1])
            voxels, coords, num_per = gen(pc)  # coords: [M, 3] as (z, y, x)
            coors_batch = F.pad(coords, (1, 0), mode='constant', value=i)
            voxels_list.append(voxels)
            coors_list.append(coors_batch)
            num_list.append(num_per)
        voxels = torch.cat(voxels_list, dim=0)
        num_points = torch.cat(num_list, dim=0)
        coors = torch.cat(coors_list, dim=0)
        return voxels, num_points, coors

    @auto_fp16()
    def forward(self, pts):
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = int(coors[-1, 0].item()) + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        if self.with_pts_backbone:
            x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x
