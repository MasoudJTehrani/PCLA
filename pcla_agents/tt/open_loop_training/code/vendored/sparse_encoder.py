# Vendored from mmdet3d/models/middle_encoders/sparse_encoder.py (Apache-2.0, OpenMMLab).
# PCLA tt port: drops the auxiliary imports (mmcv.ops points_in_boxes/three_nn/
# three_interpolate, mmdet.losses, mmdet3d.structures) that are only used by SASSD/aux
# losses, not by the forward path. Uses vendored sparse_block + spconv 2.x. Registered into
# TT_MODELS; the model actually instantiates the SparseEncoder_fp32 subclass (lidarnet.py).
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch import nn as nn

from ..tt_compat import TT_MODELS as MODELS
from .sparse_block import SparseBasicBlock, make_sparse_convmodule
from .spconv_register import (IS_SPCONV2_AVAILABLE, SparseConvTensor,  # noqa: F401
                              SparseSequential)

TwoTupleIntType = Tuple[Tuple[int]]


@MODELS.register_module()
class SparseEncoder(nn.Module):
    r"""Sparse encoder for SECOND and Part-A2 (see mmdet3d sparse_encoder.py)."""

    def __init__(
            self,
            in_channels: int,
            sparse_shape: List[int],
            order: Optional[Tuple[str]] = ('conv', 'norm', 'act'),
            norm_cfg: Optional[dict] = dict(
                type='BN1d', eps=1e-3, momentum=0.01),
            base_channels: Optional[int] = 16,
            output_channels: Optional[int] = 128,
            encoder_channels: Optional[TwoTupleIntType] = ((16, ), (32, 32, 32),
                                                           (64, 64, 64),
                                                           (64, 64, 64)),
            encoder_paddings: Optional[TwoTupleIntType] = ((1, ), (1, 1, 1),
                                                           (1, 1, 1),
                                                           ((0, 1, 1), 1, 1)),
            block_type: Optional[str] = 'conv_module',
            return_middle_feats: Optional[bool] = False):
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.return_middle_feats = return_middle_feats

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        if self.order[0] != 'conv':  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels, self.base_channels, 3, norm_cfg=norm_cfg,
                padding=1, indice_key='subm1', conv_type='SubMConv3d',
                order=('conv', ))
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels, self.base_channels, 3, norm_cfg=norm_cfg,
                padding=1, indice_key='subm1', conv_type='SubMConv3d')

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule, norm_cfg, self.base_channels,
            block_type=block_type)

        self.conv_out = make_sparse_convmodule(
            encoder_out_channels, self.output_channels, kernel_size=(3, 1, 1),
            stride=(2, 1, 1), norm_cfg=norm_cfg, padding=0,
            indice_key='spconv_down2', conv_type='SparseConv3d')

    @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, voxel_features: Tensor, coors: Tensor,
                batch_size: int) -> Union[Tensor, Tuple[Tensor, list]]:
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

        if self.return_middle_feats:
            return spatial_features, encode_features
        else:
            return spatial_features

    def make_encoder_layers(
        self,
        make_block,
        norm_cfg: Dict,
        in_channels: int,
        block_type: Optional[str] = 'conv_module',
        conv_cfg: Optional[dict] = dict(type='SubMConv3d')
    ) -> int:
        assert block_type in ['conv_module', 'basicblock']
        self.encoder_layers = SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                if i != 0 and j == 0 and block_type == 'conv_module':
                    blocks_list.append(
                        make_block(in_channels, out_channels, 3,
                                   norm_cfg=norm_cfg, stride=2, padding=padding,
                                   indice_key=f'spconv{i + 1}',
                                   conv_type='SparseConv3d'))
                elif block_type == 'basicblock':
                    if j == len(blocks) - 1 and i != len(
                            self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(in_channels, out_channels, 3,
                                       norm_cfg=norm_cfg, stride=2,
                                       padding=padding,
                                       indice_key=f'spconv{i + 1}',
                                       conv_type='SparseConv3d'))
                    else:
                        blocks_list.append(
                            SparseBasicBlock(out_channels, out_channels,
                                             norm_cfg=norm_cfg,
                                             conv_cfg=conv_cfg))
                else:
                    blocks_list.append(
                        make_block(in_channels, out_channels, 3,
                                   norm_cfg=norm_cfg, padding=padding,
                                   indice_key=f'subm{i + 1}',
                                   conv_type='SubMConv3d'))
                in_channels = out_channels
            stage_name = f'encoder_layer{i + 1}'
            stage_layers = SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels
