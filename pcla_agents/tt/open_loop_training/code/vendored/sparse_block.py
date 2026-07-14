# Vendored from mmdet3d/models/layers/sparse_block.py (Apache-2.0, OpenMMLab).
# PCLA tt port: uses the vendored BasicBlock + spconv 2.x (via spconv_register), no mmdet3d.
# Only SparseBasicBlock + make_sparse_convmodule are kept (SparseEncoder 'basicblock' path).
from typing import Optional, Tuple, Union

from mmcv.cnn import build_conv_layer, build_norm_layer
from torch import nn

from .resnet_blocks import BasicBlock
from .spconv_register import (IS_SPCONV2_AVAILABLE, SparseConvTensor,  # noqa: F401
                              SparseModule, SparseSequential)


def replace_feature(out: SparseConvTensor,
                    new_features: SparseConvTensor) -> SparseConvTensor:
    if 'replace_feature' in out.__dir__():
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out


class SparseBasicBlock(BasicBlock, SparseModule):
    """Sparse basic block implemented with submanifold sparse convolution."""

    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: Union[int, Tuple[int]] = 1,
                 downsample: nn.Module = None,
                 indice_key: Optional[str] = None,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None) -> None:
        SparseModule.__init__(self)
        if conv_cfg is None:
            conv_cfg = dict(type='SubMConv3d')
        conv_cfg.setdefault('indice_key', indice_key)
        if norm_cfg is None:
            norm_cfg = dict(type='BN1d')
        BasicBlock.__init__(
            self,
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        identity = x.features
        assert x.features.dim() == 2, f'x.features.dim()={x.features.dim()}'
        out = self.conv1(x)
        out = replace_feature(out, self.norm1(out.features))
        out = replace_feature(out, self.relu(out.features))
        out = self.conv2(out)
        out = replace_feature(out, self.norm2(out.features))
        if self.downsample is not None:
            identity = self.downsample(x).features
        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))
        return out


def make_sparse_convmodule(in_channels: int,
                           out_channels: int,
                           kernel_size: Union[int, Tuple[int]],
                           indice_key: Optional[str] = None,
                           stride: Union[int, Tuple[int]] = 1,
                           padding: Union[int, Tuple[int]] = 0,
                           conv_type: str = 'SubMConv3d',
                           norm_cfg: Optional[dict] = None,
                           order: Tuple[str] = ('conv', 'norm', 'act'),
                           **kwargs) -> SparseSequential:
    """Make sparse convolution module (SparseSequential of conv/norm/act)."""
    assert isinstance(order, tuple) and len(order) <= 3
    assert set(order) | {'conv', 'norm', 'act'} == {'conv', 'norm', 'act'}

    conv_cfg = dict(type=conv_type, indice_key=indice_key)
    if norm_cfg is None:
        norm_cfg = dict(type='BN1d')

    layers = list()
    for layer in order:
        if layer == 'conv':
            if conv_type not in [
                    'SparseInverseConv3d', 'SparseInverseConv2d',
                    'SparseInverseConv1d'
            ]:
                layers.append(
                    build_conv_layer(
                        conv_cfg,
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False))
            else:
                layers.append(
                    build_conv_layer(
                        conv_cfg,
                        in_channels,
                        out_channels,
                        kernel_size,
                        bias=False))
        elif layer == 'norm':
            layers.append(build_norm_layer(norm_cfg, out_channels)[1])
        elif layer == 'act':
            layers.append(nn.ReLU(inplace=True))

    layers = SparseSequential(*layers)
    return layers
