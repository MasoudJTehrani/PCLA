# Deformable Convolution v1 ('DCN') backed by torchvision.ops.deform_conv2d (ships
# precompiled, no nvcc), replacing mmcv.ops' CUDA DeformConv2dPack. Registered as 'DCN' in
# mmengine's MODELS so build_conv_layer(dict(type='DCN', ...)) resolves. Parameter names
# (weight / conv_offset.weight / conv_offset.bias) match mmcv's DeformConv2dPack so the
# ThinkTwice checkpoint (img_encoder.depth_net.depth_conv.4.*) loads unchanged.
import math

import torch
import torch.nn as nn
import torchvision
from mmengine.registry import MODELS


def _pair(x):
    return (x, x) if isinstance(x, int) else tuple(x)


@MODELS.register_module(name='DCN', force=True)
class DeformConv2dPack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, deform_groups=1, bias=False,
                 **kwargs):
        super().__init__()
        kh, kw = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kh, kw))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.conv_offset = nn.Conv2d(
            in_channels,
            deform_groups * 2 * kh * kw,
            kernel_size=(kh, kw),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        return torchvision.ops.deform_conv2d(
            x, offset, self.weight, self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation, mask=None)
