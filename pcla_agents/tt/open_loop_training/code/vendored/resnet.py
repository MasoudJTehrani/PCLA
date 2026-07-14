# ResNet backbone for tt's LSS image encoder, built on torchvision (avoids importing mmdet,
# which pulls mmcv._ext). torchvision's ResNet uses the SAME submodule naming as mmdet's
# ResNet (conv1 / bn1 / layer1..4, block conv1/bn1/conv2/bn2/conv3/bn3/downsample.0/.1), so
# the pretrained checkpoint keys (img_encoder.img_backbone.*) load unchanged. fc/avgpool are
# dropped so the param set matches exactly (318 params for depth=50).
from torch import nn
from torchvision.models.resnet import Bottleneck as _TVBottleneck
from torchvision.models.resnet import ResNet as _TVResNet

from ..tt_compat import TT_MODELS as MODELS

_ARCH = {50: (_TVBottleneck, [3, 4, 6, 3])}


@MODELS.register_module()
class ResNet(nn.Module):
    """torchvision ResNet exposed as a multi-scale backbone (out_indices stages)."""

    def __init__(self,
                 depth=50,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 norm_eval=False,
                 init_cfg=None,
                 **kwargs):
        super().__init__()
        assert depth in _ARCH, f'only resnet{list(_ARCH)} supported, got {depth}'
        block, layers = _ARCH[depth]
        tv = _TVResNet(block, layers)
        # re-expose stem + stages under the exact names the checkpoint uses
        self.conv1 = tv.conv1
        self.bn1 = tv.bn1
        self.relu = tv.relu
        self.maxpool = tv.maxpool
        self.layer1 = tv.layer1
        self.layer2 = tv.layer2
        self.layer3 = tv.layer3
        self.layer4 = tv.layer4
        self.out_indices = tuple(out_indices)
        self.norm_eval = norm_eval

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer in enumerate(
            [self.layer1, self.layer2, self.layer3, self.layer4]):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def init_weights(self):
        # no-op: the pretrained ThinkTwice checkpoint overwrites these weights.
        pass

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self
