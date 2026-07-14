# Importing this package registers all vendored built-ins into tt_compat.TT_MODELS:
#   ResNet, PAFPN (+FPN), SECOND, SECONDFPN, SparseEncoder, HardSimpleVFE.
# spconv_register also registers spconv conv layers into mmengine's MODELS so
# build_conv_layer(type='SubMConv3d') resolves.
from . import spconv_register  # noqa: F401  (registers spconv into mmengine MODELS)
from . import dcn  # noqa: F401  (registers torchvision-backed 'DCN' into mmengine MODELS)
from .fpn import FPN  # noqa: F401
from .pafpn import PAFPN  # noqa: F401
from .second import SECOND  # noqa: F401
from .second_fpn import SECONDFPN  # noqa: F401
from .sparse_encoder import SparseEncoder  # noqa: F401
from .hard_vfe import HardSimpleVFE  # noqa: F401
from .resnet import ResNet  # noqa: F401
