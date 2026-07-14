# PCLA tt port. Importing this package (as the config plugin does) sets up the compat layer
# and registers every model submodule into tt_compat.TT_MODELS so the network config's
# `type=` names resolve. Modules are added here as they are ported.
from . import tt_compat  # noqa: F401  deepspeed neutralize + local registry
from . import vendored    # noqa: F401  register built-ins (ResNet/PAFPN/SECOND/...)

# Ported model modules (register custom classes into TT_MODELS).
from .model_code.backbones import *   # noqa: F401,F403  LSS, LidarNet, PAFPN_fp32, SparseEncoder_fp32
from .model_code.dense_heads import *  # noqa: F401,F403 ThinkTwiceDecoder
from .encoder_decoder_framework import *  # noqa: F401,F403  EncoderDecoder
# Preprocessing transforms (register into TT_TRANSFORMS) + dataset helper.
from .datasets.pipelines import *  # noqa: F401,F403
from .datasets.carla_dataset import union2one  # noqa: F401
