"""Compatibility shim for running ThinkTwice's (mmdet3d-1.0 / mmcv-1.x era) model on the
modern PCLA stack: torch 2.2, mmcv-lite 2.1, mmengine 0.10 -- WITHOUT mmdet, mmdet3d, or
mmcv compiled ops (mmcv._ext).

Everything the model needs is provided here so the ported modules never import
mmdet/mmdet3d/mmcv.ops:

  * TT_MODELS      - one local mmengine registry for all submodules (detector, backbones,
                     necks, heads, middle-encoders). Replaces mmdet's DETECTORS/BACKBONES/
                     HEADS/NECKS and mmdet3d's MIDDLE_ENCODERS + builder.build_*.
  * build_backbone / build_neck / build_head / build_middle_encoder / build_voxel_encoder /
    build_model   - thin wrappers over TT_MODELS.build.
  * BaseModule     - re-exported from mmengine (pure python).
  * force_fp32 / auto_fp16 - no-op decorators (inference runs in fp32; the mmcv.runner
                     fp16 decorators were removed in the modern stack).

Import this module *first* in the model package: it also neutralizes deepspeed, whose
op-compatibility probe shells out to a (missing) nvcc during any mmengine import in the
PCLA environment. tt does no training and never needs deepspeed.
"""
import copy
import sys

# --- neutralize deepspeed (see module docstring) -------------------------------------
# Setting it to None makes `import deepspeed` raise ImportError, which mmengine catches
# and then skips its optional DeepSpeedStrategy. Only affects this (tt) process.
sys.modules.setdefault('deepspeed', None)

from mmengine.registry import Registry  # noqa: E402
from mmengine.model import BaseModule, ModuleList, Sequential  # noqa: E402,F401

# Single registry for every tt model submodule.
TT_MODELS = Registry('tt_models')


def _build(cfg, default_args=None):
    return TT_MODELS.build(cfg, default_args=default_args)


def build_backbone(cfg):
    return _build(cfg)


def build_neck(cfg):
    return _build(cfg)


def build_head(cfg):
    return _build(cfg)


def build_middle_encoder(cfg):
    return _build(cfg)


def build_voxel_encoder(cfg):
    return _build(cfg)


def build_model(cfg, train_cfg=None, test_cfg=None):
    """Mirror mmdet3d's build_model(cfg, train_cfg, test_cfg) call used by the agent."""
    cfg = copy.deepcopy(cfg)
    if train_cfg is not None and cfg.get('train_cfg') is None:
        cfg['train_cfg'] = train_cfg
    if test_cfg is not None and cfg.get('test_cfg') is None:
        cfg['test_cfg'] = test_cfg
    return TT_MODELS.build(cfg)


# --- no-op fp16 decorators (fp32 inference) ------------------------------------------
def force_fp32(*dargs, **dkwargs):
    def _decorator(func):
        return func
    return _decorator


def auto_fp16(*dargs, **dkwargs):
    def _decorator(func):
        return func
    return _decorator
