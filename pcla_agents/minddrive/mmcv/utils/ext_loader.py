# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os
import pkgutil
import warnings
from collections import namedtuple

import torch

class _MissingExt:
    """Stand-in for an unbuilt compiled op bundle (e.g. ``mmcv._ext``).

    PCLA port: upstream calls ``load_ext`` at *module scope* in roughly a dozen
    op modules (roi_align, nms, focal_loss, deform_conv, voxelize, ...). Those
    modules are imported transitively by ``mmcv/__init__``, so a missing
    extension makes the entire package unimportable -- even though closed-loop
    inference for this model never calls most of those ops (and the ones it does,
    such as multi-scale deformable attention, have pure-PyTorch fallbacks).

    Returning this stub defers the failure from import time to *call* time: any
    op genuinely needed at runtime raises a clear, actionable error naming
    itself, while unused ops stay harmlessly absent.
    """

    def __init__(self, name, funcs):
        self._name = name
        self._funcs = set(funcs or ())

    def __getattr__(self, item):
        def _missing(*args, **kwargs):
            raise NotImplementedError(
                f"'{item}' comes from the compiled extension 'mmcv.{self._name}', "
                f'which is not built in this inference-only PCLA port (no nvcc). '
                f'If this op is genuinely required, build the extension or use a '
                f'pure-PyTorch equivalent.')
        return _missing

    def __bool__(self):
        return False


def missing_ext(name):
    """Public factory for a deferred-failure stand-in for an unbuilt extension.

    Used by op packages that import their compiled submodule directly (e.g.
    ``from . import iou3d_cuda``) rather than through :func:`load_ext`.
    """
    return _MissingExt(name, ())


def load_ext(name, funcs):
    try:
        ext = importlib.import_module('mmcv.' + name)
    except (ImportError, ModuleNotFoundError, OSError):
        return _MissingExt(name, funcs)
    for fun in funcs:
        assert hasattr(ext, fun), f'{fun} miss in module {name}'
    return ext

def check_ops_exist():
    ext_loader = pkgutil.find_loader('mmcv._ext')
    return ext_loader is not None
