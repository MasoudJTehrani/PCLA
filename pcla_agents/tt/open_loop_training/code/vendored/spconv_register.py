# Vendored from mmdet3d/models/layers/spconv/overwrite_spconv/write_spconv2.py (Apache-2.0,
# OpenMMLab). Registers spconv 2.x conv classes into mmengine's MODELS registry so
# mmcv.cnn.build_conv_layer(dict(type='SubMConv3d')) resolves WITHOUT mmdet3d, and installs
# the SparseModule state_dict hook that converts spconv-1.x <-> 2.x kernel layouts so the
# ThinkTwice checkpoint loads correctly. Import this module once (it self-registers).
import itertools
from typing import List

from mmengine.registry import MODELS
from torch.nn.parameter import Parameter


def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                          missing_keys, unexpected_keys, error_msgs) -> None:
    """Compat conv kernel weights between spconv 1.x (MMCV) and 2.x layouts."""
    version = local_metadata.get('version', None)
    for hook in self._load_state_dict_pre_hooks.values():
        hook(state_dict, prefix, local_metadata, strict, missing_keys,
             unexpected_keys, error_msgs)

    local_name_params = itertools.chain(self._parameters.items(),
                                        self._buffers.items())
    local_state = {k: v.data for k, v in local_name_params if v is not None}

    for name, param in local_state.items():
        key = prefix + name
        if key in state_dict:
            input_param = state_dict[key]
            if len(param.shape) == 0 and len(input_param.shape) == 1:
                input_param = input_param[0]
            if version != 2:
                dims = [len(input_param.shape) - 1] + list(
                    range(len(input_param.shape) - 1))
                input_param = input_param.permute(*dims)
            if input_param.shape != param.shape:
                error_msgs.append(
                    f'size mismatch for {key}: copying a param with '
                    f'shape {key, input_param.shape} from checkpoint,'
                    f'the shape in current model is {param.shape}.')
                continue
            if isinstance(input_param, Parameter):
                input_param = input_param.data
            try:
                param.copy_(input_param)
            except Exception:
                error_msgs.append(
                    f'While copying the parameter named "{key}", whose '
                    f'dimensions in the model are {param.size()} and whose '
                    f'dimensions in the checkpoint are {input_param.size()}.')
        elif strict:
            missing_keys.append(key)

    if strict:
        for key, input_param in state_dict.items():
            if key.startswith(prefix):
                input_name = key[len(prefix):]
                input_name = input_name.split('.', 1)[0]
                if input_name not in self._modules \
                        and input_name not in local_state:
                    unexpected_keys.append(key)


def register_spconv2() -> bool:
    try:
        from spconv.pytorch import (SparseConv2d, SparseConv3d, SparseConv4d,
                                    SparseConvTranspose2d,
                                    SparseConvTranspose3d, SparseInverseConv2d,
                                    SparseInverseConv3d, SparseModule,
                                    SubMConv2d, SubMConv3d, SubMConv4d)
    except ImportError:
        return False
    for cls, name in [
        (SparseConv2d, 'SparseConv2d'), (SparseConv3d, 'SparseConv3d'),
        (SparseConv4d, 'SparseConv4d'),
        (SparseConvTranspose2d, 'SparseConvTranspose2d'),
        (SparseConvTranspose3d, 'SparseConvTranspose3d'),
        (SparseInverseConv2d, 'SparseInverseConv2d'),
        (SparseInverseConv3d, 'SparseInverseConv3d'),
        (SubMConv2d, 'SubMConv2d'), (SubMConv3d, 'SubMConv3d'),
        (SubMConv4d, 'SubMConv4d'),
    ]:
        MODELS._register_module(cls, name, force=True)
    SparseModule._version = 2
    SparseModule._load_from_state_dict = _load_from_state_dict
    return True


IS_SPCONV2_AVAILABLE = register_spconv2()

from spconv.pytorch import (  # noqa: E402  re-export for the vendored modules
    SparseConvTensor, SparseModule, SparseSequential)
