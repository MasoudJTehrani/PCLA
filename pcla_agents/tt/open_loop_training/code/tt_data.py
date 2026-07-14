"""Data-plumbing replacements for the tt runtime, standing in for mmcv.parallel
(DataContainer + collate), mmengine/mmdet Compose, and mmdet's to_tensor -- none of which
are available/compatible in the modern mmcv-lite stack. Behaviour matches how the ThinkTwice
agent uses them (samples_per_gpu=1)."""
import numpy as np
import torch

from mmengine.registry import Registry

# Local registry for preprocessing transforms (LoadPoints, IDAImageTransform, ...).
TT_TRANSFORMS = Registry('tt_transforms')


class DataContainer:
    """Minimal stand-in for mmcv.parallel.DataContainer."""

    def __init__(self, data, stack=False, padding_value=0, cpu_only=False,
                 pad_dims=2):
        self._data = data
        self.stack = stack
        self.padding_value = padding_value
        self.cpu_only = cpu_only
        self.pad_dims = pad_dims

    @property
    def data(self):
        return self._data

    def __repr__(self):
        return f'{self.__class__.__name__}(stack={self.stack}, cpu_only={self.cpu_only})'


def to_tensor(data):
    """Convert objects of various python types to torch.Tensor (mmdet to_tensor)."""
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, (list, tuple)) and not isinstance(data[0], str):
        return torch.tensor(data)
    elif isinstance(data, (int, float)):
        return torch.tensor(data)
    return torch.as_tensor(data)


def collate(batch, samples_per_gpu=1):
    """mmcv-style collate over a list of sample dicts (samples_per_gpu grouping).

    - DataContainer(cpu_only=True): -> DataContainer whose .data is a list of chunks,
      each chunk a list of the per-sample .data (metas). `.data[0]` gives the first chunk.
    - DataContainer(stack=True): -> DataContainer whose .data is a list of chunks, each a
      tensor stacked over the chunk's samples. `.data[0]` gives the first stacked tensor.
    - plain tensors/arrays/scalars: default-collated (stacked over the batch dim).
    """
    assert isinstance(batch, (list, tuple)) and len(batch) > 0
    sample = batch[0]
    out = {}
    n = len(batch)
    for key in sample:
        vals = [b[key] for b in batch]
        v0 = sample[key]
        if isinstance(v0, DataContainer):
            if v0.cpu_only:
                chunks = [[v.data for v in vals[i:i + samples_per_gpu]]
                          for i in range(0, n, samples_per_gpu)]
                out[key] = DataContainer(chunks, cpu_only=True)
            elif v0.stack:
                chunks = []
                for i in range(0, n, samples_per_gpu):
                    grp = [v.data for v in vals[i:i + samples_per_gpu]]
                    chunks.append(torch.stack(grp, 0))
                out[key] = DataContainer(chunks, stack=True)
            else:
                chunks = [[v.data for v in vals[i:i + samples_per_gpu]]
                          for i in range(0, n, samples_per_gpu)]
                out[key] = DataContainer(chunks)
        elif isinstance(v0, torch.Tensor):
            out[key] = torch.stack(vals, 0)
        elif isinstance(v0, np.ndarray):
            out[key] = torch.stack([torch.as_tensor(v) for v in vals], 0)
        elif isinstance(v0, (int, float)):
            out[key] = torch.tensor(vals)
        else:
            out[key] = vals
    return out


class LiDARPoints:
    """Minimal stand-in for mmdet3d's LiDARPoints (only `.tensor` is used by the agent)."""

    def __init__(self, tensor, points_dim=None, attribute_dims=None):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(np.asarray(tensor), dtype=torch.float32)
        self.tensor = tensor.float()
        self.points_dim = points_dim if points_dim is not None else tensor.shape[-1]
        self.attribute_dims = attribute_dims

    def __len__(self):
        return self.tensor.shape[0]


def get_points_type(points_type):
    """Return the points container class (only 'LIDAR' is used by tt)."""
    return LiDARPoints


class Compose:
    """Sequentially compose transforms; each dict is built from TT_TRANSFORMS."""

    def __init__(self, transforms):
        self.transforms = []
        for t in transforms:
            if isinstance(t, dict):
                self.transforms.append(TT_TRANSFORMS.build(t))
            elif callable(t):
                self.transforms.append(t)
            else:
                raise TypeError(f'transform must be dict or callable, got {type(t)}')

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data
