# Copyright (c) Megvii Inc. All rights reserved.
#
# PCLA port note (ThinkTwice -> modern stack): the original BEVDepth voxel_pooling is a
# custom CUDA extension that must be compiled with nvcc. To run in PCLA's environment
# (torch 2.2, no CUDA toolkit / nvcc on host) this module provides a numerically
# equivalent pure-PyTorch forward and falls back to it whenever the compiled extension is
# unavailable. The op is parameter-free (a bounded BEV scatter-add / sum-pool), so this
# substitution does not affect the pretrained checkpoint. Semantics verified against
# src/voxel_pooling_forward_cuda.cu: drop any point with x/y/z outside
# [0, voxel_num_{x,y,z}); sum features into output[b, y, x, :]; return [B, C, Y, X].
import torch
from torch.autograd import Function

try:  # use the compiled CUDA op if a user has built it (faster); otherwise pure-torch
    from . import voxel_pooling_ext as _voxel_pooling_ext
except Exception:  # pragma: no cover - expected when nvcc is unavailable
    _voxel_pooling_ext = None


class VoxelPooling(Function):
    @staticmethod
    def forward(ctx, geom_xyz: torch.Tensor, input_features: torch.Tensor,
                voxel_num: torch.Tensor) -> torch.Tensor:
        """Forward function for voxel pooling.

        Args:
            geom_xyz (Tensor): int xyz voxel coord per point, shape [B, N, 3].
            input_features (Tensor): feature per point, shape [B, N, C].
            voxel_num (Tensor): number of voxels per dim (X, Y, Z), shape [3].

        Returns:
            Tensor: [B, C, Y, X] BEV feature map.
        """
        assert geom_xyz.is_contiguous()
        assert input_features.is_contiguous()
        # no gradient for geom_xyz
        ctx.mark_non_differentiable(geom_xyz)
        geom_xyz = geom_xyz.reshape(geom_xyz.shape[0], -1, geom_xyz.shape[-1])
        input_features = input_features.reshape(
            (geom_xyz.shape[0], -1, input_features.shape[-1]))
        assert geom_xyz.shape[1] == input_features.shape[1]
        batch_size = input_features.shape[0]
        num_points = input_features.shape[1]
        num_channels = input_features.shape[2]
        vx, vy, vz = int(voxel_num[0]), int(voxel_num[1]), int(voxel_num[2])

        if _voxel_pooling_ext is not None and input_features.is_cuda:
            grad_input_features = torch.zeros_like(input_features)
            output_features = input_features.new_zeros(batch_size, vy, vx, num_channels)
            pos_memo = geom_xyz.new_ones(batch_size, num_points, 3) * -1
            _voxel_pooling_ext.voxel_pooling_forward_wrapper(
                batch_size, num_points, num_channels, vx, vy, vz,
                geom_xyz, input_features, output_features, pos_memo)
            ctx.save_for_backward(grad_input_features, pos_memo)
            return output_features.permute(0, 3, 1, 2)

        # ---- pure-PyTorch equivalent (bounded sum-pool via index_add_) ----
        x = geom_xyz[..., 0]
        y = geom_xyz[..., 1]
        z = geom_xyz[..., 2]
        valid = (x >= 0) & (x < vx) & (y >= 0) & (y < vy) & (z >= 0) & (z < vz)  # [B, N]
        batch_idx = torch.arange(batch_size, device=geom_xyz.device).view(-1, 1).expand(batch_size, num_points)
        flat = (batch_idx.long() * vy + y.long()) * vx + x.long()  # [B, N] cell index
        out = input_features.new_zeros(batch_size * vy * vx, num_channels)
        sel = valid.reshape(-1)
        out.index_add_(0, flat.reshape(-1)[sel], input_features.reshape(batch_size * num_points, num_channels)[sel])
        output_features = out.view(batch_size, vy, vx, num_channels)

        # pos_memo (b, y, x) for backward parity; -1 where dropped
        pos_memo = geom_xyz.new_full((batch_size, num_points, 3), -1)
        pos_memo[..., 0] = torch.where(valid, batch_idx.to(geom_xyz.dtype), pos_memo[..., 0])
        pos_memo[..., 1] = torch.where(valid, y, pos_memo[..., 1])
        pos_memo[..., 2] = torch.where(valid, x, pos_memo[..., 2])
        ctx.save_for_backward(torch.zeros_like(input_features), pos_memo)
        return output_features.permute(0, 3, 1, 2)

    @staticmethod
    def backward(ctx, grad_output_features):
        (grad_input_features, pos_memo) = ctx.saved_tensors
        kept = (pos_memo != -1)[..., 0]
        grad_input_features_shape = grad_input_features.shape
        grad_input_features = grad_input_features.reshape(
            grad_input_features.shape[0], -1, grad_input_features.shape[-1])
        grad_input_features[kept] = grad_output_features[
            pos_memo[kept][..., 0].long(), :, pos_memo[kept][..., 1].long(),
            pos_memo[kept][..., 2].long()]
        grad_input_features = grad_input_features.reshape(
            grad_input_features_shape)
        return None, grad_input_features, None


voxel_pooling = VoxelPooling.apply
