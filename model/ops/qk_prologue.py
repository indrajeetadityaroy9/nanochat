"""
Fused q/k prologue of model/modeling.py's attention (model/kernels/qk_prologue.cu), one launch per direction covering
q and k: per-head RMS norm with F.rms_norm's fp32 epsilon, fp32 per-dim norm weight (Qwen3, arXiv 2505.09388 §2),
rotary by -theta over the full head (karpathy/nanochat 92d63d4 gpt.py:57-65); fp32 math, one bf16 rounding per
output. Called inside model/ops/flash_attn.py's attention op.

q, k are the (B, T, H, HEAD_DIM) bf16 q and k blocks of the fused (B, T, 3C) projection (heads contiguous, token
stride 3C), dq, dk the same blocks of its gradient; norm weights (HEAD_DIM,) fp32; cos/sin (T_table >= T,
HEAD_DIM / 2) fp32.
"""

import ctypes

import torch
from torch import Tensor

from native import cuda_library, pointer, stream


def qk_forward(q: Tensor, k: Tensor, q_norm_weight: Tensor, k_norm_weight: Tensor, cos: Tensor,
               sin: Tensor) -> tuple[Tensor, Tensor]:
    """(normed and rotated q, k), contiguous."""
    B, T, H = q.shape[:3]
    q_out, k_out = q.new_empty(q.shape), k.new_empty(k.shape)
    cuda_library("qk_prologue").qkp_forward(
        pointer(q), pointer(k), pointer(q_norm_weight), pointer(k_norm_weight), pointer(q_out), pointer(k_out),
        pointer(cos), pointer(sin), ctypes.c_longlong(B * T * H), H, T, stream())
    return q_out, k_out


def qk_backward(q: Tensor, k: Tensor, q_norm_weight: Tensor, k_norm_weight: Tensor, dq_out: Tensor, dk_out: Tensor,
                cos: Tensor, sin: Tensor, dq: Tensor, dk: Tensor) -> tuple[Tensor, Tensor]:
    """Writes q's and k's gradients into dq, dk from the outputs' gradients dq_out, dk_out (contiguous); returns the
    norm weights' gradients, summed in a fixed order (bitwise reproducible)."""
    B, T, H, D = q.shape
    dq_norm_weight, dk_norm_weight = torch.empty_like(q_norm_weight), torch.empty_like(k_norm_weight)
    lib = cuda_library("qk_prologue")
    part = q_norm_weight.new_empty((2 * lib.qkp_backward_blocks(), D))  # per-block partial weight gradients
    lib.qkp_backward(
        pointer(q), pointer(k), pointer(q_norm_weight), pointer(k_norm_weight), pointer(dq_out), pointer(dk_out),
        pointer(dq), pointer(dk), pointer(dq_norm_weight), pointer(dk_norm_weight), pointer(part), pointer(cos),
        pointer(sin), ctypes.c_longlong(B * T * H), H, T, stream())
    return dq_norm_weight, dk_norm_weight
