"""
Fused q/k prologue of model/gpt.py (csrc/qk_prologue.cu), one launch per direction covering q and k: per-head RMS norm
with F.rms_norm's fp32 epsilon, fp32 per-dim gain (Qwen3, arXiv 2505.09388 §2), rotary by -theta over the full head
(karpathy/nanochat 92d63d4 gpt.py:57-65); fp32 math, one bf16 rounding per output.
"""

import ctypes
import functools

import torch
from torch import Tensor

from nanochat.native import cuda_library


@functools.cache
def qk_library(head_dim):
    return cuda_library("qk_prologue", head_dim=head_dim)


def launch(kernel, *tensors):
    """kernel(tensor pointers, rows and heads of tensors[0] = q, sequence length, current stream)."""
    q = tensors[0]
    kernel(*(ctypes.c_void_p(t.data_ptr()) for t in tensors), ctypes.c_longlong(q.numel() // q.size(-1)), q.size(2),
           q.size(1), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))


@torch.library.custom_op("nanochat::qk_norm_rope", mutates_args=(), device_types="cuda")
def qk_norm_rope(q: Tensor, k: Tensor, q_gain: Tensor, k_gain: Tensor, cos: Tensor,
                 sin: Tensor) -> tuple[Tensor, Tensor]:
    """q, k contiguous (B, T, H, D) bf16; gains (D,) fp32; cos/sin (T_table >= T, D/2) fp32; one library per D."""
    q_out, k_out = q.new_empty(q.shape), k.new_empty(k.shape)
    launch(qk_library(q.size(-1)).qkp_forward, q, k, q_gain, k_gain, q_out, k_out, cos, sin)
    return q_out, k_out


@qk_norm_rope.register_fake
def qk_norm_rope_fake(q, k, q_gain, k_gain, cos, sin):
    return q.new_empty(q.shape), k.new_empty(k.shape)


@torch.library.custom_op("nanochat::qk_norm_rope_backward", mutates_args=(), device_types="cuda")
def qk_norm_rope_backward(q: Tensor, k: Tensor, q_gain: Tensor, k_gain: Tensor, dq_out: Tensor, dk_out: Tensor,
                          cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(dq, dk, dq_gain, dk_gain); gain gradients summed in a fixed order (bitwise reproducible)."""
    dq, dk = q.new_empty(q.shape), k.new_empty(k.shape)
    dq_gain, dk_gain = torch.empty_like(q_gain), torch.empty_like(k_gain)
    lib = qk_library(q.size(-1))
    part = q_gain.new_empty((2 * lib.qkp_backward_blocks(), q.size(-1)))  # per-block partial gain gradients
    launch(lib.qkp_backward, q, k, q_gain, k_gain, dq_out, dk_out, dq, dk, dq_gain, dk_gain, part, cos, sin)
    return dq, dk, dq_gain, dk_gain


@qk_norm_rope_backward.register_fake
def qk_norm_rope_backward_fake(q, k, q_gain, k_gain, dq_out, dk_out, cos, sin):
    return q.new_empty(q.shape), k.new_empty(k.shape), q_gain.new_empty(q_gain.shape), k_gain.new_empty(k_gain.shape)


def qk_norm_rope_save(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def qk_norm_rope_grad(ctx, dq_out, dk_out):
    q, k, q_gain, k_gain, cos, sin = ctx.saved_tensors
    return *qk_norm_rope_backward(q, k, q_gain, k_gain, dq_out, dk_out, cos, sin), None, None


qk_norm_rope.register_autograd(qk_norm_rope_grad, setup_context=qk_norm_rope_save)
