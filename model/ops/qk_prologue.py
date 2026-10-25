"""
Fused q/k prologue of model/gpt.py (model/kernels/qk_prologue.cu), one launch per direction covering q and k:
per-head RMS norm with F.rms_norm's fp32 epsilon, fp32 per-dim gain (Qwen3, arXiv 2505.09388 §2), rotary by -theta
over the full head (karpathy/nanochat 92d63d4 gpt.py:57-65); fp32 math, one bf16 rounding per output.
"""

import ctypes
from collections.abc import Callable

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

from native import cuda_library, pointer, stream


def launch(kernel: Callable[..., int], *tensors: Tensor) -> None:
    """kernel(tensor pointers, rows and heads of tensors[0] = q, sequence length, current stream)."""
    q = tensors[0]
    kernel(*map(pointer, tensors), ctypes.c_longlong(q.numel() // q.size(-1)), q.size(2), q.size(1), stream())


@torch.library.custom_op("nanochat::qk_norm_rope", mutates_args=(), device_types="cuda")
def qk_norm_rope(q: Tensor, k: Tensor, q_gain: Tensor, k_gain: Tensor, cos: Tensor,
                 sin: Tensor) -> tuple[Tensor, Tensor]:
    """q, k contiguous (B, T, H, HEAD_DIM) bf16; gains (HEAD_DIM,) fp32; cos/sin (T_table >= T, HEAD_DIM/2) fp32."""
    q_out, k_out = q.new_empty(q.shape), k.new_empty(k.shape)
    launch(cuda_library("qk_prologue").qkp_forward, q, k, q_gain, k_gain, q_out, k_out, cos, sin)
    return q_out, k_out


@qk_norm_rope.register_fake
def qk_norm_rope_fake(q: Tensor, k: Tensor, q_gain: Tensor, k_gain: Tensor, cos: Tensor,
                      sin: Tensor) -> tuple[Tensor, Tensor]:
    return q.new_empty(q.shape), k.new_empty(k.shape)


@torch.library.custom_op("nanochat::qk_norm_rope_backward", mutates_args=(), device_types="cuda")
def qk_norm_rope_backward(q: Tensor, k: Tensor, q_gain: Tensor, k_gain: Tensor, dq_out: Tensor, dk_out: Tensor,
                          cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(dq, dk, dq_gain, dk_gain); gain gradients summed in a fixed order (bitwise reproducible)."""
    dq, dk = q.new_empty(q.shape), k.new_empty(k.shape)
    dq_gain, dk_gain = torch.empty_like(q_gain), torch.empty_like(k_gain)
    lib = cuda_library("qk_prologue")
    part = q_gain.new_empty((2 * lib.qkp_backward_blocks(), q.size(-1)))  # per-block partial gain gradients
    launch(lib.qkp_backward, q, k, q_gain, k_gain, dq_out, dk_out, dq, dk, dq_gain, dk_gain, part, cos, sin)
    return dq, dk, dq_gain, dk_gain


@qk_norm_rope_backward.register_fake
def qk_norm_rope_backward_fake(q: Tensor, k: Tensor, q_gain: Tensor, k_gain: Tensor, dq_out: Tensor, dk_out: Tensor,
                               cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return q.new_empty(q.shape), k.new_empty(k.shape), q_gain.new_empty(q_gain.shape), k_gain.new_empty(k_gain.shape)


def qk_norm_rope_save(ctx: FunctionCtx, inputs: tuple[Tensor, ...], output: tuple[Tensor, Tensor]) -> None:
    ctx.save_for_backward(*inputs)


def qk_norm_rope_grad(ctx: FunctionCtx, dq_out: Tensor,
                      dk_out: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, None, None]:
    q, k, q_gain, k_gain, cos, sin = ctx.saved_tensors
    return *qk_norm_rope_backward(q, k, q_gain, k_gain, dq_out, dk_out, cos, sin), None, None


qk_norm_rope.register_autograd(qk_norm_rope_grad, setup_context=qk_norm_rope_save)
