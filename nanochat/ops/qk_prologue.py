"""
The attention's q/k prologue (model/gpt.py) as one fused op per direction (csrc/qk_prologue.cu): per head RMS norm
(F.rms_norm's fp32 epsilon), a learnable fp32 per-dim gain (Qwen3, arXiv 2505.09388 §2), then rotary by -theta over the
full head (karpathy/nanochat 92d63d4 gpt.py:57-65), on q and k in one launch, fp32 inside and one bf16 rounding per
output. The library is compiled from csrc/qk_prologue.cu (and its header csrc/qk_row.cuh, shared with csrc/decode.cu)
once per source revision and GPU architecture (native.cuda_library) on first use.
"""

import ctypes
import functools

import torch
from torch import Tensor

from nanochat.native import cuda_library


@functools.cache
def _lib():
    lib = cuda_library("qk_prologue")
    shape = [ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.qkp_forward.argtypes = [ctypes.c_void_p] * 8 + shape
    lib.qkp_backward.argtypes = [ctypes.c_void_p] * 13 + shape
    lib.qkp_backward_blocks.argtypes = [ctypes.c_longlong]
    lib.qkp_backward_blocks.restype = ctypes.c_uint
    return lib


def _args(q):
    """The kernels' trailing arguments: rows and heads of q (k has q's shape), sequence length, current stream."""
    return q.numel() // q.size(-1), q.size(2), q.size(1), torch.cuda.current_stream().cuda_stream


@torch.library.custom_op("nanochat::qk_norm_rope", mutates_args=(), device_types="cuda")
def qk_norm_rope(q: Tensor, k: Tensor, q_gain: Tensor, k_gain: Tensor, cos: Tensor,
                 sin: Tensor) -> tuple[Tensor, Tensor]:
    """(rotary(rms_norm(q) * q_gain, -theta), same of k with k_gain), contiguous; q, k contiguous (B, T, H, 128) bf16 of
    one shape; gains (128,) fp32; cos/sin (T_table >= T, 64) fp32 contiguous tables of positions 0..T_table-1, rotating
    dims i and i + 64 of each head by angle i."""
    q_out, k_out = q.new_empty(q.shape), k.new_empty(k.shape)
    _lib().qkp_forward(q.data_ptr(), k.data_ptr(), q_gain.data_ptr(), k_gain.data_ptr(), q_out.data_ptr(),
                       k_out.data_ptr(), cos.data_ptr(), sin.data_ptr(), *_args(q))
    return q_out, k_out


@qk_norm_rope.register_fake
def _(q, k, q_gain, k_gain, cos, sin):
    return q.new_empty(q.shape), k.new_empty(k.shape)


@torch.library.custom_op("nanochat::qk_norm_rope_backward", mutates_args=(), device_types="cuda")
def qk_norm_rope_backward(q: Tensor, k: Tensor, q_gain: Tensor, k_gain: Tensor, dq_out: Tensor, dk_out: Tensor,
                          cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Gradients wrt q, k (contiguous) and the gains (fp32, summed over every row in a fixed order: bitwise
    reproducible) of qk_norm_rope, given those wrt its outputs (contiguous), recomputing from q and k (contiguous)."""
    rows = q.numel() // q.size(-1)
    dq, dk = q.new_empty(q.shape), k.new_empty(k.shape)
    dq_gain, dk_gain = torch.empty_like(q_gain), torch.empty_like(k_gain)
    part = q_gain.new_empty((2 * _lib().qkp_backward_blocks(rows), q.size(-1)))  # the blocks' partial gain gradients
    _lib().qkp_backward(q.data_ptr(), k.data_ptr(), q_gain.data_ptr(), k_gain.data_ptr(), dq_out.data_ptr(),
                        dk_out.data_ptr(), dq.data_ptr(), dk.data_ptr(), dq_gain.data_ptr(), dk_gain.data_ptr(),
                        part.data_ptr(), cos.data_ptr(), sin.data_ptr(), *_args(q))
    return dq, dk, dq_gain, dk_gain


@qk_norm_rope_backward.register_fake
def _(q, k, q_gain, k_gain, dq_out, dk_out, cos, sin):
    return q.new_empty(q.shape), k.new_empty(k.shape), q_gain.new_empty(q_gain.shape), k_gain.new_empty(k_gain.shape)


def _setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, dq_out, dk_out):
    q, k, q_gain, k_gain, cos, sin = ctx.saved_tensors
    return *qk_norm_rope_backward(q, k, q_gain, k_gain, dq_out, dk_out, cos, sin), None, None


qk_norm_rope.register_autograd(_backward, setup_context=_setup)
