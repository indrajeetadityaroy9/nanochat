"""
The attention's q/k prologue (model/attention.py) as one fused op per direction (qk_prologue.cu): rotary by -theta over
the first half of each head (modded-nanogpt's half-truncated RoPE), per-head RMS norm (F.rms_norm's fp32 epsilon) and
the qk scale, on q and k in one launch, fp32 inside and one bf16 rounding per output; optionally modded's key offset (a
key's stationary dims from the previous position, none at row or segment starts). The library is compiled from
qk_prologue.cu once per source revision and GPU architecture (native.cuda_library) on first use.
"""

import ctypes
import functools
import os
from typing import Optional

import torch
from torch import Tensor

from nanochat.native import cuda_library

SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qk_prologue.cu")


@functools.cache
def _lib():
    lib = cuda_library("qk_prologue", SOURCE)
    shape = ([ctypes.c_longlong] * 2 + [ctypes.c_int] * 2 + [ctypes.c_longlong] * 2 +
             [ctypes.c_int, ctypes.c_bool, ctypes.c_float, ctypes.c_void_p])
    lib.qkp_forward.argtypes = [ctypes.c_void_p] * 7 + shape
    lib.qkp_backward.argtypes = [ctypes.c_void_p] * 9 + shape
    return lib


def _args(q, k, starts, scale, key_offset):
    """The kernels' trailing arguments: the starts pointer, rows and heads of q and k, their token strides, sequence
    length, key offset, scale, current stream."""
    return (0 if starts is None else starts.data_ptr(), q.numel() // q.size(-1), k.numel() // k.size(-1), q.size(2),
            k.size(2), q.stride(1), k.stride(1), q.size(1), key_offset, scale, torch.cuda.current_stream().cuda_stream)


@torch.library.custom_op("nanochat::qk_norm_rope", mutates_args=(), device_types="cuda")
def qk_norm_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, scale: float, key_offset: bool,
                 starts: Optional[Tensor]) -> tuple[Tensor, Tensor]:
    """(scale * rms_norm(rotary(q, -theta)), same of k), contiguous; q, k (B, T, H, 128) bf16 with dense heads at any
    token stride; cos/sin (T_table >= T, 32) fp32 contiguous tables of positions 0..T_table-1, rotating the first 64
    dims of each head. key_offset: each key's dims 64..127 are the previous position's (its own normalized and scaled
    values), except at t = 0 and where starts ((B, T) bool, the document mask's segment starts; None: rows only) is
    set."""
    q_out, k_out = q.new_empty(q.shape), k.new_empty(k.shape)
    _lib().qkp_forward(q.data_ptr(), k.data_ptr(), q_out.data_ptr(), k_out.data_ptr(), cos.data_ptr(), sin.data_ptr(),
                       *_args(q, k, starts, scale, key_offset))
    return q_out, k_out


@qk_norm_rope.register_fake
def _(q, k, cos, sin, scale, key_offset, starts):
    return q.new_empty(q.shape), k.new_empty(k.shape)


@torch.library.custom_op("nanochat::qk_norm_rope_backward", mutates_args=(), device_types="cuda")
def qk_norm_rope_backward(q: Tensor, k: Tensor, dq_out: Tensor, dk_out: Tensor, cos: Tensor, sin: Tensor,
                          scale: float, key_offset: bool, starts: Optional[Tensor]) -> tuple[Tensor, Tensor]:
    """Gradients wrt q and k of qk_norm_rope (contiguous), given those wrt its outputs (contiguous), recomputing from q
    and k."""
    dq, dk = q.new_empty(q.shape), k.new_empty(k.shape)
    _lib().qkp_backward(q.data_ptr(), k.data_ptr(), dq_out.data_ptr(), dk_out.data_ptr(), dq.data_ptr(), dk.data_ptr(),
                        cos.data_ptr(), sin.data_ptr(), *_args(q, k, starts, scale, key_offset))
    return dq, dk


@qk_norm_rope_backward.register_fake
def _(q, k, dq_out, dk_out, cos, sin, scale, key_offset, starts):
    return q.new_empty(q.shape), k.new_empty(k.shape)


def _setup(ctx, inputs, output):
    q, k, cos, sin, ctx.scale, ctx.key_offset, starts = inputs
    ctx.save_for_backward(q, k, cos, sin, starts)


def _backward(ctx, dq_out, dk_out):
    q, k, cos, sin, starts = ctx.saved_tensors
    dq, dk = qk_norm_rope_backward(q, k, dq_out, dk_out, cos, sin, ctx.scale, ctx.key_offset, starts)
    return dq, dk, None, None, None, None, None


qk_norm_rope.register_autograd(_backward, setup_context=_setup)
