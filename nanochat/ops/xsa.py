"""
XSA (Zhai, arXiv 2603.09078 Eq. 2; modded-nanogpt track_1_short attention.py:224-228) at strength 1 (model/attention.py)
with the head-wise attention-output gate, as one fused op per direction (xsa.cu): per token and head out = sigmoid(z)
(y - (y . v / max(v . v, 1e-8)) v), z the gate's fp32 logits, fp32 inside and one bf16 rounding per output. The
backward recomputes the sums from y and v (the attention's output and value input, which FlashAttention keeps for its
own backward anyway). The library is compiled from xsa.cu once per source revision and GPU architecture
(native.cuda_library) on first use.
"""

import ctypes
import functools
import os

import torch
from torch import Tensor

from nanochat.native import cuda_library

SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xsa.cu")


@functools.cache
def _lib():
    lib = cuda_library("xsa", SOURCE)
    lib.xsa_forward.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_longlong, ctypes.c_void_p]
    lib.xsa_backward.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_longlong, ctypes.c_void_p]
    return lib


@torch.library.custom_op("nanochat::xsa", mutates_args=(), device_types="cuda")
def xsa(y: Tensor, v: Tensor, z: Tensor) -> Tensor:
    """sigmoid(z) (y - (y . v / max(v . v, 1e-8)) v) per (token, head) row; y, v (B, T, H, 128) bf16 contiguous, z
    (B, T, H) fp32 contiguous."""
    out = torch.empty_like(y)
    _lib().xsa_forward(y.data_ptr(), v.data_ptr(), z.data_ptr(), out.data_ptr(), z.numel(),
                       torch.cuda.current_stream().cuda_stream)
    return out


@xsa.register_fake
def _(y, v, z):
    return torch.empty_like(y)


@torch.library.custom_op("nanochat::xsa_backward", mutates_args=(), device_types="cuda")
def xsa_backward(dout: Tensor, y: Tensor, v: Tensor, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(dy, dv, dz) of xsa given dout (contiguous)."""
    dy, dv, dz = torch.empty_like(y), torch.empty_like(v), torch.empty_like(z)
    _lib().xsa_backward(dout.data_ptr(), y.data_ptr(), v.data_ptr(), z.data_ptr(), dy.data_ptr(), dv.data_ptr(),
                        dz.data_ptr(), z.numel(), torch.cuda.current_stream().cuda_stream)
    return dy, dv, dz


@xsa_backward.register_fake
def _(dout, y, v, z):
    return torch.empty_like(y), torch.empty_like(v), torch.empty_like(z)


def _setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, dout):
    return xsa_backward(dout, *ctx.saved_tensors)


xsa.register_autograd(_backward, setup_context=_setup)
