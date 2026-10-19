"""
The Engram custom ops (DeepSeek Engram, arXiv 2601.07372 v2, §2.3 Eq. 3-5) and the loader of their CUDA library
(engram.cu).

- nanochat::engram_kv(x, weight): [k; v] = W_kv x (Eq. 3) on cuBLAS (torch.mm). The fp32 master W_kv is rounded to x's
  dtype (bf16) inside the op; the forward and dX are bf16 GEMMs; dW = dY^T X in fp32 straight from the GEMM's fp32
  accumulators (autograd through F.linear would round dW to bf16).
- nanochat::engram_fuse(h, kv, norm_h, norm_k, norm_v, conv_w, starts, dilation): (H + Y (Eq. 4-5), the gate alpha per
  token, fp32, no gradient) (engram.cu: a gate kernel per token, a convolution kernel per channel vector; fp32 inside,
  bf16 V~, RMSNorm(V~) and output); its backward gives dh, dkv and the gradients of the norm and convolution weights
  (deterministic).
- nanochat::engram_taps(h, kv, norm_h, norm_k, norm_v): u = RMSNorm(V~), the values the convolution's taps read (the
  decoder's state after a prefill).

Kernel layouts (engram.cu "Layouts"): h, kv contiguous bf16, C a multiple of 128 up to 2048; norm weights (C,) and
conv_w (C, ENGRAM_CONV_KERNEL) fp32; starts (B, T) bool or None. Library calls run on the current stream.
"""

import ctypes
import functools
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from nanochat.native import cuda_library

CUDA_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engram.cu")

# The short convolution's kernel size: §2.3 "kernel size w = 4", engram.cu's TAPS (a channel's weights are one float4)
ENGRAM_CONV_KERNEL = 4


@functools.cache
def engram_library():
    """engram.cu built for the current device (nanochat.native.cuda_library), its functions typed."""
    lib = cuda_library("engram_cuda", CUDA_SOURCE)
    p, i64, c_int = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
    signatures = {
        "ec_gate_fwd": ([p, p, p, p, p, p, p, p, p, i64, c_int, p], None),
        "ec_conv_fwd": ([p, p, p, p, p, p, i64, c_int, c_int, c_int, p], None),
        "ec_conv_bwd": ([p, p, p, p, p, p, p, p, i64, c_int, c_int, c_int, p], None),
        "ec_gate_bwd": ([p, p, p, p, p, p, p, p, p, p, p, i64, c_int, p], None),
        "ec_gate_partials": ([i64, c_int], i64),
        "ec_conv_partials": ([i64, c_int], i64),
    }
    for name, (argtypes, restype) in signatures.items():
        getattr(lib, name).argtypes, getattr(lib, name).restype = argtypes, restype
    return lib


def _stream():
    return torch.cuda.current_stream().cuda_stream


def _gate(h, kv, norm_h, norm_k, norm_v, alpha=None):
    """The gate kernel (Eq. 4 and RMSNorm(V~)): V~ and u = RMSNorm(V~; norm_v) bf16 like h, r_v (tokens,) fp32; alpha
    (tokens,) fp32, if given, receives each token's gate."""
    n, c = h.numel() // h.shape[-1], h.shape[-1]
    vt, u, rv = torch.empty_like(h), torch.empty_like(h), h.new_empty(n, dtype=torch.float32)
    engram_library().ec_gate_fwd(h.data_ptr(), kv.data_ptr(), norm_h.data_ptr(), norm_k.data_ptr(), norm_v.data_ptr(),
                                 vt.data_ptr(), u.data_ptr(), rv.data_ptr(), 0 if alpha is None else alpha.data_ptr(),
                                 n, c, _stream())
    return vt, u, rv


@torch.library.custom_op("nanochat::engram_fuse", mutates_args=(), device_types="cuda")
def engram_fuse(h: Tensor, kv: Tensor, norm_h: Tensor, norm_k: Tensor, norm_v: Tensor, conv_w: Tensor,
                starts: Optional[Tensor], dilation: int) -> tuple[Tensor, Tensor]:
    """(H + Y (Eq. 4-5), the gate alpha (B, T) fp32, no gradient) per row of h (B, T, C) bf16 with [k; v] = kv
    (B, T, 2C) bf16, the RMSNorm weights norm_h, norm_k, norm_v (C,) and the convolution's conv_w
    (C, ENGRAM_CONV_KERNEL) fp32; starts (B, T) bool marks the first token of each attention segment (None: every row
    one segment)."""
    alpha = h.new_empty(h.shape[:-1], dtype=torch.float32)
    vt, u, _ = _gate(h, kv, norm_h, norm_k, norm_v, alpha)
    out = torch.empty_like(h)
    engram_library().ec_conv_fwd(h.data_ptr(), vt.data_ptr(), u.data_ptr(), conv_w.data_ptr(),
                                 0 if starts is None else starts.data_ptr(), out.data_ptr(), h.shape[0] * h.shape[1],
                                 h.shape[1], h.shape[2], dilation, _stream())
    return out, alpha


@engram_fuse.register_fake
def _(h, kv, norm_h, norm_k, norm_v, conv_w, starts, dilation):
    return torch.empty_like(h), h.new_empty(h.shape[:-1], dtype=torch.float32)


@torch.library.custom_op("nanochat::engram_fuse_bwd", mutates_args=(), device_types="cuda")
def engram_fuse_bwd(dout: Tensor, h: Tensor, kv: Tensor, norm_h: Tensor, norm_k: Tensor, norm_v: Tensor, conv_w: Tensor,
                    starts: Optional[Tensor], dilation: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """(dh, dkv, dnorm_h, dnorm_k, dnorm_v, dconv_w) of engram_fuse's output for its gradient dout. V~ and u are
    recomputed (the gate kernel, as in the forward); the weight gradients' per-block partial sums are summed here."""
    dout = dout.contiguous()  # autograd's choice of layout; the kernels read (N, C) rows
    (_, T, c), n = h.shape, h.shape[0] * h.shape[1]
    lib = engram_library()
    vt, u, rv = _gate(h, kv, norm_h, norm_k, norm_v)
    du = torch.empty(h.shape, dtype=torch.float32, device=h.device)
    conv_partial = h.new_empty(lib.ec_conv_partials(n, c), c, ENGRAM_CONV_KERNEL + 1, dtype=torch.float32)
    lib.ec_conv_bwd(dout.data_ptr(), vt.data_ptr(), u.data_ptr(), rv.data_ptr(), conv_w.data_ptr(),
                    0 if starts is None else starts.data_ptr(), du.data_ptr(), conv_partial.data_ptr(), n, T, c, dilation,
                    _stream())
    dh, dkv = torch.empty_like(h), torch.empty_like(kv)
    gate_partial = h.new_empty(lib.ec_gate_partials(n, c), c, dtype=torch.float32)
    lib.ec_gate_bwd(dout.data_ptr(), du.data_ptr(), h.data_ptr(), kv.data_ptr(), rv.data_ptr(), norm_h.data_ptr(),
                    norm_k.data_ptr(), norm_v.data_ptr(), dh.data_ptr(), dkv.data_ptr(), gate_partial.data_ptr(), n, c,
                    _stream())
    conv_sums, p = conv_partial.sum(0), gate_partial.sum(0)
    return (dh, dkv, p * norm_k, p * norm_h, conv_sums[:, ENGRAM_CONV_KERNEL].contiguous(),
            conv_sums[:, :ENGRAM_CONV_KERNEL].contiguous())


@engram_fuse_bwd.register_fake
def _(dout, h, kv, norm_h, norm_k, norm_v, conv_w, starts, dilation):
    return tuple(torch.empty_like(t) for t in (h, kv, norm_h, norm_k, norm_v, conv_w))


def _fuse_setup_context(ctx, inputs, output):
    h, kv, norm_h, norm_k, norm_v, conv_w, starts, ctx.dilation = inputs
    ctx.save_for_backward(h, kv, norm_h, norm_k, norm_v, conv_w, starts)
    ctx.mark_non_differentiable(output[1])


def _fuse_backward(ctx, dout, _dalpha):
    return *engram_fuse_bwd(dout, *ctx.saved_tensors, ctx.dilation), None, None


engram_fuse.register_autograd(_fuse_backward, setup_context=_fuse_setup_context)


@torch.library.custom_op("nanochat::engram_taps", mutates_args=(), device_types="cuda")
def engram_taps(h: Tensor, kv: Tensor, norm_h: Tensor, norm_k: Tensor, norm_v: Tensor) -> Tensor:
    """u = RMSNorm(V~; norm_v) (B, T, C) bf16 of engram_fuse's inputs, the values its convolution taps read (the
    decoder's state after a prefill; no gradient)."""
    return _gate(h, kv, norm_h, norm_k, norm_v)[1]


@engram_taps.register_fake
def _(h, kv, norm_h, norm_k, norm_v):
    return torch.empty_like(h)


@torch.library.custom_op("nanochat::engram_kv", mutates_args=(), device_types="cuda")
def engram_kv(x: Tensor, weight: Tensor) -> Tensor:
    """x (..., K) bf16, weight (N, K) fp32 -> (..., N) in x's dtype."""
    return F.linear(x, weight.to(x.dtype))


@engram_kv.register_fake
def _(x, weight):
    return x.new_empty(*x.shape[:-1], weight.shape[0])


@torch.library.custom_op("nanochat::engram_kv_bwd", mutates_args=(), device_types="cuda")
def engram_kv_bwd(dy: Tensor, x: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
    """(dX in x's dtype and shape, dW in weight's dtype) of engram_kv for its gradient dy (..., N)."""
    dy2, x2 = dy.reshape(-1, dy.shape[-1]), x.reshape(-1, x.shape[-1])
    dx = torch.mm(dy2, weight.to(x.dtype)).view(x.shape)
    return dx, torch.mm(dy2.t(), x2, out_dtype=weight.dtype)


@engram_kv_bwd.register_fake
def _(dy, x, weight):
    return x.new_empty(x.shape), torch.empty_like(weight)


def _kv_setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _kv_backward(ctx, dy):
    return engram_kv_bwd(dy, *ctx.saved_tensors)


engram_kv.register_autograd(_kv_backward, setup_context=_kv_setup_context)
