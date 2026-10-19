"""
FlashAttention-4 (Dao-AILab flash_attn.cute, sm_12x patched) as torch.library ops, causal with `window_left` keys before
each query: nanochat::flash_attn and its backward nanochat::flash_attn_bwd. FA4's own autograd.Function launches CuTe
kernels that torch.compile cannot trace: that broke the compiled graph at every layer and left the loss unfused.

Forward tiles 64x64: their 48 KB of shared memory let two CTAs share an sm_12x SM (99 KB), where FA4's default 128x64
(64 KB) fits one; measured on GB10 at nanochat's shapes -12% per forward at full and windowed attention, output
bit-identical.

Document masking (ProLong, Gao et al. 2410.02660; modded-nanogpt track_1_short trains on varlen batches of whole
documents): cu_seqlens, the segments of the flattened (B*T) tokens (data/loader.py), makes each segment attend within
itself only, causal and windowed from its own start, through FA4's varlen kernels; T bounds every segment (the kernels
size their grids from each segment's own length). Measured on sm_121: forward, dK and dV bitwise equal to running each
document alone, dQ within the backward's atomic-order noise.
"""

from typing import Optional

import torch
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
from torch import Tensor


def _varlen(t):
    """(B, T, ...) -> (B*T, ...): FA4's varlen layout."""
    return t.reshape(-1, *t.shape[2:])


@torch.library.custom_op("nanochat::flash_attn", mutates_args=())
def flash_attn(q: Tensor, k: Tensor, v: Tensor, window_left: int,
               cu_seqlens: Optional[Tensor]) -> tuple[Tensor, Tensor]:
    """(out (B, T, H, D) like q, lse fp32: (B, H, T), or (H, B*T) with cu_seqlens (FA4's layouts)) of q, k, v
    (B, T, H, D) bf16; cu_seqlens None or the int32 segment offsets of the flattened B*T tokens."""
    if cu_seqlens is None:
        out, lse = _flash_attn_fwd(q, k, v, causal=True, window_size_left=window_left, window_size_right=0,
                                   return_lse=True, tile_mn=(64, 64))[:2]
        return out, lse
    T = q.size(1)
    out, lse = _flash_attn_fwd(_varlen(q), _varlen(k), _varlen(v), cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                               max_seqlen_q=T, max_seqlen_k=T, causal=True, window_size_left=window_left,
                               window_size_right=0, return_lse=True, tile_mn=(64, 64))[:2]
    return out.view(q.shape), lse


@flash_attn.register_fake
def _(q, k, v, window_left, cu_seqlens):
    B, T, H, _ = q.shape
    return torch.empty_like(q), q.new_empty((B, H, T) if cu_seqlens is None else (H, B * T), dtype=torch.float32)


@torch.library.custom_op("nanochat::flash_attn_bwd", mutates_args=())
def flash_attn_bwd(dout: Tensor, q: Tensor, k: Tensor, v: Tensor, out: Tensor, lse: Tensor, window_left: int,
                   cu_seqlens: Optional[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    """(dq, dk, dv) like q, k, v given dout and the forward's out and lse."""
    if cu_seqlens is None:
        return tuple(_flash_attn_bwd(q, k, v, out, dout, lse, None, True, 0.0, window_size_left=window_left,
                                     window_size_right=0))
    T = q.size(1)
    grads = _flash_attn_bwd(_varlen(q), _varlen(k), _varlen(v), _varlen(out), _varlen(dout), lse, None, True, 0.0,
                            window_size_left=window_left, window_size_right=0, cu_seqlens_q=cu_seqlens,
                            cu_seqlens_k=cu_seqlens, max_seqlen_q=T, max_seqlen_k=T)
    return tuple(g.view(q.shape) for g in grads)  # q, k, v share one shape (one value head per query head)


@flash_attn_bwd.register_fake
def _(dout, q, k, v, out, lse, window_left, cu_seqlens):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_context(ctx, inputs, output):
    q, k, v, ctx.window_left, cu_seqlens = inputs
    ctx.save_for_backward(q, k, v, *output, cu_seqlens)


def _backward(ctx, dout, dlse):
    q, k, v, out, lse, cu_seqlens = ctx.saved_tensors
    return *flash_attn_bwd(dout, q, k, v, out, lse, ctx.window_left, cu_seqlens), None, None


flash_attn.register_autograd(_backward, setup_context=_setup_context)
