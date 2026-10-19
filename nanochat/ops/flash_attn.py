"""
FlashAttention-4 (Dao-AILab flash_attn.cute; on the H100 its SM90 kernels at their own tiles) as torch.library ops,
causal with `window_left` keys before each query: nanochat::flash_attn and its backward nanochat::flash_attn_bwd. FA4's
own autograd.Function launches CuTe kernels that torch.compile cannot trace: that broke the compiled graph at every
layer and left the loss unfused.

Document masking (ProLong, Gao et al. 2410.02660; modded-nanogpt track_1_short trains on varlen batches of whole
documents): cu_seqlens, the int32 segment offsets of the flattened (B*T) tokens (data/loader.py PretrainingBatches:
each piece of a row, a document, a document's cut part or a fill-in-the-middle example, and each row's padding is a
segment), makes each segment attend within itself only, causal and windowed from its own start, through FA4's varlen
kernels; T bounds every segment (the kernels size their grids from each segment's own length).
"""

import torch
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
from torch import Tensor


def _varlen(t):
    """(B, T, ...) -> (B*T, ...): FA4's varlen layout."""
    return t.reshape(-1, *t.shape[2:])


def _window(window_left, T):
    """FA4's window_size_left: None (its causal kernel) when the window spans the whole context, the local kernel
    otherwise; FA4 runs causal with any window as local (interface.py _resolve_causal_local_window). At d8 on real
    segments the causal kernel gives bitwise-equal outputs and gradients and is 1-2% faster fwd+bwd
    (benchmarks/h100.txt)."""
    return None if window_left >= T else window_left


@torch.library.custom_op("nanochat::flash_attn", mutates_args=(), device_types="cuda")
def flash_attn(q: Tensor, k: Tensor, v: Tensor, window_left: int, cu_seqlens: Tensor) -> tuple[Tensor, Tensor]:
    """(out (B, T, H, D) like q, lse fp32 (H, B*T): FA4's varlen layouts) of q, k, v (B, T, H, D) bf16."""
    T = q.size(1)
    out, lse = _flash_attn_fwd(_varlen(q), _varlen(k), _varlen(v), cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                               max_seqlen_q=T, max_seqlen_k=T, causal=True, window_size_left=_window(window_left, T),
                               return_lse=True)[:2]
    return out.view(q.shape), lse


@flash_attn.register_fake
def _(q, k, v, window_left, cu_seqlens):
    B, T, H, _ = q.shape
    return torch.empty_like(q), q.new_empty((H, B * T), dtype=torch.float32)


@torch.library.custom_op("nanochat::flash_attn_bwd", mutates_args=(), device_types="cuda")
def flash_attn_bwd(dout: Tensor, q: Tensor, k: Tensor, v: Tensor, out: Tensor, lse: Tensor, window_left: int,
                   cu_seqlens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(dq, dk, dv) like q, k, v given dout and the forward's out and lse: FA4's backward (dq accumulated with atomics,
    its fastest: benchmarks/h100.txt)."""
    T = q.size(1)
    grads = _flash_attn_bwd(_varlen(q), _varlen(k), _varlen(v), _varlen(out), _varlen(dout), lse, causal=True,
                            window_size_left=_window(window_left, T), cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                            max_seqlen_q=T, max_seqlen_k=T)
    return tuple(g.view(q.shape) for g in grads)  # q, k, v share one shape (MHA)


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
