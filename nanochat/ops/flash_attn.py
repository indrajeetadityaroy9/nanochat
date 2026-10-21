"""
FlashAttention-4 (Dao-AILab flash_attn.cute, SM90 kernels) as torch.library ops, causal with `window_left` keys before
each query; FA4's own autograd.Function cannot be traced by torch.compile.

Document masking (ProLong, Gao et al., arXiv 2410.02660): FA4's varlen kernels restrict attention to the segments in
cu_seqlens, the int32 offsets of the flattened (B*T) tokens (data/loader.py PretrainingBatches).
"""

import torch
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
from torch import Tensor
from torch.autograd.function import FunctionCtx


def fa4_window_left(window_left: int, T: int) -> int | None:
    """None (FA4's causal kernel) when the window spans T, else the local window: bitwise equal and 1-2% faster at d8;
    FA4 does not make this switch (interface.py _resolve_causal_local_window)."""
    return None if window_left >= T else window_left


@torch.library.custom_op("nanochat::flash_attn", mutates_args=(), device_types="cuda")
def flash_attn(q: Tensor, k: Tensor, v: Tensor, window_left: int, cu_seqlens: Tensor) -> tuple[Tensor, Tensor]:
    """q, k, v (B, T, H, D) bf16 -> (out like q, lse fp32 (H, B*T))."""
    T = q.size(1)
    out, lse = _flash_attn_fwd(q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), cu_seqlens_q=cu_seqlens,
                               cu_seqlens_k=cu_seqlens, max_seqlen_q=T, max_seqlen_k=T, causal=True,
                               window_size_left=fa4_window_left(window_left, T), return_lse=True)[:2]
    return out.view(q.shape), lse


@flash_attn.register_fake
def flash_attn_fake(q: Tensor, k: Tensor, v: Tensor, window_left: int, cu_seqlens: Tensor) -> tuple[Tensor, Tensor]:
    B, T, H = q.shape[:3]
    return torch.empty_like(q), q.new_empty((H, B * T), dtype=torch.float32)


@torch.library.custom_op("nanochat::flash_attn_bwd", mutates_args=(), device_types="cuda")
def flash_attn_bwd(dout: Tensor, q: Tensor, k: Tensor, v: Tensor, out: Tensor, lse: Tensor, window_left: int,
                   cu_seqlens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(dq, dk, dv); FA4's backward with atomic dq accumulation, its fastest (deterministic=True: 1-3% slower, d24)."""
    T = q.size(1)
    grads = _flash_attn_bwd(q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), out.flatten(0, 1), dout.flatten(0, 1),
                            lse, causal=True, window_size_left=fa4_window_left(window_left, T), cu_seqlens_q=cu_seqlens,
                            cu_seqlens_k=cu_seqlens, max_seqlen_q=T, max_seqlen_k=T)
    return tuple(g.view(q.shape) for g in grads)  # MHA: q, k, v share one shape


@flash_attn_bwd.register_fake
def flash_attn_bwd_fake(dout: Tensor, q: Tensor, k: Tensor, v: Tensor, out: Tensor, lse: Tensor, window_left: int,
                        cu_seqlens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def flash_attn_save(ctx: FunctionCtx, inputs: tuple[Tensor, Tensor, Tensor, int, Tensor],
                    output: tuple[Tensor, Tensor]) -> None:
    q, k, v, ctx.window_left, cu_seqlens = inputs
    ctx.save_for_backward(q, k, v, *output, cu_seqlens)


def flash_attn_grad(ctx: FunctionCtx, dout: Tensor, dlse: Tensor) -> tuple[Tensor, Tensor, Tensor, None, None]:
    q, k, v, out, lse, cu_seqlens = ctx.saved_tensors
    return *flash_attn_bwd(dout, q, k, v, out, lse, ctx.window_left, cu_seqlens), None, None


flash_attn.register_autograd(flash_attn_grad, setup_context=flash_attn_save)
