"""
Value-embedding gate and mix of model/modeling.py Attention (model/kernels/ve_gate.cu), one launch forward and one
pass backward: v + 2 sigmoid(n W^T) per head x bf16(table[input_ids]) (per-layer token-indexed value embeddings:
modded-nanogpt record #14, MoVE's LaVE, arXiv 2601.22887 §4.1 Eq. 6, behind a per-head gate; nanochat e85db6b), the
gate's logits in fp32 and the mix rounded once to bf16, as the unfused PyTorch expressions round.
The embedding lookup is the op's, so its backward writes the table's gradient (fp32, as F.embedding's backward): the
kernel adds it into table.grad itself (rows in token order, one atomic per run of equal tokens) and autograd receives
None, so micro-batches accumulate in place, with no zeroed per-call gradient and no autograd add; table must be a leaf
that requires grad. The token order is torch.sort of input_ids, one sort per micro-batch (every layer sorts the same
input_ids, which AOTAutograd's common-subexpression elimination merges: torch/_functorch/compile_utils.py
fx_graph_cse). The gate weight's gradient dz^T n is a cuBLAS GEMM from bf16 dz, fp32 out.
"""

import ctypes

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

from native import cuda_library, pointer, stream


def evenly_spaced_rows(t: Tensor) -> bool:
    """t (B, T, H, HEAD_DIM): heads contiguous and rows evenly spaced, the layout the kernels index by a row stride."""
    return t.stride(3) == 1 and t.stride(2) == t.size(3) and t.stride(0) == t.size(1) * t.stride(1)


def ve_mix(n: Tensor, v: Tensor, input_ids: Tensor, table: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    """n (B, T, C) bf16 normed stream and input_ids (B, T) int32 token ids, contiguous; v (B, T, H, HEAD_DIM) bf16 with
    evenly spaced rows (contiguous, or a view of a fused q/k/v projection); table (V, C) fp32 value embeddings (a leaf;
    its gradient accumulates into table.grad), w (H, C) fp32 gate weight -> (mixed v, contiguous, gate (B*T, H) fp32,
    saved for the backward)."""
    if not evenly_spaced_rows(v):
        raise ValueError(f"ve_mix: v's strides {v.stride()} are not evenly spaced rows of contiguous heads")
    ids, order = torch.sort(input_ids.flatten())
    return ve_mix_sorted(n, v, input_ids, table, w, ids, order)


@torch.library.custom_op("nanochat::ve_mix", mutates_args=(), device_types="cuda")
def ve_mix_sorted(n: Tensor, v: Tensor, input_ids: Tensor, table: Tensor, w: Tensor, ids: Tensor,
                  order: Tensor) -> tuple[Tensor, Tensor]:
    """ve_mix with input_ids sorted: ids (B*T,) int32 the sorted tokens, order (B*T,) int64 their rows (the
    backward's visiting order)."""
    out, gate = v.new_empty(v.shape), v.new_empty((input_ids.numel(), w.size(0)), dtype=torch.float32)
    if not cuda_library("ve_gate").ve_forward(pointer(n), pointer(v), ctypes.c_longlong(v.stride(1)),
                                              pointer(input_ids), pointer(table), pointer(w), pointer(out),
                                              pointer(gate), ctypes.c_longlong(input_ids.numel()), w.size(0),
                                              stream()):
        raise ValueError(f"model/kernels/ve_gate.cu has no kernel for {w.size(0)} heads")
    return out, gate


@ve_mix_sorted.register_fake
def ve_mix_fake(n: Tensor, v: Tensor, input_ids: Tensor, table: Tensor, w: Tensor, ids: Tensor,
                order: Tensor) -> tuple[Tensor, Tensor]:
    return v.new_empty(v.shape), v.new_empty((input_ids.numel(), w.size(0)), dtype=torch.float32)


@torch.library.custom_op("nanochat::ve_mix_backward", mutates_args=(), device_types="cuda")
def ve_mix_backward(dout: Tensor, n: Tensor, ids: Tensor, order: Tensor, table: Tensor, w: Tensor,
                    gate: Tensor) -> tuple[Tensor, Tensor]:
    """(dn like n, dw like w); dv is dout itself; the table's gradient is added into table.grad (zeros when None).
    dout with evenly spaced rows."""
    if not (table.is_leaf and table.requires_grad):
        raise ValueError("ve_mix: the table must be a leaf that requires grad (its gradient goes to table.grad)")
    if table.grad is None:
        table.grad = torch.zeros_like(table)
    dn, dz = torch.empty_like(n), torch.empty_like(gate, dtype=torch.bfloat16)
    if not cuda_library("ve_gate").ve_backward(pointer(dout), ctypes.c_longlong(dout.stride(1)), pointer(ids),
                                               pointer(order), pointer(table), pointer(w), pointer(gate), pointer(dn),
                                               pointer(table.grad), pointer(dz), ctypes.c_longlong(ids.numel()),
                                               w.size(0), stream()):
        raise ValueError(f"model/kernels/ve_gate.cu has no kernel for {w.size(0)} heads")
    return dn, torch.mm(dz.t(), n.view(-1, n.size(-1)), out_dtype=w.dtype)


@ve_mix_backward.register_fake
def ve_mix_backward_fake(dout: Tensor, n: Tensor, ids: Tensor, order: Tensor, table: Tensor, w: Tensor,
                         gate: Tensor) -> tuple[Tensor, Tensor]:
    return torch.empty_like(n), torch.empty_like(w)


def ve_mix_save(ctx: FunctionCtx, inputs: tuple[Tensor, ...], output: tuple[Tensor, Tensor]) -> None:
    n, _, _, table, w, ids, order = inputs
    ctx.save_for_backward(n, ids, order, table, w, output[1])


def ve_mix_grad(ctx: FunctionCtx, dout: Tensor, dgate: Tensor) -> tuple[Tensor | None, ...]:
    n, ids, order, table, w, gate = ctx.saved_tensors
    dn, dw = ve_mix_backward(dout if evenly_spaced_rows(dout) else dout.contiguous(), n, ids, order, table, w, gate)
    return dn, dout, None, None, dw, None, None


ve_mix_sorted.register_autograd(ve_mix_grad, setup_context=ve_mix_save)
