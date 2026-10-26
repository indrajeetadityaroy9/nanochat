"""
Value-embedding gate and mix of model/modeling.py Attention (model/kernels/ve_gate.cu), one launch forward and one
pass backward: v + 2 sigmoid(n W^T) per head x bf16(table[input_ids]) (per-layer token-indexed value embeddings:
modded-nanogpt record #14, MoVE's LaVE, arXiv 2601.22887 §4.1 Eq. 6, behind a per-head gate; nanochat e85db6b), the
gate's logits in fp32 and the mix rounded once to bf16, as the unfused PyTorch expressions round. Called inside
model/ops/flash_attn.py's attention op, which reads v from the fused q/k/v projection and hands the backward FA4's dv
in place in the projection's gradient (both at token stride 3C).
The embedding lookup is the op's, so its backward writes the table's gradient (fp32, as F.embedding's backward): the
kernel adds it into table.grad itself (rows in token order, one atomic per run of equal tokens) and autograd receives
None, so micro-batches accumulate in place, with no zeroed per-call gradient and no autograd add; table must be a leaf
that requires grad. The token order is torch.sort of input_ids, one sort per micro-batch (model/modeling.py sorts
outside the attention op, and every layer sorts the same input_ids, which AOTAutograd's common-subexpression
elimination merges: torch/_functorch/compile_utils.py fx_graph_cse). The gate weight's gradient dz^T n is a cuBLAS
GEMM from bf16 dz, fp32 out.
"""

import ctypes

import torch
from torch import Tensor

from native import cuda_library, pointer, stream


def ve_forward(n: Tensor, v: Tensor, input_ids: Tensor, table: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    """n (B, T, C) bf16 normed stream and input_ids (B, T) int32 token ids, contiguous; v (B, T, H, HEAD_DIM) bf16, the
    v block of the fused projection (heads contiguous, rows evenly spaced); table (V, C) fp32 value embeddings, w
    (H, C) fp32 gate weight -> (mixed v, contiguous, gate (B*T, H) fp32, saved for the backward)."""
    out, gate = v.new_empty(v.shape), v.new_empty((input_ids.numel(), w.size(0)), dtype=torch.float32)
    if not cuda_library("ve_gate").ve_forward(pointer(n), pointer(v), ctypes.c_longlong(v.stride(1)),
                                              pointer(input_ids), pointer(table), pointer(w), pointer(out),
                                              pointer(gate), ctypes.c_longlong(input_ids.numel()), w.size(0),
                                              stream()):
        raise ValueError(f"model/kernels/ve_gate.cu has no kernel for {w.size(0)} heads")
    return out, gate


def ve_backward(dv: Tensor, n: Tensor, ids: Tensor, order: Tensor, table: Tensor, w: Tensor,
                gate: Tensor) -> tuple[Tensor, Tensor]:
    """dv (B, T, H, HEAD_DIM) the mixed v's gradient, laid out as v (v's own gradient is dv itself), ids (B*T,) int32
    the sorted input_ids, order (B*T,) int64 their rows (torch.sort) -> (dn like n, dw like w); the table's gradient
    is added into table.grad (zeros when None); H one ve_forward accepted."""
    if table.grad is None:
        table.grad = torch.zeros_like(table)
    dn, dz = torch.empty_like(n), torch.empty_like(gate, dtype=torch.bfloat16)
    cuda_library("ve_gate").ve_backward(pointer(dv), ctypes.c_longlong(dv.stride(1)), pointer(ids), pointer(order),
                                        pointer(table), pointer(w), pointer(gate), pointer(dn), pointer(table.grad),
                                        pointer(dz), ctypes.c_longlong(ids.numel()), w.size(0), stream())
    return dn, torch.mm(dz.t(), n.view(-1, n.size(-1)), out_dtype=w.dtype)
