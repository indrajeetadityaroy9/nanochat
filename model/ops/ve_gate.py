"""
Value-embedding gate and mix of model/modeling.py Attention (model/kernels/ve_gate.cu), one launch forward and one
pass backward: v + 2 sigmoid(n W^T) per head x bf16(table[input_ids]) (per-layer token-indexed value embeddings:
modded-nanogpt record #14, MoVE's LaVE, arXiv 2601.22887 §4.1 Eq. 6, behind a per-head gate; nanochat e85db6b), the
gate's logits in fp32 and the mix rounded once to bf16, as the unfused PyTorch expressions round.
The embedding lookup is the op's, so its backward writes the table's gradient (fp32, atomics as F.embedding's backward)
and the gate weight's (fp32, a fixed summation order).
"""

import ctypes

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

from native import cuda_library, pointer, stream


@torch.library.custom_op("nanochat::ve_mix", mutates_args=(), device_types="cuda")
def ve_mix(n: Tensor, v: Tensor, input_ids: Tensor, table: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    """n (B, T, C) bf16 normed stream, v (B, T, H, HEAD_DIM) bf16, input_ids (B, T) int32 token ids (all contiguous),
    table (V, C) fp32 value embeddings, w (H, C) fp32 gate weight -> (mixed v like v, gate (B*T, H) fp32, saved for
    the backward)."""
    out, gate = torch.empty_like(v), v.new_empty((input_ids.numel(), w.size(0)), dtype=torch.float32)
    if not cuda_library("ve_gate").ve_forward(pointer(n), pointer(v), pointer(input_ids), pointer(table), pointer(w),
                                              pointer(out), pointer(gate), ctypes.c_longlong(input_ids.numel()),
                                              w.size(0), stream()):
        raise ValueError(f"model/kernels/ve_gate.cu has no kernel for {w.size(0)} heads")
    return out, gate


@ve_mix.register_fake
def ve_mix_fake(n: Tensor, v: Tensor, input_ids: Tensor, table: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    return torch.empty_like(v), v.new_empty((input_ids.numel(), w.size(0)), dtype=torch.float32)


@torch.library.custom_op("nanochat::ve_mix_backward", mutates_args=(), device_types="cuda")
def ve_mix_backward(dout: Tensor, n: Tensor, input_ids: Tensor, table: Tensor, w: Tensor,
                    gate: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(dn like n, dtable like table, dw like w); dv is dout itself. dout contiguous."""
    lib = cuda_library("ve_gate")
    dn, dtable, dw, dz = torch.empty_like(n), torch.empty_like(table), torch.empty_like(w), torch.empty_like(gate)
    part = w.new_empty((lib.ve_dw_blocks(), *w.shape))  # per-block partial gate-weight gradients
    if not lib.ve_backward(pointer(dout), pointer(n), pointer(input_ids), pointer(table), pointer(w), pointer(gate),
                           pointer(dn), pointer(dtable), ctypes.c_longlong(table.size(0)), pointer(dw), pointer(dz),
                           pointer(part), ctypes.c_longlong(input_ids.numel()), w.size(0), stream()):
        raise ValueError(f"model/kernels/ve_gate.cu has no kernel for {w.size(0)} heads")
    return dn, dtable, dw


@ve_mix_backward.register_fake
def ve_mix_backward_fake(dout: Tensor, n: Tensor, input_ids: Tensor, table: Tensor, w: Tensor,
                         gate: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return torch.empty_like(n), torch.empty_like(table), torch.empty_like(w)


def ve_mix_save(ctx: FunctionCtx, inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
                output: tuple[Tensor, Tensor]) -> None:
    n, _, input_ids, table, w = inputs
    ctx.save_for_backward(n, input_ids, table, w, output[1])


def ve_mix_grad(ctx: FunctionCtx, dout: Tensor, dgate: Tensor) -> tuple[Tensor, Tensor, None, Tensor, Tensor]:
    n, input_ids, table, w, gate = ctx.saved_tensors
    dn, dtable, dw = ve_mix_backward(dout, n, input_ids, table, w, gate)
    return dn, dout, None, dtable, dw


ve_mix.register_autograd(ve_mix_grad, setup_context=ve_mix_save)
