"""The block's GEMMs as torch.library ops, W the fp32 master weight, rounded to x's dtype inside the ops, dW in W's
dtype straight from the GEMM's fp32 accumulators (nn.Linear and QuACK's linear/mlp functions round it to bf16 first:
rel err vs fp64 1.7e-3 there, 4e-6..8e-5 here, no slower). Ops, not autograd.Functions, so a fullgraph torch.compile
traces through them.

nanochat::linear (attention projections c_q, c_k, c_v, c_proj): y = x W^T, backward nanochat::linear_bwd (dX = dY W,
dW = dY^T X). QuACK (Tri Dao, CuTe DSL) for y and dX (measured 3-26% ahead of cuBLAS on GB10), dW QuACK or cuBLAS per
width (ops/quack_gemm.py's proj_dw). quack.linear.linear_func is not used: it rounds dW to the weight's dtype.

nanochat::swiglu_mlp (the MLP): quack.mlp.mlp_func(activation="swiglu", concat_layout=True)'s GEMMs, the same kernels
and arguments (its default path: the forward stores the pre-activation for the backward instead of recomputing it),
with dW in fp32: the gated activation in the up-projection GEMM's epilogue, its derivative in the down-projection's
backward GEMM, w_fc = [gate; up] in QuACK's concat layout (the pre-activation interleaves gate and up columns).

nanochat::narrow_linear (the few-column GEMMs from the residual stream: attention-output gate logits, MUDD coefficient
hidden): y = x W^T in fp32 straight from the accumulators (cuBLAS, torch.mm out_dtype; no QuACK config: N is not a
model width and may be no multiple of 8), so the logits are not rounded to bf16. Backward, dY fp32: dX = bf16(dY) W in
x's dtype (an activation gradient, rounded to bf16 anyway); dW = dY^T X in fp32 with dY as two bf16 terms, hi = bf16(dY)
and lo = bf16(dY - hi) (16 significant bits; one GEMM over [hi, lo] reads X once): dY rounded to bf16 alone gives dW
1.6e-3 rel err vs fp64, the split 2.5e-6 (GB10, M = 8192, K = 640, N = 5).

Every QuACK GEMM runs at its config in ops/quack_gemm.py's table for the width C (no autotuning)."""
import torch
from quack.gemm_interface import gemm_dgated, gemm_gated
from torch import Tensor

from nanochat.ops.quack_gemm import CONFIGS, gemm


@torch.library.custom_op("nanochat::linear", mutates_args=(), device_types="cuda")
def linear(x: Tensor, weight: Tensor) -> Tensor:
    """x (..., K), weight (N, K) -> (..., N) in x's dtype."""
    C = x.shape[-1]
    return gemm(x.reshape(-1, C), weight.to(x.dtype).t(), "proj_fwd", C).view(*x.shape[:-1], weight.shape[0])


@linear.register_fake
def _(x, weight):
    return x.new_empty(*x.shape[:-1], weight.shape[0])


@torch.library.custom_op("nanochat::linear_bwd", mutates_args=(), device_types="cuda")
def linear_bwd(dy: Tensor, x: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
    """(dX in x's dtype and shape, dW in weight's dtype) given dY (..., N)."""
    dy2, x2 = dy.reshape(-1, dy.shape[-1]).contiguous(), x.reshape(-1, x.shape[-1])  # dY: autograd's layout
    C = x.shape[-1]
    dx = gemm(dy2, weight.to(x.dtype), "proj_dx", C).view(x.shape)
    if CONFIGS[C]["proj_dw"] == "cublas":
        return dx, torch.mm(dy2.t(), x2, out_dtype=weight.dtype)
    return dx, gemm(dy2.t(), x2, "proj_dw", C, out_dtype=weight.dtype)


@linear_bwd.register_fake
def _(dy, x, weight):
    return x.new_empty(x.shape), weight.new_empty(weight.shape)


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, dy):
    return linear_bwd(dy, *ctx.saved_tensors)


linear.register_autograd(_backward, setup_context=_setup_context)


@torch.library.custom_op("nanochat::narrow_linear", mutates_args=(), device_types="cuda")
def narrow_linear(x: Tensor, weight: Tensor) -> Tensor:
    """x (..., K) bf16, weight (N, K) fp32 -> (..., N) fp32."""
    x2 = x.reshape(-1, x.shape[-1])
    return torch.mm(x2, weight.to(x.dtype).t(), out_dtype=torch.float32).view(*x.shape[:-1], weight.shape[0])


@narrow_linear.register_fake
def _(x, weight):
    return x.new_empty(*x.shape[:-1], weight.shape[0], dtype=torch.float32)


@torch.library.custom_op("nanochat::narrow_linear_bwd", mutates_args=(), device_types="cuda")
def narrow_linear_bwd(dy: Tensor, x: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
    """(dX in x's dtype and shape, dW fp32) given dY (..., N) fp32."""
    N = dy.shape[-1]
    dy2, x2 = dy.reshape(-1, N), x.reshape(-1, x.shape[-1])
    hi = dy2.to(x.dtype)
    lo = (dy2 - hi.float()).to(x.dtype)
    dx = torch.mm(hi, weight.to(x.dtype)).view(x.shape)
    dw = torch.mm(torch.cat([hi, lo], dim=1).t(), x2, out_dtype=weight.dtype)
    return dx, dw[:N] + dw[N:]


@narrow_linear_bwd.register_fake
def _(dy, x, weight):
    return x.new_empty(x.shape), weight.new_empty(weight.shape)


def _narrow_backward(ctx, dy):
    return narrow_linear_bwd(dy, *ctx.saved_tensors)


narrow_linear.register_autograd(_narrow_backward, setup_context=_setup_context)


@torch.library.custom_op("nanochat::swiglu_mlp", mutates_args=(), device_types="cuda")
def swiglu_mlp(x: Tensor, w_fc: Tensor, w_proj: Tensor) -> tuple[Tensor, Tensor]:
    """x (..., C), w_fc (2h, C) = [gate; up], w_proj (C, h) -> (y = w_proj (silu(gate x) * up x) (..., C) in x's dtype,
    the pre-activation (M, 2h) in x's dtype, gate and up columns interleaved: kept for the backward, no gradient)."""
    C = x.shape[-1]
    x2, w_up = x.reshape(-1, C), w_fc.to(x.dtype).t()
    preact, postact = gemm_gated(x2, w_up, activation="swiglu", concat_layout=("B",), config=CONFIGS[C]["mlp_up"])
    return gemm(postact, w_proj.to(x.dtype).t(), "mlp_down", C).view(*x.shape[:-1], w_proj.shape[0]), preact


@swiglu_mlp.register_fake
def _(x, w_fc, w_proj):
    return x.new_empty(*x.shape[:-1], w_proj.shape[0]), x.new_empty(x.numel() // x.shape[-1], w_fc.shape[0])


@torch.library.custom_op("nanochat::swiglu_mlp_bwd", mutates_args=(), device_types="cuda")
def swiglu_mlp_bwd(dy: Tensor, x: Tensor, preact: Tensor, w_fc: Tensor, w_proj: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(dX in x's dtype and shape, dW_fc, dW_proj in the weights' dtype) given dY (..., C) and swiglu_mlp's
    pre-activation; the post-activation is recomputed in the dgated GEMM's epilogue."""
    C = x.shape[-1]
    dy2, x2, w_down = dy.reshape(-1, C).contiguous(), x.reshape(-1, C), w_proj.to(x.dtype)
    dpreact, postact = gemm_dgated(dy2, w_down, preact, activation="swiglu", dynamic_scheduler=True,
                                   config=CONFIGS[C]["mlp_dgated"])
    dw_proj = gemm(dy2.t(), postact, "mlp_dw_proj", C, out_dtype=w_proj.dtype)
    del postact
    dx = gemm(dpreact, w_fc.to(x.dtype), "mlp_dx", C, concat_layout=("B",)).view(x.shape)
    dw_fc = gemm(dpreact.t(), x2, "mlp_dw_fc", C, out_dtype=w_fc.dtype, concat_layout=("out",))
    return dx, dw_fc, dw_proj


@swiglu_mlp_bwd.register_fake
def _(dy, x, preact, w_fc, w_proj):
    return x.new_empty(x.shape), w_fc.new_empty(w_fc.shape), w_proj.new_empty(w_proj.shape)


def _mlp_setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs, output[1])


def _mlp_backward(ctx, dy, _dpreact):
    x, w_fc, w_proj, preact = ctx.saved_tensors
    return swiglu_mlp_bwd(dy, x, preact, w_fc, w_proj)


swiglu_mlp.register_autograd(_mlp_backward, setup_context=_mlp_setup_context)
