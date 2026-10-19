"""The block's GEMMs as torch.library ops, W the fp32 master weight, rounded to x's dtype inside the ops, dW in W's
dtype straight from the GEMM's fp32 accumulators (nn.Linear and QuACK's linear/mlp functions round it to bf16 first, a
relative error up to 2^-9 per element). Ops, not autograd.Functions, so a fullgraph torch.compile
traces through them.

nanochat::linear (attention projections c_q, c_k, c_v, c_o): y = x W^T, backward nanochat::linear_bwd (dX = dY W,
dW = dY^T X). QuACK (Tri Dao, CuTe DSL) for y and dX; every dW GEMM (here and in the MLP) QuACK or cuBLAS per width
(ops/quack_gemm.py's table). quack.linear.linear_func is not used: it rounds dW to the weight's dtype.

nanochat::relu2_mlp (the MLP, karpathy/nanochat 92d63d4 gpt.py:131-141: c_proj(relu(c_fc x)^2)): the GEMMs of
quack.mlp.mlp_func(activation="relu_sq"), its non-gated default path (quack.linear linear_act_func + act_linear_func:
the forward stores the pre-activation for the backward instead of recomputing it), the same kernels and arguments, with
dW in fp32: relu^2 in the up-projection GEMM's epilogue (gemm_act), its derivative in the down-projection's backward
GEMM (gemm_dact, which also recomputes the post-activation for dW_proj).

Every QuACK GEMM runs at its config in ops/quack_gemm.py's table for the width C (no autotuning)."""
import torch
from quack.gemm_interface import gemm_act, gemm_dact
from torch import Tensor

from nanochat.ops.quack_gemm import TABLE, gemm


@torch.library.custom_op("nanochat::linear", mutates_args=(), device_types="cuda")
def linear(x: Tensor, weight: Tensor) -> Tensor:
    """x (..., K), weight (N, K) -> (..., N) in x's dtype."""
    C = x.shape[-1]
    y = gemm(x.reshape(-1, C), weight.to(x.dtype).t(), "proj_fwd", C, x.dtype, 1)
    return y.view(*x.shape[:-1], weight.shape[0])


@linear.register_fake
def _(x, weight):
    return x.new_empty(*x.shape[:-1], weight.shape[0])


@torch.library.custom_op("nanochat::linear_bwd", mutates_args=(), device_types="cuda")
def linear_bwd(dy: Tensor, x: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
    """(dX in x's dtype and shape, dW in weight's dtype) given dY (..., N)."""
    dy2, x2 = dy.reshape(-1, dy.shape[-1]).contiguous(), x.reshape(-1, x.shape[-1])  # dY: autograd's layout
    C = x.shape[-1]
    dx = gemm(dy2, weight.to(x.dtype), "proj_dx", C, x.dtype, 1).view(x.shape)
    return dx, gemm(dy2.t(), x2, "proj_dw", C, weight.dtype, 1)


@linear_bwd.register_fake
def _(dy, x, weight):
    return x.new_empty(x.shape), weight.new_empty(weight.shape)


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, dy):
    return linear_bwd(dy, *ctx.saved_tensors)


linear.register_autograd(_backward, setup_context=_setup_context)


@torch.library.custom_op("nanochat::relu2_mlp", mutates_args=(), device_types="cuda")
def relu2_mlp(x: Tensor, w_fc: Tensor, w_proj: Tensor) -> tuple[Tensor, Tensor]:
    """x (..., C), w_fc (h, C), w_proj (C, h) -> (y = w_proj relu(w_fc x)^2 (..., C) in x's dtype, the pre-activation
    w_fc x (M, h) in x's dtype: kept for the backward, no gradient)."""
    C = x.shape[-1]
    x2 = x.reshape(-1, C)
    preact, postact = gemm_act(x2, w_fc.to(x.dtype).t(), activation="relu_sq", config=TABLE[C]["mlp_up"])
    y = gemm(postact, w_proj.to(x.dtype).t(), "mlp_down", C, x.dtype, 1)
    return y.view(*x.shape[:-1], w_proj.shape[0]), preact


@relu2_mlp.register_fake
def _(x, w_fc, w_proj):
    return x.new_empty(*x.shape[:-1], w_proj.shape[0]), x.new_empty(x.numel() // x.shape[-1], w_fc.shape[0])


@torch.library.custom_op("nanochat::relu2_mlp_bwd", mutates_args=(), device_types="cuda")
def relu2_mlp_bwd(dy: Tensor, x: Tensor, preact: Tensor, w_fc: Tensor, w_proj: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(dX in x's dtype and shape, dW_fc, dW_proj in the weights' dtype) given dY (..., C) and relu2_mlp's
    pre-activation; the post-activation is recomputed in the dact GEMM's epilogue."""
    C = x.shape[-1]
    dy2, x2 = dy.reshape(-1, C).contiguous(), x.reshape(-1, C)
    dpreact, postact = gemm_dact(dy2, w_proj.to(x.dtype), preact, activation="relu_sq", config=TABLE[C]["mlp_dact"])
    dw_proj = gemm(dy2.t(), postact, "mlp_dw_proj", C, w_proj.dtype, 1)
    del postact
    dx = gemm(dpreact, w_fc.to(x.dtype), "mlp_dx", C, x.dtype, 1).view(x.shape)
    dw_fc = gemm(dpreact.t(), x2, "mlp_dw_fc", C, w_fc.dtype, 1)
    return dx, dw_fc, dw_proj


@relu2_mlp_bwd.register_fake
def _(dy, x, preact, w_fc, w_proj):
    return x.new_empty(x.shape), w_fc.new_empty(w_fc.shape), w_proj.new_empty(w_proj.shape)


def _mlp_setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs, output[1])


def _mlp_backward(ctx, dy, _dpreact):
    x, w_fc, w_proj, preact = ctx.saved_tensors
    return relu2_mlp_bwd(dy, x, preact, w_fc, w_proj)


relu2_mlp.register_autograd(_mlp_backward, setup_context=_mlp_setup_context)
