"""The block's GEMMs as torch.library ops (traceable by fullgraph torch.compile): the fp32 master weight W rounded to x's
dtype, dW in fp32 straight from the accumulators (QuACK's linear/mlp functions round dW to bf16, rel. error up to 2^-9).
Every GEMM at its config in ops/quack_gemm.py TABLE.

nanochat::linear (c_q, c_k, c_v, c_o): y = x W^T; backward dX = dY W, dW = dY^T X.

nanochat::relu2_mlp: c_proj(relu(c_fc x)^2) (karpathy/nanochat 92d63d4 gpt.py:131-141) on the kernels of
quack.mlp.mlp_func(activation="relu_sq"): relu^2 in the up-projection's epilogue (gemm_act, which stores the
pre-activation), its derivative and the recomputed post-activation in the backward's gemm_dact."""
import torch
from quack.gemm_interface import gemm_act, gemm_dact
from torch import Tensor

from nanochat.ops.quack_gemm import TABLE, gemm


@torch.library.custom_op("nanochat::linear", mutates_args=(), device_types="cuda")
def linear(x: Tensor, weight: Tensor) -> Tensor:
    """x (..., K), weight (N, K) -> (..., N) in x's dtype."""
    C = x.shape[-1]
    y = gemm(x.reshape(-1, C), weight.to(x.dtype).t(), role="proj_fwd", width=C, out_dtype=x.dtype, alpha=1)
    return y.view(*x.shape[:-1], weight.shape[0])


@linear.register_fake
def linear_fake(x, weight):
    return x.new_empty(*x.shape[:-1], weight.shape[0])


@torch.library.custom_op("nanochat::linear_bwd", mutates_args=(), device_types="cuda")
def linear_bwd(dy: Tensor, x: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
    """(dX like x, dW in weight's dtype)."""
    C = x.shape[-1]
    dy2, x2 = dy.reshape(-1, dy.shape[-1]), x.reshape(-1, C)
    dx = gemm(dy2, weight.to(x.dtype), role="proj_dx", width=C, out_dtype=x.dtype, alpha=1).view(x.shape)
    return dx, gemm(dy2.t(), x2, role="proj_dw", width=C, out_dtype=weight.dtype, alpha=1)


@linear_bwd.register_fake
def linear_bwd_fake(dy, x, weight):
    return x.new_empty(x.shape), weight.new_empty(weight.shape)


def linear_save(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def linear_grad(ctx, dy):
    return linear_bwd(dy, *ctx.saved_tensors)


linear.register_autograd(linear_grad, setup_context=linear_save)


@torch.library.custom_op("nanochat::relu2_mlp", mutates_args=(), device_types="cuda")
def relu2_mlp(x: Tensor, w_fc: Tensor, w_proj: Tensor) -> tuple[Tensor, Tensor]:
    """x (..., C), w_fc (h, C), w_proj (C, h) -> (y (..., C), pre-activation (M, h) saved for the backward)."""
    C = x.shape[-1]
    x2 = x.reshape(-1, C)
    preact, postact = gemm_act(x2, w_fc.to(x.dtype).t(), activation="relu_sq", config=TABLE[C]["mlp_up"])
    y = gemm(postact, w_proj.to(x.dtype).t(), role="mlp_down", width=C, out_dtype=x.dtype, alpha=1)
    return y.view(*x.shape[:-1], w_proj.shape[0]), preact


@relu2_mlp.register_fake
def relu2_mlp_fake(x, w_fc, w_proj):
    return x.new_empty(*x.shape[:-1], w_proj.shape[0]), x.new_empty(x.numel() // x.shape[-1], w_fc.shape[0])


@torch.library.custom_op("nanochat::relu2_mlp_bwd", mutates_args=(), device_types="cuda")
def relu2_mlp_bwd(dy: Tensor, x: Tensor, preact: Tensor, w_fc: Tensor, w_proj: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(dX like x, dW_fc, dW_proj in the weights' dtype)."""
    C = x.shape[-1]
    dy2, x2 = dy.reshape(-1, C), x.reshape(-1, C)
    dpreact, postact = gemm_dact(dy2, w_proj.to(x.dtype), preact, activation="relu_sq", config=TABLE[C]["mlp_dact"])
    dw_proj = gemm(dy2.t(), postact, role="mlp_dw_proj", width=C, out_dtype=w_proj.dtype, alpha=1)
    del postact
    dx = gemm(dpreact, w_fc.to(x.dtype), role="mlp_dx", width=C, out_dtype=x.dtype, alpha=1).view(x.shape)
    dw_fc = gemm(dpreact.t(), x2, role="mlp_dw_fc", width=C, out_dtype=w_fc.dtype, alpha=1)
    return dx, dw_fc, dw_proj


@relu2_mlp_bwd.register_fake
def relu2_mlp_bwd_fake(dy, x, preact, w_fc, w_proj):
    return x.new_empty(x.shape), w_fc.new_empty(w_fc.shape), w_proj.new_empty(w_proj.shape)


def relu2_mlp_save(ctx, inputs, output):
    ctx.save_for_backward(*inputs, output[1])


def relu2_mlp_grad(ctx, dy, dpreact):
    x, w_fc, w_proj, preact = ctx.saved_tensors
    return relu2_mlp_bwd(dy, x, preact, w_fc, w_proj)


relu2_mlp.register_autograd(relu2_mlp_grad, setup_context=relu2_mlp_save)
