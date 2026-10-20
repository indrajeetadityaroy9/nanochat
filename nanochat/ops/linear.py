"""The block's GEMMs as torch.library ops (traceable by fullgraph torch.compile): forward and dX on QuACK's autotuned
SM90 kernels (quack.gemm_interface, tuned=True), fp32 master weight W rounded to x's dtype; dW in fp32 on cuBLAS
(torch.mm), whose split over K = M tokens beats QuACK's tiles on these small outputs in time and accuracy (d24 c_q dW
0.19 vs 0.54 ms, rel. error 1.8e-5 vs 2.3e-4; ablations/ablations.md).

nanochat::linear (c_q, c_k, c_v, c_o): y = x W^T; backward dX = dY W, dW = dY^T X.

nanochat::relu2_mlp: c_proj(relu(c_fc x)^2) (karpathy/nanochat 92d63d4 gpt.py:131-141) on the kernels of
quack.mlp.mlp_func(activation="relu_sq"): relu^2 in the up-projection's epilogue (gemm_act, which stores the
pre-activation), its derivative and the recomputed post-activation in the backward's gemm_dact."""
import torch
from quack.gemm_interface import gemm, gemm_act, gemm_dact
from torch import Tensor
from torch.autograd.function import FunctionCtx


@torch.library.custom_op("nanochat::linear", mutates_args=(), device_types="cuda")
def linear(x: Tensor, weight: Tensor) -> Tensor:
    """x (..., K), weight (N, K) -> (..., N) in x's dtype."""
    y = gemm(x.reshape(-1, x.shape[-1]), weight.to(x.dtype).t(), out_dtype=x.dtype)
    return y.view(*x.shape[:-1], weight.shape[0])


@linear.register_fake
def linear_fake(x: Tensor, weight: Tensor) -> Tensor:
    return x.new_empty(*x.shape[:-1], weight.shape[0])


@torch.library.custom_op("nanochat::linear_bwd", mutates_args=(), device_types="cuda")
def linear_bwd(dy: Tensor, x: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
    """(dX like x, dW in weight's dtype)."""
    dy2, x2 = dy.reshape(-1, dy.shape[-1]), x.reshape(-1, x.shape[-1])
    dx = gemm(dy2, weight.to(x.dtype), out_dtype=x.dtype).view(x.shape)
    return dx, torch.mm(dy2.t(), x2, out_dtype=weight.dtype)


@linear_bwd.register_fake
def linear_bwd_fake(dy: Tensor, x: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
    return x.new_empty(x.shape), weight.new_empty(weight.shape)


def linear_save(ctx: FunctionCtx, inputs: tuple[Tensor, Tensor], output: Tensor) -> None:
    ctx.save_for_backward(*inputs)


def linear_grad(ctx: FunctionCtx, dy: Tensor) -> tuple[Tensor, Tensor]:
    return linear_bwd(dy, *ctx.saved_tensors)


linear.register_autograd(linear_grad, setup_context=linear_save)


@torch.library.custom_op("nanochat::relu2_mlp", mutates_args=(), device_types="cuda")
def relu2_mlp(x: Tensor, w_fc: Tensor, w_proj: Tensor) -> tuple[Tensor, Tensor]:
    """x (..., C), w_fc (h, C), w_proj (C, h) -> (y (..., C), pre-activation (M, h) saved for the backward)."""
    x2 = x.reshape(-1, x.shape[-1])
    preact, postact = gemm_act(x2, w_fc.to(x.dtype).t(), activation="relu_sq")
    y = gemm(postact, w_proj.to(x.dtype).t(), out_dtype=x.dtype)
    return y.view(*x.shape[:-1], w_proj.shape[0]), preact


@relu2_mlp.register_fake
def relu2_mlp_fake(x: Tensor, w_fc: Tensor, w_proj: Tensor) -> tuple[Tensor, Tensor]:
    return x.new_empty(*x.shape[:-1], w_proj.shape[0]), x.new_empty(x.numel() // x.shape[-1], w_fc.shape[0])


@torch.library.custom_op("nanochat::relu2_mlp_bwd", mutates_args=(), device_types="cuda")
def relu2_mlp_bwd(dy: Tensor, x: Tensor, preact: Tensor, w_fc: Tensor, w_proj: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(dX like x, dW_fc, dW_proj in the weights' dtype)."""
    dy2, x2 = dy.reshape(-1, dy.shape[-1]), x.reshape(-1, x.shape[-1])
    dpreact, postact = gemm_dact(dy2, w_proj.to(x.dtype), preact, activation="relu_sq")
    dw_proj = torch.mm(dy2.t(), postact, out_dtype=w_proj.dtype)
    del postact
    dx = gemm(dpreact, w_fc.to(x.dtype), out_dtype=x.dtype).view(x.shape)
    dw_fc = torch.mm(dpreact.t(), x2, out_dtype=w_fc.dtype)
    return dx, dw_fc, dw_proj


@relu2_mlp_bwd.register_fake
def relu2_mlp_bwd_fake(dy: Tensor, x: Tensor, preact: Tensor, w_fc: Tensor,
                       w_proj: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return x.new_empty(x.shape), w_fc.new_empty(w_fc.shape), w_proj.new_empty(w_proj.shape)


def relu2_mlp_save(ctx: FunctionCtx, inputs: tuple[Tensor, Tensor, Tensor], output: tuple[Tensor, Tensor]) -> None:
    ctx.save_for_backward(*inputs, output[1])


def relu2_mlp_grad(ctx: FunctionCtx, dy: Tensor, dpreact: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    x, w_fc, w_proj, preact = ctx.saved_tensors
    return relu2_mlp_bwd(dy, x, preact, w_fc, w_proj)


relu2_mlp.register_autograd(relu2_mlp_grad, setup_context=relu2_mlp_save)
