"""The decoder layer's GEMMs as torch.library ops (traceable by fullgraph torch.compile): forward and dX on QuACK's
GEMMs (quack.gemm_interface, tuned=True), fp32 master weight W rounded to x's dtype; dW accumulated and stored in fp32
by torch.mm (cuBLAS).

nanochat::linear (q_proj, k_proj, v_proj, o_proj): y = x W^T; backward dX = dY W, dW = dY^T X.

nanochat::relu2_mlp: down_proj(relu(up_proj x)^2) (karpathy/nanochat 92d63d4 gpt.py:131-141) on the kernels of
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
def relu2_mlp(x: Tensor, w_up: Tensor, w_down: Tensor) -> tuple[Tensor, Tensor]:
    """x (..., C), w_up (h, C), w_down (C, h) -> (y (..., C), pre-activation (M, h) saved for the backward)."""
    x2 = x.reshape(-1, x.shape[-1])
    preact, postact = gemm_act(x2, w_up.to(x.dtype).t(), activation="relu_sq")
    y = gemm(postact, w_down.to(x.dtype).t(), out_dtype=x.dtype)
    return y.view(*x.shape[:-1], w_down.shape[0]), preact


@relu2_mlp.register_fake
def relu2_mlp_fake(x: Tensor, w_up: Tensor, w_down: Tensor) -> tuple[Tensor, Tensor]:
    return x.new_empty(*x.shape[:-1], w_down.shape[0]), x.new_empty(x.numel() // x.shape[-1], w_up.shape[0])


@torch.library.custom_op("nanochat::relu2_mlp_bwd", mutates_args=(), device_types="cuda")
def relu2_mlp_bwd(dy: Tensor, x: Tensor, preact: Tensor, w_up: Tensor, w_down: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(dX like x, dW_up, dW_down in the weights' dtype)."""
    dy2, x2 = dy.reshape(-1, dy.shape[-1]), x.reshape(-1, x.shape[-1])
    dpreact, postact = gemm_dact(dy2, w_down.to(x.dtype), preact, activation="relu_sq")
    dw_down = torch.mm(dy2.t(), postact, out_dtype=w_down.dtype)
    del postact
    dx = gemm(dpreact, w_up.to(x.dtype), out_dtype=x.dtype).view(x.shape)
    dw_up = torch.mm(dpreact.t(), x2, out_dtype=w_up.dtype)
    return dx, dw_up, dw_down


@relu2_mlp_bwd.register_fake
def relu2_mlp_bwd_fake(dy: Tensor, x: Tensor, preact: Tensor, w_up: Tensor,
                       w_down: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return x.new_empty(x.shape), w_up.new_empty(w_up.shape), w_down.new_empty(w_down.shape)


def relu2_mlp_save(ctx: FunctionCtx, inputs: tuple[Tensor, Tensor, Tensor], output: tuple[Tensor, Tensor]) -> None:
    ctx.save_for_backward(*inputs, output[1])


def relu2_mlp_grad(ctx: FunctionCtx, dy: Tensor, dpreact: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    x, w_up, w_down, preact = ctx.saved_tensors
    return relu2_mlp_bwd(dy, x, preact, w_up, w_down)


relu2_mlp.register_autograd(relu2_mlp_grad, setup_context=relu2_mlp_save)
