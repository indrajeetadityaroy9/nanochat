"""The decoder layer's GEMMs as torch.library ops (traceable by fullgraph torch.compile): forward and dX on QuACK's
GEMMs (quack.gemm_interface and quack.epilogue fused epilogues, tuned), fp32 master weight W rounded to x's dtype; dW
accumulated and stored in fp32 by torch.mm (cuBLAS).

nanochat::linear (q_proj, k_proj, v_proj): y = x W^T; backward dX = dY W, dW = dY^T X.

The post-attention half of a layer runs on the un-normed stream: the norm is gain-free, so
norm(h) W^T = diag(rstd) (h W^T) exactly, and rstd rides the GEMM epilogues as a per-row (colvec) scale; the normed
tensor is never stored.
- nanochat::residual_rms (o_proj): h = residual + x W^T and rstd = (mean(h^2) + eps)^-1/2 from the same epilogue
  (quack.gemm_interface.gemm_rms: per-tile squared sums, finalized by quack.rms_final_reduce).
- nanochat::relu2_mlp: h + down_proj(relu(rstd up_proj h)^2), the MLP of karpathy/nanochat 92d63d4 gpt.py:131-141 on
  the normed stream plus its residual: rstd and relu^2 in the up-projection's epilogue
  (quack.gemm_interface.gemm_norm_act, which stores the scaled pre-activation), the residual as the
  down-projection's C (quack.gemm_interface.gemm_add). Its backward owns the norm's backward (residual_rms's rstd
  gets no gradient of its own): with P = h W_up^T, preact = rstd P and dpreact from relu^2's derivative,
  dP = rstd dpreact, d rstd = sum(dpreact preact) / rstd and d rstd/dh = -rstd^3 h / C, so
  dh = dy + dP W_up - (rstd^2 sum(dpreact preact) / C) h; the row sum is a reduce of the dact epilogue, the rest the
  dX GEMM's epilogue (C = dy, h a tile load)."""
import torch
from quack.activation import drelu_sq
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.library import sq_reduce_mod
from quack.epilogue.ops import ColVecLoad, ColVecReduce
from quack.gemm_interface import gemm, gemm_add, gemm_norm_act
from quack.rms_final_reduce import rms_final_reduce
from torch import Tensor
from torch.autograd.function import FunctionCtx

# model/modeling.py norm's epsilon
EPS = torch.finfo(torch.float32).eps


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


@torch.library.custom_op("nanochat::residual_rms", mutates_args=(), device_types="cuda")
def residual_rms(x: Tensor, weight: Tensor, residual: Tensor) -> tuple[Tensor, Tensor]:
    """x (..., K), weight (N, K), residual (..., N) -> (h = residual + x W^T in x's dtype, rstd (M,) fp32 of h's
    rows before rounding). rstd is non-differentiable: relu2_mlp, its one consumer, differentiates through it."""
    N = weight.shape[0]
    r = sq_reduce_mod(has_c=True, has_rowvec=False, has_aux=False)(
        x.reshape(-1, x.shape[-1]), weight.to(x.dtype).t(), residual.reshape(-1, N))
    # gemm_rms's epilogue and finalize (quack/gemm_interface.py _gemm_rms_call, _rms_finalize); the per-tile partials
    # of a cold autotuned call are a strided slice, which rms_final_reduce rejects, hence contiguous()
    rstd = rms_final_reduce(r["mColVecReduce"].contiguous(), scale=1 / N, eps=EPS)
    return r["D"].view(residual.shape), rstd


@residual_rms.register_fake
def residual_rms_fake(x: Tensor, weight: Tensor, residual: Tensor) -> tuple[Tensor, Tensor]:
    return residual.new_empty(residual.shape), x.new_empty(residual.numel() // weight.shape[0], dtype=torch.float32)


def residual_rms_save(ctx: FunctionCtx, inputs: tuple[Tensor, Tensor, Tensor], output: tuple[Tensor, Tensor]) -> None:
    ctx.mark_non_differentiable(output[1])
    ctx.save_for_backward(inputs[0], inputs[1])


def residual_rms_grad(ctx: FunctionCtx, dh: Tensor, drstd: None) -> tuple[Tensor, Tensor, Tensor]:
    return *linear_bwd(dh, *ctx.saved_tensors), dh


residual_rms.register_autograd(residual_rms_grad, setup_context=residual_rms_save)


@gemm_epilogue(outputs=("postact",), ops={"rstd": ColVecLoad("rstd")},
               reduces={"dsum": ColVecReduce("dsum", scaled=True)})
def relu2_dact_rstd(acc, c, rstd):
    """acc = dY W_down (d postact), c = the stored preact: D = dP = rstd dpreact, postact recomputed (dW_down's
    operand), dsum = sum over the row of dpreact preact."""
    dpreact, postact = drelu_sq(c, acc)
    return {"D": dpreact * rstd, "postact": postact, "dsum": (dpreact, c)}


@gemm_epilogue(ops={"corr": ColVecLoad("corr")})
def norm_bwd_apply(acc, c, h, corr):
    """acc = dP W_up, c = dy (the residual's gradient), h the stream, corr = -rstd^2 dsum / C: D = dh."""
    return {"D": acc + c + h * corr}


@torch.library.custom_op("nanochat::relu2_mlp", mutates_args=(), device_types="cuda")
def relu2_mlp(h: Tensor, rstd: Tensor, w_up: Tensor, w_down: Tensor) -> tuple[Tensor, Tensor]:
    """h (..., C) the un-normed stream, rstd (M,) fp32 its rows' (residual_rms), w_up (4C, C), w_down (C, 4C) ->
    (h + MLP(norm(h)) like h, scaled pre-activation (M, 4C) saved for the backward)."""
    h2 = h.reshape(-1, h.shape[-1])
    preact, postact = gemm_norm_act(h2, w_up.to(h.dtype).t(), rstd=rstd, activation="relu_sq", store_preact=True)
    y = gemm_add(postact, w_down.to(h.dtype).t(), C=h2)
    return y.view(h.shape), preact


@relu2_mlp.register_fake
def relu2_mlp_fake(h: Tensor, rstd: Tensor, w_up: Tensor, w_down: Tensor) -> tuple[Tensor, Tensor]:
    return h.new_empty(h.shape), h.new_empty(h.numel() // h.shape[-1], w_up.shape[0])


@torch.library.custom_op("nanochat::relu2_mlp_bwd", mutates_args=(), device_types="cuda")
def relu2_mlp_bwd(dy: Tensor, h: Tensor, rstd: Tensor, preact: Tensor, w_up: Tensor,
                  w_down: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(dh like h, through the residual, the MLP and the norm; dW_up, dW_down in the weights' dtype)."""
    dy2, h2 = dy.reshape(-1, dy.shape[-1]), h.reshape(-1, h.shape[-1])
    r = relu2_dact_rstd(dy2, w_down.to(h.dtype), preact, rstd=rstd, dynamic_scheduler=True)
    dp, postact = r["D"], r["postact"]
    dw_down = torch.mm(dy2.t(), postact, out_dtype=w_down.dtype)
    del postact
    corr = rstd.square().mul_(r["dsum"]).mul_(-1 / h.shape[-1])
    dh = norm_bwd_apply(dp, w_up.to(h.dtype), dy2, h=h2, corr=corr)["D"].view(h.shape)
    dw_up = torch.mm(dp.t(), h2, out_dtype=w_up.dtype)
    return dh, dw_up, dw_down


@relu2_mlp_bwd.register_fake
def relu2_mlp_bwd_fake(dy: Tensor, h: Tensor, rstd: Tensor, preact: Tensor, w_up: Tensor,
                       w_down: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return h.new_empty(h.shape), w_up.new_empty(w_up.shape), w_down.new_empty(w_down.shape)


def relu2_mlp_save(ctx: FunctionCtx, inputs: tuple[Tensor, Tensor, Tensor, Tensor],
                   output: tuple[Tensor, Tensor]) -> None:
    ctx.save_for_backward(*inputs, output[1])


def relu2_mlp_grad(ctx: FunctionCtx, dy: Tensor, dpreact: Tensor) -> tuple[Tensor, None, Tensor, Tensor]:
    h, rstd, w_up, w_down, preact = ctx.saved_tensors
    dh, dw_up, dw_down = relu2_mlp_bwd(dy, h, rstd, preact, w_up, w_down)
    return dh, None, dw_up, dw_down


relu2_mlp.register_autograd(relu2_mlp_grad, setup_context=relu2_mlp_save)
