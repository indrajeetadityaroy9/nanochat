"""The decoder layer's GEMMs as torch.library ops (traceable by fullgraph torch.compile): forward and dX on QuACK's
GEMMs (quack.gemm_interface and quack.epilogue fused epilogues, tuned) with the weight W in x's dtype (model/modeling.py
GemmLinear's bf16 copy of the fp32 master; another dtype is cast per call); dW accumulated in fp32 by cuBLAS straight
into the fp32 master's .grad (accumulate_dw).

Weight gradients: each op takes its weights' .grad buffers (fp32, allocated and zeroed by train/loop.py) and its
backward adds dW into them in place, so the gradient-accumulation micro-batches sum in the GEMM (beta = 1) instead of
a new dW per micro-batch plus autograd's AccumulateGrad add; autograd sees no gradient for those weights. Forward-only
callers pass None.

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


def accumulate_dw(grad: Tensor, a: Tensor, b: Tensor) -> None:
    """grad += a^T b: bf16 operands, fp32 accumulation and output, cuBLAS with beta = 1 (aten addmm.dtype_out)."""
    torch.addmm(grad, a.t(), b, out_dtype=grad.dtype, out=grad)


def grad_buffers(*grads: Tensor | None) -> tuple[Tensor, ...]:
    """The weights' .grad buffers a backward accumulates into; None: the caller allocated none (forward-only)."""
    if any(g is None for g in grads):
        raise RuntimeError("backward through model/ops/linear.py needs the weights' .grad allocated (train/loop.py)")
    return grads


@torch.library.custom_op("nanochat::linear", mutates_args=(), device_types="cuda")
def linear(x: Tensor, weight: Tensor, grad: Tensor | None) -> Tensor:
    """x (..., K), weight (N, K), grad: weight's fp32 .grad or None -> (..., N) in x's dtype."""
    y = gemm(x.reshape(-1, x.shape[-1]), weight.to(x.dtype).t(), out_dtype=x.dtype)
    return y.view(*x.shape[:-1], weight.shape[0])


@linear.register_fake
def linear_fake(x: Tensor, weight: Tensor, grad: Tensor | None) -> Tensor:
    return x.new_empty(*x.shape[:-1], weight.shape[0])


@torch.library.custom_op("nanochat::linear_bwd", mutates_args=("grad",), device_types="cuda")
def linear_bwd(dy: Tensor, x: Tensor, weight: Tensor, grad: Tensor) -> Tensor:
    """dX like x; dW accumulated into grad."""
    dy2, x2 = dy.reshape(-1, dy.shape[-1]), x.reshape(-1, x.shape[-1])
    dx = gemm(dy2, weight.to(x.dtype), out_dtype=x.dtype).view(x.shape)
    accumulate_dw(grad, dy2, x2)
    return dx


@linear_bwd.register_fake
def linear_bwd_fake(dy: Tensor, x: Tensor, weight: Tensor, grad: Tensor) -> Tensor:
    return x.new_empty(x.shape)


def linear_save(ctx: FunctionCtx, inputs: tuple[Tensor, Tensor, Tensor | None], output: Tensor) -> None:
    ctx.save_for_backward(*inputs)


def linear_grad(ctx: FunctionCtx, dy: Tensor) -> tuple[Tensor, None, None]:
    x, weight, grad = ctx.saved_tensors
    return linear_bwd(dy, x, weight, *grad_buffers(grad)), None, None


linear.register_autograd(linear_grad, setup_context=linear_save)


@torch.library.custom_op("nanochat::residual_rms", mutates_args=(), device_types="cuda")
def residual_rms(x: Tensor, weight: Tensor, grad: Tensor | None, residual: Tensor) -> tuple[Tensor, Tensor]:
    """x (..., K), weight (N, K), grad its fp32 .grad or None, residual (..., N) -> (h = residual + x W^T in x's dtype,
    rstd (M,) fp32 of h's rows before rounding). rstd is non-differentiable: relu2_mlp, its one consumer,
    differentiates through it."""
    N = weight.shape[0]
    r = sq_reduce_mod(has_c=True, has_rowvec=False, has_aux=False)(
        x.reshape(-1, x.shape[-1]), weight.to(x.dtype).t(), residual.reshape(-1, N))
    # gemm_rms's epilogue and finalize (quack/gemm_interface.py _gemm_rms_call, _rms_finalize); the per-tile partials
    # of a cold autotuned call are a strided slice, which rms_final_reduce rejects, hence contiguous()
    rstd = rms_final_reduce(r["mColVecReduce"].contiguous(), scale=1 / N, eps=EPS)
    return r["D"].view(residual.shape), rstd


@residual_rms.register_fake
def residual_rms_fake(x: Tensor, weight: Tensor, grad: Tensor | None, residual: Tensor) -> tuple[Tensor, Tensor]:
    return residual.new_empty(residual.shape), x.new_empty(residual.numel() // weight.shape[0], dtype=torch.float32)


def residual_rms_save(ctx: FunctionCtx, inputs: tuple[Tensor, Tensor, Tensor | None, Tensor],
                      output: tuple[Tensor, Tensor]) -> None:
    ctx.mark_non_differentiable(output[1])
    ctx.save_for_backward(*inputs[:3])


def residual_rms_grad(ctx: FunctionCtx, dh: Tensor, drstd: None) -> tuple[Tensor, None, None, Tensor]:
    x, weight, grad = ctx.saved_tensors
    return linear_bwd(dh, x, weight, *grad_buffers(grad)), None, None, dh


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
def relu2_mlp(h: Tensor, rstd: Tensor, w_up: Tensor, w_down: Tensor, g_up: Tensor | None,
              g_down: Tensor | None) -> tuple[Tensor, Tensor]:
    """h (..., C) the un-normed stream, rstd (M,) fp32 its rows' (residual_rms), w_up (4C, C), w_down (C, 4C), g_up,
    g_down their fp32 .grad or None -> (h + MLP(norm(h)) like h, scaled pre-activation (M, 4C) saved for the
    backward)."""
    h2 = h.reshape(-1, h.shape[-1])
    preact, postact = gemm_norm_act(h2, w_up.to(h.dtype).t(), rstd=rstd, activation="relu_sq", store_preact=True)
    y = gemm_add(postact, w_down.to(h.dtype).t(), C=h2)
    return y.view(h.shape), preact


@relu2_mlp.register_fake
def relu2_mlp_fake(h: Tensor, rstd: Tensor, w_up: Tensor, w_down: Tensor, g_up: Tensor | None,
                   g_down: Tensor | None) -> tuple[Tensor, Tensor]:
    return h.new_empty(h.shape), h.new_empty(h.numel() // h.shape[-1], w_up.shape[0])


@torch.library.custom_op("nanochat::relu2_mlp_bwd", mutates_args=("g_up", "g_down"), device_types="cuda")
def relu2_mlp_bwd(dy: Tensor, h: Tensor, rstd: Tensor, preact: Tensor, w_up: Tensor, w_down: Tensor, g_up: Tensor,
                  g_down: Tensor) -> Tensor:
    """dh like h, through the residual, the MLP and the norm; dW_up, dW_down accumulated into g_up, g_down."""
    dy2, h2 = dy.reshape(-1, dy.shape[-1]), h.reshape(-1, h.shape[-1])
    r = relu2_dact_rstd(dy2, w_down.to(h.dtype), preact, rstd=rstd, dynamic_scheduler=True)
    dp, postact = r["D"], r["postact"]
    accumulate_dw(g_down, dy2, postact)
    del postact
    corr = rstd.square().mul_(r["dsum"]).mul_(-1 / h.shape[-1])
    dh = norm_bwd_apply(dp, w_up.to(h.dtype), dy2, h=h2, corr=corr)["D"].view(h.shape)
    accumulate_dw(g_up, dp, h2)
    return dh


@relu2_mlp_bwd.register_fake
def relu2_mlp_bwd_fake(dy: Tensor, h: Tensor, rstd: Tensor, preact: Tensor, w_up: Tensor, w_down: Tensor,
                       g_up: Tensor, g_down: Tensor) -> Tensor:
    return h.new_empty(h.shape)


def relu2_mlp_save(ctx: FunctionCtx, inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None],
                   output: tuple[Tensor, Tensor]) -> None:
    ctx.save_for_backward(*inputs, output[1])


def relu2_mlp_grad(ctx: FunctionCtx, dy: Tensor, dpreact: Tensor) -> tuple[Tensor, None, None, None, None, None]:
    h, rstd, w_up, w_down, g_up, g_down, preact = ctx.saved_tensors
    return relu2_mlp_bwd(dy, h, rstd, preact, w_up, w_down, *grad_buffers(g_up, g_down)), None, None, None, None, None


relu2_mlp.register_autograd(relu2_mlp_grad, setup_context=relu2_mlp_save)
