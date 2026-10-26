"""
LM-head next-token loss from hidden states x (M, C) and head weight W (V, C), z = x W^T:

    loss = sum over rows with label t != IGNORE_INDEX of (logsumexp(z) - z[t]), divided by the valid count n.

nanochat::lm_loss (training), uncapped: QuACK's scaled-exp linear cross entropy (quack/linear_cross_entropy.py,
scaled_exp_linear_cross_entropy_fwd's calls and tiles, run here per LCE_CHUNK rows so that each row's label logit is kept
for its logsumexp): the logits GEMM's epilogue (quack/epilogue/scaled_exp.py scaled_exp_target_epi) stores
E = bf16(2^(z log2 e - k)), k the rounded maximum of the row's TILE_N1-wide N tile, beside each tile's sum of
exponentials and the label's fp32 logit, never z itself; QuACK's row kernel (lce_glue) turns those into the loss
(logsumexp - z[t], from the fp32 accumulators), fixes E at the label and writes each row's power-of-two rescalings;
dx = (1/n) dz W (bf16) and dW = (1/n) dz^T x (fp32, accumulated across chunks) are GEMMs on E whose A-operand load
applies the rescalings (dz = v (u E) row-wise, u exact powers of two, v fp32). The logits are written once and read
once per gradient GEMM, with no softmax pass over them.
With a cap: a GEMM writes bf16 z; QuACK's cross_entropy_fwd_out (quack/cross_entropy.py, dx = z as with
inplace_backward) writes each row's loss and logsumexp and overwrites z with softmax(z) - onehot(t); dx = (1/n) z W on
QuACK (bf16), dW = (1/n) z^T x by torch.mm in fp32 (as model/ops/linear.py's weight gradients).

nanochat::lm_losses (evaluation): QuACK's lse_target_epi GEMM epilogue (quack/epilogue/library.py) reduces each row's
logsumexp and gathers the label's logit from the fp32 accumulators without storing z, so no bf16 rounding of the
logits enters the loss.

With an output-logit cap c (model/modeling.py ModelConfig.final_logit_softcapping; 92d63d4 gpt.py:511-515, Gemma 2's
final-logit softcapping, arXiv 2408.00118 §2) z becomes c tanh(z / c) (tanh in fp32), in kernels Inductor fuses
(cap_, capped_losses): lm_loss caps z in place (bf16) beside the cap's derivative 1 - tanh(z / c)^2 (bf16), two
logits-sized buffers, runs QuACK's cross entropy on the capped logits and multiplies its logit gradient by the
derivative before the two gradient GEMMs; lm_losses takes logsumexp minus the label's logit of the capped fp32 logits
of each row chunk's GEMM (CHUNK rows), as lse_target_epi and scaled_exp_target_epi have no tanh epilogue.
"""

import torch
from quack.cross_entropy import cross_entropy_fwd_out
from quack.epilogue.library import identity_epi, lse_target_epi
from quack.epilogue.scaled_exp import scaled_exp_target_epi
from quack.gemm_interface import gemm
from quack.linear_cross_entropy import _lce_dw_scale, _lce_dx_scale, _lce_vscale_epi, lce_glue
from quack.operand_transform.host import transform_a_operand
from torch import Tensor
from torch.autograd.function import FunctionCtx

from corpus.loader import IGNORE_INDEX

# Rows of lm_losses' capped fp32 logits per GEMM, a memory bound only (each row's loss is computed whole within its
# chunk, so the losses do not depend on it): 8,192 x 2^15 x 4 bytes = 1 GiB
CHUNK = 8192
# Rows of lm_loss's scaled-exp chunks, a multiple of 128 (the gradient GEMMs' tile_M). On a real d24 micro-batch
# (98,304 rows, GH200): 8,192 to 65,536 rows equally fast (24.1-24.4 ms per call), E (LCE_CHUNK x 2^15 bf16) and the
# op's peak growing with it (0.82 GiB at 8,192, 4.59 at 65,536); a 98,304-row chunk (E past 2^31 elements) faults in
# QuACK 0.6.5. The least memory of the equally fast.
LCE_CHUNK = 8192
# The logits GEMM's N tile, each row's k and sum-of-exponentials granularity: a multiple of 64 (lce_glue's strip is
# 64-column resolved) dividing the vocabulary 2^15, with QuACK's pingpong schedule (two consumer warp groups alternating
# tiles, hiding the epilogue; tile_N <= 208). Same micro-batch: 128 with pingpong 24.2 ms per call, 256 (no pingpong)
# 25.5, master's GEMM + cross entropy + gradient GEMMs 26.6-27.1.
TILE_N1 = 128
# The gradient GEMMs' N tile over C, QuACK's rule (scaled_exp_linear_cross_entropy_fwd): 192 when it divides C (768)
TILE_N2 = 192


@torch.compile(dynamic=False, fullgraph=True)
def cap_(z: Tensor, c: float) -> Tensor:
    """z = c tanh(z / c) in place, one kernel; returns its derivative 1 - tanh(z / c)^2; each in z's dtype."""
    t = torch.tanh(z.float() / c)
    z.copy_((c * t).to(z.dtype))
    return (1 - t.square()).to(z.dtype)


@torch.compile(dynamic=False, fullgraph=True)
def capped_losses(z: Tensor, labels: Tensor, c: float) -> Tensor:
    """logsumexp minus the label's entry of the rows of c tanh(z / c), z fp32."""
    z = c * torch.tanh(z / c)
    return torch.logsumexp(z, dim=1) - z.gather(1, labels[:, None]).squeeze(1)


def scaled_exp_loss(x: Tensor, w: Tensor, labels: Tensor, scale: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Uncapped lm_loss on QuACK's scaled-exp pipeline, x (M, C) and w (V, C) bf16 -> (per-position loss, dx bf16,
    dW fp32, per-position logsumexp; 0 at IGNORE_INDEX labels), the gradients times scale. One chunk's buffers serve
    every chunk (one stream orders them)."""
    (M, C), V = x.shape, w.size(0)
    if M % 128:
        raise ValueError(f"lm_loss: {M} rows, not a multiple of 128 (the gradient GEMMs' tile_M)")
    rows = min(LCE_CHUNK, M)
    loss, zy = x.new_empty(M, dtype=torch.float32), x.new_empty(M, dtype=torch.float32)
    dx, dw = torch.empty_like(x), x.new_empty((V, C), dtype=torch.float32)
    E, xs, v = x.new_empty((rows, V)), x.new_empty((rows, C)), x.new_empty(rows, dtype=torch.float32)
    sum_exp, k = (x.new_empty((rows, V // TILE_N1), dtype=torch.float32) for _ in range(2))
    for i, start in enumerate(range(0, M, LCE_CHUNK)):
        c = slice(start, min(start + LCE_CHUNK, M))
        n = c.stop - start
        x_c, t_c, E_c, xs_c, v_c = x[c], labels[c], E[:n], xs[:n], v[:n]
        strip = x.new_empty((V // 64, n))  # the chunk's u, contiguous along its rows
        scaled_exp_target_epi.gemm(x_c, w, E_c, epi_args={"max_log2": TILE_N1, "sum_exp": sum_exp[:n],
                                                           "max_log2_out": k[:n], "target": t_c,
                                                           "target_logit": zy[c]},
                                   tile_M=128, tile_N=TILE_N1, cluster_M=2, cluster_N=1, pingpong=True)
        lce_glue(k[:n], sum_exp[:n], x_c, t_c, zy[c], E_c, loss[c], strip, v_c, xs_c, TILE_N1,
                 ignore_index=IGNORE_INDEX, grad_scale=scale)
        _lce_vscale_epi.gemm(transform_a_operand(_lce_dx_scale, E_c, {"u": strip}, 128, 64), w.mT, dx[c],
                             epi_args={"v": v_c.unsqueeze(0)}, transform_a=_lce_dx_scale, tile_M=128, tile_N=TILE_N2,
                             tile_K=64, cluster_M=2, cluster_N=1)
        identity_epi.gemm(transform_a_operand(_lce_dw_scale, E_c.t(), {"u": strip}, 128, 64), xs_c.mT, dw,
                          epi_args={}, transform_a=_lce_dw_scale, add_to_output=i > 0, tile_M=128, tile_N=TILE_N2,
                          tile_K=64, cluster_M=1, cluster_N=2)
    return loss, dx, dw, torch.where(labels != IGNORE_INDEX, loss + zy, 0.0)


# Opaque to torch.compile: its functionalization of the in-place logit gradient (dx = z) gives x a wrong gradient.
@torch.library.custom_op("nanochat::lm_loss", mutates_args=(), device_types="cuda")
def lm_loss(x: Tensor, weight: Tensor, labels: Tensor,
            final_logit_softcapping: float | None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """x (M, C) bf16, M a multiple of 128, fp32 weight (V, C), labels (M,) int64 -> (mean loss fp32, dloss/dx in x's
    dtype, dloss/dW fp32, per-position loss (M,) fp32 (0 at IGNORE_INDEX labels), per-position logsumexp of the
    (capped) logits (M,) fp32, uncapped 0 at IGNORE_INDEX labels); the last two are QuACK's per-row outputs, without
    gradient."""
    w = weight.to(x.dtype)
    scale = 1 / (labels != IGNORE_INDEX).sum().float()
    if final_logit_softcapping is None:
        loss, dx, dw, lse = scaled_exp_loss(x, w, labels, scale)
        return loss.sum() * scale, dx, dw, loss, lse
    logits = x @ w.t()
    loss = torch.empty(labels.shape, device=x.device, dtype=torch.float32)
    lse = torch.empty(labels.shape, device=x.device, dtype=torch.float32)
    derivative = cap_(logits, final_logit_softcapping)
    cross_entropy_fwd_out(logits, labels, target_logit=None, loss=loss, lse=lse, dx=logits, weight=None,
                          ignore_index=IGNORE_INDEX)
    logits.mul_(derivative)
    return (loss.sum() * scale, gemm(logits, w, alpha=scale, out_dtype=x.dtype),
            torch.mm(logits.t(), x, out_dtype=weight.dtype) * scale, loss, lse)


@lm_loss.register_fake
def lm_loss_fake(x: Tensor, weight: Tensor, labels: Tensor,
                 final_logit_softcapping: float | None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (x.new_empty((), dtype=torch.float32), x.new_empty(x.shape), weight.new_empty(weight.shape),
            x.new_empty(labels.shape, dtype=torch.float32), x.new_empty(labels.shape, dtype=torch.float32))


def lm_loss_save(ctx: FunctionCtx, inputs: tuple[Tensor, Tensor, Tensor, float | None],
                 output: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> None:
    ctx.save_for_backward(output[1], output[2])


def lm_loss_grad(ctx: FunctionCtx, dloss: Tensor, ddx: Tensor, ddw: Tensor, dlosses: Tensor,
                 dlse: Tensor) -> tuple[Tensor, Tensor, None, None]:
    """Saved gradients times dloss; dx's product in fp32, rounded once to bf16."""
    dx, dw = ctx.saved_tensors
    return (dx.float() * dloss).to(dx.dtype), dw * dloss, None, None


lm_loss.register_autograd(lm_loss_grad, setup_context=lm_loss_save)


def step_shares(labels: list[Tensor]) -> Tensor:
    """Each micro-batch's share of a step's valid (non-IGNORE_INDEX) labels, fp32: lm_loss's per-micro-batch means
    weighted by it sum to the mean over every valid label of the step."""
    valid = torch.stack([(t != IGNORE_INDEX).sum() for t in labels]).double()
    return (valid / valid.sum()).float()


@torch.library.custom_op("nanochat::lm_losses", mutates_args=(), device_types="cuda")
def lm_losses(x: Tensor, weight: Tensor, labels: Tensor, final_logit_softcapping: float | None) -> Tensor:
    """Per-position loss (M,) fp32, 0 at IGNORE_INDEX labels; no gradient. QuACK's default tile config
    (tuned=False)."""
    valid, w = labels != IGNORE_INDEX, weight.to(x.dtype).t()
    if final_logit_softcapping is None:
        res = lse_target_epi(x, w, store_d=False, target=labels, tuned=False)
        return torch.where(valid, res["lse"] - res["target_logit"], 0.0)
    rows = torch.where(valid, labels, 0)
    losses = [capped_losses(torch.mm(xc, w, out_dtype=torch.float32), tc, final_logit_softcapping)
              for xc, tc in zip(x.split(CHUNK), rows.split(CHUNK), strict=True)]
    return torch.where(valid, torch.cat(losses), 0.0)


@lm_losses.register_fake
def lm_losses_fake(x: Tensor, weight: Tensor, labels: Tensor, final_logit_softcapping: float | None) -> Tensor:
    return x.new_empty(labels.shape, dtype=torch.float32)
