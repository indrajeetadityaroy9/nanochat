"""
The language-model head's next-token loss straight from the hidden states x (M, C) and the head's weight W (V, C),
z = x W^T:

    loss = (sum over the rows whose target t is not the loader's padding of (logsumexp(z) - z[t])) / n,

n the valid count; the padding target PretrainingBatches.PAD (data/loader.py) is the ignore index.

Training (nanochat::lm_loss): one GEMM writes the bf16 logits z; one pass of QuACK's CrossEntropy kernel (Tri Dao,
CuTe DSL; quack/cross_entropy.py cross_entropy_fwd_out, given dx = z as its cross_entropy_fwd does with
inplace_backward) reads each row once, writes the row's loss and, in place of z, the logit gradient relative to the
scale 1 / n, softmax(z) - onehot(t) (0 in ignored rows); two GEMMs take dx (bf16) and dW (fp32, straight from the
accumulators) from it with alpha = 1 / n. The backward's product of dx with dloss rounds dx to bf16 a second time; an
fp32 dx (same GEMM time at d8, benchmarks/h100.txt) would change this op's output dtype, which the shared Inductor cache
does not key on, so it is not taken. QuACK's fused linear-cross-entropy (quack/linear_cross_entropy.py
scaled_exp_linear_cross_entropy_fwd) is not faster at d8 and also returns dx in bf16 (benchmarks/h100.txt).

Evaluation (nanochat::lm_losses): QuACK's lse_target_epi GEMM epilogue (quack/epilogue/library.py, as its
linear_cross_entropy's loss-only path) reduces each row's log-sum-exp and gathers the target logit from the fp32
accumulators at QuACK's default SM90 config; the logits are never stored (benchmarks/h100.txt).
"""

import torch
from quack.cross_entropy import cross_entropy_fwd_out
from quack.epilogue.library import lse_target_epi
from torch import Tensor

from nanochat.data.loader import PretrainingBatches
from nanochat.ops.quack_gemm import gemm


# Opaque to torch.compile, as ops/flash_attn.py's FA4 ops are: the logit gradient overwrites the logits in place, an
# aliasing a traced graph must not see (functionalized, it corrupted the gradient of x).
@torch.library.custom_op("nanochat::lm_loss", mutates_args=(), device_types="cuda")
def lm_loss(x: Tensor, weight: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """x (M, C) bf16, weight (V, C) the fp32 master weight, rounded to x's dtype for the GEMMs, target (M,) int64: (the
    mean loss over the valid targets (scalar fp32), its gradient wrt x in x's dtype, its gradient wrt weight in
    weight's dtype)."""
    w = weight.to(x.dtype)
    logits = x @ w.t()
    scale = 1 / (target != PretrainingBatches.PAD).sum().float()
    loss = torch.empty(target.shape, device=x.device, dtype=torch.float32)
    cross_entropy_fwd_out(logits, target, None, loss, None, logits, None, PretrainingBatches.PAD)
    C = x.shape[1]
    return (loss.sum() * scale, gemm(logits, w, "lm_dx", C, x.dtype, scale),
            gemm(logits.t(), x, "lm_dw", C, weight.dtype, scale))


@lm_loss.register_fake
def _(x, weight, target):
    return x.new_empty((), dtype=torch.float32), x.new_empty(x.shape), weight.new_empty(weight.shape)


def _lm_loss_setup(ctx, inputs, output):
    ctx.save_for_backward(output[1], output[2])


def _lm_loss_backward(ctx, dloss, _ddx, _ddw):
    """The saved gradients times dloss (the trainer's micro-batch share of the step's targets): dx's product in fp32,
    rounded to bf16."""
    dx, dw = ctx.saved_tensors
    return (dx.float() * dloss).to(dx.dtype), dw * dloss, None


lm_loss.register_autograd(_lm_loss_backward, setup_context=_lm_loss_setup)


@torch.library.custom_op("nanochat::lm_losses", mutates_args=(), device_types="cuda")
def lm_losses(x: Tensor, weight: Tensor, target: Tensor) -> Tensor:
    """Each position's loss (M,) fp32 against its target (M,), 0 at the padding target; no gradient. weight: the master
    weight (V, C), rounded to x's dtype for the GEMM."""
    res = lse_target_epi(x, weight.to(x.dtype).t(), store_d=False, target=target, tuned=False)
    return torch.where(target != PretrainingBatches.PAD, res["lse"] - res["target_logit"], 0.0)


@lm_losses.register_fake
def _(x, weight, target):
    return x.new_empty(target.shape, dtype=torch.float32)
