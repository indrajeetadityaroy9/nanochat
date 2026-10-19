"""
The language-model head's next-token loss straight from the hidden states x (M, C) and the head's weight W (V, C),
z = x W^T:

    loss = (sum over the rows whose target t is not the loader's padding of (logsumexp(z) - z[t])) / n,

n the valid count; the padding target PretrainingBatches.PAD (data/loader.py) is the ignore index. One GEMM writes
the bf16 logits z; one pass of QuACK's CrossEntropy kernel (Tri Dao, CuTe DSL; quack/cross_entropy.py
cross_entropy_fwd_out, given dx = z as its cross_entropy_fwd does with inplace_backward) reads each row once, writes the
row's loss and, in place of z, the logit gradient relative to the scale 1 / n, softmax(z) - onehot(t) (0 in ignored
rows); two GEMMs take dx and dW from it with alpha = 1 / n, dW in fp32 straight from the accumulators. The logits are
bf16 and read once.
"""

import torch
from quack.cross_entropy import cross_entropy_fwd_out
from torch import Tensor

from nanochat.data.loader import PretrainingBatches
from nanochat.ops.quack_gemm import gemm


def _losses(logits, target, dlogits):
    """Each row's loss (fp32, 0 for an ignored target) of the logits (M, V) against the targets (M,), and, if dlogits is
    given (it may be the logits), the logit gradient into it: QuACK's cross_entropy_fwd_out without the per-row
    log-sum-exp output, class weights or precomputed target logits."""
    loss = torch.empty(target.shape, device=logits.device, dtype=torch.float32)
    cross_entropy_fwd_out(logits, target, None, loss, None, dlogits, None, PretrainingBatches.PAD)
    return loss


# Opaque to torch.compile, as ops/flash_attn.py's FA4 ops are: the logit gradient overwrites the logits in place, an
# aliasing a traced graph must not see (functionalized, it corrupted the gradient of x).
@torch.library.custom_op("nanochat::lm_loss", mutates_args=(), device_types="cuda")
def lm_loss(x: Tensor, weight: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """x (M, C) bf16, weight (V, C) the fp32 master weight, rounded to x's dtype for the GEMMs, target (M,) int64: (the
    mean loss over the valid targets (scalar fp32), its gradient wrt x, its gradient wrt weight in weight's dtype)."""
    w = weight.to(x.dtype)
    logits = x @ w.t()
    scale = 1 / (target != PretrainingBatches.PAD).sum().float()
    loss = _losses(logits, target, logits)
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
    rounded once to bf16."""
    dx, dw = ctx.saved_tensors
    return (dx.float() * dloss).to(dx.dtype), dw * dloss, None


lm_loss.register_autograd(_lm_loss_backward, setup_context=_lm_loss_setup)


@torch.library.custom_op("nanochat::lm_losses", mutates_args=(), device_types="cuda")
def lm_losses(x: Tensor, weight: Tensor, target: Tensor) -> Tensor:
    """Each position's loss (M,) fp32 against its target (M,), 0 at the padding target; no gradient. weight: the master
    weight (V, C), rounded to x's dtype for the GEMM."""
    return _losses(x @ weight.to(x.dtype).t(), target, None)


@lm_losses.register_fake
def _(x, weight, target):
    return x.new_empty(target.shape, dtype=torch.float32)
