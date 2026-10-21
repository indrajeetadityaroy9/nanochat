"""
LM-head next-token loss from hidden states x (M, C) and head weight W (V, C), z = x W^T:

    loss = sum over rows with target t != PretrainingBatches.PAD of (logsumexp(z) - z[t]), divided by the valid count n.

nanochat::lm_loss (training): a GEMM writes bf16 z; QuACK's cross_entropy_fwd_out (quack/cross_entropy.py, dx = z as
with inplace_backward) writes each row's loss and overwrites z with softmax(z) - onehot(t); dx = (1/n) z W on QuACK
(bf16), dW = (1/n) z^T x on cuBLAS in fp32 (as ops/linear.py's weight gradients). QuACK's fused linear-cross-entropy
is not faster at d8 (29.2 ms against 31.4-45.0 ms over chunk sizes 4,096-32,768).

nanochat::lm_losses (evaluation): QuACK's lse_target_epi GEMM epilogue (quack/epilogue/library.py) reduces each row's
logsumexp and gathers the target logit from the fp32 accumulators without storing z (d8: 14.9 against 16.2 ms for bf16
logits and cross entropy, relative error 5.0e-8 against 8e-4).
"""

import torch
from quack.cross_entropy import cross_entropy_fwd_out
from quack.epilogue.library import lse_target_epi
from quack.gemm_interface import gemm
from torch import Tensor
from torch.autograd.function import FunctionCtx

from nanochat.data.loader import PretrainingBatches


# Opaque to torch.compile: functionalizing the in-place logit gradient corrupted x's gradient.
@torch.library.custom_op("nanochat::lm_loss", mutates_args=(), device_types="cuda")
def lm_loss(x: Tensor, weight: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """x (M, C) bf16, fp32 weight (V, C), target (M,) int64 -> (mean loss fp32, dloss/dx in x's dtype, dloss/dW fp32)."""
    w = weight.to(x.dtype)
    logits = x @ w.t()
    scale = 1 / (target != PretrainingBatches.PAD).sum().float()
    loss = torch.empty(target.shape, device=x.device, dtype=torch.float32)
    cross_entropy_fwd_out(logits, target, target_logit=None, loss=loss, lse=None, dx=logits, weight=None,
                          ignore_index=PretrainingBatches.PAD)
    return (loss.sum() * scale, gemm(logits, w, alpha=scale, out_dtype=x.dtype),
            torch.mm(logits.t(), x, out_dtype=weight.dtype) * scale)


@lm_loss.register_fake
def lm_loss_fake(x: Tensor, weight: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return x.new_empty((), dtype=torch.float32), x.new_empty(x.shape), weight.new_empty(weight.shape)


def lm_loss_save(ctx: FunctionCtx, inputs: tuple[Tensor, Tensor, Tensor],
                 output: tuple[Tensor, Tensor, Tensor]) -> None:
    ctx.save_for_backward(output[1], output[2])


def lm_loss_grad(ctx: FunctionCtx, dloss: Tensor, ddx: Tensor, ddw: Tensor) -> tuple[Tensor, Tensor, None]:
    """Saved gradients times dloss; dx's product in fp32, rounded once to bf16."""
    dx, dw = ctx.saved_tensors
    return (dx.float() * dloss).to(dx.dtype), dw * dloss, None


lm_loss.register_autograd(lm_loss_grad, setup_context=lm_loss_save)


def step_shares(targets: list[Tensor]) -> Tensor:
    """Each micro-batch's share of a step's valid (non-PAD) targets, fp32: lm_loss's per-micro-batch means weighted by
    it sum to the mean over every valid target of the step."""
    valid = torch.stack([(t != PretrainingBatches.PAD).sum() for t in targets]).double()
    return (valid / valid.sum()).float()


@torch.library.custom_op("nanochat::lm_losses", mutates_args=(), device_types="cuda")
def lm_losses(x: Tensor, weight: Tensor, target: Tensor) -> Tensor:
    """Per-position loss (M,) fp32, 0 at PAD targets; no gradient. QuACK's default tile config (tuned=False)."""
    res = lse_target_epi(x, weight.to(x.dtype).t(), store_d=False, target=target, tuned=False)
    return torch.where(target != PretrainingBatches.PAD, res["lse"] - res["target_logit"], 0.0)


@lm_losses.register_fake
def lm_losses_fake(x: Tensor, weight: Tensor, target: Tensor) -> Tensor:
    return x.new_empty(target.shape, dtype=torch.float32)
