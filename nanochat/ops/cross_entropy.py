"""
LM-head next-token loss from hidden states x (M, C) and head weight W (V, C), z = x W^T:

    loss = sum over rows with target t != PretrainingBatches.PAD of (logsumexp(z) - z[t]), divided by the valid count n.

nanochat::lm_loss (training): a GEMM writes bf16 z; QuACK's cross_entropy_fwd_out (quack/cross_entropy.py, dx = z as
with inplace_backward) writes each row's loss and overwrites z with softmax(z) - onehot(t); dx = (1/n) z W on QuACK
(bf16), dW = (1/n) z^T x on cuBLAS in fp32 (as ops/linear.py's weight gradients). QuACK's fused linear-cross-entropy
is not faster at any chunk size.

nanochat::lm_losses (evaluation): QuACK's lse_target_epi GEMM epilogue (quack/epilogue/library.py) reduces each row's
logsumexp and gathers the target logit from the fp32 accumulators without storing z: faster and more accurate than
bf16 logits and cross entropy.

With an output-logit cap c (model/gpt.py GPTConfig.softcap; 92d63d4 gpt.py:511-515, Gemma 2's final-logit
softcapping, arXiv 2408.00118 §2) z becomes c tanh(z / c) (tanh in fp32), in kernels Inductor fuses (cap_,
capped_losses): lm_loss caps z in place (bf16) beside the cap's derivative 1 - tanh(z / c)^2 (bf16), two logits-sized
buffers, runs QuACK's cross entropy on the capped logits and multiplies its logit gradient by the derivative before
the two gradient GEMMs; lm_losses takes logsumexp minus the target logit of the capped fp32 logits of each row chunk's
GEMM (CHUNK rows), as lse_target_epi has no tanh epilogue.
Without a cap both run as above.
"""

import torch
from quack.cross_entropy import cross_entropy_fwd_out
from quack.epilogue.library import lse_target_epi
from quack.gemm_interface import gemm
from torch import Tensor
from torch.autograd.function import FunctionCtx

from nanochat.corpus.loader import PretrainingBatches

# Rows of lm_losses' capped fp32 logits per GEMM: 8,192 x 2^15 x 4 bytes = 1 GiB
CHUNK = 8192


@torch.compile(dynamic=False, fullgraph=True)
def cap_(z: Tensor, softcap: float) -> Tensor:
    """z = softcap tanh(z / softcap) in place, one kernel; returns its derivative 1 - tanh(z / softcap)^2; each in z's
    dtype."""
    t = torch.tanh(z.float() / softcap)
    z.copy_((softcap * t).to(z.dtype))
    return (1 - t.square()).to(z.dtype)


@torch.compile(dynamic=False, fullgraph=True)
def capped_losses(z: Tensor, target: Tensor, softcap: float) -> Tensor:
    """logsumexp minus the target's entry of the rows of softcap tanh(z / softcap), z fp32."""
    z = softcap * torch.tanh(z / softcap)
    return torch.logsumexp(z, dim=1) - z.gather(1, target[:, None]).squeeze(1)


# Opaque to torch.compile: functionalizing the in-place logit gradient corrupted x's gradient.
@torch.library.custom_op("nanochat::lm_loss", mutates_args=(), device_types="cuda")
def lm_loss(x: Tensor, weight: Tensor, target: Tensor, softcap: float | None) -> tuple[Tensor, Tensor, Tensor]:
    """x (M, C) bf16, fp32 weight (V, C), target (M,) int64 -> (mean loss fp32, dloss/dx in x's dtype, dloss/dW
    fp32)."""
    w = weight.to(x.dtype)
    logits = x @ w.t()
    scale = 1 / (target != PretrainingBatches.PAD).sum().float()
    loss = torch.empty(target.shape, device=x.device, dtype=torch.float32)
    derivative = None if softcap is None else cap_(logits, softcap)
    cross_entropy_fwd_out(logits, target, target_logit=None, loss=loss, lse=None, dx=logits, weight=None,
                          ignore_index=PretrainingBatches.PAD)
    if derivative is not None:
        logits.mul_(derivative)
    return (loss.sum() * scale, gemm(logits, w, alpha=scale, out_dtype=x.dtype),
            torch.mm(logits.t(), x, out_dtype=weight.dtype) * scale)


@lm_loss.register_fake
def lm_loss_fake(x: Tensor, weight: Tensor, target: Tensor, softcap: float | None) -> tuple[Tensor, Tensor, Tensor]:
    return x.new_empty((), dtype=torch.float32), x.new_empty(x.shape), weight.new_empty(weight.shape)


def lm_loss_save(ctx: FunctionCtx, inputs: tuple[Tensor, Tensor, Tensor, float | None],
                 output: tuple[Tensor, Tensor, Tensor]) -> None:
    ctx.save_for_backward(output[1], output[2])


def lm_loss_grad(ctx: FunctionCtx, dloss: Tensor, ddx: Tensor, ddw: Tensor) -> tuple[Tensor, Tensor, None, None]:
    """Saved gradients times dloss; dx's product in fp32, rounded once to bf16."""
    dx, dw = ctx.saved_tensors
    return (dx.float() * dloss).to(dx.dtype), dw * dloss, None, None


lm_loss.register_autograd(lm_loss_grad, setup_context=lm_loss_save)


def step_shares(targets: list[Tensor]) -> Tensor:
    """Each micro-batch's share of a step's valid (non-PAD) targets, fp32: lm_loss's per-micro-batch means weighted by
    it sum to the mean over every valid target of the step."""
    valid = torch.stack([(t != PretrainingBatches.PAD).sum() for t in targets]).double()
    return (valid / valid.sum()).float()


@torch.library.custom_op("nanochat::lm_losses", mutates_args=(), device_types="cuda")
def lm_losses(x: Tensor, weight: Tensor, target: Tensor, softcap: float | None) -> Tensor:
    """Per-position loss (M,) fp32, 0 at PAD targets; no gradient. QuACK's default tile config (tuned=False)."""
    valid = target != PretrainingBatches.PAD
    if softcap is None:
        res = lse_target_epi(x, weight.to(x.dtype).t(), store_d=False, target=target, tuned=False)
        return torch.where(valid, res["lse"] - res["target_logit"], 0.0)
    w, rows = weight.to(x.dtype).t(), torch.where(valid, target, 0)
    losses = [capped_losses(torch.mm(xc, w, out_dtype=torch.float32), tc, softcap)
              for xc, tc in zip(x.split(CHUNK), rows.split(CHUNK), strict=True)]
    return torch.where(valid, torch.cat(losses), 0.0)


@lm_losses.register_fake
def lm_losses_fake(x: Tensor, weight: Tensor, target: Tensor, softcap: float | None) -> Tensor:
    return x.new_empty(target.shape, dtype=torch.float32)
