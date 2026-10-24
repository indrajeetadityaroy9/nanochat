"""
AdamW (fused) on token tables, lm_head and 1-D parameters; Muon (torch.optim.Muon's update, its Newton-Schulz batched
over same-shape matrices) on all other matrices, its update scaled to AdamW's RMS (adjust_lr_fn "match_rms_adamw":
0.2 sqrt(max(fan_out, fan_in)), Moonlight, arXiv 2502.16982 §2.2) so all groups share one lr and weight decay;
momentum 0.95 Nesterov, Newton-Schulz coefficients and 5 steps PyTorch's defaults. AdamW's eps is PyTorch's default.
Each update clips the accumulated gradient's global norm first.
"""
from collections.abc import Callable

import torch
from torch.optim._muon import _adjust_lr

# DeepSeek LLM §2.3 (arXiv 2401.02954)
BETAS = (0.9, 0.95)
# Global gradient-norm clip at 1.0, DeepSeek LLM's (arXiv 2401.02954 §2.3): Adam's update is
# invariant to the gradient's scale (arXiv 1412.6980 §2.1) and Muon's Newton-Schulz divides its input by
# its norm (torch/optim/_muon.py _zeropower_via_newtonschulz, after KellerJordan/Muon), so under a slowly varying
# gradient scale the clip damps only spikes
MAX_NORM = 1.0


def tau_weight_decay(*, lr: float, steps: int, tokens_per_param: float) -> float:
    """Weight decay whose EMA timescale 1 / (lr wd) is tau* = 1.084 TPP^-0.527 of the run's steps (Power Lines,
    arXiv 2505.13738 Finding 3; arXiv 2405.13698)."""
    return 1 / (lr * 1.084 * tokens_per_param ** -0.527 * steps)


class Muon(torch.optim.Muon):
    """torch.optim.Muon (torch/optim/_muon.py _single_tensor_muon: momentum lerp, Nesterov, bf16 Newton-Schulz,
    decoupled weight decay, adjusted lr) with each step's Newton-Schulz batched over a group's same-shape matrices: one
    bmm/baddbmm chain per shape instead of one mm/addmm chain per matrix (torch.optim.Muon has no foreach path), whose
    launches made the update host-bound. The same state (momentum_buffer) and state_dict as torch.optim.Muon; the
    batched GEMMs may round differently from the per-matrix ones."""

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            a, b, c = group["ns_coefficients"]
            shapes: dict[torch.Size, list[torch.Tensor]] = {}
            for p in group["params"]:
                if p.grad is not None:
                    shapes.setdefault(p.shape, []).append(p)
            for shape, params in shapes.items():
                grads = [p.grad for p in params]
                for p in params:
                    if "momentum_buffer" not in self.state[p]:
                        self.state[p]["momentum_buffer"] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                bufs = [self.state[p]["momentum_buffer"] for p in params]
                torch._foreach_lerp_(bufs, grads, 1 - momentum)
                x = torch.stack(torch._foreach_lerp(grads, bufs, momentum) if group["nesterov"] else bufs).bfloat16()
                tall = shape[0] > shape[1]
                if tall:
                    x = x.mT
                x = x / torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True).clamp(min=group["eps"])
                for _ in range(group["ns_steps"]):
                    gram = x @ x.mT
                    x = torch.baddbmm(x, torch.baddbmm(gram, gram, gram, beta=b, alpha=c), x, beta=a)
                if tall:
                    x = x.mT
                torch._foreach_mul_(params, 1 - lr * group["weight_decay"])
                torch._foreach_add_(params, x.unbind(0), alpha=-_adjust_lr(lr, group["adjust_lr_fn"], shape))
        return loss


def optimizers(model: torch.nn.Module, *, lr: float, weight_decay: float) -> tuple[torch.optim.Optimizer, ...]:
    """(AdamW, Muon) at peak lr; weight decay on matrices, tables and lm_head, not 1-D parameters (karpathy/nanoGPT
    model.py configure_optimizers)."""
    tables = [model.wte.weight, model.lm_head.weight, *model.value_embeds.parameters()]
    table_ids = {id(p) for p in tables}
    vectors = [p for p in model.parameters() if p.ndim == 1]
    matrices = [p for p in model.parameters() if p.ndim == 2 and id(p) not in table_ids]
    adamw = torch.optim.AdamW([{"params": tables, "weight_decay": weight_decay},
                               {"params": vectors, "weight_decay": 0.0}], lr=lr, betas=BETAS, fused=True)
    return adamw, Muon(matrices, lr=lr, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw")


def update(model: torch.nn.Module, opts: tuple[torch.optim.Optimizer, ...]) -> torch.Tensor:
    """One update from the accumulated gradients: their global norm clipped to MAX_NORM before every optimizer, each
    optimizer stepped, the gradients zeroed. Returns the norm before clipping, on the device (no host
    synchronization)."""
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
    for opt in opts:
        opt.step()
    model.zero_grad()
    return norm
