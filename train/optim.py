"""
AdamW (fused) on token tables, lm_head and 1-D parameters; Muon (torch.optim.Muon's update, its Newton-Schulz batched
over same-shape matrices) on all other matrices, its update scaled to AdamW's RMS (adjust_lr_fn "match_rms_adamw":
0.2 sqrt(max(fan_out, fan_in)), Moonlight, arXiv 2502.16982 §2.2) so all groups share one lr and weight decay;
momentum 0.95 Nesterov, Newton-Schulz coefficients and 5 steps PyTorch's defaults; NorMuon's per-neuron normalization
of the orthogonalized update (arXiv 2510.05491). AdamW's eps is PyTorch's default. Each update clips the accumulated
gradient's global norm first.
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
MAX_GRAD_NORM = 1.0
# AdamW's EMA timescale law tau* = TAU_C TPP^TAU_E of the run's steps (Power Lines, arXiv 2505.13738 App. E.3.1)
TAU_C, TAU_E = 1.084, -0.527
# NorMuon's second-moment decay beta2 (arXiv 2510.05491 §4.1.1) and the eps of its normalizer (zichongli5/NorMuon
# normuon.py normuon_update)
NORMUON_BETA2, NORMUON_EPS = 0.95, 1e-10


def tau_weight_decay(*, lr: float, steps: int, tokens_per_param: float) -> float:
    """Weight decay whose EMA timescale 1 / (lr wd) is tau* = TAU_C TPP^TAU_E of the run's steps (Power Lines,
    arXiv 2505.13738 Finding 3; arXiv 2405.13698)."""
    return 1 / (lr * TAU_C * tokens_per_param ** TAU_E * steps)


class Muon(torch.optim.Muon):
    """NorMuon (arXiv 2510.05491 Alg. 1) on torch.optim.Muon's update (torch/optim/_muon.py _single_tensor_muon:
    momentum lerp, Nesterov, bf16 Newton-Schulz, decoupled weight decay, adjusted lr), each step's Newton-Schulz batched
    over a group's same-shape matrices: one bmm/baddbmm chain per shape instead of one mm/addmm chain per matrix
    (torch.optim.Muon has no foreach path; the batched GEMMs may round differently from the per-matrix ones). NorMuon
    (Alg. 1 lines 7-9): an EMA v (decay NORMUON_BETA2, state second_momentum_buffer, one fp32 per row: output neuron,
    nn.Linear's (out, in) layout) of the orthogonalized update O's mean square per row, O divided row-wise by sqrt(v) +
    NORMUON_EPS, then rescaled to O's Frobenius norm (zichongli5/NorMuon normuon.py normuon_update), so adjust_lr_fn
    "match_rms_adamw" scales it as Muon's update (Alg. 1 line 10 rescales to RMS 0.2 lr directly: the same target).
    State: torch.optim.Muon's momentum_buffer and second_momentum_buffer. After each step, update_square holds the
    squared Frobenius norm of the step's updates times their adjusted lr, summed over the matrices (the update without
    its decoupled weight decay), on the device."""

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        squares = []
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
                        self.state[p]["second_momentum_buffer"] = torch.zeros_like(p[:, :1])
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
                x = x.float()
                second = [self.state[p]["second_momentum_buffer"] for p in params]
                torch._foreach_lerp_(second, list(x.square().mean(-1, keepdim=True).unbind(0)), 1 - NORMUON_BETA2)
                normalized = x / (torch.stack(second).sqrt() + NORMUON_EPS)
                x = normalized * (torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True)
                                  / (torch.linalg.vector_norm(normalized, dim=(-2, -1), keepdim=True) + NORMUON_EPS))
                torch._foreach_mul_(params, 1 - lr * group["weight_decay"])
                adjusted = _adjust_lr(lr, group["adjust_lr_fn"], shape)
                torch._foreach_add_(params, x.unbind(0), alpha=-adjusted)
                squares.append(torch.linalg.vector_norm(x, dtype=torch.float32).square() * adjusted**2)
        self.update_square = torch.stack(squares).sum()
        return loss


# Scalar-lr arm (karpathy/nanochat 92d63d4 nanochat/gpt.py setup_optimizer lines 442-443, scalar_lr 0.5 its
# base_train.py --scalar-lr default; validated in its dev/LOG.md 2026-01-11 and 2026-01-19..22): (lr at B_REF, betas,
# eps, weight decay) of x0_lambdas and resid_lambdas; lr x sqrt(B / B_REF) (base_train.py lines 287-293)
SCALAR_B_REF = 2**19
X0_GROUP = (0.5, (0.96, 0.95), 1e-10, 0.0)
RESID_GROUP = (0.5 * 0.01, (0.8, 0.95), 1e-10, 0.05)


def optimizers(model: torch.nn.Module, *, lr: float, weight_decay: float, muon_tau_scale: float,
               adamw_learning_rate_scale: float, scalar_learning_rates: bool,
               batch_size: int) -> tuple[torch.optim.Optimizer, ...]:
    """(AdamW, Muon) at peak lr; weight decay on matrices, tables and lm_head, not 1-D parameters (karpathy/nanoGPT
    model.py configure_optimizers). Two scales, each the identity at 1: muon_tau_scale divides Muon's weight decay, so
    its EMA timescale 1 / (lr wd) grows by it (the tau law was fit with AdamW; Power Lines, arXiv 2505.13738 App. B);
    adamw_learning_rate_scale multiplies AdamW's lr and divides its weight decay, holding its timescale (Moonlight's
    one lr and weight decay for every group, arXiv 2502.16982 §2.2, matches Muon's update RMS, not the AdamW groups'
    lr). scalar_learning_rates: x0_lambdas and resid_lambdas leave the 1-D group for two AdamW groups of their own at
    92d63d4's absolute lr, betas, eps and weight decay (X0_GROUP, RESID_GROUP), lr x sqrt(batch_size / SCALAR_B_REF)
    (batch_size: tokens per step); the schedule multiplies each group's own initial lr (train/loop.py LambdaLR). Off:
    the groups are the ones without the option."""
    tables = [model.embed_tokens.weight, model.lm_head.weight, *model.value_embeds.parameters()]
    table_ids = {id(p) for p in tables}
    scalars = [model.x0_lambdas, model.resid_lambdas] if scalar_learning_rates else []
    scalar_ids = {id(p) for p in scalars}
    vectors = [p for p in model.parameters() if p.ndim == 1 and id(p) not in scalar_ids]
    matrices = [p for p in model.parameters() if p.ndim == 2 and id(p) not in table_ids]
    groups = [{"params": tables, "weight_decay": weight_decay / adamw_learning_rate_scale},
              {"params": vectors, "weight_decay": 0.0}]
    if scalar_learning_rates:
        scale = (batch_size / SCALAR_B_REF) ** 0.5
        groups += [{"params": [p], "lr": base * scale, "betas": betas, "eps": eps, "weight_decay": wd}
                   for p, (base, betas, eps, wd) in zip(scalars, (X0_GROUP, RESID_GROUP), strict=True)]
    adamw = torch.optim.AdamW(groups, lr=lr * adamw_learning_rate_scale, betas=BETAS, fused=True)
    return adamw, Muon(matrices, lr=lr, weight_decay=weight_decay / muon_tau_scale, adjust_lr_fn="match_rms_adamw")


def update(model: torch.nn.Module, opts: tuple[torch.optim.Optimizer, ...],
           accumulated: list[torch.nn.Parameter]) -> torch.Tensor:
    """One update from the accumulated gradients: their global norm clipped to MAX_GRAD_NORM before every optimizer,
    each optimizer stepped, the gradients zeroed: those of `accumulated` (parameters whose .grad the ops accumulate into
    in place, model/modeling.py accumulated_weights) in place, the rest freed. Returns the norm before clipping, on the
    device (no host synchronization)."""
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    for opt in opts:
        opt.step()
    in_place = set(accumulated)
    for p in model.parameters():
        if p not in in_place:
            p.grad = None
    torch._foreach_zero_([p.grad for p in accumulated])
    return norm
