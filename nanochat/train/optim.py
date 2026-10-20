"""
AdamW (fused) on token tables, lm_head and 1-D parameters; an orthogonalized-update optimizer on all other matrices, its
update scaled to AdamW's RMS (0.2 sqrt(max(fan_out, fan_in)), Moonlight, arXiv 2502.16982 §2.2) so all groups share one
lr and weight decay:
- muon: torch.optim.Muon (adjust_lr_fn "match_rms_adamw"); momentum 0.95 Nesterov, Newton-Schulz coefficients and 5
  steps PyTorch's defaults.
- normuon: dion.NorMuon (microsoft/dion, as OLMo-core uses it): Polar Express orthogonalization (arXiv 2505.16932),
  neuron-wise second-moment normalization (NorMuon, arXiv 2510.05491), cautious weight decay (decay only where update
  and parameter signs agree); momentum 0.95 Nesterov and adjust_lr "rms_norm" as muon's. nanochat's optimizer (92d63d4
  nanochat/optim.py; its LOG 2026-01-10: cautious weight decay a solid gain).
AdamW's eps is PyTorch's default.
"""
from typing import Literal

import torch
from dion import NorMuon

# DeepSeek LLM §2.3 (arXiv 2401.02954)
BETAS = (0.9, 0.95)


def tau_weight_decay(*, lr: float, steps: int, tokens_per_param: float) -> float:
    """Weight decay whose EMA timescale 1 / (lr wd) is tau* = 1.084 TPP^-0.527 of the run's steps (Power Lines,
    arXiv 2505.13738 Finding 3; Wang & Aitchison, arXiv 2405.13698)."""
    return 1 / (lr * 1.084 * tokens_per_param ** -0.527 * steps)


def optimizers(model: torch.nn.Module, *, lr: float, weight_decay: float,
               matrix_optimizer: Literal["muon", "normuon"]) -> tuple[torch.optim.Optimizer, ...]:
    """(AdamW, the matrix optimizer) at peak lr; weight decay on matrices, tables and lm_head, not 1-D parameters
    (nanoGPT configure_optimizers)."""
    tables = [model.wte.weight, model.lm_head.weight, *model.value_embeds.parameters()]
    table_ids = {id(p) for p in tables}
    vectors = [p for p in model.parameters() if p.ndim == 1]
    matrices = [p for p in model.parameters() if p.ndim == 2 and id(p) not in table_ids]
    adamw = torch.optim.AdamW([{"params": tables, "weight_decay": weight_decay},
                               {"params": vectors, "weight_decay": 0.0}], lr=lr, betas=BETAS, fused=True)
    if matrix_optimizer == "muon":
        return adamw, torch.optim.Muon(matrices, lr=lr, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw")
    return adamw, NorMuon(matrices, lr=lr, weight_decay=weight_decay, nesterov=True, adjust_lr="rms_norm",
                          cautious_wd=True)
