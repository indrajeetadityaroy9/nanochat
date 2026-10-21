"""
AdamW (fused) on token tables, lm_head and 1-D parameters; torch.optim.Muon on all other matrices, its update scaled to
AdamW's RMS (adjust_lr_fn "match_rms_adamw": 0.2 sqrt(max(fan_out, fan_in)), Moonlight, arXiv 2502.16982 §2.2) so all
groups share one lr and weight decay; momentum 0.95 Nesterov, Newton-Schulz coefficients and 5 steps PyTorch's
defaults. AdamW's eps is PyTorch's default. Each update clips the accumulated gradient's global norm first.
"""
import torch

# DeepSeek LLM §2.3 (arXiv 2401.02954)
BETAS = (0.9, 0.95)
# Global gradient-norm clip, every reference code model's (DeepSeek LLM, arXiv 2401.02954 §2.3): Adam's update is
# invariant to the gradient's scale (Kingma & Ba, arXiv 1412.6980 §2.1) and Muon's Newton-Schulz divides its input by
# its norm (torch/optim/_muon.py _zeropower_via_newtonschulz, after KellerJordan/Muon), so under a slowly varying
# gradient scale the clip damps only spikes
MAX_NORM = 1.0


def tau_weight_decay(*, lr: float, steps: int, tokens_per_param: float) -> float:
    """Weight decay whose EMA timescale 1 / (lr wd) is tau* = 1.084 TPP^-0.527 of the run's steps (Power Lines,
    arXiv 2505.13738 Finding 3; Wang & Aitchison, arXiv 2405.13698)."""
    return 1 / (lr * 1.084 * tokens_per_param ** -0.527 * steps)


def optimizers(model: torch.nn.Module, *, lr: float, weight_decay: float) -> tuple[torch.optim.Optimizer, ...]:
    """(AdamW, Muon) at peak lr; weight decay on matrices, tables and lm_head, not 1-D parameters (nanoGPT
    configure_optimizers)."""
    tables = [model.wte.weight, model.lm_head.weight, *model.value_embeds.parameters()]
    table_ids = {id(p) for p in tables}
    vectors = [p for p in model.parameters() if p.ndim == 1]
    matrices = [p for p in model.parameters() if p.ndim == 2 and id(p) not in table_ids]
    adamw = torch.optim.AdamW([{"params": tables, "weight_decay": weight_decay},
                               {"params": vectors, "weight_decay": 0.0}], lr=lr, betas=BETAS, fused=True)
    return adamw, torch.optim.Muon(matrices, lr=lr, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw")


def update(model: torch.nn.Module, opts: tuple[torch.optim.Optimizer, ...]) -> torch.Tensor:
    """One update from the accumulated gradients: their global norm clipped to MAX_NORM before every optimizer, each
    optimizer stepped, the gradients zeroed. Returns the norm before clipping, on the device (no host
    synchronization)."""
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
    for opt in opts:
        opt.step()
    model.zero_grad()
    return norm
