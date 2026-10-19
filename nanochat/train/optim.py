"""
The optimizers, PyTorch's own on fp32 parameters, gradients and state (optimizers): canonically AdamW
(torch.optim.AdamW, fused) on the token tables, lm_head and the 1-D parameters and Muon (torch.optim.Muon) on every
other matrix, its update scaled to AdamW's RMS (adjust_lr_fn "match_rms_adamw": lr 0.2 sqrt(max(A, B)); Liu et al.,
arXiv 2502.16982 §2.2: "Muon can directly reuse the learning rate and weight decay tuned for AdamW"; its decoupled decay
uses the unadjusted lr), so that every group shares one lr and weight decay; or (the optimizer ablation) AdamW on every
parameter. Muon's Nesterov momentum 0.95, Newton-Schulz coefficients and 5 steps are torch.optim.Muon's defaults
(KellerJordan/Muon; Moonlight §2.2: "we chose 0.95", N = 5); AdamW's eps is torch's default, Adam's 1e-8 (Kingma & Ba,
arXiv 1412.6980 Algorithm 1).
"""

from typing import Literal

import torch

# DeepSeek LLM's AdamW (arXiv 2401.02954 §2.3): "beta1 = 0.9, beta2 = 0.95, and weight_decay = 0.1"; and its clipping,
# "The gradient clipping during the training phase is set to 1.0" (DeepSeek-Coder §3.4 follows DeepSeek LLM): the step's
# gradient scaled to L2 norm GRAD_CLIP when above it (scripts/base_train.optimizer_step, torch.nn.utils.clip_grad_norm_)
BETAS, WEIGHT_DECAY, GRAD_CLIP = (0.9, 0.95), 0.1, 1.0


def tau_weight_decay(lr: float, steps: int, tokens_per_param: float) -> float:
    """The weight decay whose EMA timescale 1 / (lr wd) is the fraction tau* of the run's steps that Power Lines fits
    (Bergsma et al., arXiv 2505.13738 Finding 3: tau* = 1.084 TPP^-0.527, 111M-3.3B params, 20-1280 tokens per param;
    the EMA view of decoupled weight decay: Wang & Aitchison, arXiv 2405.13698)."""
    return 1 / (lr * 1.084 * tokens_per_param ** -0.527 * steps)


def optimizers(model: torch.nn.Module, lr: float, weight_decay: float,
               kind: Literal["muon", "adamw"]) -> tuple[torch.optim.Optimizer, ...]:
    """The optimizers of a GPT at peak lr (the trainer's LambdaLR scales every group's lr by schedule.lr_multiplier),
    weight decay on the matrices, token tables and lm_head and none on the 1-D parameters (nanoGPT model.py
    configure_optimizers: "all weight tensors in matmuls + embeddings decay, all biases and layernorms don't").
    muon: AdamW on the tables and vectors, Muon on every other matrix (Moonlight, arXiv 2502.16982 §2.2);
    adamw: AdamW on every parameter (the baseline Muon is compared against; DeepSeek LLM's optimizer)."""
    tables = [model.wte.weight, model.lm_head.weight, *model.value_embeds.parameters()]
    table_ids = {id(p) for p in tables}
    vectors = [p for p in model.parameters() if p.ndim == 1]
    matrices = [p for p in model.parameters() if p.ndim == 2 and id(p) not in table_ids]
    decayed = tables if kind == "muon" else tables + matrices
    adamw = torch.optim.AdamW([dict(params=decayed, weight_decay=weight_decay), dict(params=vectors, weight_decay=0.0)],
                              lr=lr, betas=BETAS, fused=True)
    if kind == "adamw":
        return (adamw,)
    return adamw, torch.optim.Muon(matrices, lr=lr, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw")
