"""
The optimizers, PyTorch's own on fp32 parameters, gradients and state: AdamW (torch.optim.AdamW, fused) on the token
tables, lm_head and the 1-D parameters; Muon (torch.optim.Muon) on every other matrix, its update scaled to AdamW's RMS
(adjust_lr_fn "match_rms_adamw": lr 0.2 sqrt(max(A, B)); Liu et al., arXiv 2502.16982 §2.2: "Muon can directly reuse the
learning rate and weight decay tuned for AdamW"), so that every group shares one lr and weight decay. Muon's Nesterov
momentum 0.95, Newton-Schulz coefficients and 5 steps are torch.optim.Muon's defaults (KellerJordan/Muon; Moonlight
§2.2: "we chose 0.95", N = 5); AdamW's eps is torch's default, Adam's 1e-8 (Kingma & Ba, arXiv 1412.6980 Algorithm 1).
"""

import torch

# DeepSeek LLM's AdamW (arXiv 2401.02954 §2.3): "beta1 = 0.9, beta2 = 0.95, and weight_decay = 0.1"
BETAS, WEIGHT_DECAY = (0.9, 0.95), 0.1


def optimizers(model, lr):
    """(AdamW, Muon) of a GPT at peak lr (the trainer scales every group's lr by schedule.lr_multiplier each step):
    AdamW with weight decay on the token tables and lm_head and without on the 1-D parameters (nanoGPT model.py
    configure_optimizers: "all weight tensors in matmuls + embeddings decay, all biases and layernorms don't"); Muon on
    every other matrix."""
    tables = [model.wte.weight, model.lm_head.weight, *model.value_embeds.parameters()]
    table_ids = {id(p) for p in tables}
    vectors = [p for p in model.parameters() if p.ndim == 1]
    matrices = [p for p in model.parameters() if p.ndim == 2 and id(p) not in table_ids]
    adamw = torch.optim.AdamW([dict(params=tables, weight_decay=WEIGHT_DECAY), dict(params=vectors, weight_decay=0.0)],
                              lr=lr, betas=BETAS, fused=True)
    muon = torch.optim.Muon(matrices, lr=lr, weight_decay=WEIGHT_DECAY, adjust_lr_fn="match_rms_adamw")
    return adamw, muon
