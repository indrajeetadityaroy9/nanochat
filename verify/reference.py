"""fp64 reference of the model (local://COMPOSITE_CONTRACT.md §1: model/gpt.py's forward written out in plain PyTorch,
document masking by segments), from a GPT's parameters by name. Positions are row positions. Window w: query i sees
keys j of its segment with 0 <= i - j <= w."""
import torch
import torch.nn.functional as F


def norm(x):
    return F.rms_norm(x, (x.size(-1),), eps=torch.finfo(torch.float32).eps)


def rope(x, cos, sin):
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], -1)


def reference_losses(model, idx, targets, cu_seqlens, windows, params=None, dtype=torch.float64):
    """Per-position losses (B*T,) (ignored targets 0) and the parameter copies they were computed from, in dtype (fp64:
    the reference; bf16: the plain eager formulation, softmax and logits in fp32)."""
    cfg = model.config
    P = params or {n: p.detach().to(dtype).requires_grad_() for n, p in model.named_parameters()}
    B, T = idx.shape
    C, H, D, L = cfg.n_embd, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.n_layer
    dev = idx.device
    seg = torch.searchsorted(cu_seqlens.long(), torch.arange(B * T, device=dev), right=True).view(B, T)
    t = torch.arange(T, dtype=torch.float64, device=dev)
    ang = torch.outer(t, cfg.rope_base ** (-torch.arange(D // 2, dtype=torch.float64, device=dev) / (D // 2)))
    cos, sin = ang.cos()[None, :, None, :].to(dtype), ang.sin()[None, :, None, :].to(dtype)
    wide = torch.promote_types(dtype, torch.float32)

    x = norm(P["wte.weight"][idx])
    x0 = x
    i_pos = torch.arange(T, device=dev)
    diff = i_pos[:, None] - i_pos[None, :]
    same = seg[:, :, None] == seg[:, None, :]
    for i in range(L):
        a = m = f"blocks.{i}."
        x = P["resid_lambdas"][i] * x + P["x0_lambdas"][i] * x0
        xn = norm(x)
        q = (xn @ P[a + "c_q.weight"].t()).view(B, T, H, D)
        k = (xn @ P[a + "c_k.weight"].t()).view(B, T, H, D)
        v = (xn @ P[a + "c_v.weight"].t()).view(B, T, H, D)
        if str(i) in model.value_embeds:
            ve = P[f"value_embeds.{i}.weight"][idx].view(B, T, H, D)
            g = 2 * torch.sigmoid(xn @ P[a + "ve_gate.weight"].t())
            v = v + g[..., None] * ve
        q = rope(norm(q) * P[a + "q_gain"], cos, sin)
        k = rope(norm(k) * P[a + "k_gain"], cos, sin)
        s = torch.einsum("bihd,bjhd->bhij", q, k) / D ** 0.5
        mask = same & (diff >= 0) & (diff <= windows[i])
        s = s.masked_fill(~mask[:, None], float("-inf"))
        y = torch.einsum("bhij,bjhd->bihd", s.to(wide).softmax(-1).to(dtype), v).reshape(B, T, C)
        x = x + y @ P[a + "c_o.weight"].t()
        h = F.relu(norm(x) @ P[m + "c_fc.weight"].t()).square()
        x = x + h @ P[m + "c_proj.weight"].t()
    logits = (norm(x) @ P["lm_head.weight"].t()).to(wide)
    losses = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1), ignore_index=-1, reduction="none")
    return losses, P
