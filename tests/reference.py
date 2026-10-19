"""fp64 reference of the model (nanochat/model/gpt.py's forward written out in plain PyTorch, document masking by
segments), from a GPT's parameters by name, and the helpers the verify scripts share. Positions are row positions.
Window w: query i sees keys j of its segment with 0 <= i - j <= w. The windows and the value-embedding layers are
restated here from the rules of gpt.py layer_windows / has_ve, not imported, so a bug there shows."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

QUERY_CHUNK = 1024  # queries per attention chunk: a memory bound only (fixture), the result does not depend on it


def rel(a, b):
    """Relative L2 error of a against the reference b, in fp64."""
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def norm(x):
    return F.rms_norm(x, (x.size(-1),), eps=torch.finfo(torch.float32).eps)


def rope(x, cos, sin):
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], -1)


@torch.no_grad()
def perturb(model):
    """Make every path non-zero: init_weights zeroes c_o, c_proj, lm_head and the VE gates and fixes the lambdas and
    gains. The ranges are fixtures: s is the uniform bound of std C^-0.5, the rest of order one around the init."""
    s = 3 ** 0.5 * model.config.n_embd ** -0.5
    for b in model.blocks:
        b.attn.c_o.weight.uniform_(-s, s)
        b.mlp.c_proj.weight.uniform_(-s / 2, s / 2)
        b.attn.q_gain.uniform_(0.5, 1.5)
        b.attn.k_gain.uniform_(0.5, 1.5)
        if b.attn.ve_gate is not None:
            b.attn.ve_gate.weight.uniform_(-s, s)
    model.lm_head.weight.normal_(0, 0.05)
    model.resid_lambdas.uniform_(0.8, 1.2)
    model.x0_lambdas.uniform_(0.1, 0.3)


def attention(q, k, v, seg, window, dtype):
    """Causal attention of q, k, v (B, T, H, D) within segments (seg (B, T): each position's segment), each query over
    its `window` preceding keys and itself, softmax in at least fp32. Computed per chunk of QUERY_CHUNK queries under
    torch.utils.checkpoint, which recomputes a chunk's (B, H, QUERY_CHUNK, T) scores in backward instead of storing
    them, so a 32,768-token context fits; every query's softmax still spans all its keys."""
    B, T, H, D = q.shape
    pos = torch.arange(T, device=q.device)
    wide = torch.promote_types(dtype, torch.float32)

    def chunk(qc, k, v, start):
        i = pos[start:start + qc.size(1), None]
        mask = (seg[:, start:start + qc.size(1), None] == seg[:, None, :]) & (i >= pos) & (i - pos <= window)
        s = torch.einsum("bihd,bjhd->bhij", qc, k) / D ** 0.5
        s = s.masked_fill(~mask[:, None], float("-inf"))
        return torch.einsum("bhij,bjhd->bihd", s.to(wide).softmax(-1).to(dtype), v)

    return torch.cat([checkpoint(chunk, q[:, s:s + QUERY_CHUNK], k, v, s, use_reentrant=False)
                      for s in range(0, T, QUERY_CHUNK)], dim=1)


def reference_losses(model, idx, targets, cu_seqlens, params=None, dtype=torch.float64):
    """Per-position losses (B*T,) (ignored targets 0) and the parameter copies they were computed from, in dtype (fp64:
    the reference; bf16: the plain eager formulation, softmax and logits in fp32)."""
    cfg = model.config
    P = params or {n: p.detach().to(dtype).requires_grad_() for n, p in model.named_parameters()}
    B, T = idx.shape
    C, H, D, L = cfg.n_embd, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.n_layer
    # windows: SSSL tiled over the layers, S = 512 keys capped at the context, L and the last layer the context
    ctx = cfg.sequence_len
    windows = [min(512, ctx) if i % 4 < 3 else ctx for i in range(L - 1)] + [ctx]
    dev = idx.device
    seg = torch.searchsorted(cu_seqlens.long(), torch.arange(B * T, device=dev), right=True).view(B, T)
    t = torch.arange(T, dtype=torch.float64, device=dev)
    ang = torch.outer(t, cfg.rope_base ** (-torch.arange(D // 2, dtype=torch.float64, device=dev) / (D // 2)))
    cos, sin = ang.cos()[None, :, None, :].to(dtype), ang.sin()[None, :, None, :].to(dtype)
    wide = torch.promote_types(dtype, torch.float32)

    x = norm(P["wte.weight"][idx])
    x0 = x
    for i in range(L):
        a, m = f"blocks.{i}.attn.", f"blocks.{i}.mlp."
        x = P["resid_lambdas"][i] * x + P["x0_lambdas"][i] * x0
        xn = norm(x)
        q = (xn @ P[a + "c_q.weight"].t()).view(B, T, H, D)
        k = (xn @ P[a + "c_k.weight"].t()).view(B, T, H, D)
        v = (xn @ P[a + "c_v.weight"].t()).view(B, T, H, D)
        if i % 2 == (L - 1) % 2:  # value embeddings on alternating layers, the last one included
            ve = P[f"value_embeds.{i}.weight"][idx].view(B, T, H, D)
            g = 2 * torch.sigmoid(xn @ P[a + "ve_gate.weight"].t())
            v = v + g[..., None] * ve
        q = rope(norm(q) * P[a + "q_gain"], cos, sin)
        k = rope(norm(k) * P[a + "k_gain"], cos, sin)
        y = attention(q, k, v, seg, windows[i], dtype).reshape(B, T, C)
        x = x + y @ P[a + "c_o.weight"].t()
        h = F.relu(norm(x) @ P[m + "c_fc.weight"].t()).square()
        x = x + h @ P[m + "c_proj.weight"].t()
    logits = (norm(x) @ P["lm_head.weight"].t()).to(wide)
    losses = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1), ignore_index=-1, reduction="none")
    return losses, P
