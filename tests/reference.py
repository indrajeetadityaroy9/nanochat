"""fp64 reference of the model (nanochat/model/gpt.py's forward written out in plain PyTorch, document masking by
segments), from a GPT's parameters by name, and the helpers the verify scripts share. Positions are row positions.
Window w: query i sees keys j of its segment with 0 <= i - j <= w. The windows, the value-embedding layers, the
GPTConfig switches' off states and the candidate additions are restated here from their specification (gpt.py
GPTConfig, layer_windows, has_ve, depth_attention, Attention.forward), not imported, so a bug there shows: without
value_embeds no value embedding or gate; without x0 no x0 term; without resid the residual scalar 1; without qk_gain
the gains 1; without windows every layer the full context; attn_res Block Attention Residuals over 8 blocks of
sublayers (Chen et al., arXiv 2603.15031 Eq. 6); key_offset on the full-context layers RoPE over the higher-frequency
half of the rotation pairs and the other half of each key's dims from the previous token of its segment."""
import math
import statistics
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from nanochat.config import Stage
from nanochat.data.loader import PretrainingBatches, mixture_weights, packed_rows
from nanochat.data.tokenizer import Tokenizer

QUERY_CHUNK = 1024  # queries per attention chunk: a memory bound only (fixture), the result does not depend on it


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 error of a against the reference b, in fp64."""
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),), eps=torch.finfo(torch.float32).eps)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], -1)


@torch.no_grad()
def perturb(model: nn.Module) -> None:
    """Make every path the model has non-zero: init_weights zeroes c_o, c_proj, lm_head and the VE gates and fixes the
    lambdas and the trained gains (the components a switch removes are left absent, and the off-state gains at 1). The
    ranges are fixtures: s is the uniform bound of std C^-0.5, the rest of order one around the init."""
    cfg = model.config
    s = 3 ** 0.5 * cfg.n_embd ** -0.5
    for b in model.blocks:
        b.attn.c_o.weight.uniform_(-s, s)
        b.mlp.c_proj.weight.uniform_(-s / 2, s / 2)
        if cfg.qk_gain:
            b.attn.q_gain.uniform_(0.5, 1.5)
            b.attn.k_gain.uniform_(0.5, 1.5)
        if b.attn.ve_gate is not None:
            b.attn.ve_gate.weight.uniform_(-s, s)
        if cfg.attn_res:
            b.attn.depth_query.uniform_(-s, s)
            b.mlp.depth_query.uniform_(-s, s)
    model.lm_head.weight.normal_(0, 0.05)
    if cfg.resid:
        model.resid_lambdas.uniform_(0.8, 1.2)
    if cfg.x0:
        model.x0_lambdas.uniform_(0.1, 0.3)
    if cfg.attn_res:
        model.depth_query.uniform_(-s, s)


def first_micro_batch(stage: Stage, tokenizer: Tokenizer,
                      device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(inputs, targets, cu_seqlens) of the first micro-batch of a stage's packed train rows at the stage's mixture,
    each corpus's rows permuted as base_train does by its data seed, here 42 (fixture: any seed gives real rows; all
    packed rows, where base_train permutes train/stages.py readable_rows, a subset for the repository corpora), reading
    "fim" (packed pieces, FIM examples with their sentinels, padding targets)."""
    T = stage.seq_len
    mixture = mixture_weights(stage.mixture, tokenizer, T)
    orders = {c: np.random.RandomState(42).permutation(packed_rows(c, "train", tokenizer, T)) for c in mixture}
    return next(iter(PretrainingBatches(mixture, "train", tokenizer, orders, seq_len=T, batch_rows=stage.batch_rows,
                                        device=device, reading="fim", start=0)))


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seg: torch.Tensor, window: int,
              dtype: torch.dtype) -> torch.Tensor:
    """Causal attention of q, k, v (B, T, H, D) within segments (seg (B, T): each position's segment), each query over
    its `window` preceding keys and itself, softmax in at least fp32. Computed per chunk of QUERY_CHUNK queries under
    torch.utils.checkpoint, which recomputes a chunk's (B, H, QUERY_CHUNK, T) scores in backward instead of storing
    them, so a 32,768-token context fits; every query's softmax still spans all its keys."""
    _, T, _, D = q.shape
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


def reference_losses(model: nn.Module, idx: torch.Tensor, targets: torch.Tensor, cu_seqlens: torch.Tensor,
                     params: dict[str, torch.Tensor] | None = None,
                     dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-position losses (B*T,) (ignored targets 0) and the parameter copies they were computed from, in dtype (fp64:
    the reference; bf16: the plain eager formulation, softmax and logits in fp32). The model's config switches select
    the components; an absent one is its identity (module docstring)."""
    cfg = model.config
    P = params or {n: p.detach().to(dtype).requires_grad_() for n, p in model.named_parameters()}
    B, T = idx.shape
    C, H, D, L = cfg.n_embd, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.n_layer
    # windows: SSSL tiled over the layers, S = 512 keys capped at the context, L and the last layer the context
    ctx = cfg.sequence_len
    windows = [min(512, ctx) if cfg.windows and i % 4 < 3 else ctx for i in range(L - 1)] + [ctx]
    dev = idx.device
    seg = torch.searchsorted(cu_seqlens.long(), torch.arange(B * T, device=dev), right=True).view(B, T)
    t = torch.arange(T, dtype=torch.float64, device=dev)
    ang = torch.outer(t, cfg.rope_base ** (-torch.arange(D // 2, dtype=torch.float64, device=dev) / (D // 2)))
    cos, sin = ang.cos()[None, :, None, :].to(dtype), ang.sin()[None, :, None, :].to(dtype)
    wide = torch.promote_types(dtype, torch.float32)
    # key_offset: the full-context layers rotate the higher-frequency half of the pairs (angle 0 for pairs >= D/4), and
    # the unrotated dims of a key are the previous token's in its segment
    low = torch.arange(D // 2, device=dev) >= D // 4
    cos_p, sin_p = cos.masked_fill(low, 1), sin.masked_fill(low, 0)
    previous = torch.cat([torch.zeros_like(seg[:, :1], dtype=torch.bool), seg[:, 1:] == seg[:, :-1]], 1)
    unrotated = torch.cat([low, low])

    def attn(i, xn):
        a = f"blocks.{i}.attn."
        q = (xn @ P[a + "c_q.weight"].t()).view(B, T, H, D)
        k = (xn @ P[a + "c_k.weight"].t()).view(B, T, H, D)
        v = (xn @ P[a + "c_v.weight"].t()).view(B, T, H, D)
        if cfg.value_embeds and i % 2 == (L - 1) % 2:  # value embeddings on alternating layers, the last one included
            ve = P[f"value_embeds.{i}.weight"][idx].view(B, T, H, D)
            g = 2 * torch.sigmoid(xn @ P[a + "ve_gate.weight"].t())
            v = v + g[..., None] * ve
        q, k = norm(q), norm(k)
        if cfg.qk_gain:
            q, k = q * P[a + "q_gain"], k * P[a + "k_gain"]
        offset = cfg.key_offset and windows[i] == ctx
        c, s = (cos_p, sin_p) if offset else (cos, sin)
        q, k = rope(q, c, s), rope(k, c, s)
        if offset:
            shifted = torch.cat([k[:, :1], k[:, :-1]], 1)
            k = torch.where(previous[:, :, None, None] & unrotated, shifted, k)
        return attention(q, k, v, seg, windows[i], dtype).reshape(B, T, C) @ P[a + "c_o.weight"].t()

    def mlp(i, xn):
        m = f"blocks.{i}.mlp."
        return F.relu(xn @ P[m + "c_fc.weight"].t()).square() @ P[m + "c_proj.weight"].t()

    def depth(sources, query):
        """Softmax over the sources of query . RMSNorm(source), the sources weighted by it."""
        weights = torch.stack([(norm(src.to(wide)) * query.to(wide)).sum(-1) for src in sources]).softmax(0)
        return sum(w[..., None].to(dtype) * src for w, src in zip(weights, sources, strict=True))

    x = norm(P["wte.weight"][idx])
    x0 = x
    blocks, partial = [x0], None  # attn_res: b_0 the embedding, the completed blocks; the current block's partial sum
    for i in range(L):
        if cfg.resid:
            x = P["resid_lambdas"][i] * x
        if cfg.x0:
            x = x + P["x0_lambdas"][i] * x0
        for j, (name, f) in enumerate((("attn", attn), ("mlp", mlp))):
            if cfg.attn_res:  # 8 blocks of 2 L / 8 sublayers each
                out = f(i, norm(depth(blocks + ([partial] if partial is not None else []),
                                      P[f"blocks.{i}.{name}.depth_query"])))
                partial = out if partial is None else partial + out
                if (2 * i + j + 1) % (2 * L // 8) == 0:
                    blocks, partial = [*blocks, partial], None
            else:
                x = x + f(i, norm(x))
    if cfg.attn_res:
        x = depth(blocks, P["depth_query"])
    logits = (norm(x) @ P["lm_head.weight"].t()).to(wide)
    losses = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1),
                             ignore_index=PretrainingBatches.PAD, reduction="none")
    return losses, P


def compare(model: nn.Module, compiled: Callable[..., torch.Tensor], idx: torch.Tensor, targets: torch.Tensor,
            cu_seqlens: torch.Tensor, indent: str = "") -> list[str]:
    """One forward + backward of `compiled` (the model compiled as trained) against the fp64 reference, next to the
    bf16 eager formulation (the noise baseline): the mean loss, the per-position losses (model.losses, eager) and each
    parameter's gradient, as rel L2 errors vs fp64. Negative controls on the reference, for the components the model
    has: x0_lambdas x1.01 and the VE gate weights zeroed (gate 1), the size of a real model change. Prints them;
    returns the failures: the per-position losses or a matrix's gradient (the many-element errors; the mean loss and
    the vectors' are single sums that sit on either side of bf16 eager's) whose error reaches bf16 eager's combined
    with the smallest change a control makes (in quadrature: an implementation error that size, independent of the
    rounding, added to the bf16 floor). The errors at the floor itself fall on either side of bf16 eager's (the gradient
    reductions are not in a fixed order); a real error the size of a model change does not."""
    model.zero_grad(set_to_none=True)
    loss = compiled(idx, targets, cu_seqlens=cu_seqlens)
    loss.backward()
    with torch.no_grad():
        losses = model.losses(idx, targets, cu_seqlens)
    valid = targets.reshape(-1) != PretrainingBatches.PAD
    l64, P = reference_losses(model, idx.long(), targets, cu_seqlens)
    loss64 = l64[valid].mean()
    loss64.backward()
    l16, P16 = reference_losses(model, idx.long(), targets, cu_seqlens, dtype=torch.bfloat16)
    loss16 = l16[valid].mean()
    loss16.backward()
    ref = loss64.item()
    loss_err = abs(loss.item() - ref) / ref, abs(loss16.item() - ref) / ref
    errors = {"per-position losses": (rel(losses, l64.detach()), rel(l16.detach(), l64.detach()))}
    grads = {n: (rel(p.grad, P[n].grad), rel(P16[n].grad, P[n].grad)) for n, p in model.named_parameters()}
    worst = max(grads, key=lambda n: grads[n][0])
    print(f"{indent}loss ours {loss.item():.6f} ref {ref:.6f}: rel ours {loss_err[0]:.2e} bf16 eager "
          f"{loss_err[1]:.2e}; per-position losses rel ours {errors['per-position losses'][0]:.2e} bf16 eager "
          f"{errors['per-position losses'][1]:.2e}")
    print(f"{indent}worst grad rel {worst} ours {grads[worst][0]:.2e} bf16 eager {grads[worst][1]:.2e}; ours above "
          f"bf16 eager: {[f'{n} {e:.2e} vs {e16:.2e}' for n, (e, e16) in grads.items() if e > e16]}")
    changes = []  # every control's matrix-gradient changes
    for name, part, factor in (("x0_lambdas x1.01", "x0_lambdas", 1.01), ("ve_gate zeroed", "ve_gate", 0.0)):
        edited = [n for n in P if part in n]
        if not edited:
            continue
        Q = {n: t.detach().clone() for n, t in P.items()}
        with torch.no_grad():
            for n in edited:
                Q[n].mul_(factor)
        Q = {n: t.requires_grad_() for n, t in Q.items()}
        lc = reference_losses(model, idx.long(), targets, cu_seqlens, params=Q)[0][valid].mean()
        lc.backward()
        diffs = [rel(Q[n].grad, P[n].grad) for n in P if Q[n].ndim == 2]
        changes += diffs
        print(f"{indent}control {name}: loss rel diff {abs(lc.item() - ref) / ref:.2e}, matrix-grad rel diff median "
              f"{statistics.median_high(diffs):.2e} min {min(diffs):.2e}")
    errors |= {n: e for n, e in grads.items() if P[n].ndim == 2}
    detectable = min(changes, default=0.0)  # without a control: bf16 eager's error alone
    return [n for n, (e, e16) in errors.items() if e >= math.hypot(e16, detectable)]
