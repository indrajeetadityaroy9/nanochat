"""The ops against fp64, next to the plain bf16 PyTorch formulation as the noise baseline: qk_norm_rope (per-head RMS
norm, fp32 gains, full-head rotary; contiguous (B, T, H, 128) q and k, the layout model/gpt.py passes), relu2_mlp
(QuACK relu_sq act/dact) at the table's widths, lm_loss / lm_losses (QuACK cross entropy), embedding (lookup exact;
gradient vs fp64 and bitwise equal across two calls). Prints rel L2 errors. Sizes T 2048 and M 8192 are fixtures;
the embedding's ids are Zipf-distributed (fixture), so frequent tokens span many partials as real text's do."""
import numpy as np
import torch
import torch.nn.functional as F

from nanochat.config import FILE_STAGE, VOCAB_SIZE
from nanochat.model.gpt import ASPECT_RATIO, gpt_config, rotary_tables
from nanochat.ops.cross_entropy import lm_loss, lm_losses
from nanochat.ops.embedding import embedding
from nanochat.ops.linear import relu2_mlp
from nanochat.ops.qk_prologue import qk_norm_rope
from nanochat.ops.quack_gemm import TABLE

from tests.reference import rel, rope

torch.manual_seed(0)
dev = "cuda"


def show(names, ours, base, ref):
    for n, o, b, r in zip(names, ours, base, ref):
        print(f"  {n:10s} ours {rel(o, r):.3e}   bf16 eager {rel(b, r):.3e}")


# qk_norm_rope at d24's heads, T 2048
cfg = gpt_config(24, VOCAB_SIZE, FILE_STAGE, aspect_ratio=ASPECT_RATIO)
cos, sin = rotary_tables(cfg, dev)
B, T, H, D = 2, 2048, cfg.n_head, 128
q = torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
gq = torch.empty(D, device=dev).uniform_(0.5, 1.5).requires_grad_()
gk = torch.empty(D, device=dev).uniform_(0.5, 1.5).requires_grad_()
dq, dk = torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16), torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16)
qo, ko = qk_norm_rope(q, k, gq, gk, cos, sin)
torch.autograd.backward([qo, ko], [dq, dk])
ours = [qo, ko, q.grad, k.grad, gq.grad, gk.grad]
q.grad = k.grad = gq.grad = gk.grad = None
c, s = cos[None, :T, None].to(torch.bfloat16), sin[None, :T, None].to(torch.bfloat16)
qb = rope(F.rms_norm(q, (D,)) * gq.to(torch.bfloat16), c, s)
kb = rope(F.rms_norm(k, (D,)) * gk.to(torch.bfloat16), c, s)
torch.autograd.backward([qb, kb], [dq, dk])
base = [qb, kb, q.grad, k.grad, gq.grad, gk.grad]
q64, k64 = q.detach().double().requires_grad_(), k.detach().double().requires_grad_()
gq64, gk64 = gq.detach().double().requires_grad_(), gk.detach().double().requires_grad_()
c, s = cos[None, :T, None].double(), sin[None, :T, None].double()
eps = torch.finfo(torch.float32).eps
q6 = rope(F.rms_norm(q64, (D,), eps=eps) * gq64, c, s)
k6 = rope(F.rms_norm(k64, (D,), eps=eps) * gk64, c, s)
torch.autograd.backward([q6, k6], [dq.double(), dk.double()])
print(f"qk_norm_rope B {B} T {T} H {H}")
show(["q", "k", "dq", "dk", "dq_gain", "dk_gain"], ours, base, [q6, k6, q64.grad, k64.grad, gq64.grad, gk64.grad])

# relu2_mlp at the table's widths, M 8192
for C in TABLE:
    M = 8192
    s = 3 ** 0.5 * C ** -0.5
    x = torch.randn(M, C, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w_fc = torch.empty(4 * C, C, device=dev).uniform_(-s, s).requires_grad_()
    w_proj = torch.empty(C, 4 * C, device=dev).uniform_(-s, s).requires_grad_()
    dy = torch.randn(M, C, device=dev, dtype=torch.bfloat16)
    y, _ = relu2_mlp(x, w_fc, w_proj)
    y.backward(dy)
    ours = [y, x.grad, w_fc.grad, w_proj.grad]
    x.grad = w_fc.grad = w_proj.grad = None
    yb = F.linear(F.relu(F.linear(x, w_fc.to(torch.bfloat16))).square(), w_proj.to(torch.bfloat16))
    yb.backward(dy)
    base = [yb, x.grad, w_fc.grad, w_proj.grad]
    x6, f6, p6 = (t.detach().double().requires_grad_() for t in (x, w_fc, w_proj))
    y6 = F.linear(F.relu(F.linear(x6, f6)).square(), p6)
    y6.backward(dy.double())
    print(f"relu2_mlp C {C}")
    show(["y", "dx", "dw_fc", "dw_proj"], ours, base, [y6, x6.grad, f6.grad, p6.grad])

# lm_loss / lm_losses at the table's widths, M 8192, vocabulary 2^15, 5% ignored targets
for C in TABLE:
    M, V = 8192, VOCAB_SIZE
    x = torch.randn(M, C, device=dev).to(torch.bfloat16).requires_grad_()
    w = (torch.randn(V, C, device=dev) * 0.05).requires_grad_()
    t = torch.randint(0, V, (M,), device=dev)
    t[torch.rand(M, device=dev) < 0.05] = -1
    loss = lm_loss(x, w, t)[0]
    loss.backward()
    with torch.no_grad():
        losses = lm_losses(x, w, t)
    ours = [loss, x.grad, w.grad]
    x.grad = w.grad = None
    lb = F.cross_entropy(F.linear(x, w.to(torch.bfloat16)).float(), t, ignore_index=-1)
    lb.backward()
    base = [lb, x.grad, w.grad]
    x6, w6 = x.detach().double().requires_grad_(), w.detach().double().requires_grad_()
    ls6 = F.cross_entropy(F.linear(x6, w6), t, ignore_index=-1, reduction="none")
    l6 = ls6[t != -1].mean()
    l6.backward()
    print(f"lm_loss C {C}")
    show(["loss", "dx", "dw"], ours, base, [l6, x6.grad, w6.grad])
    print(f"  lm_losses ours {rel(losses, ls6):.3e}; ignored rows max |loss| {losses[t == -1].abs().max().item():.1e}")

# embedding at d24's width, the micro-batch's tokens, vocabulary 2^15
C, V = cfg.n_embd, VOCAB_SIZE
rng = np.random.default_rng(0)
idx = torch.tensor(np.minimum(rng.zipf(1.1, (FILE_STAGE.batch_rows, FILE_STAGE.seq_len)), V) - 1, dtype=torch.int32,
                   device=dev)
table = (torch.randn(V, C, device=dev) * C ** -0.5).requires_grad_()
dy = torch.randn(*idx.shape, C, device=dev, dtype=torch.bfloat16)
out = embedding(table, idx)
grads = [torch.autograd.grad(embedding(table, idx), table, dy)[0] for _ in range(2)]
dw6 = torch.zeros(V, C, dtype=torch.float64, device=dev).index_add_(0, idx.flatten().long(), dy.reshape(-1, C).double())
wb = F.embedding(idx, table).to(torch.bfloat16)
print(f"embedding C {C}, {idx.numel()} ids, {idx.unique().numel()} distinct, most frequent "
      f"x{idx.flatten().bincount().max().item()}")
print(f"  lookup exact: {torch.equal(out, wb)}; dw ours {rel(grads[0], dw6):.3e}; bitwise equal across calls: "
      f"{torch.equal(grads[0], grads[1])}")
