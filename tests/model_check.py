"""GPT as trained (torch.compile dynamic=False fullgraph, bf16 compute, fused ops) against the fp64 reference
(reference.py) at d8 (width 256), T 1024 (fixture; short windows 512 active), two rows of several documents (fixture
boundaries), perturbed weights (reference.perturb: every path non-zero). Baseline: the same reference in bf16 eager.
Prints the loss and per-parameter gradient rel L2 errors; negative controls: the reference with x0_lambdas x1.01 / the
VE gate weights zeroed (gate 1)."""
import dataclasses
import statistics

import torch

from nanochat.config import FILE_STAGE, VOCAB_SIZE
from nanochat.data.loader import PretrainingBatches
from nanochat.model.gpt import ASPECT_RATIO, GPT, gpt_config

from tests.reference import perturb, reference_losses, rel

torch.manual_seed(0)
dev = "cuda"
T, V = 1024, VOCAB_SIZE
model = GPT(dataclasses.replace(gpt_config(8, V, FILE_STAGE, aspect_ratio=ASPECT_RATIO), sequence_len=T)).to(dev)
model.init_weights()
perturb(model)
print(f"width {model.config.n_embd} windows {model.window_sizes}")
idx = torch.randint(0, V, (2, T), device=dev, dtype=torch.int32)
targets = torch.randint(0, V, (2, T), device=dev)
targets[0, 299] = -1
targets[1, 1000:] = -1
cu = torch.tensor([0, 300, 800, 1024, 1500, 2048], dtype=torch.int32, device=dev)
compiled = torch.compile(model, dynamic=False, fullgraph=True)

loss = compiled(idx, targets, cu_seqlens=cu)
loss.backward()
with torch.no_grad():
    losses = model.losses(idx, targets, cu)
valid = targets.reshape(-1) != PretrainingBatches.PAD
l64, P = reference_losses(model, idx.long(), targets, cu)
loss64 = l64[valid].mean()
loss64.backward()
l16, P16 = reference_losses(model, idx.long(), targets, cu, dtype=torch.bfloat16)
loss16 = l16[valid].mean()
loss16.backward()
print(f"loss ours {loss.item():.6f} bf16 eager {loss16.item():.6f} ref {loss64.item():.6f}: rel ours "
      f"{abs(loss.item() - loss64.item()) / loss64.item():.2e} bf16 eager {abs(loss16.item() - loss64.item()) / loss64.item():.2e}")
print(f"per-position losses rel ours {rel(losses, l64.detach()):.2e} bf16 eager {rel(l16.detach(), l64.detach()):.2e}")
worse = []
for n, p in model.named_parameters():
    e, e16 = rel(p.grad, P[n].grad), rel(P16[n].grad, P[n].grad)
    worse += [n] if e > e16 else []
    print(f"  {n:40s} grad rel ours {e:.2e} bf16 eager {e16:.2e}")
print(f"ours worse than bf16 eager: {worse}")
for name, edit in (("x0_lambdas x1.01", lambda Q: Q["x0_lambdas"].mul_(1.01)),
                   ("ve_gate zeroed", lambda Q: [Q[n].zero_() for n in Q if "ve_gate" in n])):
    Q = {n: t.detach().clone() for n, t in P.items()}
    with torch.no_grad():
        edit(Q)
    Q = {n: t.requires_grad_() for n, t in Q.items()}
    reference_losses(model, idx.long(), targets, cu, params=Q)[0][valid].mean().backward()
    diffs = [rel(Q[n].grad, P[n].grad) for n in P if Q[n].ndim == 2]
    print(f"control {name}: matrix-grad rel diff vs ref median {statistics.median_high(diffs):.2e} "
          f"min {min(diffs):.2e}")
