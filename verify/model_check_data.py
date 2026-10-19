"""model_check.py on the data pipeline's tokens: the first micro-batch of a stage's packed train rows in seed 42 order
(all packed rows; base_train permutes train/stages.py readable_rows, a subset for the repository corpora), at the
stage's mixture, reading "fim": packed pieces, FIM examples with their sentinels, document-masking segments, padding
targets -1. A freshly initialized model at DEPTH perturbed by reference.perturb (every path non-zero). First the whole
micro-batch through the compiled model (torch.compile dynamic=False fullgraph, as base_train): loss, gradient norm,
step time, peak memory. Then ROWS rows alone (B = 1, T = the stage's context) as trained vs the fp64 reference, next
to the bf16 eager formulation; negative controls on the reference.
Usage: python -m verify.model_check_data DEPTH file|repo ROWS [CHECKPOINT_TAG (trained weights instead of the random
init)]"""
import statistics
import sys
import time

import numpy as np
import torch

from nanochat.config import FILE_STAGE, REPO_STAGE
from nanochat.data.loader import PretrainingBatches, mixture_weights, packed_rows
from nanochat.data.tokenizer import get_tokenizer
from nanochat.model.gpt import ASPECT_RATIO, GPT, gpt_config
from nanochat.train.checkpoint import load_model

from verify.reference import perturb, reference_losses, rel

depth, stage, n_rows = int(sys.argv[1]), {"file": FILE_STAGE, "repo": REPO_STAGE}[sys.argv[2]], int(sys.argv[3])
torch.manual_seed(0)
device = torch.device("cuda")
tokenizer = get_tokenizer()
T = stage.seq_len
if len(sys.argv) > 4:  # a trained checkpoint's weights (its depth must be DEPTH), at the requested stage
    model = load_model(device, sys.argv[4])[0]
    model.set_stage(stage)
else:
    with torch.device("meta"):
        model = GPT(gpt_config(depth, tokenizer.vocab_size, stage, aspect_ratio=ASPECT_RATIO))
    model.to_empty(device=device)
    model.init_weights()
    perturb(model)
mixture = mixture_weights(stage.mixture, tokenizer, T)
orders = {c: np.random.RandomState(42).permutation(packed_rows(c, "train", tokenizer, T)) for c in mixture}
x, y, cu = next(iter(PretrainingBatches(mixture, "train", tokenizer, orders, seq_len=T,
                                        batch_rows=stage.batch_rows, device=device, reading="fim", start=0)))
print(f"d{depth} width {model.config.n_embd} heads {model.config.n_head} context {T} rope {model.config.rope_base} "
      f"windows {model.window_sizes}")
compiled = torch.compile(model, dynamic=False, fullgraph=True)
for it in range(3):
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss = compiled(x, y, cu_seqlens=cu)
    loss.backward()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    grad_norm = torch.cat([p.grad.flatten() for p in model.parameters()]).norm().item()
    print(f"micro-batch {tuple(x.shape)} it {it}: loss {loss.item():.4f} grad norm {grad_norm:.4e} "
          f"{dt:.3f} s ({x.numel() / dt:,.0f} tok/s fwd+bwd) peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

sentinels = torch.tensor(sorted(tokenizer.special_tokens.values()), device=device)
for r in range(n_rows):
    idx, targets = x[r:r + 1], y[r:r + 1]
    bounds = cu[(cu >= r * T) & (cu < (r + 1) * T)] - r * T
    row_cu = torch.cat([bounds, torch.tensor([T], dtype=torch.int32, device=device)]).unique_consecutive()
    valid = targets.reshape(-1) != -1
    print(f"row {r}: {len(row_cu) - 1} segments, {valid.sum().item()} valid targets, "
          f"{torch.isin(idx, sentinels).sum().item()} special-token inputs")
    model.zero_grad(set_to_none=True)
    loss = compiled(idx, targets, cu_seqlens=row_cu)
    loss.backward()
    l64, P = reference_losses(model, idx.long(), targets, row_cu)
    loss64 = l64[valid].mean()
    loss64.backward()
    l16, P16 = reference_losses(model, idx.long(), targets, row_cu, dtype=torch.bfloat16)
    l16[valid].mean().backward()
    print(f"  loss ours {loss.item():.6f} ref {loss64.item():.6f} rel {abs(loss.item() - loss64.item()) / loss64.item():.2e}; "
          f"bf16 eager rel {abs(l16[valid].mean().item() - loss64.item()) / loss64.item():.2e}")
    worse, worst = [], 0.0
    for n, p in model.named_parameters():
        e, e16 = rel(p.grad, P[n].grad), rel(P16[n].grad, P[n].grad)
        worst = max(worst, e)
        worse += [f"{n} {e:.2e} vs {e16:.2e}"] if e > e16 else []
    print(f"  worst grad rel {worst:.2e}; ours worse than bf16 eager: {worse}")
    for name, edit in (("x0_lambdas x1.01", lambda Q: Q["x0_lambdas"].mul_(1.01)),
                       ("ve_gate zeroed", lambda Q: [Q[n].zero_() for n in Q if "ve_gate" in n])):
        Q = {n: t.detach().clone() for n, t in P.items()}
        with torch.no_grad():
            edit(Q)
        Q = {n: t.requires_grad_() for n, t in Q.items()}
        lc = reference_losses(model, idx.long(), targets, row_cu, params=Q)[0][valid].mean()
        lc.backward()
        diffs = [rel(Q[n].grad, P[n].grad) for n in P if Q[n].ndim == 2]
        print(f"  control {name}: loss rel diff {abs(lc.item() - loss64.item()) / loss64.item():.2e}, "
              f"matrix-grad rel diff median {statistics.median_high(diffs):.2e} min {min(diffs):.2e}")
    del P, P16, l64, l16
    torch.cuda.empty_cache()
