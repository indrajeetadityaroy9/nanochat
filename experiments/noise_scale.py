"""
The gradient noise scale of a run's final model (McCandlish et al., arXiv 1812.06162 §2.2 and App. A.1): B_simple =
tr(Sigma) / |G|^2, the batch (in target tokens) at which the gradient's noise matches its signal, the critical batch's
estimate. From the K micro-batch gradients g_k of one of the run's steps (each of the mean loss over its B_small valid
targets; K the plan's micro-batches per step) and their mean g (B_big = K B_small, the step's batch):
|G|^2 = (B_big |g|^2 - B_small mean |g_k|^2) / (B_big - B_small) and tr(Sigma) = (mean |g_k|^2 - |g|^2) /
(1 / B_small - 1 / B_big), unbiased estimates of the true gradient's squared norm and of the per-token gradient
covariance's trace. The micro-batches: the training mixture's val rows at the file-level stage (data/loader.py, FIM
read as in training, one permutation by seed 0), every parameter's gradient (as clip_grad_norm_ sees it), squared norms
summed in fp64. Written to <base_dir>/noise_scale/<tag>.json with the tokens the run trained.

    python -m experiments.noise_scale TAG
"""
import json
import os
import sys
from itertools import islice

import numpy as np
import torch

from nanochat.config import FILE_STAGE, MIXTURE
from nanochat.data.loader import PretrainingBatches, mixture_weights, packed_rows
from nanochat.paths import base_dir, write_json
from nanochat.train.checkpoint import load_model

tag = sys.argv[1]
with open(os.path.join(base_dir(), "metrics", f"{tag}.jsonl"), encoding="utf-8") as f:
    plan = next(r for r in map(json.loads, f) if r["kind"] == "run")["plan"]
k = plan["grad_accum_steps"]
device = torch.device("cuda")
model, tokenizer, step = load_model(device, tag)
compiled = torch.compile(model, dynamic=False, fullgraph=True)
mixture = mixture_weights(MIXTURE, tokenizer, FILE_STAGE.seq_len)
orders = {c: np.random.RandomState(0).permutation(packed_rows(c, "val", tokenizer, FILE_STAGE.seq_len)) for c in mixture}
batches = PretrainingBatches(mixture, "val", tokenizer, orders, seq_len=FILE_STAGE.seq_len,
                             batch_rows=FILE_STAGE.batch_rows, device=device, reading="fim", start=0)
params = list(model.parameters())
total = [torch.zeros_like(p) for p in params]
small = torch.zeros((), dtype=torch.float64, device=device)
valid = []
for x, y, cu_seqlens in islice(batches, k):
    compiled(x, y, cu_seqlens).backward()
    small += sum(p.grad.double().square().sum() for p in params)
    for t, p in zip(total, params, strict=True):
        t += p.grad
    model.zero_grad()
    valid.append((y != PretrainingBatches.PAD).sum())
b_small = torch.stack(valid).double().mean().item()
b_big = k * b_small
g_small2, g_big2 = (small / k).item(), sum((t / k).double().square().sum() for t in total).item()
g2 = (b_big * g_big2 - b_small * g_small2) / (b_big - b_small)
trace = (g_small2 - g_big2) / (1 / b_small - 1 / b_big)
os.makedirs(os.path.join(base_dir(), "noise_scale"), exist_ok=True)
write_json(os.path.join(base_dir(), "noise_scale", f"{tag}.json"),
           {"tag": tag, "step": step, "tokens": step * plan["total_batch_size"], "micro_batches": k,
            "b_small": b_small, "g_small2": g_small2, "g_big2": g_big2, "g2": g2, "trace": trace,
            "b_simple": trace / g2})
