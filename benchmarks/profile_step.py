"""Profile base_train's optimizer step (scripts/base_train.optimizer_step: the one step every run takes, its next step's
micro-batches read during its backward) on this GPU, on the data pipeline's packed rows of a stage (fim reading): wall
time per step over 3 steps after 3 warmup steps, then one step under torch.profiler: GPU kernel time by category and
the top kernels; GPU busy fraction of the step. The lr is the plan's peak (the schedule only scales it). SWITCHES: the
model's GPTConfig switches and additions as JSON (the canonical model without).
Usage: python -m benchmarks.profile_step DEPTH MICRO_STEPS ROWS file|repo [SWITCHES]"""
import collections
import dataclasses
import json
import statistics
import sys
import time

import numpy as np
import torch

from nanochat.config import FILE_STAGE, REPO_STAGE
from nanochat.data.loader import PretrainingBatches, mixture_weights, packed_rows
from nanochat.data.tokenizer import get_tokenizer
from nanochat.model.gpt import GPT, gpt_config
from nanochat.train.optim import WEIGHT_DECAY, optimizers
from nanochat.train.schedule import plan
from scripts.base_train import optimizer_step

depth, micro, rows = (int(a) for a in sys.argv[1:4])
stage = {"file": FILE_STAGE, "repo": REPO_STAGE}[sys.argv[4]]
SEQ_LEN = stage.seq_len
device = torch.device("cuda")
tokenizer = get_tokenizer()
run = plan(depth, 20)  # fixture: only run.lr is used; 20 tokens per parameter (Chinchilla, arXiv 2203.15556 §3)
with torch.device("meta"):
    orig = GPT(dataclasses.replace(gpt_config(depth, stage), **json.loads(sys.argv[5] if len(sys.argv) > 5 else "{}")))
orig.to_empty(device=device)
orig.init_weights(0)
model = torch.compile(orig, dynamic=False, fullgraph=True)
opts = optimizers(orig, run.lr, WEIGHT_DECAY, "muon")
mixture = mixture_weights(stage.mixture, tokenizer, SEQ_LEN)
orders = {c: np.random.RandomState(0).permutation(packed_rows(c, "train", tokenizer, SEQ_LEN)) for c in mixture}
reads = iter(PretrainingBatches(mixture, "train", tokenizer, orders, seq_len=SEQ_LEN, batch_rows=rows, device=device,
                                reading="fim", start=0))
params = list(orig.parameters())
batches = [next(reads) for _ in range(micro)]


def step():
    """One canonical optimizer step; its wall time."""
    global batches
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _, _, batches = optimizer_step(model, batches, params, opts, reads)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


for _ in range(3):
    step()
total = statistics.median(step() for _ in range(3))
tokens = micro * rows * SEQ_LEN
print(f"d{depth}: {micro} micro-batches ({tokens:,} tokens): step {total * 1e3:.0f} ms -> {tokens / total:,.0f} tok/s; "
      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    step()
events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
by_name = collections.Counter()
for e in events:
    by_name[e.name] += e.device_time_total


def category(n):
    low = n.lower()
    if "flash" in low or "fmha" in low or "attn" in low:
        return "attention (FA4)"
    if "qkp" in low:
        return "q/k prologue"
    if "crossentropy" in low:
        return "cross entropy"
    if "gemm" in low or "cutlass" in low or "sm90" in low or "xmma" in low or "nvjet" in low:
        return "GEMM"
    if "triton" in low:
        return "triton (inductor fused)"
    if "memcpy" in low or "memset" in low or "copy" in low:
        return "copy/memset"
    return "other"


cats = collections.Counter()
for n, t in by_name.items():
    cats[category(n)] += t
busy = sum(cats.values()) / 1e3
print(f"profiled step: GPU kernel time {busy:.0f} ms of wall {total*1e3:.0f} ms ({100*busy/(total*1e3):.0f}% busy)")
for c, t in cats.most_common():
    print(f"  {c:26s} {t/1e3:9.1f} ms  {100*t/1e3/busy:5.1f}%")
print("top kernels:")
for n, t in by_name.most_common(25):
    print(f"  {t/1e3:9.1f} ms  {category(n):22s} {n[:110]}")
