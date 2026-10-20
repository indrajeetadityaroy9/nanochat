"""Profile one training step (scripts/base_train.optimizer_step, tau-law weight decay, no clipping) on a stage's packed
rows (fim reading): wall time of the first steady-state step, then one step under torch.profiler (GPU kernel time by
category and per kernel, busy fraction).
Warm-up: one step (graph compile, optimizer state), then steps while each is slower than the previous (d24 12 x 12:
3,461 / 3,719 / 3,775 / 3,790 / 3,795 / 3,781 ms, ablations/ablations.md).
Usage: python -m ablations.profile_step DEPTH MICRO_STEPS ROWS file|repo"""
import collections
import math
import sys
import time

import numpy as np
import torch

from nanochat.config import FILE_STAGE, REPO_STAGE
from nanochat.data.loader import PretrainingBatches, mixture_weights, packed_rows
from nanochat.data.tokenizer import get_tokenizer
from nanochat.model.gpt import GPT, gpt_config
from nanochat.train.optim import optimizers, tau_weight_decay
from nanochat.train.schedule import plan
from nanochat.train.stages import max_ratio
from scripts.base_train import optimizer_step


def category(name):
    low = name.lower()
    if "flash" in low:
        return "attention (FA4)"
    if low.startswith(("k_forward", "k_backward", "k_gain_sum")):  # csrc/qk_prologue.cu
        return "q/k prologue"
    if "crossentropy" in low:
        return "cross entropy"
    if "gemm" in low or "nvjet" in low:  # QuACK, CUTLASS, MAGMA; cuBLAS nvjet
        return "GEMM"
    if "triton" in low:
        return "triton (inductor fused)"
    if "memcpy" in low or "memset" in low:
        return "copy/memset"
    return "other"


def main():
    depth, micro, rows = (int(a) for a in sys.argv[1:4])
    stage = {"file": FILE_STAGE, "repo": REPO_STAGE}[sys.argv[4]]
    device = torch.device("cuda")
    tokenizer = get_tokenizer()
    # production's lr and weight decay at this depth (its tokens per parameter, train/stages.py max_ratio): lr sets how
    # far weights leave zero init (GEMM operands, hence power-capped clocks)
    with torch.device("meta"):
        run = plan(model=GPT(gpt_config(depth=depth, stage=FILE_STAGE)), ratio=max_ratio(tokenizer, depth=depth))
        orig = GPT(gpt_config(depth=depth, stage=stage))
    orig.to_empty(device=device)
    orig.init_weights(seed=0)  # fixed seeds (init, data order): every profile runs the same step
    model = torch.compile(orig, dynamic=False, fullgraph=True)
    opts = optimizers(orig, lr=run.lr, weight_decay=tau_weight_decay(lr=run.lr, steps=run.num_iterations,
                                                                      tokens_per_param=run.tokens_per_param))
    mixture = mixture_weights(stage.mixture, tokenizer, stage.seq_len)
    orders = {c: np.random.RandomState(0).permutation(packed_rows(c, "train", tokenizer, stage.seq_len))
              for c in mixture}
    reads = iter(PretrainingBatches(mixture, "train", tokenizer, orders, seq_len=stage.seq_len, batch_rows=rows,
                                    device=device, reading="fim", start=0))
    params = list(orig.parameters())
    batches = [next(reads) for _ in range(micro)]

    def step():
        """Wall time (s) of one optimizer step."""
        nonlocal batches
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, _, batches = optimizer_step(model, batches, params, opts, reads, max_norm=math.inf)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    step()
    times = [step(), step()]
    while times[-1] > times[-2]:
        times.append(step())
    ms = times[-1] * 1e3  # s -> ms
    tokens = micro * rows * stage.seq_len
    print(f"d{depth}: {micro} micro-batches ({tokens:,} tokens): step {len(times) + 1} {ms:.0f} ms -> "
          f"{tokens / ms * 1e3:,.0f} tok/s; peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        step()
    by_name = collections.Counter()
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            by_name[e.name] += e.device_time_total / 1e3  # us -> ms
    cats = collections.Counter()
    for name, t in by_name.items():
        cats[category(name)] += t
    busy = cats.total()
    print(f"profiled step: GPU kernel time {busy:.0f} ms of wall {ms:.0f} ms ({busy / ms:.0%} busy)")
    for c, t in cats.most_common():
        print(f"  {c} {t:.1f} ms {t / busy:.1%}")
    print("kernels:")
    for name, t in by_name.most_common():
        print(f"  {t:.1f} ms {category(name)} {name}")


if __name__ == "__main__":
    main()
