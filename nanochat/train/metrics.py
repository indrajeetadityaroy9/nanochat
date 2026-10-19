"""
The metrics record of a training run: JSON lines in <base_dir>/metrics/<tag>.jsonl, each with "kind", "time" (Unix
seconds, added by append) and the fields of its kind.

kind "run" (one per process start): step (the step the process starts from), args (the command line), plan
    (train/schedule.py RunPlan), optimizer (every optimizer group's hyperparameters, train/optim.py param_groups; lr and
    initial_lr the peak), mixture ({corpus: row weight}).
kind "train" (every optimizer step): step (0-based), tokens (row positions after it, padding included), lrm (the lr
    multiplier), loss (the step's training loss: sum over the columns of weight x the column's summed loss over the
    step's valid targets / the step's valid next-token targets, ops/softcap_ce.py), smooth_loss (loss's debiased EMA,
    beta 0.9, nanochat 92d63d4 scripts/base_train.py:548-550), loss_columns ({column: its mean loss over its own valid
    targets of the step}), grad_norm (L2 norm of the gradient of every parameter, the Engram tables included,
    torch.nn.utils.get_total_norm), engram_gate ({Engram layer: the mean of its gate alpha (Eq. 4) over the step's
    tokens}), weight_rms and update_rms ({optimizer group: RMS over the group's elements of its weights after the step,
    of the step's change of them}), step_ms and tok_per_s (the step's compute and its loader reads), tflops (achieved
    TFLOP/s: tok_per_s x the model's training FLOPs per token, train/schedule.py flops_per_token), peak_gib (the caching
    allocator's peak allocated memory in the step). Sums are fp64 (torchmetrics SumMetric). lrm is the lr schedule's
    shape; the groups with a warmup (train/optim.py param_groups) ramp on top of it.
"""

import json
import os
import time

from nanochat.paths import base_dir


def append(tag, record):
    """Append the record (a JSON-serializable dict with "kind") as one line to the run's metrics file."""
    path = os.path.join(base_dir(), "metrics", f"{tag}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.time(), **record}) + "\n")
