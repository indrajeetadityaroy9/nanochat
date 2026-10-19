"""
A run's metrics: JSON lines in <base_dir>/metrics/<tag>.jsonl, each with "kind", "time" (Unix seconds) and its fields.

run (each process start): step, plan (schedule.RunPlan), seed, data_seed, variant (base_train.Variant), lr and
    weight_decay (as the optimizers hold them: a resumed process's are its checkpoint's), stage (config.Stage as the
    model runs it, mixture as {corpus: row weight}).
stage (switch to the repo-level stage): step, stage.
train (each optimizer step, 0-based): step; tokens (row positions after it, padding included); loss (mean next-token
    loss over the step's valid targets, fp64); lr (LambdaLR's at the step); grad_norm (before clipping); tok_per_s (row
    positions per second between the step's two CUDA synchronizations).
"""

import json
import os
import time

from nanochat.paths import base_dir


def append(tag, record):
    path = os.path.join(base_dir(), "metrics", f"{tag}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.time(), **record}) + "\n")
