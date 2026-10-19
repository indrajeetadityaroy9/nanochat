"""
The metrics record of a training run: JSON lines in <base_dir>/metrics/<tag>.jsonl, each with "kind", "time" (Unix
seconds, added by append) and the fields of its kind.

kind "run" (one per process start): step (the step the process starts from), args (the command line), plan
    (train/schedule.py RunPlan: batch, steps, the repo-level stage's first step, tokens per parameter, peak lr), stage
    (the stage of that step, config.Stage: seq_len, rope_base, mixture as {corpus: row weight}).
kind "stage" (at the switch to the repo-level stage): step, stage (as in "run").
kind "train" (every optimizer step): step (0-based), tokens (row positions after it, padding included), loss (the step's
    training loss: the mean next-token loss over the step's valid targets, summed over its micro-batches in fp64), lr
    (the lr of every group at the step: the plan's lr x train/schedule.py lr_multiplier, by LambdaLR), grad_norm (L2
    norm of the gradient of every parameter before clipping to --grad-clip, torch.nn.utils.clip_grad_norm_),
    tok_per_s (row positions per second, padding included, as tokens, over the step's time between two CUDA
    synchronizations: its loader reads, forward/backward passes and optimizer step).
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
