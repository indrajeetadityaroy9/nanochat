"""
A run's metrics: METRICS/<tag>.jsonl, UTF-8 JSON lines, each an object parseable on its own: schema (SCHEMA), tag,
attempt (the run's process start it belongs to, 0-based: a resumed run repeats the steps after its checkpoint, and a
step's record from the latest attempt supersedes the earlier ones; records are never rewritten), kind, time (Unix
seconds), then the kind's fields. Unavailable values are null; NaN and infinities are not JSON and are never written.

run (each process start): step (the first step it trains), plan (schedule.RunPlan), seed, data_seed, model
    (model/gpt.py GPTConfig as the model was built), variant (an ablation run's fields, ablations/harness.py; {} for
    production), params and trained_params (the model's parameters, all and those the optimizers train),
    flops_per_token (model/gpt.py non_embedding_flops_per_token at the stage it starts in), max_norm (the gradient clip
    threshold; null: no clipping), lr and weight_decay (as the optimizers hold them; a resumed process's come from its
    checkpoint), stage (config.Stage as the model runs it, mixture as {corpus: row weight}).
stage (switch to the repo-level stage): step, stage.
train (each optimizer step, 0-based): step; tokens (row positions after it, padding included); loss (mean next-token
    loss in nats over the step's valid targets, fp64); lr (LambdaLR's at the step); grad_norm (global L2 norm of the
    trained parameters' gradient before clipping); tok_per_s (row positions per second between the step's two CUDA
    synchronizations).
end (the last step trained): step; peak_memory_bytes (torch.cuda.max_memory_allocated of the process).
failure (a non-finite loss or gradient norm; the run stops): step, loss and grad_norm (null).

Schema 1 (the pre-launch records in artifacts/metrics): no schema, tag or attempt fields, no model, variant or
max_norm.
"""

import json
import time

from nanochat.paths import METRICS

SCHEMA = 2


def attempt(tag: str) -> int:
    """The attempt a process starting run `tag` now is: the run records already written."""
    path = METRICS / f"{tag}.jsonl"
    if not path.exists():
        return 0
    return sum(json.loads(line)["kind"] == "run" for line in path.read_text(encoding="utf-8").splitlines())


def append(tag: str, record: dict, *, attempt: int) -> None:
    line = json.dumps({"schema": SCHEMA, "tag": tag, "attempt": attempt, "time": time.time(), **record},
                      allow_nan=False)
    with (METRICS / f"{tag}.jsonl").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def records(tag: str) -> list[dict]:
    return [json.loads(line) for line in (METRICS / f"{tag}.jsonl").read_text(encoding="utf-8").splitlines()]


def train_steps(tag: str) -> dict[int, dict]:
    """{step: train record}, each step's from its latest attempt (attempts only grow down the file)."""
    return {r["step"]: r for r in records(tag) if r["kind"] == "train"}


def latest(tag: str, kind: str) -> dict:
    """The last record of a kind."""
    return [r for r in records(tag) if r["kind"] == kind][-1]
