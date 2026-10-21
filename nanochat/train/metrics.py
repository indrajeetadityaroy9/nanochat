"""
A run's training records: METRICS/<tag>.jsonl, UTF-8 JSON lines, each an object parseable on its own: tag, attempt (the
run's process start it belongs to, 0-based: a resumed run repeats the steps after its checkpoint, and a step's record
from the latest attempt supersedes the earlier ones; records are never rewritten), kind, time (Unix seconds), then the
kind's fields. NaN and infinities are not JSON and are never written. Checkpoint evaluations: eval/suite.py.

run (each process start): step (the first step it trains), plan (schedule.RunPlan), seed (initialization and data
    order), model (model/gpt.py GPTConfig as the model was built), variant (an ablation run's fields,
    ablations/harness.py; production's lr scale), params (the model's parameters), flops_per_token (model/gpt.py
    non_embedding_flops_per_token at the stage it starts in: DeepSeek LLM's M, arXiv 2401.02954 §3.1), max_norm (the
    gradient clip threshold), dataset_revision (config.DATASET_REVISION), lr and weight_decay (as the optimizers hold
    them; a resumed process's come from its checkpoint), stage (config.Stage as the model runs it, mixture as {corpus:
    row weight}).
stage (switch to the repo-level stage): step, stage.
train (each optimizer step, 0-based): step; tokens (row positions after it, padding included); loss (mean next-token
    loss in nats over the step's valid targets, fp64); lr (LambdaLR's at the step); grad_norm (global L2 norm of the
    gradient before clipping, the norm global-norm clipping thresholds: Pascanu et al., arXiv 1211.5063 §3.2);
    tok_per_s (row positions per second between the step's two CUDA synchronizations: the observed throughput of
    PaLM's model FLOPs utilization, arXiv 2204.02311 App. B).
end (the last step trained): step; peak_memory_bytes (torch.cuda.max_memory_allocated of the process: the memory
    footprint FlashAttention reports, Dao et al., arXiv 2205.14135).
failure (a non-finite loss or gradient norm; the run stops): step, loss and grad_norm (null).
"""
import json
import time

from nanochat.paths import METRICS


def append(tag: str, record: dict, *, attempt: int) -> None:
    line = json.dumps({"tag": tag, "attempt": attempt, "time": time.time(), **record}, allow_nan=False)
    with (METRICS / f"{tag}.jsonl").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def records(tag: str) -> list[dict]:
    """Run `tag`'s records in file order; none for a run not yet started."""
    path = METRICS / f"{tag}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def train_steps(tag: str) -> dict[int, dict]:
    """{step: train record}, each step's from its latest attempt (attempts only grow down the file)."""
    return {r["step"]: r for r in records(tag) if r["kind"] == "train"}


def curves(tag: str) -> dict[str, list]:
    """train_steps as columns, one per train record field (step, tokens, loss, lr, grad_norm, tok_per_s, attempt,
    time), in step order."""
    by_step = train_steps(tag)
    steps = [by_step[s] for s in sorted(by_step)]
    return {k: [s[k] for s in steps] for k in steps[0] if k not in ("tag", "kind")}
