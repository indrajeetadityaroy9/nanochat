"""
A checkpoint's evaluation and a run's tracked metrics.

evaluate: BASE_EVAL/<name>.json (name: train/checkpoint.checkpoint_name), one JSON object: run (the name), step, the
likelihood metrics (eval/bpb.py likelihood: "8k", "32k", "solution"), how healed greedy completions parse
(eval/code_eval.py parse_generation: "parse") and, with generate "benchmarks", the generation benchmarks
(eval/code_eval.py: "codegen", "fim"; null with generate "parse", the ablation runs' evaluation). NaN and infinities
are not JSON and raise.

export: RESULTS/<tag>.jsonl, one record (record): the run's records but the per-step ones (train/metrics.py: run,
stage, end, failure) and the evaluations of its checkpoints verbatim, in step order; RESULTS/<tag>-curves.jsonl, its
per-step train records as columns (train/metrics.py curves). Tables and figures are drawn from these.
"""
import json
from typing import Literal

import torch

from nanochat.corpus.tokenizer import Tokenizer
from nanochat.eval.bpb import likelihood
from nanochat.eval.code_eval import fim_generation, generation, parse_generation, tasks
from nanochat.model.gpt import GPT
from nanochat.paths import BASE_EVAL, RESULTS
from nanochat.train import metrics


def evaluate(model: GPT, tokenizer: Tokenizer, *, name: str, step: int,
             generate: Literal["benchmarks", "parse"]) -> None:
    """BASE_EVAL/<name>.json of `model` (on the GPU), the checkpoint after `step` steps."""
    benchmarks = tasks()
    results = likelihood(model, tokenizer, benchmarks, device=torch.device("cuda"))
    results |= {"parse": parse_generation(model, tokenizer, benchmarks)}
    match generate:
        case "benchmarks":
            results |= {"codegen": generation(model, tokenizer, benchmarks), "fim": fim_generation(model, tokenizer)}
        case "parse":
            results |= {"codegen": None, "fim": None}
    (BASE_EVAL / f"{name}.json").write_text(json.dumps({"run": name, "step": step} | results, allow_nan=False),
                                            encoding="utf-8")


def record(tag: str) -> dict:
    """A run's tracked metrics but the per-step ones: its non-train records and the evaluations (evaluate) of its
    checkpoints on disk, in step order (a diverged run has none)."""
    paths = BASE_EVAL.glob(f"{tag}-step*.json")
    return {"tag": tag, "records": [r for r in metrics.records(tag) if r["kind"] != "train"],
            "evaluations": sorted((json.loads(p.read_text(encoding="utf-8")) for p in paths), key=lambda r: r["step"])}


def export(tag: str) -> None:
    """RESULTS/<tag>.jsonl (record) and RESULTS/<tag>-curves.jsonl (metrics.curves)."""
    (RESULTS / f"{tag}.jsonl").write_text(json.dumps(record(tag), allow_nan=False) + "\n", encoding="utf-8")
    (RESULTS / f"{tag}-curves.jsonl").write_text(json.dumps({"tag": tag} | metrics.curves(tag), allow_nan=False) + "\n",
                                                 encoding="utf-8")
