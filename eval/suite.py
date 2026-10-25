"""
A checkpoint's evaluation and a run's tracked metrics but the per-step ones.

evaluate: RUNS/<tag>/eval-step<step>.json (evaluation_path; tag: the run's key, run.py), one JSON object: run
(the tag), step, the likelihood metrics (eval/bpb.py likelihood: "8k", "32k", "solution"), how healed greedy
completions parse (eval/code_eval.py parse_generation: "parse") and, with generate "benchmarks", the generation
benchmarks (eval/code_eval.py: "codegen", "fim"; null with generate "parse", the ablation runs' evaluation). NaN and
infinities are not JSON and raise.

record: a run's records but the per-step ones (metrics.py: run, stage, end, failure) and the evaluations of its
checkpoints verbatim, in step order; an experiment's RESULTS/<experiment>.jsonl holds one per run
(ablations/records.py).
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import torch

import metrics
from corpus.tokenizer import Tokenizer
from eval.bpb import likelihood
from eval.code_eval import fim_generation, generation, parse_generation, sanitizers, tasks
from model.gpt import GPT
from paths import RUNS


def evaluation_path(tag: str, step: int) -> Path:
    return RUNS / tag / f"eval-step{step}.json"


def evaluate(model: GPT, tokenizer: Tokenizer, *, tag: str, step: int,
             generate: Literal["benchmarks", "parse"]) -> None:
    """evaluation_path(tag, step) of `model` (on the GPU), run `tag`'s checkpoint after `step` steps; EvalPlus's
    sanitize workers (code_eval.sanitizers) live for the call. parse_generation's completions are decoded first, so
    their EvalPlus scoring (CPU) runs while the GPU computes the likelihood metrics; generate "benchmarks" starts after
    that scoring ends, so its timed checks (verify.py) never share the CPUs with it. Each part's result is the same in
    any order: decoding and scoring depend on the model and the tasks alone (decoder.NativeDecoder)."""
    benchmarks = tasks()
    with sanitizers() as pool, ThreadPoolExecutor() as scorers:
        parse = parse_generation(model, tokenizer, benchmarks, pool=pool, scorers=scorers)
        results = likelihood(model, tokenizer, benchmarks, device=torch.device("cuda"))
        results |= {"parse": {name: record.result() for name, record in parse.items()}}
        match generate:
            case "benchmarks":
                results |= {"codegen": generation(model, tokenizer, benchmarks, pool=pool),
                            "fim": fim_generation(model, tokenizer)}
            case "parse":
                results |= {"codegen": None, "fim": None}
    evaluation_path(tag, step).write_text(json.dumps({"run": tag, "step": step} | results, allow_nan=False),
                                          encoding="utf-8")


def record(tag: str) -> dict:
    """A run's tracked metrics but the per-step ones: its non-train records and the evaluations (evaluate) of its
    checkpoints on disk, in step order (a diverged run has none)."""
    paths = (RUNS / tag).glob("eval-step*.json")
    return {"tag": tag, "records": [r for r in metrics.records(tag) if r["kind"] != "train"],
            "evaluations": sorted((json.loads(p.read_text(encoding="utf-8")) for p in paths), key=lambda r: r["step"])}
