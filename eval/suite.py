"""
A checkpoint's evaluation and a run's tracked metrics but the per-step ones.

evaluate: RUNS/<run_name>/eval-<step>.json (evaluation_path; run_name: run.py), one JSON object: run_name, step, the
likelihood metrics (eval/bpb.py likelihood: "8k", "32k", "solution"), how healed greedy
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
from model.modeling import CausalLM
from paths import RUNS


def evaluation_path(run_name: str, step: int) -> Path:
    return RUNS / run_name / f"eval-{step}.json"


def evaluate(model: CausalLM, tokenizer: Tokenizer, *, run_name: str, step: int,
             generate: Literal["benchmarks", "parse"]) -> None:
    """evaluation_path(run_name, step) of `model` (on the GPU), run `run_name`'s checkpoint at `step`; EvalPlus's
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
    evaluation = {"run_name": run_name, "step": step} | results
    evaluation_path(run_name, step).write_text(json.dumps(evaluation, allow_nan=False), encoding="utf-8")


def record(run_name: str) -> dict:
    """A run's tracked metrics but the per-step ones: its non-train records and the evaluations (evaluate) of its
    checkpoints on disk, in step order (a diverged run has none)."""
    paths = (RUNS / run_name).glob("eval-*.json")
    return {"run_name": run_name, "records": [r for r in metrics.records(run_name) if r["kind"] != "train"],
            "evaluations": sorted((json.loads(p.read_text(encoding="utf-8")) for p in paths), key=lambda r: r["step"])}
