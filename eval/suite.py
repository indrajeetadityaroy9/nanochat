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
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import torch

from corpus.tokenizer import Tokenizer
from eval.bpb import likelihood
from eval.code_eval import (
    filtered_generation,
    fim_filtered_generation,
    fim_generation,
    generation,
    parse_generation,
    sanitizers,
    tasks,
)
from model.modeling import CausalLM
from paths import RUNS


def evaluation_path(run_name: str, step: int) -> Path:
    return RUNS / run_name / f"eval-{step}.json"


def resolved(records: dict) -> dict:
    """records with each future (code_eval's scoring) replaced by its result, in the same key order."""
    return {k: v.result() if isinstance(v, Future) else resolved(v) for k, v in records.items()}


def evaluate(model: CausalLM, tokenizer: Tokenizer, *, run_name: str, step: int,
             generate: Literal["benchmarks", "parse"]) -> None:
    """evaluation_path(run_name, step) of `model` (on the GPU), run `run_name`'s checkpoint at `step`; EvalPlus's
    sanitize workers (code_eval.sanitizers) live for the call. The GPU work runs back to back: the unconstrained
    decodes (parse_generation; with generate "benchmarks", generation and fim_generation), the likelihood metrics, then
    the filtered searches (filtered_generation, fim_filtered_generation); each benchmark's scoring (CPU: its official
    evaluator, a worker per CPU) is queued on one scorer thread as soon as its completions exist and runs while the
    GPU works on. The searches come last: their own checks are timed (verify.py), and the likelihood pass, which uses
    little CPU, has drained most of the scoring queue by then. The records are collected at the end. Each part's
    result is the same in any order (decoding and scoring depend on the model and the tasks alone, decoder.NativeDecoder)
    but for timed checks, which heavy CPU load can fail (EvalPlus's time limits, verify.py's and the evaluators'
    timeouts)."""
    benchmarks = tasks()
    codegen = fim = None
    with sanitizers() as pool, ThreadPoolExecutor(1) as scorers:
        parse = parse_generation(model, tokenizer, benchmarks, pool=pool, scorers=scorers)
        if generate == "benchmarks":
            codegen = generation(model, tokenizer, benchmarks, pool=pool, scorers=scorers)
            fim = fim_generation(model, tokenizer, scorers=scorers)
        results = likelihood(model, tokenizer, benchmarks, device=torch.device("cuda"))
        if generate == "benchmarks":
            for name, record in filtered_generation(model, tokenizer, benchmarks, pool=pool, scorers=scorers).items():
                codegen[name]["filtered"] = record
            for key, record in fim_filtered_generation(model, tokenizer, scorers=scorers).items():
                fim[key]["filtered"] = record
        results |= {"parse": resolved(parse), "codegen": None if codegen is None else resolved(codegen),
                    "fim": None if fim is None else resolved(fim)}
    evaluation = {"run_name": run_name, "step": step} | results
    evaluation_path(run_name, step).write_text(json.dumps(evaluation, allow_nan=False), encoding="utf-8")


def record(run_name: str, log: list[dict]) -> dict:
    """A run's tracked metrics but the per-step ones: the non-train records of its log (metrics.records) and the
    evaluations (evaluate) of its checkpoints on disk, in step order (a diverged run has none)."""
    paths = (RUNS / run_name).glob("eval-*.json")
    return {"run_name": run_name, "records": [r for r in log if r["kind"] != "train"],
            "evaluations": sorted((json.loads(p.read_text(encoding="utf-8")) for p in paths), key=lambda r: r["step"])}
