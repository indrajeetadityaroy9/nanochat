"""
A run's likelihood trajectory: eval/bpb.py likelihood of each saved model from a step on, the evaluated checkpoints'
from their evaluations (eval-<step>.json), the analysis checkpoints' (docs/training.md §6) scored here into
RUNS/<run>/likelihood-<step>.json; then each group's bits per byte, its paired change from a reference step and
between consecutive steps over identical rows and solutions (metrics.paired_rows), and the summed learning rate of the
updates between consecutive steps.

    python -m eval.trajectory <run_name> <reference step> <first step> <out.json>
"""
import gc
import json
import sys
import time
from pathlib import Path

import torch

import metrics
from corpus.mixture import FILE_LEVEL_STAGE, REPO_LEVEL_STAGE, Stage
from corpus.tokenizer import Tokenizer, get_tokenizer
from eval.bpb import likelihood
from eval.cache import environment
from eval.code_eval import tasks
from eval.suite import evaluation_path, source_digest, write_atomically
from paths import RUNS
from train.checkpoint import load_model

STAGES = {"8k": FILE_LEVEL_STAGE, "32k": REPO_LEVEL_STAGE}


def likelihood_path(run_name: str, step: int) -> Path:
    return RUNS / run_name / f"likelihood-{step}.json"


def saved_steps(run_name: str) -> list[int]:
    """The steps of the run's saved models: full states (checkpoint-<step>.pt) and weights-only files."""
    paths = [*(RUNS / run_name).glob("checkpoint-*.pt"), *(RUNS / run_name).glob("weights-*.pt")]
    return sorted({int(p.stem.split("-")[1]) for p in paths})


def likelihoods(run_name: str, step: int) -> dict | None:
    """A step's {"8k", "32k", "solution"}: its evaluation's, else its likelihood record's, else None."""
    for path in (evaluation_path(run_name, step), likelihood_path(run_name, step)):
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            return {key: record[key] for key in ("8k", "32k", "solution")}
    return None


def score(tokenizer: Tokenizer, benchmarks: dict, *, run_name: str, step: int) -> None:
    """Write likelihood_path(run_name, step): likelihood of the step's saved model on the GPU, with an evaluation's
    "provenance" (source digest taken at the start, environment, start and end times)."""
    started, source = time.time(), source_digest()
    model = load_model(run_name, step, device=torch.device("cuda"))
    results = likelihood(model, tokenizer, benchmarks, device=torch.device("cuda"))
    del model
    torch._dynamo.reset()  # compiled graphs and their guards' references
    gc.collect()
    torch.cuda.empty_cache()
    provenance = {"source_sha256": source, "environment": json.loads(environment()), "started": started,
                  "finished": time.time()}
    write_atomically(likelihood_path(run_name, step), json.dumps(
        {"run_name": run_name, "step": step} | results | {"provenance": provenance}, allow_nan=False).encode())


def groups(stage: Stage, weights: dict[str, float]) -> dict[str, dict[str, float]]:
    """A val record's corpus weights per group: each of the stage's groups, "code" (Stage.code) and "all"."""
    members = {name: group.sources for name, group in stage.groups.items()} | {"code": stage.code, "all": list(weights)}
    return {name: {c: weights[c] for c in sources if c in weights} for name, sources in members.items()}


def rows_by_measure(record: dict) -> dict[str, tuple[dict, dict[str, dict[str, float]]]]:
    """{measure: (rows by stratum, {group: stratum weights})} of a likelihoods record: "8k" and "32k" (when the model
    is at that context) by the stage's groups at the record's weights; "solution" per benchmark (a cluster per
    problem) and "solutions" pooled over metrics.solution_strata."""
    out = {ctx: (record[ctx]["corpora"], groups(stage, record[ctx]["weights"]))
           for ctx, stage in STAGES.items() if record[ctx] is not None}
    benchmarks = {name: {"row_bits": s["row_bits"], "row_bytes": s["row_bytes"],
                         "row_clusters": range(len(s["row_bits"])), "rows": len(s["row_bits"])}
                  for name, s in record["solution"].items()}
    strata = metrics.solution_strata(record["solution"])
    out["solution"] = (benchmarks | strata, {name: {name: 1.0} for name in benchmarks}
                       | {"solutions": {name: s["rows"] for name, s in strata.items()}})
    return out


def summaries(rows: dict, weights: dict[str, dict[str, float]], key: str) -> dict[str, dict]:
    """{group: {key, "se", "interval"}} of metrics.mixture_bpb_summary."""
    out = {}
    for group, w in weights.items():
        s = metrics.mixture_bpb_summary(rows, w)
        out[group] = {key: s["bpb"], "se": s["se"], "interval": s["interval"]}
    return out


def trajectory(run_name: str, steps: list[int], *, reference: int) -> dict:
    """Per step and measure, each group's bits per byte with its standard error ("bpb") and its paired change from the
    reference step ("change"; from the first step holding the measure where the reference lacks it, as a model at 8K
    has no "32k"); per interval between consecutive steps, the paired change and the summed learning rate of the
    updates between them (checkpoint s holds the model after steps 0 … s − 1)."""
    if reference not in steps:
        raise ValueError(f"reference step {reference} is not among {steps}")
    loaded = {step: likelihoods(run_name, step) for step in steps}
    missing = [step for step, record in loaded.items() if record is None]
    if missing:
        raise ValueError(f"steps without likelihoods: {missing}")
    parts = {step: rows_by_measure(record) for step, record in loaded.items() if record is not None}
    names = list(dict.fromkeys(m for p in parts.values() for m in p))
    references = {m: reference if m in parts[reference] else next(s for s in steps if m in parts[s]) for m in names}
    lr = {step: r["learning_rate"] for step, r in metrics.train_steps(metrics.records(run_name)).items()}
    bpb, change, intervals = {}, {}, []
    for step in steps:
        bpb[step] = {m: summaries(rows, weights, "bpb") for m, (rows, weights) in parts[step].items()}
        change[step] = {m: summaries(metrics.paired_rows(parts[references[m]][m][0], rows), weights, "change")
                        for m, (rows, weights) in parts[step].items()}
    for a, b in zip(steps, steps[1:]):
        intervals.append({"from": a, "to": b, "summed_lr": sum(lr[s] for s in range(a, b)),
                          "change": {m: summaries(metrics.paired_rows(parts[a][m][0], rows), weights, "change")
                                     for m, (rows, weights) in parts[b].items() if m in parts[a]}})
    return {"run_name": run_name, "steps": steps, "reference": references, "bpb": bpb, "change": change,
            "intervals": intervals}


def main(run_name: str, reference: int, first: int, out: Path) -> None:
    """Score the run's saved models from step `first` on that have no likelihoods yet, then write their trajectory to
    out (JSON)."""
    steps = [step for step in saved_steps(run_name) if step >= first]
    todo = [step for step in steps if likelihoods(run_name, step) is None]
    if todo:
        tokenizer, benchmarks = get_tokenizer(), tasks()
        for step in todo:
            score(tokenizer, benchmarks, run_name=run_name, step=step)
            print(f"{run_name}: scored step {step}", flush=True)
    out.write_text(json.dumps(trajectory(run_name, steps, reference=reference), allow_nan=False) + "\n",
                   encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]))
