"""
An experiment's records once its sweep has ended (Hydra callback): <results>/<experiment>.jsonl and -curves.jsonl, in
results/posttrain/ for runs from a base model's checkpoint (config.base), else in results/pretrain/. A run's curves
alone, at any time: python -m ablations.records <run_name> <experiment> [field[:n] ...] (`curves`).
"""
import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf

import metrics
from eval import suite
from paths import POSTTRAIN_RESULTS, PRETRAIN_RESULTS
from run import EVALUATED, RunConfig, plan_of, run_name


def results_dir(config: RunConfig) -> Path:
    """The directory of a run's experiment records: post-training (a run from a base checkpoint) or pretraining."""
    return POSTTRAIN_RESULTS if config.base is not None else PRETRAIN_RESULTS


def curves(experiment: str, name: str, log: list[dict], fields: dict[str, int] | None = None) -> dict:
    """A run's line of <results>/<experiment>-curves.jsonl: its train records as columns (metrics.curves), its stage
    switches (each `stage` record's step, seq_len and tokens) and its validations (each `val` record's step, loss and
    labels); a resumed attempt's record of a step supersedes earlier ones. With `fields` ({field: n}): the step and
    only those columns, each at the steps divisible by its n (None elsewhere)."""
    columns = metrics.curves(log)
    if fields is not None:
        if missing := set(fields) - set(columns):
            raise ValueError(f"no train record of {name} has {sorted(missing)}")
        steps = columns["step"]
        columns = {"step": steps} | {
            k: [v if s % fields[k] == 0 else None for s, v in zip(steps, column, strict=True)]
            for k, column in columns.items() if k in fields and k != "step"}
    stages = {r["step"]: {"step": r["step"], "seq_len": r["stage"]["seq_len"], "tokens": r["stage"]["tokens"]}
              for r in log if r["kind"] == "stage"}
    validations = {r["step"]: {"step": r["step"], "loss": r["loss"], "labels": r["labels"]}
                   for r in log if r["kind"] == "val"}
    return {"experiment": experiment, "run_name": name, "stages": [stages[s] for s in sorted(stages)],
            "validations": [validations[s] for s in sorted(validations)]} | columns


def export_curves(name: str, experiment: str, fields: dict[str, int] | None = None) -> Path:
    """Run `name`'s curves (`curves`) as <results>/<experiment>-curves.jsonl, routed by its first `run` record's
    config (`results_dir`); without `fields`, a finished run's file equals the one its sweep's callback writes."""
    log = metrics.records(name)
    started = next(r for r in log if r["kind"] == "run")
    path = results_dir(RunConfig(**started["config"])) / f"{experiment}-curves.jsonl"
    path.write_text(json.dumps(curves(experiment, name, log, fields), allow_nan=False) + "\n", encoding="utf-8")
    return path


def record(config: RunConfig, experiment: str, log: list[dict]) -> dict:
    """A run's line of <results>/<experiment>.jsonl; its measures from its final checkpoint's evaluation (none with
    generate "none": its log's val records hold what it measured)."""
    name, run = run_name(config), plan_of(config)
    stopped = metrics.diverged(log)
    # the first attempt's run record: a resumed attempt's flops_per_token is at its own stage's context
    started = next(r for r in log if r["kind"] == "run")
    peaks = [r for r in log if r["kind"] in ("stage", "end")]
    measured = errors = offset = efficiency = None
    if stopped is None:
        efficiency = {"num_parameters": started["num_parameters"], "flops_per_token": started["flops_per_token"],
                      "model_flops_per_token": started["model_flops_per_token"],
                      "peak_memory_bytes": max(r["peak_memory_bytes"] for r in peaks),
                      "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in peaks)}
    if stopped is None and config.generate in EVALUATED:
        evaluation = json.loads(suite.evaluation_path(name, run.max_steps).read_text(encoding="utf-8"))
        measured = metrics.measures(evaluation, metrics.train_steps(log), repo_level_start=run.repo_level_start)
        errors, offset = metrics.measure_errors(evaluation), evaluation["32k"]["offset_bpb"]["code"]
    return {"experiment": experiment} | suite.record(name, log) | {
        "config": dataclasses.asdict(config), "measures": measured, "errors": errors, "offset_bpb": offset,
        "efficiency": efficiency, "diverged_at": stopped}


class Records(Callback):
    """The sweep's records (module docstring)."""

    def __init__(self) -> None:
        self.configs: list[RunConfig] = []

    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
        """Every job's RunConfig, a skipped run's included."""
        self.configs.append(OmegaConf.to_object(job_return.cfg))

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        # the sweep's config (run_job restores HydraConfig to its pre-job state, unset in a multirun)
        experiment = config.hydra.runtime.choices.experiment
        logs = [metrics.records(run_name(c)) for c in self.configs]
        lines = [record(c, experiment, log) for c, log in zip(self.configs, logs, strict=True)]
        (directory,) = {results_dir(c) for c in self.configs}
        (directory / f"{experiment}.jsonl").write_text(
            "".join(json.dumps(line, allow_nan=False) + "\n" for line in lines), encoding="utf-8")
        (directory / f"{experiment}-curves.jsonl").write_text(
            "".join(json.dumps(curves(experiment, run_name(c), log), allow_nan=False) + "\n"
                    for c, log in zip(self.configs, logs, strict=True)), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name")
    parser.add_argument("experiment")
    parser.add_argument("fields", nargs="*", metavar="field[:n]",
                        help="the train fields to keep, each at the steps divisible by n (default 1); default: all")
    args = parser.parse_args()
    fields = {f.partition(":")[0]: int(f.partition(":")[2] or 1) for f in args.fields}
    print(export_curves(args.run_name, args.experiment, fields or None))
