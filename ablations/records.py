"""
An experiment's records (the Records callback of conf/experiment/<name>.yaml; run.py) once every job of its
sweep has ended: RESULTS/<experiment>.jsonl, a record per run (eval/suite.py record, the run's config, its compared
measures (metrics.measures) and their error bars (metrics.measure_errors), the 32K code bits per byte by byte offset
(eval/bpb.py "offset_bpb"), efficiency, the step it diverged at), RESULTS/<experiment>-curves.jsonl (per run,
metrics.curves) and, with an analysis (the experiment's pre-registered decision, a function of the .jsonl's records),
RESULTS/<experiment>.json = {"analysis": analysis(lines)}. Each run's log (metrics.records) is read once.
"""
import dataclasses
import json
from typing import Any

from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

import metrics
from eval import suite
from paths import RESULTS
from run import RunConfig, plan_of, run_name


def record(config: RunConfig, experiment: str, log: list[dict]) -> dict:
    """A run's line of RESULTS/<experiment>.jsonl from its run log; its measures from its final checkpoint's
    evaluation."""
    name, run = run_name(config), plan_of(config)
    stopped = metrics.diverged(log)
    started = [r for r in log if r["kind"] == "run"][-1]
    # the stages' peaks: the stage switch's (the file-level stage) and the end's (the last stage)
    peaks = [r for r in log if r["kind"] in ("stage", "end")]
    measured = errors = offset = efficiency = None
    if stopped is None:
        evaluation = json.loads(suite.evaluation_path(name, run.max_steps).read_text(encoding="utf-8"))
        measured = metrics.measures(evaluation, metrics.train_steps(log), repo_level_start=run.repo_level_start)
        errors, offset = metrics.measure_errors(evaluation), evaluation["32k"]["offset_bpb"]["code"]
        # model size and cost (metrics.py run, stage and end records)
        efficiency = {"num_parameters": started["num_parameters"], "flops_per_token": started["flops_per_token"],
                      "model_flops_per_token": started["model_flops_per_token"],
                      "peak_memory_bytes": max(r["peak_memory_bytes"] for r in peaks),
                      "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in peaks)}
    return {"experiment": experiment} | suite.record(name, log) | {
        "config": dataclasses.asdict(config), "measures": measured, "errors": errors, "offset_bpb": offset,
        "efficiency": efficiency, "diverged_at": stopped}


class Records(Callback):
    """The sweep's records (module docstring); `analysis`: the dotted path of the experiment's analysis, or None."""

    def __init__(self, analysis: str | None) -> None:
        self.analysis = analysis
        self.configs: list[RunConfig] = []

    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
        """Every job's RunConfig, a skipped run's included."""
        self.configs.append(OmegaConf.to_object(job_return.cfg))

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        # the sweep's config (run_job restores HydraConfig to its pre-job state, unset in a multirun)
        experiment = config.hydra.runtime.choices.experiment
        logs = [metrics.records(run_name(c)) for c in self.configs]
        lines = [record(c, experiment, log) for c, log in zip(self.configs, logs, strict=True)]
        (RESULTS / f"{experiment}.jsonl").write_text(
            "".join(json.dumps(line, allow_nan=False) + "\n" for line in lines), encoding="utf-8")
        (RESULTS / f"{experiment}-curves.jsonl").write_text(
            "".join(json.dumps({"experiment": experiment, "run_name": run_name(c)} | metrics.curves(log),
                               allow_nan=False) + "\n" for c, log in zip(self.configs, logs, strict=True)),
            encoding="utf-8")
        if self.analysis is not None:
            (RESULTS / f"{experiment}.json").write_text(
                json.dumps({"analysis": get_method(self.analysis)(lines)}, allow_nan=False), encoding="utf-8")
