"""
An experiment's records (the Records callback of conf/experiment/<name>.yaml; run.py) once every job of its
sweep has ended: RESULTS/<experiment>.jsonl, a record per run (eval/suite.py record, the run's config, its compared
measures (metrics.measures), efficiency, the step it diverged at), RESULTS/<experiment>-curves.jsonl (per run,
metrics.curves) and, with an analysis (the experiment's pre-registered decision, a function of the .jsonl's records),
RESULTS/<experiment>.json = {"analysis": analysis(lines)}.
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


def record(config: RunConfig, experiment: str) -> dict:
    """A run's line of RESULTS/<experiment>.jsonl; its measures from its final checkpoint's evaluation."""
    name, run = run_name(config), plan_of(config)
    stopped = metrics.diverged(name)
    records = metrics.records(name)
    started = [r for r in records if r["kind"] == "run"][-1]
    ended = [r for r in records if r["kind"] == "end"]
    measured = efficiency = None
    if stopped is None:
        evaluation = json.loads(suite.evaluation_path(name, run.max_steps).read_text(encoding="utf-8"))
        measured = metrics.measures(evaluation, metrics.train_steps(name), repo_level_start=run.repo_level_start)
        # model size and cost (metrics.py run and end records)
        efficiency = {"num_parameters": started["num_parameters"], "flops_per_token": started["flops_per_token"],
                      "peak_memory_bytes": ended[-1]["peak_memory_bytes"]}
    return {"experiment": experiment} | suite.record(name) | {
        "config": dataclasses.asdict(config), "measures": measured, "efficiency": efficiency, "diverged_at": stopped}


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
        lines = [record(c, experiment) for c in self.configs]
        (RESULTS / f"{experiment}.jsonl").write_text(
            "".join(json.dumps(line, allow_nan=False) + "\n" for line in lines), encoding="utf-8")
        (RESULTS / f"{experiment}-curves.jsonl").write_text(
            "".join(json.dumps({"experiment": experiment, "run_name": run_name(c)} | metrics.curves(run_name(c)),
                               allow_nan=False)
                    + "\n" for c in self.configs), encoding="utf-8")
        if self.analysis is not None:
            (RESULTS / f"{experiment}.json").write_text(
                json.dumps({"analysis": get_method(self.analysis)(lines)}, allow_nan=False), encoding="utf-8")
