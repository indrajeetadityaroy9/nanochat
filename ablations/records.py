"""
An experiment's records (the Records callback of conf/experiment/<name>.yaml; run.py) once every job of its
sweep has ended: RESULTS/<experiment>.jsonl, a record per run (eval/suite.py record, the run's config, its compared
measures, efficiency, the step it diverged at), RESULTS/<experiment>-curves.jsonl (per run, train/metrics.py curves)
and, with an analysis (the experiment's pre-registered decision, a function of the .jsonl's records),
RESULTS/<experiment>.json = {"analysis": analysis(lines)}.
"""
import dataclasses
import json
import statistics
from typing import Any

from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

from corpus.mixture import REPO_STAGE
from eval import bpb, suite
from paths import RESULTS
from run import RunConfig, key, plan_of
from train import metrics


def measures(config: RunConfig) -> dict[str, float]:
    """A run's compared measures (bits per byte; shares; tokens per second) from its final checkpoint's evaluation and
    its per-step records: code, text and math, the 8K validation bits per byte of each group at its corpus weights
    (bpb.val_bits "bpb"); solutions, the bits per byte of the HumanEval+, MBPP+ (EvalPlus, arXiv 2305.01210) and
    HumanEval-X C++ (arXiv 2303.17568) reference solutions given their prompts, pooled (bpb.solution_bits); repo_32k,
    the 32K repository bits per byte of the code group, and repo_beyond_8k, the same over the targets beyond the
    file-level context (bpb.CONTEXTS' last bucket) at the evaluation's code corpus weights (bpb.val_bits "weights");
    unparsed, indentation and syntax, the shares of HumanEval+ and MBPP+ problems, pooled, whose healed greedy
    completion does not compile, by an IndentationError and by another SyntaxError (code_eval.parse_generation), and
    greedy, their base pass@1, pooled; tok_per_s, the median over the file-level stage's steps of the step's tokens per
    second."""
    tag, run = key(config), plan_of(config)
    result = json.loads(suite.evaluation_path(tag, run.num_iterations).read_text(encoding="utf-8"))
    rates = [r["tok_per_s"] for step, r in metrics.train_steps(tag).items() if step < run.repo_start]
    parsed = result["parse"].values()
    problems = sum(len(p["outcomes"]) for p in parsed)

    def share(name: str) -> float:
        return sum(p[name] * len(p["outcomes"]) for p in parsed) / problems

    return {"code": result["8k"]["bpb"]["code"],
            "solutions": bpb.bpb(*result["solution"].values()),
            "repo_32k": result["32k"]["bpb"]["code"],
            "repo_beyond_8k": bpb.mixture_bpb(
                result["32k"]["corpora"],
                {c: w for c, w in result["32k"]["weights"].items() if c in REPO_STAGE.code}, buckets=slice(-1, None)),
            "text": result["8k"]["bpb"]["text"], "math": result["8k"]["bpb"]["math"],
            "unparsed": 1 - share("parses"), "indentation": share("indentation"), "syntax": share("syntax"),
            "greedy": share("base pass@1"),
            "tok_per_s": statistics.median(rates)}


def record(config: RunConfig, experiment: str) -> dict:
    """A run's line of RESULTS/<experiment>.jsonl."""
    tag = key(config)
    stopped = metrics.diverged(tag)
    records = metrics.records(tag)
    started = [r for r in records if r["kind"] == "run"][-1]
    ended = [r for r in records if r["kind"] == "end"]
    return {"experiment": experiment} | suite.record(tag) | {
        "config": dataclasses.asdict(config),
        "measures": None if stopped is not None else measures(config),
        # model size and cost (train/metrics.py run and end records)
        "efficiency": None if stopped is not None else {"params": started["params"],
                                                        "flops_per_token": started["flops_per_token"],
                                                        "peak_memory_bytes": ended[-1]["peak_memory_bytes"]},
        "diverged_at": stopped}


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
            "".join(json.dumps({"experiment": experiment, "tag": key(c)} | metrics.curves(key(c)), allow_nan=False)
                    + "\n" for c in self.configs), encoding="utf-8")
        if self.analysis is not None:
            (RESULTS / f"{experiment}.json").write_text(
                json.dumps({"analysis": get_method(self.analysis)(lines)}, allow_nan=False), encoding="utf-8")
