"""
The mechanism the ablation experiments (ablations/lr.py, tokenizer.py) share. An experiment is production at each of
its settings (depth, tokens per scaling parameter, seeds) and one arm per alternative, the arms differing from
production only in their Run fields, each on the same seeds. A run trains production's GPT by
scripts/base_train.train and is scored by eval/suite.py evaluate (its likelihood metrics and parse measure), the
production code paths; its plan (batch, steps, peak lr) is the production model's at its depth and ratio, so every arm
matches in tokens and hyperparameters.

Runs train one at a time (one d8 run saturates the GPU), each in its own process (torch.compile's caches and limits are
per process) launched with its fields, its output, a failure's traceback included, appended to LOGS/<tag>.log. A
rerun experiment resumes: a run whose evaluation exists is not trained again, nor is one that diverged (its
train/metrics.py failure record: a non-finite loss or gradient norm, an outcome of a lr sweep's high end,
arXiv 2309.14322); any other failure stops the experiment, and a stopped run resumes from its repo_start checkpoint
or restarts (train/metrics.py attempts keep both traceable). Then each alternative is compared with production over
the runs that did not diverge; the experiment writes every tracked metric of every run, RESULTS/<experiment>.jsonl (a
record per run: eval/suite.py record, setting, fields, the compared measures, efficiency, the step it diverged at)
and RESULTS/<experiment>-curves.jsonl (per run, train/metrics.py curves), and RESULTS/<experiment>.json (per setting,
each alternative's fields, its evaluated runs and, per measure, its difference from production, p and confidence
interval; null where one run per arm leaves them undefined), the data of the paper's decisions and figures.

    python -m ablations.<experiment>       (every missing run, each in a child process, then the comparison)
    python -m ablations.harness '<fields>'  (one run: harness.Run's fields as JSON)
"""
import dataclasses
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from scipy.stats import ttest_ind

from nanochat.config import FILE_STAGE, REPO_STAGE, run_tag
from nanochat.data.loader import mixture_weights
from nanochat.data.tokenizer import get_tokenizer
from nanochat.eval import bpb, suite
from nanochat.model.gpt import GPT, gpt_config
from nanochat.paths import BASE_EVAL, LOGS, RESULTS
from nanochat.train import metrics
from nanochat.train.checkpoint import checkpoint_name
from nanochat.train.schedule import RunPlan, plan
from scripts.base_train import train

# The conventional significance level ("Statistical Methods for Research Workers" (1925), §20): the confidence
# intervals' 1 - ALPHA
ALPHA = 0.05


@dataclass(frozen=True, kw_only=True)
class Run:
    """A run: production except these fields."""
    n_layer: int
    ratio: float         # tokens per scaling parameter
    lr_scale: float      # times the plan's peak lr (schedule.plan)
    tokenizer: str       # data/tokenizer.get_tokenizer's: "production" or a tokenizer ablation arm (its own data)
    seed: int            # init and data order


def plan_of(run: Run) -> RunPlan:
    """The production model's plan at the run's depth and ratio (schedule.plan), at the run's peak lr."""
    with torch.device("meta"):
        production_plan = plan(model=GPT(gpt_config(depth=run.n_layer, stage=FILE_STAGE)), ratio=run.ratio)
    return dataclasses.replace(production_plan, lr=run.lr_scale * production_plan.lr)


def tag_of(run: Run) -> str:
    """config.run_tag over the run's fields and its peak lr (so a change of the plan's lr, an adopted lr.py result, is
    a new run)."""
    return run_tag(depth=run.n_layer, ratio=run.ratio, seed=run.seed,
                   variant=dataclasses.asdict(run) | {"lr": plan_of(run).lr})


def evaluation_path(run: Run) -> Path:
    """BASE_EVAL/<checkpoint_name>.json of the run's final checkpoint, its likelihood evaluation."""
    return BASE_EVAL / f"{checkpoint_name(tag_of(run), plan_of(run).num_iterations)}.json"


def execute(run: Run) -> None:
    """Train and evaluate one run: its metrics as production's (train/metrics.py), its final checkpoint's evaluation
    as production's (eval/suite.py evaluate) without the generation benchmarks, with the parse measure."""
    tag, run_plan = tag_of(run), plan_of(run)
    steps = run_plan.num_iterations
    with torch.device("meta"):
        model = GPT(gpt_config(depth=run.n_layer, stage=FILE_STAGE))
    model = train(model=model, run=run_plan, tokenizer_name=run.tokenizer, variant=dataclasses.asdict(run),
                  seed=run.seed, checkpoints=frozenset({steps}), tag=tag)
    suite.evaluate(model, get_tokenizer(run.tokenizer), name=checkpoint_name(tag, steps), step=steps, generate="parse")


def evaluation(run: Run) -> dict[str, float]:
    """A run's compared measures (bits per byte; shares; tokens per second) from its evaluation and its per-step
    records: code, text and math, the 8K validation bits per byte of each group at its corpus weights (bpb.val_bits
    "bpb"); solutions, the bits per byte of the HumanEval+, MBPP+ (EvalPlus, arXiv 2305.01210) and HumanEval-X C++
    (arXiv 2303.17568) reference solutions given their prompts, pooled (bpb.solution_bits); repo_32k, the 32K repository
    bits per byte of the code group, and repo_beyond_8k, the same over the targets beyond the file-level context
    (bpb.CONTEXTS' last bucket); unparsed, indentation and syntax, the shares of HumanEval+ and MBPP+ problems, pooled,
    whose healed greedy completion does not compile, by an IndentationError and by another SyntaxError
    (code_eval.parse_generation), and greedy, their base pass@1, pooled; tok_per_s, the median over the file-level
    stage's steps of the step's tokens per second."""
    tag, repo_start = tag_of(run), plan_of(run).repo_start
    result = json.loads(evaluation_path(run).read_text(encoding="utf-8"))
    rates = [r["tok_per_s"] for step, r in metrics.train_steps(tag).items() if step < repo_start]
    parsed = result["parse"].values()
    problems = sum(len(p["outcomes"]) for p in parsed)

    def share(key: str) -> float:
        return sum(p[key] * len(p["outcomes"]) for p in parsed) / problems

    return {"code": result["8k"]["bpb"]["code"],
            "solutions": bpb.bpb(*result["solution"].values()),
            "repo_32k": result["32k"]["bpb"]["code"],
            "repo_beyond_8k": bpb.mixture_bpb(result["32k"]["corpora"], mixture_weights(
                REPO_STAGE.mixture, get_tokenizer(run.tokenizer), REPO_STAGE.seq_len), buckets=slice(-1, None)),
            "text": result["8k"]["bpb"]["text"], "math": result["8k"]["bpb"]["math"],
            "unparsed": 1 - share("parses"), "indentation": share("indentation"), "syntax": share("syntax"),
            "greedy": share("base pass@1"),
            "tok_per_s": statistics.median(rates)}


def compare(alternative: list[dict[str, float]],
            baseline: list[dict[str, float]]) -> dict[str, dict[str, float | None]]:
    """{statistic: {measure: value}} over two arms' runs (each a measure dict): the alternative's mean minus the
    baseline's, the two-sided pooled t-test's p and the difference's 1 - ALPHA confidence interval (scipy ttest_ind);
    None where one run per arm leaves them undefined."""
    stats = {"difference": {}, "p": {}, "ci_low": {}, "ci_high": {}}
    for m in baseline[0]:
        a, b = [r[m] for r in alternative], [r[m] for r in baseline]
        test = ttest_ind(a, b)
        interval = test.confidence_interval(confidence_level=1 - ALPHA)
        stats["difference"][m] = statistics.fmean(a) - statistics.fmean(b)
        for k, v in (("p", test.pvalue), ("ci_low", interval.low), ("ci_high", interval.high)):
            stats[k][m] = None if math.isnan(v) else float(v)
    return stats


def diverged(tag: str) -> int | None:
    """The step at which run `tag` stopped on a non-finite loss or gradient norm (its failure record), or None."""
    failures = [r["step"] for r in metrics.records(tag) if r["kind"] == "failure"]
    return failures[-1] if failures else None


def main(*, settings: tuple[tuple[int, float, range, dict[str, dict]], ...], production: dict) -> None:
    """Run the calling experiment (python -m ablations.<experiment>): at each setting (depth, tokens per scaling
    parameter, seeds, alternatives), production and an arm per alternative {name: Run fields}; then compare each
    alternative with production at each setting. `production`: the Run fields in which the experiment's production arm,
    and so its every arm, departs from production (the tokenizer ablation's: production's vocabulary on its
    documents)."""
    experiment = sys.modules["__main__"].__spec__.name.rsplit(".", 1)[1]
    cells = []
    for depth, ratio, seeds, alternatives in settings:
        # production's settings (scripts/base_train.py main, nanochat/model/gpt.py) at this depth and ratio
        baseline = [dataclasses.replace(Run(n_layer=depth, ratio=ratio, lr_scale=1.0, tokenizer="production",
                                            seed=seed), **production) for seed in seeds]
        cells.append({"production": baseline} | {name: [dataclasses.replace(run, **changes) for run in baseline]
                                                 for name, changes in alternatives.items()})
    tags = {run: tag_of(run) for arms in cells for arm in arms.values() for run in arm}
    for run, tag in tags.items():
        if evaluation_path(run).exists() or diverged(tag) is not None:
            continue
        with (LOGS / f"{tag}.log").open("a", encoding="utf-8") as log:
            child = subprocess.run([sys.executable, "-m", "ablations.harness", json.dumps(dataclasses.asdict(run))],
                                   stdout=log, stderr=subprocess.STDOUT)
        if child.returncode and diverged(tag) is None:
            raise RuntimeError(f"run {tag} failed: {LOGS / f'{tag}.log'}")
    summaries = []
    with ((RESULTS / f"{experiment}.jsonl").open("w", encoding="utf-8") as f,
          (RESULTS / f"{experiment}-curves.jsonl").open("w", encoding="utf-8") as curve_lines):
        for (depth, ratio, seeds, alternatives), arms in zip(settings, cells, strict=True):
            measures = {name: {tags[run]: evaluation(run) for run in arm if diverged(tags[run]) is None}
                        for name, arm in arms.items()}
            for name, arm in arms.items():
                for run in arm:
                    tag, records = tags[run], metrics.records(tags[run])
                    started = [r for r in records if r["kind"] == "run"][-1]
                    ended = [r for r in records if r["kind"] == "end"]
                    identity = {"experiment": experiment, "arm": name, "tag": tag}
                    f.write(json.dumps(identity | suite.record(tag) | {
                        "setting": {"depth": depth, "tokens_per_param": ratio, "seeds": len(seeds)},
                        "fields": dataclasses.asdict(run),
                        "measures": measures[name].get(tag),
                        # model size and cost (train/metrics.py run and end records)
                        "efficiency": ({"params": started["params"], "flops_per_token": started["flops_per_token"],
                                        "peak_memory_bytes": ended[-1]["peak_memory_bytes"]}
                                       if tag in measures[name] else None),
                        "diverged_at": diverged(tag)}, allow_nan=False) + "\n")
                    curve_lines.write(json.dumps(identity | metrics.curves(tag), allow_nan=False) + "\n")
            summaries.append({"depth": depth, "tokens_per_param": ratio, "seeds": len(seeds),
                              "production_runs": len(measures["production"]),
                              "alternatives": {name: {"fields": changes, "runs": len(measures[name])}
                                               | (compare([*measures[name].values()],
                                                          [*measures["production"].values()])
                                                  if measures[name] and measures["production"] else {})
                                               for name, changes in alternatives.items()}})
    (RESULTS / f"{experiment}.json").write_text(json.dumps({"settings": summaries}, allow_nan=False),
                                                encoding="utf-8")


if __name__ == "__main__":
    execute(Run(**json.loads(sys.argv[1])))
