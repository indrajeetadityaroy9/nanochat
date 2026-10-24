"""
The mechanism the ablation experiments (ablations/lr.py, ablations/tokenizer.py) share. An experiment is the baseline
at each of its settings (Setting: depth, tokens per scaling parameter, seeds) and one arm per alternative, the arms
differing from the baseline only in their Run fields, each on the same seeds. A run trains the base model's GPT at its
depth by train/loop.py train and is scored by eval/suite.py evaluate (its likelihood metrics and parse measure), the
pre-training code paths; its plan (batch, steps, peak lr) is the base model's at its depth and ratio (base_plan), so
arms match in hyperparameters and, unless an arm sets ratio (the tokenizer experiment's equal bytes), in tokens.

Runs train one at a time (one d8 run saturates the GPU), each in its own process (torch.compile's caches and limits are
per process) launched with its fields, its output, a failure's traceback included, appended to LOGS/<tag>.log. A
rerun experiment resumes: a run whose evaluation exists is not trained again, nor is one that diverged (its
train/metrics.py failure record: a non-finite loss or gradient norm, an outcome of a lr sweep's high end,
arXiv 2309.14322 §3.1); any other failure stops the experiment, and a stopped run trains again from its start (its one
checkpoint is its final model; train/metrics.py attempts keep both traceable). The experiment then writes every
tracked metric of every run, RESULTS/<experiment>.jsonl (a record per run: eval/suite.py record, setting, fields, the
compared measures, efficiency, the step it diverged at) and RESULTS/<experiment>-curves.jsonl (per run,
train/metrics.py curves), and RESULTS/<experiment>.json (per setting, each alternative's fields and its evaluated
runs), to which the experiment adds its own pre-registered analysis (lr.py analysis, tokenizer.py runs' decision).

    python -m ablations.<experiment>       (every missing run, each in a child process, then the records: main)
    python -m ablations.harness '<fields>'  (one run: harness.Run's fields as JSON)
"""
import dataclasses
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from nanochat.config import run_tag
from nanochat.corpus.mixture import FILE_STAGE, REPO_STAGE
from nanochat.corpus.tokenizer import get_tokenizer
from nanochat.eval import bpb, suite
from nanochat.model.gpt import GPT, gpt_config
from nanochat.paths import BASE_EVAL, CORPUS, LOGS, RESULTS
from nanochat.train import metrics
from nanochat.train.checkpoint import checkpoint_name
from nanochat.train.loop import train
from nanochat.train.schedule import RunPlan, plan


@dataclass(frozen=True, kw_only=True)
class Run:
    """A run: the base model's recipe except these fields."""
    n_layer: int
    ratio: float         # tokens per scaling parameter
    lr_scale: float      # times the plan's peak lr (schedule.plan)
    tokenizer: str       # its vocabulary's directory under CORPUS: "tokenizer" (the corpus's) or an ablation arm's
    seed: int            # init and data order

    @property
    def vocabulary(self) -> Path:
        return CORPUS / self.tokenizer


@dataclass(frozen=True, kw_only=True)
class Setting:
    """An experiment's setting: the base model at `depth` trained at `ratio` tokens per scaling parameter on each of
    `seeds`, and its alternatives {arm name: the Run fields in which the arm departs from the baseline}."""
    depth: int
    ratio: float
    seeds: range
    alternatives: dict[str, dict]


def base_plan(*, depth: int, ratio: float) -> RunPlan:
    """The base model's plan at a depth and tokens per scaling parameter (schedule.plan)."""
    with torch.device("meta"):
        return plan(model=GPT(gpt_config(depth=depth, stage=FILE_STAGE)), ratio=ratio)


def plan_of(run: Run) -> RunPlan:
    """The base model's plan at the run's depth and ratio, at the run's peak lr."""
    base = base_plan(depth=run.n_layer, ratio=run.ratio)
    return dataclasses.replace(base, lr=run.lr_scale * base.lr)


def tag_of(run: Run) -> str:
    """config.run_tag over the run's fields (its ratio and lr_scale as floats, so equal Runs share a tag), its
    tokenizer's data key (so a rebuilt vocabulary is a new run) and its peak lr (so a change of the plan's lr, an
    adopted lr.py result, is a new run)."""
    return run_tag(depth=run.n_layer, ratio=run.ratio, seed=run.seed,
                   variant=dataclasses.asdict(run) | {"ratio": float(run.ratio), "lr_scale": float(run.lr_scale),
                                                      "lr": plan_of(run).lr,
                                                      "tokenizer_data": get_tokenizer(run.vocabulary).data_key})


def evaluation_path(run: Run) -> Path:
    """BASE_EVAL/<checkpoint_name>.json of the run's final checkpoint, its likelihood evaluation."""
    return BASE_EVAL / f"{checkpoint_name(tag_of(run), plan_of(run).num_iterations)}.json"


def execute(run: Run) -> None:
    """Train and evaluate one run: its metrics as the base model's (train/metrics.py), its final checkpoint's
    evaluation as the base model's (eval/suite.py evaluate) without the generation benchmarks, with the parse
    measure."""
    tag, run_plan = tag_of(run), plan_of(run)
    steps = run_plan.num_iterations
    with torch.device("meta"):
        model = GPT(gpt_config(depth=run.n_layer, stage=FILE_STAGE))
    model = train(model=model, run=run_plan, vocabulary=run.vocabulary, variant=dataclasses.asdict(run),
                  seed=run.seed, checkpoints=frozenset({steps}), tag=tag)
    suite.evaluate(model, get_tokenizer(run.vocabulary), name=checkpoint_name(tag, steps), step=steps,
                   generate="parse")


def evaluation(run: Run) -> dict[str, float]:
    """A run's compared measures (bits per byte; shares; tokens per second) from its evaluation and its per-step
    records: code, text and math, the 8K validation bits per byte of each group at its corpus weights (bpb.val_bits
    "bpb"); solutions, the bits per byte of the HumanEval+, MBPP+ (EvalPlus, arXiv 2305.01210) and HumanEval-X C++
    (arXiv 2303.17568) reference solutions given their prompts, pooled (bpb.solution_bits); repo_32k, the 32K repository
    bits per byte of the code group, and repo_beyond_8k, the same over the targets beyond the file-level context
    (bpb.CONTEXTS' last bucket) at the evaluation's code corpus weights (bpb.val_bits "weights"); unparsed,
    indentation and syntax, the shares of HumanEval+ and MBPP+ problems, pooled, whose healed greedy completion does not
    compile, by an IndentationError and by another SyntaxError (code_eval.parse_generation), and greedy, their base
    pass@1, pooled; tok_per_s, the median over the file-level stage's steps of the step's tokens per second."""
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
            "repo_beyond_8k": bpb.mixture_bpb(
                result["32k"]["corpora"],
                {c: w for c, w in result["32k"]["weights"].items() if c in REPO_STAGE.code}, buckets=slice(-1, None)),
            "text": result["8k"]["bpb"]["text"], "math": result["8k"]["bpb"]["math"],
            "unparsed": 1 - share("parses"), "indentation": share("indentation"), "syntax": share("syntax"),
            "greedy": share("base pass@1"),
            "tok_per_s": statistics.median(rates)}


def diverged(tag: str) -> int | None:
    """The step at which run `tag` stopped on a non-finite loss or gradient norm (its failure record), or None."""
    failures = [r["step"] for r in metrics.records(tag) if r["kind"] == "failure"]
    return failures[-1] if failures else None


def main(*, experiment: str, settings: list[Setting], baseline: dict) -> None:
    """Run an experiment (its records RESULTS/<experiment>.json, .jsonl and -curves.jsonl): at each setting, the
    baseline and an arm per alternative, then every run's record. `baseline`: the Run fields in which the experiment's
    baseline, and so its every arm, departs from the base model's recipe (the tokenizer ablation's: its sa-bpe
    vocabulary)."""
    cells = []
    for setting in settings:
        # the base model's recipe (nanochat/train/loop.py, nanochat/model/gpt.py) at this depth and ratio
        runs = [dataclasses.replace(Run(n_layer=setting.depth, ratio=setting.ratio, lr_scale=1.0,
                                        tokenizer="tokenizer", seed=seed), **baseline) for seed in setting.seeds]
        cells.append({"baseline": runs} | {name: [dataclasses.replace(run, **changes) for run in runs]
                                           for name, changes in setting.alternatives.items()})
    tags = {run: tag_of(run) for arms in cells for arm in arms.values() for run in arm}
    for run, tag in tags.items():
        if evaluation_path(run).exists() or diverged(tag) is not None:
            continue
        with (LOGS / f"{tag}.log").open("a", encoding="utf-8") as log:
            child = subprocess.run([sys.executable, "-m", "ablations.harness", json.dumps(dataclasses.asdict(run))],
                                   stdout=log, stderr=subprocess.STDOUT)
        if child.returncode and diverged(tag) is None:
            raise RuntimeError(f"run {tag} failed: {LOGS / f'{tag}.log'}")
    stopped = {tag: diverged(tag) for tag in tags.values()}
    summaries = []
    with ((RESULTS / f"{experiment}.jsonl").open("w", encoding="utf-8") as f,
          (RESULTS / f"{experiment}-curves.jsonl").open("w", encoding="utf-8") as curve_lines):
        for setting, arms in zip(settings, cells, strict=True):
            measures = {name: {tags[run]: evaluation(run) for run in arm if stopped[tags[run]] is None}
                        for name, arm in arms.items()}
            for name, arm in arms.items():
                for run in arm:
                    tag = tags[run]
                    records = metrics.records(tag)
                    started = [r for r in records if r["kind"] == "run"][-1]
                    ended = [r for r in records if r["kind"] == "end"]
                    identity = {"experiment": experiment, "arm": name, "tag": tag}
                    f.write(json.dumps(identity | suite.record(tag) | {
                        "setting": {"depth": setting.depth, "tokens_per_param": setting.ratio,
                                    "seeds": len(setting.seeds)},
                        "fields": dataclasses.asdict(run),
                        "measures": measures[name].get(tag),
                        # model size and cost (train/metrics.py run and end records)
                        "efficiency": ({"params": started["params"], "flops_per_token": started["flops_per_token"],
                                        "peak_memory_bytes": ended[-1]["peak_memory_bytes"]}
                                       if tag in measures[name] else None),
                        "diverged_at": stopped[tag]}, allow_nan=False) + "\n")
                    curve_lines.write(json.dumps(identity | metrics.curves(tag), allow_nan=False) + "\n")
            summaries.append({"depth": setting.depth, "tokens_per_param": setting.ratio, "seeds": len(setting.seeds),
                              "baseline_runs": len(measures["baseline"]),
                              "alternatives": {name: {"fields": changes, "runs": len(measures[name])}
                                               for name, changes in setting.alternatives.items()}})
    (RESULTS / f"{experiment}.json").write_text(json.dumps({"settings": summaries}, allow_nan=False),
                                                encoding="utf-8")


if __name__ == "__main__":
    execute(Run(**json.loads(sys.argv[1])))
