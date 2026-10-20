"""
The mechanism the ablation experiments (ablations/lr.py, ablations/value_embeds.py; ablations.md) share. An experiment
is production at each of its settings (depth, tokens per scaling parameter, seeds) and one arm per alternative, the arms
differing from production only in their Run fields, each on the same seeds. A run trains by scripts/base_train.train
and is scored by scripts/base_eval.loss_evaluation, the production code paths; its model is ablations/model.py's
Candidate (production's GPT when every structural field is production's); its plan (batch, steps, peak lr) is the
production model's at its depth and ratio, so every arm matches in tokens and hyperparameters.

Runs train one at a time (one d8 run saturates the GPU), each in its own process (torch.compile's caches and limits are
per process) launched with its fields, its output, a failure's traceback included, appended to LOGS/<tag>.log. A run
whose evaluation exists is skipped (arms shared between experiments train once), as is one that diverged (its
train/metrics.py failure record: a non-finite loss or gradient norm, an outcome of a lr sweep's high end, Wortsman et
al., arXiv 2309.14322); any other failure stops the experiment, and a stopped run resumes from its repo_start checkpoint
or restarts (train/metrics.py attempts keep both traceable). Then each alternative is compared with production over
the evaluated runs; the experiment writes every tracked metric of every run, RESULTS/<experiment>.jsonl (a record per
run: setting, fields, its train/metrics.py records but the per-step ones, its scripts/base_eval.py evaluation, the
compared measures, efficiency, the step it diverged at) and RESULTS/<experiment>-curves.jsonl (per run, its per-step
train records as columns), and RESULTS/<experiment>.json (per setting, each alternative's fields, its evaluated runs
and, per measure, its difference from production, p and confidence interval; null where one run per arm leaves them
undefined), for the decisions and figures ablations.md fixes.

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

import pandas as pd
import torch
from scipy.stats import ttest_ind

from ablations.model import Candidate
from nanochat.config import CODE, FILE_STAGE, REPO_STAGE, run_tag
from nanochat.data.loader import mixture_weights
from nanochat.data.tokenizer import get_tokenizer
from nanochat.eval.code_eval import tasks
from nanochat.eval.loss_eval import bpb, mixture_bpb
from nanochat.model.gpt import GPT, gpt_config
from nanochat.paths import BASE_EVAL, LOGS, METRICS, RESULTS
from nanochat.train.metrics import latest, records, train_steps
from nanochat.train.optim import tau_weight_decay
from nanochat.train.schedule import RunPlan, plan
from scripts.base_eval import loss_evaluation, save
from scripts.base_train import train

# The conventional significance level (Fisher, Statistical Methods for Research Workers, 1925, §20): the confidence
# intervals' 1 - ALPHA
ALPHA = 0.05
# Version of RESULTS/<experiment>.jsonl's records
SCHEMA = 1
# The code a run's result depends on: production's (config.CODE) and this mechanism
RUN_CODE = (*CODE, "ablations/harness.py", "ablations/model.py")


@dataclass(frozen=True, kw_only=True)
class Run:
    """A run: production except these fields (the structural one: ablations/model.py)."""
    n_layer: int
    ratio: float         # tokens per scaling parameter
    lr_scale: float      # times the plan's peak lr (schedule.plan)
    value_embeds: str    # "alternating" or "none"
    seed: int            # init and data order


def production(*, depth: int, ratio: float, lr_scale: float, seed: int) -> Run:
    """Production's settings (scripts/base_train.py main, nanochat/model/gpt.py) at this depth and ratio, its peak lr
    `lr_scale` times the plan's."""
    return Run(n_layer=depth, ratio=ratio, lr_scale=lr_scale, value_embeds="alternating", seed=seed)


def model_of(run: Run) -> Candidate:
    """The run's model on the meta device."""
    with torch.device("meta"):
        return Candidate(gpt_config(depth=run.n_layer, stage=FILE_STAGE), value_embeds=run.value_embeds)



def plan_of(run: Run) -> RunPlan:
    """The production model's plan at the run's depth and ratio (schedule.plan), at the run's peak lr."""
    with torch.device("meta"):
        production_plan = plan(model=GPT(gpt_config(depth=run.n_layer, stage=FILE_STAGE)), ratio=run.ratio)
    return dataclasses.replace(production_plan, lr=run.lr_scale * production_plan.lr)

def tag_of(run: Run) -> str:
    """config.run_tag over the run's fields and RUN_CODE."""
    return run_tag(depth=run.n_layer, ratio=run.ratio, seed=run.seed, data_seed=run.seed,
                   variant=dataclasses.asdict(run), code=RUN_CODE)


def execute(run: Run) -> None:
    """Train and evaluate one run: its metrics as production's (train/metrics.py), its evaluation as base_eval's schema
    with loss_evaluation only; tau-law weight decay (production's); checkpoints at repo_start and the end only."""
    tag, run_plan = tag_of(run), plan_of(run)
    steps = run_plan.num_iterations
    model = train(model=model_of(run), run=run_plan,
                  weight_decay=tau_weight_decay(lr=run_plan.lr, steps=steps,
                                                tokens_per_param=run_plan.tokens_per_param),
                  variant=dataclasses.asdict(run), seed=run.seed,
                  data_seed=run.seed, save_every=steps, snapshot_every=steps, tag=tag)
    save(tag, steps, loss_evaluation(model, get_tokenizer(), tasks(), device=torch.device("cuda")) |
         {"codegen": None, "fim": None})


def evaluation(run: Run) -> dict[str, float]:
    """A run's compared measures (bits per byte; tokens per second) from its evaluation and its per-step records: code,
    text and math, the 8K validation bits per byte of each group at production's corpus weights (loss_eval.val_bits
    "bpb"); solutions, the bits per byte of the HumanEval+, MBPP+ and HumanEval-X C++ reference solutions given their
    prompts, pooled (code_eval.solution_bits); repo_32k, the 32K repository bits per byte of the code group, and
    repo_beyond_8k, the same over the targets beyond the file-level context (loss_eval CONTEXTS' last bucket);
    tok_per_s, the median over the file-level stage's steps of the step's tokens per second."""
    tag, repo_start = tag_of(run), plan_of(run).repo_start
    result = json.loads((BASE_EVAL / f"{tag}.json").read_text(encoding="utf-8"))
    solution = result["solution"].values()
    beyond = {c: {"bits": r["bits"][-1:], "bytes": r["bytes"][-1:], "rows": r["rows"]}
              for c, r in result["32k"]["corpora"].items()}
    rates = [r["tok_per_s"] for step, r in train_steps(tag).items() if step < repo_start]
    return {"code": result["8k"]["bpb"]["code"],
            "solutions": bpb({k: [sum(sum(s[k]) for s in solution)] for k in ("bits", "bytes")}),
            "repo_32k": result["32k"]["bpb"]["code"],
            "repo_beyond_8k": mixture_bpb(beyond, mixture_weights(REPO_STAGE.mixture, get_tokenizer(),
                                                                  REPO_STAGE.seq_len)),
            "text": result["8k"]["bpb"]["text"], "math": result["8k"]["bpb"]["math"],
            "tok_per_s": statistics.median(rates)}


def efficiency(run: Run) -> dict[str, int]:
    """A run's model size and cost from its records (train/metrics.py run and end records)."""
    tag = tag_of(run)
    started, ended = latest(tag, "run"), latest(tag, "end")
    return {"params": started["params"], "flops_per_token": started["flops_per_token"],
            "peak_memory_bytes": ended["peak_memory_bytes"]}


def compare(alternative: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """{statistic: {measure (column): value}}: the alternative's mean minus the baseline's, the two-sided pooled
    t-test's p and the difference's 1 - ALPHA confidence interval (scipy ttest_ind); None where one run per arm leaves
    them undefined."""
    stats = {"difference": {}, "p": {}, "ci_low": {}, "ci_high": {}}
    for m in baseline:
        test = ttest_ind(alternative[m], baseline[m])
        interval = test.confidence_interval(confidence_level=1 - ALPHA)
        stats["difference"][m] = float(alternative[m].mean() - baseline[m].mean())
        for k, v in (("p", test.pvalue), ("ci_low", interval.low), ("ci_high", interval.high)):
            stats[k][m] = None if math.isnan(v) else float(v)
    return stats


def diverged(tag: str) -> int | None:
    """The step at which run `tag` stopped on a non-finite loss or gradient norm (its failure record), or None."""
    if not (METRICS / f"{tag}.jsonl").exists():
        return None
    failures = [r["step"] for r in records(tag) if r["kind"] == "failure"]
    return failures[-1] if failures else None


def main(*, settings: tuple[tuple[int, float, range, dict[str, dict]], ...], lr_scale: float) -> None:
    """Run the calling experiment (python -m ablations.<experiment>): at each setting (depth, tokens per scaling
    parameter, seeds, alternatives), production at peak lr `lr_scale` times the plan's and an arm per alternative
    {name: Run fields}; then compare each alternative with production at each setting."""
    experiment = sys.modules["__main__"].__spec__.name.rsplit(".", 1)[1]
    cells = []
    for depth, ratio, seeds, alternatives in settings:
        baseline = [production(depth=depth, ratio=ratio, lr_scale=lr_scale, seed=seed) for seed in seeds]
        cells.append({"production": baseline} | {name: [dataclasses.replace(run, **changes) for run in baseline]
                                                 for name, changes in alternatives.items()})
    for run in (run for arms in cells for arm in arms.values() for run in arm):
        tag = tag_of(run)
        if (BASE_EVAL / f"{tag}.json").exists() or diverged(tag) is not None:
            continue
        with (LOGS / f"{tag}.log").open("a", encoding="utf-8") as log:
            child = subprocess.run([sys.executable, "-m", "ablations.harness", json.dumps(dataclasses.asdict(run))],
                                   stdout=log, stderr=subprocess.STDOUT)
        if child.returncode and diverged(tag) is None:
            raise RuntimeError(f"run {tag} failed: {LOGS / f'{tag}.log'}")
    summaries = []
    with ((RESULTS / f"{experiment}.jsonl").open("w", encoding="utf-8") as f,
          (RESULTS / f"{experiment}-curves.jsonl").open("w", encoding="utf-8") as curves):
        for (depth, ratio, seeds, alternatives), arms in zip(settings, cells, strict=True):
            evaluated = {name: [run for run in arm if (BASE_EVAL / f"{tag_of(run)}.json").exists()]
                         for name, arm in arms.items()}
            frames = {name: pd.DataFrame([evaluation(run) for run in runs],
                                         index=pd.Index(map(tag_of, runs), name="tag"))
                      for name, runs in evaluated.items()}
            for name, arm in arms.items():
                for run in arm:
                    tag = tag_of(run)
                    scored = run in evaluated[name]
                    identity = {"schema": SCHEMA, "experiment": experiment, "arm": name, "tag": tag}
                    f.write(json.dumps(identity | {
                        "setting": {"depth": depth, "tokens_per_param": ratio, "seeds": len(seeds)},
                        "fields": dataclasses.asdict(run),
                        "records": [r for r in records(tag) if r["kind"] != "train"],
                        "evaluation": (json.loads((BASE_EVAL / f"{tag}.json").read_text(encoding="utf-8"))
                                       if scored else None),
                        "measures": frames[name].loc[tag].to_dict() if scored else None,
                        "efficiency": efficiency(run) if scored else None,
                        "diverged_at": diverged(tag)}, allow_nan=False) + "\n")
                    steps = list(train_steps(tag).values())
                    curves.write(json.dumps(identity | {k: [s[k] for s in steps] for k in (steps[0] if steps else {})
                                                        if k not in identity and k != "kind"},
                                            allow_nan=False) + "\n")
            summaries.append({"depth": depth, "tokens_per_param": ratio, "seeds": len(seeds), "lr_scale": lr_scale,
                              "production_runs": len(frames["production"]),
                              "alternatives": {name: {"fields": changes, "runs": len(frames[name])}
                                               | (compare(frames[name], frames["production"])
                                                  if len(frames[name]) and len(frames["production"]) else {})
                                               for name, changes in alternatives.items()}})
    # HEAD names the runs' code: their tags, computed at HEAD from RUN_CODE's object ids, matched their evaluations
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    (RESULTS / f"{experiment}.json").write_text(json.dumps({"commit": commit, "settings": summaries},
                                                           allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    execute(Run(**json.loads(sys.argv[1])))
