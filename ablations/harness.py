"""
The mechanism every ablation experiment (ablations/<experiment>.py, one per open decision of docs/h100-pretraining.md;
ablations.md) shares. An experiment is production at a setting (depth, tokens per scaling parameter) and one arm per
alternative, the arms differing from production only in their Run fields, each on the same seeds. A run trains by
scripts/base_train.train and is scored by scripts/base_eval.loss_evaluation, the production code paths; its model is
ablations/model.py's Candidate (production's GPT when every structural field is production's); its plan (batch, steps,
peak lr) is the production model's at its depth and ratio, so every arm matches in tokens and hyperparameters.

Runs train one at a time (one d8 run saturates the GPU), each in its own process (torch.compile's caches and limits are
per process), its output, a failure's traceback included, appended to LOGS/<tag>.log. A run whose evaluation exists is
skipped (arms shared between experiments train once); a stopped run resumes from its repo_start checkpoint or restarts
(train/metrics.py attempts keep both traceable). Then each alternative is compared with production; the experiment
writes RESULTS/<experiment>.jsonl (a record per run: its fields, evaluation and efficiency), RESULTS/<experiment>.json
(each alternative's fields, differences, p and confidence intervals, ablations/confirm.py's input) and
RESULTS/<experiment>.md (the tables).

    python -m ablations.<experiment>            (every missing run, each in a child process, then the comparison)
    python -m ablations.<experiment> <index>    (one run)
"""
import dataclasses
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from functools import partial

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
from nanochat.paths import BASE_EVAL, LOGS, RESULTS
from nanochat.train.metrics import latest, train_steps
from nanochat.train.optim import tau_weight_decay
from nanochat.train.schedule import RunPlan, lr_multiplier, plan
from scripts.base_eval import loss_evaluation, save
from scripts.base_train import train

# Chinchilla's compute-optimal tokens per parameter (arXiv 2203.15556 §3)
CHINCHILLA = 20
# (depth, tokens per scaling parameter): the architecture experiments' d8 (14.7M scaling parameters) at CHINCHILLA; the
# regime experiments' d8 at 100, about sqrt(20 x 490.8) = 99, the geometric midpoint between CHINCHILLA and production's
# ratio (runs/h100.sh), toward the horizon a regime is chosen for; the confirmation's d16 (67.1M) at CHINCHILLA
ARCHITECTURE = (8, CHINCHILLA)
REGIME = (8, 100)
CONFIRM = (16, CHINCHILLA)
# Seeds per arm: d8's and d16's (ablations.md, Power)
SEEDS = range(8)
CONFIRM_SEEDS = range(4)
# The conventional significance level (Fisher, Statistical Methods for Research Workers, 1925, §20)
ALPHA = 0.05
# The metric that decides and those an alternative must not worsen (metrics)
DECISION, GUARDS = "code", ("solutions", "repo_32k")
# Version of RESULTS/<experiment>.jsonl's records
SCHEMA = 1
# The code a run's result depends on: production's (config.CODE) and this mechanism
RUN_CODE = (*CODE, "ablations/harness.py", "ablations/model.py", "ablations/swiglu.py", "ablations/depth_attention.py",
            "ablations/depth_attention.cu")


@dataclass(frozen=True, kw_only=True)
class Run:
    """A run: production except these fields (the structural ones: ablations/model.py)."""
    n_layer: int
    ratio: float         # tokens per scaling parameter
    lr_scale: float      # times the plan's peak lr (schedule.plan)
    max_norm: float      # gradient clip threshold; inf: none
    schedule: str        # "decay" (schedule.lr_multiplier over the run) or "rewarm" (lr_schedule_of)
    mlp: str             # "relu2" or "swiglu"
    value_embeds: str    # "alternating" or "none"
    residual: str        # "scalars", "identity" or "attn_res"
    qk_gains: str        # "learned" or "fixed"
    windows: str         # "SSSL" or "full"
    rope: str            # "one" or "staged"
    key_offset: str      # "off" or "on"
    seed: int            # init and data order


def production(*, depth: int, ratio: float, seed: int) -> Run:
    """Production's settings (scripts/base_train.py main, nanochat/model/gpt.py) at this depth and ratio."""
    return Run(n_layer=depth, ratio=ratio, lr_scale=1.0, max_norm=math.inf, schedule="decay", mlp="relu2",
               value_embeds="alternating", residual="scalars", qk_gains="learned", windows="SSSL", rope="one",
               key_offset="off", seed=seed)


def fields(run: Run) -> dict:
    """The run's fields as JSON values (no clipping, an infinite threshold, as null)."""
    return {k: None if v == math.inf else v for k, v in dataclasses.asdict(run).items()}


def model_of(run: Run) -> Candidate:
    """The run's model on the meta device."""
    with torch.device("meta"):
        return Candidate(gpt_config(depth=run.n_layer, stage=FILE_STAGE), mlp=run.mlp, value_embeds=run.value_embeds,
                         residual=run.residual, qk_gains=run.qk_gains, windows=run.windows, rope=run.rope,
                         key_offset=run.key_offset)


def plan_of(run: Run) -> RunPlan:
    """The production model's plan at the run's depth and ratio (schedule.plan), at the run's peak lr."""
    with torch.device("meta"):
        production_plan = plan(model=GPT(gpt_config(depth=run.n_layer, stage=FILE_STAGE)), ratio=run.ratio)
    return dataclasses.replace(production_plan, lr=run.lr_scale * production_plan.lr)


def rewarm(it: int, *, repo_start: int, steps: int) -> float:
    """Production's multiplier in the file-level stage; from repo_start, warmed up again to 1 and decayed to zero over
    the repository stage's steps (schedule.lr_multiplier; re-warming for a new data distribution, Gupta et al., arXiv
    2308.04014)."""
    if it < repo_start:
        return lr_multiplier(it, steps=steps)
    return lr_multiplier(it - repo_start, steps=steps - repo_start)


def lr_schedule_of(run: Run, run_plan: RunPlan):
    if run.schedule == "decay":
        return partial(lr_multiplier, steps=run_plan.num_iterations)
    return partial(rewarm, repo_start=run_plan.repo_start, steps=run_plan.num_iterations)


def tag_of(run: Run) -> str:
    """config.run_tag over the run's fields and RUN_CODE."""
    return run_tag(depth=run.n_layer, ratio=run.ratio, seed=run.seed, data_seed=run.seed, variant=fields(run),
                   code=RUN_CODE)


def execute(run: Run) -> None:
    """Train and evaluate one run: its metrics as production's (train/metrics.py), its evaluation as base_eval's schema
    with loss_evaluation only; tau-law weight decay (production's); checkpoints at repo_start and the end only."""
    tag, run_plan = tag_of(run), plan_of(run)
    steps = run_plan.num_iterations
    model = train(model=model_of(run), run=run_plan,
                  weight_decay=tau_weight_decay(lr=run_plan.lr, steps=steps,
                                                tokens_per_param=run_plan.tokens_per_param),
                  lr_schedule=lr_schedule_of(run, run_plan), max_norm=run.max_norm, variant=fields(run), seed=run.seed,
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
    return {"params": started["params"], "trained_params": started["trained_params"],
            "flops_per_token": started["flops_per_token"], "peak_memory_bytes": ended["peak_memory_bytes"]}


def compare(alternative: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Per measure (column): the alternative's mean minus the baseline's, the two-sided pooled t-test's p and the
    difference's 1 - ALPHA confidence interval (scipy ttest_ind)."""
    rows = {}
    for m in baseline:
        test = ttest_ind(alternative[m], baseline[m])
        interval = test.confidence_interval(confidence_level=1 - ALPHA)
        rows[m] = {"difference": alternative[m].mean() - baseline[m].mean(), "p": test.pvalue,
                   "ci_low": interval.low, "ci_high": interval.high}
    return pd.DataFrame(rows).T


def better(table: pd.DataFrame, decision_p: float) -> bool:
    """The decision rule: DECISION lower with `decision_p` (its p, adjusted where a family is tested) < ALPHA and no
    guard higher with p < ALPHA."""
    return (table.at[DECISION, "difference"] < 0 and decision_p < ALPHA
            and not any(table.at[m, "difference"] > 0 and table.at[m, "p"] < ALPHA for m in GUARDS))


def main(*, alternatives: dict[str, dict], setting: tuple[int, float], seeds: range) -> None:
    """Run the calling experiment (python -m ablations.<experiment>): production at `setting` and an arm per
    alternative {name: Run fields}, on `seeds`; then compare each alternative with production. RESULTS/<experiment>.md
    marks each better or not by its own p; RESULTS/<experiment>.json holds each alternative's fields, differences and
    p, from which ablations/confirm.py decides across every experiment's alternatives."""
    module = sys.modules["__main__"].__spec__.name
    experiment = module.rsplit(".", 1)[1]
    depth, ratio = setting
    baseline = [production(depth=depth, ratio=ratio, seed=seed) for seed in seeds]
    arms = {"production": baseline} | {name: [dataclasses.replace(run, **changes) for run in baseline]
                                       for name, changes in alternatives.items()}
    runs = [run for arm in arms.values() for run in arm]
    if len(sys.argv) == 2:
        execute(runs[int(sys.argv[1])])
        return
    for i, run in enumerate(runs):
        tag = tag_of(run)
        if not (BASE_EVAL / f"{tag}.json").exists():
            with (LOGS / f"{tag}.log").open("a", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-m", module, str(i)], stdout=log, stderr=subprocess.STDOUT,
                               check=True)
    frames = {name: pd.DataFrame([evaluation(run) for run in arm], index=pd.Index(map(tag_of, arm), name="tag"))
              for name, arm in arms.items()}
    costs = {name: pd.DataFrame([efficiency(run) for run in arm], index=frames[name].index)
             for name, arm in arms.items()}
    tables = {name: compare(frames[name], frames["production"]) for name in alternatives}
    with (RESULTS / f"{experiment}.jsonl").open("w", encoding="utf-8") as f:
        for name, arm in arms.items():
            for run in arm:
                tag = tag_of(run)
                f.write(json.dumps({"schema": SCHEMA, "experiment": experiment, "arm": name, "tag": tag,
                                    "fields": fields(run), "evaluation": frames[name].loc[tag].to_dict(),
                                    "efficiency": costs[name].loc[tag].to_dict()}, allow_nan=False) + "\n")
    lines = [f"# {experiment}: d{depth} at {ratio} tokens per parameter, {len(seeds)} seeds per arm\n", "## Means\n",
             pd.DataFrame({name: pd.concat([frames[name].mean(), costs[name].mean()]) for name in arms}).to_markdown()]
    for name, table in tables.items():
        verdict = "better" if better(table, table.at[DECISION, "p"]) else "not better"
        lines += [f"\n## {name} against production: {verdict}\n", table.to_markdown()]
    lines += ["\n## Runs\n", pd.concat(frames, names=["arm"]).to_markdown()]
    (RESULTS / f"{experiment}.md").write_text("\n".join(lines), encoding="utf-8")
    (RESULTS / f"{experiment}.json").write_text(json.dumps(
        {"alternatives": {name: {"fields": alternatives[name]} | table.to_dict() for name, table in tables.items()}},
        allow_nan=False), encoding="utf-8")
