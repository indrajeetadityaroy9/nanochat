"""
The mechanism every ablation experiment (ablations/{lr,mlp,weight_decay,confirm}.py; ablations.md) shares. An experiment
is production at a setting (depth, tokens per scaling parameter) and one arm per alternative, the arms differing from
production only in their Run fields, each on the same seeds. A run trains by scripts/base_train.train and is scored by
scripts/base_eval.loss_evaluation, the production code paths; its plan (batch, steps, peak lr) is the production
model's at its depth and ratio, so every arm matches in tokens and hyperparameters.

Runs train one at a time (one d8 run saturates the GPU), each in its own process (torch.compile's caches and limits are
per process), its output, a failure's traceback included, appended to LOGS/<tag>.log. A run whose evaluation exists is
skipped (arms shared between experiments train once); a stopped run resumes from its repo_start checkpoint or restarts
(train/metrics.py attempts keep both traceable). Then each alternative is compared with production and the experiment
writes RESULTS/<experiment>.md and RESULTS/<experiment>.json ({"adopted": the Run fields of the adopted alternative,
{} for none}).

    python -m ablations.<experiment>            (every missing run, each in a child process, then the comparison)
    python -m ablations.<experiment> <index>    (one run)
"""
import dataclasses
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass

import pandas as pd
import torch
from scipy.stats import ttest_ind

from ablations.swiglu import SwiGLU
from nanochat.config import CODE, FILE_STAGE, run_tag
from nanochat.data.tokenizer import get_tokenizer
from nanochat.eval.code_eval import tasks
from nanochat.eval.loss_eval import bpb
from nanochat.model.gpt import GPT, gpt_config
from nanochat.paths import BASE_EVAL, LOGS, RESULTS
from nanochat.train.metrics import train_steps
from nanochat.train.optim import tau_weight_decay
from nanochat.train.schedule import RunPlan, plan
from scripts.base_eval import loss_evaluation, save
from scripts.base_train import train

# Chinchilla's compute-optimal tokens per parameter (arXiv 2203.15556 §3)
CHINCHILLA = 20
# The experiments' setting: d8 (14.7M scaling parameters) at 100 tokens per parameter, where the weight-decay contrast
# is production's (DeepSeek LLM's 0.1 decays on an EMA timescale 3.1x the tau law's, 3.2x at d24 and 490.8;
# ablations.md); the confirmation's: d16 (67.1M) at CHINCHILLA
SETTING = (8, 100)
CONFIRM = (16, CHINCHILLA)
# Seeds per arm: d8's and d16's (ablations.md, Power)
SEEDS = range(8)
CONFIRM_SEEDS = range(4)
# The conventional significance level (Fisher, Statistical Methods for Research Workers, 1925, §20)
ALPHA = 0.05
# The metric that decides and those an alternative must not worsen (metrics)
DECISION, GUARDS = "code", ("solutions", "repo_32k")
# The code a run's result depends on: production's (config.CODE) and this mechanism
RUN_CODE = (*CODE, "ablations/harness.py", "ablations/swiglu.py")


@dataclass(frozen=True, kw_only=True)
class Run:
    """A run: production except these fields."""
    n_layer: int
    ratio: float                # tokens per scaling parameter
    mlp: str                    # "relu2" (gpt.MLP) or "swiglu" (ablations/swiglu.py)
    weight_decay: float | str   # "tau" (optim.tau_weight_decay) or a fixed value
    lr_scale: float             # times the plan's peak lr (schedule.plan)
    seed: int                   # init and data order


def production(*, depth: int, ratio: float, seed: int) -> Run:
    """Production's MLP, weight decay and lr (scripts/base_train.py main) at this depth and ratio."""
    return Run(n_layer=depth, ratio=ratio, mlp="relu2", weight_decay="tau", lr_scale=1.0, seed=seed)


def model_of(run: Run) -> GPT:
    """Production's model at the run's depth with its MLP, on the meta device."""
    with torch.device("meta"):
        model = GPT(gpt_config(depth=run.n_layer, stage=FILE_STAGE))
        if run.mlp == "swiglu":
            for block in model.blocks:
                block.mlp = SwiGLU(model.config)
    return model


def plan_of(run: Run) -> RunPlan:
    """The production model's plan at the run's depth and ratio (schedule.plan), at the run's peak lr."""
    with torch.device("meta"):
        production_plan = plan(model=GPT(gpt_config(depth=run.n_layer, stage=FILE_STAGE)), ratio=run.ratio)
    return dataclasses.replace(production_plan, lr=run.lr_scale * production_plan.lr)


def weight_decay_of(run: Run, run_plan: RunPlan) -> float:
    if run.weight_decay == "tau":
        return tau_weight_decay(lr=run_plan.lr, steps=run_plan.num_iterations,
                                tokens_per_param=run_plan.tokens_per_param)
    return run.weight_decay


def tag_of(run: Run) -> str:
    """config.run_tag over the run's fields and RUN_CODE."""
    return run_tag(depth=run.n_layer, ratio=run.ratio, seed=run.seed, data_seed=run.seed,
                   variant=dataclasses.asdict(run), code=RUN_CODE)


def execute(run: Run) -> None:
    """Train and evaluate one run: its metrics as production's (train/metrics.py), its evaluation as base_eval's schema
    with loss_evaluation only. Checkpoints at repo_start and the end only."""
    tag, run_plan = tag_of(run), plan_of(run)
    steps = run_plan.num_iterations
    model = train(model=model_of(run), run=run_plan, weight_decay=weight_decay_of(run, run_plan), seed=run.seed,
                  data_seed=run.seed, save_every=steps, snapshot_every=steps, tag=tag)
    save(tag, steps, loss_evaluation(model, get_tokenizer(), tasks(), device=torch.device("cuda")) |
         {"codegen": None, "fim": None})


def metrics(run: Run) -> dict[str, float]:
    """A run's metrics (bits per byte; tokens per second) from its evaluation and its per-step records: code, text and
    math, the 8K validation bits per byte of each group at production's corpus weights (loss_eval.val_bits "bpb");
    solutions, the bits per byte of the HumanEval+, MBPP+ and HumanEval-X C++ reference solutions given their prompts,
    pooled (code_eval.solution_bits); repo_32k, the 32K repository bits per byte of the code group; tok_per_s, the
    median over the file-level stage's steps of the step's tokens per second."""
    tag = tag_of(run)
    result = json.loads((BASE_EVAL / f"{tag}.json").read_text(encoding="utf-8"))
    solution = result["solution"].values()
    rates = [r["tok_per_s"] for step, r in train_steps(tag).items() if step < plan_of(run).repo_start]
    return {"code": result["8k"]["bpb"]["code"],
            "solutions": bpb({k: [sum(sum(s[k]) for s in solution)] for k in ("bits", "bytes")}),
            "repo_32k": result["32k"]["bpb"]["code"], "text": result["8k"]["bpb"]["text"],
            "math": result["8k"]["bpb"]["math"], "tok_per_s": statistics.median(rates)}


def compare(alternative: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Per metric (column): the alternative's mean minus the baseline's, and the two-sided pooled t-test's p (scipy)."""
    return pd.DataFrame({m: {"difference": alternative[m].mean() - baseline[m].mean(),
                             "p": ttest_ind(alternative[m], baseline[m]).pvalue} for m in baseline}).T


def better(table: pd.DataFrame) -> bool:
    """The decision rule: DECISION lower with p < ALPHA and no guard higher with p < ALPHA (a false adoption has
    probability ALPHA / 2 per alternative)."""
    return (table.at[DECISION, "difference"] < 0 and table.at[DECISION, "p"] < ALPHA
            and not any(table.at[m, "difference"] > 0 and table.at[m, "p"] < ALPHA for m in GUARDS))


def main(*, alternatives: dict[str, dict], setting: tuple[int, float], seeds: range) -> None:
    """Run the calling experiment (python -m ablations.<experiment>): production at `setting` and an arm per
    alternative {name: Run fields}, on `seeds`; then compare each alternative with production and adopt, of those
    better by the rule, the one with the lowest mean DECISION."""
    module = sys.modules["__main__"].__spec__.name
    experiment = module.rsplit(".", 1)[1]
    depth, ratio = setting
    baseline = [production(depth=depth, ratio=ratio, seed=seed) for seed in seeds]
    arms = {"production": baseline} | {name: [dataclasses.replace(run, **fields) for run in baseline]
                                       for name, fields in alternatives.items()}
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
    frames = {name: pd.DataFrame([metrics(run) for run in arm], index=pd.Index(map(tag_of, arm), name="tag"))
              for name, arm in arms.items()}
    tables = {name: compare(frames[name], frames["production"]) for name in alternatives}
    passing = [name for name in alternatives if better(tables[name])]
    adopted = min(passing, key=lambda name: frames[name][DECISION].mean(), default=None)
    lines = [f"# {experiment}: d{depth} at {ratio} tokens per parameter, {len(seeds)} seeds per arm\n",
             f"Adopted: {adopted or 'none'}\n", "## Means\n",
             pd.DataFrame({name: frame.mean() for name, frame in frames.items()}).to_markdown()]
    for name, table in tables.items():
        lines += [f"\n## {name} against production\n", table.to_markdown()]
    lines += ["\n## Runs\n", pd.concat(frames, names=["arm"]).to_markdown()]
    (RESULTS / f"{experiment}.md").write_text("\n".join(lines), encoding="utf-8")
    (RESULTS / f"{experiment}.json").write_text(json.dumps({"adopted": alternatives[adopted] if adopted else {}},
                                                           allow_nan=False), encoding="utf-8")
