"""
The pre-registered architecture and regime ablations (ablations/ablations.md): designs, then one run per design point,
trained by scripts/base_train.train and scored by scripts/base_eval.loss_evaluation, the production code paths. A run
differs from production only in its Run fields; its plan (batch, steps, lr, weight decay) is the production model's at
its shape, so arms match in tokens and hyperparameters, and every other setting (schedule, stages, data order,
evaluation rows) is production's.

- screen: d8 and d12 at 20 tokens per scaling parameter, at each a 2^(8-4) fractional factorial (generators E = BCD,
  F = ACD, G = ABC, H = ABD, resolution IV: main effects unaliased with two-factor interactions; Box, Hunter & Hunter,
  Statistics for Experimenters, 2nd ed., Table 6.22) over eight production choices against their alternatives (pydoe
  fracfact), each of its 16 points twice; 64 runs, each its own seed.
- lr: production at d8 and d12 with the peak lr times 2^(k/2), k = -2..2; 2 seeds each.
- shape: production's d16 against one head narrower and deeper at its scaling parameters; 2 shapes x 4 seeds.
- confirm: each change the screen and the lr sweep adopt (ablations/report.py writes RESULTS/screen.json, lr.json), at
  d16 against the shape design's production runs; 4 seeds each.

Runs train one at a time (one d8 run saturates the GPU, ablations.md), each in its own process (torch.compile's caches
and limits are per process). A run whose evaluation exists is skipped; a stopped run resumes from its repo_start
checkpoint or restarts (train/metrics.py attempts keep both traceable).

    python -m ablations.study screen|lr|shape|confirm            (every missing run, each in a child process:)
    python -m ablations.study screen|lr|shape|confirm <index>    (one run of the design)
"""
import dataclasses
import itertools
import json
import math
import subprocess
import sys
from dataclasses import dataclass

import torch
from pydoe import fracfact
from torch import nn

from ablations.swiglu import SwiGLU
from nanochat.config import CODE, FILE_STAGE, REPO_STAGE, run_tag
from nanochat.data.tokenizer import get_tokenizer
from nanochat.eval.code_eval import tasks
from nanochat.model.gpt import GPT, gpt_config, scaling_params
from nanochat.paths import BASE_EVAL, LOGS, RESULTS
from nanochat.train.optim import tau_weight_decay
from nanochat.train.schedule import RunPlan, plan
from scripts.base_eval import loss_evaluation, save
from scripts.base_train import train

# Chinchilla's compute-optimal tokens per parameter (arXiv 2203.15556 §3): every ablation run's horizon
RATIO = 20
# The screen and the lr sweep at two depths: d12 (33.8M scaling parameters) decides, d8 (14.7M) replicates one scale
# down; the shape and confirm designs at d16 (67.1M)
SCREEN_DEPTHS, SHAPE_DEPTH = (8, 12), 16
# Each screen point and lr multiplier runs twice; the shape and confirm arms four times (ablations.md, Power)
REPLICATES = 2
SEEDS = range(4)
# The lr sweep: half-octave steps, one octave each side of the law's lr
LR_SCALES = tuple(2 ** (k / 2) for k in range(-2, 3))
# The code an ablation run's result depends on: production's (config.CODE) and these modules
RUN_CODE = (*CODE, "ablations/study.py", "ablations/swiglu.py")


@dataclass(frozen=True, kw_only=True)
class Run:
    """A design point: production's settings except these."""
    n_layer: int
    n_embd: int
    window_pattern: str         # gpt.GPTConfig.window_pattern
    mlp: str                    # "relu2" (gpt.MLP) or "swiglu" (ablations/swiglu.py)
    value_embeds: str           # "alternating" (gpt.has_ve) or "none"
    residual_scalars: str       # "learned" or "identity" (resid_lambdas 1 and x0_lambdas 0, frozen)
    max_norm: float             # gradient clip threshold; inf: none
    matrix_optimizer: str       # train/optim.py optimizers
    lr_scale: float             # times the plan's peak lr (schedule.plan)
    weight_decay: float | str   # "tau" (optim.tau_weight_decay) or a fixed value
    repo_stage: str             # "on" (the plan's repo_start) or "off" (every step file-level)
    seed: int                   # init and data order


def production(*, depth: int, seed: int) -> Run:
    """Production's settings at this depth (gpt_config; scripts/base_train.py main's MLP, value embeddings, residual
    scalars, clipping, matrix optimizer, lr, weight decay and stages)."""
    config = gpt_config(depth=depth, stage=FILE_STAGE)
    return Run(n_layer=depth, n_embd=config.n_embd, window_pattern=config.window_pattern, mlp="relu2",
               value_embeds="alternating", residual_scalars="learned", max_norm=math.inf, matrix_optimizer="muon",
               lr_scale=1.0, weight_decay="tau", repo_stage="on", seed=seed)


# Screen factors in the design's column order (A..H): name -> (Run field, alternative level); production(), the one
# source of production's levels, is the -1 level
FACTORS = {
    "windows": ("window_pattern", "L"),                   # Gemma 3's local windows vs DeepSeek-Coder's full attention
    "clip": ("max_norm", 1.0),                            # none vs DeepSeek LLM's 1.0 (arXiv 2401.02954 §2.3)
    "mlp": ("mlp", "swiglu"),                             # Primer's ReLU^2 vs DeepSeek-Coder's SwiGLU
    "optimizer": ("matrix_optimizer", "normuon"),         # torch.optim.Muon vs nanochat's NorMuon (dion)
    "weight_decay": ("weight_decay", 0.1),                # the tau law vs DeepSeek LLM's 0.1 (§2.3)
    "value_embeds": ("value_embeds", "none"),             # nanochat's value embeddings vs none, as DeepSeek-Coder
    "residual_scalars": ("residual_scalars", "identity"),  # nanochat's per-layer scalars vs a plain residual stream
    "repo_stage": ("repo_stage", "off"),                  # Qwen2.5-Coder's repository-level stage vs file-level only
}
# The 2^(8-4) resolution-IV generators (Box, Hunter & Hunter Table 6.22), in pydoe's fracfact notation
GENERATORS = "a b c d bcd acd abc abd"


def alternative(run: Run, names) -> Run:
    """`run` with the named factors at their alternative levels."""
    return dataclasses.replace(run, **dict(FACTORS[name] for name in names))


def screen() -> list[Run]:
    rows = [*fracfact(GENERATORS)] * REPLICATES
    return [alternative(production(depth=depth, seed=seed),
                        [name for name, sign in zip(FACTORS, row, strict=True) if sign > 0])
            for depth in SCREEN_DEPTHS for seed, row in enumerate(rows)]


def lr() -> list[Run]:
    return [dataclasses.replace(production(depth=depth, seed=seed), lr_scale=scale)
            for depth, scale, seed in itertools.product(SCREEN_DEPTHS, LR_SCALES, range(REPLICATES))]


def model_of(run: Run) -> GPT:
    """The run's model on the meta device: production's config at the run's depth with its width (whole heads of
    production's head_dim) and windows; its MLP; without value embeddings, no tables and no gates; with identity
    residual scalars, resid_lambdas and x0_lambdas frozen at their init (1 and 0, GPT.init_weights)."""
    config = gpt_config(depth=run.n_layer, stage=FILE_STAGE)
    head_dim = config.n_embd // config.n_head
    with torch.device("meta"):
        model = GPT(dataclasses.replace(config, n_embd=run.n_embd, n_head=run.n_embd // head_dim,
                                        window_pattern=run.window_pattern))
        if run.mlp == "swiglu":
            for block in model.blocks:
                block.mlp = SwiGLU(model.config)
    if run.value_embeds == "none":
        model.value_embeds = nn.ModuleDict()
        for block in model.blocks:
            block.attn.ve_gate = None
    if run.residual_scalars == "identity":
        model.resid_lambdas.requires_grad_(False)
        model.x0_lambdas.requires_grad_(False)
    return model


def plan_of(run: Run) -> RunPlan:
    """The plan of production's model at the run's shape (schedule.plan at RATIO), with the run's peak lr and stages."""
    shape_plan = plan(model=model_of(dataclasses.replace(production(depth=run.n_layer, seed=run.seed),
                                                         n_embd=run.n_embd)), ratio=RATIO)
    return dataclasses.replace(shape_plan, lr=run.lr_scale * shape_plan.lr,
                               repo_start=shape_plan.num_iterations if run.repo_stage == "off"
                               else shape_plan.repo_start)


def weight_decay_of(run: Run, run_plan: RunPlan) -> float:
    if run.weight_decay == "tau":
        return tau_weight_decay(lr=run_plan.lr, steps=run_plan.num_iterations,
                                tokens_per_param=run_plan.tokens_per_param)
    return run.weight_decay


def iso_param_depth(run: Run, *, width: int, params: int) -> int:
    """The depth at which `run` at this width has the scaling-parameter count nearest `params` (the count grows with
    depth, so the nearest is one of the two depths around it)."""
    def count(depth):
        return scaling_params(model_of(dataclasses.replace(run, n_layer=depth, n_embd=width)))
    depth = 1
    while count(depth + 1) <= params:
        depth += 1
    return min((depth, depth + 1), key=lambda d: abs(count(d) - params))


def shape() -> list[Run]:
    """Production's d16 and, at its scaling parameters, one head narrower and deeper (MobileLLM, arXiv 2402.14905
    §2.2.2: deeper and thinner is better at fixed parameters)."""
    runs = []
    for seed in SEEDS:
        reference = production(depth=SHAPE_DEPTH, seed=seed)
        config = model_of(reference).config
        width = reference.n_embd - config.n_embd // config.n_head
        params = scaling_params(model_of(reference))
        runs += [reference, dataclasses.replace(reference, n_layer=iso_param_depth(reference, width=width,
                                                                                   params=params), n_embd=width)]
    return runs


def adopted() -> dict[str, dict]:
    """{change: its Run fields} adopted by the screen and the lr sweep (RESULTS/screen.json, lr.json)."""
    return {name: change for design in ("screen", "lr")
            for name, change in json.loads((RESULTS / f"{design}.json").read_text())["adopted"].items()}


def confirm() -> list[Run]:
    """Each adopted change on the d16 production shape, per seed; the baseline is the shape design's."""
    return [dataclasses.replace(production(depth=SHAPE_DEPTH, seed=seed), **change)
            for change, seed in itertools.product(adopted().values(), SEEDS)]


DESIGNS = {"screen": screen, "lr": lr, "shape": shape, "confirm": confirm}


def tag_of(run: Run) -> str:
    """config.run_tag over the run's fields (no clipping, an infinite threshold, as null) and RUN_CODE."""
    return run_tag(depth=run.n_layer, ratio=RATIO, seed=run.seed, data_seed=run.seed, code=RUN_CODE,
                   variant={k: None if v == math.inf else v for k, v in dataclasses.asdict(run).items()})


def execute(run: Run) -> None:
    """Train and evaluate one run: its metrics as production's (train/metrics.py; the run record holds its model, plan,
    weight decay, clip threshold and matrix optimizer), its evaluation as base_eval's schema with loss_evaluation only,
    at 32K too for a run without the repository-level stage."""
    tag, run_plan = tag_of(run), plan_of(run)
    steps = run_plan.num_iterations
    # resume checkpoints only at repo_start and the end; the model-only snapshot at repo_start
    model = train(model=model_of(run), run=run_plan, weight_decay=weight_decay_of(run, run_plan), seed=run.seed,
                  data_seed=run.seed, save_every=steps, snapshot_every=steps, tag=tag, max_norm=run.max_norm,
                  matrix_optimizer=run.matrix_optimizer)
    model.set_stage(REPO_STAGE)
    save(tag, steps, loss_evaluation(model, get_tokenizer(), tasks(), device=torch.device("cuda")) |
         {"codegen": None, "fim": None})


def main():
    design = DESIGNS[sys.argv[1]]()
    if len(sys.argv) == 3:
        execute(design[int(sys.argv[2])])
        return
    for i, run in enumerate(design):
        tag = tag_of(run)
        if not (BASE_EVAL / f"{tag}.json").exists():
            # every attempt's output, a failure's traceback included, appended to the run's log
            with (LOGS / f"{tag}.log").open("a", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-m", "ablations.study", sys.argv[1], str(i)], stdout=log,
                               stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
