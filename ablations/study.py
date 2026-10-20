"""
The pre-registered architecture and regime ablations (ablations/ablations.md): designs, then one run per design point,
trained by scripts/base_train.train and scored by scripts/base_eval.loss_evaluation, the production code paths. A run
differs from production only in its Run fields; every other setting (plan, schedule, weight decay, stages, data order,
evaluation rows) is production's.

- screen: d8 at 20 tokens per scaling parameter, a 2^(6-2) fractional factorial (generators E = ABC, F = BCD,
  resolution IV: main effects unaliased with two-factor interactions; Box, Hunter & Hunter, Statistics for
  Experimenters, 2nd ed., Table 6.22) over six production choices against their alternatives (pydoe fracfact), run in
  two blocks (seeds); 32 runs.
- shape: depth against width at the d16 scaling-parameter count, production otherwise; 4 shapes x 2 seeds.
- confirm: each factor the screen adopts (ablations/report.py writes RESULTS/screen.json), at d16 against the shape
  design's production-shape runs; 2 seeds each.

Runs train one at a time (one d8 run saturates the GPU, ablations.md), each in its own process (torch.compile's caches
and limits are per process). A run whose evaluation exists is skipped; a stopped run resumes from its repo_start
checkpoint or restarts (train/metrics.py attempts keep both traceable).

    python -m ablations.study screen|shape|confirm            (every missing run, each in a child process:)
    python -m ablations.study screen|shape|confirm <index>    (one run of the design)
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

from ablations.swiglu import SwiGLU
from nanochat.config import CODE, FILE_STAGE, MIXTURE, ROPE_BASE, VOCAB_SIZE, Stage, run_tag
from nanochat.data.tokenizer import get_tokenizer
from nanochat.eval.code_eval import tasks
from nanochat.model.gpt import GPT, GPTConfig, gpt_config, scaling_params
from nanochat.paths import BASE_EVAL, RESULTS
from nanochat.train.schedule import plan
from scripts.base_eval import loss_evaluation, save
from scripts.base_train import train

# Chinchilla's compute-optimal tokens per parameter (arXiv 2203.15556 §3): every ablation run's horizon
RATIO = 20
# production's mixture and Qwen2.5-Coder's chosen code:text:math = 70:20:10 (arXiv 2409.12186 Table 3) on its corpora
MIXTURES = {"production": MIXTURE, "qwen": {group: (share, MIXTURE[group][1])
                                            for group, share in (("code", 0.70), ("text", 0.20), ("math", 0.10))}}
# Two blocks: the screen's power (ablations.md, Power) at 32 runs
SEEDS = (0, 1)
# The screen's d8 (14.7M scaling parameters) and the shape and confirm designs' d16 (67.1M)
SCREEN_DEPTH, SHAPE_DEPTH = 8, 16
# The code an ablation run's result depends on: production's (config.CODE) and these modules
RUN_CODE = (*CODE, "ablations/study.py", "ablations/swiglu.py")


@dataclass(frozen=True, kw_only=True)
class Run:
    """A design point: production's settings except these."""
    n_layer: int
    n_embd: int
    head_dim: int
    window_pattern: str    # gpt.GPTConfig.window_pattern
    mlp: str               # "relu2" (gpt.MLP) or "swiglu" (ablations/swiglu.py)
    max_norm: float        # gradient clip threshold; inf: none
    mixture: str           # MIXTURES key: the file-level stage's mixture
    matrix_optimizer: str  # train/optim.py optimizers
    seed: int              # init and data order


def production(*, depth: int, seed: int) -> Run:
    """Production's settings at this depth (gpt_config; scripts/base_train.py main's MLP, clipping, mixture, matrix
    optimizer)."""
    config = gpt_config(depth=depth, stage=FILE_STAGE)
    return Run(n_layer=depth, n_embd=config.n_embd, head_dim=config.n_embd // config.n_head,
               window_pattern=config.window_pattern, mlp="relu2", max_norm=math.inf, mixture="production",
               matrix_optimizer="muon", seed=seed)


# Screen factors in the design's column order (A..F): name -> (Run field, alternative level); production(), the one
# source of production's levels, is the -1 level
FACTORS = {
    "windows": ("window_pattern", "L"),             # Gemma 3's local windows vs full attention in every layer
    "head_dim": ("head_dim", 64),                   # nanochat LOG 2026-01-17's 128 vs twice the heads (MobileLLM §2.2.4)
    "clip": ("max_norm", 1.0),                      # none vs DeepSeek LLM's 1.0 (arXiv 2401.02954 §2.3)
    "mixture": ("mixture", "qwen"),                 # 87:10:3 vs Qwen2.5-Coder's 70:20:10
    "mlp": ("mlp", "swiglu"),                       # Primer's ReLU^2 vs SwiGLU (Shazeer, arXiv 2002.05202)
    "optimizer": ("matrix_optimizer", "normuon"),   # torch.optim.Muon vs nanochat's NorMuon (dion)
}
# The 2^(6-2) resolution-IV generators (Box, Hunter & Hunter Table 6.22), in pydoe's fracfact notation
GENERATORS = "a b c d abc bcd"


def alternative(run: Run, names) -> Run:
    """`run` with the named factors at their alternative levels."""
    return dataclasses.replace(run, **dict(FACTORS[name] for name in names))


def screen() -> list[Run]:
    return [alternative(production(depth=SCREEN_DEPTH, seed=seed), [name for name, sign in zip(FACTORS, row, strict=True)
                                                         if sign > 0])
            for seed, row in itertools.product(SEEDS, fracfact(GENERATORS))]


def model_of(run: Run) -> GPT:
    """The run's model on the meta device."""
    with torch.device("meta"):
        model = GPT(GPTConfig(sequence_len=FILE_STAGE.seq_len, vocab_size=VOCAB_SIZE, rope_base=ROPE_BASE,
                              n_layer=run.n_layer, n_head=run.n_embd // run.head_dim, n_embd=run.n_embd,
                              window_pattern=run.window_pattern))
        if run.mlp == "swiglu":
            for block in model.blocks:
                block.mlp = SwiGLU(model.config)
    return model


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
    """Widths of whole heads around the d16 reference's (one head narrower to two wider), each at the depth matching the
    reference's scaling parameters."""
    runs = []
    for seed in SEEDS:
        reference = production(depth=SHAPE_DEPTH, seed=seed)
        params = scaling_params(model_of(reference))
        for k in (-1, 0, 1, 2):
            width = reference.n_embd + k * reference.head_dim
            runs.append(dataclasses.replace(reference, n_layer=iso_param_depth(reference, width=width, params=params),
                                            n_embd=width))
    return runs


def confirm() -> list[Run]:
    """Each screen factor in RESULTS/screen.json's "adopted" at its alternative on the d16 production shape; the
    baseline is the shape design's."""
    adopted = json.loads((RESULTS / "screen.json").read_text())["adopted"]
    return [alternative(production(depth=SHAPE_DEPTH, seed=seed), [name])
            for name, seed in itertools.product(adopted, SEEDS)]


DESIGNS = {"screen": screen, "shape": shape, "confirm": confirm}


def tag_of(run: Run) -> str:
    """config.run_tag over the run's fields (no clipping, an infinite threshold, as null) and RUN_CODE."""
    return run_tag(depth=run.n_layer, ratio=RATIO, seed=run.seed, data_seed=run.seed, code=RUN_CODE,
                   variant={k: None if v == math.inf else v for k, v in dataclasses.asdict(run).items()})


def execute(run: Run) -> None:
    """Train and evaluate one run: its metrics as production's (train/metrics.py; the run record holds its model, clip
    threshold, matrix optimizer and mixture), its evaluation as base_eval's schema with loss_evaluation only."""
    tag, model = tag_of(run), model_of(run)
    steps = plan(model=model, ratio=RATIO).num_iterations
    # resume checkpoints only at repo_start and the end; the model-only snapshot at repo_start
    model = train(model=model, ratio=RATIO, seed=run.seed, data_seed=run.seed, save_every=steps, snapshot_every=steps,
                  tag=tag, max_norm=run.max_norm, matrix_optimizer=run.matrix_optimizer,
                  file_stage=Stage(seq_len=FILE_STAGE.seq_len, mixture=MIXTURES[run.mixture]))
    save(tag, steps, loss_evaluation(model, get_tokenizer(), tasks(), device=torch.device("cuda")) |
         {"codegen": None, "fim": None})


def main():
    design = DESIGNS[sys.argv[1]]()
    if len(sys.argv) == 3:
        execute(design[int(sys.argv[2])])
        return
    for i, run in enumerate(design):
        if not (BASE_EVAL / f"{tag_of(run)}.json").exists():
            subprocess.run([sys.executable, "-m", "ablations.study", sys.argv[1], str(i)], check=True)


if __name__ == "__main__":
    main()
