"""
A run's training rows per stage: the packed train rows of the stage's corpora (corpus/pack.py packs each stage's
documents apart, so no document is read twice), drawn by the run's weights in the stage (corpus/mixture.py weights: the
stage's shares of the packed rows at the rows the run draws there). max_ratio: the most tokens per scaling parameter at
which no stage draws more rows of a corpus than it holds (PretrainingBatches stops at an exhausted corpus).
stage_batches: a stage's training micro-batches in the run's data order.
"""

import bisect
from collections.abc import Iterator

import numpy as np
import torch

from corpus.layout import stage_rows
from corpus.loader import PretrainingBatches
from corpus.mixture import FILE_STAGE, MICRO_BATCH_TOKENS, REPO_STAGE, Stage, weights
from corpus.tokenizer import Tokenizer
from model.gpt import GPT, gpt_config, scaling_params
from train.schedule import RunPlan, plan


def drawn(stage: Stage, run: RunPlan) -> int:
    """The rows the run draws in the stage: its steps there x micro-batches x rows per micro-batch."""
    steps = run.num_iterations - run.repo_start if stage is REPO_STAGE else run.repo_start
    return steps * run.grad_accum_steps * stage.batch_rows


def stage_batches(tokenizer: Tokenizer, *, stage: Stage, run: RunPlan, seed: int, device: torch.device,
                  start: int) -> tuple[dict[str, float], Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """({corpus: weight}, the stage's training micro-batches from run row `start` on): each corpus's packed rows
    permuted once by `seed` (RandomState's stream is frozen across numpy versions, so a resume sees the same order),
    FIM pieces read as FIM examples. A run that would overdraw a corpus raises (max_ratio bounds the ratio)."""
    rows = stage_rows(tokenizer, stage)
    mixture, _, over = weights(stage, rows, drawn(stage, run))
    if over:
        raise RuntimeError(f"the run draws {over} rows beyond the T{stage.seq_len} packs (train/stages.py max_ratio)")
    orders = {corpus: np.random.RandomState(seed).permutation(rows[corpus]) for corpus in mixture}
    return mixture, iter(PretrainingBatches(mixture, "train", tokenizer, orders, seq_len=stage.seq_len,
                                            batch_rows=stage.batch_rows, device=device, reading="fim", start=start))


def max_ratio(tokenizer: Tokenizer, *, depth: int) -> float:
    """Bisection over whole micro-batches of tokens on "overdraws". Not monotone (grad_accum_steps steps up with the
    count, so trained tokens dip): the result fits, but a larger fitting count may exist."""
    with torch.device("meta"):
        model = GPT(gpt_config(depth=depth, stage=FILE_STAGE))
    params = scaling_params(model)
    rows = [(stage, stage_rows(tokenizer, stage)) for stage in (FILE_STAGE, REPO_STAGE)]

    def ratio(n: int) -> float:
        return n * MICRO_BATCH_TOKENS / params

    def overdrawn(n: int) -> bool:
        run = plan(model=model, ratio=ratio(n))
        return any(weights(stage, held, drawn(stage, run))[2] for stage, held in rows if drawn(stage, run))

    n = bisect.bisect_left(range(sum(rows[0][1].values()) * FILE_STAGE.seq_len // MICRO_BATCH_TOKENS + 1), True,
                           key=overdrawn)
    return ratio(n - 1)
