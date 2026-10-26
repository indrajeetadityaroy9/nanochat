"""
A run's training rows per stage: the packed train rows of the stage's corpora (corpus/pack.py packs each stage's
documents apart, so no document is read twice), drawn by the run's weights in the stage (corpus/mixture.py weights: the
stage's shares of the packed rows at the rows the run draws there). max_tokens_per_param: the most tokens per
non-embedding parameter at which no stage draws more rows of a corpus than it holds (PretrainingBatches stops at an
exhausted corpus). stage_batches: a stage's training micro-batches in the run's data order.
"""

import bisect
from collections.abc import Iterator

import numpy as np
import torch

from corpus.layout import stage_rows
from corpus.loader import PretrainingBatches
from corpus.mixture import FILE_LEVEL_STAGE, MICRO_BATCH_TOKENS, REPO_LEVEL_STAGE, Stage, weights
from corpus.tokenizer import Tokenizer
from model.modeling import CausalLM, model_config, num_non_embedding_params
from train.schedule import RunPlan, plan


def drawn(stage: Stage, run: RunPlan) -> int:
    """The rows the run draws in the stage: its steps there x micro-batches x rows per micro-batch."""
    steps = run.max_steps - run.repo_level_start if stage is REPO_LEVEL_STAGE else run.repo_level_start
    return steps * run.gradient_accumulation_steps * stage.micro_batch_size


def stage_batches(tokenizer: Tokenizer, *, stage: Stage, run: RunPlan, seed: int, device: torch.device, start: int
                  ) -> tuple[dict[str, float], Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                          np.ndarray]]]:
    """({corpus: weight}, the stage's training micro-batches from run row `start` on): each corpus's packed rows
    permuted once by `seed` (RandomState's stream is frozen across numpy versions, so a resume sees the same order),
    FIM pieces read as FIM examples. A run that would overdraw a corpus raises (max_tokens_per_param bounds the
    tokens per parameter)."""
    rows = stage_rows(tokenizer, stage)
    mixture, _, over = weights(stage, rows, drawn(stage, run))
    if over:
        raise RuntimeError(
            f"the run draws {over} rows beyond the T{stage.seq_len} packs (train/stages.py max_tokens_per_param)")
    orders = {corpus: np.random.RandomState(seed).permutation(rows[corpus]) for corpus in mixture}
    return mixture, iter(PretrainingBatches(mixture, "train", tokenizer, orders, seq_len=stage.seq_len,
                                            micro_batch_size=stage.micro_batch_size, device=device, reading="fim",
                                            start=start))


def max_tokens_per_param(tokenizer: Tokenizer, *, num_hidden_layers: int) -> float:
    """Bisection over whole micro-batches of tokens on "overdraws". Not monotone (gradient_accumulation_steps steps
    up with the count, so trained tokens dip): the result fits, but a larger fitting count may exist."""
    with torch.device("meta"):
        model = CausalLM(model_config(num_hidden_layers=num_hidden_layers, stage=FILE_LEVEL_STAGE))
    params = num_non_embedding_params(model)
    rows = [(stage, stage_rows(tokenizer, stage)) for stage in (FILE_LEVEL_STAGE, REPO_LEVEL_STAGE)]

    def tokens_per_param(n: int) -> float:
        return n * MICRO_BATCH_TOKENS / params

    def overdrawn(n: int) -> bool:
        if n == 0:  # no tokens overdraw nothing (and plan needs a positive count)
            return False
        run = plan(model=model, tokens_per_param=tokens_per_param(n))
        return any(weights(stage, held, drawn(stage, run))[2] for stage, held in rows if drawn(stage, run))

    n = bisect.bisect_left(range(sum(rows[0][1].values()) * FILE_LEVEL_STAGE.seq_len // MICRO_BATCH_TOKENS + 1), True,
                           key=overdrawn)
    if n == 1:
        raise ValueError(f"one micro-batch ({MICRO_BATCH_TOKENS} tokens) already overdraws the data")
    return tokens_per_param(n - 1)
