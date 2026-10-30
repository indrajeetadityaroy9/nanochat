"""
A run's training rows per stage: each corpus's packed rows, drawn by the stage's mixture weights; the most tokens per
parameter bisection finds that overdraws no corpus; an SFT run's held-out rows.
"""

import bisect
from collections.abc import Iterator

import numpy as np
import torch

from corpus.layout import packed_rows, stage_rows
from corpus.loader import PretrainingBatches
from corpus.mixture import FILE_LEVEL_STAGE, MICRO_BATCH_TOKENS, REPO_LEVEL_STAGE, Stage, weights
from corpus.tokenizer import Tokenizer
from model.modeling import meta_model, num_non_embedding_params
from train.schedule import RunPlan, plan


def drawn(stage: Stage, run: RunPlan) -> int:
    """The rows the run draws in the stage: its steps there x micro-batches x rows per micro-batch."""
    steps = run.max_steps - run.repo_level_start if stage is run.last_stage else run.repo_level_start
    return steps * run.gradient_accumulation_steps * stage.micro_batch_size


def stage_batches(tokenizer: Tokenizer, *, stage: Stage, run: RunPlan, seed: int, device: torch.device, start: int
                  ) -> tuple[dict[str, float], Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                          np.ndarray]]]:
    """({corpus: weight} of the run's draw in the stage, the stage's micro-batches from run row `start` on), each
    corpus's rows permuted by `seed` (RandomState: stream frozen across numpy versions, so a resume sees the same
    order); a corpus drawn beyond its rows (an SFT run of more than one epoch) is read on through further epochs,
    epoch e a fresh permutation by seed + e, so no row repeats within an epoch."""
    rows = stage_rows(tokenizer, stage)
    mixture, draws, _ = weights(stage, rows, drawn(stage, run))
    orders = {corpus: np.concatenate([np.random.RandomState(seed + e).permutation(rows[corpus])
                                      for e in range(max(1, -(-draws[corpus] // rows[corpus])))])
              for corpus in mixture}
    return mixture, iter(PretrainingBatches(mixture, "train", tokenizer, orders, seq_len=stage.seq_len,
                                            micro_batch_size=stage.micro_batch_size, device=device, reading="fim",
                                            start=start))


def val_batches(tokenizer: Tokenizer, *, stage: Stage, device: torch.device) -> PretrainingBatches:
    """Every row of the val split of a one-corpus stage (SFT_STAGE's held-out prompts) in order, the last micro-batch
    filled with blank rows."""
    (corpus,) = stage.sources
    rows = packed_rows(corpus, "val", tokenizer, stage.seq_len)
    order = np.concatenate([np.arange(rows), np.full(-rows % stage.micro_batch_size, -1)])
    return PretrainingBatches({corpus: 1.0}, "val", tokenizer, {corpus: order}, seq_len=stage.seq_len,
                              micro_batch_size=stage.micro_batch_size, device=device, reading="left_to_right", start=0)


def max_tokens_per_param(tokenizer: Tokenizer) -> float:
    """Bisection over whole micro-batches of tokens on "overdraws". Not monotone (gradient_accumulation_steps steps
    up with the count, so trained tokens dip): the result fits, but a larger fitting count may exist."""
    model = meta_model()
    params = num_non_embedding_params(model)
    rows = [(stage, stage_rows(tokenizer, stage)) for stage in (FILE_LEVEL_STAGE, REPO_LEVEL_STAGE)]

    def tokens_per_param(n: int) -> float:
        return n * MICRO_BATCH_TOKENS / params

    def overdrawn(n: int) -> bool:
        run = plan(model=model, tokens_per_param=tokens_per_param(n))
        return any(weights(stage, held, drawn(stage, run))[2] for stage, held in rows)

    n = bisect.bisect_left(range(sum(rows[0][1].values()) * FILE_LEVEL_STAGE.seq_len // MICRO_BATCH_TOKENS + 1), True,
                           key=overdrawn)
    return tokens_per_param(n - 1)
