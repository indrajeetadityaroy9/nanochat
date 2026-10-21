"""
Packed rows each stage reads, no document twice: per repository corpus, the repo-level stage reads the fewest trailing
curated shards (shuffled by repository, data/curate.py curate_partition) holding the rows it draws, among the trailing
shards data/pack.py packs at its rows; the file-level stage reads the rest. max_ratio: the most tokens per scaling
parameter at which no stage draws more rows of a corpus than it may read (data/loader.py PretrainingBatches stops at an
exhausted corpus). stage_batches: a stage's training micro-batches in the run's data order.
"""

import bisect
from collections.abc import Iterator

import numpy as np
import torch

from nanochat.config import FILE_STAGE, MICRO_BATCH_TOKENS, REPO_STAGE, Stage
from nanochat.data.loader import PretrainingBatches, mixture_rows, mixture_weights, packed_rows, row_shards
from nanochat.data.tokenizer import Tokenizer
from nanochat.model.gpt import GPT, gpt_config, scaling_params
from nanochat.train.schedule import RunPlan, plan


def drawn_rows(tokenizer: Tokenizer, *, stage: Stage, run: RunPlan) -> dict[str, int]:
    """{corpus: rows the stage (FILE_STAGE or REPO_STAGE) draws}: steps x micro-batches x rows per micro-batch, split
    by mixture_rows."""
    steps = run.num_iterations - run.repo_start if stage is REPO_STAGE else run.repo_start
    mixture = mixture_weights(stage.mixture, tokenizer, stage.seq_len)
    count = steps * run.grad_accum_steps * stage.batch_rows
    drawn = np.bincount(mixture_rows(np.array(list(mixture.values())), start=0, count=count), minlength=len(mixture))
    return dict(zip(mixture, drawn.tolist(), strict=True))


def readable_rows(tokenizer: Tokenizer, *, stage: Stage, run: RunPlan) -> dict[str, np.ndarray]:
    """{corpus: rows the stage may read}."""
    rows = {}
    for corpus, need in drawn_rows(tokenizer, stage=REPO_STAGE, run=run).items():
        per_shard = np.bincount(row_shards(corpus, "train", tokenizer, REPO_STAGE.seq_len))
        first = len(per_shard) - 1 - int(np.searchsorted(np.cumsum(per_shard[::-1]), need))
        shards = row_shards(corpus, "train", tokenizer, stage.seq_len)
        rows[corpus] = np.flatnonzero(shards >= first if stage is REPO_STAGE else shards < first)
    return {corpus: rows[corpus] if corpus in rows else np.arange(packed_rows(corpus, "train", tokenizer, stage.seq_len))
            for corpus in mixture_weights(stage.mixture, tokenizer, stage.seq_len)}


def stage_batches(tokenizer: Tokenizer, *, stage: Stage, run: RunPlan, seed: int, device: torch.device,
                  start: int) -> tuple[dict[str, float], Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """({corpus: weight}, the stage's training micro-batches from run row `start` on): the rows it may read, each
    corpus's permuted once by `seed` (RandomState's stream is frozen across numpy versions, so a resume sees the same
    order), FIM pieces read as FIM examples."""
    mixture = mixture_weights(stage.mixture, tokenizer, stage.seq_len)
    readable = readable_rows(tokenizer, stage=stage, run=run)
    orders = {corpus: np.random.RandomState(seed).permutation(readable[corpus]) for corpus in mixture}
    return mixture, iter(PretrainingBatches(mixture, "train", tokenizer, orders, seq_len=stage.seq_len,
                                            batch_rows=stage.batch_rows, device=device, reading="fim", start=start))


def max_ratio(tokenizer: Tokenizer, *, depth: int) -> float:
    """Bisection over whole micro-batches of tokens on "overdraws". Not monotone (grad_accum_steps steps up with the
    count, so trained tokens dip): the result fits, but a larger fitting count may exist."""
    with torch.device("meta"):
        model = GPT(gpt_config(depth=depth, stage=FILE_STAGE))
    params = scaling_params(model)
    rows = sum(packed_rows(c, "train", tokenizer, FILE_STAGE.seq_len)
               for c in mixture_weights(FILE_STAGE.mixture, tokenizer, FILE_STAGE.seq_len))

    def ratio(n):
        return n * MICRO_BATCH_TOKENS / params

    def overdrawn(n):
        run = plan(model=model, ratio=ratio(n))
        for stage in (FILE_STAGE, REPO_STAGE):
            readable = readable_rows(tokenizer, stage=stage, run=run)
            if any(drawn > len(readable[c]) for c, drawn in drawn_rows(tokenizer, stage=stage, run=run).items()):
                return True
        return False

    n = bisect.bisect_left(range(rows * FILE_STAGE.seq_len // MICRO_BATCH_TOKENS + 1), True, key=overdrawn)
    return ratio(n - 1)
