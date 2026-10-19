"""
The packed rows each training stage of a run reads (config.Stage, train/schedule.py RunPlan). No document is read twice:
the repo-level stage reads, of each of its corpora, the fewest trailing curated shards (shuffle partitions by repository,
data/prepare.py curate_partition) whose 32K rows its mixture draws from that corpus over the stage; the file-level stage
reads every other shard. A run fits the data when no stage draws more rows of a corpus than it may read (data/loader.py
PretrainingBatches stops at an exhausted corpus); max_ratio is the most tokens per scaling parameter that fits.

python -m nanochat.train.stages <depth> prints max_ratio.
"""

import bisect
import sys

import numpy as np

from nanochat.config import FILE_STAGE, MICRO_BATCH_TOKENS, REPO_STAGE
from nanochat.data.loader import mixture_rows, mixture_weights, packed_rows, row_shards
from nanochat.data.tokenizer import get_tokenizer
from nanochat.train.schedule import plan


def stage_mixture(stage, tokenizer):
    """{corpus: row weight} of a stage (data/loader.py mixture_weights at its rows)."""
    return mixture_weights(stage.mixture, tokenizer, stage.seq_len)


def drawn_rows(stage, run, tokenizer):
    """{corpus: rows a stage of the run draws}: its steps x micro-batches x rows per micro-batch, by mixture_rows."""
    steps = run.repo_start if stage is FILE_STAGE else run.num_iterations - run.repo_start
    mixture = stage_mixture(stage, tokenizer)
    count = steps * run.grad_accum_steps * stage.batch_rows
    drawn = np.bincount(mixture_rows(np.array(list(mixture.values())), start=0, count=count), minlength=len(mixture))
    return dict(zip(mixture, drawn.tolist(), strict=True))


def repo_first_shards(run, tokenizer):
    """{repository corpus: the first of the trailing curated shards the repo-level stage of the run reads}: the fewest
    whose 32K rows hold the rows its mixture draws from that corpus."""
    first = {}
    for corpus, need in drawn_rows(REPO_STAGE, run, tokenizer).items():
        per_shard = np.bincount(row_shards(corpus, "train", tokenizer, REPO_STAGE.seq_len))
        first[corpus] = len(per_shard) - 1 - int(np.searchsorted(np.cumsum(per_shard[::-1]), need))
    return first


def readable_rows(stage, run, tokenizer):
    """{corpus: the rows a stage of the run may read}."""
    first = repo_first_shards(run, tokenizer)
    rows = {}
    for corpus in stage_mixture(stage, tokenizer):
        if corpus not in first:
            rows[corpus] = np.arange(packed_rows(corpus, "train", tokenizer, stage.seq_len))
            continue
        shards = row_shards(corpus, "train", tokenizer, stage.seq_len)
        rows[corpus] = np.flatnonzero(shards >= first[corpus] if stage is REPO_STAGE else shards < first[corpus])
    return rows


def fits(run, tokenizer):
    """Whether every stage draws at most the rows of each corpus it may read."""
    for stage in (FILE_STAGE, REPO_STAGE):
        rows = readable_rows(stage, run, tokenizer)
        if any(drawn > len(rows[corpus]) for corpus, drawn in drawn_rows(stage, run, tokenizer).items()):
            return False
    return True


def max_ratio(depth, tokenizer):
    """The most tokens per scaling parameter a run of this depth fits, in whole micro-batches of tokens (finer than a
    step, the unit a run's trained tokens move in): bisect.bisect_left over the micro-batch counts up to every train row
    of the file-level stage's corpora, on the predicate "does not fit". The predicate is not monotone in the count (a
    run trains floor(n / g) x g micro-batches with g = grad_accum_steps rising with n, so the trained tokens dip where g
    steps up): the ratio returned fits, but a larger count that also fits may exist."""
    params = plan(depth, 1.0).scaling_params
    rows = sum(packed_rows(c, "train", tokenizer, FILE_STAGE.seq_len) for c in stage_mixture(FILE_STAGE, tokenizer))

    def ratio(n):
        return n * MICRO_BATCH_TOKENS / params

    n = bisect.bisect_left(range(rows * FILE_STAGE.seq_len // MICRO_BATCH_TOKENS + 1), True,
                           key=lambda n: not fits(plan(depth, ratio(n)), tokenizer))
    return ratio(n - 1)


if __name__ == "__main__":
    print(repr(max_ratio(int(sys.argv[1]), get_tokenizer())))
