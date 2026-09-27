"""
Micro-batches of compiled token rows (compile.py) from a weighted mixture of corpora, in an order that does not depend
on the number of GPUs.

A compiled split is a memory-mapped file of rows of seq_len + 1 tokens (inputs row[:-1], targets row[1:]). Global row
position k goes to the corpus whose cumulative-weight interval holds frac((k + 1) * PHI), a low-discrepancy sequence,
so every stretch of the run follows the weights, and a corpus's i-th position takes the i-th row of its own sequence
(row_order). Rank r takes the r-th slice of each micro-batch of world_size * batch_rows global rows, so a step covers
the same rows at any world size, and a run resumes, on any number of GPUs, from the number of rows consumed.
"""

import os
import itertools

import numpy as np
import torch

from nanochat.data.storage import get_data_dir

# the golden ratio conjugate: of all Weyl sequences, its multiples modulo 1 are the most evenly spread
PHI = (np.sqrt(5) - 1) / 2
# positions per pass of mixture_rows, which bounds its temporaries to 16 MiB for any span (~16 bytes per position)
CHUNK = 1 << 20


def compiled_path(dataset, split, seq_len, tokenizer):
    """Path of a compiled split, keyed by corpus, sequence length and tokenizer."""
    return os.path.join(get_data_dir(), "compiled", f"{dataset}-T{seq_len}-{tokenizer.fingerprint()}", f"{split}.bin")


def token_dtype(tokenizer):
    """The smallest unsigned integer type that holds every token id."""
    return np.min_scalar_type(tokenizer.get_vocab_size() - 1)


def mixture_schedule(weights, start, stop):
    """The corpus of each global row position in [start, stop). Only the interior cumulative weights are searched: the
    last is 1 up to rounding."""
    return np.searchsorted(np.cumsum(weights)[:-1], (np.arange(start + 1, stop + 1) * PHI) % 1, side="right")


def mixture_rows(weights, start, stop):
    """Rows each corpus takes over global row positions [start, stop)."""
    rows = np.zeros(len(weights), dtype=np.int64)
    for chunk in range(start, stop, CHUNK):
        rows += np.bincount(mixture_schedule(weights, chunk, min(chunk + CHUNK, stop)), minlength=len(weights))
    return rows


def row_order(num_rows, seed, start):
    """Row ids of a corpus's own sequence from position start: its rows epoch after epoch, each epoch permuted by
    (seed, epoch), or in stored order without a seed."""
    epoch, offset = divmod(start, num_rows)
    for epoch in itertools.count(epoch):
        # RandomState: its streams are frozen across numpy versions, so a resumed run sees the same order
        order = np.arange(num_rows) if seed is None else np.random.RandomState([seed, epoch]).permutation(num_rows)
        yield from order[offset:]
        offset = 0


class PretrainingBatches:
    """Re-iterable (inputs int32, targets int64) micro-batches on device of a mixture [(compiled corpus, weight)],
    weights summing to 1, from global row start_row."""

    def __init__(self, mixture, split, tokenizer, *, seq_len, batch_rows, device, rank, world_size, start_row=0, seed=None):
        self.weights = np.array([weight for _, weight in mixture])
        self.rows = [np.memmap(compiled_path(name, split, seq_len, tokenizer), dtype=token_dtype(tokenizer), mode="r").reshape(-1, seq_len + 1)
                     for name, _ in mixture]
        self.batch_rows, self.device, self.rank, self.world_size = batch_rows, device, rank, world_size
        self.start_row, self.seed = start_row, seed

    def __iter__(self):
        taken = mixture_rows(self.weights, 0, self.start_row)  # rows each corpus gave before start_row
        orders = [row_order(len(rows), self.seed, start) for rows, start in zip(self.rows, taken)]
        micro_rows = self.world_size * self.batch_rows
        for start in itertools.count(self.start_row, micro_rows):
            # every rank draws the whole micro-batch, so every corpus's sequence advances alike on all ranks
            drawn = [(c, next(orders[c])) for c in mixture_schedule(self.weights, start, start + micro_rows)]
            mine = drawn[self.rank * self.batch_rows:(self.rank + 1) * self.batch_rows]
            batch = torch.from_numpy(np.stack([self.rows[c][i] for c, i in mine], dtype=np.int32))
            batch = batch.pin_memory().to(self.device, non_blocking=True)  # page-locked, so the copy is asynchronous
            yield batch[:, :-1].contiguous(), batch[:, 1:].to(torch.int64)
