"""
Micro-batches of compiled token rows (compile.py), in a deterministic order that does not depend on the number of GPUs.

A compiled split is one file of packed rows of seq_len + 1 tokens (inputs row[:-1], targets row[1:]), memory-mapped in
place. The global row sequence runs through the rows epoch after epoch, each epoch in a permutation seeded by
(seed, epoch), or in stored order (row_order). It is consumed in micro-batches of world_size * batch_rows rows and rank
r takes the r-th slice of each, so an optimizer step covers the same global rows at any world size, and a run resumes,
on any number of GPUs, from the number of rows consumed.
"""

import os
import itertools

import numpy as np
import torch

from nanochat.data.storage import get_data_dir


def compiled_path(dataset, split, seq_len, tokenizer):
    """Path of a compiled split, keyed by corpus, sequence length and tokenizer."""
    return os.path.join(get_data_dir(), "compiled", f"{dataset}-T{seq_len}-{tokenizer.fingerprint()}", f"{split}.bin")


def token_dtype(tokenizer):
    """The smallest unsigned integer type that holds every token id."""
    return np.min_scalar_type(tokenizer.get_vocab_size() - 1)


def row_order(num_rows, seed, start):
    """Row ids of the global sequence from position start: the rows epoch after epoch, each epoch permuted by
    (seed, epoch), or in stored order without a seed."""
    epoch, offset = divmod(start, num_rows)
    for epoch in itertools.count(epoch):
        # RandomState: its streams are frozen across numpy versions, so a resumed run sees the same order
        order = np.arange(num_rows) if seed is None else np.random.RandomState([seed, epoch]).permutation(num_rows)
        yield from order[offset:]
        offset = 0


class PretrainingBatches:
    """Re-iterable (inputs int32, targets int64) micro-batches on device of a compiled corpus, from global row
    start_row."""

    def __init__(self, dataset, split, tokenizer, *, seq_len, batch_rows, device, rank, world_size, start_row=0, seed=None):
        self.rows = np.memmap(compiled_path(dataset, split, seq_len, tokenizer), dtype=token_dtype(tokenizer), mode="r").reshape(-1, seq_len + 1)
        self.batch_rows, self.device, self.rank, self.world_size = batch_rows, device, rank, world_size
        self.start_row, self.seed = start_row, seed

    def __iter__(self):
        order = row_order(len(self.rows), self.seed, self.start_row)
        first = self.rank * self.batch_rows
        while True:
            # every rank draws the whole micro-batch, so the global sequence advances alike on all ranks
            ids = list(itertools.islice(order, self.world_size * self.batch_rows))[first:first + self.batch_rows]
            batch = torch.from_numpy(self.rows[ids].astype(np.int32))
            batch = batch.pin_memory().to(self.device, non_blocking=True)  # page-locked, so the copy is asynchronous
            yield batch[:, :-1].contiguous(), batch[:, 1:].to(torch.int64)
