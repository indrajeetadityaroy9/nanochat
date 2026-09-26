"""
Micro-batches of compiled token rows (compile.py), in a deterministic order that does not depend on the number of GPUs.

A compiled split is one file of packed rows of seq_len + 1 tokens (inputs row[:-1], targets row[1:]), memory-mapped in
place, so the OS page cache holds the hot data and nothing is decoded. The global row sequence runs through the rows
epoch after epoch, each epoch in a permutation seeded by (seed, epoch), or in stored order. It is consumed in
micro-batches of world_size * batch_rows rows and rank r takes the r-th slice of each, so an optimizer step covers the
same global rows at any world size, and a run resumes, on any number of GPUs, from the number of rows consumed.
"""

import os
import itertools

import numpy as np
import torch

from nanochat.data.storage import get_data_dir


def compiled_path(dataset, split, seq_len, tokenizer):
    """Compiled data is keyed by corpus, sequence length and tokenizer."""
    return os.path.join(get_data_dir(), "compiled", f"{dataset}-T{seq_len}-{tokenizer.fingerprint()}", f"{split}.bin")


def token_dtype(tokenizer):
    """The smallest unsigned integer type that holds every token id."""
    return np.min_scalar_type(tokenizer.get_vocab_size() - 1)


class PretrainingBatches:
    """
    Re-iterable (inputs int32, targets int64) micro-batches on device, from row start_row of the global sequence.
    Without a seed, every epoch reads the rows in stored order.
    """

    def __init__(self, dataset, split, tokenizer, *, seq_len, batch_rows, device, rank, world_size, start_row=0, seed=None):
        path = compiled_path(dataset, split, seq_len, tokenizer)
        self.rows = np.memmap(path, dtype=token_dtype(tokenizer), mode="r").reshape(-1, seq_len + 1)
        self.batch_rows, self.device, self.rank, self.world_size = batch_rows, device, rank, world_size
        self.start_row, self.seed = start_row, seed

    def __iter__(self):
        n = len(self.rows)
        # RandomState: its streams are frozen across numpy versions, so a resumed run sees the same order
        epochs = (range(n) if self.seed is None else np.random.RandomState([self.seed, epoch]).permutation(n) for epoch in itertools.count())
        order = itertools.islice(itertools.chain.from_iterable(epochs), self.start_row, None)
        first = self.rank * self.batch_rows
        while True:
            ids = list(itertools.islice(order, self.world_size * self.batch_rows))[first:first + self.batch_rows]
            batch = torch.from_numpy(self.rows[ids].astype(np.int32))
            if self.device.type == "cuda":
                batch = batch.pin_memory() # page-locked, so the copy to the GPU is asynchronous
            batch = batch.to(self.device, non_blocking=True)
            yield batch[:, :-1].contiguous(), batch[:, 1:].to(torch.int64)
