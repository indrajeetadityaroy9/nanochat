"""
Micro-batches of compiled token rows (compile.py) of one corpus or a weighted mixture of corpora, in a deterministic
order that does not depend on the number of GPUs.

A compiled split is one file of packed rows of seq_len + 1 tokens (inputs row[:-1], targets row[1:]), memory-mapped in
place. Each corpus's row sequence is one pass over its rows, permuted by the seed, or in stored order (row_order): no row
is read twice, so a run needs as many compiled rows of each corpus as it reads, and require() fails before training,
with the files to compile, when a corpus holds fewer. A run interleaves the corpora by their weights (mixture_rows) and
starts each corpus's sequence at the rows earlier runs consumed of it, so a run initialized from another continues every
corpus where that run stopped. The run's rows are consumed in micro-batches of world_size * batch_rows rows and rank r
takes the r-th slice of each, so an optimizer step covers the same global rows at any world size, and a run resumes,
on any number of GPUs, from the number of rows it consumed.
"""

import os
import json
import math

import numpy as np
import torch

from nanochat.data.storage import get_data_dir
from nanochat.data.pretrain.sources import fetch_command

GOLDEN = (math.sqrt(5) - 1) / 2  # the Weyl sequence frac(n * GOLDEN) spreads every prefix of itself most evenly over [0, 1)


def compiled_path(dataset, split, seq_len, tokenizer):
    """Path of a compiled split, keyed by corpus, sequence length and tokenizer."""
    return os.path.join(get_data_dir(), "compiled", f"{dataset}-T{seq_len}-{tokenizer.fingerprint()}", f"{split}.bin")


def token_dtype(tokenizer):
    """The smallest unsigned integer type that holds every token id."""
    return np.min_scalar_type(tokenizer.get_vocab_size() - 1)


def parse_mixture(spec):
    """{corpus: weight} of a corpus name or 'name:weight,name:weight,...', the weights normalized to sum to 1."""
    parts = [part.split(":") for part in spec.split(",")]
    weights = {part[0]: float(part[1]) if len(part) == 2 else 1.0 for part in parts}
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()}


def mixture_rows(weights, start, count):
    """Corpus index (into weights) of each run row start..start+count-1: row n goes to the corpus whose share of the
    cumulative weights holds the Weyl point frac((n + 1) * GOLDEN), so every prefix of the run holds each corpus in
    proportion to its weight, and any stretch of the run is computed without the rows before it."""
    bounds = np.cumsum(weights)
    bounds /= bounds[-1]
    points = np.modf(np.arange(start + 1, start + count + 1) * GOLDEN)[0]
    return np.searchsorted(bounds, points, side="right")


def row_order(num_rows, seed, start):
    """Row ids of a corpus's row sequence from position start: its rows once, permuted by seed, or in stored order
    without a seed."""
    # RandomState: its streams are frozen across numpy versions, so a resumed run sees the same order
    order = np.arange(num_rows) if seed is None else np.random.RandomState(seed).permutation(num_rows)
    return iter(order[start:])


class PretrainingBatches:
    """Re-iterable (inputs int32, targets int64) micro-batches on device of a mixture {corpus: weight} of compiled
    corpora, from row run_start of the run; start: {corpus: rows of its row sequence consumed before the run}."""

    def __init__(self, mixture, split, tokenizer, *, seq_len, batch_rows, device, rank, world_size, start=None, run_start=0, seed=None):
        self.names, self.weights, self.split = list(mixture), np.array(list(mixture.values())), split
        self.paths = {name: compiled_path(name, split, seq_len, tokenizer) for name in self.names}
        self.rows = {name: np.memmap(path, dtype=token_dtype(tokenizer), mode="r").reshape(-1, seq_len + 1)
                     for name, path in self.paths.items()}
        self.batch_rows, self.device, self.rank, self.world_size = batch_rows, device, rank, world_size
        self.start, self.run_start, self.seed = start or {}, run_start, seed

    def rows_after(self, run_rows):
        """{corpus: rows of its row sequence consumed} once the run has consumed run_rows rows."""
        counts = np.bincount(mixture_rows(self.weights, 0, run_rows), minlength=len(self.names))
        return {name: self.start.get(name, 0) + int(count) for name, count in zip(self.names, counts)}

    def shortfall(self, name, rows):
        """The error for a corpus that holds fewer compiled rows than the rows of it a run reads."""
        have = len(self.rows[name])
        if self.split == "val":
            return ValueError(f"'{name}' val holds {have:,} rows, fewer than the {rows:,} to evaluate: its val is its "
                              f"held-out file(s), so evaluate fewer tokens")
        files = len(json.load(open(os.path.splitext(self.paths[name])[0] + ".json"))["files"])
        needed = math.ceil(rows / have * files)  # at its files' mean rows
        return ValueError(f"'{name}' {self.split} holds {have:,} rows compiled from {files} files, fewer than the {rows:,} "
                          f"the run reads, each once: fetch and compile about {needed} files: `{fetch_command(name, needed)}`, "
                          f"then `python -m nanochat.data.pretrain.compile --dataset={name} --max-files={needed}`")

    def require(self, run_rows):
        """Fail unless every corpus holds the rows the run has read of it once it has consumed run_rows rows."""
        for name, rows in self.rows_after(run_rows).items():
            if rows > len(self.rows[name]):
                raise self.shortfall(name, rows)

    def __iter__(self):
        consumed = self.rows_after(self.run_start)
        orders = [row_order(len(self.rows[name]), self.seed, consumed[name]) for name in self.names]
        position, first = self.run_start, self.rank * self.batch_rows
        while True:
            # every rank draws the whole micro-batch, so every corpus's sequence advances alike on all ranks
            corpora = mixture_rows(self.weights, position, self.world_size * self.batch_rows)
            position += len(corpora)
            ids = [next(orders[c], None) for c in corpora]
            if None in ids:  # a caller that skipped require() read past a corpus's rows
                name = self.names[corpora[ids.index(None)]]
                raise self.shortfall(name, self.rows_after(position)[name])
            rows = [self.rows[self.names[c]][i] for c, i in zip(corpora, ids)][first:first + self.batch_rows]
            batch = torch.from_numpy(np.stack(rows).astype(np.int32))
            batch = batch.pin_memory().to(self.device, non_blocking=True)  # page-locked, so the copy is asynchronous
            yield batch[:, :-1].contiguous(), batch[:, 1:].to(torch.int64)
