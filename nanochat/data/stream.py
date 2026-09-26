"""
Loader for compiled token shards (shards.py): deterministic, elastic, and cheap on the CPU.

Training reads the node's compiled directory in place (put <base_dir>/data on fast local NVMe):
shards are memory-mapped, so the OS page cache holds the hot data and nothing is decoded.

Order. Each epoch permutes the shard order and shuffles rows within consecutive blocks of
BLOCK_SHARDS shards: rows mix across many shards, while each worker only maps the block it reads.
The global row sequence depends only on the seed and the index, never on the number of GPUs,
nodes or workers.

Elasticity. The global sequence is consumed in micro-batches of world_size * batch_rows rows and rank
r takes the r-th slice of each, so an optimizer step of rows_per_step rows always covers global rows
[step * rows_per_step, (step + 1) * rows_per_step) whatever the world size. A run resumes, on any
number of GPUs, from the number of rows consumed.

Throughput. DataLoader worker processes assemble each micro-batch by gathering rows from mmap'd
shards (no tokenization, no packing); the DataLoader pins it and the copy to the GPU is asynchronous.
"""

import os
import json
import hashlib

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from nanochat.data.shards import INDEX_FILE, check_index, get_compiled_dir, open_shard

BLOCK_SHARDS = 16 # shards per shuffle block: rows mix within a block, and a worker maps one block at a time


def _seed(*parts):
    """32-bit seed for np.random.RandomState, whose streams are frozen across numpy versions."""
    return int.from_bytes(hashlib.blake2b("/".join(map(str, parts)).encode(), digest_size=4).digest(), "little")


class _EpochOrder:
    """Row order of one epoch: shards permuted and cut into blocks, rows shuffled within each block."""

    def __init__(self, shard_rows, epoch, seed, shuffle, block_shards):
        rows = np.asarray(shard_rows, dtype=np.int64)
        order = np.random.RandomState(_seed(seed, epoch)).permutation(len(rows)) if shuffle else np.arange(len(rows))
        self.blocks = [order[i:i + block_shards] for i in range(0, len(order), block_shards)]
        self.block_start = np.concatenate([[0], np.cumsum([rows[b].sum() for b in self.blocks])])
        self.shard_start = [np.concatenate([[0], np.cumsum(rows[b])]) for b in self.blocks]
        self.seed, self.epoch, self.shuffle = seed, epoch, shuffle
        self._perms = {}

    def blocks_of(self, offsets):
        return np.searchsorted(self.block_start, offsets, side="right") - 1

    def locate(self, block, offsets):
        """(shard ids, rows within those shards) for row offsets of this epoch that fall in one block."""
        local = offsets - self.block_start[block]
        if self.shuffle:
            local = self._perm(block)[local]
        which = np.searchsorted(self.shard_start[block], local, side="right") - 1
        return self.blocks[block][which], local - self.shard_start[block][which]

    def _perm(self, block):
        if block not in self._perms:
            if len(self._perms) == 2:
                self._perms.pop(next(iter(self._perms)))
            size = int(self.block_start[block + 1] - self.block_start[block])
            self._perms[block] = np.random.RandomState(_seed(self.seed, self.epoch, block)).permutation(size)
        return self._perms[block]


class TokenStream(IterableDataset):
    """One rank's micro-batches of token rows, shape (batch_rows, row_len), in the global deterministic order."""

    def __init__(self, index, split_dir, *, batch_rows, rank=0, world_size=1, start_row=0, shuffle=True, seed=42, block_shards=BLOCK_SHARDS):
        stride = world_size * batch_rows
        assert start_row % stride == 0, f"start_row {start_row:,} is not a multiple of world_size * batch_rows = {stride}"
        self.shards = index["shards"]
        self.total_rows = index["rows"]
        self.dtype, self.row_len = np.dtype(index["dtype"]), index["row_len"]
        self.split_dir = split_dir
        self.batch_rows, self.rank, self.world_size = batch_rows, rank, world_size
        self.first_batch = start_row // stride
        self.shuffle, self.seed, self.block_shards = shuffle, seed, block_shards

    def __iter__(self):
        worker = get_worker_info()
        worker_id, num_workers = (worker.id, worker.num_workers) if worker is not None else (0, 1)
        reader = _Reader(self)
        batch = self.first_batch + worker_id # the DataLoader interleaves its workers in order
        while True:
            start = (batch * self.world_size + self.rank) * self.batch_rows
            yield reader.read(np.arange(start, start + self.batch_rows, dtype=np.int64))
            batch += num_workers


class _Reader:
    """Per-worker state: epoch orders, and memmaps of the shards of the block being read."""

    def __init__(self, stream):
        self.s = stream
        self.epochs = {}
        self.maps = {} # shard id -> memmap
        self.block = None # (epoch, block) being read

    def read(self, positions):
        s = self.s
        out = np.empty((len(positions), s.row_len), dtype=s.dtype)
        epochs, offsets = np.divmod(positions, s.total_rows)
        for epoch in np.unique(epochs):
            order = self._order(int(epoch))
            in_epoch = np.flatnonzero(epochs == epoch)
            blocks = order.blocks_of(offsets[in_epoch])
            for block in np.unique(blocks):
                self._enter(int(epoch), int(block))
                sel = in_epoch[blocks == block]
                shard_ids, rows = order.locate(int(block), offsets[sel])
                for shard in np.unique(shard_ids):
                    mask = shard_ids == shard
                    out[sel[mask]] = self._map(int(shard))[rows[mask]]
        return torch.from_numpy(out.astype(np.int32))

    def _order(self, epoch):
        if epoch not in self.epochs:
            if len(self.epochs) == 3:
                self.epochs.pop(min(self.epochs))
            self.epochs[epoch] = _EpochOrder([sh["rows"] for sh in self.s.shards], epoch, self.s.seed, self.s.shuffle, self.s.block_shards)
        return self.epochs[epoch]

    def _enter(self, epoch, block):
        """Unmap the shards outside the block now being read: every mapping holds a file descriptor."""
        if self.block == (epoch, block):
            return
        self.block = (epoch, block)
        current = set(int(i) for i in self._order(epoch).blocks[block])
        for shard in [sh for sh in self.maps if sh not in current]:
            del self.maps[shard]

    def _map(self, shard):
        if shard not in self.maps:
            prefix = os.path.join(self.s.split_dir, self.s.shards[shard]["name"])
            rows = open_shard(prefix)
            assert rows.shape == (self.s.shards[shard]["rows"], self.s.row_len) and rows.dtype == self.s.dtype, f"Shard {prefix} does not match the index"
            self.maps[shard] = rows
        return self.maps[shard]


class DeviceBatches:
    """Re-iterable (inputs int32, targets int64) on device, fed by DataLoader workers through pinned memory."""

    def __init__(self, stream, device, num_workers):
        self.device = device
        on_cuda = device.type == "cuda"
        self.loader = DataLoader(
            stream,
            batch_size=None, # the stream yields whole micro-batches
            num_workers=num_workers,
            pin_memory=on_cuda, # page-locked host memory: asynchronous copies to the GPU
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=num_workers > 0, # e.g. the val loader is re-iterated at every eval
            multiprocessing_context="fork" if num_workers > 0 else None, # entry scripts are not import-safe for spawn
        )

    def __iter__(self):
        non_blocking = self.device.type == "cuda"
        for rows in self.loader:
            rows = rows.to(self.device, non_blocking=non_blocking)
            yield rows[:, :-1].contiguous(), rows[:, 1:].to(torch.int64)


def pretraining_batches(dataset, split, tokenizer, *, seq_len, batch_rows, device, rank, world_size, start_row=0,
                        shuffle=True, seed=42, num_workers=2):
    """(batches, index) for one split of a corpus compiled on this node (python -m nanochat.data.compile)."""
    fingerprint = tokenizer.fingerprint()
    split_dir = os.path.join(get_compiled_dir(dataset, seq_len, fingerprint), split)
    index_path = os.path.join(split_dir, INDEX_FILE)
    assert os.path.exists(index_path), f"No compiled data at {split_dir}: run `python -m nanochat.data.compile`"
    with open(index_path) as f:
        index = check_index(json.load(f))
    assert index["seq_len"] == seq_len, f"Compiled for seq_len {index['seq_len']}, training uses {seq_len}"
    assert index["tokenizer"]["fingerprint"] == fingerprint and index["tokenizer"]["bos_id"] == tokenizer.get_bos_token_id(), "Compiled with a different tokenizer"
    stream = TokenStream(index, split_dir, batch_rows=batch_rows, rank=rank, world_size=world_size,
                         start_row=start_row, shuffle=shuffle, seed=seed)
    return DeviceBatches(stream, device, num_workers), index
