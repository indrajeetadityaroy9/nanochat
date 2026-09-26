"""
SFT data: a deterministic mixture of chat tasks (TaskMixture), and the loader (SFTLoader) that packs its conversations
with BOS-aligned best-fit packing, padded instead of cropped:
   - Every row starts with BOS, the start of a conversation
   - Conversations are packed largest-fit first and are never split across rows
   - When no conversation fits the rest of a row, it is padded with BOS, so no token is ever discarded
   - Targets are -1 (ignore_index) where the render mask is 0 (prompts, special tokens, tool outputs) and at padding
Conversations are rendered to at most one row (max_seq_len + 1 tokens), so every one of them fits. One whose
prompt fills the row keeps no assistant token to train on, so it is skipped rather than spending a row on nothing.
"""

import random

import torch

from nanochat.data.task import Task


class TaskMixture(Task):
    """
    For SFT Training it becomes useful to train on a mixture of datasets.
    Fun trick: if you wish to oversample any task, just pass it in multiple times in the list.
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        # tasks is a list of Task objects
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)
        # Build list of all (task_idx, local_idx) pairs
        self.index_map = []
        for task_idx, task_length in enumerate(self.lengths):
            for local_idx in range(task_length):
                self.index_map.append((task_idx, local_idx))
        # Deterministically shuffle to mix tasks throughout training
        rng = random.Random(42)
        rng.shuffle(self.index_map)
        # Note: this is not the most elegant or best solution, but it's ok for now

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        """
        Access conversations according to a deterministic shuffle of all examples.
        This ensures tasks are mixed throughout training, regardless of dataset size.
        """
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for mixture with {self.num_conversations} conversations"
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]


class SFTLoader:
    """
    Infinite iterator of (inputs int32, targets int64) batches of shape (batch_size, max_seq_len).
    Ranks shard the conversations by stride. The number of steps of a pass is not known in advance,
    so the loader exposes where it is, as of the most recent batch it returned (a caller that
    prefetches reads these before fetching the next batch):
        progress   0 -> 1 over one pass of the dataset (or over num_batches batches, if > 0), never above 1
        last_step  the pass is complete (or num_batches batches were produced)
        epoch      1-based pass the most recently packed conversation belongs to
    """

    def __init__(self, dataset, tokenizer, *, batch_size, max_seq_len, device, rank, world_size, num_batches=-1, buffer_size=100):
        self.dataset = dataset
        self.num_rows = len(dataset)
        assert self.num_rows > 0
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.world_size = world_size
        self.num_batches = num_batches
        self.buffer_size = buffer_size
        self.pin_memory = torch.device(device).type == "cuda"
        self.bos = tokenizer.get_bos_token_id()
        self.buffer = [] # rendered conversations waiting to be packed, (ids, mask, skipped before it) tuples
        self.cursor = rank # next conversation to fetch, this rank takes every world_size-th one
        self.consumed = rank # the same, but only counting conversations done with: packed into rows, or skipped
        self.steps = 0 # batches produced
        self.progress = 0.0
        self.last_step = False
        self.epoch = 1

    def __iter__(self):
        return self

    def _refill(self):
        while len(self.buffer) < self.buffer_size:
            skipped = 0
            while True:
                conversation = self.dataset[self.cursor]
                self.cursor = (self.cursor + self.world_size) % self.num_rows
                ids, mask = self.tokenizer.render_conversation(conversation, max_tokens=self.max_seq_len + 1)
                if any(mask):
                    break
                skipped += 1
                if skipped >= self.num_rows: # a whole cycle of this rank's conversations
                    raise ValueError(f"No conversation keeps a token to train on within max_seq_len={self.max_seq_len}")
            # skipped conversations count as consumed along with this one, as the buffer runs ahead of packing
            self.buffer.append((ids, mask, skipped))

    def _pack_row(self):
        row_capacity = self.max_seq_len + 1 # +1 for the target at the last position
        row, mask = [], []
        while len(row) < row_capacity:
            self._refill()
            remaining = row_capacity - len(row)
            # best fit: the largest conversation that fits entirely
            best_idx, best_len = -1, 0
            for i, (ids, _, _) in enumerate(self.buffer):
                if best_len < len(ids) <= remaining:
                    best_idx, best_len = i, len(ids)
            if best_idx < 0:
                # nothing fits: pad the remainder (mask 0) instead of cropping a conversation
                row.extend([self.bos] * remaining)
                mask.extend([0] * remaining)
                break
            ids, conv_mask, skipped = self.buffer.pop(best_idx)
            row.extend(ids)
            mask.extend(conv_mask)
            self.consumed += self.world_size * (1 + skipped)
        return row, mask

    def __next__(self):
        rows, masks = zip(*(self._pack_row() for _ in range(self.batch_size)))
        rows = torch.tensor(rows, dtype=torch.long)
        masks = torch.tensor(masks, dtype=torch.bool)
        # inputs and targets share one fresh staging buffer, so a single transfer moves the batch. It is
        # never reused while the async copy may still read it: the pinned allocator waits for the copy.
        batch = torch.empty((2, self.batch_size, self.max_seq_len), dtype=torch.long, pin_memory=self.pin_memory)
        batch[0] = rows[:, :-1]
        batch[1] = rows[:, 1:]
        batch[1].masked_fill_(~masks[:, 1:], -1) # mask[1:] aligns with the targets
        batch = batch.to(self.device, non_blocking=self.pin_memory)
        inputs, targets = batch[0].to(torch.int32), batch[1]

        self.steps += 1
        # progress counts consumed conversations, not fetched ones, to account for the buffer
        progress = self.steps / self.num_batches if self.num_batches > 0 else self.consumed / self.num_rows
        self.progress = min(progress, 1.0) # the pass can end mid-batch: rows after it wrap into the next one
        self.last_step = self.consumed >= self.num_rows or 0 < self.num_batches <= self.steps
        self.epoch = 1 + (self.consumed - self.world_size) // self.num_rows
        return inputs, targets
