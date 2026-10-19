"""
Bits per byte of a model on each corpus's packed val split: the summed loss (nats) of the target tokens over (ln 2 x
their summed length in bytes), a vocabulary-independent metric, both sums fp64 on the device (no host synchronization
per batch; a NaN loss propagates to the result).
"""
import math
from itertools import islice

import numpy as np
import torch

from nanochat.data.loader import PretrainingBatches, packed_rows

# The val sample per corpus: nanochat's evaluation budget, --eval-tokens 80 x 2^19 tokens (karpathy/nanochat 92d63d4
# scripts/base_train.py:73), or the whole split when it is smaller
VAL_TOKENS = 80 * 2**19


@torch.no_grad()
def evaluate_bpb(losses, batches, steps, token_bytes, num_segments):
    """Bits per byte of the first `steps` batches (the loader's (inputs, targets, cu_seqlens)) under losses (the model's
    per-position losses, GPT.losses). token_bytes (vocab_size,): each token's length in bytes, 0 for special tokens,
    which are therefore excluded; padding targets count 0 bytes and lose 0. cu_seqlens is padded to num_segments with
    empty segments (its last offset repeated, as the loader's unused entries), so loaders of different corpora give the
    compiled model one shape."""
    nats, nbytes = (torch.zeros((), dtype=torch.float64, device=token_bytes.device) for _ in range(2))
    for x, y, cu_seqlens in islice(batches, steps):
        cu_seqlens = torch.cat([cu_seqlens, cu_seqlens[-1:].expand(num_segments + 1 - cu_seqlens.numel())])
        y = y.flatten()
        num_bytes = token_bytes[y] * (y != PretrainingBatches.PAD)
        nats += (losses(x, y.view_as(x), cu_seqlens) * (num_bytes > 0)).double().sum()
        nbytes += num_bytes.double().sum()
    return (nats / (math.log(2) * nbytes)).item()


def sample_rows(num_rows, *, batch_rows, seq_len):
    """Row ids of a split's systematic sample: n = ceil(VAL_TOKENS / (batch_rows x seq_len)) x batch_rows rows (all
    full micro-batches of a smaller split) (j x num_rows) // n for j < n, spread evenly over the whole split. The packer
    orders each shard's rows longest pieces first (Best-Fit-Decreasing), so a prefix of the split over-represents long
    documents; an even stride samples every position of every shard."""
    n = min(num_rows // batch_rows, -(-VAL_TOKENS // (batch_rows * seq_len))) * batch_rows
    return np.arange(n, dtype=np.int64) * num_rows // n


def val_bpb(losses, tokenizer, datasets, *, stage, device):
    """{dataset: bits per byte} of each corpus's packed val split in rows of a training stage's context (config.Stage:
    seq_len, batch_rows per batch), read left to right over a fixed systematic sample of its rows (sample_rows), by
    losses (GPT.losses, the training forward's per-position losses). Every corpus's batches share one shape (cu_seqlens
    padded to the corpora's most segments): one compiled graph scores all."""
    token_bytes = tokenizer.get_token_bytes(device=device)
    samples = {name: sample_rows(packed_rows(name, "val", tokenizer, stage.seq_len), batch_rows=stage.batch_rows,
                                 seq_len=stage.seq_len) for name in datasets}
    loaders = {name: PretrainingBatches({name: 1.0}, "val", tokenizer, {name: rows}, seq_len=stage.seq_len,
                                        batch_rows=stage.batch_rows, device=device, reading="left_to_right", start=0)
               for name, rows in samples.items()}
    num_segments = max(loader.num_segments for loader in loaders.values())
    return {name: evaluate_bpb(losses, loader, len(samples[name]) // stage.batch_rows, token_bytes, num_segments)
            for name, loader in loaders.items()}
