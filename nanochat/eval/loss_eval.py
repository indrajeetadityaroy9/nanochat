"""
Bits per byte of a model on each corpus's packed val split: the summed loss (nats) of the target tokens over (ln 2 x
their summed length in bytes), a vocabulary-independent metric, both sums fp64 on the device (torchmetrics SumMetric,
nan_strategy "disable": no host synchronization per batch, and a NaN loss shows in the result instead of being dropped).
"""
import math
from itertools import islice

import numpy as np
import torch
from torchmetrics.aggregation import SumMetric

from nanochat.config import SEQ_LEN
from nanochat.data.loader import PretrainingBatches, packed_rows

# The val sample per corpus: large enough that which rows are scored adds at most a quarter of one run's seed noise
# to a paired difference of two runs on the same rows (independent parts: sqrt(1 + 1/16) = 1.03 x the seed noise).
# - seed noise of val bpb per run: 0.0015 (d8, seed 42 vs 43 of the split-pattern A/B's new pattern: |difference|
#   0.0018 Stack-Edu / 0.0023 Stack v3 val bpb, each / sqrt 2, their mean)
# - paired per-problem difference of two same-config seeds in solution bits per byte: sd 0.13 at the HumanEval
#   reference solutions' median 153 bytes (d4 ratio 4, 5 seed pairs: RedesignEval M5, M16), so 0.13 x sqrt(153) bits
#   per byte at one byte if bytes are independent
# - bytes needed: (0.13 sqrt(153) / (0.0015 / 4))^2; tokens at the tokenizer's held-out 3.9419 bytes per token (every
#   corpus's val split, dev/D20_SPEC.md "Tokenizer")
VAL_TOKENS = math.ceil((0.13 * math.sqrt(153) / (0.0015 / 4)) ** 2 / 3.9419)  # 4,664,553


@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes, num_segments):
    """Bits per byte of the first `steps` batches (the loader's (inputs, targets, cu_seqlens)). token_bytes
    (vocab_size,): each token's length in bytes, 0 for special tokens, which are therefore excluded; targets < 0 count
    0 bytes and lose 0. cu_seqlens is padded to num_segments with empty segments (its last offset repeated, as the
    loader's unused entries), so loaders of different corpora give the compiled model one shape."""
    nats, nbytes = (SumMetric(nan_strategy="disable").set_dtype(torch.float64).to(token_bytes.device) for _ in range(2))
    for x, y, cu_seqlens in islice(batches, steps):
        cu_seqlens = torch.cat([cu_seqlens, cu_seqlens[-1:].expand(num_segments + 1 - cu_seqlens.numel())])
        loss = model(x, y, loss_reduction='none', cu_seqlens=cu_seqlens).view(-1)
        y = y.view(-1)
        num_bytes = token_bytes[y.clamp(min=0)] * (y >= 0)
        nats.update(loss * (num_bytes > 0))
        nbytes.update(num_bytes)
    return (nats.compute() / (math.log(2) * nbytes.compute())).item()


def sample_rows(num_rows, *, batch_rows):
    """Row ids of a split's systematic sample: n = ceil(VAL_TOKENS / (batch_rows x SEQ_LEN)) x batch_rows rows (all
    full micro-batches of a smaller split) (j x num_rows) // n for j < n, spread evenly over the whole split. The packer
    orders each shard's rows longest pieces first (Best-Fit-Decreasing), so a prefix of the split over-represents long
    documents; an even stride samples every position of every shard."""
    n = min(num_rows // batch_rows, -(-VAL_TOKENS // (batch_rows * SEQ_LEN))) * batch_rows
    return np.arange(n, dtype=np.int64) * num_rows // n


def val_bpb(model, tokenizer, datasets, *, batch_rows, device):
    """{dataset: bits per byte} of each corpus's packed val split, read left to right over a fixed systematic sample of
    its rows (sample_rows). model(x, y, loss_reduction='none', cu_seqlens) gives each position's next-token loss (the
    training forward). Every corpus's batches share one shape (cu_seqlens padded to the corpora's most segments): one
    compiled graph scores all."""
    token_bytes = tokenizer.get_token_bytes(device=device)
    samples = {name: sample_rows(packed_rows(name, "val", tokenizer), batch_rows=batch_rows) for name in datasets}
    loaders = {name: PretrainingBatches({name: 1.0}, "val", tokenizer, {name: rows}, batch_rows=batch_rows, device=device,
                                        reading="left_to_right", start=0)
               for name, rows in samples.items()}
    num_segments = max(loader.num_segments for loader in loaders.values())
    return {name: evaluate_bpb(model, loader, len(samples[name]) // batch_rows, token_bytes, num_segments)
            for name, loader in loaders.items()}
