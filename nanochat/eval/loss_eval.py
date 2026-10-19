"""
Bits per byte of a model on held-out text: the summed next-token loss of the target tokens in bits over their summed
length in UTF-8 bytes, a vocabulary-independent metric (special tokens count 0 bytes and padding is excluded). Every
evaluation returns the two sums, not their ratio, per context bucket (CONTEXTS), so corpora, buckets and samples pool
exactly; bpb and mixture_bpb form the ratios. The sums are fp64 on the device (no host synchronization per batch; a NaN
loss propagates to the result) and reduced in a fixed order (no atomics: bitwise reproducible).
"""
import math
from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F

from nanochat.config import FILE_STAGE, Stage
from nanochat.data.loader import PretrainingBatches, mixture_corpora, mixture_weights, packed_rows

# The val sample per corpus: nanochat's evaluation budget, --eval-tokens 80 x 2^19 tokens (karpathy/nanochat 92d63d4
# scripts/base_train.py:73), or the whole split when it is smaller
VAL_TOKENS = 80 * 2**19
# The context buckets of a target, by the tokens it is predicted from (its input's position in its piece + 1, the
# piece's BOS included; a piece is a document or the part of one a row holds): at most the file-level stage's context
# (what the 8K stage trains), at most twice it, and beyond (the two halves the 32K repo-level stage adds)
CONTEXTS = (FILE_STAGE.seq_len, 2 * FILE_STAGE.seq_len)


@torch.no_grad()
def evaluate(losses, batches, steps: int, token_bytes: torch.Tensor, num_segments: int) -> dict[str, list[float]]:
    """{"bits", "bytes"}: per context bucket (CONTEXTS: bucket i holds contexts in (CONTEXTS[i-1], CONTEXTS[i]]), the
    summed loss in bits and length in bytes of the targets of the first `steps` batches (the loader's (inputs, targets,
    cu_seqlens)) under losses (GPT.losses: each position's loss, 0 at padding targets). token_bytes (vocab_size,): each
    token's length in bytes, 0 for special tokens, which are therefore excluded. cu_seqlens is padded to num_segments
    with empty segments (its last offset repeated, as the loader's unused entries), so loaders of different corpora
    give the compiled model one shape; a position's piece is its segment (searchsorted right: a segment's first
    position belongs to it, and the repeated offsets lie past every position)."""
    device = token_bytes.device
    edges = torch.tensor(CONTEXTS, dtype=torch.int32, device=device)
    sums = torch.zeros(2, len(CONTEXTS) + 1, dtype=torch.float64, device=device)
    for x, y, cu_seqlens in islice(batches, steps):
        cu_seqlens = torch.cat([cu_seqlens, cu_seqlens[-1:].expand(num_segments + 1 - cu_seqlens.numel())])
        y = y.flatten()
        nbytes = torch.where(y == PretrainingBatches.PAD, 0, token_bytes[y.clamp_min(0)])
        nats = losses(x, y.view_as(x), cu_seqlens) * (nbytes > 0)
        t = torch.arange(y.numel(), dtype=torch.int32, device=device)
        context = t - cu_seqlens[torch.searchsorted(cu_seqlens, t, right=True) - 1] + 1
        bucket = F.one_hot(torch.bucketize(context, edges), len(CONTEXTS) + 1).double()
        sums += (torch.stack([nats.double(), nbytes.double()])[:, :, None] * bucket).sum(1)
    bits, nbytes = sums.cpu().numpy()
    return {"bits": (bits / math.log(2)).tolist(), "bytes": nbytes.tolist()}


def bpb(sums: dict) -> float:
    """Bits per byte of evaluated sums (evaluate), pooled over their context buckets."""
    return sum(sums["bits"]) / sum(sums["bytes"])


def mixture_bpb(corpora: dict, weights: dict[str, float]) -> float:
    """Bits per byte of val rows drawn as a mixture draws its rows ({corpus: row weight}, data/loader.py
    mixture_weights): the ratio of the corpora's weighted bits per row to their weighted bytes per row (corpora: val_bits'
    {corpus: {"bits", "bytes", "rows"}})."""
    bits = sum(w * sum(corpora[c]["bits"]) / corpora[c]["rows"] for c, w in weights.items())
    nbytes = sum(w * sum(corpora[c]["bytes"]) / corpora[c]["rows"] for c, w in weights.items())
    return bits / nbytes


def sample_rows(num_rows: int, *, batch_rows: int, seq_len: int) -> np.ndarray:
    """Row ids of a split's systematic sample: n = ceil(VAL_TOKENS / (batch_rows x seq_len)) x batch_rows rows (all
    full micro-batches of a smaller split) (j x num_rows) // n for j < n, spread evenly over the whole split. The packer
    orders each shard's rows longest pieces first (Best-Fit-Decreasing), so a prefix of the split over-represents long
    documents; an even stride samples every position of every shard."""
    n = min(num_rows // batch_rows, -(-VAL_TOKENS // (batch_rows * seq_len))) * batch_rows
    return np.arange(n, dtype=np.int64) * num_rows // n


def val_bits(losses, tokenizer, groups: dict, *, stage: Stage, device: torch.device) -> dict:
    """The val evaluation of a mixture {group: (share, corpora)} in rows of a training stage's context (config.Stage:
    seq_len, batch_rows per batch) by losses (GPT.losses): {"corpora": {corpus: {"bits", "bytes" (evaluate), "rows"}},
    "bpb": {group: mixture_bpb of its corpora, "all": of every corpus}}, each corpus's packed val split read left to
    right over a fixed systematic sample of its rows (sample_rows), weighted as the stage's mixture weights its train
    rows. Every corpus's batches share one shape (cu_seqlens padded to the corpora's most segments): one compiled graph
    scores all."""
    token_bytes = tokenizer.get_token_bytes(device=device)
    samples = {name: sample_rows(packed_rows(name, "val", tokenizer, stage.seq_len), batch_rows=stage.batch_rows,
                                 seq_len=stage.seq_len) for name in mixture_corpora(groups)}
    loaders = {name: PretrainingBatches({name: 1.0}, "val", tokenizer, {name: rows}, seq_len=stage.seq_len,
                                        batch_rows=stage.batch_rows, device=device, reading="left_to_right", start=0)
               for name, rows in samples.items()}
    num_segments = max(loader.num_segments for loader in loaders.values())
    corpora = {name: {**evaluate(losses, loader, len(samples[name]) // stage.batch_rows, token_bytes, num_segments),
                      "rows": len(samples[name])} for name, loader in loaders.items()}
    weights = mixture_weights(groups, tokenizer, stage.seq_len)
    groups_bpb = {group: mixture_bpb(corpora, {c: weights[c] for c in members}) for group, (_, members) in groups.items()}
    return {"corpora": corpora, "bpb": groups_bpb | {"all": mixture_bpb(corpora, weights)}}
