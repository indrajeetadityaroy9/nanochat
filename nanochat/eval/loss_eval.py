"""
Bits per byte: summed next-token loss in bits over summed UTF-8 bytes of the targets (special tokens 0 bytes, padding
excluded). Evaluations return both sums per context bucket (CONTEXTS) so corpora, buckets and samples pool exactly; bpb
and mixture_bpb form the ratios. Sums are fp64 on the device, reduced in a fixed order.
"""
import math
from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F

from nanochat.config import FILE_STAGE, Stage
from nanochat.data.loader import PretrainingBatches, mixture_corpora, mixture_weights, packed_rows

# Val sample per corpus: nanochat's --eval-tokens 80 x 2^19 (karpathy/nanochat 92d63d4 scripts/base_train.py:73), or the
# whole split if smaller; arms are scored on identical rows, so the subsample's sampling error cancels in differences
VAL_TOKENS = 80 * 2**19
# Context buckets of a target (its input's position in its piece + 1): positions both stages train (at most the
# file-level context) and positions only the repo-level stage trains
CONTEXTS = (FILE_STAGE.seq_len,)


@torch.no_grad()
def evaluate(losses, batches, token_bytes: torch.Tensor, *, steps: int, num_segments: int) -> dict[str, list[float]]:
    """{"bits", "bytes"} per context bucket (bucket i: contexts in (CONTEXTS[i-1], CONTEXTS[i]]) of the targets of the
    first `steps` batches under losses (GPT.losses). token_bytes: bytes per token, 0 for special tokens. cu_seqlens is
    padded to num_segments by repeating its last offset (one compiled shape across corpora); searchsorted(right) maps a
    position to its segment."""
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
    """Bits per byte of val rows drawn with the mixture's row weights (data/loader.py mixture_weights): weighted bits
    per row over weighted bytes per row of corpora {corpus: {"bits", "bytes", "rows"}}."""
    bits = sum(w * sum(corpora[c]["bits"]) / corpora[c]["rows"] for c, w in weights.items())
    nbytes = sum(w * sum(corpora[c]["bytes"]) / corpora[c]["rows"] for c, w in weights.items())
    return bits / nbytes


def sample_rows(num_rows: int, *, batch_rows: int, seq_len: int) -> np.ndarray:
    """Systematic sample of a split's rows: n = min(all full micro-batches, ceil(VAL_TOKENS / (batch_rows x seq_len))
    micro-batches) rows, row (j x num_rows) // n for j < n. The packer orders each shard's rows longest first
    (Best-Fit-Decreasing), so a prefix would over-represent long documents."""
    n = min(num_rows // batch_rows, -(-VAL_TOKENS // (batch_rows * seq_len))) * batch_rows
    return np.arange(n, dtype=np.int64) * num_rows // n


def val_bits(losses, tokenizer, groups: dict, *, stage: Stage, device: torch.device) -> dict:
    """Val evaluation of a mixture {group: (share, corpora)} in a stage's rows: {"corpora": {corpus: {"bits", "bytes",
    "rows"}}, "bpb": {group: mixture_bpb, "all": mixture_bpb of every corpus}}, each corpus's packed val split read left
    to right over sample_rows, weighted by the stage's mixture weights. cu_seqlens is padded to the corpora's most
    segments: one compiled graph scores all."""
    token_bytes = tokenizer.get_token_bytes(device=device)
    samples = {name: sample_rows(packed_rows(name, "val", tokenizer, stage.seq_len), batch_rows=stage.batch_rows,
                                 seq_len=stage.seq_len) for name in mixture_corpora(groups)}
    loaders = {name: PretrainingBatches({name: 1.0}, "val", tokenizer, {name: rows}, seq_len=stage.seq_len,
                                        batch_rows=stage.batch_rows, device=device, reading="left_to_right", start=0)
               for name, rows in samples.items()}
    num_segments = max(loader.num_segments for loader in loaders.values())
    corpora = {name: {**evaluate(losses, loader, token_bytes, steps=len(samples[name]) // stage.batch_rows,
                                 num_segments=num_segments),
                      "rows": len(samples[name])} for name, loader in loaders.items()}
    weights = mixture_weights(groups, tokenizer, stage.seq_len)
    groups_bpb = {group: mixture_bpb(corpora, {c: weights[c] for c in members}) for group, (_, members) in groups.items()}
    return {"corpora": corpora, "bpb": groups_bpb | {"all": mixture_bpb(corpora, weights)}}
