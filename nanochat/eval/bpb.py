"""
Likelihood metrics of a checkpoint in bits per byte (BPB): summed next-token loss in bits (nats / ln 2) over the
summed UTF-8 bytes of the targets, targets of 0 bytes (special tokens, padding) excluded from both (The Pile's
bits per UTF-8 byte, Gao et al., arXiv 2101.00027; tokenizer-independent, so models and vocabularies compare).
- val_bits: each training corpus's held-out val split, the test loss of autoregressive training (Bavarian et al.,
  arXiv 2207.14255 §3: the "AR test loss"), per context bucket (CONTEXTS: Code Llama's loss as a function of context
  length, arXiv 2308.12950 §3.3, Fig. 4a), per corpus, per group (code, text, math) and overall, pooled by the stage's
  mixture weights (mixture_bpb), as The Pile reports its BPB per component and over the whole.
- solution_bits: each benchmark's reference solutions given their prompts: the task loss that predicts pass@1 at
  scales where pass@1 is near zero (Bhagia et al., arXiv 2412.04403 §2; DataDecide, arXiv 2504.11393 §3.3: the correct
  answer's likelihood the best small-scale predictor for HumanEval and MBPP; Heineman et al., arXiv 2508.13144 §5.3:
  BPB raises the signal-to-noise ratio of most benchmarks; Xing et al., arXiv 2604.07769: it tracks pass@1 across
  checkpoints where corpus loss does not).
Each returns the bits and bytes sums per context bucket, so corpora, buckets and benchmarks pool exactly (bpb,
mixture_bpb). Sums are fp64 on the device, reduced in a fixed order.
"""
import bisect
import math
from collections.abc import Callable, Iterable
from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F

from nanochat.config import FILE_STAGE, REPO_STAGE, Stage
from nanochat.data.loader import PretrainingBatches, mixture_corpora, mixture_weights, packed_rows
from nanochat.data.tokenizer import Tokenizer
from nanochat.model.gpt import GPT

# Val sample per corpus: nanochat's --eval-tokens 80 x 2^19 (karpathy/nanochat 92d63d4 scripts/base_train.py:73), or the
# whole split if smaller; arms are scored on identical rows, so the subsample's sampling error cancels in differences
VAL_TOKENS = 80 * 2**19
# Context buckets of a target (its input's position in its piece + 1): positions both stages train (at most the
# file-level context) and positions only the repo-level stage trains
CONTEXTS = (FILE_STAGE.seq_len,)
# The benchmarks with reference solutions (eval/code_eval.py tasks): Python's EvalPlus canonical solutions, C++'s
# HumanEval-X's (MultiPL-E has none)
SOLUTION_BENCHMARKS = ("humaneval_plus", "mbpp_plus", "humaneval_x_cpp")


@torch.no_grad()
def sums(losses: Callable, batches: Iterable, token_bytes: torch.Tensor, *, steps: int,
         num_segments: int) -> dict[str, list[float]]:
    """{"bits", "bytes"} per context bucket (bucket i: contexts in (CONTEXTS[i-1], CONTEXTS[i]]) of the targets of the
    first `steps` batches under losses (GPT.losses). token_bytes: bytes per token, 0 for special tokens. cu_seqlens is
    padded to num_segments by repeating its last offset (one compiled shape across corpora); searchsorted(right) maps a
    position to its segment."""
    device = token_bytes.device
    edges = torch.tensor(CONTEXTS, dtype=torch.int32, device=device)
    total = torch.zeros(2, len(CONTEXTS) + 1, dtype=torch.float64, device=device)
    for x, y, cu_seqlens in islice(batches, steps):
        cu_seqlens = torch.cat([cu_seqlens, cu_seqlens[-1:].expand(num_segments + 1 - cu_seqlens.numel())])
        y = y.flatten()
        nbytes = torch.where(y == PretrainingBatches.PAD, 0, token_bytes[y.clamp_min(0)])
        nats = torch.where(nbytes > 0, losses(x, y.view_as(x), cu_seqlens), 0.0)
        t = torch.arange(y.numel(), dtype=torch.int32, device=device)
        context = t - cu_seqlens[torch.searchsorted(cu_seqlens, t, right=True) - 1] + 1
        bucket = F.one_hot(torch.bucketize(context, edges), len(CONTEXTS) + 1).double()
        total += (torch.stack([nats.double(), nbytes.double()])[:, :, None] * bucket).sum(1)
    bits, nbytes = total.cpu().numpy()
    return {"bits": (bits / math.log(2)).tolist(), "bytes": nbytes.tolist()}


def bpb(*pooled: dict) -> float:
    """Bits per byte of sums (sums) pooled over their context buckets and over every argument."""
    return sum(sum(s["bits"]) for s in pooled) / sum(sum(s["bytes"]) for s in pooled)


def mixture_bpb(corpora: dict, weights: dict[str, float], *, buckets: slice = slice(None)) -> float:
    """Bits per byte of val rows drawn with the mixture's row weights (data/loader.py mixture_weights): weighted bits
    per row over weighted bytes per row of corpora {corpus: {"bits", "bytes", "rows"}}, over the context buckets
    `buckets` (all by default)."""
    bits = sum(w * sum(corpora[c]["bits"][buckets]) / corpora[c]["rows"] for c, w in weights.items())
    nbytes = sum(w * sum(corpora[c]["bytes"][buckets]) / corpora[c]["rows"] for c, w in weights.items())
    return bits / nbytes


def val_bits(losses: Callable, tokenizer: Tokenizer, *, stage: Stage, device: torch.device) -> dict:
    """Val evaluation of a stage's mixture {group: (share, corpora)} in its rows: {"corpora": {corpus: {"bits", "bytes",
    "rows"}}, "bpb": {group: mixture_bpb, "all": mixture_bpb of every corpus}}, each corpus's packed val split read left
    to right over a systematic sample of its rows, weighted by the stage's mixture weights. cu_seqlens is padded to the
    corpora's most segments: one compiled graph scores all."""
    token_bytes = tokenizer.get_token_bytes(device=device)
    # the sample: n = min(all full micro-batches, ceil(VAL_TOKENS / (batch_rows x seq_len)) micro-batches) rows, row
    # (j x num_rows) // n for j < n; the packer orders each shard's rows longest first (Best-Fit-Decreasing), so a prefix
    # would over-represent long documents
    samples = {}
    for name in mixture_corpora(stage.mixture):
        num_rows = packed_rows(name, "val", tokenizer, stage.seq_len)
        n = min(num_rows // stage.batch_rows, -(-VAL_TOKENS // (stage.batch_rows * stage.seq_len))) * stage.batch_rows
        samples[name] = np.arange(n, dtype=np.int64) * num_rows // n
    loaders = {name: PretrainingBatches({name: 1.0}, "val", tokenizer, {name: rows}, seq_len=stage.seq_len,
                                        batch_rows=stage.batch_rows, device=device, reading="left_to_right", start=0)
               for name, rows in samples.items()}
    num_segments = max(loader.num_segments for loader in loaders.values())
    corpora = {name: {**sums(losses, loader, token_bytes, steps=len(samples[name]) // stage.batch_rows,
                             num_segments=num_segments),
                      "rows": len(samples[name])} for name, loader in loaders.items()}
    weights = mixture_weights(stage.mixture, tokenizer, stage.seq_len)
    groups_bpb = {group: mixture_bpb(corpora, {c: weights[c] for c in members})
                  for group, (_, members) in stage.mixture.items()}
    return {"corpora": corpora, "bpb": groups_bpb | {"all": mixture_bpb(corpora, weights)}}


def solution_bits(losses: Callable, tokenizer: Tokenizer, benchmarks: dict, *, device: torch.device
                  ) -> dict[str, dict[str, list[float]]]:
    """{benchmark: bits and bytes (sums) of its reference solutions} for SOLUTION_BENCHMARKS, in FILE_STAGE's
    micro-batches (one compiled shape), one example and one attention segment per row: BOS + the tokens of prompt +
    solution, targets padding before the first token starting at or after the prompt's end (decode_with_offsets); a
    token straddling it is not scored."""
    batch_rows, seq_len = FILE_STAGE.batch_rows, FILE_STAGE.seq_len
    bos, token_bytes = tokenizer.bos_token_id, tokenizer.get_token_bytes(device=device)
    cu_seqlens = torch.arange(0, batch_rows * seq_len + 1, seq_len, dtype=torch.int32, device=device)
    results = {}
    for name in SOLUTION_BENCHMARKS:
        seqs = []
        for t in benchmarks[name]:
            ids = tokenizer.enc.encode_ordinary(t.prompt + t.solution)
            _, starts = tokenizer.enc.decode_with_offsets(ids)
            seqs.append(([bos, *ids], bisect.bisect_left(starts, len(t.prompt)) + 1))  # + 1: the BOS before
        batches = []
        for g in range(0, len(seqs), batch_rows):
            x = torch.full((batch_rows, seq_len), bos, dtype=torch.int32)
            y = torch.full((batch_rows, seq_len), PretrainingBatches.PAD, dtype=torch.int64)
            for r, (ids, first) in enumerate(seqs[g:g + batch_rows]):
                x[r, :len(ids) - 1] = torch.tensor(ids[:-1])
                y[r, first - 1:len(ids) - 1] = torch.tensor(ids[first:])
            batches.append((x.to(device), y.to(device), cu_seqlens))
        results[name] = sums(losses, batches, token_bytes, steps=len(batches), num_segments=batch_rows)
    return results


def likelihood(model: GPT, tokenizer: Tokenizer, benchmarks: dict, *, device: torch.device) -> dict:
    """{"8k": val_bits of FILE_STAGE (config.MIXTURE in file-level rows), "32k": val_bits of REPO_STAGE (null for a
    model not at its context), "solution": solution_bits} by the compiled GPT.losses, one graph per row shape;
    benchmarks: eval/code_eval.py tasks()."""
    losses = torch.compile(model.losses, dynamic=False, fullgraph=True)
    return {"8k": val_bits(losses, tokenizer, stage=FILE_STAGE, device=device),
            "32k": (val_bits(losses, tokenizer, stage=REPO_STAGE, device=device)
                    if model.config.sequence_len == REPO_STAGE.seq_len else None),
            "solution": solution_bits(losses, tokenizer, benchmarks, device=device)}
