"""
Likelihood metrics of a checkpoint in bits per byte (metrics.bpb, metrics.mixture_bpb); targets of 0 bytes (special
tokens, padding) are excluded from both sums (karpathy/nanochat evaluate_bpb, nanochat/loss_eval.py).
- val_bits: each training corpus's held-out val split, the AR loss of arXiv 2207.14255 §2, per context bucket
  (CONTEXTS; arXiv 2308.12950 §3.3, Fig. 4a), per corpus, per group (code, text, math) and overall, pooled by the
  stage's mixture weights as The Pile pools its components.
- solution_bits: each benchmark's reference solutions given their prompts, the task loss of arXiv 2412.04403 §2.2.
Each returns bits and bytes sums per context bucket (corpora, buckets and benchmarks pool exactly) and per sampled
row, the unit of the row-level standard errors (metrics.mixture_bpb_summary). Sums are fp64 on the device, reduced in
a fixed order.
"""
import math
import re
from collections.abc import Callable, Iterable
from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F

from corpus.layout import packed_rows, stage_rows
from corpus.loader import IGNORE_INDEX, PretrainingBatches
from corpus.mixture import FILE_LEVEL_STAGE, NON_CODE, REPO_LEVEL_STAGE, Stage
from corpus.tokenizer import Tokenizer
from metrics import mixture_bpb, mixture_bpb_summary, offset_bpb
from model.modeling import CausalLM

# Val sample per corpus: nanochat's --eval-tokens 80 x 2^19 (karpathy/nanochat 92d63d4 scripts/base_train.py:73), or
# the whole split if smaller
VAL_TOKENS = 80 * 2**19
# Context buckets of a target (its input's position in its piece + 1): positions both stages train (at most the
# file-level context) and positions only the repo-level stage trains
CONTEXTS = (FILE_LEVEL_STAGE.seq_len,)
# Byte-offset buckets of a target (the bytes of its piece before it), doublings from 1 KiB to 128 KiB (beyond a 32K
# row's span in bytes): arXiv 2308.12950 Fig. 4a's loss by context length, in bytes (arXiv 2503.13423 §3.1, App. B.1)
OFFSETS = tuple(2**k for k in range(10, 18))
# Benchmarks with reference solutions (eval/code_eval.py tasks): EvalPlus's (arXiv 2305.01210) canonical solutions,
# HumanEval-X's C++ ones (arXiv 2303.17568)
SOLUTION_BENCHMARKS = ("humaneval_plus", "mbpp_plus", "humaneval_x_cpp")


@torch.no_grad()
def sums(losses: Callable, batches: Iterable, token_bytes: torch.Tensor, *, steps: int,
         num_segments: int) -> dict[str, list[float]]:
    """{"bits", "bytes"} per context bucket (bucket i: contexts in (CONTEXTS[i-1], CONTEXTS[i]]), {"offset_bits",
    "offset_bytes"} per byte-offset bucket (bucket i: targets after [OFFSETS[i-1], OFFSETS[i]) bytes of their segment)
    and {"row_bits", "row_bytes"} per row of the batches in order, of the targets of the first `steps` batches under
    losses (CausalLM.losses). batches: corpus/loader.py PretrainingBatches' tuples. token_bytes: bytes per token, 0 for
    special tokens. cu_seqlens is padded to num_segments by repeating its last offset (one compiled shape across
    corpora; the attention reads only each batch's segments, corpus/loader.py PretrainingBatches); searchsorted(right)
    maps a position to its segment."""
    device = token_bytes.device
    edges = torch.tensor(CONTEXTS, dtype=torch.int32, device=device)
    offset_edges = torch.tensor(OFFSETS, dtype=torch.int64, device=device)
    total = torch.zeros(2, len(CONTEXTS) + 1, dtype=torch.float64, device=device)
    offset_total = torch.zeros(2, len(OFFSETS) + 1, dtype=torch.float64, device=device)
    rows = []
    for x, y, cu_seqlens, segments, _ in islice(batches, steps):
        cu_seqlens = torch.cat([cu_seqlens, cu_seqlens[-1:].expand(num_segments + 1 - cu_seqlens.numel())])
        y = y.flatten()
        nbytes = torch.where(y == IGNORE_INDEX, 0, token_bytes[y.clamp_min(0)])
        nats = torch.where(nbytes > 0, losses(x, y.view_as(x), cu_seqlens, segments), 0.0)
        t = torch.arange(y.numel(), dtype=torch.int32, device=device)
        start = cu_seqlens[torch.searchsorted(cu_seqlens, t, right=True) - 1]
        context = t - start + 1
        before = nbytes.cumsum(0, dtype=torch.int64) - nbytes  # bytes of the targets before each one
        offset = before - before[start]
        scored = torch.stack([nats.double(), nbytes.double()])[:, :, None]
        total += (scored * F.one_hot(torch.bucketize(context, edges), len(CONTEXTS) + 1).double()).sum(1)
        offset_bucket = F.one_hot(torch.bucketize(offset, offset_edges, right=True), len(OFFSETS) + 1).double()
        offset_total += (scored * offset_bucket).sum(1)
        rows.append(scored.view(2, x.size(0), -1).sum(2))
    (bits, nbytes), (offset_bits, offset_nbytes) = total.cpu().numpy(), offset_total.cpu().numpy()
    row_bits, row_nbytes = torch.cat(rows, 1).cpu().numpy()
    return {"bits": (bits / math.log(2)).tolist(), "bytes": nbytes.tolist(),
            "offset_bits": (offset_bits / math.log(2)).tolist(), "offset_bytes": offset_nbytes.tolist(),
            "row_bits": (row_bits / math.log(2)).tolist(), "row_bytes": row_nbytes.tolist()}


def sample_rows(stage: Stage) -> int:
    """The most rows of a corpus's val split val_bits scores at a stage's rows: VAL_TOKENS' worth in whole
    micro-batches."""
    return -(-VAL_TOKENS // (stage.micro_batch_size * stage.seq_len)) * stage.micro_batch_size


def val_bits(losses: Callable, tokenizer: Tokenizer, *, stage: Stage, device: torch.device) -> dict:
    """Val evaluation of a stage's corpora in its rows: {"corpora": {corpus: {"bits", "bytes", "offset_bits",
    "offset_bytes", "row_bits", "row_bytes" (sums: its sampled rows), "rows"}}, "bpb": {group: mixture_bpb, "all":
    mixture_bpb of every corpus}, "errors": {group: metrics.mixture_bpb_summary}, "offset_bpb": {group:
    metrics.offset_bpb}, "weights": {corpus: weight} mixture_bpb pools by}, each corpus's packed val split read left
    to right over a systematic sample of its rows. The weights are the stage's shares of its train packed rows
    (Stage.shares at their total), the groups code (every code group), text and math. cu_seqlens is padded to the
    corpora's most segments: one compiled graph scores all."""
    token_bytes = torch.from_numpy(tokenizer.token_lengths).to(device=device, dtype=torch.int32)
    # n = min(num_rows, sample_rows) rows, row (j x num_rows) // n for j < n (systematic: the packer orders each
    # shard's rows longest first), the last micro-batch filled with empty rows
    sampled, samples = {}, {}
    for name in stage.sources:
        num_rows = packed_rows(name, "val", tokenizer, stage.seq_len)
        n = sampled[name] = min(num_rows, sample_rows(stage))
        samples[name] = np.concatenate([np.arange(n, dtype=np.int64) * num_rows // n,
                                        np.full(-n % stage.micro_batch_size, -1, dtype=np.int64)])
    loaders = {name: PretrainingBatches({name: 1.0}, "val", tokenizer, {name: rows}, seq_len=stage.seq_len,
                                        micro_batch_size=stage.micro_batch_size, device=device,
                                        reading="left_to_right", start=0)
               for name, rows in samples.items()}
    num_segments = max(loader.num_segments for loader in loaders.values())
    corpora = {}
    for name, loader in loaders.items():
        scored = sums(losses, loader, token_bytes, steps=len(samples[name]) // stage.micro_batch_size,
                      num_segments=num_segments)
        n = sampled[name]  # the sample's rows, without the empty rows filling its last micro-batch
        corpora[name] = scored | {"row_bits": scored["row_bits"][:n], "row_bytes": scored["row_bytes"][:n], "rows": n}
    rows = stage_rows(tokenizer, stage)
    weights = stage.shares(rows, sum(rows.values()))
    groups = {"code": stage.code} | {name: stage.groups[name].sources for name in NON_CODE}
    group_weights = {group: {c: weights[c] for c in members} for group, members in groups.items()} | {"all": weights}
    return {"corpora": corpora, "bpb": {group: mixture_bpb(corpora, w) for group, w in group_weights.items()},
            "errors": {group: mixture_bpb_summary(corpora, w) for group, w in group_weights.items()},
            "offset_bpb": {group: offset_bpb(corpora, w) for group, w in group_weights.items()}, "weights": weights}


def solution_bits(losses: Callable, tokenizer: Tokenizer, benchmarks: dict, *, device: torch.device
                  ) -> dict[str, dict[str, list[float]]]:
    """{benchmark: sums of its reference solutions ("row_bits", "row_bytes": per solution, in benchmark order)} for
    SOLUTION_BENCHMARKS, in FILE_LEVEL_STAGE's micro-batches (one compiled shape), one example and one attention
    segment per row: BOS + the tokens of prompt + solution, scored from the token starting at the prompt's end
    (decode_with_offsets; none starting there raises).
    The solution's leading blank lines, through its last leading line break, count as prompt (EvalPlus's solutions
    begin with them; a tokenizer would join them to the prompt's last line break)."""
    micro_batch_size, seq_len = FILE_LEVEL_STAGE.micro_batch_size, FILE_LEVEL_STAGE.seq_len
    bos = tokenizer.bos_token_id
    token_bytes = torch.from_numpy(tokenizer.token_lengths).to(device=device, dtype=torch.int32)
    cu_seqlens = torch.arange(0, micro_batch_size * seq_len + 1, seq_len, dtype=torch.int32, device=device)
    segments = torch.tensor(micro_batch_size, dtype=torch.int32)
    results = {}
    for name in SOLUTION_BENCHMARKS:
        seqs = []
        for t in benchmarks[name]:
            prompt_end = len(t.prompt) + len(re.match(r"\s*[\r\n]|", t.canonical_solution).group())
            ids = tokenizer.enc.encode_ordinary(t.prompt + t.canonical_solution)
            _, starts = tokenizer.enc.decode_with_offsets(ids)
            seqs.append(([bos, *ids], starts.index(prompt_end) + 1))  # + 1: the BOS before
        batches = []
        for g in range(0, len(seqs), micro_batch_size):
            x = torch.full((micro_batch_size, seq_len), bos, dtype=torch.int32)
            y = torch.full((micro_batch_size, seq_len), IGNORE_INDEX, dtype=torch.int64)
            for r, (ids, first) in enumerate(seqs[g:g + micro_batch_size]):
                x[r, :len(ids) - 1] = torch.tensor(ids[:-1])
                y[r, first - 1:len(ids) - 1] = torch.tensor(ids[first:])
            batches.append((x.to(device), y.to(device), cu_seqlens, segments, None))  # sums reads no lengths
        scored = sums(losses, batches, token_bytes, steps=len(batches), num_segments=micro_batch_size)
        results[name] = scored | {k: scored[k][:len(seqs)] for k in ("row_bits", "row_bytes")}
    return results


def likelihood(model: CausalLM, tokenizer: Tokenizer, benchmarks: dict, *, device: torch.device) -> dict:
    """{"8k": val_bits of FILE_LEVEL_STAGE, "32k": val_bits of REPO_LEVEL_STAGE (null for a model not at its context),
    "solution": solution_bits} by the compiled CausalLM.losses, one graph per row shape; benchmarks:
    eval/code_eval.py tasks()."""
    losses = torch.compile(model.losses, dynamic=False, fullgraph=True)
    return {"8k": val_bits(losses, tokenizer, stage=FILE_LEVEL_STAGE, device=device),
            "32k": (val_bits(losses, tokenizer, stage=REPO_LEVEL_STAGE, device=device)
                    if model.config.max_position_embeddings == REPO_LEVEL_STAGE.seq_len else None),
            "solution": solution_bits(losses, tokenizer, benchmarks, device=device)}
