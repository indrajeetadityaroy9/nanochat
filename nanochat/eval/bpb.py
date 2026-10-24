"""
Likelihood metrics of a checkpoint in bits per byte (BPB): summed next-token loss in bits (nats / ln 2) over the
summed UTF-8 bytes of the targets (The Pile's bits per UTF-8 byte, arXiv 2101.00027 §3.1; tokenizer-independent, so
models and vocabularies compare), targets of 0 bytes (special tokens, padding) excluded from both (karpathy/nanochat
evaluate_bpb, nanochat/loss_eval.py).
- val_bits: each training corpus's held-out val split, the test loss of autoregressive training (arXiv 2207.14255
  §2: its "AR loss"; "AR test loss", §1.1), per context bucket (CONTEXTS: Code Llama's loss as a function of context
  length, arXiv 2308.12950 §3.3, Fig. 4a), per corpus, per group (code, text, math) and overall, pooled by the stage's
  mixture weights (mixture_bpb), as The Pile reports its BPB per component and over the whole.
- solution_bits: each benchmark's reference solutions given their prompts, the task loss of arXiv 2412.04403 §2.2
  (the correct answer's bits per byte; its tasks multiple-choice, without pass@1), a predictor of pass@1 at scales
  where pass@1 is near zero (DataDecide, arXiv 2504.11393 §3.4, Fig. 6: the correct answer's probability makes
  HumanEval and MBPP decisions predictable where accuracy is trivial; arXiv 2508.13144 §5.3: BPB raises the
  signal-to-noise ratio of most benchmarks; arXiv 2604.07769 §3.3.1, Table 2: the loss on a benchmark's problems and
  reference solutions, its "valid-loss", tracks pass@1 where held-out corpus loss does not).
Each returns the bits and bytes sums per context bucket, so corpora, buckets and benchmarks pool exactly (bpb,
mixture_bpb). Sums are fp64 on the device, reduced in a fixed order.
"""
import math
import re
from collections.abc import Callable, Iterable
from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F

from nanochat.corpus.loader import PretrainingBatches, packed_rows
from nanochat.corpus.mixture import FILE_STAGE, NON_CODE, REPO_STAGE, Stage
from nanochat.corpus.tokenizer import Tokenizer
from nanochat.model.gpt import GPT

# Val sample per corpus: nanochat's --eval-tokens 80 x 2^19 (karpathy/nanochat 92d63d4 scripts/base_train.py:73), or the
# whole split if smaller; arms are scored on identical rows (the tokenizer ablation's, whose rows differ, on val splits
# within the sample: identical documents), so the subsample's sampling error cancels in differences
VAL_TOKENS = 80 * 2**19
# Context buckets of a target (its input's position in its piece + 1): positions both stages train (at most the
# file-level context) and positions only the repo-level stage trains
CONTEXTS = (FILE_STAGE.seq_len,)
# Byte-offset buckets of a target (the bytes of its piece before it), doublings from 1 KiB to 128 KiB (beyond a 32K
# row's span in bytes): loss at doubling context lengths, as arXiv 2308.12950 Fig. 4a samples perplexity,
# in bytes so that tokenizers compare at the same context (arXiv 2503.13423 §3.1 fixes context in bytes for this
# reason; App. B.1, Fig. 6: BPB by bytes of context)
OFFSETS = tuple(2**k for k in range(10, 18))
# The benchmarks with reference solutions (eval/code_eval.py tasks): Python's EvalPlus (arXiv 2305.01210) canonical
# solutions, C++'s HumanEval-X's (arXiv 2303.17568; MultiPL-E, arXiv 2208.08227, has none)
SOLUTION_BENCHMARKS = ("humaneval_plus", "mbpp_plus", "humaneval_x_cpp")


@torch.no_grad()
def sums(losses: Callable, batches: Iterable, token_bytes: torch.Tensor, *, steps: int,
         num_segments: int) -> dict[str, list[float]]:
    """{"bits", "bytes"} per context bucket (bucket i: contexts in (CONTEXTS[i-1], CONTEXTS[i]]) and {"offset_bits",
    "offset_bytes"} per byte-offset bucket (bucket i: targets after [OFFSETS[i-1], OFFSETS[i]) bytes of their segment)
    of the targets of the first `steps` batches under losses (GPT.losses). token_bytes: bytes per token, 0 for special
    tokens. cu_seqlens is padded to num_segments by repeating its last offset (one compiled shape across corpora; the
    attention reads only each batch's segments, corpus/loader.py PretrainingBatches); searchsorted(right) maps a
    position to its segment."""
    device = token_bytes.device
    edges = torch.tensor(CONTEXTS, dtype=torch.int32, device=device)
    offset_edges = torch.tensor(OFFSETS, dtype=torch.int64, device=device)
    total = torch.zeros(2, len(CONTEXTS) + 1, dtype=torch.float64, device=device)
    offset_total = torch.zeros(2, len(OFFSETS) + 1, dtype=torch.float64, device=device)
    for x, y, cu_seqlens, segments in islice(batches, steps):
        cu_seqlens = torch.cat([cu_seqlens, cu_seqlens[-1:].expand(num_segments + 1 - cu_seqlens.numel())])
        y = y.flatten()
        nbytes = torch.where(y == PretrainingBatches.PAD, 0, token_bytes[y.clamp_min(0)])
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
    (bits, nbytes), (offset_bits, offset_nbytes) = total.cpu().numpy(), offset_total.cpu().numpy()
    return {"bits": (bits / math.log(2)).tolist(), "bytes": nbytes.tolist(),
            "offset_bits": (offset_bits / math.log(2)).tolist(), "offset_bytes": offset_nbytes.tolist()}


def bpb(*pooled: dict) -> float:
    """Bits per byte of sums (sums) pooled over their context buckets and over every argument."""
    return sum(sum(s["bits"]) for s in pooled) / sum(sum(s["bytes"]) for s in pooled)


def mixture_bpb(corpora: dict, weights: dict[str, float], *, buckets: slice = slice(None)) -> float:
    """Bits per byte of val rows drawn with row weights {corpus: weight}: weighted bits per row over weighted bytes per
    row of corpora {corpus: {"bits", "bytes", "rows"}}, over the context buckets `buckets` (all by default)."""
    bits = sum(w * sum(corpora[c]["bits"][buckets]) / corpora[c]["rows"] for c, w in weights.items())
    nbytes = sum(w * sum(corpora[c]["bytes"][buckets]) / corpora[c]["rows"] for c, w in weights.items())
    return bits / nbytes


def sample_rows(stage: Stage) -> int:
    """The most rows of a corpus's val split val_bits scores at a stage's rows: VAL_TOKENS' worth in whole
    micro-batches."""
    return -(-VAL_TOKENS // (stage.batch_rows * stage.seq_len)) * stage.batch_rows


def val_bits(losses: Callable, tokenizer: Tokenizer, *, stage: Stage, device: torch.device) -> dict:
    """Val evaluation of a stage's corpora in its rows: {"corpora": {corpus: {"bits", "bytes", "rows"}}, "bpb": {group:
    mixture_bpb, "all": mixture_bpb of every corpus}, "weights": {corpus: weight} mixture_bpb pools by}, each corpus's
    packed val split read left to right over a systematic sample of its rows. The weights are the stage's shares of its
    train packed rows (Stage.shares at their total), the groups code (every code group), text and math. cu_seqlens is
    padded to the corpora's most segments: one compiled graph scores all."""
    token_bytes = torch.from_numpy(tokenizer.token_lengths).to(device=device, dtype=torch.int32)
    # the sample: n = min(num_rows, sample_rows) rows, row (j x num_rows) // n for j < n (the packer orders each shard's
    # rows longest first, Best-Fit-Decreasing, so a prefix would over-represent long documents), its last micro-batch
    # filled with empty rows: a split within the sample is scored whole, every row of it
    sampled, samples = {}, {}
    for name in stage.sources:
        num_rows = packed_rows(name, "val", tokenizer, stage.seq_len)
        n = sampled[name] = min(num_rows, sample_rows(stage))
        samples[name] = np.concatenate([np.arange(n, dtype=np.int64) * num_rows // n,
                                        np.full(-n % stage.batch_rows, -1, dtype=np.int64)])
    loaders = {name: PretrainingBatches({name: 1.0}, "val", tokenizer, {name: rows}, seq_len=stage.seq_len,
                                        batch_rows=stage.batch_rows, device=device, reading="left_to_right", start=0)
               for name, rows in samples.items()}
    num_segments = max(loader.num_segments for loader in loaders.values())
    corpora = {name: {**sums(losses, loader, token_bytes, steps=len(samples[name]) // stage.batch_rows,
                             num_segments=num_segments),
                      "rows": sampled[name]} for name, loader in loaders.items()}
    rows = {corpus: packed_rows(corpus, "train", tokenizer, stage.seq_len) for corpus in stage.sources}
    weights = stage.shares(rows, sum(rows.values()))
    groups = {"code": stage.code} | {name: stage.groups[name].sources for name in NON_CODE}
    groups_bpb = {group: mixture_bpb(corpora, {c: weights[c] for c in members}) for group, members in groups.items()}
    return {"corpora": corpora, "bpb": groups_bpb | {"all": mixture_bpb(corpora, weights)}, "weights": weights}


def solution_bits(losses: Callable, tokenizer: Tokenizer, benchmarks: dict, *, device: torch.device
                  ) -> dict[str, dict[str, list[float]]]:
    """{benchmark: bits and bytes (sums) of its reference solutions} for SOLUTION_BENCHMARKS, in FILE_STAGE's
    micro-batches (one compiled shape), one example and one attention segment per row: BOS + the tokens of prompt +
    solution, scored from the token starting at the prompt's end (decode_with_offsets; none starting there raises).
    The solution's leading blank lines, through its last leading line break, count as prompt: EvalPlus's solutions
    begin with them, and a tokenizer joins them to the prompt's last line break, so the prompt would end inside a token
    (every HumanEval+ and MBPP+ prompt, in each tokenizer measured) and tokenizers would score different bytes."""
    batch_rows, seq_len = FILE_STAGE.batch_rows, FILE_STAGE.seq_len
    bos = tokenizer.bos_token_id
    token_bytes = torch.from_numpy(tokenizer.token_lengths).to(device=device, dtype=torch.int32)
    cu_seqlens = torch.arange(0, batch_rows * seq_len + 1, seq_len, dtype=torch.int32, device=device)
    segments = torch.tensor(batch_rows, dtype=torch.int32)
    results = {}
    for name in SOLUTION_BENCHMARKS:
        seqs = []
        for t in benchmarks[name]:
            prompt_end = len(t.prompt) + len(re.match(r"\s*[\r\n]|", t.solution).group())
            ids = tokenizer.enc.encode_ordinary(t.prompt + t.solution)
            _, starts = tokenizer.enc.decode_with_offsets(ids)
            seqs.append(([bos, *ids], starts.index(prompt_end) + 1))  # + 1: the BOS before
        batches = []
        for g in range(0, len(seqs), batch_rows):
            x = torch.full((batch_rows, seq_len), bos, dtype=torch.int32)
            y = torch.full((batch_rows, seq_len), PretrainingBatches.PAD, dtype=torch.int64)
            for r, (ids, first) in enumerate(seqs[g:g + batch_rows]):
                x[r, :len(ids) - 1] = torch.tensor(ids[:-1])
                y[r, first - 1:len(ids) - 1] = torch.tensor(ids[first:])
            batches.append((x.to(device), y.to(device), cu_seqlens, segments))
        results[name] = sums(losses, batches, token_bytes, steps=len(batches), num_segments=batch_rows)
    return results


def likelihood(model: GPT, tokenizer: Tokenizer, benchmarks: dict, *, device: torch.device) -> dict:
    """{"8k": val_bits of FILE_STAGE, "32k": val_bits of REPO_STAGE (null for a model not at its context), "solution":
    solution_bits} by the compiled GPT.losses, one graph per row shape; benchmarks: eval/code_eval.py tasks()."""
    losses = torch.compile(model.losses, dynamic=False, fullgraph=True)
    return {"8k": val_bits(losses, tokenizer, stage=FILE_STAGE, device=device),
            "32k": (val_bits(losses, tokenizer, stage=REPO_STAGE, device=device)
                    if model.config.sequence_len == REPO_STAGE.seq_len else None),
            "solution": solution_bits(losses, tokenizer, benchmarks, device=device)}
