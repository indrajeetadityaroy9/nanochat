"""
Every tracked metric: a run's training records, its models' loss evaluations, and the export the paper's tables and
figures are drawn from.

Training records: METRICS/<tag>.jsonl, UTF-8 JSON lines, each an object parseable on its own: schema (SCHEMA), tag,
attempt (the run's process start it belongs to, 0-based: a resumed run repeats the steps after its checkpoint, and a
step's record from the latest attempt supersedes the earlier ones; records are never rewritten), kind, time (Unix
seconds), then the kind's fields. NaN and infinities are not JSON and are never written.

run (each process start): step (the first step it trains), plan (schedule.RunPlan), seed (initialization and data
    order), model (model/gpt.py GPTConfig as the model was built), variant (an ablation run's fields,
    ablations/harness.py; production's lr scale), params (the model's parameters), flops_per_token (model/gpt.py
    non_embedding_flops_per_token at the stage it starts in), max_norm (the gradient clip threshold), dataset_revision
    (config.DATASET_REVISION), lr and weight_decay (as the optimizers hold them; a resumed process's come from its
    checkpoint), stage (config.Stage as the model runs it, mixture as {corpus: row weight}).
stage (switch to the repo-level stage): step, stage.
train (each optimizer step, 0-based): step; tokens (row positions after it, padding included); loss (mean next-token
    loss in nats over the step's valid targets, fp64); lr (LambdaLR's at the step); grad_norm (global L2 norm of the
    gradient before clipping); tok_per_s (row positions per second between the step's two CUDA synchronizations).
end (the last step trained): step; peak_memory_bytes (torch.cuda.max_memory_allocated of the process).
failure (a non-finite loss or gradient norm; the run stops): step, loss and grad_norm (null).

Bits per byte: summed next-token loss in bits (nats / ln 2) over summed UTF-8 bytes of the targets, targets of 0 bytes
(special tokens, padding) excluded from both. Evaluations return both sums per context bucket (CONTEXTS) so corpora,
buckets and samples pool exactly; bpb and mixture_bpb form the ratios. Sums are fp64 on the device, reduced in a fixed
order.

Loss evaluation of a checkpoint: BASE_EVAL/<name>.json (name: checkpoint.checkpoint_name), one JSON object: schema
(EVALUATION_SCHEMA), run (the name), step, then loss_evaluation's results and the caller's (scripts/base_eval.py:
"codegen", "fim"; ablations/harness.py: null).

Export: RESULTS/<tag>.jsonl, one record: the run's records but the per-step ones (run, stage, end, failure) and the
evaluations of its final model and of its model-only snapshots verbatim, in step order; RESULTS/<tag>-curves.jsonl,
its per-step train records as columns (curves). Aggregates (bits per byte, pass@1 tables) are drawn from these
downstream.
"""
import bisect
import json
import math
import time
from collections.abc import Callable, Iterable
from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F

from nanochat.config import FILE_STAGE, REPO_STAGE, Stage
from nanochat.data.loader import PretrainingBatches, mixture_corpora, mixture_weights, packed_rows
from nanochat.data.tokenizer import Tokenizer
from nanochat.model.gpt import GPT
from nanochat.paths import BASE_EVAL, METRICS, RESULTS

SCHEMA = 2
EVALUATION_SCHEMA = 1
# Val sample per corpus: nanochat's --eval-tokens 80 x 2^19 (karpathy/nanochat 92d63d4 scripts/base_train.py:73), or the
# whole split if smaller; arms are scored on identical rows, so the subsample's sampling error cancels in differences
VAL_TOKENS = 80 * 2**19
# Context buckets of a target (its input's position in its piece + 1): positions both stages train (at most the
# file-level context) and positions only the repo-level stage trains
CONTEXTS = (FILE_STAGE.seq_len,)
# The benchmarks with reference solutions (eval/code_eval.py tasks): their bits per byte track pass@1 across checkpoints
# where corpus loss does not (Xing et al., arXiv 2604.07769)
SOLUTION_BENCHMARKS = ("humaneval_plus", "mbpp_plus", "humaneval_x_cpp")


def attempt(tag: str) -> int:
    """The attempt a process starting run `tag` now is: the run records already written (0 for a new run)."""
    return sum(r["kind"] == "run" for r in records(tag)) if (METRICS / f"{tag}.jsonl").exists() else 0


def append(tag: str, record: dict, *, attempt: int) -> None:
    line = json.dumps({"schema": SCHEMA, "tag": tag, "attempt": attempt, "time": time.time(), **record},
                      allow_nan=False)
    with (METRICS / f"{tag}.jsonl").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def records(tag: str) -> list[dict]:
    return [json.loads(line) for line in (METRICS / f"{tag}.jsonl").read_text(encoding="utf-8").splitlines()]


def train_steps(tag: str) -> dict[int, dict]:
    """{step: train record}, each step's from its latest attempt (attempts only grow down the file)."""
    return {r["step"]: r for r in records(tag) if r["kind"] == "train"}


def curves(tag: str) -> dict[str, list]:
    """train_steps as columns, one per train record field (step, tokens, loss, lr, grad_norm, tok_per_s, attempt,
    time), in step order."""
    by_step = train_steps(tag)
    steps = [by_step[s] for s in sorted(by_step)]
    return {k: [s[k] for s in steps] for k in steps[0] if k not in ("schema", "tag", "kind")}


def latest(tag: str, kind: str) -> dict:
    """The last record of a kind."""
    return [r for r in records(tag) if r["kind"] == kind][-1]


@torch.no_grad()
def evaluate(losses: Callable, batches: Iterable, token_bytes: torch.Tensor, *, steps: int,
             num_segments: int) -> dict[str, list[float]]:
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
        nats = torch.where(nbytes > 0, losses(x, y.view_as(x), cu_seqlens), 0.0)
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
    corpora = {name: {**evaluate(losses, loader, token_bytes, steps=len(samples[name]) // stage.batch_rows,
                                 num_segments=num_segments),
                      "rows": len(samples[name])} for name, loader in loaders.items()}
    weights = mixture_weights(stage.mixture, tokenizer, stage.seq_len)
    groups_bpb = {group: mixture_bpb(corpora, {c: weights[c] for c in members})
                  for group, (_, members) in stage.mixture.items()}
    return {"corpora": corpora, "bpb": groups_bpb | {"all": mixture_bpb(corpora, weights)}}


def solution_bits(losses: Callable, tokenizer: Tokenizer, benchmarks: dict, *, device: torch.device
                  ) -> dict[str, dict[str, list[float]]]:
    """{benchmark: bits and bytes (evaluate) of its reference solutions} for SOLUTION_BENCHMARKS, in FILE_STAGE's
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
        results[name] = evaluate(losses, batches, token_bytes, steps=len(batches), num_segments=batch_rows)
    return results


def loss_evaluation(model: GPT, tokenizer: Tokenizer, benchmarks: dict, *, device: torch.device) -> dict:
    """{"8k": val_bits of FILE_STAGE (config.MIXTURE in file-level rows), "32k": val_bits of REPO_STAGE (null for a
    model not at its context), "solution": solution_bits} by the compiled GPT.losses, one graph per row shape;
    benchmarks: eval/code_eval.py tasks()."""
    losses = torch.compile(model.losses, dynamic=False, fullgraph=True)
    return {"8k": val_bits(losses, tokenizer, stage=FILE_STAGE, device=device),
            "32k": (val_bits(losses, tokenizer, stage=REPO_STAGE, device=device)
                    if model.config.sequence_len == REPO_STAGE.seq_len else None),
            "solution": solution_bits(losses, tokenizer, benchmarks, device=device)}


def save_evaluation(name: str, step: int, results: dict) -> None:
    """BASE_EVAL/<name>.json; NaN and infinities are not JSON and raise."""
    (BASE_EVAL / f"{name}.json").write_text(json.dumps({"schema": EVALUATION_SCHEMA, "run": name, "step": step}
                                                       | results, allow_nan=False), encoding="utf-8")


def record(tag: str) -> dict:
    """A run's tracked metrics but the per-step ones: its non-train records and the evaluations (save_evaluation) of
    its final model and snapshots on disk, in step order (a diverged run has none)."""
    paths = [*BASE_EVAL.glob(f"{tag}.json"), *BASE_EVAL.glob(f"{tag}-step*.json")]
    return {"tag": tag, "records": [r for r in records(tag) if r["kind"] != "train"],
            "evaluations": sorted((json.loads(p.read_text(encoding="utf-8")) for p in paths), key=lambda r: r["step"])}


def export(tag: str) -> None:
    """RESULTS/<tag>.jsonl (record) and RESULTS/<tag>-curves.jsonl (curves)."""
    (RESULTS / f"{tag}.jsonl").write_text(json.dumps(record(tag), allow_nan=False) + "\n", encoding="utf-8")
    (RESULTS / f"{tag}-curves.jsonl").write_text(json.dumps({"tag": tag} | curves(tag), allow_nan=False) + "\n",
                                                 encoding="utf-8")
