"""
Metrics: the run log every training process writes, the measures checkpoints are scored and runs compared by, their
error bars, the hardware metrics of training and decoding, and the tokenizer metrics. Torch-free, so training,
evaluation and the ablations' analyses import it alike; the one source of the repository's metrics.

The run log, RUNS/<run_name>/metrics.jsonl (run_name: run.py run_name): UTF-8 JSON lines, each an object parseable on
its own: run_name, attempt (the run's process start it belongs to, 0-based: a resumed run repeats the steps after its
checkpoint, and a step's record from the latest attempt supersedes the earlier ones; records are never rewritten),
kind, time (Unix seconds), then the kind's fields. NaN and infinities are not JSON and are never written.
run (each process start): step (the first step it trains), plan (schedule.RunPlan), seed (initialization and data
    order), config (run.py RunConfig, the run's fields), data_key (corpus/tokenizer.py Tokenizer.data_key of
    the run's tokenizer; with plan, what run.py's stale guard compares), model (model/modeling.py ModelConfig at
    the stage of the weights it starts from: the file-level stage, or its checkpoint's), num_parameters (the model's
    parameters), flops_per_token (model/modeling.py non_embedding_flops_per_token of that model: DeepSeek LLM's M,
    arXiv 2401.02954 §3.2 Eq. 2, the compute of the scaling plan), model_flops_per_token (model_flops_per_token at that
    stage: PaLM's nominal per-token model FLOPs), max_grad_norm (the gradient clip threshold), learning_rate and
    weight_decay ({"adamw": [per group], "muon": [per group]}, as the optimizers hold them; a resumed process's come
    from its checkpoint), stage (corpus/mixture.py Stage as the model runs it, mixture as {corpus: row weight}).
stage (switch to the repo-level stage): step, stage, model_flops_per_token (at the repo-level stage),
    peak_memory_bytes and peak_reserved_bytes (torch.cuda.max_memory_allocated and max_memory_reserved over the
    file-level stage; both peaks reset at the switch, the stages' shapes differing).
train (each optimizer step, 0-based): step; num_input_tokens_seen (row positions after it, padding included); loss
    (mean next-token loss in nats over the step's valid labels, fp64); learning_rate (LambdaLR's at the step);
    grad_norm (global L2 norm of the gradient before clipping, the norm global-norm clipping thresholds:
    arXiv 1211.5063 §3.2); step_seconds (the step's device time between CUDA events at the previous step's end and its
    own; train/loop.py reads a step's values one step late, so the host never drains the device);
    tokens_per_second (row positions per second over step_seconds: the observed throughput of PaLM's model FLOPs
    utilization, arXiv 2204.02311 App. B); model_flops (executed_model_flops of the step's micro-batches); mfu
    (model_flops per step_seconds over PEAK_FLOPS); train/diagnostics.py's: max_logsumexp and mean_square_logsumexp
    (the output logits' logsumexp over the step's valid positions: maximum and mean square, PaLM's z-loss quantity,
    arXiv 2204.02311 §5), qk_norm_gain_max (the largest QK-norm gain magnitude), corpus_loss ({corpus: mean loss of
    the step's rows drawn from it}); every 10th step update_rms ({optimizer[group]: RMS of the step's update without
    its decoupled weight decay}) and param_norm ({optimizer[group]: L2 norm after the step}).
end (the last step trained): step; peak_memory_bytes and peak_reserved_bytes over the last stage (since the switch,
    or the process start): the memory footprint FlashAttention reports (arXiv 2205.14135) and the allocator's.
failure (a non-finite loss or gradient norm; the run stops): step, loss and grad_norm (null).

Bits per byte (bpb, mixture_bpb): summed next-token loss in bits (nats / ln 2) over the summed UTF-8 bytes of the
targets (The Pile's bits per UTF-8 byte, arXiv 2101.00027 §3.1; tokenizer-independent, so models and vocabularies
compare), from the bits and bytes sums eval/bpb.py computes per context bucket, pooled exactly.

Parse outcome (parse_outcome, parse_shares): how a completion compiles. Compared measures (measures): a run's
measures from its final checkpoint's evaluation and its train records.

Completion similarity (no execution; per sample, averaged by the caller): line exact match (line_exact_match), the
single-line infilling metric of SantaCoder (arXiv 2301.03988 §6.1.2; bigcode-evaluation-harness santacoder_fim.py
process_results) and DeepSeek-Coder (arXiv 2401.14196 §4.2, Table 6); CrossCodeEval's exact match, edit similarity and
identifier exact match / F1 (exact_match, edit_similarity, identifier_match; arXiv 2310.11248 §4, amazon-science/cceval
@40c68d2 scripts/eval_metric.py process_examples and compute_metric_stmt, eval_utils.py cal_edit_sim; DeepSeek-Coder
§4.3, Table 7), as cceval computes them on the same strings.

pass@k (pass_at_k): Codex's unbiased estimator per problem (arXiv 2107.03374 §2.1), the implementation the benchmarks'
own evaluators call (human-eval / human_eval_infilling estimate_pass_at_k).

Error bars (summary, bpb_summary; Miller, arXiv 2411.00640): each score is a mean of question-level scores (a
problem's pass@k estimate, its mean over samples, Miller §3.1); its CLT standard error (Eq. 1), clustered by the
problem a question derives from when several do (Eq. 4, as statsmodels' cluster-robust CR0: the infilling benchmarks'
lines and spans of one HumanEval problem); 95% intervals by the normal quantile (Eq. 5-6).

Hardware (model_flops_per_token, executed_model_flops, mfu, latency): model FLOPs utilization (PaLM, arXiv 2204.02311
App. B: 6N + 12 L H Q T per token, 2 FLOPs per multiply-add, rematerialization not counted) at the GPU's dense BF16
peak, N the matmul parameters. Nominal (model_flops_per_token): each layer's T the most keys a query attends, itself
and up to its window before it, min(window + 1, context), without the causal halving (torchtitan
quadratic_attention_flops_per_token). Executed
(executed_model_flops): the keys each query attends within its segment (cu_seqlens: a document, a FIM example, a row's
padding) and window, causally, the work FlashAttention does (Megatron-LM num_floating_point_operations' causal and
per-document counting); torch.utils.flop_counter counts the same once each custom op registers its formula, but per
step it would read cu_seqlens on the host (a device sync) and dispatch every op through Python. DeepSeek LLM's M stays
the compute of the scaling plan, not an MFU numerator.
Decoding latency as vLLM's benchmark reports it (vllm/benchmarks/serve.py calculate_metrics): time to first token from
the request taking a decoder slot, time per output token after the first, end-to-end latency, output tokens per second.

Fits (parabola_vertex): the vertex of a least-squares parabola, the optimum of a sweep (ablations/peak_lr.py).
"""
import difflib
import json
import keyword
import re
import statistics
import time
import warnings
from collections import Counter
from collections.abc import Sequence
from typing import Literal

import numpy as np
import statsmodels.api as sm
from human_eval_infilling.evaluation import estimate_pass_at_k
from numpy.typing import ArrayLike

from corpus.mixture import REPO_LEVEL_STAGE
from paths import RUNS

PARSE_OUTCOMES = ("parses", "indentation", "syntax")


def append(run_name: str, record: dict, *, attempt: int) -> None:
    line = json.dumps({"run_name": run_name, "attempt": attempt, "time": time.time(), **record}, allow_nan=False)
    with (RUNS / run_name / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def records(run_name: str) -> list[dict]:
    """Run `run_name`'s records in file order (its run log, read once; the functions below take it); none for a run not
    yet started."""
    path = RUNS / run_name / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def train_steps(log: list[dict]) -> dict[int, dict]:
    """{step: train record} of a run log, each step's from its latest attempt (attempts only grow down the file)."""
    return {r["step"]: r for r in log if r["kind"] == "train"}


def diverged(log: list[dict]) -> int | None:
    """The step at which a run stopped on a non-finite loss or gradient norm (its log's failure record), or None."""
    failures = [r["step"] for r in log if r["kind"] == "failure"]
    return failures[-1] if failures else None


def curves(log: list[dict]) -> dict[str, list]:
    """train_steps as columns, one per train record field (module docstring) in first-seen order, in step order, None
    at the steps whose record lacks the field (update_rms and param_norm: every 10th step); none for a run without
    train steps (diverged at step 0)."""
    by_step = train_steps(log)
    steps = [by_step[s] for s in sorted(by_step)]
    columns = dict.fromkeys(k for s in steps for k in s if k not in ("run_name", "kind"))
    return {k: [s.get(k) for s in steps] for k in columns}


def bpb(*pooled: dict) -> float:
    """Bits per byte of bits and bytes sums (eval/bpb.py sums) pooled over their context buckets and every argument."""
    return sum(sum(s["bits"]) for s in pooled) / sum(sum(s["bytes"]) for s in pooled)


def mixture_bpb(corpora: dict, weights: dict[str, float], *, buckets: slice = slice(None)) -> float:
    """Bits per byte of val rows drawn with row weights {corpus: weight}: weighted bits per row over weighted bytes per
    row of corpora {corpus: {"bits", "bytes", "rows"}}, over the context buckets `buckets` (all by default)."""
    bits = sum(w * sum(corpora[c]["bits"][buckets]) / corpora[c]["rows"] for c, w in weights.items())
    nbytes = sum(w * sum(corpora[c]["bytes"][buckets]) / corpora[c]["rows"] for c, w in weights.items())
    return bits / nbytes


def offset_bpb(corpora: dict, weights: dict[str, float]) -> list[float | None]:
    """mixture_bpb per byte-offset bucket (eval/bpb.py OFFSETS: the targets after [OFFSETS[i-1], OFFSETS[i]) bytes of
    their segment; None where no target is): bits per byte as a function of the context in bytes (arXiv 2308.12950
    Fig. 4a; 2503.13423 App. B.1), from corpora {corpus: {"offset_bits", "offset_bytes", "rows"}}. A bucket pools the
    documents long enough to reach it, so the curve mixes context use with document composition."""
    n = len(next(iter(corpora.values()))["offset_bits"])
    bits = [sum(w * corpora[c]["offset_bits"][i] / corpora[c]["rows"] for c, w in weights.items()) for i in range(n)]
    nbytes = [sum(w * corpora[c]["offset_bytes"][i] / corpora[c]["rows"] for c, w in weights.items()) for i in range(n)]
    return [b / y if y else None for b, y in zip(bits, nbytes, strict=True)]


def bpb_summary(bits: ArrayLike, nbytes: ArrayLike, *, row_weights: ArrayLike | None = None) -> dict:
    """Bits per byte Σw bits / Σw bytes of sampled val rows (per-row sums; row weights w, 1 by default; rows of 0 bytes
    dropped) with its row-level standard error: the weighted least-squares fit of each row's bits per byte on a
    constant, weights w x its bytes (statsmodels WLS, whose estimate is the ratio of sums), sandwich covariance HC1 (the
    delta-method ratio standard error, times √(n / (n - 1)) as Eq. 1's n - 1)."""
    y = np.asarray(nbytes, dtype=np.float64)
    w = y if row_weights is None else y * np.asarray(row_weights, dtype=np.float64)
    keep = y > 0
    r = np.asarray(bits, dtype=np.float64)[keep] / y[keep]
    fit = sm.WLS(r, np.ones_like(r), weights=w[keep]).fit(cov_type="HC1", use_t=False)
    low, high = fit.conf_int()[0]
    return {"bpb": float(fit.params[0]), "se": float(fit.bse[0]), "interval": (float(low), float(high)),
            "rows": int(keep.sum())}


def mixture_bpb_summary(corpora: dict, weights: dict[str, float]) -> dict:
    """mixture_bpb over all context buckets with its bpb_summary standard error, from corpora {corpus: {"row_bits",
    "row_bytes", "rows"}} (eval/bpb.py val_bits: each sampled row's sums): a row of corpus c weighs weights[c] /
    rows, mixture_bpb's weight of its bits and bytes."""
    names = list(weights)
    return bpb_summary(np.concatenate([corpora[c]["row_bits"] for c in names]),
                       np.concatenate([corpora[c]["row_bytes"] for c in names]),
                       row_weights=np.concatenate([np.full(len(corpora[c]["row_bits"]), weights[c] / corpora[c]["rows"])
                                                   for c in names]))


def parse_outcome(source: str) -> Literal["parses", "indentation", "syntax"]:
    """How a program fails to compile: IndentationError (TabError among them) checked before any other SyntaxError,
    the classification of TokEval's companion post (cimeister.github.io/blog/code_tokenizer_ablations, "How the failing
    generations fail"; linked from arXiv 2608.18062 App. F.3, footnote 15); the parser's overflow (MemoryError,
    RecursionError) is a SyntaxError, as syntax_constraint.syntax_error counts it."""
    with warnings.catch_warnings(action="ignore"):
        try:
            compile(source, "<completion>", "exec")
        except IndentationError:
            return "indentation"
        except (SyntaxError, MemoryError, RecursionError):
            return "syntax"
    return "parses"


def parse_shares(outcomes: list[str]) -> dict[str, float]:
    """Each parse outcome's share of a benchmark's completions."""
    return {kind: outcomes.count(kind) / len(outcomes) for kind in PARSE_OUTCOMES}


def parsed(outcomes: list[str]) -> list[bool]:
    """Per completion, whether it compiles (parse_outcome "parses"): the question scores of the parse shares."""
    return [o == "parses" for o in outcomes]


def measures(evaluation: dict, steps: dict[int, dict], *, repo_level_start: int) -> dict[str, float]:
    """A run's compared measures (bits per byte; shares; throughput) from its final checkpoint's evaluation
    (eval/suite.py evaluate) and its train records (train_steps): code, text and math, the 8K validation bits per byte
    of each group at its corpus weights (eval/bpb.py val_bits "bpb"); solutions, the bits per byte of the HumanEval+,
    MBPP+ (EvalPlus, arXiv 2305.01210) and HumanEval-X C++ (arXiv 2303.17568) reference solutions given their prompts,
    pooled (eval/bpb.py solution_bits); repo_32k, the 32K repository bits per byte of the code group, and
    repo_beyond_8k, the same over the targets beyond the file-level context (the last context bucket) at the
    evaluation's code corpus weights; unparsed, indentation and syntax, the shares of HumanEval+ and MBPP+ problems,
    pooled, whose healed greedy completion does not compile, by an IndentationError and by another SyntaxError
    (parse_outcome), and pass@1, their greedy base pass@1, pooled; tokens_per_second and mfu, the medians over the
    file-level stage's steps (before repo_level_start) of the step's tokens per second and executed MFU."""
    parse = evaluation["parse"].values()
    shares = parse_shares([o for p in parse for o in p["outcomes"]])
    file_level = [r for step, r in steps.items() if step < repo_level_start]
    return {"code": evaluation["8k"]["bpb"]["code"],
            "solutions": bpb(*evaluation["solution"].values()),
            "repo_32k": evaluation["32k"]["bpb"]["code"],
            "repo_beyond_8k": mixture_bpb(
                evaluation["32k"]["corpora"],
                {c: w for c, w in evaluation["32k"]["weights"].items() if c in REPO_LEVEL_STAGE.code},
                buckets=slice(-1, None)),
            "text": evaluation["8k"]["bpb"]["text"], "math": evaluation["8k"]["bpb"]["math"],
            "unparsed": 1 - shares["parses"], "indentation": shares["indentation"], "syntax": shares["syntax"],
            "pass@1": statistics.fmean(s[0] == "pass" for p in parse for s in p["base_status"]),
            "tokens_per_second": statistics.median(r["tokens_per_second"] for r in file_level),
            "mfu": statistics.median(r["mfu"] for r in file_level)}


def measure_errors(evaluation: dict) -> dict[str, dict]:
    """Error bars of measures' evaluation measures (Miller, arXiv 2411.00640): the bits per byte by their val rows
    (eval/bpb.py "errors": mixture_bpb_summary per group; solutions: bpb_summary over the reference solutions), the
    parse shares and pass@1 by summary of their per-problem scores (Eq. 1)."""
    parse = evaluation["parse"].values()
    outcomes = [o for p in parse for o in p["outcomes"]]
    solutions = evaluation["solution"].values()
    return {"code": evaluation["8k"]["errors"]["code"], "text": evaluation["8k"]["errors"]["text"],
            "math": evaluation["8k"]["errors"]["math"], "repo_32k": evaluation["32k"]["errors"]["code"],
            "solutions": bpb_summary(np.concatenate([s["row_bits"] for s in solutions]),
                                     np.concatenate([s["row_bytes"] for s in solutions])),
            "unparsed": summary([not p for p in parsed(outcomes)]),
            "indentation": summary([o == "indentation" for o in outcomes]),
            "syntax": summary([o == "syntax" for o in outcomes]),
            "pass@1": summary([s[0] == "pass" for p in parse for s in p["base_status"]])}


# CrossCodeEval's comment stripping (eval_utils.py remove_comments: "#" and "//" to the line end, strings included),
# string literals (string_pattern, removed before identifiers are read) and identifiers (IDENTIFIER_REGEX matched at
# the start of NLTK RegexpTokenizer(r"\w+")'s words); Python's keywords as dpu-utils keywordlist.py lists them
# (keyword.kwlist without True and False)
COMMENT = re.compile(r"#.*|//.*")
STRING = re.compile(r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\'')
WORD, IDENTIFIER = re.compile(r"\w+", re.UNICODE | re.MULTILINE | re.DOTALL), re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")
PYTHON_KEYWORDS = frozenset(keyword.kwlist) - {"True", "False"}


def line_exact_match(prediction: str, reference: str) -> bool:
    """Whether a middle equals the reference up to surrounding whitespace (santacoder_fim.py: g.strip() ==
    reference.strip())."""
    return prediction.strip() == reference.strip()


def code_lines(code: str) -> list[str]:
    """The stripped non-empty lines of code without its comments (cceval process_examples)."""
    return [line.strip() for line in COMMENT.sub("", code).split("\n") if line.strip()]


def exact_match(prediction: str, reference: str) -> bool:
    """CrossCodeEval's exact match: equal code_lines."""
    return code_lines(prediction) == code_lines(reference)


def edit_similarity(prediction: str, reference: str) -> int:
    """CrossCodeEval's edit similarity of comment-stripped code in [0, 100]: fuzzywuzzy 0.18 fuzz.ratio of the stripped
    strings as cceval runs it (its requirements install fuzzywuzzy without python-Levenshtein, so difflib's
    SequenceMatcher, autojunk included): 100 when equal, 0 when either is empty, else round(100 ratio), Python's
    round (fuzzywuzzy utils.intr)."""
    a, b = COMMENT.sub("", prediction).strip(), COMMENT.sub("", reference).strip()
    if a == b:
        return 100
    if not a or not b:
        return 0
    return round(100 * difflib.SequenceMatcher(None, a, b).ratio())


def identifiers(code: str) -> list[str]:
    """The identifiers of comment-stripped Python code in order, repeats kept (cceval extract_identifiers)."""
    return [w for w in WORD.findall(STRING.sub("", COMMENT.sub("", code)))
            if IDENTIFIER.match(w) and w not in PYTHON_KEYWORDS]


def identifier_match(prediction: str, reference: str) -> tuple[bool, float]:
    """CrossCodeEval's identifier exact match (equal identifier lists) and F1 (over the sets; 0 when both are empty)."""
    p, r = identifiers(prediction), identifiers(reference)
    tp = len(set(p) & set(r))
    wrong = len(set(p) ^ set(r))
    return p == r, 2 * tp / (2 * tp + wrong) if tp or wrong else 0.0


def completion_scores(predictions: Sequence[str], references: Sequence[str]) -> dict[str, list[float]]:
    """Per sample: line_exact_match, exact_match, edit_similarity, identifier exact match and F1 (cceval's per-sample
    detailed_results), each a 0-1 or 0-100 question score for summary."""
    out = {"line_em": [], "em": [], "es": [], "id_em": [], "id_f1": []}
    for p, r in zip(predictions, references, strict=True):
        id_em, id_f1 = identifier_match(p, r)
        scores = (line_exact_match(p, r), exact_match(p, r), edit_similarity(p, r), id_em, id_f1)
        for k, v in zip(out, scores, strict=True):
            out[k].append(float(v))
    return out


def pass_at_k(passed: Sequence[Sequence[bool]], k: int) -> np.ndarray:
    """Each problem's pass@k estimate from its samples' outcomes (at least k samples each)."""
    n = np.array([len(p) for p in passed])
    return estimate_pass_at_k(n, np.array([sum(p) for p in passed]), k)


def summary(scores: ArrayLike, clusters: ArrayLike | None = None) -> dict:
    """{"mean", "se", "interval", "n", "clusters"} of question scores: the least-squares fit of the scores on a
    constant (statsmodels OLS, normal reference distribution). Unclustered, its standard error is Miller's Eq. 1
    (sample variance over n - 1); clustered by `clusters` (each question's cluster label), the Liang-Zeger
    cluster-robust CR0 (cov_type "cluster" without the small-sample correction), Miller's Eq. 4 but for Eq. 4's (n - 1)
    in its diagonal term; the 95% interval (Eq. 5-6). A constant score has se 0 and its interval the point."""
    s = np.asarray(scores, dtype=np.float64)
    mean = float(s.mean())
    groups = None if clusters is None else np.unique(np.asarray(clusters), return_inverse=True)[1]
    out = {"n": int(s.size), "clusters": None if groups is None else int(groups.max() + 1), "mean": mean}
    if np.ptp(s) == 0:
        return out | {"se": 0.0, "interval": (mean, mean)}
    model = sm.OLS(s, np.ones_like(s))
    fit = (model.fit(use_t=False) if groups is None else
           model.fit(cov_type="cluster", cov_kwds={"groups": groups, "use_correction": False}, use_t=False))
    low, high = fit.conf_int()[0]
    return out | {"se": float(fit.bse[0]), "interval": (float(low), float(high))}


# The GPU's dense BF16 peak: H100 SXM5 / GH200 (NVIDIA H100 datasheet, 989.4 TFLOP/s without sparsity)
PEAK_FLOPS = 989.4e12


def model_flops_per_token(*, matmul_params: int, hidden_size: int, windows: Sequence[int], context: int) -> int:
    """Nominal training FLOPs per token (forward and backward) of PaLM's MFU: 6 x the matmul parameters (decoder
    layers' and lm_head's weights; embedding tables are lookups, norm gains elementwise) + 12 hidden_size per key per
    layer, each layer's most keys a query attends: itself and up to `window` keys before it (model/modeling.py
    layer_windows, as attended_keys counts them) within the context, min(window + 1, context)."""
    return 6 * matmul_params + 12 * hidden_size * sum(min(window + 1, context) for window in windows)


def attended_keys(lengths: np.ndarray, window: int) -> int:
    """Σ over segments of Σ_p min(p, window + 1), p = 1..length: the keys each query attends causally within its
    segment, itself and up to `window` keys before it (model/modeling.py layer_windows)."""
    lengths = np.asarray(lengths, dtype=np.int64)
    m = np.minimum(lengths, window + 1)
    return int((m * (m + 1) // 2 + (lengths - m) * (window + 1)).sum())


def executed_model_flops(lengths: np.ndarray, *, matmul_params: int, hidden_size: int, windows: Sequence[int]) -> int:
    """Training FLOPs (forward and backward) a micro-batch of segments `lengths` (its cu_seqlens' differences, every row
    position in one) executes: 6 x the matmul parameters per position + 12 hidden_size per attended_keys per layer
    (QK^T and PV, 4 hidden_size per key forward, twice that backward; FA4's recomputation of the scores not counted)."""
    keys = sum(count * attended_keys(lengths, window) for window, count in Counter(windows).items())
    return 6 * matmul_params * int(np.asarray(lengths).sum()) + 12 * hidden_size * keys


def mfu(flops: float, seconds: float) -> float:
    """Model FLOPs utilization: the model FLOPs done per second over PEAK_FLOPS."""
    return flops / seconds / PEAK_FLOPS


def parabola_vertex(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """(curvature, vertex) of the least-squares parabola a x² + b x + c through the points (numpy polyfit, degree 2):
    (a, -b / (2a))."""
    a, b, _ = np.polyfit(x, y, 2)
    return float(a), float(-b / (2 * a))


def latency(starts: ArrayLike, firsts: ArrayLike, ends: ArrayLike, new_tokens: ArrayLike, *, duration: float) -> dict:
    """Decoding latency of requests (seconds; vLLM calculate_metrics): ttft (first token - start), tpot ((end - first)
    / (new tokens - 1), requests of more than one token; None if no request has more than one, where vLLM reports 0),
    e2el (end - start), each {"mean", "median", "p99"} (the percentile serve.py reports by default,
    --metric-percentiles 99), and output_throughput (output tokens / duration)."""
    start, first, end = (np.asarray(t, dtype=np.float64) for t in (starts, firsts, ends))
    n = np.asarray(new_tokens)
    many = n > 1

    def describe(x: np.ndarray) -> dict:
        return {"mean": float(x.mean()), "median": float(np.median(x)), "p99": float(np.percentile(x, 99))}

    return {"ttft": describe(first - start),
            "tpot": describe((end - first)[many] / (n[many] - 1)) if many.any() else None,
            "e2el": describe(end - start), "output_throughput": float(n.sum() / duration),
            "requests": int(n.size)}


# Tokenizer metrics (TokEval): tokenizer_analysis (cimeister/tokenizer-intrinsic-evals 7d03a61, arXiv 2608.18062) run on
# the repository's tiktoken vocabularies (corpus/tokenizer.py Tokenizer). Its package imports transformers, so it is
# imported inside the functions: this module's import stays light for the forkserver workers that load it.


def token_offsets(text: str, lengths: np.ndarray) -> list[tuple[int, int]]:
    """Each token's (start, end) character range in `text` from its byte length: the characters its bytes belong to,
    so a token holding part of a multi-byte character overlaps the next token's range at that character (as
    HuggingFace ByteLevel offsets do). The tokens' bytes are text's UTF-8 bytes in order (its encoding)."""
    data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    ends = np.cumsum(lengths)
    # the character each byte belongs to: the lead and ASCII bytes up to it, less one
    char_of_byte = np.cumsum((data & 0xC0) != 0x80) - 1
    return list(zip(char_of_byte[ends - lengths].tolist(), (char_of_byte[ends - 1] + 1).tolist(), strict=True))


def tokeval_wrapper(name: str, tokenizer):
    """A corpus/tokenizer.py Tokenizer as a TokEval TokenizerWrapper (core/tokenizer_wrapper.py). encode is tiktoken's
    encode_ordinary (special tokens' text encodes as ordinary text, as corpus/tokenize.py does); decode joins the
    tokens' bytes; pretokenize is the tokenizer's split pattern.

    Vocabulary keys are GPT-2's byte-to-unicode rendering of each mergeable token's bytes (transformers'
    bytes_to_unicode, openai/gpt-2 src/encoder.py: HuggingFace's ByteLevel form of tiktoken vocabularies), a bijection
    on byte strings, so keys are distinct and keep partial characters. TokEval
    reads the keys as this form: utf8_token_integrity (metrics/utf8_integrity.py _detect_gpt2_encoding) recognizes the
    GPT-2 marker keys and maps each key back to its exact bytes; indentation_consistency (metrics/code_ast.py
    _is_whitespace_token) and BaseMetrics._build_char_decode_table read a space, line feed, tab and carriage return as
    Ġ, Ċ, ĉ, č. Special tokens (the ids after the mergeable tokens, in order: corpus/tokenizer.py) keep their text as
    key."""
    import regex
    from tokenizer_analysis.core.tokenizer_wrapper import TokenizerWrapper
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    byte_to_unicode = bytes_to_unicode()

    class Wrapper(TokenizerWrapper):
        def __init__(self) -> None:
            self._enc, self._mergeable, self._lengths = tokenizer.enc, tokenizer.bos_token_id, tokenizer.token_lengths
            self._strings = ["".join(byte_to_unicode[b] for b in token)
                             for token in tokenizer.token_bytes[:self._mergeable]] + list(tokenizer.special_tokens)
            self._vocab = {s: i for i, s in enumerate(self._strings)}
            self._split = regex.compile(tokenizer.pattern)

        def get_name(self) -> str:
            return name

        def get_vocab_size(self) -> int:
            return tokenizer.vocab_size

        def get_vocab(self) -> dict[str, int]:
            return dict(self._vocab)

        def can_encode(self) -> bool:
            return True

        def encode(self, text: str) -> list[int]:
            return self._enc.encode_ordinary(text)

        def can_pretokenize(self) -> bool:
            return True

        def pretokenize(self, text: str) -> list[str]:
            return self._split.findall(text)

        def pretokenize_with_spans(self, text: str) -> list[tuple[str, tuple[int, int]]]:
            return [(m.group(), m.span()) for m in self._split.finditer(text)]

        def get_special_token_strings(self) -> set[str]:
            return set(tokenizer.special_tokens)

        @classmethod
        def from_config(cls, name: str, config: dict):
            """{"path": a tokenizer directory (corpus/tokenizer.py get_tokenizer)}."""
            from pathlib import Path

            from corpus.tokenizer import get_tokenizer
            return tokeval_wrapper(name, get_tokenizer(Path(config["path"])))

        def can_decode(self) -> bool:
            return True

        def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
            if skip_special_tokens:
                token_ids = [i for i in token_ids if i < self._mergeable]
            return self._enc.decode_bytes(token_ids).decode("utf-8", errors="replace")

        def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
            ids = self._enc.encode_ordinary(text)
            return ids, token_offsets(text, self._lengths[np.asarray(ids, dtype=np.int64)])

        def encode_batch_with_offsets(self, texts: list[str]) -> list[tuple[list[int], list[tuple[int, int]]]]:
            batch = self._enc.encode_ordinary_batch(texts)
            return [(ids, token_offsets(text, self._lengths[np.asarray(ids, dtype=np.int64)]))
                    for text, ids in zip(texts, batch, strict=True)]

        def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
            return [self._strings[i] for i in token_ids]

        def get_special_token_ids(self) -> set[int]:
            return set(tokenizer.special_tokens.values())

        def get_unk_token_id(self) -> None:
            return None  # byte-level: every byte has a token

    return Wrapper()


def tokeval(tokenizers: dict, texts: dict[str, list[str]], code: dict[str, list[str]], math: list[str]) -> dict:
    """TokEval's full analysis (main.py UnifiedTokenizerAnalyzer.run_analysis) of corpus/tokenizer.py Tokenizers
    ({name: Tokenizer}) in its published form (cli/run_analysis.py normalize_results, non-finite floats as null).

    texts: {language: documents}, the prose corpus; its language names must differ from the code languages' and from
    "math", the domain keys TokEval reports the code and math corpora under (metrics/basic.py). code: {"python" | "cpp"
    | ...: documents}, the code corpus (code_ast_config, read through CodeDataLoader: CRLF to LF, trailing whitespace
    stripped), which the AST, identifier and indentation metrics read. math: the math corpus (math_data_path), which the
    digit metrics and operator isolation's math domain read. MorphScore off (it needs its submodule and data), plots
    off, CER unbudgeted (cer_time_budget_s 0: every text decodes exactly). The tree-sitter parsers run in subprocesses
    (metrics/code_ast.py); TokEval's own corpus files are written to a temporary directory."""
    import tempfile
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizer_analysis import UnifiedTokenizerAnalyzer
    from tokenizer_analysis.cli.run_analysis import _convert_for_json_public, normalize_results
    from tokenizer_analysis.core.input_providers import create_input_provider
    from tokenizer_analysis.core.input_utils import create_simple_specifications

    wrappers = {name: tokeval_wrapper(name, tokenizer) for name, tokenizer in tokenizers.items()}
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for language, documents in code.items():
            pq.write_table(pa.table({"content": documents}), directory / f"code_{language}.parquet")
        (directory / "math.json").write_text(json.dumps({"texts": math}), encoding="utf-8")
        provider = create_input_provider(create_simple_specifications(
            {name: (wrapper, texts) for name, wrapper in wrappers.items()}))
        analyzer = UnifiedTokenizerAnalyzer(
            provider, plot_save_dir=str(directory / "plots"),
            code_ast_config={language: str(directory / f"code_{language}.parquet") for language in code},
            math_data_path=str(directory / "math.json"))
        results = analyzer.run_analysis(save_plots=False, include_morphscore=False, verbose=False, cer_time_budget_s=0)
    return _convert_for_json_public(normalize_results(results))
