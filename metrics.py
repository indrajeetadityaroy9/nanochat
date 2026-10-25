"""
Metrics: the run log every training process writes, and the measures checkpoints are scored and runs compared by.
Torch-free, so training, evaluation and the ablations' analyses import it alike.

The run log, RUNS/<run_name>/metrics.jsonl (run_name: run.py run_name): UTF-8 JSON lines, each an object parseable on
its own: run_name, attempt (the run's process start it belongs to, 0-based: a resumed run repeats the steps after its
checkpoint, and a step's record from the latest attempt supersedes the earlier ones; records are never rewritten),
kind, time (Unix seconds), then the kind's fields. NaN and infinities are not JSON and are never written.
run (each process start): step (the first step it trains), plan (schedule.RunPlan), seed (initialization and data
    order), config (run.py RunConfig, the run's fields), data_key (corpus/tokenizer.py Tokenizer.data_key of
    the run's tokenizer; with plan, what run.py's stale guard compares), model (model/modeling.py ModelConfig at
    the stage of the weights it starts from: the file-level stage, or its checkpoint's), num_parameters (the model's
    parameters), flops_per_token (model/modeling.py non_embedding_flops_per_token of that model: DeepSeek LLM's M,
    arXiv 2401.02954 §3.2 Eq. 2), max_grad_norm (the gradient clip threshold), learning_rate and weight_decay
    ({"adamw": [per group], "muon": [per group]}, as the optimizers hold them; a resumed process's come from its
    checkpoint), stage (corpus/mixture.py Stage as the model runs it, mixture as {corpus: row weight}).
stage (switch to the repo-level stage): step, stage.
train (each optimizer step, 0-based): step; num_input_tokens_seen (row positions after it, padding included); loss
    (mean next-token loss in nats over the step's valid labels, fp64); learning_rate (LambdaLR's at the step);
    grad_norm (global L2 norm of the gradient before clipping, the norm global-norm clipping thresholds:
    arXiv 1211.5063 §3.2); tokens_per_second (row positions per second between the step's two CUDA
    synchronizations: the observed throughput of PaLM's model FLOPs utilization, arXiv 2204.02311 App. B).
end (the last step trained): step; peak_memory_bytes (torch.cuda.max_memory_allocated of the process: the memory
    footprint FlashAttention reports, arXiv 2205.14135).
failure (a non-finite loss or gradient norm; the run stops): step, loss and grad_norm (null).

Bits per byte (bpb, mixture_bpb): summed next-token loss in bits (nats / ln 2) over the summed UTF-8 bytes of the
targets (The Pile's bits per UTF-8 byte, arXiv 2101.00027 §3.1; tokenizer-independent, so models and vocabularies
compare), from the bits and bytes sums eval/bpb.py computes per context bucket, pooled exactly.

Parse outcome (parse_outcome, parse_shares): how a completion compiles. Compared measures (measures): a run's
measures from its final checkpoint's evaluation and its train records.
"""
import json
import statistics
import time
import warnings
from typing import Literal

from corpus.mixture import REPO_LEVEL_STAGE
from paths import RUNS

PARSE_OUTCOMES = ("parses", "indentation", "syntax")


def append(run_name: str, record: dict, *, attempt: int) -> None:
    line = json.dumps({"run_name": run_name, "attempt": attempt, "time": time.time(), **record}, allow_nan=False)
    with (RUNS / run_name / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def records(run_name: str) -> list[dict]:
    """Run `run_name`'s records in file order; none for a run not yet started."""
    path = RUNS / run_name / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def train_steps(run_name: str) -> dict[int, dict]:
    """{step: train record}, each step's from its latest attempt (attempts only grow down the file)."""
    return {r["step"]: r for r in records(run_name) if r["kind"] == "train"}


def diverged(run_name: str) -> int | None:
    """The step at which run `run_name` stopped on a non-finite loss or gradient norm (its failure record), or None."""
    failures = [r["step"] for r in records(run_name) if r["kind"] == "failure"]
    return failures[-1] if failures else None


def curves(run_name: str) -> dict[str, list]:
    """train_steps as columns, one per train record field (step, num_input_tokens_seen, loss, learning_rate,
    grad_norm, tokens_per_second, attempt, time), in step order; none for a run without train steps (diverged at step
    0)."""
    by_step = train_steps(run_name)
    steps = [by_step[s] for s in sorted(by_step)]
    return {k: [s[k] for s in steps] for k in (steps[0] if steps else ()) if k not in ("run_name", "kind")}


def bpb(*pooled: dict) -> float:
    """Bits per byte of bits and bytes sums (eval/bpb.py sums) pooled over their context buckets and every argument."""
    return sum(sum(s["bits"]) for s in pooled) / sum(sum(s["bytes"]) for s in pooled)


def mixture_bpb(corpora: dict, weights: dict[str, float], *, buckets: slice = slice(None)) -> float:
    """Bits per byte of val rows drawn with row weights {corpus: weight}: weighted bits per row over weighted bytes per
    row of corpora {corpus: {"bits", "bytes", "rows"}}, over the context buckets `buckets` (all by default)."""
    bits = sum(w * sum(corpora[c]["bits"][buckets]) / corpora[c]["rows"] for c, w in weights.items())
    nbytes = sum(w * sum(corpora[c]["bytes"][buckets]) / corpora[c]["rows"] for c, w in weights.items())
    return bits / nbytes


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


def measures(evaluation: dict, steps: dict[int, dict], *, repo_level_start: int) -> dict[str, float]:
    """A run's compared measures (bits per byte; shares; tokens per second) from its final checkpoint's evaluation
    (eval/suite.py evaluate) and its train records (train_steps): code, text and math, the 8K validation bits per byte
    of each group at its corpus weights (eval/bpb.py val_bits "bpb"); solutions, the bits per byte of the HumanEval+,
    MBPP+ (EvalPlus, arXiv 2305.01210) and HumanEval-X C++ (arXiv 2303.17568) reference solutions given their prompts,
    pooled (eval/bpb.py solution_bits); repo_32k, the 32K repository bits per byte of the code group, and
    repo_beyond_8k, the same over the targets beyond the file-level context (the last context bucket) at the
    evaluation's code corpus weights; unparsed, indentation and syntax, the shares of HumanEval+ and MBPP+ problems,
    pooled, whose healed greedy completion does not compile, by an IndentationError and by another SyntaxError
    (parse_outcome), and pass@1, their greedy base pass@1, pooled; tokens_per_second, the median over the file-level
    stage's steps (before repo_level_start) of the step's tokens per second."""
    parsed = evaluation["parse"].values()
    problems = sum(len(p["outcomes"]) for p in parsed)

    def share(name: str) -> float:
        return sum(p[name] * len(p["outcomes"]) for p in parsed) / problems

    return {"code": evaluation["8k"]["bpb"]["code"],
            "solutions": bpb(*evaluation["solution"].values()),
            "repo_32k": evaluation["32k"]["bpb"]["code"],
            "repo_beyond_8k": mixture_bpb(
                evaluation["32k"]["corpora"],
                {c: w for c, w in evaluation["32k"]["weights"].items() if c in REPO_LEVEL_STAGE.code},
                buckets=slice(-1, None)),
            "text": evaluation["8k"]["bpb"]["text"], "math": evaluation["8k"]["bpb"]["math"],
            "unparsed": 1 - share("parses"), "indentation": share("indentation"), "syntax": share("syntax"),
            "pass@1": share("base pass@1"),
            "tokens_per_second": statistics.median(r["tokens_per_second"] for step, r in steps.items()
                                                   if step < repo_level_start)}
