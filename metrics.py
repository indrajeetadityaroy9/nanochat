"""
Metrics, torch-free: the run log, bits per byte (arXiv 2101.00027 §3.1), parse outcomes, completion similarity
(arXiv 2310.11248 §3.2), pass@k (arXiv 2107.03374 §2.1), error bars and paired comparisons (arXiv 2411.00640), model
FLOPs utilization (arXiv 2204.02311 App. B) and decoding latency.
"""
import difflib
import json
import keyword
import re
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from typing import Literal

import numpy as np
import statsmodels.api as sm
from human_eval_infilling.evaluation import estimate_pass_at_k
from numpy.typing import ArrayLike
from scipy.stats import norm
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from corpus.mixture import REPO_LEVEL_STAGE
from paths import RUNS

# eval/code_eval.py truncated (cut at the token budget, not compiled), else parse_outcome's
PARSE_OUTCOMES = ("parses", "indentation", "syntax", "truncated")


def append(run_name: str, record: dict, *, attempt: int) -> None:
    line = json.dumps({"run_name": run_name, "attempt": attempt, "time": time.time(), **record}, allow_nan=False)
    with (RUNS / run_name / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def records(run_name: str) -> list[dict]:
    """Run `run_name`'s records (RUNS/<run_name>/metrics.jsonl) in file order; none for a run not yet started."""
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
    """train_steps as columns in step order, None where a record lacks the field; none without train steps."""
    by_step = train_steps(log)
    steps = [by_step[s] for s in sorted(by_step)]
    columns = dict.fromkeys(k for s in steps for k in s if k not in ("run_name", "kind"))
    return {k: [s.get(k) for s in steps] for k in columns}


def bpb(*pooled: dict) -> float:
    """Bits per byte of bits and bytes sums (eval/bpb.py sums) pooled over their context buckets and every argument."""
    return sum(sum(s["bits"]) for s in pooled) / sum(sum(s["bytes"]) for s in pooled)


def mixture_bpb(corpora: dict, weights: dict[str, float], *, buckets: slice = slice(None)) -> float:
    """Bits per byte of val rows drawn with row weights w: Σ w bits / rows over Σ w bytes / rows, over `buckets`.
    corpora: {corpus: {"bits", "bytes", "rows"}}; weights: {corpus: w}."""
    bits = sum(w * sum(corpora[c]["bits"][buckets]) / corpora[c]["rows"] for c, w in weights.items())
    nbytes = sum(w * sum(corpora[c]["bytes"][buckets]) / corpora[c]["rows"] for c, w in weights.items())
    return bits / nbytes


def offset_bpb(corpora: dict, weights: dict[str, float]) -> list[float | None]:
    """mixture_bpb per byte-offset bucket, None where no target is (arXiv 2308.12950 Fig. 4a; 2503.13423 App. B.1)."""
    n = len(next(iter(corpora.values()))["offset_bits"])
    bits = [sum(w * corpora[c]["offset_bits"][i] / corpora[c]["rows"] for c, w in weights.items()) for i in range(n)]
    nbytes = [sum(w * corpora[c]["offset_bytes"][i] / corpora[c]["rows"] for c, w in weights.items()) for i in range(n)]
    return [b / y if y else None for b, y in zip(bits, nbytes, strict=True)]


def mixture_bpb_summary(corpora: dict, weights: dict[str, float]) -> dict:
    """mixture_bpb of rows with its standard error and 95% normal interval (arXiv 2411.00640 Eq. 3; the strata and the
    ratio's delta method are this code's). corpora:
    {corpus: {"row_bits", "row_bytes", "row_clusters", "rows"}}, the strata of a sample at fixed weights; within one,
    clusters of rows are independent (eval/bpb.py row_clusters). With a_c = weights[c] / rows_c, R = Σ a_c bits_c /
    Σ a_c bytes_c and, by the delta method, se² = Σ_c a_c² m_c s_c² / (Σ_c a_c bytes_c)², s_c² the sample variance
    (ddof 1) of corpus c's m_c cluster sums of bits - R bytes (clustered errors, arXiv 2411.00640 Eq. 4, per
    stratum)."""
    scale = {c: w / corpora[c]["rows"] for c, w in weights.items()}
    nbytes = sum(a * np.sum(corpora[c]["row_bytes"]) for c, a in scale.items())
    bpb = float(sum(a * np.sum(corpora[c]["row_bits"]) for c, a in scale.items()) / nbytes)
    variance, clusters = 0.0, 0
    for c, a in scale.items():
        residuals = np.asarray(corpora[c]["row_bits"]) - bpb * np.asarray(corpora[c]["row_bytes"])
        sums = np.bincount(corpora[c]["row_clusters"], weights=residuals)
        variance += a**2 * sums.size * sums.var(ddof=1)
        clusters += sums.size
    se = float(np.sqrt(variance) / nbytes)
    half = statistics.NormalDist().inv_cdf(0.975) * se
    return {"bpb": bpb, "se": se, "interval": (bpb - half, bpb + half),
            "rows": sum(corpora[c]["rows"] for c in weights), "clusters": clusters}


def paired_rows(a: dict, b: dict) -> dict:
    """{stratum: b's rows with each row's bits minus a's}, for two scorings of identical rows (equal row_bytes and
    row_clusters, else ValueError): mixture_bpb_summary of them is the paired change b − a in bits per byte with its
    clustered standard error (arXiv 2411.00640 §4.2 Eq. 8, within each stratum)."""
    out = {}
    for c, x in a.items():
        y = b[c]
        if list(x["row_bytes"]) != list(y["row_bytes"]) or list(x["row_clusters"]) != list(y["row_clusters"]):
            raise ValueError(f"{c}: the two scorings hold different rows")
        out[c] = {"row_bits": np.subtract(y["row_bits"], x["row_bits"]), "row_bytes": x["row_bytes"],
                  "row_clusters": x["row_clusters"], "rows": x["rows"]}
    return out


def parse_outcome(source: str) -> Literal["parses", "indentation", "syntax"]:
    """How a program compiles: "indentation" (IndentationError, checked first), "syntax" or "parses".
    TokEval companion post's classification (linked from arXiv 2608.18062 App. F.3, fn. 15); parser overflow counts as
    syntax, and so does CPython 3.12.3's ValueError for f"{x:{y=}}" (python/cpython#121130, fixed in 3.12.5)."""
    try:
        compile(source, "<completion>", "exec")
    except IndentationError:
        return "indentation"
    except (SyntaxError, MemoryError, RecursionError, ValueError):
        return "syntax"
    return "parses"


def parse_shares(outcomes: list[str]) -> dict[str, float]:
    """Each parse outcome's share of a benchmark's completions."""
    return {kind: outcomes.count(kind) / len(outcomes) for kind in PARSE_OUTCOMES}


def measures(evaluation: dict, steps: dict[int, dict], *, repo_level_start: int) -> dict[str, float]:
    """A run's compared measures from its final checkpoint's evaluation (eval/suite.py evaluate) and train records.
    tokens_per_second and mfu: medians over the file-level stage's steps (before repo_level_start)."""
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
            "unparsed": shares["indentation"] + shares["syntax"], "indentation": shares["indentation"],
            "syntax": shares["syntax"], "truncated": shares["truncated"],
            "pass@1": statistics.fmean(s[0] == "pass" for p in parse for s in p["base_status"]),
            "tokens_per_second": statistics.median(r["tokens_per_second"] for r in file_level),
            "mfu": statistics.median(r["mfu"] for r in file_level)}


def solution_strata(solution: dict) -> dict:
    """An evaluation's "solution" rows as two strata: HumanEval (HumanEval+ and HumanEval-X C++ hold its problems in
    one order, HumanEval/i and CPP/i, a cluster each) and MBPP+ (a cluster per problem); mixture_bpb_summary at weights
    {stratum: rows} is their pooled bpb."""
    plus, cpp, mbpp = (solution[name] for name in ("humaneval_plus", "humaneval_x_cpp", "mbpp_plus"))
    problems = len(plus["row_bits"])
    return {"humaneval": {k: [*plus[k], *cpp[k]] for k in ("row_bits", "row_bytes")}
            | {"row_clusters": [*range(problems)] * 2, "rows": 2 * problems},
            "mbpp": {"row_bits": mbpp["row_bits"], "row_bytes": mbpp["row_bytes"],
                     "row_clusters": range(len(mbpp["row_bits"])), "rows": len(mbpp["row_bits"])}}


def measure_errors(evaluation: dict) -> dict[str, dict]:
    """Error bars of measures' evaluation measures (arXiv 2411.00640): bpb by mixture_bpb_summary, shares by problem;
    solutions over solution_strata at their solution counts (the pooled bpb)."""
    parse = evaluation["parse"].values()
    outcomes = [o for p in parse for o in p["outcomes"]]
    strata = solution_strata(evaluation["solution"])
    return {"code": evaluation["8k"]["errors"]["code"], "text": evaluation["8k"]["errors"]["text"],
            "math": evaluation["8k"]["errors"]["math"], "repo_32k": evaluation["32k"]["errors"]["code"],
            "solutions": mixture_bpb_summary(strata, {name: s["rows"] for name, s in strata.items()}),
            "unparsed": summary([o in ("indentation", "syntax") for o in outcomes]),
            "indentation": summary([o == "indentation" for o in outcomes]),
            "syntax": summary([o == "syntax" for o in outcomes]),
            "truncated": summary([o == "truncated" for o in outcomes]),
            "pass@1": summary([s[0] == "pass" for p in parse for s in p["base_status"]])}


# CrossCodeEval's patterns (arXiv 2310.11248; its released scripts/eval_utils.py remove_comments, string_pattern,
# IDENTIFIER_REGEX); keywords as dpu-utils keywordlist.py lists them (keyword.kwlist without True and False)
COMMENT = re.compile(r"#.*|//.*")
STRING = re.compile(r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\'')
WORD, IDENTIFIER = re.compile(r"\w+", re.UNICODE | re.MULTILINE | re.DOTALL), re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")
PYTHON_KEYWORDS = frozenset(keyword.kwlist) - {"True", "False"}


def line_exact_match(prediction: str, reference: str) -> bool:
    """Whether a middle equals the reference up to surrounding whitespace (arXiv 2301.03988 §5.4)."""
    return prediction.strip() == reference.strip()


def code_lines(code: str) -> list[str]:
    """The stripped non-empty lines of code without its comments (cceval process_examples)."""
    return [line.strip() for line in COMMENT.sub("", code).split("\n") if line.strip()]


def exact_match(prediction: str, reference: str) -> bool:
    """CrossCodeEval's exact match: equal code_lines."""
    return code_lines(prediction) == code_lines(reference)


def edit_similarity(prediction: str, reference: str) -> int:
    """CrossCodeEval's edit similarity in [0, 100]: fuzzywuzzy 0.18 fuzz.ratio without python-Levenshtein.
    That is difflib's SequenceMatcher (autojunk included), rounded by Python's round (fuzzywuzzy utils.intr)."""
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
    """Per-sample line_em, em, es (0-100), id_em and id_f1 question scores (cceval detailed_results)."""
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
    """{"mean", "se", "interval", "n", "clusters"} of question scores by OLS on a constant (arXiv 2411.00640).
    Unclustered: Eq. 1; clustered: Liang-Zeger CR0 without small-sample correction; 95% normal interval (Eq. 3)."""
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


def pass_cell(passed: Sequence[Sequence[bool]]) -> dict:
    """A pass@1 table cell: summary of each problem's mean pass over its samples (arXiv 2411.00640 §3.1: a question's
    K answers averaged first, the error over the question means by Eq. 1), and, when every problem has one sample,
    "wilson": the Wilson 95% score interval of the passes (statsmodels proportion_confint), inside [0, 1] and not
    degenerate at 0 or all passes, where summary's normal interval is."""
    cell = summary([statistics.fmean(p) for p in passed])
    if all(len(p) == 1 for p in passed):
        low, high = proportion_confint(sum(p[0] for p in passed), len(passed), alpha=0.05, method="wilson")
        cell["wilson"] = (float(low), float(high))
    return cell


def evalplus_passed(record: dict) -> tuple[list[list[bool]], list[list[bool]]]:
    """An EvalPlus record's base and plus pass outcomes [problem][sample]: base status "pass"; plus, both "pass"."""
    base = [[s == "pass" for s in ss] for ss in record["base_status"]]
    plus = [[s == t == "pass" for s, t in zip(ss, ts, strict=True)]
            for ss, ts in zip(record["base_status"], record["plus_status"], strict=True)]
    return base, plus


def evalplus_average(humaneval: dict, mbpp: dict) -> dict:
    """Qwen3's EvalPlus score (arXiv 2505.09388 §3.3: the average of HumanEval, MBPP, HumanEval+ and MBPP+) of one
    mode's humaneval_plus and mbpp_plus records, as a summary cell. With a task's score t = (base + plus) / 2 of its
    pass@1 (samples averaged first, arXiv 2411.00640 §3.1) the score is ½(mean_HE t + mean_MBPP t): a task's base and
    plus columns are one cluster, the benchmarks independent, so se² = ¼(se²_HE + se²_MBPP), each by summary over the
    task scores (Eq. 1); 95% normal interval (Eq. 3). n counts tasks."""
    cells = []
    for record in (humaneval, mbpp):
        base, plus = evalplus_passed(record)
        scores = [(statistics.fmean(b) + statistics.fmean(p)) / 2 for b, p in zip(base, plus, strict=True)]
        cells.append(summary(scores))
    mean = (cells[0]["mean"] + cells[1]["mean"]) / 2
    se = float(np.hypot(cells[0]["se"], cells[1]["se"]) / 2)
    half = float(norm.ppf(0.975)) * se
    return {"n": cells[0]["n"] + cells[1]["n"], "clusters": None, "mean": mean, "se": se,
            "interval": (mean - half, mean + half)}


def pass_columns(benchmark: str, record: dict) -> dict[str, list[list[bool]]]:
    """A codegen, reasoning or fim record's pass outcomes per score column, [problem][sample]: EvalPlus base and plus
    (columns humaneval and humaneval_plus, or mbpp and mbpp_plus); else "pass@1" of status "OK" (MultiPL-E), "passed"
    (lm-eval MBPP) or True (LiveCodeBench), or of result "passed" (human_eval_infilling); or SantaCoder's
    "exact_match"."""
    if "base_status" in record:
        return dict(zip((benchmark.removesuffix("_plus"), benchmark), evalplus_passed(record), strict=True))
    if "status" in record:
        return {"pass@1": [[s is True or s in ("OK", "passed") for s in ss] for ss in record["status"]]}
    if "result" in record:
        return {"pass@1": [[r == "passed"] for r in record["result"]]}
    return {"exact_match": [[bool(e)] for e in record["exact_match"]]}


def problem_clusters(record: dict) -> list[str] | None:
    """A fim record's clusters, its rows' HumanEval problems: human_eval_infilling ids "<Benchmark>/HumanEval/<i>/L<j>"
    as eval/code_eval.py infilling clusters them, SantaCoder ids "<problem>.json-L<line>"; None for one row each."""
    if "result" in record:
        return ["/".join(i.split("/")[1:3]) for i in record["ids"]]
    if "exact_match" in record:
        return [i.rsplit(".json-L", 1)[0] for i in record["ids"]]
    return None


# z_{α/2} + z_β at α 0.05 and 80% power (arXiv 2411.00640 Eq. 10)
DETECTABLE_Z = float(norm.ppf(0.975) + norm.ppf(0.8))


def compare(a: dict, b: dict) -> dict[str, dict]:
    """Paired comparison of evaluation records a and b (RUNS/<run>/eval-<step>.json) over the codegen (every mode and
    column), fim and reasoning sections both hold, mode by mode on identical problems (equal ids, else ValueError).
    Rows "<section>/<benchmark>/<mode>/<column>" (reasoning's mode "sampled"), each: n, means a and b of the
    per-problem pass rates (samples averaged first, arXiv 2411.00640 §3.1), diff = b − a with se by summary of the
    per-problem differences (Eq. 7; Eq. 8 clustered by HumanEval problem for fim rows) and its 95% interval (Eq. 3);
    corr, the Pearson correlation of the per-problem scores (Table 5; None when either is constant); p: exact McNemar
    (statsmodels) when every problem has one sample in both (greedy, 1@10, mbpp_3shot, fim; it treats a fim row's
    lines as independent), else the two-sided normal test of diff / se; p_holm, Holm-adjusted over all rows
    (statsmodels multipletests); mde, the difference detectable at α 0.05 with 80% power, (z_0.025 + z_0.2) · se
    (Eq. 10). The variance of the per-problem difference, estimated by its sample variance over n problems
    (n · se²), is Eq. 10's bracket ω² + σ²_A/K_A + σ²_B/K_B less twice the covariance of A's and B's sample means:
    the sampled modes seed sample k of problem i alike in every record (eval/code_eval.py), so A's and B's samples are
    positively correlated, and this paired se and mde are the ones for that design (common random numbers), smaller
    than with independent seeds; for fim rows the clustered se² is App. C's cluster-adjusted bracket over n."""
    rows = {}
    for section in ("codegen", "fim", "reasoning"):
        if not a.get(section) or not b.get(section):
            continue
        for benchmark in [k for k in a[section] if k in b[section]]:
            modes_a, modes_b = (s[section][benchmark] for s in (a, b))
            if section == "reasoning":
                modes_a, modes_b = {"sampled": modes_a}, {"sampled": modes_b}
            for mode in [m for m in modes_a if m in modes_b]:
                x, y = modes_a[mode], modes_b[mode]
                if x["ids"] != y["ids"]:
                    raise ValueError(f"{section}/{benchmark}/{mode}: the two records hold different problems")
                clusters = problem_clusters(x) if section == "fim" else None
                columns_b = pass_columns(benchmark, y)
                for column, passed_a in pass_columns(benchmark, x).items():
                    passed_b = columns_b[column]
                    pa, pb = (np.array([statistics.fmean(p) for p in passed]) for passed in (passed_a, passed_b))
                    diff = summary(pb - pa, clusters)
                    if all(len(p) == 1 for p in (*passed_a, *passed_b)):
                        both, only_a, only_b = (int(np.sum(u * v)) for u, v in ((pa, pb), (pa, 1 - pb), (1 - pa, pb)))
                        table = [[both, only_a], [only_b, pa.size - both - only_a - only_b]]
                        test, p = "mcnemar", float(mcnemar(table, exact=True).pvalue)
                    else:
                        test, se = "normal", diff["se"]
                        p = 2 * float(norm.sf(abs(diff["mean"]) / se)) if se else float(diff["mean"] == 0)
                    corr = None if np.ptp(pa) == 0 or np.ptp(pb) == 0 else float(np.corrcoef(pa, pb)[0, 1])
                    rows[f"{section}/{benchmark}/{mode}/{column}"] = {
                        "n": int(pa.size), "a": float(pa.mean()), "b": float(pb.mean()), "diff": diff["mean"],
                        "se": diff["se"], "interval": diff["interval"], "clusters": diff["clusters"], "corr": corr,
                        "test": test, "p": p, "mde": DETECTABLE_Z * diff["se"]}
    if rows:
        for row, adjusted in zip(rows.values(), multipletests([r["p"] for r in rows.values()], method="holm")[1],
                                 strict=True):
            row["p_holm"] = float(adjusted)
    return rows


# The GPU's dense BF16 peak: H100 SXM5 / GH200 (half the 1,979 TFLOP/s "with sparsity" of NVIDIA's H100 SXM specs)
PEAK_FLOPS = 989.4e12


def model_flops_per_token(*, matmul_params: int, hidden_size: int, windows: Sequence[int], context: int) -> int:
    """Nominal training FLOPs per token (arXiv 2204.02311 App. B), N the decoder layers' and lm_head's matmul weights:
    6 N + 12 hidden_size Σ_layers min(window + 1, context)."""
    return 6 * matmul_params + 12 * hidden_size * sum(min(window + 1, context) for window in windows)


def attended_keys(lengths: np.ndarray, window: int) -> int:
    """Σ over segments of Σ_p min(p, window + 1), p = 1..length: the keys attended causally within a segment."""
    lengths = np.asarray(lengths, dtype=np.int64)
    m = np.minimum(lengths, window + 1)
    return int((m * (m + 1) // 2 + (lengths - m) * (window + 1)).sum())


def executed_model_flops(lengths: np.ndarray, *, matmul_params: int, hidden_size: int, windows: Sequence[int]) -> int:
    """Training FLOPs a micro-batch of segment `lengths` executes: 6 N per position + 12 hidden_size per attended key
    per layer (QK^T and PV, 4 hidden_size per key forward, twice backward; FA4's score recomputation not counted)."""
    keys = sum(count * attended_keys(lengths, window) for window, count in Counter(windows).items())
    return 6 * matmul_params * int(np.asarray(lengths).sum()) + 12 * hidden_size * keys


def mfu(flops: float, seconds: float) -> float:
    """Model FLOPs utilization: the model FLOPs done per second over PEAK_FLOPS."""
    return flops / seconds / PEAK_FLOPS


def parabola_vertex(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """(curvature a, vertex -b / (2a)) of the least-squares parabola a x² + b x + c (numpy polyfit)."""
    a, b, _ = np.polyfit(x, y, 2)
    return float(a), float(-b / (2 * a))


def latency(starts: ArrayLike, firsts: ArrayLike, ends: ArrayLike, new_tokens: ArrayLike, *, duration: float) -> dict:
    """Decoding latency in seconds (vllm-project/vllm v0.32.0 vllm/benchmarks/serve.py calculate_metrics): ttft, tpot,
    e2el and output_throughput; tpot over requests of more than one token, None if none (vLLM reports 0)."""
    start, first, end = (np.asarray(t, dtype=np.float64) for t in (starts, firsts, ends))
    n = np.asarray(new_tokens)
    many = n > 1

    def describe(x: np.ndarray) -> dict:
        return {"mean": float(x.mean()), "median": float(np.median(x)), "p99": float(np.percentile(x, 99))}

    return {"ttft": describe(first - start),
            "tpot": describe((end - first)[many] / (n[many] - 1)) if many.any() else None,
            "e2el": describe(end - start), "output_throughput": float(n.sum() / duration),
            "requests": int(n.size)}
