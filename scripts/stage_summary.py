"""
Paired summary of S1 (runs/s1.sh): the framework F (--objective spec) against the next-token baseline B0 (--objective
ntp), one pair of runs per seed, from their evaluations (<base_dir>/base_eval/<run>.json, scripts/base_eval.py; the
tail-averaged weights, the estimator S1 compares). Each metric's F and B0 values and difference F - B0 per seed, then
the mean, the sample standard deviation of the differences, its standard error and the paired Student-t test of the
differences (scipy.stats.ttest_rel: its 95% confidence interval of the mean difference and two-sided p-value, n - 1
degrees of freedom; the seed effect is shared by the problems of a run, so the pairs are the samples of a difference).
Pre-registered primary metric: the mean val bpb over the code corpora (its p unadjusted). Secondary rows, per language
(Python, C++, then text and math): per-corpus val bpb, reference-solution bpb, greedy pass@1, fill-in-the-middle pass@1
(F only); their p-values Holm-adjusted together (Holm 1979, step-down family-wise error control).
Written as Markdown to --out and printed.

    python -m scripts.stage_summary --pairs d10-r8-spec-s42=d10-r8-ntp-s42,d10-r8-spec-s43=d10-r8-ntp-s43 --out runs/logs/s1/summary.md
"""
import argparse
import json
import os
import statistics

from scipy import stats

from nanochat.data.sources import SOURCES
from nanochat.paths import base_dir

CI_LEVEL = 0.95  # the summary's two-sided interval
ESTIMATOR = "tail"  # base_eval's estimator S1 compares
GENERATED = {"Python": {"humaneval_plus": ("base pass@1", "plus pass@1"), "mbpp_plus": ("base pass@1", "plus pass@1")},
             "C++": {"humaneval_cpp": ("pass@1",), "mbpp_cpp": ("pass@1",)}}
SOLUTIONS = {"Python": ("humaneval_plus", "mbpp_plus"), "C++": ("humaneval_x_cpp",)}
# codeforces_cots holds both languages; grouped by its token majority: C++ 61.3%, Python 38.6% of its train tokens
# (AuditData D5: C++ R1 59.8% + C++ submissions 1.5%)
MAJORITY = {"codeforces_cots": "C++"}


def language(corpus):
    """The mixture group of a corpus: its code language (a two-language source's token majority), else None."""
    lang = MAJORITY.get(corpus) or SOURCES[corpus].language
    return lang if lang in GENERATED else None


def metric_rows(evals):
    """(primary (label, getter, format), {group: [(label, getter, format)]})."""
    corpora = sorted({c for e in evals for c in e["bpb"][ESTIMATOR]}, key=list(SOURCES).index)
    code = [c for c in corpora if language(c) is not None]
    fim = sorted({b for e in evals for b in e.get("fim", {})})
    pct = lambda x: None if x is None else 100 * x
    bpb = lambda c: (f"{c} val bpb", lambda e: e["bpb"][ESTIMATOR][c], "{:.4f}")
    primary = ("mean code-corpus val bpb", lambda e: statistics.fmean(e["bpb"][ESTIMATOR][c] for c in code), "{:.4f}")
    groups = {}
    for lang in GENERATED:
        rows = [bpb(c) for c in corpora if language(c) == lang]
        rows += [(f"{b} solution bpb", lambda e, b=b: e["solution_bpb"][ESTIMATOR][b], "{:.4f}") for b in SOLUTIONS[lang]]
        rows += [(f"{b} {k} %", lambda e, b=b, k=k: pct(e["codegen"][b][k]), "{:.2f}")
                 for b, keys in GENERATED[lang].items() for k in keys]
        if lang == "Python":  # HumanEval infilling: spec runs only, a B0 cell stays empty
            rows += [(f"{b} FIM pass@1 %", lambda e, b=b: pct(e.get("fim", {}).get(b, {}).get("pass@1")), "{:.2f}") for b in fim]
        groups[lang] = rows
    groups["text and math"] = [bpb(c) for c in corpora if language(c) is None]
    return primary, groups


def holm(pvalues):
    """Holm's step-down adjusted p-values, in the input order."""
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def fmt(value, form, signed=False):
    return "–" if value is None else ("+" if signed and value >= 0 else "") + form.format(value)


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows])


def paired(row, pairs, evals):
    """(cells, p or None) of one metric row."""
    metric, get, form = row
    cells, values = [metric], []
    for f, b in pairs:
        vf, vb = get(evals[f]), get(evals[b])
        d = None if vf is None or vb is None else vf - vb
        cells += [fmt(vf, form), fmt(vb, form), fmt(d, form, signed=True)]
        values.append((vf, vb, d))
    if any(d is None for _, _, d in values):
        return cells + ["–"] * 5, None
    vf, vb, diffs = zip(*values, strict=True)
    test = stats.ttest_rel(vf, vb)
    ci = test.confidence_interval(CI_LEVEL)
    return cells + [fmt(statistics.fmean(diffs), form, signed=True), fmt(statistics.stdev(diffs), form),
                    fmt(stats.sem(diffs), form), f"[{fmt(ci.low, form, signed=True)}, {fmt(ci.high, form, signed=True)}]",
                    f"{test.pvalue:.3f}"], test.pvalue


def main():
    parser = argparse.ArgumentParser(description="S1 paired summary: F - B0 per seed")
    parser.add_argument("--pairs", type=str, required=True, help="comma-separated F_RUN=B0_RUN, one per seed")
    parser.add_argument("--out", type=str, required=True, help="Markdown output path")
    args = parser.parse_args()
    pairs = [tuple(item.split("=")) for item in args.pairs.split(",")]
    evals = {}
    for run in {run for pair in pairs for run in pair}:
        with open(os.path.join(base_dir(), "base_eval", f"{run}.json"), encoding="utf-8") as f:
            evals[run] = json.load(f)
    n = len(pairs)
    header = (["metric"] + [h for i, (f, b) in enumerate(pairs, 1) for h in (f"F {f}", f"B0 {b}", f"F - B0 pair {i}")]
              + ["mean F - B0", "sd of differences", "se", f"{CI_LEVEL:.0%} CI", "p"])
    primary, groups = metric_rows(list(evals.values()))
    secondary = {group: [paired(row, pairs, evals) for row in rows] for group, rows in groups.items()}
    tested = [p for lines in secondary.values() for _, p in lines if p is not None]
    adjusted = iter(holm(tested))
    out = [f"# S1: F vs B0 ({n} seeds, {ESTIMATOR} weights)", "", f"se = sd / sqrt(n); CI and two-sided p of the paired "
           f"t-test (scipy.stats.ttest_rel, df = {n - 1}): mean ± t x se, t = {stats.t.ppf((1 + CI_LEVEL) / 2, n - 1):.4f}.",
           "", "## Primary (pre-registered)", "", table(header, [paired(primary, pairs, evals)[0]]), "",
           f"Secondary rows: Holm p over their {len(tested)} tests.", ""]
    for group, lines in secondary.items():
        rows = [cells + ["–" if p is None else f"{next(adjusted):.3f}"] for cells, p in lines]
        out += [f"## {group}", "", table(header + ["Holm p"], rows), ""]
    text = "\n".join(out)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
