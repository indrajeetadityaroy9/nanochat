"""
Analysis of the ablation designs (ablations/study.py) from each run's raw records, by the rules ablations.md fixes before
any run: its evaluation (BASE_EVAL/<tag>.json, scripts/base_eval's schema) and its per-step metrics (METRICS/<tag>.jsonl,
each step's latest attempt). Tables in results/<design>.md; the screen's and the lr sweep's decisions in
results/screen.json and results/lr.json (study.py confirm's input).

Metrics. Primary: code, the 8K validation bits per byte of the code group at production's corpus weights
(loss_eval.val_bits "bpb"); solutions, the bits per byte of the HumanEval+, MBPP+ and HumanEval-X C++ reference solutions
given their prompts, pooled (code_eval.solution_bits); repo_32k, the 32K repository bits per byte of the code group.
Secondary, reported only: text, math, the repository bits per byte beyond the file-level context, each benchmark's
solutions, and tok_per_s, the median over the file-level stage's steps of the step's tokens per second.

Rule. At a depth, a candidate is better iff its effect is negative with Holm-adjusted p < ALPHA (Holm across the
design's candidates) on a primary metric, and not positive with unadjusted p < ALPHA on any. A candidate is adopted iff
it is better at the deciding depth (the deepest) and its effect on each of those metrics is negative at every other
depth too.

- screen: per depth, OLS of each metric on the eight factors coded -1 (production) / +1 (alternative) (statsmodels);
  every run has its own seed (no blocks: in the paired comparison a seed moved the two configurations in opposite
  directions, ablations.md Power). The design is resolution IV, so main effects are free of two-factor interactions,
  which are aliased in groups and left to the residual. Effect = 2 x coefficient, the alternative minus production.
- lr: per depth, OLS of each metric on the multiplier (treatment contrasts against 1); of the adopted multipliers, the
  one with the lowest mean code at the deciding depth.
- shape: OLS of each metric on the shape (treatment contrast against the production shape); the deeper shape is better
  by the rule at its one depth.
- confirm: per adopted change, its d16 runs against the shape design's production-shape runs: one-sided pooled t-test
  (statsmodels ttest_ind) on each metric the change was adopted for; confirmed iff every p < ALPHA.

    python -m ablations.report screen|lr|shape|confirm
"""
import json
import statistics
import sys

import pandas as pd
import statsmodels.formula.api as smf
from pydoe import fracfact, fracfact_aliasing
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import ttest_ind

from ablations.study import (
    FACTORS,
    GENERATORS,
    LR_SCALES,
    SEEDS,
    SHAPE_DEPTH,
    adopted,
    confirm,
    lr,
    plan_of,
    production,
    screen,
    shape,
    tag_of,
)
from nanochat.config import FILE_STAGE, REPO_STAGE
from nanochat.data.loader import mixture_weights
from nanochat.data.tokenizer import get_tokenizer
from nanochat.eval.loss_eval import bpb, mixture_bpb
from nanochat.model.gpt import gpt_config
from nanochat.paths import BASE_EVAL, RESULTS
from nanochat.train.metrics import train_steps

ALPHA = 0.05  # the conventional significance level (Fisher, Statistical Methods for Research Workers, 1925, §20)
PRIMARY = ("code", "solutions", "repo_32k")


def metrics(run, repo_weights: dict[str, float]) -> dict[str, float]:
    """A run's metrics from its evaluation and its per-step records."""
    tag = tag_of(run)
    result = json.loads((BASE_EVAL / f"{tag}.json").read_text(encoding="utf-8"))
    solution = result["solution"]
    pooled = {k: [sum(sum(s[k]) for s in solution.values())] for k in ("bits", "bytes")}
    repo = result["32k"]["corpora"]
    beyond = {c: {"bits": r["bits"][-1:], "bytes": r["bytes"][-1:], "rows": r["rows"]} for c, r in repo.items()}
    repo_start = plan_of(run).repo_start
    rates = [r["tok_per_s"] for step, r in train_steps(tag).items() if step < repo_start]
    return {"code": result["8k"]["bpb"]["code"], "solutions": bpb(pooled), "repo_32k": result["32k"]["bpb"]["code"],
            "text": result["8k"]["bpb"]["text"], "math": result["8k"]["bpb"]["math"],
            "repo_beyond_8k": mixture_bpb(beyond, repo_weights),
            **{f"solutions_{b}": bpb(s) for b, s in solution.items()}, "tok_per_s": statistics.median(rates)}


def frame(runs, columns) -> pd.DataFrame:
    """One row per run: columns(run), its depth and seed, and its metrics."""
    weights = mixture_weights(REPO_STAGE.mixture, get_tokenizer(), REPO_STAGE.seq_len)
    return pd.DataFrame([columns(run) | {"depth": run.n_layer, "seed": run.seed} | metrics(run, weights)
                         for run in runs])


def ols(df: pd.DataFrame, design: list[str], rhs: str, terms: dict[str, str], scale: float):
    """Per metric (every column but the design's, depth and seed), OLS on `rhs`: for each candidate, `scale` x the
    coefficient terms[candidate], its standard error and p. Frames indexed by candidate, one column per metric."""
    effects, se, p = {}, {}, {}
    for metric in df.columns.drop([*design, "depth", "seed"]):
        fit = smf.ols(f"{metric} ~ {rhs}", data=df).fit()
        coefficients = list(terms.values())
        effects[metric] = scale * fit.params[coefficients].set_axis(list(terms))
        se[metric] = scale * fit.bse[coefficients].set_axis(list(terms))
        p[metric] = fit.pvalues[coefficients].set_axis(list(terms))
    return pd.DataFrame(effects), pd.DataFrame(se), pd.DataFrame(p)


def holm(p_raw: pd.DataFrame) -> pd.DataFrame:
    """Holm-adjusted p-values within each metric (column)."""
    return p_raw.apply(lambda column: pd.Series(multipletests(column, alpha=ALPHA, method="holm")[1],
                                                index=column.index))


def better(effects: pd.DataFrame, p_raw: pd.DataFrame) -> dict[str, list[str]]:
    """{candidate: primary metrics it is better on} by the rule at one depth."""
    p_holm = holm(p_raw[list(PRIMARY)])
    decisions = {}
    for candidate in effects.index:
        on = [m for m in PRIMARY if effects.at[candidate, m] < 0 and p_holm.at[candidate, m] < ALPHA]
        worse = [m for m in PRIMARY if effects.at[candidate, m] > 0 and p_raw.at[candidate, m] < ALPHA]
        if on and not worse:
            decisions[candidate] = on
    return decisions


def replicated(fits: dict[int, tuple]) -> dict[str, list[str]]:
    """{candidate: metrics} better at the deepest depth with negative effects on those metrics at every other depth."""
    deciding = max(fits)
    return {c: on for c, on in better(fits[deciding][0], fits[deciding][2]).items()
            if all(fits[d][0].at[c, m] < 0 for d in fits for m in on)}


def tables(fits: dict[int, tuple]) -> list[str]:
    lines = []
    for depth, (effects, se, p_raw) in sorted(fits.items()):
        lines += [f"## d{depth}\n", "### Effects\n", effects.to_markdown(), "\n### Standard errors\n", se.to_markdown(),
                  "\n### p (primary metrics: Holm across candidates)\n", holm(p_raw[list(PRIMARY)]).to_markdown(),
                  "\n### p (unadjusted)\n", p_raw.to_markdown(), ""]
    return lines


def write_decisions(design: str, decisions: dict[str, list[str]], changes: dict[str, dict], fits) -> None:
    (RESULTS / f"{design}.json").write_text(json.dumps(
        {"adopted": {c: changes[c] for c in decisions}, "better_on": decisions,
         "effects": {depth: fit[0][list(PRIMARY)].to_dict() for depth, fit in fits.items()}}, allow_nan=False),
        encoding="utf-8")


def screen_report() -> str:
    df = frame(screen(), lambda run: {name: 1 if getattr(run, field) == level else -1
                                      for name, (field, level) in FACTORS.items()})
    fits = {depth: ols(group, list(FACTORS), " + ".join(FACTORS), {name: name for name in FACTORS}, 2)
            for depth, group in df.groupby("depth")}
    decisions = replicated(fits)
    write_decisions("screen", decisions, {name: dict([FACTORS[name]]) for name in FACTORS}, fits)
    letters = ", ".join(f"{chr(ord('a') + i)} = {name}" for i, name in enumerate(FACTORS))
    aliases = "\n".join(f"- {chain}" for chain in fracfact_aliasing(fracfact(GENERATORS))[0])
    return "\n".join([f"# Screen: {len(df)} runs\n",
                      f"Design: generators {GENERATORS} ({letters}); aliasing (pydoe fracfact_aliasing):\n\n{aliases}\n",
                      "Effect: alternative minus production (bits per byte; tokens per second for tok_per_s).\n",
                      *tables(fits), f"## Adopted\n\n{json.dumps(decisions) if decisions else 'none'}\n",
                      "## Runs\n", df.to_markdown()])


def lr_report() -> str:
    scales = {f"x{s:.3g}": s for s in LR_SCALES}  # label -> multiplier; production's is "x1"
    df = frame(lr(), lambda run: {"lr": f"x{run.lr_scale:.3g}"})
    contrast = "C(lr, Treatment('x1'))"
    terms = {label: f"{contrast}[T.{label}]" for label, s in scales.items() if s != 1}
    fits = {depth: ols(group, ["lr"], contrast, terms, 1) for depth, group in df.groupby("depth")}
    candidates = replicated(fits)
    # at most one multiplier: the adopted one with the lowest mean code at the deciding depth
    code = df[df.depth == max(fits)].groupby("lr").code.mean()
    decisions = {c: candidates[c] for c in sorted(candidates, key=code.get)[:1]}
    write_decisions("lr", decisions, {label: {"lr_scale": s} for label, s in scales.items()}, fits)
    means = df.groupby(["depth", "lr"]).mean(numeric_only=True).drop(columns="seed")
    return "\n".join(["# Peak lr: multiples of the law's\n", "## Means over seeds\n", means.to_markdown(), "",
                      "Effect: the multiplier minus 1 (bits per byte; tokens per second for tok_per_s).\n",
                      *tables(fits), f"## Adopted\n\n{json.dumps(decisions) if decisions else 'none'}\n",
                      "## Runs\n", df.to_markdown()])


def shape_report() -> str:
    label = f"{SHAPE_DEPTH}x{gpt_config(depth=SHAPE_DEPTH, stage=FILE_STAGE).n_embd}"
    df = frame(shape(), lambda run: {"shape": f"{run.n_layer}x{run.n_embd}"})
    other = next(s for s in df["shape"] if s != label)
    effects, se, p_raw = ols(df, ["shape"], f"C(shape, Treatment('{label}'))",
                             {other: f"C(shape, Treatment('{label}'))[T.{other}]"}, 1)
    decisions = better(effects, p_raw)
    means = df.groupby("shape").mean(numeric_only=True).drop(columns=["depth", "seed"])
    return "\n".join([f"# Shape: depth vs width at {label}'s scaling parameters\n", "## Means over seeds\n",
                      means.to_markdown(), f"\n## Contrast against {label}\n", effects.to_markdown(),
                      "\n## Standard errors\n", se.to_markdown(), "\n## p\n", p_raw.to_markdown(),
                      f"\n## Better than {label}\n\n{json.dumps(decisions) if decisions else 'none'}\n",
                      "## Runs\n", df.to_markdown()])


def confirm_report() -> str:
    better_on = {c: on for design in ("screen", "lr")
                 for c, on in json.loads((RESULTS / f"{design}.json").read_text())["better_on"].items()}
    baseline = frame([production(depth=SHAPE_DEPTH, seed=seed) for seed in SEEDS], lambda run: {})
    runs = confirm()
    lines = [f"# Confirmation at d{SHAPE_DEPTH}\n"]
    for name, fields in adopted().items():
        change = frame([r for r in runs if all(getattr(r, f) == v for f, v in fields.items())], lambda run: {})
        # adopted on a metric only for a negative (better) effect: the change's mean is tested smaller
        p = {m: ttest_ind(change[m], baseline[m], alternative="smaller")[1] for m in better_on[name]}
        lines += [f"## {name}: {'confirmed' if all(v < ALPHA for v in p.values()) else 'not confirmed'}\n",
                  f"One-sided p at d{SHAPE_DEPTH}: {json.dumps(p)}\n", "Mean difference (change minus production):\n",
                  (change.mean() - baseline.mean()).drop(["depth", "seed"]).to_frame("difference").to_markdown(),
                  "\nRuns:\n", change.to_markdown(), ""]
    lines += [f"## Production d{SHAPE_DEPTH}\n", baseline.to_markdown()]
    return "\n".join(lines)


def main():
    design = sys.argv[1]
    report = {"screen": screen_report, "lr": lr_report, "shape": shape_report, "confirm": confirm_report}[design]()
    (RESULTS / f"{design}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
