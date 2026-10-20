"""
Analysis of the ablation designs (ablations/study.py) from each run's raw records, by the rules ablations.md fixes before
any run: its evaluation (BASE_EVAL/<tag>.json, scripts/base_eval's schema) and its per-step metrics (METRICS/<tag>.jsonl,
each step's latest attempt). Tables in results/<design>.md, the screen's decisions in results/screen.json (study.py
confirm's input).

Metrics. Primary: code, the 8K validation bits per byte of the code group at production's corpus weights
(loss_eval.val_bits "bpb"); solutions, the bits per byte of the HumanEval+, MBPP+ and HumanEval-X C++ reference solutions
given their prompts, pooled (code_eval.solution_bits). Secondary, reported only: text, math, the 32K repository bits per
byte overall and beyond the file-level context, each benchmark's solutions, and tok_per_s, the median over the
file-level stage's steps of the step's tokens per second.

- screen: OLS of each metric on the six factors coded -1 (production) / +1 (alternative) and the seed block
  (statsmodels); the design is resolution IV, so main effects are free of two-factor interactions, which are aliased in
  groups and left to the residual. Effect = 2 x coefficient, the alternative minus production. A factor is adopted iff
  its effect is negative with Holm-adjusted p < ALPHA (Holm across the six factors) on code or on solutions, and not
  positive with unadjusted p < ALPHA on the other.
- shape: OLS of each metric on the shape (treatment contrasts against the production shape) and the seed block; Holm
  across the three contrasts; a shape is better by the screen's rule.
- confirm: per adopted factor, the d16 alternative minus the shape design's production-shape run, per seed; confirmed
  iff every seed's difference has the screen effect's sign on each metric the factor was adopted for.

    python -m ablations.report screen|shape|confirm
"""
import json
import statistics
import sys

import pandas as pd
import statsmodels.formula.api as smf
from pydoe import fracfact, fracfact_aliasing
from statsmodels.stats.multitest import multipletests

from ablations.study import (
    FACTORS,
    GENERATORS,
    RATIO,
    RESULTS,
    SEEDS,
    confirm,
    model_of,
    production,
    screen,
    shape,
    tag_of,
)
from nanochat.config import REPO_STAGE
from nanochat.data.loader import mixture_weights
from nanochat.data.tokenizer import get_tokenizer
from nanochat.eval.loss_eval import bpb, mixture_bpb
from nanochat.paths import BASE_EVAL
from nanochat.train.metrics import train_steps
from nanochat.train.schedule import plan

ALPHA = 0.05  # the conventional significance level (Fisher, Statistical Methods for Research Workers, 1925, §20)
PRIMARY = ("code", "solutions")


def metrics(run, repo_weights: dict[str, float]) -> dict[str, float]:
    """A run's metrics from its evaluation and its per-step records."""
    tag = tag_of(run)
    result = json.loads((BASE_EVAL / f"{tag}.json").read_text(encoding="utf-8"))
    solution = result["solution"]
    pooled = {k: [sum(sum(s[k]) for s in solution.values())] for k in ("bits", "bytes")}
    repo = result["32k"]["corpora"]
    beyond = {c: {"bits": r["bits"][-1:], "bytes": r["bytes"][-1:], "rows": r["rows"]} for c, r in repo.items()}
    repo_start = plan(model=model_of(run), ratio=RATIO).repo_start
    rates = [r["tok_per_s"] for step, r in train_steps(tag).items() if step < repo_start]
    return {"code": result["8k"]["bpb"]["code"], "solutions": bpb(pooled), "text": result["8k"]["bpb"]["text"],
            "math": result["8k"]["bpb"]["math"], "repo_32k": result["32k"]["bpb"]["code"],
            "repo_beyond_8k": mixture_bpb(beyond, repo_weights),
            **{f"solutions_{b}": bpb(s) for b, s in solution.items()}, "tok_per_s": statistics.median(rates)}


def frame(runs, columns) -> pd.DataFrame:
    """One row per run: columns(run) and its metrics."""
    weights = mixture_weights(REPO_STAGE.mixture, get_tokenizer(), REPO_STAGE.seq_len)
    return pd.DataFrame([columns(run) | {"seed": run.seed} | metrics(run, weights) for run in runs])


def adopted_by_rule(effects: pd.DataFrame, p_holm: pd.DataFrame, p_raw: pd.DataFrame) -> dict[str, list[str]]:
    """{candidate: metrics it is better on} for candidates better on a primary metric and worse on none."""
    decisions = {}
    for candidate in effects.index:
        better = [m for m in PRIMARY if effects.at[candidate, m] < 0 and p_holm.at[candidate, m] < ALPHA]
        worse = [m for m in PRIMARY if effects.at[candidate, m] > 0 and p_raw.at[candidate, m] < ALPHA]
        if better and not worse:
            decisions[candidate] = better
    return decisions


def holm(p_raw: pd.DataFrame) -> pd.DataFrame:
    """Holm-adjusted p-values within each metric (column)."""
    return p_raw.apply(lambda column: pd.Series(multipletests(column, alpha=ALPHA, method="holm")[1],
                                                index=column.index))


def screen_report() -> str:
    df = frame(screen(), lambda run: {name: -1 if getattr(run, field) == low else 1
                                      for name, (field, low, _) in FACTORS.items()})
    effects, se, p_raw = {}, {}, {}
    for metric in df.columns.drop([*FACTORS, "seed"]):
        fit = smf.ols(f"{metric} ~ {' + '.join(FACTORS)} + C(seed)", data=df).fit()
        effects[metric], se[metric], p_raw[metric] = 2 * fit.params, 2 * fit.bse, fit.pvalues
    effects, se, p_raw = (pd.DataFrame(x).loc[list(FACTORS)] for x in (effects, se, p_raw))
    p_holm = holm(p_raw[list(PRIMARY)])
    decisions = adopted_by_rule(effects, p_holm, p_raw)
    (RESULTS / "screen.json").write_text(json.dumps({"adopted": list(decisions), "better_on": decisions,
                                                     "effects": effects[list(PRIMARY)].to_dict()}, allow_nan=False),
                                         encoding="utf-8")
    letters = ", ".join(f"{chr(ord('a') + i)} = {name}" for i, name in enumerate(FACTORS))
    aliases = "\n".join(f"- {chain}" for chain in fracfact_aliasing(fracfact(GENERATORS))[0])
    lines = [f"# Screen: d8, {len(df)} runs\n",
             f"Design: 2^(6-2), generators {GENERATORS} ({letters}); aliasing (pydoe fracfact_aliasing):\n\n{aliases}\n",
             "Effect: alternative minus production (bits per byte; tokens per second for tok_per_s).\n",
             "## Effects\n", effects.to_markdown(floatfmt=".5f"), "\n## Standard errors\n", se.to_markdown(floatfmt=".5f"),
             "\n## p (primary metrics: Holm across factors)\n", p_holm.to_markdown(floatfmt=".4f"),
             "\n## p (unadjusted)\n", p_raw.to_markdown(floatfmt=".4f"),
             f"\n## Adopted\n\n{json.dumps(decisions) if decisions else 'none'}\n",
             "\n## Runs\n", df.to_markdown(floatfmt=".5f")]
    return "\n".join(lines)


def shape_report() -> str:
    reference = production(depth=16, seed=0)
    label = f"{reference.n_layer}x{reference.n_embd}"
    df = frame(shape(), lambda run: {"shape": f"{run.n_layer}x{run.n_embd}"})
    effects, p_raw = {}, {}
    for metric in df.columns.drop(["shape", "seed"]):
        fit = smf.ols(f"{metric} ~ C(shape, Treatment('{label}')) + C(seed)", data=df).fit()
        contrasts = {k: k.split("[T.")[1].rstrip("]") for k in fit.params.index if k.startswith("C(shape")}
        effects[metric] = fit.params[list(contrasts)].rename(contrasts)
        p_raw[metric] = fit.pvalues[list(contrasts)].rename(contrasts)
    effects, p_raw = pd.DataFrame(effects), pd.DataFrame(p_raw)
    p_holm = holm(p_raw[list(PRIMARY)])
    decisions = adopted_by_rule(effects, p_holm, p_raw)
    means = df.groupby("shape").mean(numeric_only=True).drop(columns="seed")
    lines = [f"# Shape: depth vs width at {label}'s scaling parameters\n", "## Means over seeds\n",
             means.to_markdown(floatfmt=".5f"), f"\n## Contrasts against {label}\n", effects.to_markdown(floatfmt=".5f"),
             "\n## p (primary metrics: Holm across contrasts)\n", p_holm.to_markdown(floatfmt=".4f"),
             f"\n## Better than {label}\n\n{json.dumps(decisions) if decisions else 'none'}\n",
             "\n## Runs\n", df.to_markdown(floatfmt=".5f")]
    return "\n".join(lines)


def confirm_report() -> str:
    screen_result = json.loads((RESULTS / "screen.json").read_text())
    runs = confirm()  # each adopted factor's runs, one per seed
    diff = (frame(runs, lambda run: {}) - frame([production(depth=16, seed=run.seed) for run in runs],
                                                lambda run: {})).drop(columns="seed")
    lines = ["# Confirmation at d16\n"]
    for i, name in enumerate(screen_result["adopted"]):
        factor = diff.iloc[i * len(SEEDS):(i + 1) * len(SEEDS)]
        signs = {m: screen_result["effects"][m][name] for m in screen_result["better_on"][name]}
        confirmed = all((factor[m] * signs[m] > 0).all() for m in signs)
        lines += [f"## {name}: {'confirmed' if confirmed else 'not confirmed'}\n",
                  f"Screen effects: {json.dumps(signs)}\n", factor.to_markdown(floatfmt=".5f"), ""]
    return "\n".join(lines)


def main():
    design = sys.argv[1]
    report = {"screen": screen_report, "shape": shape_report, "confirm": confirm_report}[design]()
    (RESULTS / f"{design}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
