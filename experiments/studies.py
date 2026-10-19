"""
The appendix ablations: each study is a set of runs (identified by configuration and code, config.run_tag), a decision
rule fixed in advance, and a report in results/studies/. Metric: validation bits per byte of the code
mixture at 8K (VAL_METRIC); for the repository-stage schedule also of the repository mixture at 32K (LONG_VAL_METRIC).

Adopted without a study, on published evidence: linear decay to zero (Bergsma et al., arXiv 2502.15938: the gain over
WSD grows with tokens per parameter), one RoPE base 10^6 for both stages (Men et al., arXiv 2405.14591 lower bound;
Llama 3 trains at a large base from the start), SSSL windows (Gemma 3, arXiv 2503.19786 §5.2), learned QK-norm gains
(Qwen3, arXiv 2505.09388 §2), the x0 and residual scalars (karpathy/nanochat dev/LOG.md 2026-01-11), the peak lr of
DeepSeek LLM's law (a d24 lr-horizon fit at affordable horizons extrapolates 38x with an interval too wide to override
it).

- regime (small scale: d8, 100 tokens per scaling parameter): 2^3 full factorial over weight decay (tau law / 0.1),
  clipping (none / 1.0) and the repository stage's schedule (continue / rewarm), at two seed pairs; effects by OLS with
  the seed pair as a block, their SEs from the replicates. Rules (regime_decision), Holm across the three: a hand-set
  value (weight decay 0.1, clip 1.0) replaces the baseline's only if significantly better at 8K; rewarm is adopted only
  if significantly better at 32K and not significantly worse at 8K.
- scale_up (full scale: d24, file-level stage, two seed pairs): the regime decision ("selected") at SCALE_UP_TOKENS and
  each architecture candidate on it: value embeddings removed (~5% of the step, 302M table parameters), Block Attention
  Residuals added, the partial key offset added. The first pair's selected run also at twice the tokens: its slope per
  doubling. Rule (scale_up_decision): a candidate is adopted if on both pairs it beats the selected run at equal H100
  time (the selected run's metric moved along that slope to the candidate's wall-clock).
Runs train one at a time (experiments/run.sh): scale_up's throughput needs the GPU alone.

    python -m experiments.studies runs <study>
    python -m experiments.studies report <study>|final
"""
import argparse
import json
import math
import os
from dataclasses import dataclass, fields
from functools import cache
from typing import Literal, get_args, get_origin

import pandas as pd
import pydoe
import scipy.stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from nanochat.config import run_tag
from nanochat.data.tokenizer import get_tokenizer
from nanochat.paths import base_dir
from nanochat.train.stages import max_ratio
from scripts.base_train import Variant

RESULTS = "results/studies"
VAL_METRIC = ("8k", "code")
LONG_VAL_METRIC = ("32k", "all")
ALPHA = 0.05
# Small scale: d8 at 100 tokens per scaling parameter (5x Chinchilla's 20, arXiv 2203.15556), the studies' budget
SMALL_DEPTH, SMALL_RATIO = 8, 100
# Each Literal field's (baseline, alternative): its first and second choice
LEVELS = {f.name: get_args(f.type) for f in fields(Variant) if get_origin(f.type) is Literal}
FACTORS = ("weight_decay", "clip", "repo_schedule")
SEED_PAIRS = ((0, 0), (1, 1))
FULL_DEPTH = 24
# 0.5B tokens (2.6 per scaling parameter at d24)
SCALE_UP_TOKENS = 5e8
# Value embeddings removed; Block Attention Residuals (arXiv 2603.15031 Table 2: 194M activated params, val loss
# 1.931 -> 1.909), which replaces the resid and x0 scalars; p-RoPE with the partial key offset (modded-nanogpt #49)
CANDIDATES = {"no_value_embeds": {"value_embeds": "none"},
              "attn_res": {"attn_res": "block", "resid": "one", "x0": "none"},
              "key_offset": {"key_offset": "partial"}}


@dataclass(frozen=True, kw_only=True)
class Run:
    study: str
    name: str
    depth: int
    ratio: float
    seed: int
    data_seed: int
    config: dict  # the Variant fields that differ from the baseline

    @property
    def tag(self) -> str:
        return run_tag(depth=self.depth, ratio=self.ratio, seed=self.seed, data_seed=self.data_seed,
                       changes=self.config)


def changes(values: dict) -> dict:
    """The fields that differ from the baseline Variant."""
    baseline = Variant()
    return {k: v for k, v in values.items() if getattr(baseline, k) != v}


def val_bpb(tag: str, *, which: tuple[str, str]) -> float:
    with open(os.path.join(base_dir(), "evals", f"{tag}.json"), encoding="utf-8") as f:
        return json.load(f)["eval"][which[0]]["bpb"][which[1]]


def throughput(tag: str) -> float:
    """Median training tok/s (train/metrics.py), each process's first (compiling) step excluded."""
    log = pd.read_json(os.path.join(base_dir(), "metrics", f"{tag}.jsonl"), lines=True)
    steps = log[log.kind == "train"]
    return float(steps.tok_per_s[~steps.step.isin(log[log.kind == "run"].step)].median())


@cache
def production_ratio() -> float:
    return max_ratio(get_tokenizer(), depth=FULL_DEPTH)


# Runs


def design_points() -> list[dict[str, int]]:
    """{factor: +1 baseline / -1 alternative} per point of the full factorial."""
    return [dict(zip(FACTORS, map(int, row), strict=True)) for row in pydoe.ff2n(len(FACTORS))]


def regime() -> list[Run]:
    return [Run(study="regime", name="".join("1" if p[f] > 0 else "0" for f in FACTORS) + f"-s{s}-d{d}",
                depth=SMALL_DEPTH, ratio=SMALL_RATIO, seed=s, data_seed=d,
                config=changes({f: LEVELS[f][p[f] < 0] for f in FACTORS}))
            for s, d in SEED_PAIRS for p in design_points()]


def full_scale_run(*, name: str, config: dict, seed: int, data_seed: int, tokens: float) -> Run:
    """A d24 run at the production run's batch and lr, cut to `tokens`."""
    return Run(study="scale_up", name=name, depth=FULL_DEPTH, ratio=production_ratio(), seed=seed, data_seed=data_seed,
               config=changes(config | {"tokens": tokens}))


def scale_up() -> list[Run]:
    selected = regime_decision()
    runs = []
    for s, d in SEED_PAIRS:
        configs = {f"selected-s{s}-d{d}": (selected, SCALE_UP_TOKENS)}
        configs |= {f"selected-s{s}-d{d}-2x": (selected, 2 * SCALE_UP_TOKENS)} if (s, d) == SEED_PAIRS[0] else {}
        configs |= {f"{c}-s{s}-d{d}": (selected | extra, SCALE_UP_TOKENS) for c, extra in CANDIDATES.items()}
        runs += [full_scale_run(name=n, config=c, seed=s, data_seed=d, tokens=t) for n, (c, t) in configs.items()]
    return runs


STUDIES = {"regime": regime, "scale_up": scale_up}


# Decisions


def regime_results() -> pd.DataFrame:
    """The factorial's runs on +-1 coding (+1 baseline) with their seed pair and both metrics."""
    return pd.DataFrame([{"run": r.name, "pair": f"s{r.seed}-d{r.data_seed}",
                          **{f: 1 if f not in r.config else -1 for f in FACTORS},
                          "bpb_8k": val_bpb(r.tag, which=VAL_METRIC), "bpb_32k": val_bpb(r.tag, which=LONG_VAL_METRIC)}
                         for r in regime()]).set_index("run")


def regime_effects() -> pd.DataFrame:
    """Per factor and metric: effect = alternative - baseline (OLS on the full factorial with a seed-pair block, -2 x
    the coefficient), its SE and one-sided p from the replicates' residual; the rule's p-values Holm-adjusted together,
    and `adopt`."""
    results = regime_results()
    rows = {}
    for metric in ("bpb_8k", "bpb_32k"):
        fit = smf.ols(f"{metric} ~ {' * '.join(FACTORS)} + C(pair)", data=results).fit()
        for f in FACTORS:
            effect, se = -2 * fit.params[f], 2 * fit.bse[f]
            rows[(f, metric)] = {"effect": effect, "se": se, "df": fit.df_resid,
                                 "p_better": scipy.stats.t.cdf(effect / se, fit.df_resid),
                                 "p_worse": scipy.stats.t.sf(effect / se, fit.df_resid)}
    effects = pd.DataFrame(rows).T.rename_axis(["factor", "metric"])
    tests = [("weight_decay", "bpb_8k"), ("clip", "bpb_8k"), ("repo_schedule", "bpb_32k")]
    holm = multipletests([effects.loc[t, "p_better"] for t in tests], alpha=ALPHA, method="holm")[1]
    effects["p_better_holm"] = pd.Series(dict(zip(tests, holm, strict=True)))
    adopt = {f: p < ALPHA for (f, _), p in zip(tests, holm, strict=True)}
    adopt["repo_schedule"] &= effects.loc[("repo_schedule", "bpb_8k"), "p_worse"] >= ALPHA
    effects["adopt"] = [adopt[f] if (f, m) in tests else None for f, m in effects.index]
    return effects


def regime_decision() -> dict:
    effects = regime_effects()
    return changes({f: LEVELS[f][1] for (f, _), a in effects.adopt.dropna().items() if a})


def scale_up_decision() -> tuple[pd.DataFrame, list[str]]:
    """Per seed pair the metrics, each candidate's time ratio r (selected / candidate throughput) and the selected
    run's metric at r x SCALE_UP_TOKENS (the first pair's slope per doubling); the candidates adopted."""
    runs = {r.name: r for r in scale_up()}
    s0, d0 = SEED_PAIRS[0]
    slope = (val_bpb(runs[f"selected-s{s0}-d{d0}-2x"].tag, which=VAL_METRIC)
             - val_bpb(runs[f"selected-s{s0}-d{d0}"].tag, which=VAL_METRIC))
    rows = []
    for s, d in SEED_PAIRS:
        pair = f"s{s}-d{d}"
        selected = runs[f"selected-{pair}"]
        row = {"seeds": pair, "selected": val_bpb(selected.tag, which=VAL_METRIC), "slope per doubling": slope}
        for c in CANDIDATES:
            run = runs[f"{c}-{pair}"]
            ratio = throughput(selected.tag) / throughput(run.tag)
            row |= {c: val_bpb(run.tag, which=VAL_METRIC), f"{c}: time ratio": ratio,
                    f"{c}: selected at equal time": row["selected"] + slope * math.log2(ratio)}
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("seeds")
    return frame, [c for c in CANDIDATES if (frame[c] < frame[f"{c}: selected at equal time"]).all()]


def final_config() -> dict:
    """The production run's changes from the baseline: the regime decision plus the adopted candidates."""
    config = regime_decision()
    for c in scale_up_decision()[1]:
        config |= CANDIDATES[c]
    return changes(config)


# Reports: results/studies/<study>.md and <study>.csv


def markdown(frame: pd.DataFrame, *, title: str) -> str:
    return f"### {title}\n\n{frame.to_markdown(floatfmt='.5g')}\n"


def report_regime() -> tuple[list[str], pd.DataFrame]:
    effects = regime_effects()
    return [markdown(regime_results(), title="Regime: runs"),
            markdown(effects, title="Regime: effects, alternative - baseline, and decisions"),
            f"Adopted: `{json.dumps(regime_decision())}`\n"], effects


def report_scale_up() -> tuple[list[str], pd.DataFrame]:
    frame, added = scale_up_decision()
    return [markdown(frame, title=f"Scale-up to d24 on {json.dumps(regime_decision())} (bits per byte at "
                                  f"{SCALE_UP_TOKENS:g} tokens; the selected run at equal H100 time along its "
                                  f"log2-token slope)"),
            f"Candidates adopted: {added}\n"], frame


def report_final() -> tuple[list[str], pd.DataFrame]:
    config = final_config()
    frame = pd.DataFrame({"value": {k: str(v) for k, v in config.items()}}).rename_axis("Variant field")
    return [markdown(frame, title="The production run's changes from the baseline"), f"`{json.dumps(config)}`\n"], frame


REPORTS = {"regime": report_regime, "scale_up": report_scale_up, "final": report_final}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("runs", "report"))
    parser.add_argument("study", choices=REPORTS)
    args = parser.parse_args()
    if args.command == "runs":
        for run in STUDIES[args.study]():
            print("run", run.name, run.tag)
        return
    sections, numbers = REPORTS[args.study]()
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"{args.study}.md"), "w", encoding="utf-8") as f:
        f.write(f"## {args.study}\n\n" + "\n".join(sections))
    numbers.to_csv(os.path.join(RESULTS, f"{args.study}.csv"))


if __name__ == "__main__":
    main()
