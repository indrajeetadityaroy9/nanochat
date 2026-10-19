"""
The paper's appendix studies (docs/h100-pretraining.md, "Appendix studies"): each a set of arms (training runs, one
experiments/arm.py run each, identified by their configuration so a run shared by studies trains once), a
pre-registered decision rule, and an analysis into Markdown and tidy tables under results/studies/ (the figures' data).
The metric of every study: METRIC, the code group's mixture bits per byte of the 8K val rows (eval/loss_eval.py).

- noise (d8, 100 tokens per scaling parameter): the canonical model at 4 init seeds x 2 data seeds: the run-to-run SD
  of the metric (the yardstick of every d8 effect) and its two-way ANOVA variance components (method of moments).
- screen (d8, 100 TPP): a 2^(7-3) resolution-IV fractional factorial (pydoe.fracfact DESIGN; Box & Hunter 1961) over
  four parameterized components, the attention windows and two regime choices (FACTORS), every arm at init and data
  seed 0 (common random numbers: model/gpt.py init_weights seeds each tensor by its name; the all-current point is the
  noise study's (0, 0) arm). Main effects by OLS on the +-1 design, each tested against the noise SD (t, the noise
  arms' df, one-sided in its rule's direction, Holm across the factors) and against Lenth's margin of error over the
  design's 15 contrasts (Technometrics 31(4), 1989). The noise SD is between seeds; a contrast at shared seeds carries
  the trajectory divergence of a changed component, bounded by it (Summers & Dinneen, arXiv 2103.04514). Rule
  (verdict): a component that costs parameters stays only if it significantly helps; the windows, which save compute,
  go only if they significantly hurt; a regime alternative is adopted only if significantly better.
- long_context (d8, 100 TPP): 2 x 2 over the repo-level stage's lr schedule (continue / re-warmed) and the RoPE base
  (switched at the stage / the repo-level stage's from the start): bits per byte at 32K per context bucket and at 8K.
- confirm (d24, CONFIRM_TOKENS, the file-level stage): the screen's architecture verdict against the canonical model
  at two seed pairs, on the screen's regime verdict. Rule (final): the verdict stands unless it loses on both.
- lr_horizon (d24, the file-level stage): peak lr LR_SCALES x the law's production lr (schedule.plan at the production
  ratio) over HORIZONS, on the final variant: the lr minimizing a quadratic in ln lr per horizon, and the WLS line of
  ln eta* on ln D extrapolated to the production run's file-level tokens. With the wsd schedule and fixed weight decay
  each lr's horizons share one constant-lr phase (schedule.lr_multiplier), so a horizon continues the shorter one's
  kept checkpoint at its cooldown start (Hagele et al., arXiv 2405.18392 §3.2): the same run.
- optimizer (d24, OPTIMIZER_TOKENS): AdamW on every parameter at LR_SCALES against lr_horizon's Muon arms there,
  each at its own fitted optimum.
- batch (d24): the gradient noise scale (experiments/noise_scale.py) at lr_horizon's kept checkpoints of the law's lr,
  its log-log trend in the tokens trained, extrapolated to the production run, against its batch.

    python -m experiments.studies arms <study>        # "arm NAME TAG" per line (experiments/run.sh)
    python -m experiments.studies analyze <study>     # results/studies/<study>.md and .csv
    python -m experiments.studies collect             # results/studies/{arms.csv, eval.csv, curves.parquet}
"""
import argparse
import dataclasses
import json
import math
import os
from dataclasses import dataclass, field
from functools import cache

import numpy as np
import pandas as pd
import pydoe
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from nanochat.config import REPO_STAGE, run_tag
from nanochat.data.loader import mixture_weights
from nanochat.data.tokenizer import get_tokenizer
from nanochat.eval.loss_eval import CONTEXTS, mixture_bpb
from nanochat.paths import base_dir
from nanochat.train.optim import GRAD_CLIP
from nanochat.train.schedule import cooldown_start, plan
from nanochat.train.stages import max_ratio
from scripts.base_train import Variant

RESULTS = "results/studies"
METRIC = ("8k", "code")
ALPHA = 0.05
# The d8 proxy: the smallest model of the family at 100 tokens per scaling parameter (5x Chinchilla's 20, arXiv
# 2203.15556; the production ratio is ~490)
D8, RATIO8 = 8, 100
NOISE_SEEDS, NOISE_DATA_SEEDS = (0, 1, 2, 3), (0, 1)
# {factor: (current, alternative)}: each choice under test against its simplest alternative (Variant fields), in the
# order of DESIGN's columns: schedule = VE x0 resid, weight_decay = VE x0 qk_gain, windows = VE resid qk_gain, which put
# the suspected interactions (VE x x0, x0 x resid, schedule x weight decay, qk_gain x windows) in four alias chains
FACTORS = {"value_embeds": (True, False), "x0": (True, False), "resid": (True, False), "qk_gain": (True, False),
           "schedule": ("wsd", "d2z"), "weight_decay": ("fixed", "tau"), "windows": (True, False)}
DESIGN = "a b c d abc abd acd"
COMPONENTS, COMPUTE, REGIME = ("value_embeds", "x0", "resid", "qk_gain"), ("windows",), ("schedule", "weight_decay")
LONG_CONTEXT = {"canonical": {}, "rewarm": {"repo_schedule": "rewarm"}, "rope": {"rope": "single"},
                "rewarm-rope": {"repo_schedule": "rewarm", "rope": "single"}}
D24 = 24
CONFIRM_SEEDS = ((0, 0), (1, 1))
# Octaves of the law's production lr, bracketing the short horizons' optimum, which lies above it (eta* falls with the
# horizon: Bjorck et al., arXiv 2409.19913); horizons doubling from 0.5B tokens (2.6 tokens per scaling parameter)
LR_SCALES = (0.5, 1.0, 2.0, 4.0)
HORIZONS = (5e8, 1e9, 2e9)
CONFIRM_TOKENS = OPTIMIZER_TOKENS = HORIZONS[0]


@dataclass(frozen=True, kw_only=True)
class Arm:
    """A study's arm: a training run's configuration (Variant fields departing from the canonical model), the parent
    run whose kept checkpoint (tag, step) it continues, and the steps whose checkpoints it keeps."""
    study: str
    name: str
    depth: int
    ratio: float
    seed: int
    data_seed: int
    variant: dict = field(default_factory=dict)
    parent: tuple[str, int] | None = None
    keep: tuple[int, ...] = ()

    @property
    def tag(self) -> str:
        """The run's identity: config.run_tag, the data seed and every departing field."""
        fields = "".join(f"-{k}={v:g}" if isinstance(v, float) else f"-{k}={v}" for k, v in sorted(self.variant.items()))
        return f"{run_tag(self.depth, self.ratio, self.seed)}-d{self.data_seed}{fields}"


def departures(**fields) -> dict:
    """The fields that depart from the canonical Variant (its defaults)."""
    canonical = Variant()
    return {k: v for k, v in fields.items() if getattr(canonical, k) != v}


def record(tag: str) -> dict:
    """An arm's evaluation record (experiments/arm.py)."""
    with open(os.path.join(base_dir(), "evals", f"{tag}.json"), encoding="utf-8") as f:
        return json.load(f)


def metric(tag: str) -> float:
    return record(tag)["eval"][METRIC[0]]["bpb"][METRIC[1]]


@cache
def production_ratio() -> float:
    """The production run's tokens per scaling parameter (runs/h100.sh: train/stages.py max_ratio at d24)."""
    return max_ratio(D24, get_tokenizer())


# The studies' arms


def noise() -> list[Arm]:
    return [Arm(study="noise", name=f"s{s}-d{d}", depth=D8, ratio=RATIO8, seed=s, data_seed=d)
            for s in NOISE_SEEDS for d in NOISE_DATA_SEEDS]


def screen_points() -> list[dict[str, int]]:
    """The design's 16 points: {factor: +1 current / -1 alternative} (pydoe's Yates order)."""
    return [dict(zip(FACTORS, map(int, row), strict=True)) for row in pydoe.fracfact(DESIGN)]


def screen() -> list[Arm]:
    return [Arm(study="screen", name="".join("1" if p[f] > 0 else "0" for f in FACTORS), depth=D8, ratio=RATIO8,
                seed=0, data_seed=0, variant=departures(**{f: FACTORS[f][p[f] < 0] for f in FACTORS}))
            for p in screen_points()]


def long_context() -> list[Arm]:
    return [Arm(study="long_context", name=name, depth=D8, ratio=RATIO8, seed=0, data_seed=0, variant=v)
            for name, v in LONG_CONTEXT.items()]


def confirm() -> list[Arm]:
    """The d24 pairs (canonical, pruned) per seed pair; none when the screen keeps the canonical architecture."""
    adopted = verdict()
    architecture = {k: v for k, v in adopted.items() if k not in REGIME}
    regime = {k: v for k, v in adopted.items() if k in REGIME}
    if not architecture:
        return []
    return [Arm(study="confirm", name=f"{name}-s{s}-d{d}", depth=D24, ratio=production_ratio(), seed=s, data_seed=d,
                variant=departures(**regime, **(architecture if name == "pruned" else {}), tokens=CONFIRM_TOKENS))
            for s, d in CONFIRM_SEEDS for name in ("canonical", "pruned")]


def lr_horizon() -> list[Arm]:
    variant, ratio = final(), production_ratio()
    run = Variant(**variant)
    branch = run.schedule == "wsd" and run.weight_decay == "fixed"
    arms = []
    for scale in LR_SCALES:
        parent = None
        for i, tokens in enumerate(HORIZONS):
            child = branch and i + 1 < len(HORIZONS)
            keep = (cooldown_start(plan(D24, ratio, tokens).num_iterations),) if child else ()
            arm = Arm(study="lr_horizon", name=f"lr{scale:g}-t{tokens:g}", depth=D24, ratio=ratio, seed=0,
                      data_seed=0, variant=departures(**variant, lr_scale=scale, tokens=tokens), parent=parent,
                      keep=keep)
            arms.append(arm)
            parent = (arm.tag, keep[0]) if child else None
    return arms


def optimizer() -> list[Arm]:
    variant, ratio = final(), production_ratio()
    return [Arm(study="optimizer", name=f"adamw-lr{scale:g}", depth=D24, ratio=ratio, seed=0, data_seed=0,
                variant=departures(**variant, optimizer="adamw", lr_scale=scale, tokens=OPTIMIZER_TOKENS))
            for scale in LR_SCALES]


def batch() -> list[Arm]:
    """The noise-scale measurements' runs: lr_horizon's at the law's lr, one per horizon (their final models)."""
    return [a for a in lr_horizon() if "lr_scale" not in a.variant]


STUDIES = {"noise": noise, "screen": screen, "long_context": long_context, "confirm": confirm,
           "lr_horizon": lr_horizon, "optimizer": optimizer}


# Decisions


def noise_sd() -> tuple[float, int]:
    """The noise study's SD of the metric and its degrees of freedom."""
    values = [metric(a.tag) for a in noise()]
    return float(np.std(values, ddof=1)), len(values) - 1


def screen_effects() -> pd.DataFrame:
    """Per factor: effect = metric(alternative) - metric(current) (OLS on the +-1 design: -2 x its coefficient; the
    residual holds the aliased interaction chains, so the coefficients' own SEs are not used), its noise SE and t,
    p one-sided in the rule's direction (components: the current choice helps; windows and regime: the alternative
    is better), Holm-adjusted across the factors, Lenth's PSE / ME / SME over the 15 contrasts, the effect on tok/s
    (relative), and the decision `adopt` (the alternative replaces the current choice)."""
    points = screen_points()
    frame = pd.DataFrame([{**p, "bpb": metric(a.tag), "tok_per_s": summary(a)["tok_per_s"]}
                          for a, p in zip(screen(), points, strict=True)])
    factors = list(FACTORS)
    effect = -2 * smf.ols("bpb ~ " + " + ".join(factors), data=frame).fit().params[factors]
    speed = -2 * smf.ols("tok_per_s ~ " + " + ".join(factors), data=frame).fit().params[factors] / frame.tok_per_s.mean()
    sigma, df = noise_sd()
    se = 2 * sigma / math.sqrt(len(points))
    contrasts = 2 * smf.ols("bpb ~ " + " * ".join(factors[:4]), data=frame).fit().params.drop("Intercept").abs()
    s0 = 1.5 * contrasts.median()
    pse = 1.5 * contrasts[contrasts < 2.5 * s0].median()
    m = len(contrasts)
    me, sme = scipy.stats.t.ppf([0.975, (1 + 0.95 ** (1 / m)) / 2], m / 3) * pse
    direction = pd.Series({f: 1 if f in COMPONENTS else -1 for f in factors})
    p = scipy.stats.t.sf(direction * effect / se, df)
    holm = multipletests(p, alpha=ALPHA, method="holm")[1]
    significant = (holm < ALPHA) & (direction * effect > me)
    adopt = np.where(direction > 0, ~significant, significant)
    return pd.DataFrame({"alternative": [FACTORS[f][1] for f in factors], "effect": effect, "se": se,
                         "t": effect / se, "df": df, "p": p, "p_holm": holm, "lenth_pse": pse, "lenth_me": me,
                         "lenth_sme": sme, "tok_per_s_effect": speed, "adopt": adopt}, index=factors)


def verdict() -> dict:
    """The screen's decision: the Variant fields of the alternatives it adopts."""
    effects = screen_effects()
    return departures(**{f: FACTORS[f][1] for f in effects.index[effects.adopt]})


def final() -> dict:
    """The variant of the d24 regime studies: the screen's regime verdict, and its architecture verdict unless the
    pruned model loses on both of confirm's seed pairs."""
    adopted = verdict()
    pairs = confirm()
    if pairs and all(metric(pruned.tag) > metric(canonical.tag) for canonical, pruned in zip(pairs[::2], pairs[1::2],
                                                                                              strict=True)):
        return {k: v for k, v in adopted.items() if k in REGIME}
    return adopted


# Tidy tables


def summary(arm: Arm) -> dict:
    """The run's train summary from its metrics (curves): median tok/s (each process's first step, which compiles,
    excluded) and the fraction of steps whose gradient norm exceeded the clip (optim.GRAD_CLIP)."""
    frame = curves(arm.tag, {})
    return {"tok_per_s": frame.tok_per_s[~frame.first_of_process].median(),
            "clipped": float((frame.grad_norm > GRAD_CLIP).mean())}


def metrics(tag: str) -> pd.DataFrame:
    return pd.read_json(os.path.join(base_dir(), "metrics", f"{tag}.jsonl"), lines=True)


def runs(tag: str) -> pd.DataFrame:
    frame = metrics(tag)
    return frame[frame.kind == "run"]


def curves(tag: str, parents: dict) -> pd.DataFrame:
    """The run's train records, one per step (a relaunch's repeat of a step replaces it), each process's first step
    marked; a branch's ({tag: (parent tag, branch step)}) preceded by its ancestry's steps before the branch point."""
    frame = metrics(tag)
    starts = set(frame[frame.kind == "run"].step)
    train = frame[frame.kind == "train"].drop_duplicates("step", keep="last").sort_values("step")
    train = train.assign(first_of_process=train.step.isin(starts))[
        ["step", "tokens", "loss", "lr", "grad_norm", "tok_per_s", "first_of_process"]]
    if parents.get(tag):
        parent, step = parents[tag]
        ancestry = curves(parent, parents)
        train = pd.concat([ancestry[ancestry.step < step], train], ignore_index=True)
    return train.reset_index(drop=True)


def registered() -> list[Arm]:
    return [a for arms in STUDIES.values() for a in arms()]


def evaluated(arms: list[Arm]) -> list[Arm]:
    """The arms with an evaluation record."""
    return [a for a in arms if os.path.exists(os.path.join(base_dir(), "evals", f"{a.tag}.json"))]


def collect() -> None:
    """results/studies/arms.csv (one row per study arm: its configuration, every Variant field, the plan and the
    optimizer hyperparameters of its first process, its train summary, its group bits per byte), eval.csv (one row per
    arm, evaluation, corpus or benchmark and context bucket: bits, bytes, rows) and curves.parquet (one row per arm and
    step)."""
    arms, evals, steps = [], [], []
    everything = registered()
    parents = {a.tag: a.parent for a in everything}
    for arm in evaluated(everything):
        r, first = record(arm.tag), runs(arm.tag).iloc[0]
        row = {"study": arm.study, "arm": arm.name, "tag": arm.tag, "commit": r["commit"], "depth": arm.depth,
               "ratio": arm.ratio, "seed": arm.seed, "data_seed": arm.data_seed,
               **dataclasses.asdict(Variant(**arm.variant)), "parent": arm.parent and arm.parent[0],
               "branch_step": arm.parent and arm.parent[1], **{f"plan_{k}": v for k, v in first["plan"].items()},
               "lr": first.get("lr"), "weight_decay": first.get("weight_decay"), "eval_step": r["step"],
               **summary(arm)}
        for name, result in r["eval"].items():
            if name == "solution":
                evals += [{"tag": arm.tag, "eval": name, "corpus": b, "bucket": i, **bucket(s, i)}
                          for b, s in result.items() for i in range(len(s["bits"]))]
                continue
            row |= {f"bpb_{name}_{g}": v for g, v in result["bpb"].items()}
            evals += [{"tag": arm.tag, "eval": name, "corpus": c, "bucket": i, "rows": s["rows"], **bucket(s, i)}
                      for c, s in result["corpora"].items() for i in range(len(s["bits"]))]
        arms.append(row)
        steps.append(curves(arm.tag, parents).assign(tag=arm.tag))
    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame(arms).to_csv(os.path.join(RESULTS, "arms.csv"), index=False)
    pd.DataFrame(evals).drop_duplicates().to_csv(os.path.join(RESULTS, "eval.csv"), index=False)
    pd.concat(steps).drop_duplicates(["tag", "step"]).to_parquet(os.path.join(RESULTS, "curves.parquet"), index=False)


def bucket(sums: dict, i: int) -> dict:
    """Context bucket i of evaluated sums: its context range (loss_eval.CONTEXTS) and its bits and bytes."""
    edges = (0, *CONTEXTS, math.inf)
    return {"context_min": edges[i] + 1, "context_max": edges[i + 1], "bits": sums["bits"][i],
            "bytes": sums["bytes"][i]}


# Analyses: each writes results/studies/<study>.md (the appendix table) and <study>.csv (its numbers)


def lr_optimum(frame: pd.DataFrame) -> dict:
    """The lr minimizing a quadratic in x = ln lr fitted by OLS to the metric: ln eta* = -b1 / (2 b2) with its
    delta-method SE (gradient (-1 / (2 b2), b1 / (2 b2^2)) through the coefficients' covariance), the fitted minimum,
    the curvature b2 (> 0 for a minimum) and whether eta* lies inside the grid."""
    fit = smf.ols("bpb ~ x + I(x ** 2)", data=frame.assign(x=np.log(frame.lr))).fit()
    b1, b2 = fit.params["x"], fit.params["I(x ** 2)"]
    x = -b1 / (2 * b2)
    gradient = np.array([-1 / (2 * b2), b1 / (2 * b2 ** 2)])
    cov = fit.cov_params().loc[["x", "I(x ** 2)"], ["x", "I(x ** 2)"]].to_numpy()
    return {"ln_lr": x, "ln_lr_se": float(np.sqrt(gradient @ cov @ gradient)), "lr": math.exp(x), "curvature": b2,
            "bpb_min": float(fit.predict(pd.DataFrame({"x": [x]}))[0]), "df_resid": fit.df_resid,
            "inside": bool(np.log(frame.lr).min() <= x <= np.log(frame.lr).max())}


def table(frame: pd.DataFrame, title: str) -> str:
    """A titled Markdown table of a DataFrame (pandas to_markdown, tabulate's pipe format)."""
    return f"### {title}\n\n{frame.to_markdown(floatfmt='.5g')}\n"


def per_arm(arms: list[Arm]) -> pd.DataFrame:
    rows = {a.name: {"tag": a.tag, "bpb": metric(a.tag), **summary(a)} for a in arms}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("arm")


def analyze_noise() -> tuple[list[str], pd.DataFrame]:
    frame = pd.DataFrame([{"seed": a.seed, "data_seed": a.data_seed, "bpb": metric(a.tag)} for a in noise()])
    anova = sm.stats.anova_lm(smf.ols("bpb ~ C(seed) + C(data_seed)", data=frame).fit(), typ=1)
    ms = anova["mean_sq"]
    components = pd.DataFrame({"variance": [
        max(0.0, (ms["C(seed)"] - ms["Residual"]) / len(NOISE_DATA_SEEDS)),
        max(0.0, (ms["C(data_seed)"] - ms["Residual"]) / len(NOISE_SEEDS)), ms["Residual"]]},
        index=pd.Index(["init seed", "data seed", "residual"], name="component"))
    sigma, df = noise_sd()
    return [table(frame.set_index("seed"), "Noise: the metric per init x data seed"),
            table(anova, "Two-way ANOVA (type 1)"), table(components, "Variance components (method of moments)"),
            f"SD of all {df + 1} runs {sigma:.5f} ({df} df)\n"], frame


def analyze_screen() -> tuple[list[str], pd.DataFrame]:
    effects = screen_effects()
    aliases = pydoe.fracfact_aliasing(pydoe.fracfact(DESIGN))[0]
    letters = dict(zip("abcdefg", FACTORS, strict=True))
    chains = ["- " + " = ".join("*".join(letters[c] for c in word) for word in chain.split(" = ")) for chain in aliases]
    return [table(per_arm(screen()), "Screen: arms"), table(effects.rename_axis("factor"), "Screen: main effects"),
            "### Alias chains\n\n" + "\n".join(chains) + "\n"], effects


def analyze_long_context() -> tuple[list[str], pd.DataFrame]:
    weights = mixture_weights(REPO_STAGE.mixture, get_tokenizer(), REPO_STAGE.seq_len)
    rows = []
    for arm in long_context():
        result = record(arm.tag)["eval"]
        corpora = result["32k"]["corpora"]
        # each context bucket's mixture bits per byte (eval/loss_eval.py mixture_bpb, the corpora's bucket alone)
        buckets = {f"bpb_32k_context_le_{edge:g}": mixture_bpb(
            {c: {"bits": [s["bits"][i]], "bytes": [s["bytes"][i]], "rows": s["rows"]} for c, s in corpora.items()},
            weights) for i, edge in enumerate((*CONTEXTS, math.inf))}
        rows.append({"arm": arm.name, "rewarm": -1 if "repo_schedule" in arm.variant else 1,
                     "rope": -1 if "rope" in arm.variant else 1, "bpb_8k": result["8k"]["bpb"]["code"],
                     "bpb_32k": result["32k"]["bpb"]["all"], **buckets})
    frame = pd.DataFrame(rows).set_index("arm")
    sigma, _ = noise_sd()
    outcomes = [c for c in frame.columns if c.startswith("bpb")]
    # one replicate per cell: each effect's SE is the noise SD (2 sigma / sqrt(4)); the interaction as -2 x its
    # coefficient, the alternatives' joint departure from additivity
    effects = pd.DataFrame({y: -2 * smf.ols(f"{y} ~ rewarm * rope", data=frame).fit().params[
        ["rewarm", "rope", "rewarm:rope"]] for y in outcomes}).rename_axis("effect")
    return [table(frame, "Long-context stage: arms"),
            table(effects, f"Long-context stage: effects, alternative - current (SE: the noise SD {sigma:.5f})")], \
        effects


def analyze_confirm() -> tuple[list[str], pd.DataFrame]:
    arms = confirm()
    if not arms:
        return ["The screen kept the canonical architecture: nothing to confirm.\n"], pd.DataFrame()
    frame = pd.DataFrame([{"seeds": c.name.split("-", 1)[1], "canonical": metric(c.tag), "pruned": metric(p.tag)}
                          for c, p in zip(arms[::2], arms[1::2], strict=True)]).set_index("seeds")
    frame["pruned - canonical"] = frame.pruned - frame.canonical
    return [table(frame, f"Confirmation at d24 of {json.dumps(verdict())}"),
            f"Final variant of the d24 studies: `{json.dumps(final())}`\n"], frame


def horizon_frame(arms: list[Arm]) -> pd.DataFrame:
    rows = []
    for a in arms:
        run = runs(a.tag).iloc[0]
        rows.append({"arm": a.name, "tokens": Variant(**a.variant).tokens, "lr": run["lr"], "bpb": metric(a.tag)})
    return pd.DataFrame(rows)


def analyze_lr_horizon() -> tuple[list[str], pd.DataFrame]:
    """The optimum per horizon, then (from the horizons whose fit has an interior minimum: positive curvature, eta*
    inside the grid) the WLS line of ln eta* on ln D with the optima's delta-method variances as known variances,
    extrapolated to the production run's file-level tokens (decision: its lr replaces the law's only if the 95%
    interval excludes the law's)."""
    frame = horizon_frame(lr_horizon())
    optima = pd.DataFrame({t: lr_optimum(g) for t, g in frame.groupby("tokens")}).T.rename_axis("tokens")
    sections = [table(frame.set_index("arm"), "lr x horizon: arms"), table(optima, "Optimal lr per horizon")]
    valid = optima[(optima.curvature.astype(float) > 0) & optima.inside.astype(bool)]
    if len(valid) < 2:
        return [*sections, f"{len(valid)} horizon(s) with an interior optimum: no horizon law.\n"], optima
    fit = sm.WLS(valid.ln_lr.astype(float), sm.add_constant(np.log(valid.index.to_numpy(dtype=float))),
                 weights=1 / valid.ln_lr_se.astype(float) ** 2).fit(cov_type="fixed scale")
    production = plan(D24, production_ratio())
    file_tokens = production.repo_start * production.total_batch_size
    prediction = fit.get_prediction(np.array([[1.0, math.log(file_tokens)]])).summary_frame(alpha=ALPHA)
    lower, upper = (math.exp(prediction[k].iloc[0]) for k in ("mean_ci_lower", "mean_ci_upper"))
    law = pd.DataFrame({"law lr": [production.lr], "extrapolated lr": [math.exp(prediction["mean"].iloc[0])],
                        "lower": [lower], "upper": [upper], "beta": [-fit.params.iloc[1]], "beta se": [fit.bse.iloc[1]],
                        "replace the law": [not lower <= production.lr <= upper]},
                       index=pd.Index([f"{file_tokens:.4g} tokens"], name="horizon"))
    return [*sections, table(law, "eta*(D) = a D^-beta at the production run's file-level tokens")], optima


def analyze_optimizer() -> tuple[list[str], pd.DataFrame]:
    muon = horizon_frame([a for a in lr_horizon() if Variant(**a.variant).tokens == OPTIMIZER_TOKENS])
    adamw = horizon_frame(optimizer())
    optima = pd.DataFrame({"muon": lr_optimum(muon), "adamw": lr_optimum(adamw)}).T.rename_axis("optimizer")
    return [table(pd.concat([muon.assign(optimizer="muon"), adamw.assign(optimizer="adamw")]).set_index("arm"),
                  f"Optimizers at {OPTIMIZER_TOKENS:g} tokens: arms"),
            table(optima, "Optimizers: fitted optimum each")], optima


def analyze_batch() -> tuple[list[str], pd.DataFrame]:
    production = plan(D24, production_ratio())
    rows = []
    for arm in batch():
        with open(os.path.join(base_dir(), "noise_scale", f"{arm.tag}.json"), encoding="utf-8") as f:
            rows.append(json.load(f))
    frame = pd.DataFrame(rows).set_index("tokens")
    fit = sm.OLS(np.log(frame.b_simple), sm.add_constant(np.log(frame.index.to_numpy(dtype=float)))).fit()
    file_tokens = production.repo_start * production.total_batch_size
    prediction = fit.get_prediction(np.array([[1.0, math.log(file_tokens)]])).summary_frame(alpha=ALPHA)
    trend = pd.DataFrame({"batch": [production.total_batch_size], "B_simple": [math.exp(prediction["mean"].iloc[0])],
                          "lower": [math.exp(prediction["mean_ci_lower"].iloc[0])],
                          "upper": [math.exp(prediction["mean_ci_upper"].iloc[0])], "slope": [fit.params.iloc[1]]},
                         index=pd.Index([f"{file_tokens:.4g} tokens"], name="horizon"))
    return [table(frame, "Gradient noise scale at the law's lr"),
            table(trend, "ln B_simple ~ ln D (OLS), at the production run's file-level tokens")], frame


ANALYSES = {"noise": analyze_noise, "screen": analyze_screen, "long_context": analyze_long_context,
            "confirm": analyze_confirm, "lr_horizon": analyze_lr_horizon, "optimizer": analyze_optimizer,
            "batch": analyze_batch}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("arms", "analyze", "collect"))
    parser.add_argument("study", nargs="?", choices=(*STUDIES, "batch"))
    args = parser.parse_args()
    if args.command == "arms":
        for arm in (batch() if args.study == "batch" else STUDIES[args.study]()):
            print("arm", arm.name, arm.tag)
    elif args.command == "analyze":
        sections, numbers = ANALYSES[args.study]()
        os.makedirs(RESULTS, exist_ok=True)
        with open(os.path.join(RESULTS, f"{args.study}.md"), "w", encoding="utf-8") as f:
            f.write(f"## {args.study}\n\n" + "\n".join(sections))
        numbers.to_csv(os.path.join(RESULTS, f"{args.study}.csv"))
    else:
        collect()


if __name__ == "__main__":
    main()
