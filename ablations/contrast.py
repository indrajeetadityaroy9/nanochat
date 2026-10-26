"""
The ablations' pre-registered tests of an arm against its baseline, over the seeds where both have an evaluation (a
diverged run drops its seed from the arm's contrasts on both sides):
- parsing: per PARSED problem, the seeds whose healed greedy completion compiles under the arm minus under the
  baseline; the exact sign test (scipy.stats.binomtest) over the problems whose difference is not 0 (McNemar's exact
  test at one seed, the test of TokEval's companion post, cimeister.github.io/blog/code_tokenizer_ablations, its
  App. D); better or worse at p < ALPHA, else equal. MBPP+ alone: its prompts leave the function header to the model,
  and "an evaluation that uses HumanEval alone does not expose this type of defect" (the post, which finds that code
  bits per byte "shows no association with MBPP at any level");
- code bits per byte: the difference of the arms' means, better or worse beyond its 1 - ALPHA two-sided normal bound
  z CODE_SIGMA √(1 / n + 1 / n), z = scipy.stats.norm.ppf(1 - ALPHA / 2), n the contrast's seeds per arm, else equal.
An arm every seed of which diverged (no evaluation) counts as worse than any evaluated one, as ablations/peak_lr.py fit
counts a diverged run: no seeds, so parsing equal (untested) and code worse (better if the baseline is the one without
an evaluation), its difference and bound None.
An arm is adopted iff its code is better and its parsing not worse (adopted). A scale swept over {low, 1, high}
(scaled): the vertex of the exact 3-point quadratic in ln(scale) through the arms' mean code bits per byte, clamped to
[low, high] (a concave or flat fit, or an arm without an evaluation, whose mean is +inf, reported None: the lower-code
end), adopted iff the lower-code arm is adopted against 1; else 1.
"""
import math
import statistics

import numpy as np
from scipy.stats import binomtest, norm

import metrics

# The decisions' significance level, the conventional 0.05
ALPHA = 0.05
# The pooled seed standard deviation of a run's 8K code bits per byte (seed variances pooled over configurations): two
# d8 configurations at 20 tokens per parameter on the reference tokenizer's data, data seeds 0 to 2 (six runs),
# assumed to carry to other depths and to the corpus
CODE_SIGMA = 0.0034
# The benchmark of the parse test
PARSED = "mbpp_plus"


def evaluated(lines: list[dict], field: str) -> dict[tuple, tuple[float, list[bool]]]:
    """{(the run's `field`, seed): (code bits per byte, per PARSED problem in the benchmark's order whether its healed
    greedy completion compiles)} of every evaluated run in an experiment's records (a diverged run has none)."""
    return {(line["config"][field], line["config"]["seed"]): (
        line["measures"]["code"], metrics.parsed(line["evaluations"][-1]["parse"][PARSED]["outcomes"]))
        for line in lines if line["measures"] is not None}


def contrast(runs: dict[tuple, tuple[float, list[bool]]], arm: object, baseline: object) -> dict:
    """An arm against the baseline (module docstring; evaluated's runs) over the seeds where both have an evaluation:
    parsing by the exact sign test, code bits per byte by the difference of the means beyond its bound; an arm without
    an evaluation (every seed diverged) worse."""
    seeds = sorted(seed for value, seed in runs if value == arm and (baseline, seed) in runs)
    # per problem, the seeds whose completion parses under the arm minus under the baseline (0 without seeds)
    seeds_parsing = (np.sum([runs[arm, seed][1] for seed in seeds], axis=0)
                     - np.sum([runs[baseline, seed][1] for seed in seeds], axis=0))
    here, there = int((seeds_parsing > 0).sum()), int((seeds_parsing < 0).sum())
    p = binomtest(here, here + there).pvalue if here + there else 1.0
    difference = margin = None
    if seeds:
        difference = (statistics.fmean(runs[arm, seed][0] for seed in seeds)
                      - statistics.fmean(runs[baseline, seed][0] for seed in seeds))
        margin = norm.ppf(1 - ALPHA / 2) * CODE_SIGMA * math.sqrt(2 / len(seeds))
        code = "better" if difference < -margin else "worse" if difference > margin else "equal"
    elif all(value != arm for value, _ in runs):
        code = "worse"
    elif all(value != baseline for value, _ in runs):
        code = "better"
    else:
        raise ValueError(f"{arm} and {baseline} have evaluations at no common seed")
    return {"seeds": seeds, "problems_parsing_more_here": here, "problems_parsing_more_in_baseline": there, "p": p,
            "parsing": "equal" if p >= ALPHA else "better" if here > there else "worse",
            "code_difference": difference, "code_margin": margin, "code": code}


def adopted(c: dict) -> bool:
    """Whether a contrast adopts its arm (module docstring)."""
    return c["code"] == "better" and c["parsing"] != "worse"


def scaled(lines: list[dict], field: str) -> dict:
    """A scale's decision over the arms {low, 1, high} of `field`, the values its records swept (module docstring):
    "scale" the adopted scale."""
    runs = evaluated(lines, field)
    scales = low, _, high = sorted({line["config"][field] for line in lines})
    codes = {s: [code for (v, _), (code, _) in runs.items() if v == s] for s in scales}
    means = {s: statistics.fmean(c) if c else math.inf for s, c in codes.items()}
    curvature, x = (metrics.parabola_vertex([math.log(s) for s in scales], [means[s] for s in scales])
                    if all(map(math.isfinite, means.values())) else (None, None))
    end = min((low, high), key=means.__getitem__)
    vertex = min(max(math.exp(x), low), high) if curvature is not None and curvature > 0 else end
    contrasts = {str(s): contrast(runs, s, 1.0) for s in (low, high)}
    return {"means": {str(s): m if math.isfinite(m) else None for s, m in means.items()}, "curvature": curvature,
            "vertex": vertex, "contrasts": contrasts, "scale": vertex if adopted(contrasts[str(end)]) else 1.0}
