"""
The ablation study of the mechanism (docs/h100-pretraining.md): which components earn their place, by the val bits per
byte of d8 proxies at DEPTH and RATIO, each arm one run of experiments/arm.py.

- noise: the canonical model at init seeds x data seeds (SEEDS x DATA_SEEDS): the spread of the metric across runs
  that differ only by randomness (Summers & Dinneen, arXiv 2103.04514), the noise every effect is judged against.
- screen: a 2^(7-3) resolution-IV fractional factorial over the components (FACTORS; Box & Hunter 1961), every arm at
  init and data seed 0 (common random numbers: model/gpt.py init_weights seeds each tensor by its name). Main effects
  are not aliased with two-factor interactions; the generators (schedule = value_embeds x0 resid, weight_decay =
  value_embeds x0 qk_gain, windows = value_embeds resid qk_gain) put the suspected interactions (value_embeds x x0,
  x0 x resid, schedule x weight_decay, qk_gain x windows) in four different alias chains. A main effect in a two-level
  factorial is the component's Banzhaf value (Wang & Jia, arXiv 2205.15466).

The metric is the mean bits per byte over the code corpora (config.MIXTURE "code"); effects are bpb(alternative) -
bpb(current), positive when the current component helps, with standard error sigma / 2 (16 runs, sigma from the noise
arms) and Lenth's pseudo standard error from the effects themselves (Technometrics 31(4), 1989).

    python3 -m experiments.design runs                   # TAG SEED DATA_SEED VARIANT per line (experiments/run.sh)
    python -m experiments.design analyze --out=<file.md>  # in the image: the noise and the effects
"""
import argparse
import itertools
import json
import math
import os
import statistics

# The proxy: d8, the smallest model of the family, at 100 tokens per scaling parameter, a screening budget below the
# production 490 (effects are confirmed at the production ratio before they are acted on)
DEPTH, RATIO = 8, 100
SEEDS, DATA_SEEDS = (0, 1, 2, 3), (0, 1)
# {factor: (current, alternative)}: each component of the mechanism against its simplest alternative
FACTORS = {"value_embeds": (True, False), "x0": (True, False), "resid": (True, False), "qk_gain": (True, False),
           "schedule": ("wsd", "d2z"), "weight_decay": ("fixed", "tau"), "windows": (True, False)}
GENERATORS = {"schedule": ("value_embeds", "x0", "resid"), "weight_decay": ("value_embeds", "x0", "qk_gain"),
              "windows": ("value_embeds", "resid", "qk_gain")}
CODE = ("stack_edu_python", "ultradata_code_l2_python", "ultradata_stack_v3_l2_python", "kaggle_notebooks",
        "codeforces_cots", "stack_edu_cpp", "ultradata_code_l2_cpp", "ultradata_stack_v3_l2_cpp")


def screen():
    """The 16 design points: {factor: +1 current / -1 alternative}."""
    base = [f for f in FACTORS if f not in GENERATORS]
    points = []
    for signs in itertools.product((1, -1), repeat=len(base)):
        point = dict(zip(base, signs, strict=True))
        for factor, word in GENERATORS.items():
            point[factor] = point[word[0]] * point[word[1]] * point[word[2]]
        points.append(point)
    return points


def variant(point):
    """The Variant fields of a design point (the alternatives only)."""
    return {f: FACTORS[f][1] for f, s in point.items() if s < 0}


def tag(seed, data_seed, point):
    return f"ab-d{DEPTH}-r{RATIO}-s{seed}-d{data_seed}-" + "".join("1" if point[f] > 0 else "0" for f in FACTORS)


def runs():
    """Every arm: the noise arms, then the screen's (its canonical point is the noise arm at seeds 0, 0)."""
    canonical = {f: 1 for f in FACTORS}
    arms = [(tag(s, d, canonical), s, d, {}) for s in SEEDS for d in DATA_SEEDS]
    arms += [(tag(0, 0, p), 0, 0, variant(p)) for p in screen() if p != canonical]
    return arms


def analyze(out):
    from nanochat.paths import base_dir

    def metric(t):
        with open(os.path.join(base_dir(), "ablation", f"{t}.json"), encoding="utf-8") as f:
            r = json.load(f)
        return statistics.fmean(r["bpb"][c] for c in CODE), r

    canonical = {f: 1 for f in FACTORS}
    noise = {(s, d): metric(tag(s, d, canonical))[0] for s in SEEDS for d in DATA_SEEDS}
    sigma = statistics.stdev(noise.values())
    lines = ["### Noise: canonical model, code bits per byte by init seed x data seed", "",
             "| init seed | " + " | ".join(f"data {d}" for d in DATA_SEEDS) + " |", "|---" * (len(DATA_SEEDS) + 1) + "|"]
    lines += [f"| {s} | " + " | ".join(f"{noise[s, d]:.5f}" for d in DATA_SEEDS) + " |" for s in SEEDS]
    seed_means = [statistics.fmean(noise[s, d] for d in DATA_SEEDS) for s in SEEDS]
    data_means = [statistics.fmean(noise[s, d] for s in SEEDS) for d in DATA_SEEDS]
    lines += ["", f"sigma (all {len(noise)} runs) {sigma:.5f}; spread of init-seed means "
              f"{statistics.stdev(seed_means):.5f}; of data-seed means {statistics.stdev(data_means):.5f}", ""]
    points = screen()
    values = [metric(tag(0, 0, p)) for p in points]
    effects = {f: statistics.fmean(v for (v, _), p in zip(values, points) if p[f] < 0)
               - statistics.fmean(v for (v, _), p in zip(values, points) if p[f] > 0) for f in FACTORS}
    # Lenth's PSE over all 15 orthogonal contrasts of the 16 runs (the products of the base factors' signs)
    base = [f for f in FACTORS if f not in GENERATORS]
    contrasts = []
    for k in range(1, len(base) + 1):
        for subset in itertools.combinations(base, k):
            signs = [math.prod(p[f] for f in subset) for p in points]
            contrasts.append(statistics.fmean(v * s for (v, _), s in zip(values, signs)) * 2)
    s0 = 1.5 * statistics.median(abs(c) for c in contrasts)
    pse = 1.5 * statistics.median(abs(c) for c in contrasts if abs(c) < 2.5 * s0)
    se = sigma / 2
    lines += ["### Screen: main effects, bpb(alternative) - bpb(current)", "",
              f"| component | alternative | effect | t (noise SE {se:.5f}) | t (Lenth PSE {pse:.5f}) |",
              "|---|---|---|---|---|"]
    lines += [f"| {f} | {FACTORS[f][1]} | {e:+.5f} | {e / se:+.2f} | {e / pse:+.2f} |" for f, e in effects.items()]
    lines += ["", "### Screen: arms", "", "| tag | code bpb | final train loss | tok/s | clipped |", "|---|---|---|---|---|"]
    lines += [f"| `{r['tag']}` | {v:.5f} | {r['final_loss']:.4f} | {r['tok_per_s']:,.0f} | {r['clipped']:.3f} |"
              for v, r in values]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("runs", "analyze"))
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.command == "runs":
        for t, s, d, v in runs():
            print(t, s, d, json.dumps(v, separators=(",", ":")))
    else:
        analyze(args.out)
