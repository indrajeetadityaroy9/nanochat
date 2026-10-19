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

- regime: one-at-a-time departures of the canonical model at seeds 0, 0: the peak lr on a half-octave log grid (lr
  sensitivity, Wortsman et al., arXiv 2309.14322 §2), the batch at 2x and 4x the law's (at d8 the law's batch is one
  micro-batch, the floor), no gradient clipping, the repo-level stage's own re-warmed schedule, the single RoPE base,
  and both; with a repeat of the canonical arm under its own tag (equal to the screen's bit for bit: the harness leaves
  the canonical run unchanged).

The metric is the mean bits per byte over the code corpora (config.MIXTURE "code"); effects are bpb(alternative) -
bpb(current), positive when the current component helps, with standard error sigma / 2 (16 runs, sigma from the noise
arms) and Lenth's pseudo standard error from the effects themselves (Technometrics 31(4), 1989); a regime arm's
difference from the canonical has standard error sigma sqrt(2). The regime phase also reports the repository corpora's
bits per byte at 32K (bpb32).

    python3 -m experiments.design runs <phase>                   # TAG SEED DATA_SEED VARIANT per line (run.sh)
    python -m experiments.design analyze <phase> --out=<file.md>  # in the image: the tables
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


REGIME = {"canonical": {}, "lr0.5": {"lr_scale": 0.5}, "lr0.71": {"lr_scale": 2 ** -0.5},
          "lr1.41": {"lr_scale": 2 ** 0.5}, "lr2": {"lr_scale": 2.0}, "batch2": {"batch_scale": 2.0},
          "batch4": {"batch_scale": 4.0}, "noclip": {"grad_clip": math.inf}, "rewarm": {"repo_schedule": "rewarm"},
          "rope1e6": {"rope": "single"}, "rewarm-rope1e6": {"repo_schedule": "rewarm", "rope": "single"}}


def runs(phase):
    """The arms of a phase. screen: the noise arms, then the screen's (its canonical point is the noise arm at seeds 0,
    0); regime: REGIME's."""
    if phase == "regime":
        return [(f"ab-d{DEPTH}-r{RATIO}-s0-d0-{name}", 0, 0, v) for name, v in REGIME.items()]
    canonical = {f: 1 for f in FACTORS}
    arms = [(tag(s, d, canonical), s, d, {}) for s in SEEDS for d in DATA_SEEDS]
    arms += [(tag(0, 0, p), 0, 0, variant(p)) for p in screen() if p != canonical]
    return arms


def analyze(phase, out):
    from nanochat.paths import base_dir

    def metric(t):
        with open(os.path.join(base_dir(), "ablation", f"{t}.json"), encoding="utf-8") as f:
            r = json.load(f)
        return statistics.fmean(r["bpb"][c] for c in CODE), r

    canonical = {f: 1 for f in FACTORS}
    noise = {(s, d): metric(tag(s, d, canonical))[0] for s in SEEDS for d in DATA_SEEDS}
    sigma = statistics.stdev(noise.values())
    if phase == "regime":
        lines = write_regime(metric, sigma)
    else:
        lines = write_screen(metric, noise, sigma)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_regime(metric, sigma):
    """The regime arms against the repeat of the canonical arm: code bpb at 8K, the repository corpora's at 32K."""
    arms = {name: metric(t) for name, (t, _, _, _) in zip(REGIME, runs("regime"))}
    base8, base = arms["canonical"]
    base32 = statistics.fmean(base["bpb32"].values())
    se = sigma * 2 ** 0.5
    lines = [f"### Regime: one-at-a-time departures (difference SE {se:.5f})", "",
             "| arm | variant | code bpb | diff | t | repo bpb at 32K | diff | steps | tok/s | clipped |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, (v, r) in arms.items():
        b32 = statistics.fmean(r["bpb32"].values())
        lines.append(f"| {name} | `{json.dumps(r['variant'])}` | {v:.5f} | {v - base8:+.5f} | {(v - base8) / se:+.2f} | "
                     f"{b32:.5f} | {b32 - base32:+.5f} | {r['steps']} | {r['tok_per_s']:,.0f} | {r['clipped']:.3f} |")
    return lines


def write_screen(metric, noise, sigma):
    """The noise table and the screen's main effects."""
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
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("runs", "analyze"))
    parser.add_argument("phase", choices=("screen", "regime"))
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.command == "runs":
        for t, s, d, v in runs(args.phase):
            print(t, s, d, json.dumps(v, separators=(",", ":")))
    else:
        analyze(args.phase, args.out)
