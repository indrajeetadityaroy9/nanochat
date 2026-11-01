"""
Figures of the base pretraining run's logged metrics for the paper: vector PDFs at ICML's column (3.25 in) and text
(6.75 in) widths, a Times-like serif at 8 pt, fonts embedded as TrueType (no Type 3); per-step traces are rasterized.

Inputs, all written by the run itself: metrics.jsonl (one `train` record per step, train/loop.py), diagnostics-step.npy
and diagnostics-update.npy with their column names in diagnostics.json (train/diagnostics.py), and the likelihood
trajectory results/pretrain/trajectory.json (eval/trajectory.py). Needs numpy and matplotlib (not in the training
image) and the repository on PYTHONPATH (the source groups come from corpus/mixture.py).

    PYTHONPATH=. python paper/figures/plot_training.py <run dir> [--trajectory results/pretrain/trajectory.json]
        [--out paper/figures] [--png <dir for 200-dpi previews>]
"""

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from corpus.mixture import FILE_LEVEL_STAGE, REPO_LEVEL_STAGE

BATCH = 1_474_560  # tokens per step (docs/training.md §2)
CLIP = 1.0  # train/optim.py MAX_GRAD_NORM
CAP = 15.0  # model/modeling.py FINAL_LOGIT_SOFTCAPPING
VOCAB = 32_768
COLUMN, TEXT = 3.25, 6.75  # ICML widths, inches
# Okabe-Ito, by mixture group (corpus/mixture.py); the 32K stage's repositories share the file stage's colour
GROUP_COLOR = {"repository": "#0072B2", "repositories": "#0072B2", "synthetic": "#E69F00", "notebooks": "#009E73",
               "functions": "#56B4E9", "text": "#D55E00", "math": "#CC79A7", "traces": "#000000"}
GROUP_LABEL = {"repository": "repository code", "synthetic": "synthetic code", "notebooks": "notebooks",
               "functions": "functions", "text": "web text", "math": "mathematics", "traces": "Codeforces traces"}
SCALARS = ("loss", "learning_rate", "grad_norm", "tokens_per_second", "mfu", "max_logsumexp", "mean_square_logsumexp",
           "qk_norm_gain_max", "gpu_power_watts", "gpu_sm_clock_mhz")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Times", "Nimbus Roman"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "lines.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "figure.dpi": 200,
})


def load(run: Path) -> dict:
    """Per-step arrays of the run's train records (a resumed attempt's record of a step supersedes the earlier one,
    as metrics.train_steps does) and its stage switch."""
    rows, sources, stage = {}, {}, None
    with open(run / "metrics.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] == "stage":
                stage = r["step"]
            if r["kind"] != "train":
                continue
            rows[r["step"]] = [math.nan if r.get(k) is None else r[k] for k in SCALARS]
            for s, v in (r.get("corpus_loss") or {}).items():
                sources.setdefault(s, {})[r["step"]] = v
    steps = max(rows) + 1
    assert len(rows) == steps, "missing train steps"
    a = np.array([rows[s] for s in range(steps)], dtype=np.float64)
    data = {k: a[:, i] for i, k in enumerate(SCALARS)}
    for s, v in sources.items():
        x = np.full(steps, np.nan)
        x[list(v)] = list(v.values())
        sources[s] = x
    names = json.loads((run / "diagnostics.json").read_text())
    data |= {"steps": steps, "stage": stage, "sources": sources,
             "step_cols": names["diagnostics-step.npy"], "update_cols": names["diagnostics-update.npy"],
             "update_every": names["update_stats_every"],
             "diag_step": np.load(run / "diagnostics-step.npy", mmap_mode="r"),
             "diag_update": np.load(run / "diagnostics-update.npy", mmap_mode="r")}
    return data


def tokens(step) -> np.ndarray:
    """Training tokens (billions) seen after `step`."""
    return (np.asarray(step) + 1) * BATCH / 1e9


def windows(y: np.ndarray, w: int, start: int, stop: int, reduce=np.nanmean):
    """Window statistics of y over [start, stop) in windows of w steps (a tail shorter than w/2 dropped): the windows'
    middle steps and values."""
    mids, vals = [], []
    for a in range(start, stop, w):
        b = min(a + w, stop)
        if b - a < w // 2:
            break
        seg = y[a:b]
        if np.isfinite(seg).any():
            mids.append((a + b - 1) / 2)
            vals.append(reduce(seg))
    return np.array(mids), np.array(vals)


def mark_stage(ax, d, label=True):
    """Shade the repository-level (32K) stage."""
    x0, x1 = tokens(d["stage"] - 1), tokens(d["steps"] - 1)
    ax.axvspan(x0, x1, color="0.92", lw=0, zorder=0)
    if label:
        ax.text(x1, 1.0, "32K", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=6.5, color="0.35")


def save(fig, name: str, out: Path, png: Path | None):
    fig.savefig(out / f"{name}.pdf")
    if png:
        fig.savefig(png / f"{name}.png", dpi=200)
    plt.close(fig)


def fig_loss(d, out, png):
    """Training loss (window means) and the learning-rate schedule."""
    fig, ax = plt.subplots(figsize=(COLUMN, 2.0))
    mark_stage(ax, d)
    for a, b, w in ((500, d["stage"], 500), (d["stage"], d["steps"], 250)):
        m, v = windows(d["loss"], w, a, b)
        ax.plot(tokens(m), v, color="k", lw=1.1)
    ax.set_ylim(0.84, 1.40)
    ax.set_xlim(0, tokens(d["steps"] - 1))
    ax.set_xlabel("training tokens (B)")
    ax.set_ylabel("training loss (nats)")
    lr = ax.twinx()
    lr.spines["right"].set_visible(True)
    lr.plot(tokens(np.arange(d["steps"])), d["learning_rate"] * 1e3, color="0.45", ls="--", lw=0.9)
    lr.set_ylim(0, 2.6)
    lr.set_ylabel(r"learning rate ($10^{-3}$)", color="0.35")
    lr.tick_params(axis="y", colors="0.35")
    save(fig, "fig_loss", out, png)


def source_groups() -> dict[str, tuple[str, str]]:
    """Source -> (stage, group), from corpus/mixture.py."""
    groups = {}
    for name, stage in (("8K", FILE_LEVEL_STAGE), ("32K", REPO_LEVEL_STAGE)):
        for group, g in stage.groups.items():
            for s in g.sources:
                groups[(name, s)] = group
    return groups


def style(source: str) -> str:
    return "--" if source.endswith("cpp") else "-"


def fig_sources(d, out, png):
    """Each source's training loss (log scale) and, in the file-level stage, its loss relative to steps 500-1,499."""
    groups = source_groups()
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT, 2.3), gridspec_kw={"wspace": 0.28})
    mark_stage(a, d)
    for (stage, s), group in sorted(groups.items()):
        if s not in d["sources"]:
            continue
        lo, hi = (500, d["stage"]) if stage == "8K" else (d["stage"], d["steps"])
        m, v = windows(d["sources"][s], 1000 if stage == "8K" else 250, lo, hi)
        if len(m) == 0:
            continue
        a.plot(tokens(m), v, color=GROUP_COLOR[group], ls=style(s), lw=0.8)
        if stage == "8K":
            b.plot(tokens(m), v / v[0], color=GROUP_COLOR[group], ls=style(s), lw=0.8)
    a.set_yscale("log")
    a.set_yticks([0.4, 0.6, 0.8, 1, 1.5, 2, 3])
    a.set_yticklabels(["0.4", "0.6", "0.8", "1", "1.5", "2", "3"])
    a.set_ylabel("training loss (nats, log scale)")
    b.set_ylabel("loss relative to steps 500–1,499")
    for ax in (a, b):
        ax.set_xlabel("training tokens (B)")
        ax.set_xlim(0, tokens(d["steps"] - 1) if ax is a else tokens(d["stage"] - 1))
    a.text(-0.16, 1.02, "(a)", transform=a.transAxes, fontweight="bold")
    b.text(-0.16, 1.02, "(b)", transform=b.transAxes, fontweight="bold")
    handles = [Line2D([], [], color=GROUP_COLOR[g], label=GROUP_LABEL[g])
               for g in ("repository", "synthetic", "notebooks", "functions", "text", "math", "traces")]
    handles += [Line2D([], [], color="0.3", ls="-", label="Python or other"),
                Line2D([], [], color="0.3", ls="--", label="C++")]
    b.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), handlelength=1.8)
    save(fig, "fig_sources", out, png)


def cols(d, key: str, pattern: str) -> list[int]:
    return [i for i, c in enumerate(d[key]) if re.fullmatch(pattern, c)]


def fig_stability(d, out, png):
    """Gradient norm, log-sum-exp of the capped logits, and each layer's largest QK-norm scale."""
    x = tokens(np.arange(d["steps"]))
    fig, axs = plt.subplots(1, 3, figsize=(TEXT, 1.9), gridspec_kw={"wspace": 0.42})
    a, b, c = axs
    a.plot(x, d["grad_norm"], color="0.72", lw=0.3, rasterized=True)
    m, v = windows(d["grad_norm"], 250, 0, d["steps"], np.nanmedian)
    a.plot(tokens(m), v, color="k", lw=0.9)
    a.axhline(CLIP, color="#D55E00", ls="--", lw=0.8)
    a.text(0.98, CLIP * 1.12, "clipping threshold", transform=a.get_yaxis_transform(), ha="right", fontsize=6.5,
           color="#D55E00")
    a.set_yscale("log")
    a.set_ylim(0.05, 5)
    a.set_ylabel("gradient norm")
    b.plot(x, d["max_logsumexp"], color="0.72", lw=0.3, rasterized=True, label="maximum")
    m, v = windows(np.sqrt(d["mean_square_logsumexp"]), 250, 0, d["steps"])
    b.plot(tokens(m), v, color="k", lw=0.9, label="root mean square")
    bound = CAP + math.log(VOCAB)
    b.axhline(bound, color="#D55E00", ls="--", lw=0.8)
    b.text(0.98, bound - 0.5, r"bound $c + \ln V$", transform=b.get_yaxis_transform(), ha="right", va="top",
           fontsize=6.5, color="#D55E00")
    b.set_ylim(0, 27)
    b.set_ylabel("log-sum-exp of the logits")
    b.legend(loc="lower right", handlelength=1.2)
    idx = d["step_cols"]
    sub = np.arange(0, d["steps"], 50)
    gains = np.asarray(d["diag_step"][sub])
    cmap = plt.get_cmap("viridis")
    layers = 24
    for layer in range(layers):
        q = idx.index(f"gain_max/layers.{layer}.self_attn.q_norm_weight")
        k = idx.index(f"gain_max/layers.{layer}.self_attn.k_norm_weight")
        c.plot(tokens(sub), np.maximum(gains[:, q], gains[:, k]), color=cmap(layer / (layers - 1)), lw=0.6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, layers))
    cb = fig.colorbar(sm, ax=c, pad=0.02, fraction=0.06, ticks=[1, 12, 24])
    cb.set_label("layer", labelpad=1)
    cb.outline.set_linewidth(0.4)
    c.set_ylabel("largest QK-norm scale")
    for ax, tag in zip(axs, "abc"):
        mark_stage(ax, d, label=False)
        ax.set_xlabel("training tokens (B)")
        ax.set_xlim(0, tokens(d["steps"] - 1))
        ax.text(-0.27, 1.02, f"({tag})", transform=ax.transAxes, fontweight="bold")
    save(fig, "fig_stability", out, png)


def fig_residual(d, out, png):
    """The per-layer residual scalars lambda_r and lambda_0 over training."""
    idx = d["step_cols"]
    sub = np.arange(0, d["steps"], 50)
    rows = np.asarray(d["diag_step"][sub])
    fig, axs = plt.subplots(1, 2, figsize=(TEXT, 2.0), gridspec_kw={"wspace": 0.42})
    extent = [tokens(sub[0]), tokens(sub[-1]), 0.5, 24.5]
    for ax, key, label, center, cmap in ((axs[0], "resid_lambdas", r"$\lambda_r$", 1.0, "PuOr_r"),
                                          (axs[1], "x0_lambdas", r"$\lambda_0$", 0.0, "RdBu_r")):
        z = rows[:, [idx.index(f"{key}/{i}") for i in range(24)]].T
        norm = TwoSlopeNorm(vcenter=center, vmin=min(z.min(), center - 1e-3), vmax=max(z.max(), center + 1e-3))
        im = ax.imshow(z, aspect="auto", origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest",
                       rasterized=True)
        cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.05)
        cb.set_label(label, labelpad=2)
        cb.outline.set_linewidth(0.4)
        ax.axvline(tokens(d["stage"] - 1), color="k", ls=":", lw=0.7)
        ax.set_yticks([1, 4, 8, 12, 16, 20, 24])
        ax.set_ylabel("layer")
        ax.set_xlabel("training tokens (B)")
    axs[0].text(-0.13, 1.02, "(a)", transform=axs[0].transAxes, fontweight="bold")
    axs[1].text(-0.13, 1.02, "(b)", transform=axs[1].transAxes, fontweight="bold")
    save(fig, "fig_residual", out, png)


# Parameter tensors by optimizer group and module (train/optim.py optimizers; model/modeling.py names)
MODULES = {
    "attention $q, k$": r"layers\.\d+\.self_attn\.[qk]_proj\.weight",
    "attention $v$": r"layers\.\d+\.self_attn\.v_proj\.weight",
    "attention output": r"layers\.\d+\.self_attn\.o_proj\.weight",
    "MLP input": r"layers\.\d+\.mlp\.up_proj\.weight",
    "MLP output": r"layers\.\d+\.mlp\.down_proj\.weight",
    "value-embedding gates": r"layers\.\d+\.self_attn\.ve_gate\.weight",
    "value-embedding tables": r"value_embeds\.\d+\.weight",
    "token embedding": r"embed_tokens\.weight",
    "output layer": r"lm_head\.weight",
    "vectors": r"resid_lambdas|x0_lambdas|layers\.\d+\.self_attn\.[qk]_norm_weight",
}
MODULE_COLOR = dict(zip(MODULES, ("#0072B2", "#56B4E9", "#009E73", "#E69F00", "#F0E442", "#CC79A7", "#882255",
                                  "#D55E00", "#000000", "0.6")))
OPTIMIZER = {"NorMuon: matrices": ("attention $q, k$", "attention $v$", "attention output", "MLP input", "MLP output",
                                   "value-embedding gates"),
             "AdamW: tables": ("value-embedding tables", "token embedding", "output layer"),
             "AdamW: vectors": ("vectors",)}


def fig_optimizer(d, out, png):
    """Parameter norms by module, update size over the learning rate by optimizer, and each module's share of the
    squared gradient norm."""
    every = d["update_every"]
    upd = np.asarray(d["diag_update"], dtype=np.float64)
    usteps = np.arange(len(upd)) * every
    fig, axs = plt.subplots(1, 3, figsize=(TEXT, 2.1), gridspec_kw={"wspace": 0.45})
    a, b, c = axs
    for module, pat in MODULES.items():
        idx = cols(d, "update_cols", rf"param_norm/(?:{pat})")
        a.plot(tokens(usteps), np.sqrt((upd[:, idx] ** 2).sum(1)), color=MODULE_COLOR[module], lw=0.8,
               label=module)
    a.set_yscale("log")
    a.set_ylim(5, 3000)
    a.set_ylabel("parameter norm")
    lr = d["learning_rate"][usteps]
    for (name, modules), color in zip(OPTIMIZER.items(), ("#0072B2", "#D55E00", "0.5")):
        pat = "|".join(MODULES[m] for m in modules)
        idx = cols(d, "update_cols", rf"update_rms/(?:{pat})")
        ratio = np.median(upd[:, idx], axis=1) / lr
        m, v = windows(ratio, 50, 10, len(ratio))
        b.plot(tokens(m * every), v, color=color, lw=0.9, label=name)
    b.set_ylim(0, 0.3)
    b.set_ylabel("update RMS / learning rate")
    b.legend(loc="lower left", handlelength=1.2)
    g = np.asarray(d["diag_step"], dtype=np.float64)
    total = (g[:, cols(d, "step_cols", r"grad_norm/.*")] ** 2).sum(1)
    shares = []
    for module, pat in MODULES.items():
        sq = (g[:, cols(d, "step_cols", rf"grad_norm/(?:{pat})")] ** 2).sum(1) / total
        m, v = windows(sq, 500, 100, d["steps"])
        shares.append(v)
    c.stackplot(tokens(m), np.array(shares), colors=[MODULE_COLOR[k] for k in MODULES], lw=0)
    c.set_ylim(0, 1)
    c.set_ylabel("share of squared gradient norm")
    for ax, tag in zip(axs, "abc"):
        mark_stage(ax, d, label=False)
        ax.set_xlabel("training tokens (B)")
        ax.set_xlim(0, tokens(d["steps"] - 1))
        ax.text(-0.28, 1.02, f"({tag})", transform=ax.transAxes, fontweight="bold")
    a.legend(loc="upper center", bbox_to_anchor=(1.85, -0.28), ncol=5, handlelength=1.4, columnspacing=1.0)
    save(fig, "fig_optimizer", out, png)


def fig_systems(d, out, png):
    """Throughput and MFU, and the GPU's power and SM clock (NVML, every 10th step)."""
    fig, axs = plt.subplots(1, 2, figsize=(TEXT, 1.8), gridspec_kw={"wspace": 0.5})
    a, b = axs
    for lo, hi in ((0, d["stage"]), (d["stage"], d["steps"])):
        m, v = windows(d["tokens_per_second"], 250, lo, hi, np.nanmedian)
        a.plot(tokens(m), v / 1e3, color="k", lw=0.9)
    a.set_ylabel("throughput (K tokens/s)")
    mfu = a.twinx()
    mfu.spines["right"].set_visible(True)
    for lo, hi in ((0, d["stage"]), (d["stage"], d["steps"])):
        m, v = windows(d["mfu"], 250, lo, hi, np.nanmedian)
        mfu.plot(tokens(m), v, color="#0072B2", lw=0.9, ls="--")
    mfu.set_ylim(0.45, 0.55)
    mfu.set_ylabel("MFU", color="#0072B2")
    mfu.tick_params(axis="y", colors="#0072B2")
    a.set_ylim(250, 400)
    m, v = windows(d["gpu_power_watts"], 250, 0, d["steps"])
    b.plot(tokens(m), v, color="k", lw=0.9)
    b.axhline(700, color="#D55E00", ls="--", lw=0.8)
    b.text(0.02, 703, "700 W limit", transform=b.get_yaxis_transform(), fontsize=6.5, color="#D55E00", va="bottom")
    b.set_ylim(550, 760)
    b.set_ylabel("GPU power (W)")
    clock = b.twinx()
    clock.spines["right"].set_visible(True)
    m, v = windows(d["gpu_sm_clock_mhz"], 250, 0, d["steps"])
    clock.plot(tokens(m), v, color="#0072B2", lw=0.9, ls="--")
    clock.set_ylabel("SM clock (MHz)", color="#0072B2")
    clock.tick_params(axis="y", colors="#0072B2")
    clock.set_ylim(1200, 1700)
    for ax, tag in zip(axs, "ab"):
        mark_stage(ax, d, label=False)
        ax.set_xlabel("training tokens (B)")
        ax.set_xlim(0, tokens(d["steps"] - 1))
        ax.text(-0.2, 1.02, f"({tag})", transform=ax.transAxes, fontweight="bold")
    save(fig, "fig_systems", out, png)


def fig_likelihood(trajectory: dict, d, out, png):
    """Validation bits per byte by domain at the late checkpoints, as a change from the reference checkpoint."""
    groups = (("8k", ("repository", "synthetic", "notebooks", "text", "math", "all"), "8K context"),
              ("32k", ("repositories", "traces", "text", "math", "all"), "32K context"),
              ("solution", ("humaneval_plus", "mbpp_plus", "humaneval_x_cpp"), "benchmark solutions"))
    color = GROUP_COLOR | {"all": "0.4", "humaneval_plus": "#0072B2", "mbpp_plus": "#009E73",
                           "humaneval_x_cpp": "#E69F00"}
    label = GROUP_LABEL | {"repositories": "repositories", "all": "all", "humaneval_plus": "HumanEval+",
                           "mbpp_plus": "MBPP+", "humaneval_x_cpp": "HumanEval-X C++"}
    fig, axs = plt.subplots(1, 3, figsize=(TEXT, 2.5), gridspec_kw={"wspace": 0.42})
    for ax, (ctx, domains, title), tag in zip(axs, groups, "abc"):
        steps = [s for s in trajectory["steps"] if ctx in trajectory["change"][str(s)]]
        ref = trajectory["reference"][ctx]
        for dom in domains:
            ch = [trajectory["change"][str(s)][ctx][dom] for s in steps]
            y = np.array([c["change"] for c in ch]) * 1e3
            lo = y - np.array([c["interval"][0] for c in ch]) * 1e3
            hi = np.array([c["interval"][1] for c in ch]) * 1e3 - y
            ax.errorbar(tokens(steps), y, yerr=[lo, hi], color=color[dom], lw=0.9, marker="o", ms=2, capsize=1.2,
                        elinewidth=0.6, label=label[dom])
        ax.axhline(0, color="0.6", lw=0.5)
        ax.axvline(tokens(d["stage"] - 1), color="k", ls=":", lw=0.7)
        ax.set_title(f"{title}, against step {ref:,}", fontsize=7.5)
        ax.set_xlabel("training tokens (B)")
        ax.text(-0.27, 1.06, f"({tag})", transform=ax.transAxes, fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, handlelength=1.2, fontsize=6,
                  columnspacing=0.8)
    axs[0].set_ylabel(r"change in bits per byte ($10^{-3}$)")
    save(fig, "fig_likelihood", out, png)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", type=Path)
    p.add_argument("--trajectory", type=Path, default=Path("results/pretrain/trajectory.json"))
    p.add_argument("--out", type=Path, default=Path("paper/figures"))
    p.add_argument("--png", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.png:
        args.png.mkdir(parents=True, exist_ok=True)
    d = load(args.run)
    fig_loss(d, args.out, args.png)
    fig_sources(d, args.out, args.png)
    fig_stability(d, args.out, args.png)
    fig_residual(d, args.out, args.png)
    fig_optimizer(d, args.out, args.png)
    fig_systems(d, args.out, args.png)
    fig_likelihood(json.loads(args.trajectory.read_text()), d, args.out, args.png)


if __name__ == "__main__":
    main()
