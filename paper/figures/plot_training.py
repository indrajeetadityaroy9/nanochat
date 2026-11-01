"""
The paper's figures of the base pretraining run: vector PDFs at ICML's column (3.25 in) and text (6.75 in) widths, a
Times-metric serif at 8 pt, fonts embedded as TrueType (no Type 3), per-step traces rasterized, no timestamp.

- fig_loss: training loss (window means) and the learning-rate schedule;
- fig_stability: gradient norm with the clipping threshold, log-sum-exp of the capped logits with its bound c + ln V,
  and each layer's largest QK-norm scale;
- fig_residual: the residual scalars lambda_r and lambda_0 of every layer over training.

Inputs, written by the run itself: metrics.jsonl (one `train` record per step, train/loop.py) and
diagnostics-step.npy with its column names in diagnostics.json (train/diagnostics.py). Needs numpy and matplotlib
(not in the training image).

    python paper/figures/plot_training.py <run dir> [--out paper/figures] [--png <dir for 200-dpi previews>]
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

BATCH = 1_474_560  # tokens per step (docs/training.md §2)
CLIP = 1.0  # train/optim.py MAX_GRAD_NORM
CAP = 15.0  # model/modeling.py FINAL_LOGIT_SOFTCAPPING
VOCAB = 32_768
LAYERS = 24
COLUMN, TEXT = 3.25, 6.75  # ICML widths, inches
ACCENT = "#D55E00"  # Okabe-Ito vermilion: reference lines
SCALARS = ("loss", "learning_rate", "grad_norm", "max_logsumexp", "mean_square_logsumexp")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Times", "Nimbus Roman"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 200,
})


def load(run: Path) -> dict:
    """Per-step arrays of the run's train records (a resumed attempt's record of a step supersedes the earlier one,
    as metrics.train_steps does), its stage switch and the per-step diagnostics."""
    rows, stage = {}, None
    with open(run / "metrics.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] == "stage":
                stage = r["step"]
            elif r["kind"] == "train":
                rows[r["step"]] = [math.nan if r.get(k) is None else r[k] for k in SCALARS]
    steps = max(rows) + 1
    assert len(rows) == steps, "missing train steps"
    a = np.array([rows[s] for s in range(steps)], dtype=np.float64)
    names = json.loads((run / "diagnostics.json").read_text())
    return {k: a[:, i] for i, k in enumerate(SCALARS)} | {
        "steps": steps, "stage": stage, "columns": names["diagnostics-step.npy"],
        "diagnostics": np.load(run / "diagnostics-step.npy", mmap_mode="r")}


def tokens(step) -> np.ndarray:
    """Training tokens (billions) seen after `step`."""
    return (np.asarray(step) + 1) * BATCH / 1e9


def windows(y: np.ndarray, w: int, start: int, stop: int, reduce=np.nanmean):
    """Statistics of y over [start, stop) in windows of w steps (a tail shorter than w/2 dropped): the windows'
    middle steps and values."""
    mids, vals = [], []
    for a in range(start, stop, w):
        b = min(a + w, stop)
        if b - a < w // 2:
            break
        mids.append((a + b - 1) / 2)
        vals.append(reduce(y[a:b]))
    return np.array(mids), np.array(vals)


def mark_stage(ax, d, label=True):
    """Shade the repository-level (32K) stage."""
    x0, x1 = tokens(d["stage"] - 1), tokens(d["steps"] - 1)
    ax.axvspan(x0, x1, color="0.92", lw=0, zorder=0)
    if label:
        ax.text(x1, 1.0, "32K", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=6.5, color="0.35")


def tag(ax, letter: str, x: float):
    ax.text(x, 1.02, f"({letter})", transform=ax.transAxes, fontweight="bold")


def save(fig, name: str, out: Path, png: Path | None):
    fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None})  # no timestamp: identical bytes on a rerun
    if png:
        fig.savefig(png / f"{name}.png", dpi=200)
    plt.close(fig)


def fig_loss(d, out, png):
    """Training loss (means over 500 steps, 250 in the 32K stage, from step 500) and the learning rate."""
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


def fig_stability(d, out, png):
    """Gradient norm, log-sum-exp of the capped logits, and each layer's largest QK-norm scale."""
    x = tokens(np.arange(d["steps"]))
    fig, axs = plt.subplots(1, 3, figsize=(TEXT, 1.9), gridspec_kw={"wspace": 0.42})
    a, b, c = axs
    a.plot(x, d["grad_norm"], color="0.72", lw=0.3, rasterized=True)
    m, v = windows(d["grad_norm"], 250, 0, d["steps"], np.nanmedian)
    a.plot(tokens(m), v, color="k", lw=0.9)
    a.axhline(CLIP, color=ACCENT, ls="--", lw=0.8)
    a.text(0.98, CLIP * 1.12, "clipping threshold", transform=a.get_yaxis_transform(), ha="right", fontsize=6.5,
           color=ACCENT)
    a.set_yscale("log")
    a.set_ylim(0.05, 5)
    a.set_ylabel("gradient norm")
    b.plot(x, d["max_logsumexp"], color="0.72", lw=0.3, rasterized=True, label="maximum")
    m, v = windows(np.sqrt(d["mean_square_logsumexp"]), 250, 0, d["steps"])
    b.plot(tokens(m), v, color="k", lw=0.9, label="root mean square")
    bound = CAP + math.log(VOCAB)
    b.axhline(bound, color=ACCENT, ls="--", lw=0.8)
    b.text(0.98, bound - 0.5, r"bound $c + \ln V$", transform=b.get_yaxis_transform(), ha="right", va="top",
           fontsize=6.5, color=ACCENT)
    b.set_ylim(0, 27)
    b.set_ylabel("log-sum-exp of the logits")
    b.legend(loc="lower right", handlelength=1.2)
    sub = np.arange(0, d["steps"], 50)
    rows = np.asarray(d["diagnostics"][sub])
    cmap = plt.get_cmap("viridis")
    for layer in range(LAYERS):
        q = d["columns"].index(f"gain_max/layers.{layer}.self_attn.q_norm_weight")
        k = d["columns"].index(f"gain_max/layers.{layer}.self_attn.k_norm_weight")
        c.plot(tokens(sub), np.maximum(rows[:, q], rows[:, k]), color=cmap(layer / (LAYERS - 1)), lw=0.6)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, LAYERS)), ax=c, pad=0.02,
                      fraction=0.06, ticks=[1, 12, 24])
    cb.set_label("layer", labelpad=1)
    cb.outline.set_linewidth(0.4)
    c.set_ylabel("largest QK-norm scale")
    for ax, letter in zip(axs, "abc"):
        mark_stage(ax, d, label=False)
        ax.set_xlabel("training tokens (B)")
        ax.set_xlim(0, tokens(d["steps"] - 1))
        tag(ax, letter, -0.27)
    save(fig, "fig_stability", out, png)


def fig_residual(d, out, png):
    """The residual scalars lambda_r and lambda_0 of every layer over training (every 50th step)."""
    sub = np.arange(0, d["steps"], 50)
    rows = np.asarray(d["diagnostics"][sub])
    fig, axs = plt.subplots(1, 2, figsize=(TEXT, 2.0), gridspec_kw={"wspace": 0.42})
    extent = [tokens(sub[0]), tokens(sub[-1]), 0.5, LAYERS + 0.5]
    for ax, key, label, center, cmap, letter in ((axs[0], "resid_lambdas", r"$\lambda_r$", 1.0, "PuOr_r", "a"),
                                                  (axs[1], "x0_lambdas", r"$\lambda_0$", 0.0, "RdBu_r", "b")):
        z = rows[:, [d["columns"].index(f"{key}/{i}") for i in range(LAYERS)]].T
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
        tag(ax, letter, -0.13)
    save(fig, "fig_residual", out, png)


def main():
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, default=Path("paper/figures"))
    p.add_argument("--png", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.png:
        args.png.mkdir(parents=True, exist_ok=True)
    d = load(args.run)
    fig_loss(d, args.out, args.png)
    fig_stability(d, args.out, args.png)
    fig_residual(d, args.out, args.png)


if __name__ == "__main__":
    main()
