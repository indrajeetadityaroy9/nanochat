"""
The recipe ablations (ACTIONS.md §8), each a knob of run.py RunConfig at its default in the baseline, the
tokenizer ablation's d16 or d8 recipe, and their pre-registered analyses (conf/experiment/<name>.yaml's analysis):

- softcap (D5, conf/experiment/softcap.yaml): the output logits capped at c = 15 (karpathy/nanochat 92d63d4
  gpt.py:511-515; modded-nanogpt record #18 "lowers that to 15") against none, d16 at Chinchilla's 20 tokens per
  parameter (arXiv 2203.15556 Table 3) on the sa-bpe arm's data, 4 paired seeds, at lr_scale 2.7972 (the base model's
  multiplier, ablations/lr.py analysis, as the cap's published effect is a high-lr one). Adopt c = 15 iff code is
  better and parsing is not worse (contrast); else no cap ("equal" keeps: the cap costs the loss, eval and decoder
  paths).
- adamw_lr (D6, conf/experiment/adamw_lr.yaml): the AdamW groups' lr x 1/4 and x 4 at their decay timescale (their
  weight decay / scale), Muon's untouched, d16 at the sa-bpe arm's max_ratio, 2 seeds; the baseline is the tokenizer
  ablation's sa-bpe runs (the same keys). k*: the vertex of the exact 3-point quadratic in ln k through the arms'
  mean code bits per byte over {1/4, 1, 4} (ablations/lr.py's estimator), clamped to [1/4, 4]; adopt
  adamw_lr_scale = k* iff the better of the two arms is better on code than the baseline and not worse on parsing,
  else keep one lr (scaled).
- muon_tau (D7, conf/experiment/muon_tau_20.yaml then muon_tau_200.yaml): Muon's decay timescale x 1/2 and x 2 (its
  weight decay x 2 and x 1/2) under the tau law (train/optim.py tau_weight_decay: Power Lines, arXiv 2505.13738, fit
  with AdamW; its App. B extends the EMA view to decoupled-decay optimizers), AdamW's untouched, d8 at 20 tokens per
  parameter on the sa-bpe arm's data (2 seeds) and at 200 on the corpus's (2 seeds; beyond the arm's set). Per setting
  m: the vertex of the exact 3-point quadratic in ln m over {1/2, 1, 2}, clamped to [1/2, 2], or 1 unless the better
  arm is better on code than the baseline and not worse on parsing (scaled). If m_20 = m_200 = 1 the law stays for
  Muon; else Muon's own law through both points, tau* = c TPP^e with c = TAU_C m_20 20^-D, e = TAU_E + D,
  D = ln(m_200 / m_20) / ln 10 (muon_tau_law; TAU_C, TAU_E the law's, train/optim.py).

contrast: the tokenizer ablation's tests (ablations/tokenizer.py contrast) of an arm against the baseline on the seeds
where both have an evaluation: code bits per byte better or worse beyond 1.96 CODE_SIGMA sqrt(2 / n), n the seeds per
arm; MBPP+ parsing by the exact sign test over the problems whose parsing seeds differ, at ALPHA.
"""
import json
import math
import statistics

import numpy as np

from ablations.tokenizer import contrast, evaluated
from paths import RESULTS
from train.optim import TAU_C, TAU_E


def adopted(c: dict) -> bool:
    return c["code"] == "better" and c["parsing"] != "worse"


def softcap(lines: list[dict]) -> dict:
    """D5's decision (module docstring)."""
    c = contrast(evaluated(lines, "softcap"), 15.0, None)
    return {"contrast": c, "decision": {"softcap": 15.0 if adopted(c) else None}}


def scaled(lines: list[dict], field: str, low: float, high: float) -> dict:
    """A scale's decision over the arms {low, 1, high} (module docstring: D6, D7): the vertex of the exact quadratic in
    ln(scale) through the arms' mean code bits per byte, clamped to [low, high] (a concave or flat fit: the lower-code
    end), adopted iff the lower-code arm's contrast with the baseline (1) is better on code and not worse on parsing;
    else 1."""
    runs = evaluated(lines, field)
    scales = (low, 1.0, high)
    means = {s: statistics.fmean(code for (v, _), (code, _) in runs.items() if v == s) for s in scales}
    a, b, _ = np.polyfit([math.log(s) for s in scales], [means[s] for s in scales], 2)
    end = min((low, high), key=means.__getitem__)
    vertex = min(max(math.exp(-b / (2 * a)), low), high) if a > 0 else end
    contrasts = {str(s): contrast(runs, s, 1.0) for s in (low, high)}
    return {"means": {str(s): m for s, m in means.items()}, "curvature": float(a), "vertex": vertex,
            "contrasts": contrasts, "scale": vertex if adopted(contrasts[str(end)]) else 1.0}


def adamw_lr(lines: list[dict]) -> dict:
    """D6's decision: "scale" the adopted adamw_lr_scale."""
    return scaled(lines, "adamw_lr_scale", 0.25, 4.0)


def muon_tau(lines: list[dict]) -> dict:
    """D7's m at one setting: "scale"."""
    return scaled(lines, "muon_tau_scale", 0.5, 2.0)


def muon_tau_law(lines: list[dict]) -> dict:
    """D7's m at 200 tokens per parameter and the decision with m at 20 (RESULTS/muon_tau_20.json, run first): keep
    the tau law for Muon, or Muon's own constants (module docstring)."""
    m_200 = muon_tau(lines)
    m_20 = json.loads((RESULTS / "muon_tau_20.json").read_text(encoding="utf-8"))["analysis"]["scale"]
    if m_20 == m_200["scale"] == 1.0:
        return m_200 | {"decision": "law"}
    d = math.log(m_200["scale"] / m_20) / math.log(200 / 20)
    return m_200 | {"decision": {"c_tau": TAU_C * m_20 * 20 ** -d, "m_tau": TAU_E + d}}
