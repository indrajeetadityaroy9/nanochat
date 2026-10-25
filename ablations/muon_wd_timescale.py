"""
The muon_wd_timescale ablation (conf/experiment/muon_wd_timescale_20tpp.yaml, then muon_wd_timescale_200tpp.yaml):
Muon's weight-decay EMA timescale x 1/2 and x 2 (its weight decay x 2 and x 1/2) under the tau law (train/optim.py
tau_weight_decay: Power Lines, arXiv 2505.13738, fit with AdamW; its App. B extends the EMA view to decoupled-decay
optimizers), AdamW's untouched, d8 at 20 tokens per parameter on the sa-bpe arm's data (2 seeds) and at 200 on the
corpus tokenizer's (2 seeds). Its pre-registered analysis: per setting, m the scale sweep over {1/2, 1, 2}
(ablations/contrast.py scaled; analysis); if m_20 = m_200 = 1 the law stays for Muon, else Muon's own law through both
points, tau* = c TPP^e with c = TAU_C m_20 20^-D, e = TAU_E + D, D = ln(m_200 / m_20) / ln 10 (law; TAU_C, TAU_E the
law's, train/optim.py).
"""
import json
import math

from ablations.contrast import scaled
from paths import RESULTS
from train.optim import TAU_C, TAU_E


def analysis(lines: list[dict]) -> dict:
    """Muon's timescale multiplier m at one setting (muon_wd_timescale_20tpp): "scale"."""
    return scaled(lines, "muon_tau_scale", 0.5, 2.0)


def law(lines: list[dict]) -> dict:
    """m at 200 tokens per parameter (muon_wd_timescale_200tpp) and the decision with m at 20
    (RESULTS/muon_wd_timescale_20tpp.json): keep the tau law for Muon, or Muon's own constants (module docstring)."""
    m_200 = analysis(lines)
    m_20 = json.loads((RESULTS / "muon_wd_timescale_20tpp.json").read_text(encoding="utf-8"))["analysis"]["scale"]
    if m_20 == m_200["scale"] == 1.0:
        return m_200 | {"decision": "law"}
    d = math.log(m_200["scale"] / m_20) / math.log(200 / 20)
    return m_200 | {"decision": {"c_tau": TAU_C * m_20 * 20 ** -d, "m_tau": TAU_E + d}}
