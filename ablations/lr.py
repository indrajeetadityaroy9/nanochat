"""
Peak lr. Production's peak lr is DeepSeek LLM's law eta = 0.3118 C^-0.125 (arXiv 2401.02954 §3.1 Eq. 1;
train/schedule.py plan), fit for AdamW under a multi-step schedule (arXiv 2401.02954 §2.3) at compute below
production's; production applies it to Muon (matched to AdamW's update RMS, train/optim.py) as the peak of linear decay
to zero, and at a fixed model it falls with tokens as D^-0.125.

The production lr is measured, not swept at production's length: its optimum at d24 (production's model, so no width
or depth transfer is assumed; its parameterisation is standard, not muP) at SHORT tokens per parameter, times the
optimum's horizon trend measured at d8 between SHORT and LONG, assumed width-independent ([inference]: arXiv 2409.19913
§4 finds its horizon exponent beta size-independent only for models of at least 760M parameters, larger for smaller
ones). The published rules disagree on the trend's direction: Complete^(d)P's AdamW SDE rule eta ∝ sqrt(m_B / m_D), for
batch ratio m_B and token ratio m_D (arXiv 2512.22382 Table 1, §2.3), lowers the lr with the horizon, as arXiv
2409.19913 §3.2 Eq. 1 and the law do; Step-Law eta = 1.79 N^-0.713 D^0.307, fit over AdamW models (arXiv 2503.04715 §1
Eq. 1), raises it. Alternatives: the law's lr times 2 and 4, and 8 at d8 at LONG, where Step-Law points; one seed per
setting.

Analysis, pre-registered, written to RESULTS/lr.json's "analysis":
- per setting, code bits per byte fitted as a parabola in u = ln(multiplier) (fit): its vertex u* or the grid extension
  the rule calls for;
- the measured d8 ratio r = exp(u*_d8@LONG - u*_d8@SHORT) against the SDE rule's and Step-Law's predictions, each over
  the law's own fall between the two plans;
- the horizon exponent ln r / ln m_D,d8;
- production's multiplier of the law's lr, exp(u*_d24@SHORT) * (r / sqrt(m_B,d8))^(ln m_D,prod / ln m_D,d8)
  * sqrt(m_B,prod): d8's trend net of its batch growth by the square-root batch rule (arXiv 2205.10287 Def. 5.2, its lr
  factor; arXiv 2512.22382 Table 1), its exponent (horizon_exponent_net) carried to production's horizon range (every
  rule in play is a power law in D), and production's batch growth by the same rule.
m_D,prod is production's tokens over d24 at SHORT's, m_D,d8 d8 at LONG's over d8 at SHORT's. Batches, token counts and
the law's lrs are the runs' recorded plans (production's: harness.production_plan at max_ratio of d24).

    python -m ablations.lr    (ablations/harness.py)
"""
import json
import math

import numpy as np

from ablations.harness import Setting, main, production_plan
from nanochat.data.tokenizer import get_tokenizer
from nanochat.paths import RESULTS
from nanochat.train.stages import max_ratio

# The short horizon in tokens per scaling parameter: the longest d24 run a sweep affords
SHORT = 5
# d8's long horizon: the shortest at which d8's law batch grows to 2 micro-batches (schedule.plan's rounding), so the
# trend is measured under batch co-scaling as production trains, its batch growth netted out by the square-root rule
LONG = 200
# Step-Law's token exponent of the optimal lr (arXiv 2503.04715 §1 Eq. 1: eta = 1.79 N^-0.713 D^0.307)
STEP_LAW_D = 0.307


def setting(lines: list[dict], depth: int, ratio: float) -> list[dict]:
    """The records of a setting's runs."""
    return [line for line in lines if (line["setting"]["depth"], line["setting"]["tokens_per_param"]) == (depth, ratio)]


def fit(lines: list[dict]) -> dict:
    """A setting's optimum from its runs in lr order, a diverged run counting as worse than any evaluated one: the grid
    extension by a factor of 2 ("down" or "up") while its lowest-code run is at an end of the grid; else the vertex (as
    a multiplier) of the parabola in u = ln(multiplier) through the lowest-code run and its two neighbours (positive
    curvature by construction; a local bracket, as arXiv 2409.19913 §3.1 fits quadratics in log lr), or the lowest-code
    run itself when the neighbour above it diverged."""
    runs = sorted(lines, key=lambda line: line["fields"]["lr_scale"])
    if len({line["fields"]["lr_scale"] for line in runs}) != len(runs):
        raise RuntimeError("a setting holds two runs at one lr_scale")
    u = [math.log(line["fields"]["lr_scale"]) for line in runs]
    code = [math.inf if line["measures"] is None else line["measures"]["code"] for line in runs]
    best = int(np.argmin(code))
    evaluated = sum(math.isfinite(c) for c in code)
    if best == 0 or best == len(runs) - 1:
        return {"runs": evaluated, "curvature": None, "vertex": None, "extend": "down" if best == 0 else "up"}
    if not math.isfinite(code[best + 1]):
        return {"runs": evaluated, "curvature": None, "vertex": math.exp(u[best]), "extend": None}
    if not math.isfinite(code[best - 1]):
        raise RuntimeError("the lowest-code run's lower neighbour diverged: extend the grid down by hand")
    a, b, _ = np.polyfit(u[best - 1:best + 2], code[best - 1:best + 2], 2)
    return {"runs": evaluated, "curvature": float(a), "vertex": math.exp(-b / (2 * a)), "extend": None}


def analysis(lines: list[dict]) -> dict:
    """The pre-registered estimator over RESULTS/lr.jsonl's records (module docstring); the derived values are None
    while any setting's grid needs extending."""
    def recorded_plan(depth, ratio):
        """The setting's plan at the law's lr (every arm's batch and steps are the same)."""
        line = setting(lines, depth, ratio)[0]
        recorded = next(r for r in line["records"] if r["kind"] == "run")["plan"]
        return recorded | {"lr": recorded["lr"] / line["fields"]["lr_scale"]}

    def tokens(p):
        return p["num_iterations"] * p["total_batch_size"]

    fits = {f"d{depth} at {ratio:g}": fit(setting(lines, depth, ratio))
            for depth, ratio in ((8, SHORT), (8, LONG), (24, SHORT))}
    d8_short, d8_long, d24_short = recorded_plan(8, SHORT), recorded_plan(8, LONG), recorded_plan(24, SHORT)
    production = production_plan(depth=24, ratio=max_ratio(get_tokenizer(), depth=24))
    m_b_production = production.total_batch_size / d24_short["total_batch_size"]
    m_b_d8 = d8_long["total_batch_size"] / d8_short["total_batch_size"]
    m_d_production = production.num_iterations * production.total_batch_size / tokens(d24_short)
    m_d_d8 = tokens(d8_long) / tokens(d8_short)
    law_fall_d8 = d8_long["lr"] / d8_short["lr"]
    out = {"settings": fits, "m_b_production": m_b_production, "m_b_d8": m_b_d8, "m_d_production": m_d_production,
           "m_d_d8": m_d_d8, "law_fall_d8": law_fall_d8,
           "predicted_d8_ratio": {"sde": math.sqrt(m_b_d8 / m_d_d8) / law_fall_d8,
                                  "step_law": m_d_d8 ** STEP_LAW_D / law_fall_d8},
           "measured_d8_ratio": None, "horizon_exponent": None, "horizon_exponent_net": None,
           "production_multiplier": None, "production_lr": None}
    if all(f["extend"] is None for f in fits.values()):
        short, long, d24 = (fits[k]["vertex"] for k in fits)
        ratio = long / short
        out["measured_d8_ratio"] = ratio
        out["horizon_exponent"] = math.log(ratio) / math.log(m_d_d8)
        out["horizon_exponent_net"] = math.log(ratio / math.sqrt(m_b_d8)) / math.log(m_d_d8)
        out["production_multiplier"] = (d24 * m_d_production ** out["horizon_exponent_net"]
                                        * math.sqrt(m_b_production))
        out["production_lr"] = out["production_multiplier"] * production.lr
    return out


if __name__ == "__main__":
    # multipliers of the law's lr per (depth, tokens per parameter), the cheap settings first; a grid the rule extends
    # (fit) gains the next factor of 2 and the experiment runs again (harness.main skips evaluated runs)
    grids = {(8, SHORT): [1, 2, 4], (8, LONG): [1, 2, 4, 8], (24, SHORT): [1, 2, 4]}
    while True:
        main(experiment="lr", settings=[Setting(depth=depth, ratio=ratio, seeds=range(1),
                                                alternatives={f"lr x{s:g}": {"lr_scale": float(s)}
                                                              for s in scales if s != 1})
                                        for (depth, ratio), scales in grids.items()], production={})
        lines = [json.loads(line) for line in (RESULTS / "lr.jsonl").read_text(encoding="utf-8").splitlines()]
        extend = {(depth, ratio): fit(setting(lines, depth, ratio))["extend"] for depth, ratio in grids}
        if not any(extend.values()):
            break
        for key, direction in extend.items():
            if direction is not None:
                grids[key].append(min(grids[key]) / 2 if direction == "down" else max(grids[key]) * 2)
    summary = json.loads((RESULTS / "lr.json").read_text(encoding="utf-8"))
    (RESULTS / "lr.json").write_text(json.dumps(summary | {"analysis": analysis(lines)}, allow_nan=False),
                                     encoding="utf-8")
