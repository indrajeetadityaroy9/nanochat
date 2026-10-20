"""
Peak lr. Production: DeepSeek LLM's law eta = 0.3118 C^-0.125 (arXiv 2401.02954 §3.1; train/schedule.py plan) as the
peak lr. The law was fit for AdamW over C = 1e17 to 2e19 FLOPs under a multi-step schedule; production applies it to
Muon (matched to AdamW's update RMS, train/optim.py) as the peak of linear decay to zero at C = 1.5e20, and at a fixed
model it falls with tokens as D^-0.125. At d8 at 20 tokens per parameter the law's lr times 2 lowers code bits per byte
by 0.018 on both of 2 seeds (ablations.md, Records): the law is low at short horizons.

The production lr is measured, not swept at production's length: its optimum at production's width and depth at a short
horizon, d24 at 5 tokens per parameter (production's model, so no width or depth transfer is assumed: its
parameterisation is standard, not muP), times the optimum's horizon trend measured at d8 between 5 and LONG tokens per
parameter, assumed width-independent (Bjorck et al.'s size-independent horizon exponent, arXiv 2409.19913). The
published transfer rules disagree on that trend's direction: Complete^(d)P's AdamW SDE rule, eta ∝ sqrt(m_B / m_D) for
batch ratio m_B and token ratio m_D (Mlodozeniec et al., arXiv 2512.22382, Table 1), lowers the lr with the horizon, as
Bjorck et al. (D^-0.32) and the law do; Step-Law (Li et al., arXiv 2503.04715: eta = 1.79 N^-0.713 D^0.307, 3,700
AdamW models) raises it. Alternatives: the law's lr times 2 and 4, and 8 at d8 at LONG, where Step-Law points; one seed
per setting.

Analysis (ablations.md, Analysis), written to RESULTS/lr.json's "analysis": per setting, code bits per byte fitted as a
quadratic in u = ln(multiplier) over its evaluated runs (numpy polyfit), its vertex u* and the grid extension the
pre-registered rule calls for; the measured d8 ratio r = exp(u*_d8@LONG - u*_d8@5) against the SDE rule's and
Step-Law's predictions (each over the law's own fall between the two plans); the horizon exponent ln r / ln m_D,d8;
production's multiplier of the law's lr, exp(u*_d24@5) * r^(ln m_D,prod / ln m_D,d8) * sqrt(m_B,prod / m_B,d8): the
exponent carried to production's horizon range (every rule in play is a power law in D: the SDE rule, Bjorck et al.,
Step-Law, the law), and d8's smaller batch growth corrected by the square-root batch rule (Malladi et al., arXiv
2205.10287; Complete^(d)P's sqrt(m_B)). m_D,prod is production's tokens over d24 at 5's, m_D,d8 d8 at LONG's over d8
at 5's. Batches, token counts and the law's lrs are the runs' recorded plans (production's: harness.plan_of at
max_ratio of d24).

    python -m ablations.lr    (ablations/harness.py; ablations.md, Analysis)
"""
import json
import math

import numpy as np

from ablations.harness import Run, main, plan_of
from nanochat.data.tokenizer import get_tokenizer
from nanochat.paths import RESULTS
from nanochat.train.stages import max_ratio

# The short horizon in tokens per scaling parameter: the longest d24 run a sweep affords (ablations.md, Settings)
SHORT = 5
# d8's long horizon: the shortest at which d8's law batch has grown as production's batch grows in kind, to 2
# micro-batches (B_opt 1.52 micro-batches at 200, 2.04 at production's 490.8; schedule.plan rounds both to 2), so the
# batch correction is the same sqrt(m_B,prod / m_B,d8) as at 490.8 at less than half its cost; m_D,d8 about 40
LONG = 200
# Step-Law's token exponent of the optimal lr (Li et al., arXiv 2503.04715, eta = 1.79 N^-0.713 D^0.307)
STEP_LAW_D = 0.307


def fit(lines: list[dict]) -> dict:
    """A setting's quadratic of code in u = ln(multiplier) over its evaluated runs: curvature, vertex (as a multiplier;
    None without a minimum) and the grid extension by a factor of 2 the rule calls for ("down", "up" or None): toward
    the vertex when it lies outside the grid, toward the grid's lower-code end when the curvature is not positive, and
    down when divergences leave fewer than the fit's 3 runs (the high end diverged)."""
    runs = sorted((line for line in lines if line["measures"] is not None), key=lambda line: line["fields"]["lr_scale"])
    if len(runs) < 3:
        return {"runs": len(runs), "curvature": None, "vertex": None, "extend": "down"}
    u = [math.log(line["fields"]["lr_scale"]) for line in runs]
    code = [line["measures"]["code"] for line in runs]
    a, b, _ = np.polyfit(u, code, 2)
    vertex = -b / (2 * a)
    if a <= 0:
        extend = "down" if code[0] < code[-1] else "up"
    else:
        extend = "down" if vertex < u[0] else "up" if vertex > u[-1] else None
    return {"runs": len(runs), "curvature": float(a), "vertex": math.exp(vertex) if a > 0 else None, "extend": extend}


def analysis(lines: list[dict]) -> dict:
    """The pre-registered estimator over RESULTS/lr.jsonl's records (module docstring); the derived values are None
    while any setting's grid needs extending."""
    def setting(depth, ratio):
        return [line for line in lines if (line["setting"]["depth"], line["setting"]["tokens_per_param"]) == (depth, ratio)]

    def recorded_plan(depth, ratio):
        """The setting's plan at the law's lr (every arm's batch and steps are the same)."""
        line = setting(depth, ratio)[0]
        recorded = next(r for r in line["records"] if r["kind"] == "run")["plan"]
        return recorded | {"lr": recorded["lr"] / line["fields"]["lr_scale"]}

    def tokens(p):
        return p["num_iterations"] * p["total_batch_size"]

    fits = {f"d{depth} at {ratio:g}": fit(setting(depth, ratio)) for depth, ratio in ((8, SHORT), (8, LONG), (24, SHORT))}
    d8_short, d8_long, d24_short = recorded_plan(8, SHORT), recorded_plan(8, LONG), recorded_plan(24, SHORT)
    production = plan_of(Run(n_layer=24, ratio=max_ratio(get_tokenizer(), depth=24), lr_scale=1.0,
                             value_embeds="alternating", seed=0))
    m_b_production = production.total_batch_size / d24_short["total_batch_size"]
    m_b_d8 = d8_long["total_batch_size"] / d8_short["total_batch_size"]
    m_d_production = production.num_iterations * production.total_batch_size / tokens(d24_short)
    m_d_d8 = tokens(d8_long) / tokens(d8_short)
    law_fall_d8 = d8_long["lr"] / d8_short["lr"]
    out = {"settings": fits, "m_b_production": m_b_production, "m_b_d8": m_b_d8, "m_d_production": m_d_production,
           "m_d_d8": m_d_d8, "law_fall_d8": law_fall_d8,
           "predicted_d8_ratio": {"sde": math.sqrt(m_b_d8 / m_d_d8) / law_fall_d8,
                                  "step_law": m_d_d8 ** STEP_LAW_D / law_fall_d8},
           "measured_d8_ratio": None, "horizon_exponent": None, "production_multiplier": None, "production_lr": None}
    if all(f["extend"] is None for f in fits.values()):
        short, long, d24 = (fits[k]["vertex"] for k in fits)
        ratio = long / short
        out["measured_d8_ratio"] = ratio
        out["horizon_exponent"] = math.log(ratio) / math.log(m_d_d8)
        out["production_multiplier"] = (d24 * ratio ** (math.log(m_d_production) / math.log(m_d_d8))
                                        * math.sqrt(m_b_production / m_b_d8))
        out["production_lr"] = out["production_multiplier"] * production.lr
    return out


if __name__ == "__main__":
    grid = {"lr x2": {"lr_scale": 2.0}, "lr x4": {"lr_scale": 4.0}}
    # the cheap settings first
    main(settings=((8, SHORT, range(1), grid), (8, LONG, range(1), grid | {"lr x8": {"lr_scale": 8.0}}),
                   (24, SHORT, range(1), grid)))
    lines = [json.loads(line) for line in (RESULTS / "lr.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((RESULTS / "lr.json").read_text(encoding="utf-8"))
    (RESULTS / "lr.json").write_text(json.dumps(summary | {"analysis": analysis(lines)}, allow_nan=False),
                                     encoding="utf-8")
