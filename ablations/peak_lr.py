"""
Peak-lr calibration (pre-registered): the base model's multiplier of DeepSeek LLM's law eta = 0.3118 C^-0.125
(arXiv 2401.02954 §3.1 Eq. 1), exp(u*_d24@SHORT) (r / sqrt(m_B,d8))^(ln m_D,base / ln m_D,d8) sqrt(m_B,base), with
r = exp(u*_d8@LONG - u*_d8@SHORT); r / sqrt(m_B,d8) nets d8's batch growth by the square-root batch rule (arXiv
2205.10287 Def. 5.2). r is compared with arXiv 2512.22382 §2.1 Table 1 and Step-Law (arXiv 2503.04715 §1 Eq. 1).
"""
import math

import numpy as np

import metrics
from run import RunConfig, base_plan, plan_of

# short horizon in tokens per parameter: the longest d24 run a sweep affords
SHORT = 5
# d8's long horizon, where its law batch is 2 micro-batches (batch growth netted out by the square-root rule)
LONG = 200
# Step-Law's token exponent of the optimal lr (arXiv 2503.04715 §1 Eq. 1: eta = 1.79 N^-0.713 D^0.307)
STEP_LAW_D = 0.307


def setting(lines: list[dict], num_hidden_layers: int, tokens_per_param: float) -> list[dict]:
    """The records of a setting's runs."""
    return [line for line in lines
            if (line["setting"]["depth"], line["setting"]["tokens_per_param"]) == (num_hidden_layers, tokens_per_param)]


def fit(lines: list[dict]) -> dict:
    """A setting's optimum: grid extension x2 ("down"/"up") while the lowest-code run is at a grid end, else the vertex
    of the parabola in u = ln(multiplier) through it and its neighbours (as arXiv 2409.19913 §3.1), or that run itself
    if its upper neighbour diverged; a diverged run counts as worse than any evaluated one."""
    runs = sorted(lines, key=lambda line: line["fields"]["lr_scale"])
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
    curvature, vertex = metrics.parabola_vertex(u[best - 1:best + 2], code[best - 1:best + 2])
    return {"runs": evaluated, "curvature": curvature, "vertex": math.exp(vertex), "extend": None}


def analysis(lines: list[dict]) -> dict:
    """The estimator over results/peak_lr.jsonl's records; derived values None while a grid needs extending. Vertices
    are re-based onto the current law's lr; raises unless the recorded batch and steps are the current plan's."""
    def recorded_plan(num_hidden_layers: int, tokens_per_param: float) -> dict:
        """The setting's recorded plan (old record keys) with the current and the recorded law's lr."""
        line = setting(lines, num_hidden_layers, tokens_per_param)[0]
        recorded = next(r for r in line["records"] if r["kind"] == "run")["plan"]
        current = base_plan(num_hidden_layers=num_hidden_layers, tokens_per_param=tokens_per_param)
        if (recorded["total_batch_size"], recorded["num_iterations"]) != (current.batch_size, current.max_steps):
            raise RuntimeError(f"d{num_hidden_layers} at {tokens_per_param:g}: the recorded runs' batch and steps are "
                               "not the current plan's")
        return recorded | {"lr": current.learning_rate, "recorded_lr": recorded["lr"] / line["fields"]["lr_scale"]}

    def tokens(p: dict) -> int:
        return p["num_iterations"] * p["total_batch_size"]

    fits = {f"d{num_hidden_layers} at {tokens_per_param:g}": fit(setting(lines, num_hidden_layers, tokens_per_param))
            for num_hidden_layers, tokens_per_param in ((8, SHORT), (8, LONG), (24, SHORT))}
    d8_short, d8_long, d24_short = recorded_plan(8, SHORT), recorded_plan(8, LONG), recorded_plan(24, SHORT)
    base = plan_of(RunConfig())
    m_b_base = base.batch_size / d24_short["total_batch_size"]
    m_b_d8 = d8_long["total_batch_size"] / d8_short["total_batch_size"]
    m_d_base = base.max_steps * base.batch_size / tokens(d24_short)
    m_d_d8 = tokens(d8_long) / tokens(d8_short)
    law_fall_d8 = d8_long["lr"] / d8_short["lr"]
    out = {"settings": fits, "m_b_base": m_b_base, "m_b_d8": m_b_d8, "m_d_base": m_d_base,
           "m_d_d8": m_d_d8, "law_fall_d8": law_fall_d8,
           "predicted_d8_ratio": {"sde": math.sqrt(m_b_d8 / m_d_d8) / law_fall_d8,
                                  "step_law": m_d_d8 ** STEP_LAW_D / law_fall_d8},
           "measured_d8_ratio": None, "horizon_exponent": None, "horizon_exponent_net": None,
           "base_multiplier": None, "base_lr": None}
    if all(f["extend"] is None for f in fits.values()):
        short, long, d24 = (fits[k]["vertex"] * p["recorded_lr"] / p["lr"]
                            for k, p in zip(fits, (d8_short, d8_long, d24_short), strict=True))
        ratio = long / short
        out["measured_d8_ratio"] = ratio
        out["horizon_exponent"] = math.log(ratio) / math.log(m_d_d8)
        out["horizon_exponent_net"] = math.log(ratio / math.sqrt(m_b_d8)) / math.log(m_d_d8)
        out["base_multiplier"] = d24 * m_d_base ** out["horizon_exponent_net"] * math.sqrt(m_b_base)
        out["base_lr"] = out["base_multiplier"] * base.learning_rate
    return out
