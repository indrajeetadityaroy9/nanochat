"""
Peak lr. The base model's peak lr is DeepSeek LLM's law eta = 0.3118 C^-0.125 (arXiv 2401.02954 §3.1 Eq. 1;
train/schedule.py plan), fit for AdamW under a multi-step schedule (arXiv 2401.02954 §2.3) at compute below the base
model's; pre-training applies it to Muon (matched to AdamW's update RMS, train/optim.py) as the peak of linear decay to
zero, and at a fixed model it falls with tokens as D^-0.125.

The base model's lr is measured, not swept at its length: its optimum at d24 (the base model, so no width
or depth transfer is assumed; its parameterisation is standard, not muP) at SHORT tokens per parameter, times the
optimum's horizon trend measured at d8 between SHORT and LONG, assumed width-independent (arXiv 2409.19913
§4 finds its horizon exponent beta size-independent only for models of at least 760M parameters, larger for smaller
ones). The published rules disagree on the trend's direction: Complete^(d)P's AdamW SDE rule eta ∝ sqrt(m_B / m_D), for
batch ratio m_B and token ratio m_D (arXiv 2512.22382 Table 1, §2.3), lowers the lr with the horizon, as
arXiv 2409.19913 §3.2 Eq. 1 and the law do; Step-Law eta = 1.79 N^-0.713 D^0.307, fit over AdamW models
(arXiv 2503.04715 §1 Eq. 1), raises it. Alternatives: the law's lr times 2 and 4, and 8 at d8 at LONG, where Step-Law
points; one seed per setting.

Analysis, pre-registered, computed when the pretrain experiment launches (run.py lr_multiplier, from
RESULTS/peak_lr.jsonl; it depends on the adopted tokenizer's max_tokens_per_param, so no copy is stored):
- per setting, code bits per byte fitted as a parabola in u = ln(multiplier) (fit): its vertex u* or the grid extension
  the rule calls for;
- the measured d8 ratio r = exp(u*_d8@LONG - u*_d8@SHORT) against the SDE rule's and Step-Law's predictions, each over
  the law's own fall between the two plans;
- the horizon exponent ln r / ln m_D,d8;
- the base model's multiplier of the law's lr, exp(u*_d24@SHORT) * (r / sqrt(m_B,d8))^(ln m_D,base / ln m_D,d8)
  * sqrt(m_B,base): d8's trend net of its batch growth by the square-root batch rule (arXiv 2205.10287 Def. 5.2, its lr
  factor; arXiv 2512.22382 Table 1), its exponent (horizon_exponent_net) carried to the base model's horizon range
  (every rule in play is a power law in D), and the base model's batch growth by the same rule.
m_D,base is the base model's tokens over d24 at SHORT's, m_D,d8 d8 at LONG's over d8 at SHORT's. Batches, token counts
and the law's lrs are the runs' recorded plans (the base model's: run.py plan_of(RunConfig()), d24 at its
data's max_tokens_per_param).

The recorded runs (RESULTS/peak_lr.jsonl, peak_lr-curves.jsonl, peak_lr.json) use an earlier record format: a run's
fields under "fields" (n_layer, ratio, lr_scale, seed) and its setting under "setting" (depth, tokens_per_param,
seeds); the grids: d8 at SHORT x1, 2, 4, 8, d8 at LONG x1, 2, 4, 8, d24 at SHORT x1, 2, 4, one seed each. The
estimate is the pretrain experiment's learning_rate_scale (conf/experiment/pretrain.yaml, run.py lr_multiplier).
"""
import math

import numpy as np

import metrics
from run import RunConfig, base_plan, plan_of

# The short horizon in tokens per scaling parameter: the longest d24 run a sweep affords
SHORT = 5
# d8's long horizon: its law batch is 2 micro-batches there (schedule.plan's rounding), so the trend is measured under
# batch co-scaling as the base model trains, its batch growth netted out by the square-root rule
LONG = 200
# Step-Law's token exponent of the optimal lr (arXiv 2503.04715 §1 Eq. 1: eta = 1.79 N^-0.713 D^0.307)
STEP_LAW_D = 0.307


def setting(lines: list[dict], num_hidden_layers: int, tokens_per_param: float) -> list[dict]:
    """The records of a setting's runs."""
    return [line for line in lines
            if (line["setting"]["depth"], line["setting"]["tokens_per_param"]) == (num_hidden_layers, tokens_per_param)]


def fit(lines: list[dict]) -> dict:
    """A setting's optimum from its runs in lr order, a diverged run counting as worse than any evaluated one: the grid
    extension by a factor of 2 ("down" or "up") while its lowest-code run is at an end of the grid; else the vertex (as
    a multiplier) of the parabola in u = ln(multiplier) through the lowest-code run and its two neighbours (positive
    curvature by construction; a local bracket, as arXiv 2409.19913 §3.1 fits quadratics in log lr), or the lowest-code
    run itself when the neighbour above it diverged."""
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
    """The pre-registered estimator over RESULTS/peak_lr.jsonl's records (module docstring); the derived values are
    None while any setting's grid needs extending. The vertices are multipliers of the law's lr the runs recorded; they
    are re-based onto the current law's lr (base_plan), whose batch and steps must be the recorded runs'."""
    def recorded_plan(num_hidden_layers, tokens_per_param):
        """The setting's current plan (every arm's batch and steps are the same), with the law's lr its runs
        recorded; the old record keys of RESULTS/peak_lr.jsonl."""
        line = setting(lines, num_hidden_layers, tokens_per_param)[0]
        recorded = next(r for r in line["records"] if r["kind"] == "run")["plan"]
        current = base_plan(num_hidden_layers=num_hidden_layers, tokens_per_param=tokens_per_param)
        if (recorded["total_batch_size"], recorded["num_iterations"]) != (current.batch_size, current.max_steps):
            raise RuntimeError(f"d{num_hidden_layers} at {tokens_per_param:g}: the recorded runs' batch and steps are "
                               "not the current plan's")
        return recorded | {"lr": current.learning_rate, "recorded_lr": recorded["lr"] / line["fields"]["lr_scale"]}

    def tokens(p):
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
