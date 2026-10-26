"""
Peak-lr calibration (pre-registered): the base model's multiplier of DeepSeek LLM's law eta = 0.3118 C^-0.125
(arXiv 2401.02954 §3.1 Eq. 1), exp(u*_d24@SHORT) (r / sqrt(m_B,d8))^(ln m_D,base / ln m_D,d8) sqrt(m_B,base), with
r = exp(u*_d8@LONG - u*_d8@SHORT); r / sqrt(m_B,d8) nets d8's batch growth by the square-root batch rule (arXiv
2205.10287 Def. 5.2). r is compared with arXiv 2512.22382 §2.1 Table 1 and Step-Law (arXiv 2503.04715 §1 Eq. 1).
The runs: conf/experiment/peak_lr_d8.yaml, peak_lr_d24.yaml.
"""
import json
import math
from dataclasses import asdict

import numpy as np

import metrics
from corpus.mixture import FILE_LEVEL_STAGE, REPO_LEVEL_STAGE
from corpus.tokenizer import get_tokenizer
from paths import RESULTS
from run import RunConfig, base_plan, plan_of, tokenizer_path
from train.schedule import RunPlan
from train.stages import stage_mixture

EXPERIMENTS = ("peak_lr_d8", "peak_lr_d24")
# short horizon in tokens per parameter: the longest d24 run a sweep affords
SHORT = 5
# d8's long horizon, where its law batch is 2 micro-batches (batch growth netted out by the square-root rule)
LONG = 200
# Step-Law's token exponent of the optimal lr (arXiv 2503.04715 §1 Eq. 1: eta = 1.79 N^-0.713 D^0.307)
STEP_LAW_D = 0.307
# (num_hidden_layers, tokens_per_param) of each setting, as the experiments sweep them
SETTINGS = {"d8 at 5": (8, SHORT), "d8 at 200": (8, LONG), "d24 at 5": (24, SHORT)}


def records() -> list[dict]:
    """The experiments' records, results/<experiment>.jsonl (ablations/records.py)."""
    return [json.loads(line) for name in EXPERIMENTS
            for line in (RESULTS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]


def check(line: dict) -> None:
    """Raises unless the run trained the current recipe: its recorded plan, tokenizer data key and each stage's
    mixture are those the current code computes for its RunConfig."""
    config = RunConfig(**line["config"])
    run = plan_of(config)
    tokenizer = get_tokenizer(tokenizer_path(config))
    mixtures = {stage.seq_len: stage_mixture(tokenizer, stage, run)[0]
                for stage in (FILE_LEVEL_STAGE, REPO_LEVEL_STAGE)}
    for r in line["records"]:
        changed = r["kind"] == "run" and (r["plan"] != asdict(run) or r["data_key"] != tokenizer.data_key)
        if changed or (r["kind"] in ("run", "stage") and r["stage"]["mixture"] != mixtures[r["stage"]["seq_len"]]):
            raise RuntimeError(f"{line['run_name']}: not trained on the current recipe; re-run its experiment")


def fit(lines: list[dict]) -> dict:
    """A setting's optimum: the vertex of the parabola in u = ln(multiplier) through the lowest-code run and its
    neighbours (as arXiv 2409.19913 §3.1), or that run itself if its upper neighbour diverged; a diverged run counts as
    worse than any evaluated one. Raises where the rule extends the grid x2: the lowest-code run at a grid end."""
    runs = sorted(lines, key=lambda line: line["config"]["learning_rate_scale"])
    u = [math.log(line["config"]["learning_rate_scale"]) for line in runs]
    code = [math.inf if line["measures"] is None else line["measures"]["code"] for line in runs]
    best = int(np.argmin(code))
    if best in (0, len(runs) - 1) or not math.isfinite(code[best - 1]):
        raise RuntimeError("the lowest-code run is at a grid end or its lower neighbour diverged: extend the grid")
    if not math.isfinite(code[best + 1]):
        return {"curvature": None, "vertex": math.exp(u[best])}
    curvature, vertex = metrics.parabola_vertex(u[best - 1:best + 2], code[best - 1:best + 2])
    return {"curvature": curvature, "vertex": math.exp(vertex)}


def analysis(lines: list[dict]) -> dict:
    """The estimator over the experiments' records, each checked against the current recipe (`check`)."""
    for line in lines:
        check(line)
    fits = {name: fit([line for line in lines
                       if (line["config"]["num_hidden_layers"], line["config"]["tokens_per_param"]) == setting])
            for name, setting in SETTINGS.items()}
    d8_short, d8_long, d24_short = (base_plan(num_hidden_layers=n, tokens_per_param=t) for n, t in SETTINGS.values())
    base = plan_of(RunConfig())

    def tokens(p: RunPlan) -> int:
        return p.max_steps * p.batch_size

    m_b_base = base.batch_size / d24_short.batch_size
    m_b_d8 = d8_long.batch_size / d8_short.batch_size
    m_d_base = tokens(base) / tokens(d24_short)
    m_d_d8 = tokens(d8_long) / tokens(d8_short)
    law_fall_d8 = d8_long.learning_rate / d8_short.learning_rate
    short, long, d24 = (f["vertex"] for f in fits.values())
    ratio = long / short
    horizon_exponent_net = math.log(ratio / math.sqrt(m_b_d8)) / math.log(m_d_d8)
    base_multiplier = d24 * m_d_base ** horizon_exponent_net * math.sqrt(m_b_base)
    return {"settings": fits, "m_b_base": m_b_base, "m_b_d8": m_b_d8, "m_d_base": m_d_base,
            "m_d_d8": m_d_d8, "law_fall_d8": law_fall_d8,
            "predicted_d8_ratio": {"sde": math.sqrt(m_b_d8 / m_d_d8) / law_fall_d8,
                                   "step_law": m_d_d8 ** STEP_LAW_D / law_fall_d8},
            "measured_d8_ratio": ratio, "horizon_exponent": math.log(ratio) / math.log(m_d_d8),
            "horizon_exponent_net": horizon_exponent_net, "base_multiplier": base_multiplier,
            "base_lr": base_multiplier * base.learning_rate}
