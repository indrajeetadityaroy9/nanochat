"""
Peak-lr calibration (d24): the base model's multiplier of DeepSeek LLM's law eta = 0.3118 C^-0.125 (arXiv 2401.02954
§3.1 Eq. 1), the optimum of d24 at 5 tokens per parameter; the law, with its batch, carries the compute dependence
to the base plan. The runs: conf/experiment/peak_lr_d24.yaml; their records: results/peak_lr_d24.jsonl
(ablations/records.py).
"""
import json
import math

import metrics
from paths import RESULTS


def lr_multiplier() -> float:
    """The base multiplier: the vertex of the parabola in u = ln(multiplier) through the runs' code bits per byte (as
    arXiv 2409.19913 §3.1)."""
    lines = [json.loads(line) for line in (RESULTS / "peak_lr_d24.jsonl").read_text(encoding="utf-8").splitlines()]
    u = [math.log(line["config"]["learning_rate_scale"]) for line in lines]
    _, vertex = metrics.parabola_vertex(u, [line["measures"]["code"] for line in lines])
    return math.exp(vertex)
