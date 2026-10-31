"""
Peak-lr calibration (d24): the base model's multiplier of DeepSeek LLM's law eta = 0.3118 C^-0.125 (arXiv 2401.02954
§3.1 Eq. 1), the optimum of d24 at 5 tokens per parameter; the law, with its batch, carries the compute dependence
to the base plan. The runs: conf/experiment/peak_lr_d24.yaml; their records: results/pretrain/peak_lr_d24.jsonl
(ablations/records.py).
"""
import json
import math

import metrics
from paths import PRETRAIN_RESULTS


def lr_multiplier() -> float:
    """The base multiplier: the vertex of the parabola in u = ln(multiplier) through the runs' code bits per byte (as
    arXiv 2409.19913 §3.1). Raises ValueError unless every arm was measured (a diverged arm was not) and the curvature
    is positive and the vertex inside the grid (then an arm 2x beyond the edge with the lowest bpb is added)."""
    lines = [json.loads(line)
             for line in (PRETRAIN_RESULTS / "peak_lr_d24.jsonl").read_text(encoding="utf-8").splitlines()]
    if diverged := [line["run_name"] for line in lines if line["measures"] is None]:
        raise ValueError(f"peak lr: arms {diverged} have no measures (diverged)")
    scales = [line["config"]["learning_rate_scale"] for line in lines]
    code = [line["measures"]["code"] for line in lines]
    u = [math.log(scale) for scale in scales]
    curvature, vertex = metrics.parabola_vertex(u, code)
    if not (curvature > 0 and min(u) < vertex < max(u)):
        raise ValueError(f"peak lr: no interior minimum (curvature {curvature:.4g}, vertex x{math.exp(vertex):.4g}); "
                         f"code bpb by scale {dict(zip(scales, code, strict=True))}")
    return math.exp(vertex)
