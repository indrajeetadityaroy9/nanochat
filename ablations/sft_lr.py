"""
SFT peak-lr calibration (docs/post-training.md §11.2): complete schedules at 1/10, 1/3 and 1 times the base run's peak
lr over the same first 2% of the SFT train rows, each scored by the response loss on the held-out problems after its
last step; the full run's scale is the fitted optimum moved to the full run's steps (its epochs of the train rows). The
runs: conf/experiment/sft_lr.yaml; their records: results/sft_lr.jsonl (ablations/records.py).
"""
import json
import math

import metrics
from paths import RESULTS

# The optimum lr falls with the run's steps at a fixed batch as steps^-STEPS_EXPONENT: 0.32 in arXiv 2409.19913 §6
# Eq. 5, 0.30 in its §4.1 (Llama-1), 0.24-0.30 from the repository's d8 pair (docs/training.md §5) [inference:
# transferred to SFT]
STEPS_EXPONENT = 0.30


def sft_lr_fit(epochs: float) -> dict:
    """The arms' final held-out losses against u = ln(scale), fitted by a parabola (metrics.parabola_vertex, as
    ablations/peak_lr.py lr_multiplier); its vertex times (run steps / arm steps)^-STEPS_EXPONENT, the steps from the
    arms' plans (their run records) and the full run's (train/schedule.py sft_plan over `epochs` of the train rows).
    Raises ValueError unless every arm has a val record (a diverged arm has none) and the curvature is positive and the
    vertex inside the grid (then an arm 3x beyond the edge with the lowest loss is added to the grid)."""
    from corpus.layout import packed_rows
    from corpus.tokenizer import get_tokenizer
    from model.modeling import meta_model
    from posttrain.sft import NAME, SFT_STAGE
    from train.schedule import sft_plan

    lines = [json.loads(line) for line in (RESULTS / "sft_lr.jsonl").read_text(encoding="utf-8").splitlines()]
    scales = [line["config"]["learning_rate_scale"] for line in lines]
    losses = []
    for line in lines:
        validated = [r for r in line["records"] if r["kind"] == "val"]
        if not validated:
            raise ValueError(f"SFT lr: arm x{line['config']['learning_rate_scale']} ({line['run_name']}) has no val "
                             f"record (diverged at step {line['diverged_at']})")
        losses.append(max(validated, key=lambda r: r["step"])["loss"])
    u = [math.log(scale) for scale in scales]
    curvature, vertex = metrics.parabola_vertex(u, losses)
    if not (curvature > 0 and min(u) < vertex < max(u)):
        raise ValueError(f"SFT lr: no interior minimum (curvature {curvature:.4g}, vertex x{math.exp(vertex):.4g}); "
                         f"final val loss by scale {dict(zip(scales, losses, strict=True))}")
    (arm_steps,) = {next(r for r in line["records"] if r["kind"] == "run")["plan"]["max_steps"] for line in lines}
    rows = packed_rows(NAME, "train", get_tokenizer(), SFT_STAGE.seq_len)
    run_steps = sft_plan(model=meta_model(), rows=rows, epochs=epochs, learning_rate=1.0).max_steps
    factor = (run_steps / arm_steps) ** -STEPS_EXPONENT
    return {"losses": dict(zip(scales, losses, strict=True)), "curvature": curvature, "vertex": math.exp(vertex),
            "arm_steps": arm_steps, "run_steps": run_steps, "factor": factor, "scale": math.exp(vertex) * factor}


def sft_lr_scale(epochs: float) -> float:
    """The learning_rate_scale of a full SFT run of `epochs` (sft_lr_fit), its fit printed."""
    fit = sft_lr_fit(epochs)
    print(f"SFT lr calibration: {json.dumps(fit)}", flush=True)
    return fit["scale"]
