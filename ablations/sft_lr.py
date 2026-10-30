"""
SFT peak-lr calibration (docs/post-training.md §11.2): complete schedules at 1/30, 1/10 and 1/3 of the base run's peak
lr over the same 2% of the SFT train rows, scored by the response loss on the held-out prompts after their last step.
The runs: conf/experiment/sft_lr.yaml; their records: results/sft_lr.jsonl (ablations/records.py).
"""
import json

from paths import RESULTS


def sft_lr_scale() -> float:
    """The scale (RunConfig.learning_rate_scale) of the run with the lowest final held-out response loss."""
    lines = [json.loads(line) for line in (RESULTS / "sft_lr.jsonl").read_text(encoding="utf-8").splitlines()]

    def final(line: dict) -> float:
        return max((r for r in line["records"] if r["kind"] == "val"), key=lambda r: r["step"])["loss"]

    return min(lines, key=final)["config"]["learning_rate_scale"]
