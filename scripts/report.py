"""
Every tracked metric of a base_train run as JSON lines in RESULTS, the data the paper's tables and figures are drawn
from: RESULTS/<tag>.jsonl, one record: the run's train/metrics.py records but the per-step ones (run, stage, end,
failure), the evaluations of its final model and of its model-only snapshots verbatim (BASE_EVAL/<tag>[-step<NNNNNN>]
.json, scripts/base_eval.py: val bits and bytes per corpus and context bucket, reference-solution bits, greedy and
system decoding's per-task results) and its probe results (PROBES/<tag>[-step<NNNNNN>].json, eval/probes.py), each
list in step order; RESULTS/<tag>-curves.jsonl, its per-step train records as columns (train/metrics.py curves).
Aggregates (bits per byte, pass@1 tables) are drawn from these downstream.

    python -m scripts.report <tag>    (scripts/production.py, after the run's and its snapshots' evaluations)
"""
import json
import sys
from pathlib import Path

from nanochat.paths import BASE_EVAL, PROBES, RESULTS
from nanochat.train.metrics import curves, records


def outputs(directory: Path, tag: str) -> list[dict]:
    """The JSON results of a run's final model and snapshots in `directory`, in step order."""
    paths = [p for p in (directory / f"{tag}.json", *directory.glob(f"{tag}-step*.json")) if p.exists()]
    return sorted((json.loads(p.read_text(encoding="utf-8")) for p in paths), key=lambda r: r["step"])


def record(tag: str) -> dict:
    """A run's tracked metrics but the per-step ones."""
    return {"tag": tag, "records": [r for r in records(tag) if r["kind"] != "train"],
            "evaluations": outputs(BASE_EVAL, tag), "probes": outputs(PROBES, tag)}


def main():
    tag = sys.argv[1]
    (RESULTS / f"{tag}.jsonl").write_text(json.dumps(record(tag), allow_nan=False) + "\n", encoding="utf-8")
    (RESULTS / f"{tag}-curves.jsonl").write_text(json.dumps({"tag": tag} | curves(tag), allow_nan=False) + "\n",
                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
