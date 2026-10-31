"""
An SFT run's held-out loss per source: a saved model's per-label losses over the val rows and labels of
train/loop.py validate (train/stages.py val_batches), summed over each val document's scored labels
(corpus/posttrain/build.py val_labels: [start + context, end) of its row) and pooled per group, per KodCode style
("kodcode/<style>") and over all ("all": the run log's val record's loss at that step, up to summation order).
Written to RUNS/<run>/sft-val-<step>.json: {source: {"loss" (nats per label), "labels", "documents"}} and
"document_losses" (each val document's summed loss and label count, in document order).

    python -m eval.sft_val <run_name> <step>
"""
import json
import sys
from collections import defaultdict

import numpy as np
import torch

from corpus.posttrain.build import labels_path
from corpus.posttrain.sft import SFT_STAGE
from corpus.tokenizer import get_tokenizer
from eval.suite import write_atomically
from paths import RUNS
from train.checkpoint import load_model
from train.stages import val_batches


@torch.no_grad()
def row_losses(run_name: str, step: int) -> np.ndarray:
    """Saved model `step` of the run's per-label loss (rows, seq_len) fp32 over the val rows in order (the last
    micro-batch's blank rows after them), 0 at ignored labels."""
    device = torch.device("cuda")
    model = load_model(run_name, step, device=device)
    losses = torch.compile(model.losses, dynamic=False, fullgraph=True)
    rows = val_batches(get_tokenizer(), stage=SFT_STAGE, device=device)
    return torch.cat([losses(input_ids, labels, cu_seqlens, segments).view(-1, SFT_STAGE.seq_len).cpu()
                      for input_ids, labels, cu_seqlens, segments, _ in rows]).numpy()


def source_losses(losses: np.ndarray, labels: list[dict]) -> dict:
    """Per-label `losses` (rows, seq_len) pooled per source of the val documents `labels` (val_labels' lines)."""
    sums, counts, documents = defaultdict(float), defaultdict(int), defaultdict(int)
    per_document = []
    for label in labels:
        span = losses[label["row"], label["start"] + label["context"]:label["end"]]
        total = float(span.sum(dtype=np.float64))
        per_document.append([total, len(span)])
        for source in ("all", label["group"], *([f"kodcode/{label['style']}"] if label["style"] else [])):
            sums[source] += total
            counts[source] += len(span)
            documents[source] += 1
    return {source: {"loss": sums[source] / counts[source], "labels": counts[source],
                     "documents": documents[source]} for source in sums} | {"document_losses": per_document}


def main(run_name: str, step: int) -> None:
    labels = [json.loads(line) for line in labels_path().read_text(encoding="utf-8").splitlines()]
    result = source_losses(row_losses(run_name, step), labels)
    write_atomically(RUNS / run_name / f"sft-val-{step}.json", json.dumps(result, allow_nan=False).encode())
    print(json.dumps({k: v for k, v in result.items() if k != "document_losses"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
