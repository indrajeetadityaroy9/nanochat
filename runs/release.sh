#!/bin/bash
# Publish a run as a Hugging Face model repository in the checkout's layout (nanochat/paths.py): the checkpoint
# artifacts/checkpoints/<tag>.pt and its snapshots <tag>-step<NNNNNN>.pt, the metrics record, the base_eval and probe
# results, and the tables results/<tag>*.md. `runs/container.sh hf download <repo> --local-dir /workspace/nanochat`
# restores them for train/checkpoint.load_model, scripts/base_eval.py and a resumed base_train. Needs a write token
# (runs/container.sh hf auth login).
#
#   runs/release.sh <tag> <repo>
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TAG=$1 REPO=$2
# /workspace/nanochat: the checkout inside the container (runs/container.sh)
"$ROOT"/runs/container.sh hf upload "$REPO" /workspace/nanochat . --repo-type model \
  --include "artifacts/checkpoints/$TAG.pt" --include "artifacts/checkpoints/$TAG-step*.pt" \
  --include "artifacts/metrics/$TAG.jsonl" --include "artifacts/base_eval/$TAG*.json" \
  --include "artifacts/probes/$TAG*" --include "results/$TAG*.md"
