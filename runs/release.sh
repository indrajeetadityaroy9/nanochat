#!/bin/bash
# Publish a run as a Hugging Face model repository in the checkout's layout (nanochat/paths.py): its run directory
# artifacts/runs/<key>/ (step<step>.pt, metrics.jsonl, eval-step<step>.json, .hydra) and results/.
# `runs/container.sh hf download <repo> --local-dir /workspace/nanochat` restores them for train/checkpoint.load_model
# and a resumed pre-training run. Needs a write token (runs/container.sh hf auth login).
#
#   runs/release.sh <key> <repo>          (key: the run's directory name under artifacts/runs)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
KEY=$1 REPO=$2
# /workspace/nanochat: the checkout inside the container (runs/container.sh)
"$ROOT"/runs/container.sh hf upload "$REPO" /workspace/nanochat . --repo-type model \
  --include "artifacts/runs/$KEY/step*.pt" --include "artifacts/runs/$KEY/metrics.jsonl" \
  --include "artifacts/runs/$KEY/eval-step*.json" --include "artifacts/runs/$KEY/.hydra/*" --include "results/*"
