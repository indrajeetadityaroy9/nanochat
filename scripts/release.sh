#!/bin/bash
# Publish a run as a Hugging Face model repository in the checkout's layout (paths.py): its run directory
# artifacts/runs/<run_name>/ (checkpoint-<step>.pt, metrics.jsonl, eval-<step>.json, .hydra) and results/.
# `scripts/container.sh hf download <repo> --local-dir /workspace/nanochat` restores them for
# train/checkpoint.load_model and a resumed pre-training run. Needs a write token (scripts/container.sh hf auth login).
#
#   scripts/release.sh <run_name> <repo>     (run_name: the run's directory name under artifacts/runs)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_NAME=$1 REPO=$2
# /workspace/nanochat: the checkout inside the container (scripts/container.sh)
"$ROOT"/scripts/container.sh hf upload "$REPO" /workspace/nanochat . --repo-type model \
  --include "artifacts/runs/$RUN_NAME/checkpoint-*.pt" --include "artifacts/runs/$RUN_NAME/metrics.jsonl" \
  --include "artifacts/runs/$RUN_NAME/eval-*.json" --include "artifacts/runs/$RUN_NAME/.hydra/*" \
  --include "results/*"
