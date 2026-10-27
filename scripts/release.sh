#!/bin/bash
# Upload a run's directory (checkpoints, metrics.jsonl, evaluations and their sidecars, .hydra) and results/ to a
# Hugging Face model repository in the checkout's layout. Usage: scripts/release.sh <run_name> <repo>
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_NAME=$1 REPO=$2
"$ROOT"/scripts/container.sh hf upload "$REPO" /workspace/nanochat . --repo-type model \
  --include "artifacts/runs/$RUN_NAME/checkpoint-*.pt" --include "artifacts/runs/$RUN_NAME/metrics.jsonl" \
  --include "artifacts/runs/$RUN_NAME/eval-*.json" --include "artifacts/runs/$RUN_NAME/eval-*.samples.json.gz" \
  --include "artifacts/runs/$RUN_NAME/.hydra/*" \
  --include "results/*"
