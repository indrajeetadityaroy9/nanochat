#!/bin/bash
# Publish a run as a Hugging Face model repository in the checkout's layout (nanochat/paths.py): the checkpoint
# artifacts/checkpoints/<tag>.pt and its snapshots <tag>-step<NNNNNN>.pt, the run records under artifacts/ (metrics,
# base_eval) and results/.
# `runs/container.sh hf download <repo> --local-dir /workspace/nanochat` restores them for train/checkpoint.load_model
# and a resumed production run. Needs a write token (runs/container.sh hf auth login).
#
#   runs/release.sh <tag> <repo>
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TAG=$1 REPO=$2
# /workspace/nanochat: the checkout inside the container (runs/container.sh)
"$ROOT"/runs/container.sh hf upload "$REPO" /workspace/nanochat . --repo-type model \
  --include "artifacts/checkpoints/$TAG.pt" --include "artifacts/checkpoints/$TAG-step*.pt" \
  --include "artifacts/metrics/*" --include "artifacts/base_eval/*" --include "results/*"
