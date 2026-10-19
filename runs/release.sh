#!/bin/bash
# Publish a run as a Hugging Face model repository in the experiment directory's layout: final checkpoint
# (base_checkpoints/<tag>/checkpoint.pt), metrics record, base_eval results. `hf download <repo> --local-dir
# ~/nanochat-runs` restores them for train/checkpoint.load_model, scripts/base_eval.py and a resumed base_train. Needs a
# write token (runs/container.sh hf auth login).
#
#   runs/release.sh <tag> <repo>
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=$1 REPO=$2
runs/container.sh hf upload "$REPO" /runs . --repo-type model --include "base_checkpoints/$TAG/checkpoint.pt" \
  --include "metrics/$TAG.jsonl" --include "base_eval/$TAG.json"
