#!/bin/bash
# Publish a run on the Hugging Face Hub as a model repository laid out like the experiment directory: its final
# checkpoint (base_checkpoints/<tag>/checkpoint.pt: weights, optimizer state and model config, train/checkpoint.py), its
# metrics record and its base_eval results. `hf download <repo> --local-dir ~/nanochat-runs` puts them back where
# train/checkpoint.load_model, scripts/base_eval.py and a resumed scripts/base_train.py read them. Needs a write token
# (runs/container.sh hf auth login).
#
#   runs/release.sh <tag> <repo>
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=$1 REPO=$2
runs/container.sh hf upload "$REPO" /runs . --repo-type model \
  --include "base_checkpoints/$TAG/checkpoint.pt" "metrics/$TAG.jsonl" "base_eval/$TAG.json"
