#!/bin/bash
# The production run on one H100 80 GB SXM5: d24 at the most tokens per scaling parameter the data holds without reading
# a document twice (nanochat/train/stages.py max_ratio), seed 0 (the d24 studies' seed-0 init and data order), both
# stages, then base_eval and its tables (scripts/report.py). Rerunning on the same code (config.run_tag) continues from
# the newest checkpoint.
#
#   tmux new -d -s h100 'cd ~/projects/nanochat-next && runs/h100.sh'
set -euo pipefail
cd "$(dirname "$0")/.."
DEPTH=24 SEED=0 LOGS=$HOME/nanochat-runs/logs/h100
mkdir -p "$LOGS"
echo "$(date -Is) commit $(git rev-parse HEAD)$(git diff --quiet HEAD || echo -dirty)" \
  "image $(docker image inspect -f '{{.Id}}' nanochat)" >>"$LOGS/runner.log"
RATIO=$(runs/container.sh python -m nanochat.train.stages "$DEPTH" | tail -n 1)
read -r TAG STEPS _ < <(runs/container.sh python -m nanochat.train.schedule "$DEPTH" "$RATIO" "$SEED" | tail -n 1)
# a checkpoint every 1% of the steps: a failure costs at most 1% of the run
runs/container.sh python -m scripts.base_train --depth "$DEPTH" --target-param-data-ratio "$RATIO" --seed "$SEED" \
  --save-every $(((STEPS + 99) / 100)) >>"$LOGS/train.log" 2>&1
runs/container.sh python -m scripts.base_eval --model-tag "$TAG" >>"$LOGS/eval.log" 2>&1
runs/container.sh python -m scripts.report --out "/runs/results/$TAG.md" "$TAG" >>"$LOGS/eval.log" 2>&1
