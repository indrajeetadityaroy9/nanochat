#!/bin/bash
# Capability curves of a run: base_eval (loss, greedy and system decoding) and the probe set on each model-only snapshot
# artifacts/checkpoints/<tag>-step<NNNNNN>.pt (scripts/base_train.py --snapshot-every; runs/h100.sh keeps one every 10%),
# then one table across the snapshots and the final model, results/<tag>-curves.md (scripts/report.py). Run after the
# run's own evaluation (runs/h100.sh). Log: artifacts/logs/curves-<tag>.log.
#
#   tmux new -d -s curves runs/curves.sh <tag>    (from the checkout)
set -euo pipefail
shopt -s failglob  # no snapshots is an error, not an empty table
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TAG=$1
LOG=$ROOT/artifacts/logs/curves-$TAG.log
NAMES=()
for snapshot in "$ROOT/artifacts/checkpoints/$TAG"-step*.pt; do
  name=$(basename "$snapshot" .pt)
  step=$((10#${name##*-step}))
  "$ROOT"/runs/container.sh python -m scripts.base_eval --model-tag "$TAG" --step "$step" >>"$LOG" 2>&1
  "$ROOT"/runs/container.sh python -m nanochat.eval.probes --model-tag "$TAG" --step "$step" >>"$LOG" 2>&1
  NAMES+=("$name")
done
"$ROOT"/runs/container.sh python -m scripts.report --name "$TAG-curves" "${NAMES[@]}" "$TAG" >>"$LOG" 2>&1
