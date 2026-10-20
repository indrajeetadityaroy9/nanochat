#!/bin/bash
# Production run on one H100 80 GB SXM5: d24 at the most tokens per scaling parameter the data holds without rereading a
# document (nanochat/train/stages.py max_ratio), seed 0, both stages; then base_eval (loss, greedy and system decoding),
# the probe set (nanochat/eval/probes.py) and the table results/<tag>.md (scripts/report.py). Rerunning on the same code
# (config.run_tag) resumes from the checkpoint. Log: artifacts/logs/h100-<tag>.log.
#
#   tmux new -d -s h100 runs/h100.sh    (from the checkout)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEPTH=24 SEED=0
RATIO=$("$ROOT"/runs/container.sh python -m nanochat.train.stages "$DEPTH" | tail -n 1)
read -r TAG STEPS _ < <("$ROOT"/runs/container.sh python -m nanochat.train.schedule "$DEPTH" "$RATIO" "$SEED" | tail -n 1)
LOG=$ROOT/artifacts/logs/h100-$TAG.log
echo "$(date -Is) commit $(git -C "$ROOT" rev-parse HEAD)$(git -C "$ROOT" diff --quiet HEAD || echo -dirty)" \
  "image $(docker image inspect -f '{{.Id}}' nanochat)" >>"$LOG"
# checkpoint every 1% of steps: a failure costs at most 1% of the run; a model-only snapshot every 10% (~2.1 GB each)
# for evaluating intermediate models (capability curves)
"$ROOT"/runs/container.sh python -m scripts.base_train --depth "$DEPTH" --target-param-data-ratio "$RATIO" \
  --seed "$SEED" --save-every $(((STEPS + 99) / 100)) --snapshot-every $(((STEPS + 9) / 10)) >>"$LOG" 2>&1
# the evaluation runs the code that trained: the tag hashes config.CODE (config.run_tag), so a changed checkout refuses
read -r CHECK _ < <("$ROOT"/runs/container.sh python -m nanochat.train.schedule "$DEPTH" "$RATIO" "$SEED" | tail -n 1)
[ "$CHECK" = "$TAG" ] || { echo "$(date -Is) code changed since training ($CHECK != $TAG)" >>"$LOG"; exit 1; }
"$ROOT"/runs/container.sh python -m scripts.base_eval --model-tag "$TAG" >>"$LOG" 2>&1
"$ROOT"/runs/container.sh python -m nanochat.eval.probes --model-tag "$TAG" >>"$LOG" 2>&1
"$ROOT"/runs/container.sh python -m scripts.report --name "$TAG" "$TAG" >>"$LOG" 2>&1
