#!/bin/bash
# The production run on one H100 80 GB SXM5: d24 (nanochat/model/gpt.py ASPECT_RATIO: 24 x 768) at the most tokens per
# scaling parameter the data holds without reading a document twice (nanochat/train/stages.py max_ratio), seed 42, both
# stages (file-level 8K, then repo-level 32K), then base_eval on its final checkpoint and its tables
# (~/nanochat-runs/results/<tag>.md, scripts/report.py). The first launch copies the repository to
# ~/nanochat-runs/trees/h100 and records the ratio; every launch runs that copy at that ratio. base_train resumes from
# the run's newest checkpoint; after a failure the runner relaunches it as long as the failed attempt saved a newer
# checkpoint than the one it started from (a failure without progress stops the runner).
#
#   tmux new -d -s h100 'cd ~/projects/nanochat && runs/h100.sh'
set -euo pipefail
cd "$(dirname "$0")/.."
TREE=$HOME/nanochat-runs/trees/h100 LOGS=$HOME/nanochat-runs/logs/h100
if [[ $(pwd -P) != "$(realpath -m "$TREE")" ]]; then
  [[ -e $TREE ]] || rsync -a --mkpath --exclude /dev-ignore --exclude /.git --exclude /runs/logs --exclude __pycache__ \
    --exclude .ruff_cache "$(pwd)/" "$TREE/"
  exec "$TREE/runs/h100.sh"
fi
DEPTH=24 SEED=42
mkdir -p "$LOGS"
# the run's identity, appended at every launch: the image and a hash of the frozen tree's files
{
  echo "$(date -Is) image $(docker image inspect -f '{{.Id}}' nanochat)"
  echo "$(date -Is) tree $(find . -type f -not -path '*/__pycache__/*' -not -path './.ruff_cache/*' -print0 | sort -z \
    | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
} >>"$LOGS/runner.log"
[[ -s $LOGS/ratio ]] || runs/container.sh python -m nanochat.train.stages "$DEPTH" | tail -n 1 >"$LOGS/ratio"
RATIO=$(cat "$LOGS/ratio")
read -r TAG STEPS REPO_START < <(runs/container.sh python -m nanochat.train.schedule "$DEPTH" "$RATIO" "$SEED" | tail -n 1)
# a checkpoint every 1% of the plan's steps: a failure loses at most 1% of the run; kept besides: the model before the
# repo-level stage (what the 32K stage adds)
SAVE_EVERY=$(( (STEPS + 99) / 100 ))
echo "$(date -Is) $TAG ratio $RATIO steps $STEPS save_every $SAVE_EVERY keep $REPO_START" >>"$LOGS/runner.log"
CKPT=$HOME/nanochat-runs/base_checkpoints/$TAG/checkpoint.pt
# the checkpoint's modification time (0 before the first one): a relaunch continues only after progress
saved() { stat -c %Y "$CKPT" 2>/dev/null || echo 0; }
until
  start=$(saved)
  runs/container.sh python -m scripts.base_train --depth "$DEPTH" --target-param-data-ratio "$RATIO" --seed "$SEED" \
    --save-every "$SAVE_EVERY" --keep "$REPO_START" >>"$LOGS/train.log" 2>&1
do
  end=$(saved)
  echo "$(date -Is) base_train failed: checkpoint saved at $start -> $end" >>"$LOGS/runner.log"
  [[ $end != "$start" ]] || exit 1
done
runs/container.sh python -m scripts.base_eval --model-tag "$TAG" >>"$LOGS/eval.log" 2>&1
runs/container.sh python -m scripts.report --out "/runs/results/$TAG.md" "$TAG" >>"$LOGS/eval.log" 2>&1
echo "$(date -Is) done" >>"$LOGS/runner.log"
