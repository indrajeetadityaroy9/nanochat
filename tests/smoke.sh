#!/bin/bash
# Two-stage smoke on the H100: d8 at 20 tokens per scaling parameter, a checkpoint every repo_start steps (one at the
# switch to the repo-level stage). The first launch is killed 8 steps after the switch; the relaunch resumes from the
# switch checkpoint in the repo-level stage and finishes; then smoke_check (the steps both launches took must have
# equal losses) and decoder_check on the trained model. Only the smoke's own container is stopped: runs/container.sh
# execs `docker run` (sig-proxy on by default), so the signal to its pid reaches that container alone.
set -euo pipefail
cd "$(dirname "$0")/.."
DEPTH=8 RATIO=20 SEED=42 RUNS=$HOME/nanochat-runs
read -r TAG _ SWITCH < <(runs/container.sh python -m nanochat.train.schedule "$DEPTH" "$RATIO" "$SEED" | tail -n 1)
rm -rf "$RUNS/base_checkpoints/$TAG" "$RUNS/metrics/$TAG.jsonl"
echo "$TAG: switch at step $SWITCH"
train=(runs/container.sh python -m scripts.base_train --depth "$DEPTH" --target-param-data-ratio "$RATIO" --seed "$SEED"
  --save-every "$SWITCH")
"${train[@]}" >"$HOME/smoke-1.log" 2>&1 &
pid=$!
until [[ -s $RUNS/metrics/$TAG.jsonl ]] && python3 -c "
import json, sys
steps = [r['step'] for r in map(json.loads, open('$RUNS/metrics/$TAG.jsonl')) if r['kind'] == 'train']
sys.exit(0 if steps and max(steps) >= $SWITCH + 8 else 1)"; do
  kill -0 "$pid" 2>/dev/null || { echo "first launch ended before the kill"; exit 1; }
  sleep 2
done
kill "$pid"
wait "$pid" || true
echo "killed; checkpoints: $(ls "$RUNS/base_checkpoints/$TAG")"
"${train[@]}" >"$HOME/smoke-2.log" 2>&1
echo "done"
runs/container.sh python -m tests.smoke_check "$TAG"
runs/container.sh python -m tests.decoder_check "$TAG" 32 64
