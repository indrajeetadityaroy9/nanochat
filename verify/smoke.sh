#!/bin/bash
# Two-stage smoke on the H100: d8 at 20 tokens per scaling parameter, a checkpoint every repo_start steps (one at the
# switch to the repo-level stage). The first launch is killed 8 steps after the switch; the relaunch resumes from the
# switch checkpoint in the repo-level stage and finishes.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=d8-r20-s42 RUNS=$HOME/nanochat-runs
rm -rf "$RUNS/base_checkpoints/$TAG" "$RUNS/metrics/$TAG.jsonl" "$RUNS/base_eval/$TAG.json"
SWITCH=$(runs/container.sh python -c "from nanochat.data.tokenizer import get_tokenizer; \
from nanochat.train.schedule import plan; print(plan(8, 20, get_tokenizer().vocab_size).repo_start)" | tail -n 1)
echo "switch at step $SWITCH"
train() {
  runs/container.sh python -m scripts.base_train --depth 8 --target-param-data-ratio 20 --seed 42 \
    --save-every "$SWITCH"
}
train >"$HOME/smoke-1.log" 2>&1 &
pid=$!
until [[ -s $RUNS/metrics/$TAG.jsonl ]] && python3 -c "
import json, sys
steps = [r['step'] for r in map(json.loads, open('$RUNS/metrics/$TAG.jsonl')) if r['kind'] == 'train']
sys.exit(0 if steps and max(steps) >= $SWITCH + 8 else 1)"; do
  kill -0 "$pid" 2>/dev/null || { echo "first launch ended before the kill"; exit 1; }
  sleep 2
done
docker kill $(docker ps -q) >/dev/null
wait || true
echo "killed; checkpoints: $(ls "$RUNS/base_checkpoints/$TAG")"
train >"$HOME/smoke-2.log" 2>&1
echo "done"
