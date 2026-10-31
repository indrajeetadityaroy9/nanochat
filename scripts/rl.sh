#!/bin/bash
# RL with verifiable rewards from a post-trained checkpoint (docs/post-training.md §6): the trainer (onpolicy/rl.py,
# the training image: the native model) and the sampler (onpolicy/sampler.py, the vLLM image: the engines and the
# tests) on the GPU together for the trainer's time budget (onpolicy/rl.py BUDGET), exchanging policies and batches
# under /dev/shm/nanochat-rl-<step>, cleared first. The trainer resumes from rl/state.pt when present, the sampler from
# the policy the trainer publishes. Both containers run without a network (the tests run the model's code); either
# one's failure stops the other. Needs scripts/onpolicy.sh's export (onpolicy-<step>/model) and val prompts
# (onpolicy-<step>/eval-model/val).
#
#   scripts/rl.sh <run> <step>
#
# Output: RUNS/<run>/onpolicy-<step>/rl/: train.jsonl (a line per batch), evals.jsonl (KodCode val pass@1 of the
# first and the last policy), step-<k>/ (each evaluated policy, the export's format), state.pt.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_DIR=$ROOT/artifacts/runs/$1 STEP=$2
OUT=$RUN_DIR/onpolicy-$STEP W=/workspace/nanochat X=/dev/shm/nanochat-rl-$STEP
w() { echo "${1/#$ROOT/$W}"; }
for CHECKPOINT in "$RUN_DIR"/{checkpoint,weights}-"$STEP".pt ""; do [ -f "$CHECKPOINT" ] && break; done
[ -n "$CHECKPOINT" ] || { echo "no checkpoint or weights of step $STEP in $RUN_DIR" >&2; exit 1; }
[ -f "$OUT/model/model.safetensors" ] && [ -f "$OUT/eval-model/val/prompts.jsonl" ] ||
  { echo "no export or val prompts in $OUT: run scripts/onpolicy.sh first" >&2; exit 1; }
docker build -q -f "$ROOT/docker/Dockerfile.eval" -t nanochat-eval "$ROOT/docker" > /dev/null
docker build -q -f "$ROOT/docker/Dockerfile.onpolicy" -t nanochat-onpolicy "$ROOT/docker" > /dev/null
rm -rf "$X"
mkdir -p "$X" "$OUT/rl"
DOCKER=(docker run --rm --ipc=host --init --network none --ulimit core=0 --device nvidia.com/gpu=all
        --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_OFFLINE=1 -v "$ROOT":$W
        -v "$(realpath "$ROOT/artifacts")":$W/artifacts -v "$X":"$X" -w $W)
"${DOCKER[@]}" --name nanochat-rl-trainer --ulimit memlock=-1 --ulimit stack=67108864:-1 -e PYTHONPATH=$W nanochat \
  python -m onpolicy.rl "$(w "$CHECKPOINT")" "$(w "$OUT/model")" "$X" "$(w "$OUT/rl")" &
"${DOCKER[@]}" --name nanochat-rl-sampler -e XDG_CACHE_HOME=$W/artifacts/cache \
  -e HF_HOME=$W/artifacts/cache/huggingface -e VLLM_CACHE_ROOT=$W/artifacts/cache/vllm \
  -e PYTHONPATH=$W/eval/official_site:$W nanochat-onpolicy \
  python -m onpolicy.sampler "$(w "$OUT/model")" "$X" "$(w "$OUT/rl")" &
if ! wait -n; then
  docker rm -f nanochat-rl-trainer nanochat-rl-sampler > /dev/null 2>&1 || true
  wait || true
  exit 1
fi
wait
rm -rf "$X"
