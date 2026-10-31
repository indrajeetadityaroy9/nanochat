#!/bin/bash
# The serving path's faithfulness at long context for one post-trained checkpoint (eval/serving_check.py): the
# checkpoint exported with --chat to a temporary directory (CPU); the training model's per-token NLL of 8 SFT val
# documents (nanochat image, GPU); vLLM's prompt logprobs of the same ids and its greedy generations from 4 val prompts
# (nanochat-eval image, GPU, offline, gpu_memory_utilization VLLM_GPU_MEMORY_UTILIZATION, default 0.12); the training
# model's teacher-forced forward on those generations (nanochat image). One GPU probe at a time (the termination lock).
#
#   scripts/serving_check.sh <run name> <step>
#
# Output: RUNS/<run>/serving-check-<step>.json.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_DIR=$ROOT/artifacts/runs/$1 STEP=$2
CHECKPOINT=$RUN_DIR/checkpoint-$STEP.pt
[ -f "$CHECKPOINT" ] || CHECKPOINT=$RUN_DIR/weights-$STEP.pt
[ -f "$CHECKPOINT" ] || { echo "no checkpoint or weights of step $STEP in $RUN_DIR" >&2; exit 1; }
OUT=$RUN_DIR/serving-check-$STEP.json W=/workspace/nanochat
docker build -q -f "$ROOT/docker/Dockerfile.eval" -t nanochat-eval "$ROOT/docker" > /dev/null
DOCKER=(docker run --rm --ipc=host --init --network host --ulimit core=0 --user "$(id -u):$(id -g)" -e HOME=/tmp
        -e PYTHONPATH=$W -v "$ROOT":$W -v "$(realpath "$ROOT/artifacts")":$W/artifacts -w $W)

mkdir -p "$ROOT/artifacts/cache"
exec 9> "$ROOT/artifacts/cache/termination.lock"
flock 9
MODEL=$(mktemp -d "$ROOT/artifacts/cache/serving-check-$STEP-XXXXXX")
trap 'rm -rf "$MODEL" "$OUT.partial"' EXIT
"${DOCKER[@]}" --cpuset-mems="${CPU_NODE:-0}" nanochat python -m inference.export_hf "${CHECKPOINT/#$ROOT/$W}" \
  "${MODEL/#$ROOT/$W}" --chat
# each stage's output kept if a later one fails
[ -f "$OUT.train" ] || "${DOCKER[@]}" --device nvidia.com/gpu=all nanochat python -m eval.serving_check train "$1" \
  "$STEP" "${OUT/#$ROOT/$W}.train"
[ -f "$OUT.vllm" ] || "${DOCKER[@]}" --device nvidia.com/gpu=all -e VLLM_CACHE_ROOT=$W/artifacts/cache/vllm \
  -e PYTHONPATH=$W/eval/official_site:$W -e VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.12}" \
  nanochat-eval python -m eval.serving_check vllm "${MODEL/#$ROOT/$W}" "${OUT/#$ROOT/$W}.train" \
  "${OUT/#$ROOT/$W}.vllm"
"${DOCKER[@]}" --device nvidia.com/gpu=all nanochat python -m eval.serving_check decode "$1" "$STEP" \
  "${OUT/#$ROOT/$W}.vllm" "${OUT/#$ROOT/$W}.partial"
mv "$OUT.partial" "$OUT"
rm "$OUT.train" "$OUT.vllm"
