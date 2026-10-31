#!/bin/bash
# The natural-termination probe of one post-trained checkpoint (eval/official_termination.py; docs/post-training.md
# §5), also beside a running training job: the checkpoint exported with --chat to a temporary directory (CPU), then
# sampled by vLLM offline on the GPU with gpu_memory_utilization VLLM_GPU_MEMORY_UTILIZATION (default 0.12), one
# probe at a time (a lock); skipped (exit 2) if the GPU has less than MIN_FREE_MIB (default 15,360) free at its start.
# The run's metrics.jsonl gives its tokens/s during the probe against before it and the time it lost, as information.
# scripts/official_reasoning.sh runs the same probe on its export before serving.
#
#   scripts/official_termination.sh <run name> <step> [<prompts per set, default 64>]
#
# Output: RUNS/<run>/termination-<step>.json.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_DIR=$ROOT/artifacts/runs/$1 STEP=$2 PROMPTS=${3:-64}
for CHECKPOINT in "$RUN_DIR"/{checkpoint,weights}-"$STEP".pt ""; do [ -f "$CHECKPOINT" ] && break; done
[ -n "$CHECKPOINT" ] || { echo "no checkpoint or weights of step $STEP in $RUN_DIR" >&2; exit 1; }
OUT=$RUN_DIR/termination-$STEP.json W=/workspace/nanochat
docker build -q -f "$ROOT/docker/Dockerfile.eval" -t nanochat-eval "$ROOT/docker" > /dev/null
DOCKER=(docker run --rm --ipc=host --init --network host --ulimit core=0 --user "$(id -u):$(id -g)" -e HOME=/tmp
        -e PYTHONPATH=$W -v "$ROOT":$W -v "$(realpath "$ROOT/artifacts")":$W/artifacts -w $W)

mkdir -p "$ROOT/artifacts/cache"
exec 9> "$ROOT/artifacts/cache/termination.lock"
flock 9
SINCE=$(date +%s)
MODEL=$(mktemp -d "$ROOT/artifacts/cache/termination-$STEP-XXXXXX")
trap 'rm -rf "$MODEL" "$OUT.partial"' EXIT
"${DOCKER[@]}" --cpuset-mems="${CPU_NODE:-0}" nanochat python -m inference.export_hf "${CHECKPOINT/#$ROOT/$W}" \
  "${MODEL/#$ROOT/$W}" --chat
free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1 - $2}')
[ "$free" -ge "${MIN_FREE_MIB:-15360}" ] || { echo "GPU memory free $free MiB: probe skipped" >&2; exit 2; }
METRICS=()
[ -f "$RUN_DIR/metrics.jsonl" ] && METRICS=("${RUN_DIR/#$ROOT/$W}/metrics.jsonl" "$SINCE")
"${DOCKER[@]}" --device nvidia.com/gpu=all -e VLLM_CACHE_ROOT=$W/artifacts/cache/vllm -e OPENBLAS_NUM_THREADS=1 \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e PYTHONPATH=$W/eval/official_site:$W -e VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.12}" \
  nanochat-eval python -m eval.official_termination "${MODEL/#$ROOT/$W}" "${OUT/#$ROOT/$W}.partial" "${METRICS[@]}" \
  --prompts "$PROMPTS"
mv "$OUT.partial" "$OUT"
