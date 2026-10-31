#!/bin/bash
# The natural-termination probe of one post-trained checkpoint (eval/official_termination.py; docs/post-training.md
# §5), also beside a running training job: the checkpoint exported with --chat to a temporary directory (CPU), then
# sampled by vLLM offline on the GPU with gpu_memory_utilization VLLM_GPU_MEMORY_UTILIZATION (default 0.12), one
# probe at a time (a lock); skipped (exit 2) if the GPU has less than MIN_FREE_MIB (default 15,360) free at its start.
# The run's metrics.jsonl gives its tokens/s during the probe against before it and the time it lost, as information.
# The responses are scored afterwards in a CPU container of their own (single-threaded BLAS), never in the vLLM process;
# the unscored responses stay in RUNS/<run>/<output>.generated if scoring fails.
# scripts/official_reasoning.sh runs the same probe on its export before serving. With --budget, the forced
# thinking-budget probe and the closure's sensitivity to sampling instead (eval/official_termination.py --budget);
# with --dry, the same under the protocol's sampling with and without DRY (eval/dry.py).
#
#   [TAG=<suffix>] scripts/official_termination.sh [--budget | --dry] <run name> <step> [<prompts per set, default 64>]
#
# Output: RUNS/<run>/termination-<step>.json; with --budget, RUNS/<run>/termination-budget-<step>.json; with --dry,
# RUNS/<run>/termination-dry-<step>.json; each name with -$TAG before .json if TAG is set.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODE=() KIND=termination
case "${1:-}" in
  --budget) MODE=(--budget) KIND=termination-budget; shift ;;
  --dry) MODE=(--budget --configs protocol,dry) KIND=termination-dry; shift ;;
esac
RUN_DIR=$ROOT/artifacts/runs/$1 STEP=$2 PROMPTS=${3:-64}
for CHECKPOINT in "$RUN_DIR"/{checkpoint,weights}-"$STEP".pt ""; do [ -f "$CHECKPOINT" ] && break; done
[ -n "$CHECKPOINT" ] || { echo "no checkpoint or weights of step $STEP in $RUN_DIR" >&2; exit 1; }
OUT=$RUN_DIR/$KIND-$STEP${TAG:+-$TAG}.json W=/workspace/nanochat
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
"${DOCKER[@]}" --device nvidia.com/gpu=all -e VLLM_CACHE_ROOT=$W/artifacts/cache/vllm \
  -e PYTHONPATH=$W/eval/official_site:$W -e VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.12}" \
  nanochat-eval python -m eval.official_termination "${MODEL/#$ROOT/$W}" "${OUT/#$ROOT/$W}.generated" "${METRICS[@]}" \
  --prompts "$PROMPTS" "${MODE[@]}"
"${DOCKER[@]}" -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 nanochat-eval \
  python -m eval.official_termination --score "${OUT/#$ROOT/$W}.generated" "${OUT/#$ROOT/$W}.partial"
mv "$OUT.partial" "$OUT"
rm "$OUT.generated"
