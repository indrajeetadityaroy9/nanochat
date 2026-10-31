#!/bin/bash
# The SFT model's export and evaluation before RL (onpolicy/, docs/post-training.md §5), each step skipped when its
# output exists (a rerun resumes):
#   1. the checkpoint exported with --chat (inference/export_hf.py, CPU) to RUNS/<run>/onpolicy-<step>/model;
#   2. its evaluation to eval-model/: the natural-termination probe on PROBE_PROMPTS (default 128) prompts per set
#      (eval/official_termination.py, scored in a CPU container), or the step's own probe,
#      RUNS/<run>/termination-<step>[-p<PROBE_PROMPTS>].json, when it has as many prompts (the same prompts, seeds and
#      generation); then pass@1 on VAL_PROMPTS (default 512) held-out KodCode problems of the SFT val split, 4 samples
#      each at T 0.6 (onpolicy/rollouts.py, verified by onpolicy/verify.py, summarized by onpolicy/outcomes.py: the
#      rollouts, verdicts and outcomes.json under eval-model/val), whose prompts and seeds scripts/rl.sh's evaluations
#      reuse.
# CPU_SET (docker --cpuset-cpus, default all) confines every CPU container (export, verification, scoring);
# verification runs as many workers as the set has cores.
#
#   [VLLM_GPU_MEMORY_UTILIZATION=0.85] scripts/onpolicy.sh <run> <step>
#
# Output: RUNS/<run>/onpolicy-<step>/: model/, eval-model/, timings.log.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_DIR=$ROOT/artifacts/runs/$1 STEP=$2
VAL_PROMPTS=${VAL_PROMPTS:-512} PROBE_PROMPTS=${PROBE_PROMPTS:-128} UTIL=${VLLM_GPU_MEMORY_UTILIZATION:-0.85}
for CHECKPOINT in "$RUN_DIR"/{checkpoint,weights}-"$STEP".pt ""; do [ -f "$CHECKPOINT" ] && break; done
[ -n "$CHECKPOINT" ] || { echo "no checkpoint or weights of step $STEP in $RUN_DIR" >&2; exit 1; }
OUT=$RUN_DIR/onpolicy-$STEP W=/workspace/nanochat
EVAL=$OUT/eval-model
w() { echo "${1/#$ROOT/$W}"; }
docker build -q -f "$ROOT/docker/Dockerfile.eval" -t nanochat-eval "$ROOT/docker" > /dev/null
docker build -q -f "$ROOT/docker/Dockerfile.onpolicy" -t nanochat-onpolicy "$ROOT/docker" > /dev/null
DOCKER=(docker run --rm --ipc=host --init --ulimit core=0 --user "$(id -u):$(id -g)" -e HOME=/tmp
        -e XDG_CACHE_HOME=$W/artifacts/cache -e HF_HOME=$W/artifacts/cache/huggingface -e PYTHONPATH=$W -v "$ROOT":$W
        -v "$(realpath "$ROOT/artifacts")":$W/artifacts -w $W)
CPU=("${DOCKER[@]}" --network none --cpuset-mems="${CPU_NODE:-0}" ${CPU_SET:+--cpuset-cpus "$CPU_SET"}
     -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1)
GPU=("${DOCKER[@]}" --network host --device nvidia.com/gpu=all -e VLLM_CACHE_ROOT=$W/artifacts/cache/vllm
     -e PYTHONPATH=$W/eval/official_site:$W -e VLLM_GPU_MEMORY_UTILIZATION="$UTIL")
mkdir -p "$EVAL"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/timings.log"; }
timed() {  # <name> <command...>
  local start=$SECONDS; log "start $1"; "${@:2}"; log "end $1: $((SECONDS - start)) s"
}
prompts_per_set() {  # <probe record>: its prompts per set
  python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["prompts_per_set"])' "$1"
}

[ -f "$OUT/model/model.safetensors" ] ||
  timed export "${CPU[@]}" nanochat python -m inference.export_hf "$(w "$CHECKPOINT")" "$(w "$OUT/model")" --chat
if [ ! -f "$EVAL/termination.json" ]; then
  for probe in "$RUN_DIR/termination-$STEP.json" "$RUN_DIR/termination-$STEP-p$PROBE_PROMPTS.json"; do
    if [ -f "$probe" ] && [ "$(prompts_per_set "$probe")" = "$PROBE_PROMPTS" ]; then
      cp "$probe" "$EVAL/termination.json"
      break
    fi
  done
fi
if [ ! -f "$EVAL/termination.json" ]; then
  timed termination "${GPU[@]}" nanochat-onpolicy python -m eval.official_termination "$(w "$OUT/model")" \
    "$(w "$EVAL/termination.json.generated")" --prompts "$PROBE_PROMPTS"
  "${CPU[@]}" nanochat-onpolicy python -m eval.official_termination --score "$(w "$EVAL/termination.json.generated")" \
    "$(w "$EVAL/termination.json")"
fi
[ -f "$EVAL/val/meta.json" ] || timed "val rollouts" "${GPU[@]}" nanochat-onpolicy python -m onpolicy.rollouts \
  "$(w "$OUT/model")" "$(w "$EVAL/val")" --prompts "$VAL_PROMPTS"
[ -f "$EVAL/val/verdicts.jsonl" ] || timed "val verify" "${CPU[@]}" nanochat-onpolicy python -m onpolicy.verify \
  "$(w "$EVAL/val")"
[ -f "$EVAL/val/outcomes.json" ] || "${CPU[@]}" nanochat-onpolicy python -m onpolicy.outcomes "$(w "$EVAL/val")" \
  > /dev/null
log done
