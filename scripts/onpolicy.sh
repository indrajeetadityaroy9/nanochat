#!/bin/bash
# The on-policy stage of one post-trained checkpoint (onpolicy/, docs/post-training.md §8), each step skipped when its
# output exists (a rerun resumes):
#   1. the checkpoint exported with --chat (inference/export_hf.py, CPU) to RUNS/<run>/onpolicy-<step>/model;
#   2.-3. rollout waves (`waves`): PROMPTS distinct train-pool prompts per wave (default 8000; each wave excludes the
#      earlier waves' prompts) under SAMPLES (default 0.6:4) by onpolicy/rollouts.py on the GPU at
#      VLLM_GPU_MEMORY_UTILIZATION (default 0.85) to rollouts/wave-<k>/, and after each wave the selection over every
#      wave (onpolicy/selection.py: rollouts/ftpo.jsonl, rollouts/selection.json; the rows need no verdicts, so the
#      train rollouts are not verified); waves stop once the FTPO rows before rebalancing reach TARGET_ROWS (default
#      20,000: Antidoom's generation "stops after collecting 20k pairs", liquid.ai/blog/antidoom) or a wave finds
#      fewer than PROMPTS prompts left (the usable pool is exhausted);
#   5. FTPO from the export (onpolicy/train.py ftpo) to ftpo/;
#   6. the evaluation of each of EVAL_MODELS (default "ftpo model": ftpo/, the export; `evaluate`): the
#      natural-termination probe on the model directory (eval/official_termination.py, scored in a CPU container; for
#      the export, the step's own termination-<step>.json when it exists: the same prompts, seeds and generation) and
#      pass@1 on VAL_PROMPTS (default 256) held-out KodCode problems of the SFT val split, 4 samples each at T 0.6
#      (rollouts, verdicts and selection.json under eval-<model>/).
# STAGES (default "rollouts train evaluate") limits the steps after the export: "rollouts" is 2-3, "train" 5.
# CPU_SET (docker --cpuset-cpus, default all) confines every CPU container (export, selection, verification, scoring);
# verification runs as many workers as the set has cores.
#
# Output: RUNS/<run>/onpolicy-<step>/.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_DIR=$ROOT/artifacts/runs/$1 STEP=$2
PROMPTS=${PROMPTS:-8000} SAMPLES=${SAMPLES:-0.6:4} VAL_PROMPTS=${VAL_PROMPTS:-256} TARGET_ROWS=${TARGET_ROWS:-20000}
STAGES=${STAGES:-rollouts train evaluate} UTIL=${VLLM_GPU_MEMORY_UTILIZATION:-0.85}
for CHECKPOINT in "$RUN_DIR"/{checkpoint,weights}-"$STEP".pt ""; do [ -f "$CHECKPOINT" ] && break; done
[ -n "$CHECKPOINT" ] || { echo "no checkpoint or weights of step $STEP in $RUN_DIR" >&2; exit 1; }
OUT=$RUN_DIR/onpolicy-$STEP W=/workspace/nanochat
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
METRICS=()
[ -f "$RUN_DIR/metrics.jsonl" ] && METRICS=("$(w "$RUN_DIR/metrics.jsonl")" "$(date +%s)")
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/timings.log"; }
timed() {  # <name> <command...>
  local start=$SECONDS; log "start $1"; "${@:2}"; log "end $1: $((SECONDS - start)) s"
}

rollouts() {  # <model dir> <out dir> <rollouts args...>: rollouts, verdicts, selection
  [ -f "$2/meta.json" ] || timed "rollouts $(basename "$2")" "${GPU[@]}" nanochat-onpolicy \
    python -m onpolicy.rollouts "$(w "$1")" "$(w "$2")" "${@:3}"
  # the cpuset's cores: GNU nproc honours OMP_NUM_THREADS (1 in CPU containers), so it is unset for the count
  [ -f "$2/verdicts.jsonl" ] || timed "verify $(basename "$2")" "${CPU[@]}" nanochat-onpolicy \
    python -m onpolicy.verify "$(w "$2")" --workers "$("${CPU[@]}" nanochat-onpolicy env -u OMP_NUM_THREADS nproc)"
  [ -f "$2/selection.json" ] || "${CPU[@]}" nanochat-onpolicy python -m onpolicy.selection "$(w "$2")" > /dev/null
}

evaluate() {  # <model dir> <name>
  local out=$OUT/eval-$2
  mkdir -p "$out"
  if [ "$2" = model ] && [ -f "$RUN_DIR/termination-$STEP.json" ] && [ ! -f "$out/termination.json" ]; then
    cp "$RUN_DIR/termination-$STEP.json" "$out/termination.json"
  fi
  if [ ! -f "$out/termination.json" ]; then
    timed "termination $2" "${GPU[@]}" nanochat-onpolicy python -m eval.official_termination "$(w "$1")" \
      "$(w "$out/termination.json.generated")"
    "${CPU[@]}" nanochat-onpolicy python -m eval.official_termination --score "$(w "$out/termination.json.generated")" \
      "$(w "$out/termination.json")"
  fi
  rollouts "$1" "$out/val" --split val --prompts "$VAL_PROMPTS" --samples 0.6:4
}

waves() {  # rollout waves until TARGET_ROWS rows or the pool's end; the selection over all waves after each
  local k=0 rows prompts dir seen=()
  while :; do
    dir=$OUT/rollouts/wave-$k
    [ -f "$dir/meta.json" ] || timed "rollouts wave $k" "${GPU[@]}" nanochat-onpolicy python -m onpolicy.rollouts \
      "$(w "$OUT/model")" "$(w "$dir")" "${METRICS[@]}" --split train --prompts "$PROMPTS" --samples "$SAMPLES" \
      ${seen[@]:+--exclude "${seen[@]}"}
    seen+=("$(w "$dir")")
    "${CPU[@]}" nanochat-onpolicy python -m onpolicy.selection "$(w "$OUT/rollouts")" "${seen[@]}" > /dev/null
    rows=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["ftpo"]["rows_before_rebalance"])' \
      "$OUT/rollouts/selection.json")
    prompts=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["prompts"])' "$dir/meta.json")
    log "wave $k: $prompts prompts; $rows FTPO rows before rebalancing over waves 0-$k"
    [ "$rows" -lt "$TARGET_ROWS" ] && [ "$prompts" -ge "$PROMPTS" ] || break
    k=$((k + 1))
  done
}

[ -f "$OUT/model/model.safetensors" ] ||
  timed export "${CPU[@]}" nanochat python -m inference.export_hf "$(w "$CHECKPOINT")" "$(w "$OUT/model")" --chat
if [[ " $STAGES " == *" rollouts "* ]]; then
  waves
fi
if [[ " $STAGES " == *" train "* ]]; then
  [ -f "$OUT/ftpo/model.safetensors" ] || timed ftpo "${GPU[@]}" nanochat-onpolicy python -m onpolicy.train ftpo \
    "$(w "$OUT/model")" "$(w "$OUT/rollouts/ftpo.jsonl")" "$(w "$OUT/ftpo")"
fi
if [[ " $STAGES " == *" evaluate "* ]]; then
  for name in ${EVAL_MODELS:-ftpo model}; do
    if [ -f "$OUT/$name/model.safetensors" ]; then evaluate "$OUT/$name" "$name"; fi
  done
fi
log done
