#!/bin/bash
# The on-policy stage of one post-trained checkpoint (onpolicy/, docs/post-training.md §8), each step skipped when its
# output exists (a rerun resumes):
#   1. the checkpoint exported with --chat (inference/export_hf.py, CPU) to RUNS/<run>/onpolicy-<step>/model;
#   2.-3. rollout waves (`waves`): PROMPTS distinct prompts per wave (default 24,000) of the mining pool, every
#      source's SFT train contexts (onpolicy/prompts.py --mix, built here if absent; each wave excludes the earlier
#      waves' prompts) under SAMPLES (default 0.6:1) by onpolicy/rollouts.py on the GPU at VLLM_GPU_MEMORY_UTILIZATION
#      (default 0.85) to rollouts/wave-<k>/, and after each wave the selection over every wave (onpolicy/selection.py:
#      rollouts/ftpo.jsonl, rollouts/selection.json; the rows need no verdicts, so the train rollouts are not
#      verified); waves stop once the FTPO rows before rebalancing reach TARGET_ROWS (default 20,000: Antidoom's
#      generation "stops after collecting 20k pairs", liquid.ai/blog/antidoom) or a wave finds fewer than PROMPTS
#      prompts left (the pool is exhausted);
#   5. FTPO from the export (onpolicy/train.py ftpo, FREEZE none or head, default none) to ftpo/: the candidates
#      ftpo/ftpo-<threshold>/ (and ftpo/ftpo-final/) and ftpo/train-log.json;
#   6. the evaluation (`evaluate`) of the export (model/) and of every candidate present, or only of EVAL_MODELS
#      (names: model, ftpo-0.15, ...) when set, to eval-<name>/: the natural-termination probe on PROBE_PROMPTS
#      (default 128) prompts per set (eval/official_termination.py, scored in a CPU container; for the export, the
#      step's own probe, RUNS/<run>/termination-<step>[-p<PROBE_PROMPTS>].json, when it has as many prompts: the same
#      prompts, seeds and generation); pass@1 on VAL_PROMPTS (default 512) held-out KodCode problems of the SFT val
#      split, 4 samples each at T 0.6 (rollouts, verdicts and selection.json under eval-<name>/val); and the NLL of the
#      export's passing val responses (onpolicy/train.py nll on eval-model/val, eval-<name>/nll.json);
#   7. the pick (onpolicy/pick.py, CPU): pick.json, the candidate kept by the pre-registered rule, or none.
# STAGES (default "rollouts train evaluate pick") limits the steps after the export: "rollouts" is 2-3, "train" 5.
# CPU_SET (docker --cpuset-cpus, default all) confines every CPU container (export, pool, selection, verification,
# scoring, pick); verification runs as many workers as the set has cores.
#
# Output: RUNS/<run>/onpolicy-<step>/.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_DIR=$ROOT/artifacts/runs/$1 STEP=$2
PROMPTS=${PROMPTS:-24000} SAMPLES=${SAMPLES:-0.6:1} VAL_PROMPTS=${VAL_PROMPTS:-512} TARGET_ROWS=${TARGET_ROWS:-20000}
PROBE_PROMPTS=${PROBE_PROMPTS:-128} FREEZE=${FREEZE:-none}
STAGES=${STAGES:-rollouts train evaluate pick} UTIL=${VLLM_GPU_MEMORY_UTILIZATION:-0.85}
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
prompts_per_set() {  # <probe record>: its prompts per set
  python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["prompts_per_set"])' "$1"
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
  local out=$OUT/eval-$2 probe
  mkdir -p "$out"
  if [ "$2" = model ] && [ ! -f "$out/termination.json" ]; then
    for probe in "$RUN_DIR/termination-$STEP.json" "$RUN_DIR/termination-$STEP-p$PROBE_PROMPTS.json"; do
      if [ -f "$probe" ] && [ "$(prompts_per_set "$probe")" = "$PROBE_PROMPTS" ]; then
        cp "$probe" "$out/termination.json"
        break
      fi
    done
  fi
  if [ ! -f "$out/termination.json" ]; then
    timed "termination $2" "${GPU[@]}" nanochat-onpolicy python -m eval.official_termination "$(w "$1")" \
      "$(w "$out/termination.json.generated")" --prompts "$PROBE_PROMPTS"
    "${CPU[@]}" nanochat-onpolicy python -m eval.official_termination --score "$(w "$out/termination.json.generated")" \
      "$(w "$out/termination.json")"
  fi
  rollouts "$1" "$out/val" --split val --prompts "$VAL_PROMPTS" --samples 0.6:4
  [ -f "$out/nll.json" ] || timed "nll $2" "${GPU[@]}" nanochat-onpolicy python -m onpolicy.train nll "$(w "$1")" \
    "$(w "$OUT/eval-model/val")" "$(w "$out/nll.json")"
}

waves() {  # rollout waves until TARGET_ROWS rows or the pool's end; the selection over all waves after each
  local k=0 rows prompts dir seen=()
  "${CPU[@]}" nanochat-onpolicy python -c 'import sys; from onpolicy.prompts import pool_path
sys.exit(not pool_path("mix").exists())' ||
    timed pool "${CPU[@]}" nanochat-onpolicy python -m onpolicy.prompts --mix
  while :; do
    dir=$OUT/rollouts/wave-$k
    [ -f "$dir/meta.json" ] || timed "rollouts wave $k" "${GPU[@]}" nanochat-onpolicy python -m onpolicy.rollouts \
      "$(w "$OUT/model")" "$(w "$dir")" "${METRICS[@]}" --split mix --prompts "$PROMPTS" --samples "$SAMPLES" \
      --seed "$k" ${seen[@]:+--exclude "${seen[@]}"}
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
  [ -f "$OUT/ftpo/train-log.json" ] || timed ftpo "${GPU[@]}" nanochat-onpolicy python -m onpolicy.train ftpo \
    "$(w "$OUT/model")" "$(w "$OUT/rollouts/ftpo.jsonl")" "$(w "$OUT/ftpo")" --freeze "$FREEZE"
fi
if [[ " $STAGES " == *" evaluate "* ]]; then
  for dir in "$OUT/model" "$OUT"/ftpo/ftpo-*; do
    name=$(basename "$dir")
    if [ -f "$dir/model.safetensors" ] && { [ -z "${EVAL_MODELS:-}" ] || [[ " $EVAL_MODELS " == *" $name "* ]]; }; then
      evaluate "$dir" "$name"
    fi
  done
fi
if [[ " $STAGES " == *" pick "* ]] && compgen -G "$OUT/ftpo/ftpo-*/model.safetensors" > /dev/null; then
  [ -f "$OUT/pick.json" ] || "${CPU[@]}" nanochat-onpolicy python -m onpolicy.pick "$(w "$OUT")"
fi
log done
