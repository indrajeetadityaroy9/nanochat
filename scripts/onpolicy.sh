#!/bin/bash
# The on-policy stage of one post-trained checkpoint (onpolicy/, docs/post-training.md §8), each step skipped when its
# output exists (a rerun resumes):
#   1. the checkpoint exported with --chat (inference/export_hf.py, CPU) to RUNS/<run>/onpolicy-<step>/model;
#   2.-3. rollout waves (`waves`): PROMPTS distinct prompts per wave (default 6,000) of the mining pool, every
#      source's SFT train contexts (onpolicy/prompts.py --mix, built here if absent), under SAMPLES (default 0.6:1) by
#      onpolicy/rollouts.py to rollouts/wave-<k>/, in groups of ENGINES (default 4) concurrent engines (`engines`:
#      wave k seeded k and excluding waves 0..k-1's prompts, each at VLLM_GPU_MEMORY_UTILIZATION (default 0.85) /
#      ENGINES, each started once the previous one has written its prompts and logged "KV cache size", vLLM's
#      memory profiling; vLLM's engine core is CPU-bound, so one engine leaves the GPU idle), and after each group
#      the selection over every wave (onpolicy/selection.py: rollouts/ftpo.jsonl, rollouts/selection.json; the rows
#      need no verdicts, so the train rollouts are not verified); groups stop once the FTPO rows before rebalancing
#      reach TARGET_ROWS (default 8,000, docs/post-training.md §8.7 Waves) or the group's last wave finds fewer than
#      PROMPTS prompts left (the pool is exhausted);
#   4. FTPO from the export (onpolicy/train.py ftpo, FREEZE none or head, default none) to ftpo/: one model, written
#      when the held-out chosen_win reaches 0.35 or at the epoch's end, and ftpo/train-log.json;
#   5. DPO waves (`dpo_waves`): PROMPTS_DPO distinct KodCode train-pool prompts per wave (default 8,000) x 4 samples
#      at T 0.6 from ftpo/ to dpo-rollouts/wave-<k>/, in groups of at most ENGINES concurrent engines as in 2.-3.,
#      verified (onpolicy/verify.py), and after each group the selection over every DPO wave (dpo-rollouts/dpo.jsonl,
#      dpo-rollouts/selection.json); groups stop once the pairs reach DPO_PAIRS (default 8,000, DeLoop's count),
#      after DPO_WAVES (default 3) waves, or when the group's last wave finds fewer than PROMPTS_DPO prompts left;
#      then DPO from ftpo/ on dpo.jsonl (onpolicy/train.py dpo) to dpo/;
#   6. the evaluation (`evaluate`) of each of model/, ftpo/ and dpo/ present, or only of EVAL_MODELS (names: model,
#      ftpo, dpo) when set, to eval-<name>/: the natural-termination probe on PROBE_PROMPTS (default 128) prompts per
#      set (eval/official_termination.py, scored in a CPU container; for the export, the step's own probe,
#      RUNS/<run>/termination-<step>[-p<PROBE_PROMPTS>].json, when it has as many prompts: the same prompts, seeds and
#      generation); pass@1 on VAL_PROMPTS (default 512) held-out KodCode problems of the SFT val split, 4 samples each
#      at T 0.6 (rollouts, verdicts and selection.json under eval-<name>/val).
# STAGES (default "rollouts train dpo evaluate") limits the steps after the export: "rollouts" is 2-3, "train" 4,
# "dpo" 5, "evaluate" 6.
# CPU_SET (docker --cpuset-cpus, default all) confines every CPU container (export, pool, selection, verification,
# scoring); verification runs as many workers as the set has cores.
#
# Output: RUNS/<run>/onpolicy-<step>/.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_DIR=$ROOT/artifacts/runs/$1 STEP=$2
PROMPTS=${PROMPTS:-6000} SAMPLES=${SAMPLES:-0.6:1} VAL_PROMPTS=${VAL_PROMPTS:-512} TARGET_ROWS=${TARGET_ROWS:-8000}
PROBE_PROMPTS=${PROBE_PROMPTS:-128} FREEZE=${FREEZE:-none} ENGINES=${ENGINES:-4}
PROMPTS_DPO=${PROMPTS_DPO:-8000} DPO_PAIRS=${DPO_PAIRS:-8000} DPO_WAVES=${DPO_WAVES:-3}
STAGES=${STAGES:-rollouts train dpo evaluate} UTIL=${VLLM_GPU_MEMORY_UTILIZATION:-0.85}
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
VLLM=("${DOCKER[@]}" --network host --device nvidia.com/gpu=all -e VLLM_CACHE_ROOT=$W/artifacts/cache/vllm
      -e PYTHONPATH=$W/eval/official_site:$W)
GPU=("${VLLM[@]}" -e VLLM_GPU_MEMORY_UTILIZATION="$UTIL")
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

verify() {  # <rollouts dir>: its verdicts unless written
  # the cpuset's cores: GNU nproc honours OMP_NUM_THREADS (1 in CPU containers), so it is unset for the count
  [ -f "$1/verdicts.jsonl" ] || timed "verify $(basename "$1")" "${CPU[@]}" nanochat-onpolicy \
    python -m onpolicy.verify "$(w "$1")" --workers "$("${CPU[@]}" nanochat-onpolicy env -u OMP_NUM_THREADS nproc)"
}

rollouts() {  # <model dir> <out dir> <rollouts args...>: rollouts, verdicts, selection
  [ -f "$2/meta.json" ] || timed "rollouts $(basename "$2")" "${GPU[@]}" nanochat-onpolicy \
    python -m onpolicy.rollouts "$(w "$1")" "$(w "$2")" "${@:3}"
  verify "$2"
  [ -f "$2/selection.json" ] || "${CPU[@]}" nanochat-onpolicy python -m onpolicy.selection "$(w "$2")" > /dev/null
}

engines() {  # <model dir> <waves dir> <first wave> <waves> <rollouts args...>: the waves without meta.json, one
  # engine each at UTIL / ENGINES, concurrently; wave k seeded k, excluding waves 0..k-1's prompts; each engine
  # started once the previous one has written its prompts and logged "KV cache size" (vLLM's memory profiling)
  local k dir share prev="" pid="" pids=() exclude=() failed=0 start=$SECONDS
  share=$(python3 -c 'import sys; print(float(sys.argv[1]) / int(sys.argv[2]))' "$UTIL" "$ENGINES")
  for ((k = 0; k < $3; k++)); do exclude+=("$(w "$2/wave-$k")"); done
  for ((k = $3; k < $3 + $4; k++)); do
    dir=$2/wave-$k
    if [ ! -f "$dir/meta.json" ]; then
      if [ -n "$pid" ]; then
        until grep -qsF "KV cache size" "$prev/rollouts.log" || ! kill -0 "$pid" 2> /dev/null; do sleep 5; done
        [ -f "$prev/prompts.jsonl" ] || { log "$(basename "$prev") wrote no prompts.jsonl"; wait; return 1; }
      fi
      mkdir -p "$dir"
      log "start rollouts $(basename "$2")/wave-$k at $share"
      "${VLLM[@]}" -e VLLM_GPU_MEMORY_UTILIZATION="$share" nanochat-onpolicy python -m onpolicy.rollouts \
        "$(w "$1")" "$(w "$dir")" "${@:5}" --seed "$k" ${exclude[@]:+--exclude "${exclude[@]}"} \
        > "$dir/rollouts.log" 2>&1 &
      pid=$! prev=$dir
      pids+=("$pid")
    fi
    exclude+=("$(w "$dir")")
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  log "end rollouts $(basename "$2")/wave-$3..$(($3 + $4 - 1)): exit $failed, $((SECONDS - start)) s"
  return "$failed"
}

selected() {  # <waves dir> <waves>: the selection over waves 0..<waves>-1 into <waves dir>
  local k seen=()
  for ((k = 0; k < $2; k++)); do seen+=("$(w "$1/wave-$k")"); done
  "${CPU[@]}" nanochat-onpolicy python -m onpolicy.selection "$(w "$1")" "${seen[@]}" > /dev/null
}

json() {  # <file> <key> [<key>]: the value
  python3 -c 'import json, sys; d = json.load(open(sys.argv[1]))
for k in sys.argv[2:]: d = d[k]
print(d)' "$@"
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
}

waves() {  # groups of ENGINES mining waves until TARGET_ROWS rows or the pool's end; the selection after each
  local k=0 rows prompts
  "${CPU[@]}" nanochat-onpolicy python -c 'import sys; from onpolicy.prompts import pool_path
sys.exit(not pool_path("mix").exists())' ||
    timed pool "${CPU[@]}" nanochat-onpolicy python -m onpolicy.prompts --mix
  while :; do
    engines "$OUT/model" "$OUT/rollouts" "$k" "$ENGINES" "${METRICS[@]}" --split mix --prompts "$PROMPTS" \
      --samples "$SAMPLES"
    k=$((k + ENGINES))
    selected "$OUT/rollouts" "$k"
    rows=$(json "$OUT/rollouts/selection.json" ftpo rows_before_rebalance)
    prompts=$(json "$OUT/rollouts/wave-$((k - 1))/meta.json" prompts)
    log "waves 0-$((k - 1)): $rows FTPO rows before rebalancing; the last wave $prompts prompts"
    [ "$rows" -lt "$TARGET_ROWS" ] && [ "$prompts" -ge "$PROMPTS" ] || break
  done
}

dpo_waves() {  # verified KodCode train waves from ftpo/, in groups of at most ENGINES, until DPO_PAIRS pairs,
  # DPO_WAVES waves or the pool's end
  local k=0 n j pairs prompts
  while :; do
    n=$((DPO_WAVES - k < ENGINES ? DPO_WAVES - k : ENGINES))
    engines "$OUT/ftpo" "$OUT/dpo-rollouts" "$k" "$n" --split train --prompts "$PROMPTS_DPO" --samples 0.6:4
    for ((j = k; j < k + n; j++)); do verify "$OUT/dpo-rollouts/wave-$j"; done
    k=$((k + n))
    selected "$OUT/dpo-rollouts" "$k"
    pairs=$(json "$OUT/dpo-rollouts/selection.json" dpo pairs)
    prompts=$(json "$OUT/dpo-rollouts/wave-$((k - 1))/meta.json" prompts)
    log "dpo waves 0-$((k - 1)): $pairs DPO pairs; the last wave $prompts prompts"
    [ "$pairs" -lt "$DPO_PAIRS" ] && [ "$prompts" -ge "$PROMPTS_DPO" ] && [ "$k" -lt "$DPO_WAVES" ] || break
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
if [[ " $STAGES " == *" dpo "* ]] && [ -f "$OUT/ftpo/train-log.json" ] && [ ! -f "$OUT/dpo/train-log.json" ]; then
  dpo_waves
  timed dpo "${GPU[@]}" nanochat-onpolicy python -m onpolicy.train dpo "$(w "$OUT/ftpo")" \
    "$(w "$OUT/dpo-rollouts/dpo.jsonl")" "$(w "$OUT/dpo")"
fi
if [[ " $STAGES " == *" evaluate "* ]]; then
  for name in model ftpo dpo; do
    if [ -f "$OUT/$name/model.safetensors" ] && { [ -z "${EVAL_MODELS:-}" ] || [[ " $EVAL_MODELS " == *" $name "* ]]; }
    then
      evaluate "$OUT/$name" "$name"
    fi
  done
fi
log done
