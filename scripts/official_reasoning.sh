#!/bin/bash
# The reasoning protocol of a post-trained checkpoint (docs/post-training.md §5) by the benchmarks' official tools:
# the checkpoint exported with the SFT's chat format (inference/export_hf.py --chat), served by `vllm serve` (vLLM
# 0.32.0's Transformers backend, the head in fp32) with the "nanochat" reasoning parser (eval/official_reasoning_parser.py:
# the answer after the last "</think>"); EvalPlus's OpenAI backend, greedy (n 1, T 0) and sampled (its chat prompt,
# n 20, T 0.6, top-p 0.95 set by its openai_request.make_request), each sample's ending recorded
# (eval/official_evalplus.py: its finish reason and whether "</think>" closed), and lcb_runner (eval/official_lcb.py:
# LiveCodeBench v6, contest dates 2024-08-01 to 2025-05-01, n 16, T 0.6, top-p 0.95), each judged by its own
# evaluator. max_tokens 32,768 in every request: vLLM caps it at the context less the prompt
# (api_utils.get_max_tokens). First, the natural-termination probe (eval/official_termination.py) on the export; then
# every prompt either tool will send is checked for "\r" and the first's served ids against the SFT context
# (eval/official_tokenize.py). The parser is registered in every server process by eval/official_site/sitecustomize.py
# (NANOCHAT_REASONING_PARSER). The server stops loops as the probe does (eval/official_loop_stop.py: a response with
# no "</think>" whose last 4,096 tokens repeat, checked every 2,048, is ended, an empty answer either way), each stop
# a line of loop-stops.jsonl; LOOP_STOP=0 serves without it.
#
#   [MODEL=<model dir> TAG=<name>] scripts/official_reasoning.sh <run name> <step> [<problems per benchmark, a smoke>]
#
# Outputs in RUNS/<run>/official-<step>/: model/, tokenize-{evalplus,lcb}.json, evalplus/ (each samples file with its
# .endings.jsonl), lcb/, loop-stops.jsonl, server.log, timings.tsv, summary.json (eval/official_reasoning.py: EvalPlus's
# closure rate, loop stops, pass@1 and pass@1 given closure; both tools' loop stops); RUNS/<run>/termination-<step>.json
# (generated, then scored in a CPU container).
# With MODEL and TAG, a model directory in the export's format (onpolicy/train.py's) is evaluated in place of the
# step's export, its outputs in RUNS/<run>/official-<step>-<TAG>/ with no termination probe (scripts/onpolicy.sh
# evaluates it). VLLM_GPU_MEMORY_UTILIZATION and CPU_NODE as scripts/official_eval.sh; CPU_SET (docker --cpuset-cpus,
# default all) confines the containers other than the server (the clients and both tools' scoring).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_DIR=$ROOT/artifacts/runs/$1 STEP=$2 LIMIT=${3:-} MODEL=${MODEL:-} TAG=${TAG:-}
[ -z "$MODEL" ] || [ -n "$TAG" ] || { echo "MODEL needs TAG" >&2; exit 1; }
CHECKPOINT=
[ -n "$MODEL" ] ||
  for CHECKPOINT in "$RUN_DIR"/{checkpoint,weights}-"$STEP".pt ""; do [ -f "$CHECKPOINT" ] && break; done
[ -n "$MODEL$CHECKPOINT" ] || { echo "no checkpoint or weights of step $STEP in $RUN_DIR" >&2; exit 1; }
OUT=$RUN_DIR/official-$STEP${TAG:+-$TAG}
mkdir -p "$OUT"/{model,evalplus,lcb}
W=/workspace/nanochat C=/workspace/nanochat/${OUT#"$ROOT"/} PORT=${PORT:-8000} SERVER=nanochat-eval-serve-$$
LCB_WINDOW=(release_v6 2024-08-01 2025-05-01)  # lcb_runner's --release_version, --start_date, --end_date
SERVE_ENV=() SERVE_ARGS=()
if [ "${LOOP_STOP:-1}" != 0 ]; then
  touch "$OUT/loop-stops.jsonl"
  SERVE_ENV=(-e NANOCHAT_LOOP_STOP_LOG=$C/loop-stops.jsonl)
  SERVE_ARGS=(--logits-processors eval.official_loop_stop:LoopStopLogitsProcessor)
fi
docker build -q -f "$ROOT/docker/Dockerfile.eval" -t nanochat-eval "$ROOT/docker" > /dev/null

# XDG_CACHE_HOME as the nanochat image's (docker/Dockerfile): EvalPlus's datasets and ground-truth caches
# (artifacts/cache/evalplus) persist across its containers instead of downloading and recomputing in each
DOCKER=(docker run --rm --ipc=host --init --network host --ulimit memlock=-1 --ulimit stack=67108864:-1 --ulimit core=0
        --user "$(id -u):$(id -g)" -e HOME=/tmp -e XDG_CACHE_HOME=$W/artifacts/cache
        -e HF_HOME=$W/artifacts/cache/huggingface -e PYTHONPATH=$W
        -v "$ROOT":$W -v "$(realpath "$ROOT/artifacts")":$W/artifacts -w $W ${CPU_SET:+--cpuset-cpus "$CPU_SET"})
CPU=("${DOCKER[@]}" --cpuset-mems="${CPU_NODE:-0}" -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1)
step() {  # <name> <command...>: run, and append the wall time to timings.tsv
  local start=$SECONDS; "${@:2}"; printf '%s\t%d\n' "$1" $((SECONDS - start)) >> "$OUT/timings.tsv"
}

if [ -n "$MODEL" ]; then
  [ -f "$OUT/model/model.safetensors" ] || cp -r "$MODEL"/. "$OUT/model/"
else
  [ -f "$OUT/model/model.safetensors" ] ||
    step export "${CPU[@]}" nanochat python -m inference.export_hf "${CHECKPOINT/#$ROOT/$W}" $C/model --chat
fi
"${DOCKER[@]}" nanochat-eval cp /opt/tools.json $C/tools.json
PROBE=${RUN_DIR/#$ROOT/$W}/termination-$STEP.json
if [ -z "$MODEL" ] && [ ! -f "$RUN_DIR/termination-$STEP.json" ] && [ -z "$LIMIT" ]; then
  step termination "${DOCKER[@]}" --device nvidia.com/gpu=all -e VLLM_CACHE_ROOT=$W/artifacts/cache/vllm \
    -e PYTHONPATH=$W/eval/official_site:$W ${VLLM_GPU_MEMORY_UTILIZATION:+-e VLLM_GPU_MEMORY_UTILIZATION} \
    nanochat-eval python -m eval.official_termination $C/model "$PROBE.generated"
  "${CPU[@]}" nanochat-eval python -m eval.official_termination --score "$PROBE.generated" "$PROBE"
fi

docker run -d --rm --name $SERVER --ipc=host --init --network host --ulimit core=0 --device nvidia.com/gpu=all \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e VLLM_CACHE_ROOT=$W/artifacts/cache/vllm \
  -e PYTHONPATH=$W/eval/official_site:$W -e NANOCHAT_REASONING_PARSER=1 "${SERVE_ENV[@]}" \
  -v "$ROOT":$W -v "$(realpath "$ROOT/artifacts")":$W/artifacts -v "$OUT/model":/model:ro nanochat-eval \
  vllm serve /model --served-model-name nanochat --trust-remote-code --dtype bfloat16 --port $PORT \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.9}" --reasoning-parser nanochat "${SERVE_ARGS[@]}" \
  > /dev/null
trap 'docker logs $SERVER > "$OUT/server.log" 2>&1; docker stop $SERVER > /dev/null' EXIT
until curl -sf http://127.0.0.1:$PORT/health > /dev/null; do
  docker inspect $SERVER > /dev/null 2>&1 || { echo "vllm serve exited" >&2; exit 1; }; sleep 5
done
API=(-e OPENAI_BASE_URL=http://127.0.0.1:$PORT/v1 -e OPENAI_API_KEY=none -e OPENAI_KEY=none)

"${DOCKER[@]}" nanochat-eval python -m eval.official_tokenize evalplus http://127.0.0.1:$PORT $C/tokenize-evalplus.json
"${DOCKER[@]}" -w /opt/LiveCodeBench nanochat-eval /opt/lcb/bin/python -m eval.official_tokenize lcb \
  http://127.0.0.1:$PORT $C/tokenize-lcb.json "${LCB_WINDOW[@]}"
"${CPU[@]}" nanochat python -m eval.official_tokenize check $C/tokenize-{evalplus,lcb}.json

EVALPLUS=(--model nanochat --backend openai --base_url http://127.0.0.1:$PORT/v1 --max_new_tokens 32768
          --root $C/evalplus ${LIMIT:+--id_range "[0,$LIMIT]"})
for dataset in humaneval mbpp; do
  # the samples files: nanochat_openai_temp_0.0 (greedy), nanochat_openai_temp_0.6 (sampled)
  step "evalplus.codegen $dataset greedy" "${DOCKER[@]}" "${API[@]}" nanochat-eval python -m eval.official_evalplus \
    --dataset $dataset --greedy "${EVALPLUS[@]}"
  step "evalplus.codegen $dataset sampled" "${DOCKER[@]}" "${API[@]}" nanochat-eval python -m eval.official_evalplus \
    --dataset $dataset --n_samples 20 --temperature 0.6 "${EVALPLUS[@]}"
  [ -n "$LIMIT" ] ||  # evalplus.evaluate requires every task's samples
    for temperature in 0.0 0.6; do
      step "evalplus.evaluate $dataset $temperature" "${CPU[@]}" nanochat-eval python -m evalplus.evaluate \
        --dataset $dataset --samples $C/evalplus/$dataset/nanochat_openai_temp_$temperature.jsonl
    done
done

# lcb_runner reads its prompt files relative to the working directory and writes output/ there
step "lcb_runner" "${DOCKER[@]}" "${API[@]}" -w $C/lcb nanochat-eval bash -c "ln -sfn /opt/LiveCodeBench/lcb_runner . &&
  /opt/lcb/bin/python -m eval.official_lcb --model nanochat --scenario codegeneration \
    --release_version ${LCB_WINDOW[0]} --start_date ${LCB_WINDOW[1]} --end_date ${LCB_WINDOW[2]} --n 16 \
    --temperature 0.6 --top_p 0.95 --max_tokens 32768 --openai_timeout 86400 --multiprocess 64 --evaluate \
    --num_process_evaluate 32 ${LIMIT:+--debug}"
[ -n "$LIMIT" ] || "${CPU[@]}" nanochat python -m eval.official_reasoning $C
