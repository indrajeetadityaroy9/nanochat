#!/bin/bash
# The reasoning protocol of a post-trained checkpoint (docs/post-training.md §11.4) by the benchmarks' official tools:
# the checkpoint exported with the SFT's chat format (inference/export_hf.py --chat), served by `vllm serve` (vLLM
# 0.32.0's Transformers backend, the head in fp32) with the "nanochat" reasoning parser (eval/official_reasoning_parser.py:
# the answer after the last "</think>"); EvalPlus's OpenAI backend (its chat prompt, n 20, T 0.6) and lcb_runner
# (eval/official_lcb.py: LiveCodeBench v6, contest dates 2024-08-01 to 2025-05-01, n 16, T 0.6, top-p 0.95), each
# judged by its own evaluator; the server's generation config adds top-p 0.95 to EvalPlus's requests, which send none.
# max_tokens 32,768 in every request: vLLM caps it at the context less the prompt (api_utils.get_max_tokens).
#
#   scripts/official_reasoning.sh <run name> <step> [<problems per benchmark, a smoke>]
#
# Outputs in RUNS/<run>/official-<step>/: model/, evalplus/, lcb/, server.log, timings.tsv. VLLM_GPU_MEMORY_UTILIZATION
# and CPU_NODE as scripts/official_eval.sh.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_DIR=$ROOT/artifacts/runs/$1 STEP=$2 LIMIT=${3:-}
for CHECKPOINT in "$RUN_DIR"/{checkpoint,weights}-"$STEP".pt ""; do [ -f "$CHECKPOINT" ] && break; done
[ -n "$CHECKPOINT" ] || { echo "no checkpoint or weights of step $STEP in $RUN_DIR" >&2; exit 1; }
OUT=$RUN_DIR/official-$STEP
mkdir -p "$OUT"/{model,evalplus,lcb}
W=/workspace/nanochat C=/workspace/nanochat/${OUT#"$ROOT"/} PORT=${PORT:-8000} SERVER=nanochat-eval-serve-$$
docker build -q -f "$ROOT/docker/Dockerfile.eval" -t nanochat-eval "$ROOT/docker" > /dev/null

DOCKER=(docker run --rm --ipc=host --init --network host --ulimit memlock=-1 --ulimit stack=67108864:-1 --ulimit core=0
        --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=$W/artifacts/cache/huggingface -e PYTHONPATH=$W
        -v "$ROOT":$W -v "$(realpath "$ROOT/artifacts")":$W/artifacts -w $W)
CPU=("${DOCKER[@]}" --cpuset-mems="${CPU_NODE:-0}" -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1)
step() {  # <name> <command...>: run, and append the wall time to timings.tsv
  local start=$SECONDS; "${@:2}"; printf '%s\t%d\n' "$1" $((SECONDS - start)) >> "$OUT/timings.tsv"
}

[ -f "$OUT/model/model.safetensors" ] ||
  step export "${CPU[@]}" nanochat python -m inference.export_hf "${CHECKPOINT/#$ROOT/$W}" $C/model --chat
"${DOCKER[@]}" nanochat-eval cp /opt/tools.json $C/tools.json

docker run -d --rm --name $SERVER --ipc=host --init --network host --ulimit core=0 --device nvidia.com/gpu=all \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e VLLM_CACHE_ROOT=$W/artifacts/cache/vllm -e PYTHONPATH=$W \
  -v "$ROOT":$W -v "$(realpath "$ROOT/artifacts")":$W/artifacts -v "$OUT/model":/model:ro nanochat-eval \
  vllm serve /model --served-model-name nanochat --trust-remote-code --dtype bfloat16 --port $PORT \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.9}" \
  --reasoning-parser-plugin $W/eval/official_reasoning_parser.py --reasoning-parser nanochat > /dev/null
trap 'docker logs $SERVER > "$OUT/server.log" 2>&1; docker stop $SERVER > /dev/null' EXIT
until curl -sf http://127.0.0.1:$PORT/health > /dev/null; do
  docker inspect $SERVER > /dev/null 2>&1 || { echo "vllm serve exited" >&2; exit 1; }; sleep 5
done
API=(-e OPENAI_BASE_URL=http://127.0.0.1:$PORT/v1 -e OPENAI_API_KEY=none -e OPENAI_KEY=none)

for dataset in humaneval mbpp; do
  step "evalplus.codegen $dataset" "${DOCKER[@]}" "${API[@]}" nanochat-eval python -m evalplus.codegen \
    --model nanochat --dataset $dataset --backend openai --base_url http://127.0.0.1:$PORT/v1 --n_samples 20 \
    --temperature 0.6 --max_new_tokens 32768 --root $C/evalplus ${LIMIT:+--id_range "[0,$LIMIT]"}
  [ -n "$LIMIT" ] ||  # evalplus.evaluate requires every task's samples
    step "evalplus.evaluate $dataset" "${CPU[@]}" nanochat-eval bash -c "python -m evalplus.evaluate \
      --dataset $dataset --samples \$(ls $C/evalplus/$dataset/*.jsonl | grep -v '\.raw\.jsonl$')"
done

# lcb_runner reads its prompt files relative to the working directory and writes output/ there
step "lcb_runner" "${DOCKER[@]}" "${API[@]}" -w $C/lcb nanochat-eval bash -c "ln -sfn /opt/LiveCodeBench/lcb_runner . &&
  /opt/lcb/bin/python -m eval.official_lcb --model nanochat --scenario codegeneration --release_version release_v6 \
    --start_date 2024-08-01 --end_date 2025-05-01 --n 16 --temperature 0.6 --top_p 0.95 --max_tokens 32768 \
    --openai_timeout 86400 --multiprocess 64 --evaluate --num_process_evaluate 32 ${LIMIT:+--debug}"
