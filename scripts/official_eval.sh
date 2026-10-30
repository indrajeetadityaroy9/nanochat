#!/bin/bash
# The benchmarks' official tools, unmodified, on one checkpoint (docs/evaluation.md §8): the checkpoint exported by
# inference/export_hf.py, served by vLLM 0.32.0's Transformers backend with the head in fp32 (or loaded by transformers
# for bigcode-evaluation-harness); each tool's own generator and judge in the nanochat-eval image
# (docker/Dockerfile.eval); eval/official.py's summary, and with a record its paired comparison, in the nanochat image.
#
#   scripts/official_eval.sh <run name> <step> [<eval record to compare against, relative to the checkout>]
#
# Outputs in RUNS/<run>/official-<step>/: model/ (the export), evalplus/, multipl-e/{greedy,sampled}/, lm-eval/,
# bigcode/, tools.json, timings.tsv, summary.json, compare-*.json. Rerun on an existing directory, the export is kept
# and the EvalPlus and MultiPL-E generators and judges skip the samples and verdicts already there.
# VLLM_GPU_MEMORY_UTILIZATION=<fraction> shares the GPU (eval/official_site/sitecustomize.py); CPU work is bound to
# the memory of NUMA node CPU_NODE (default 0: on GH200 node 1 is the GPU's memory).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_DIR=$ROOT/artifacts/runs/$1 STEP=$2 COMPARE=${3:-}
for CHECKPOINT in "$RUN_DIR"/{checkpoint,weights}-"$STEP".pt ""; do [ -f "$CHECKPOINT" ] && break; done
[ -n "$CHECKPOINT" ] || { echo "no checkpoint or weights of step $STEP in $RUN_DIR" >&2; exit 1; }
OUT=$RUN_DIR/official-$STEP
mkdir -p "$OUT"/{model,bigcode,lm-eval} "$OUT"/multipl-e/{greedy,sampled}/{humaneval,mbpp}-cpp
# the export at /model in the generators' containers: run names hold commas, which lm-eval's model_args split on
W=/workspace/nanochat C=/workspace/nanochat/${OUT#"$ROOT"/} M=/model
docker build -q -f "$ROOT/docker/Dockerfile.eval" -t nanochat-eval "$ROOT/docker" > /dev/null

DOCKER=(docker run --rm --ipc=host --init --ulimit memlock=-1 --ulimit stack=67108864:-1 --ulimit core=0
        --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=$W/artifacts/cache/huggingface -e PYTHONPATH=$W
        -v "$ROOT":$W -v "$(realpath "$ROOT/artifacts")":$W/artifacts -w $W)
# one BLAS/OpenMP thread: evalplus.evaluate's reliability_guard caps a test process's address space (RLIMIT_AS),
# which counts the mappings it inherits from evalplus.evaluate's process, inflated by per-core thread pools on 64
# cores, so the largest-allocation tests fail with MemoryError (HumanEval/15's plus input [1000000])
CPU=("${DOCKER[@]}" --cpuset-mems="${CPU_NODE:-0}" -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1)
GPU=("${DOCKER[@]}" --device nvidia.com/gpu=all -v "$OUT/model":/model:ro -e VLLM_CACHE_ROOT=$W/artifacts/cache/vllm
     -e PYTHONPATH=$W/eval/official_site:$W -e VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}")
step() {  # <name> <command...>: run, and append the wall time to timings.tsv
  local start=$SECONDS; "${@:2}"; printf '%s\t%d\n' "$1" $((SECONDS - start)) >> "$OUT/timings.tsv"
}

[ -f "$OUT/model/model.safetensors" ] ||
  step export "${CPU[@]}" nanochat python -m inference.export_hf "${CHECKPOINT/#$ROOT/$W}" $C/model
"${DOCKER[@]}" nanochat-eval cp /opt/tools.json $C/tools.json

# EvalPlus: greedy, 768 new tokens, its EOS list and sanitizer (codegen.py, provider/vllm.py)
for dataset in humaneval mbpp; do
  step "evalplus.codegen $dataset" "${GPU[@]}" nanochat-eval python -m evalplus.codegen --model $M \
    --dataset $dataset --backend vllm --greedy --trust_remote_code True --root $C/evalplus
  step "evalplus.evaluate $dataset" "${CPU[@]}" nanochat-eval bash -c "python -m evalplus.evaluate --dataset $dataset \
    --samples \$(ls $C/evalplus/$dataset/*.jsonl | grep -v '\.raw\.jsonl$')"
done

# MultiPL-E C++: greedy (temperature 0, one completion) and sampled (T 0.2, top-p 0.95, 20), 1,024 new tokens; judged
# one problem file per main.py process (main.py --dir runs the problems serially)
for dataset in humaneval mbpp; do
  for mode in "greedy 0 1" "sampled 0.2 20"; do
    set -- $mode
    step "multipl-e $dataset-cpp $1" "${GPU[@]}" -w /opt/MultiPL-E nanochat-eval python automodel_vllm.py --name $M \
      --root-dataset $dataset --lang cpp --temperature $2 --top-p 0.95 --completion-limit $3 \
      --output-dir $C/multipl-e/$1/$dataset-cpp
    step "multipl-e judge $dataset-cpp $1" "${CPU[@]}" nanochat-eval bash -c "ls $C/multipl-e/$1/$dataset-cpp/*.json.gz |
      grep -v '\.results\.json\.gz$' | xargs -P $((32 / ($3 > 4 ? 4 : $3))) -I{} python \$MULTIPL_E/evaluation/src/main.py \
      --file {} --output-dir $C/multipl-e/$1/$dataset-cpp --max-workers $(($3 > 4 ? 4 : $3)) > /dev/null"
  done
done
"${CPU[@]}" nanochat-eval bash -c "python /opt/MultiPL-E/pass_k.py $C/multipl-e/*/*" | tee "$OUT/multipl-e/pass_k.csv"

# lm-evaluation-harness mbpp (3-shot, greedy, until "[DONE]", 256 new tokens), BOS by the tokenizer's post-processor
ls "$OUT"/lm-eval/*/samples_mbpp_*.jsonl > /dev/null 2>&1 ||
  step "lm-eval mbpp" "${GPU[@]}" -e HF_ALLOW_CODE_EVAL=1 nanochat-eval lm_eval --model vllm \
    --model_args pretrained=$M,trust_remote_code=True,add_bos_token=True --tasks mbpp --confirm_run_unsafe_code \
    --log_samples --output_path $C/lm-eval

# bigcode-evaluation-harness SantaCoder FIM (all three languages) with this tokenizer's sentinels, generated by vLLM
# and scored by the task's own code (eval/official_fim.py)
[ -f "$OUT/bigcode/evaluation_results.json" ] ||
  step "bigcode santacoder_fim" "${GPU[@]}" nanochat-eval python -m eval.official_fim $M $C/bigcode

# human-eval-infilling: samples by eval/official_infilling.py (greedy PSM and SPM; sampled SPM, T 0.2, 100 or 200 a
# task), judged by the package's evaluate_functional_correctness (3 s a sample)
for mode in greedy sampled; do
  ls "$OUT"/infilling/$mode/*.jsonl > /dev/null 2>&1 ||
    step "infilling $mode" "${GPU[@]}" nanochat-eval python -m eval.official_infilling $M $C/infilling/$mode $mode
  for samples in "$OUT"/infilling/$mode/*.jsonl; do  # its default k "1,10,100" (fire parses a given k as numbers)
    [ -f "$samples" ] && [ ! -f "${samples}_results.jsonl.gz" ] || continue
    name=$(basename "$samples" .jsonl)
    step "infilling judge $name $mode" "${CPU[@]}" nanochat-eval python -m \
      human_eval_infilling.evaluate_functional_correctness --benchmark_name "${name%-*}" \
      --sample_file "${samples/#$ROOT/$W}" --n_workers 32
  done
done

step summary "${CPU[@]}" nanochat python -m eval.official $C ${COMPARE:+$W/$COMPARE}
