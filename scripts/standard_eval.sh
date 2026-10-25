#!/bin/bash
# The standard-protocol cross-check of eval/code_eval.py: the benchmarks' own harness CLIs, unmodified, on an exported
# checkpoint (inference/export_hf.py) served by vLLM (the nanochat-vllm image, docker/Dockerfile.vllm):
# - EvalPlus (evalplus pinned as pyproject.toml pins it): `evalplus.codegen --backend vllm --greedy` on HumanEval+ and
#   MBPP+ (direct completion: the tokenizer has no chat template), scored by `evalplus.evaluate` in the nanochat image;
# - MultiPL-E (nuprl/MultiPL-E at docker/Dockerfile's commit): `automodel_vllm.py` on humaneval-cpp and mbpp-cpp at
#   T = 0.2, top-p 0.95, 20 completions per problem, judged by its evaluation/src/main.py and pass_k.py in the nanochat
#   image ($MULTIPL_E).
# Results in <out dir>: evalplus/<dataset>/..._eval_results.json, multipl-e/<dataset>/ and pass_k.csv.
#
#   scripts/standard_eval.sh <exported model dir> <out dir> [generate|score]   (mounted at /model and /out; default
#   both: generate on the GPU, then score on the CPU)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$2"
MODEL=$(realpath "$1") OUT=$(realpath "$2") PHASE=${3:-both}
# GitHub archives of the pinned commits (the vLLM image has no git; evalplus's version comes from setuptools-scm)
EVALPLUS=$(grep -o 'evalplus = { git = "[^"]*", rev = "[^"]*"' "$ROOT/pyproject.toml" | sed -E 's|.*git = "([^"]*)", rev = "([^"]*)"|\1/archive/\2.tar.gz|')
MULTIPL_E_REV=$(grep -o 'checkout [0-9a-f]\{40\}' "$ROOT/docker/Dockerfile" | tail -1 | cut -d' ' -f2)
RUN=(docker run --rm --device nvidia.com/gpu=all --ipc=host --init --ulimit memlock=-1 --ulimit stack=67108864:-1
     --user "$(id -u):$(id -g)" -e HOME=/tmp -e UV_CACHE_DIR=/tmp/uv
     -e HF_HOME=/workspace/nanochat/artifacts/cache/huggingface
     -v "$ROOT":/workspace/nanochat -v "$MODEL":/model:ro -v "$OUT":/out)

# generation: the harnesses' own vLLM generators, in a virtual environment over the image's vLLM
[ "$PHASE" = score ] || "${RUN[@]}" nanochat-vllm bash -euc "
  uv venv -q --system-site-packages /tmp/venv && . /tmp/venv/bin/activate
  SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv pip install -q 'evalplus @ $EVALPLUS' datasets
  for dataset in humaneval mbpp; do
    python -m evalplus.codegen --model /model --dataset \$dataset --backend vllm --greedy --trust_remote_code True \
      --root /out/evalplus
  done
  mkdir /tmp/MultiPL-E && curl -sL https://github.com/nuprl/MultiPL-E/archive/$MULTIPL_E_REV.tar.gz |
    tar xz --strip-components=1 -C /tmp/MultiPL-E
  for dataset in humaneval mbpp; do
    (cd /tmp/MultiPL-E && python automodel_vllm.py --name /model --root-dataset \$dataset \
       --lang cpp --temperature 0.2 --top-p 0.95 --completion-limit 20 --batch-size 20 \
       --output-dir /out/multipl-e/\$dataset-cpp)
  done"

# scoring: the harnesses' evaluators in the nanochat image
[ "$PHASE" = generate ] || "${RUN[@]}" nanochat bash -euc "
  for samples in /out/evalplus/*/*.jsonl; do
    dataset=\$(basename \$(dirname \$samples)); python -m evalplus.evaluate --dataset \${dataset%%_*} --samples \$samples
  done
  for dataset in humaneval-cpp mbpp-cpp; do
    python \$MULTIPL_E/evaluation/src/main.py --dir /out/multipl-e/\$dataset --output-dir /out/multipl-e/\$dataset
  done
  python \$MULTIPL_E/pass_k.py /out/multipl-e/* | tee /out/pass_k.csv"
