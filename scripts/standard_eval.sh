#!/bin/bash
# Standard-protocol cross-check: the benchmarks' own harness CLIs, unmodified (EvalPlus codegen --greedy; MultiPL-E
# automodel_vllm.py, T = 0.2, top-p 0.95, 20 completions: copies of eval/code_eval.py TEMPERATURE, TOP_P, N_SAMPLE),
# on an exported checkpoint served by vLLM 0.32.0 through its Transformers backend; scored by the harnesses'
# evaluators in the nanochat image. <out dir> must be empty or new: both harnesses skip samples already in it.
#
#   scripts/standard_eval.sh <exported model dir> <out dir> [generate|score]   (default both)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ "${3:-both}" = score ] || [ -z "$(ls -A "$2" 2> /dev/null)" ] || { echo "$2 is not empty" >&2; exit 1; }
mkdir -p "$2"
MODEL=$(realpath "$1") OUT=$(realpath "$2") PHASE=${3:-both}
# GitHub archives of the pinned commits (the vLLM image has no git; evalplus's version comes from setuptools-scm)
EVALPLUS=$(grep -o 'evalplus = { git = "[^"]*", rev = "[^"]*"' "$ROOT/pyproject.toml" | sed -E 's|.*git = "([^"]*)", rev = "([^"]*)"|\1/archive/\2.tar.gz|')
MULTIPL_E_REV=$(grep -oP '\$MULTIPL_E checkout \K[0-9a-f]{40}' "$ROOT/docker/Dockerfile")
VLLM=vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
RUN=(docker run --rm --device nvidia.com/gpu=all --ipc=host --init --ulimit memlock=-1 --ulimit stack=67108864:-1
     --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/workspace/nanochat/artifacts/cache/huggingface
     -v "$ROOT":/workspace/nanochat -v "$MODEL":/model:ro -v "$OUT":/out)

# generation: the harnesses' own vLLM generators, in a virtual environment over the image's vLLM
[ "$PHASE" = score ] || "${RUN[@]}" -e VLLM_CACHE_ROOT=/workspace/nanochat/artifacts/cache/vllm --entrypoint bash \
  "$VLLM" -o pipefail -euc "
  uv venv -q --system-site-packages /tmp/venv && . /tmp/venv/bin/activate
  SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv pip install -q 'evalplus @ $EVALPLUS' datasets
  for dataset in humaneval mbpp; do
    python -m evalplus.codegen --model /model --dataset \$dataset --backend vllm --greedy --trust_remote_code True \
      --root /out/evalplus
  done
  mkdir -p /tmp/MultiPL-E /out/multipl-e && curl -sL https://github.com/nuprl/MultiPL-E/archive/$MULTIPL_E_REV.tar.gz |
    tar xz --strip-components=1 -C /tmp/MultiPL-E
  for dataset in humaneval mbpp; do
    (cd /tmp/MultiPL-E && python automodel_vllm.py --name /model --root-dataset \$dataset \
       --lang cpp --temperature 0.2 --top-p 0.95 --completion-limit 20 --output-dir /out/multipl-e/\$dataset-cpp)
  done"

# scoring: the harnesses' evaluators in the nanochat image
[ "$PHASE" = generate ] || "${RUN[@]}" nanochat bash -o pipefail -euc "
  for samples in /out/evalplus/*/*.jsonl; do
    case \$samples in *.raw.jsonl) continue ;; esac  # evalplus codegen's unsanitized copy of the samples
    dataset=\$(basename \$(dirname \$samples)); python -m evalplus.evaluate --dataset \${dataset%%_*} --samples \$samples
  done
  for dataset in humaneval-cpp mbpp-cpp; do
    python \$MULTIPL_E/evaluation/src/main.py --dir /out/multipl-e/\$dataset --output-dir /out/multipl-e/\$dataset
  done
  python \$MULTIPL_E/pass_k.py /out/multipl-e/* | tee /out/pass_k.csv"
