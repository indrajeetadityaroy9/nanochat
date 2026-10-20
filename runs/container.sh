#!/bin/bash
# Run a command in the nanochat image (docker/Dockerfile) on this GPU, with this checkout mounted at the image's working
# directory /workspace/nanochat: nanochat/paths.py's ROOT, artifacts/ included. Hugging Face token:
# artifacts/data/huggingface/token (the image's HF_HOME), written by `runs/container.sh hf auth login`.
#
#   runs/container.sh python -m nanochat.data.compile
#
# --ipc=host --ulimit memlock=-1 --ulimit stack=67108864: the NGC PyTorch container's documented run flags; stack hard
# limit unlimited (:-1) for evalplus's reliability_guard (evalplus/eval/utils.py setrlimit RLIMIT_STACK).
# TORCHINDUCTOR_COORDINATE_DESCENT_TUNING: Inductor's kernels -33 ms of a 3.74 s d24 step (ablations/ablations.md).
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec docker run --rm --device nvidia.com/gpu=all --ipc=host --init --ulimit memlock=-1 --ulimit stack=67108864:-1 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1 \
  -v "$ROOT":/workspace/nanochat nanochat "$@"
