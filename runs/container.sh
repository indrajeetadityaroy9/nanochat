#!/bin/bash
# Run a command in the nanochat image (docker/Dockerfile) on this GPU: the experiment directory ~/nanochat-runs at /runs
# (tokenizer, data, checkpoints, metrics, caches: nanochat/paths.py) and this checkout at /workspace/nanochat. The
# Hugging Face token is the Hub library's own file under the image's HF_HOME (~/nanochat-runs/data/huggingface/token,
# written by `runs/container.sh hf auth login`), read only by the calls that need it.
#
#   runs/container.sh python -m nanochat.data.compile
#
# --ipc=host --ulimit memlock=-1 --ulimit stack=67108864: the NGC PyTorch container's documented run flags, the stack's
# hard limit unlimited (:-1) so evalplus's reliability_guard can raise its test processes' stack (evalplus/eval/utils.py
# setrlimit RLIMIT_STACK)
REPO=$(cd "$(dirname "$0")/.." && pwd)
exec docker run --rm --device nvidia.com/gpu=all --ipc=host --init --ulimit memlock=-1 --ulimit stack=67108864:-1 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$HOME/nanochat-runs":/runs -v "$REPO":/workspace/nanochat nanochat "$@"
