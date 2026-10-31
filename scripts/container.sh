#!/bin/bash
# Run a command in the nanochat image as the invoking user, this checkout mounted at /workspace/nanochat (paths.py
# ROOT); HF_TOKEN passes through from the environment (the private dataset, corpus/pretrain/publish.py). NGC's
# documented run flags; stack hard limit unlimited for evalplus's reliability_guard (setrlimit RLIMIT_STACK); SYS_NICE
# for decode.cu's prompt store, bound to the CPU's NUMA node (mbind; Docker's default seccomp profile requires it).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec docker run --rm --device nvidia.com/gpu=all --ipc=host --init --ulimit memlock=-1 --ulimit stack=67108864:-1 \
  --cap-add SYS_NICE \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_TOKEN -v "$ROOT":/workspace/nanochat nanochat "$@"
