#!/bin/bash
# Run a command in the nanochat image with this checkout mounted at /workspace/nanochat (paths.py ROOT).
# NGC's documented run flags; stack hard limit unlimited for evalplus's reliability_guard (setrlimit RLIMIT_STACK).
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec docker run --rm --device nvidia.com/gpu=all --ipc=host --init --ulimit memlock=-1 --ulimit stack=67108864:-1 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$ROOT":/workspace/nanochat nanochat "$@"
