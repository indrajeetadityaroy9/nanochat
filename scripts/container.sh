#!/bin/bash
# Run a command in the nanochat image as the invoking user, this checkout mounted at /workspace/nanochat (paths.py
# ROOT); HF_TOKEN passes through from the environment (setup.sh's private dataset, release.sh's upload). NGC's
# documented run flags; stack hard limit unlimited for evalplus's reliability_guard (setrlimit RLIMIT_STACK).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec docker run --rm --device nvidia.com/gpu=all --ipc=host --init --ulimit memlock=-1 --ulimit stack=67108864:-1 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_TOKEN -v "$ROOT":/workspace/nanochat nanochat "$@"
