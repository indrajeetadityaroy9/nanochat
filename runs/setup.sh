#!/bin/bash
# Prepare a fresh H100 host with Docker and the NVIDIA driver (Lambda's Ubuntu image) for runs/h100.sh: the CDI spec for
# runs/container.sh's `--device nvidia.com/gpu=all`, docker group membership, the nanochat image, the artifacts layout
# (nanochat/paths.py; the only place it is created) and the tokenizer and compiled + packed data
# (artifacts/{tokenizer,data/compiled}) from the private Hugging Face dataset repo at the revision nanochat/config.py
# pins (DATASET, DATASET_REVISION). Idempotent: docker build cache, `mkdir -p`, `hf download` skips present files.
#
#   HF_TOKEN=<read token> runs/setup.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
# the docker group applies at the next login; sg runs this script's docker commands in it
sudo usermod -aG docker "$USER"
sg docker -c "docker build -f '$ROOT/docker/Dockerfile' -t nanochat '$ROOT'"
# nanochat/paths.py's ARTIFACTS directories
mkdir -p "$ROOT"/artifacts/{data,tokenizer,checkpoints,metrics,base_eval,probes,logs,cache}
sg docker -c "'$ROOT/runs/container.sh' hf auth login --token $HF_TOKEN"
read -r DATASET REVISION < <(sg docker -c "'$ROOT/runs/container.sh' python -c \
  'from nanochat.config import DATASET, DATASET_REVISION; print(DATASET, DATASET_REVISION)'" | tail -n 1)
# /workspace/nanochat: the checkout inside the container (runs/container.sh)
sg docker -c "'$ROOT/runs/container.sh' hf download $DATASET --type dataset --revision $REVISION \
  --local-dir /workspace/nanochat/artifacts"
echo "log in again (docker group), then from the checkout: tmux new -d -s h100 runs/h100.sh"
