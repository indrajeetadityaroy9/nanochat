#!/bin/bash
# Prepare a fresh H100 host with Docker and the NVIDIA driver (Lambda's Ubuntu image): the CDI spec for
# runs/container.sh's `--device nvidia.com/gpu=all`, docker group membership, the nanochat image, then (nanochat/paths.py)
# the artifacts layout and the tokenizer and compiled + packed data at the dataset revision nanochat/config.py pins.
# Idempotent: docker build cache, mkdir exist_ok, snapshot_download skips present files.
#
#   HF_TOKEN=<read token> runs/setup.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
# the docker group applies at the next login; sg runs this script's docker commands in it
sudo usermod -aG docker "$USER"
sg docker -c "docker build -f '$ROOT/docker/Dockerfile' -t nanochat '$ROOT'"
sg docker -c "'$ROOT/runs/container.sh' hf auth login --token $HF_TOKEN"
sg docker -c "'$ROOT/runs/container.sh' python -m nanochat.paths"
echo "log in again (docker group), then from the checkout: tmux new -d -s h100 runs/h100.sh"
