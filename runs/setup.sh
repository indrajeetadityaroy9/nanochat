#!/bin/bash
# Prepare a fresh GH200 or H100 host with Docker and the NVIDIA driver (Lambda's Ubuntu image): the CDI spec for
# runs/container.sh's `--device nvidia.com/gpu=all`, docker group membership, the nanochat image, then (nanochat/paths.py)
# the artifacts layout and the curated store and tokenizer (the dataset at main), which
# `python -m nanochat.data.compile` and `python -m nanochat.data.pack` then build into the training rows.
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
echo "log in again (docker group), then from the checkout: build the training rows and check them and the download"
echo "  runs/container.sh python -m nanochat.data.compile && runs/container.sh python -m nanochat.data.pack"
echo "  (cd artifacts && sha256sum --quiet -c data/curated.sha256 data/compiled.sha256)"
echo "then the tokenizer experiment (docs/h100-pretraining.md) and production:"
echo "  tmux new -d -s run 'runs/container.sh python -m scripts.production 2>&1 | tee -a artifacts/logs/production.log'"
