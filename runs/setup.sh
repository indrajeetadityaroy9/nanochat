#!/bin/bash
# Prepare a fresh H100 host with Docker and the NVIDIA driver (Lambda's Ubuntu image) for runs/h100.sh: the CDI spec for
# runs/container.sh's `--device nvidia.com/gpu=all`, docker group membership, the nanochat image, and the tokenizer and
# compiled + packed data (~/nanochat-runs/{tokenizer,data/compiled}) from the private Hugging Face dataset repo.
# Idempotent: docker build cache, `hf download` skips present files.
#
#   HF_TOKEN=<read token> runs/setup.sh <dataset repo>    (e.g. iaroy/nanochat-data)
set -euo pipefail
REPO=$1
cd "$(dirname "$0")/.."
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
# the docker group applies at the next login; sg runs this script's docker commands in it
sudo usermod -aG docker "$USER"
sg docker -c "docker build -f docker/Dockerfile -t nanochat ."
mkdir -p "$HOME/nanochat-runs"
sg docker -c "runs/container.sh hf auth login --token $HF_TOKEN"
sg docker -c "runs/container.sh hf download $REPO --type dataset --local-dir /runs"
echo "log in again (docker group), then: tmux new -d -s h100 'cd $(pwd) && runs/h100.sh'"
