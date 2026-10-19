#!/bin/bash
# Set up a fresh single-H100 host that has Docker and the NVIDIA driver (Lambda's Ubuntu image) for runs/h100.sh: the
# CDI spec runs/container.sh's `--device nvidia.com/gpu=all` names, this user in the docker group, the nanochat image,
# and the trained tokenizer and the compiled + packed data (~/nanochat-runs/{tokenizer,data/compiled}) from the private
# Hugging Face dataset repo that holds them, downloaded with the image's `hf` (huggingface-hub) into the Hub library's
# token file under the image's HF_HOME. Every step is idempotent: a rerun rebuilds from the docker build cache and
# `hf download` skips the files already in place.
#
#   HF_TOKEN=<read token> runs/setup.sh <dataset repo>    (e.g. iaroy/nanochat-data)
set -euo pipefail
REPO=$1
cd "$(dirname "$0")/.."
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
# the group takes effect at the next login; until then sg runs this script's docker commands in it
sudo usermod -aG docker "$USER"
sg docker -c "docker build -f docker/Dockerfile -t nanochat ."
mkdir -p "$HOME/nanochat-runs"
sg docker -c "runs/container.sh hf auth login --token $HF_TOKEN"
sg docker -c "runs/container.sh hf download $REPO --type dataset --local-dir /runs"
echo "log in again (docker group), then: tmux new -d -s h100 'cd $(pwd) && runs/h100.sh'"
