#!/bin/bash
# Prepare a GH200 host (Docker, the NVIDIA driver and Container Toolkit): the CDI spec, the locked SM clock,
# the docker group, the image, and the curated corpus with its tokenizer (the public dataset
# corpus/pretrain/publish.py DATASET, pulled without a token). Idempotent; re-run after a reboot, which resets the
# clock lock. Usage: scripts/setup.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
# maximum SM clock: under a power cap the driver otherwise lowers it; speed only, not numerics
MAX_SM_CLOCK=$(nvidia-smi --id=0 --query-gpu=clocks.max.sm --format=csv,noheader,nounits)
sudo nvidia-smi -lgc "$MAX_SM_CLOCK,$MAX_SM_CLOCK"
# the docker group applies at the next login; sg runs this script's docker commands in it
sudo usermod -aG docker "$USER"
sg docker -c "docker build -f '$ROOT/docker/Dockerfile' -t nanochat '$ROOT' &&
  '$ROOT/scripts/container.sh' python -m corpus.pretrain.publish pull"
echo "log in again (docker group); every command then runs in the image through scripts/container.sh"
