#!/bin/bash
# Prepare a fresh GH200 or H100 host (Docker, NVIDIA driver): CDI spec, locked SM clock, docker group, image, artifacts/
# and the curated corpus. Idempotent. Usage: HF_TOKEN=<read token> scripts/setup.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
# maximum SM clock: under a power cap the driver otherwise lowers it; speed only, not numerics; reset by a reboot
MAX_SM_CLOCK=$(nvidia-smi --id=0 --query-gpu=clocks.max.sm --format=csv,noheader,nounits)
sudo nvidia-smi -lgc "$MAX_SM_CLOCK,$MAX_SM_CLOCK"
# the docker group applies at the next login; sg runs this script's docker commands in it
sudo usermod -aG docker "$USER"
sg docker -c "docker build -f '$ROOT/docker/Dockerfile' -t nanochat '$ROOT'"
sg docker -c "'$ROOT/scripts/container.sh' hf auth login --token $HF_TOKEN"
sg docker -c "'$ROOT/scripts/container.sh' python -m paths"
sg docker -c "'$ROOT/scripts/container.sh' python -m corpus.publish pull"
echo "log in again (docker group), then follow docs/training.md §9"
