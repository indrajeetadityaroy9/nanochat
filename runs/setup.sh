#!/bin/bash
# Prepare a fresh GH200 or H100 host with Docker and the NVIDIA driver (Lambda's Ubuntu image): the CDI spec for
# runs/container.sh's `--device nvidia.com/gpu=all`, docker group membership, the nanochat image, the artifacts layout
# (nanochat/paths.py) and the curated corpus (nanochat/corpus/publish.py pull), from which the tokenizer ablation,
# compile and pack build the training rows (nanochat/corpus/gh200.md). Idempotent: docker build cache, mkdir exist_ok.
#
#   HF_TOKEN=<read token> runs/setup.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
# the GPU at its maximum SM clock (1980 MHz on the GH200): under a power cap the driver otherwise lowers the clock, and
# locking it shortened the d24 8K step by 3.0% (measured on the GH200); speed only, not numerics; reset by a reboot
sudo nvidia-smi -lgc 1980,1980
# the docker group applies at the next login; sg runs this script's docker commands in it
sudo usermod -aG docker "$USER"
sg docker -c "docker build -f '$ROOT/docker/Dockerfile' -t nanochat '$ROOT'"
sg docker -c "'$ROOT/runs/container.sh' hf auth login --token $HF_TOKEN"
sg docker -c "'$ROOT/runs/container.sh' python -m nanochat.paths"
sg docker -c "'$ROOT/runs/container.sh' python -m nanochat.corpus.publish pull"
echo "log in again (docker group), then from the checkout follow nanochat/corpus/gh200.md: the tokenizer ablation"
echo "  runs/container.sh python -m ablations.tokenizer vocabularies|intrinsic|data"
echo "  runs/container.sh python -m nanochat.run experiment=tokenizer"
echo "  runs/container.sh python -m ablations.tokenizer adopt"
echo "then compile, pack and the base model's pre-training (output in artifacts/runs/<key>/run.log):"
echo "  runs/container.sh python -m nanochat.corpus.compile && runs/container.sh python -m nanochat.corpus.pack"
echo "  tmux new -d -s run 'runs/container.sh python -m nanochat.run experiment=base'"
