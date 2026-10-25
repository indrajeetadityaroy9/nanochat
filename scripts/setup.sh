#!/bin/bash
# Prepare a fresh GH200 or H100 host with Docker and the NVIDIA driver (Lambda's Ubuntu image): the CDI spec
# for scripts/container.sh's `--device nvidia.com/gpu=all`, docker group membership, the nanochat image, the artifacts
# layout (paths.py) and the curated corpus (corpus/publish.py pull), from which the tokenizer ablation,
# tokenize and pack build the training rows (docs/pretraining.md, Reproducing the run). Idempotent: docker build
# cache, mkdir exist_ok.
#
#   HF_TOKEN=<read token> scripts/setup.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
# the GPU at its maximum SM clock (1980 MHz on the GH200): under a power cap the driver otherwise lowers the clock
# (ACTIONS.md E2); speed only, not numerics; reset by a reboot
sudo nvidia-smi -lgc 1980,1980
# the docker group applies at the next login; sg runs this script's docker commands in it
sudo usermod -aG docker "$USER"
sg docker -c "docker build -f '$ROOT/docker/Dockerfile' -t nanochat '$ROOT'"
sg docker -c "'$ROOT/scripts/container.sh' hf auth login --token $HF_TOKEN"
sg docker -c "'$ROOT/scripts/container.sh' python -m paths"
sg docker -c "'$ROOT/scripts/container.sh' python -m corpus.publish pull"
echo "log in again (docker group), then follow docs/pretraining.md's Reproducing the run: the tokenizer ablation"
echo "  scripts/container.sh python -m ablations.tokenizer vocabularies|intrinsic|data"
echo "  scripts/container.sh python run.py experiment=tokenizer_split_merge"
echo "  scripts/container.sh python -m ablations.tokenizer adopt"
echo "then tokenize, pack and the base model's pre-training (output in artifacts/runs/<run_name>/run.log):"
echo "  scripts/container.sh python -m corpus.tokenize && scripts/container.sh python -m corpus.pack"
echo "  tmux new -d -s run 'scripts/container.sh python run.py experiment=pretrain'"
