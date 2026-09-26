#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning).
# The time-to-GPT-2 reference is an 8XH100 node (~1.5 hours), but it runs on any number of GPUs:
# NPROC_PER_NODE defaults to every visible GPU and gradient accumulation makes up the difference.
# Assumes an activated environment (`uv sync --extra gpu && source .venv/bin/activate`) or the docker/ image.

# 1) Example launch (simplest):
# bash runs/speedrun.sh
# 2) Example launch in a screen session (because the run takes ~1.5 hours):
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p $NANOCHAT_BASE_DIR
NPROC_PER_NODE="${NPROC_PER_NODE:-gpu}"

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Data: raw corpus -> tokenizer -> compiled token rows

# train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data (it downloads the raw files it reads)
python -m nanochat.tokenizer
# download, tokenize and pack the first 170 raw files (~150 for GPT-2 capability, plus 20 of padding) into token
# rows once, on all CPUs
python -m nanochat.data.pretrain.compile --max-files=170

# -----------------------------------------------------------------------------
# Base model (pretraining)

# d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN
# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# run SFT and eval the model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft
