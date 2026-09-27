# syntax=docker/dockerfile:1
#
# nanochat image for NVIDIA Hopper and Blackwell GPUs. Datasets are not baked in: mount a volume at /runs
# (NANOCHAT_BASE_DIR: tokenizer, checkpoints, eval results) and fast node-local NVMe at /runs/data (raw corpora,
# compiled data). See docker/compose.yaml.
#
# Built on NVIDIA's NGC PyTorch image the way NVIDIA's own stacks (Megatron-LM, Megatron-Bridge, NeMo Automodel,
# NeMo-RL) build on it: the image's Python is left as NVIDIA built it, and uv.lock goes into a virtual environment that
# sees the image's packages. The environment takes torch, everything only torch needs (Triton, the CUDA libraries) and
# numpy (which the image's numba pins below 2.5) from the image, and every other package at its locked version.
#
#   docker build -f docker/Dockerfile -t nanochat .

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.08-py3
FROM ghcr.io/astral-sh/uv:0.12.19 AS uv

FROM ${BASE_IMAGE}
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
WORKDIR /workspace/nanochat

# dependency layer, rebuilt only when pyproject.toml or uv.lock change. The image's torchrun starts the environment's
# Python (as NeMo Automodel does), and the import confirms FlashAttention-4 loads against the image's torch.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --system-site-packages --python /usr/bin/python3 $VIRTUAL_ENV && \
    uv export --frozen --no-dev --no-emit-project --no-hashes --prune torch --prune numpy -o /tmp/requirements.txt && \
    uv pip install --no-deps -r /tmp/requirements.txt && \
    sed -i "1s|.*|#!$VIRTUAL_ENV/bin/python|" /usr/local/bin/torchrun && \
    python -c "import torch, flash_attn.cute"

COPY . .
# HF_XET_HIGH_PERFORMANCE: saturate network and CPUs when pulling raw corpora from the Hugging Face Hub
ENV NANOCHAT_BASE_DIR=/runs HF_HOME=/runs/data/huggingface \
    HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
