# syntax=docker/dockerfile:1
#
# nanochat image for NVIDIA GPUs. Datasets are not baked in: mount a volume at /runs (NANOCHAT_BASE_DIR:
# tokenizer, checkpoints, eval results) and fast node-local NVMe at /runs/data (raw corpora, compiled data).
# See docker/compose.yaml.
#
# TORCH=base (default): an NGC PyTorch image. NVIDIA's PyTorch/CUDA/NCCL/cuDNN stack is kept as built and
#   nanochat's other dependencies are installed on top at their uv.lock versions (see ngc_requirements.py).
#   25.10 ships PyTorch 2.9 (CUDA 13.0), the release nanochat pins; newer NGC tags carry newer PyTorch.
# TORCH=lock: any Debian/Ubuntu base. Installs the exact uv.lock environment, including the PyTorch 2.9.1
#   CUDA 12.8 wheels (EXTRA=cpu for a CPU-only image, e.g. for data preparation nodes).
#
#   docker build -f docker/Dockerfile -t nanochat .
#   docker build -f docker/Dockerfile --build-arg BASE_IMAGE=ubuntu:24.04 --build-arg TORCH=lock -t nanochat:lock .

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ghcr.io/astral-sh/uv:0.12.15 AS uv

FROM ${BASE_IMAGE}
ARG TORCH=base
ARG EXTRA=gpu
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/nanochat-venv UV_PYTHON_INSTALL_DIR=/opt/uv-python
WORKDIR /workspace/nanochat

# dependency layer, rebuilt only when pyproject.toml or uv.lock change. The plain base also needs g++ at runtime:
# torch.compile's Inductor and Triton's kernel launchers compile C/C++ (NGC images already ship a toolchain).
COPY pyproject.toml uv.lock .python-version ./
COPY docker/ngc_requirements.py docker/
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TORCH" = "lock" ]; then \
        apt-get update && apt-get install -y --no-install-recommends ca-certificates g++ && rm -rf /var/lib/apt/lists/* && \
        uv sync --frozen --no-dev --no-install-project --extra "$EXTRA"; \
    else \
        uv export --quiet --frozen --no-hashes --no-dev --no-emit-project --extra gpu -o /tmp/lock.txt && \
        python docker/ngc_requirements.py pyproject.toml /tmp/lock.txt /tmp/requirements.txt /tmp/constraints.txt && \
        uv pip install --system --break-system-packages -r /tmp/requirements.txt -c /tmp/constraints.txt; \
    fi
ENV PATH=/opt/nanochat-venv/bin:$PATH

COPY . .
# HF_XET_HIGH_PERFORMANCE: saturate network and CPUs when pulling raw corpora from the Hugging Face Hub
ENV NANOCHAT_BASE_DIR=/runs HF_HOME=/runs/data/huggingface \
    HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
