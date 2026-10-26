# syntax=docker/dockerfile:1
# nanochat image (sm_90) on NGC PyTorch (torch, Triton, CUDA, nvcc and numpy from the image, outside uv.lock); uv.lock
# in a venv over the image's packages. No code or data baked in: scripts/container.sh mounts the checkout at WORKDIR.
# Base images pinned by manifest digest.

FROM ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b AS uv

FROM nvcr.io/nvidia/pytorch:26.09-py3@sha256:6e8ccc607fc2a51e3741667b86316a0889418ba8b78ed0ba96e8d282b648a7a6
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
WORKDIR /workspace/nanochat

# Rust toolchain (pinned, minimal profile) for the crates, built on first use; CARGO_HOME in artifacts/cache (set below)
ENV RUSTUP_HOME=/usr/local/rustup PATH=/usr/local/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | CARGO_HOME=/usr/local/cargo sh -s -- -y --no-modify-path --profile minimal --default-toolchain 1.99.1
# the image's `flash_attn` (FA2 with an FA4 beta) would hide the locked flash_attn.cute
RUN python3 -m pip uninstall -y flash-attn flash-attn-4

# dependency layer, rebuilt only when pyproject.toml or uv.lock change
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --system-site-packages --python /usr/bin/python3 $VIRTUAL_ENV && \
    uv sync --frozen --active

# openai/human-eval-infilling: the checkout on sys.path (.pth; read_problems reads <package>/../data), execution enabled
# (its README's step) and HumanEval-RandomSpan.jsonl.gz, the name data.py:13 reads, linked to the shipped file
RUN git clone https://github.com/openai/human-eval-infilling /opt/human-eval-infilling && \
    git -C /opt/human-eval-infilling checkout 88062ff9859c875d04db115b698ed4b0f0395170 && \
    sed -i 's/^#\( *exec(check_program, exec_globals)\)/\1/' /opt/human-eval-infilling/human_eval_infilling/execution.py && \
    ln -s HumanEval-RandomSpanInfilling.jsonl.gz /opt/human-eval-infilling/data/HumanEval-RandomSpan.jsonl.gz && \
    echo /opt/human-eval-infilling > "$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/human_eval_infilling.pth"

# nuprl/MultiPL-E's official C++ judge (eval/code_eval.py runs its scripts from $MULTIPL_E)
ENV MULTIPL_E=/opt/MultiPL-E
RUN git clone https://github.com/nuprl/MultiPL-E $MULTIPL_E && \
    git -C $MULTIPL_E checkout 3025a531af7450e7df8b96fe0440e9804480bbad

# TF32 override off (the NGC image sets 1: fp32 cuBLAS matmuls in TF32); one OpenMP thread (parallelism is by process);
# compiler, kernel and user caches in the mounted artifacts/cache, reused across containers
ENV HF_HOME=/workspace/nanochat/artifacts/cache/huggingface \
    HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    TORCHINDUCTOR_CACHE_DIR=/workspace/nanochat/artifacts/cache/inductor \
    TRITON_CACHE_DIR=/workspace/nanochat/artifacts/cache/triton QUACK_CACHE_DIR=/workspace/nanochat/artifacts/cache/quack \
    FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/nanochat/artifacts/cache/flash_attn \
    CARGO_HOME=/workspace/nanochat/artifacts/cache/cargo XDG_CACHE_HOME=/workspace/nanochat/artifacts/cache \
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
