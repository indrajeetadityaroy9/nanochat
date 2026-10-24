# syntax=docker/dockerfile:1
#
# nanochat image for the NVIDIA H100 (sm_90). Neither code nor data is baked in: runs/container.sh mounts the checkout at
# the working directory /workspace/nanochat, nanochat/paths.py's ROOT; data, checkpoints, results and caches live in its
# artifacts/ (created by runs/setup.sh).
#
# Built on NVIDIA's NGC PyTorch image, which supplies PyTorch as NVIDIA built it: torch, everything only torch needs
# (Triton, the CUDA libraries, nvcc for nanochat/native.py) and numpy (which the image's numba pins below 2.5), none of
# them in uv.lock (pyproject.toml exclude-dependencies). uv.lock goes into a virtual environment that sees the image's
# packages: every locked package is the lock's version, over the image's copy. The image's own FlashAttention is
# uninstalled: its regular `flash_attn` package (FA2, with an FA4 beta inside) would hide the environment's
# flash_attn.cute. UV_COMPILE_BYTECODE: the container's user cannot write bytecode into /opt/venv; UV_LINK_MODE=copy:
# the uv cache is a separate build mount, which hard links cannot cross.
#
#   runs/setup.sh builds it: docker build -f docker/Dockerfile -t nanochat .
#
# Every base image is pinned by its manifest digest (each tag as resolved when pinned), so a rebuild anywhere starts
# from the same bytes.

FROM ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b AS uv

FROM nvcr.io/nvidia/pytorch:26.09-py3@sha256:6e8ccc607fc2a51e3741667b86316a0889418ba8b78ed0ba96e8d282b648a7a6
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
WORKDIR /workspace/nanochat

# Rust toolchain for the Rust crates (nanochat/corpus/crate, nanochat/inference/syntax), built on first use like the CUDA
# kernels: rustup's minimal profile at a pinned version. Crates fetched on that first build go to CARGO_HOME in
# artifacts/cache (set below): the container's user cannot write the image's /usr/local.
ENV RUSTUP_HOME=/usr/local/rustup PATH=/usr/local/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | CARGO_HOME=/usr/local/cargo sh -s -- -y --no-modify-path --profile minimal --default-toolchain 1.99.1
RUN python3 -m pip uninstall -y flash-attn flash-attn-4

# dependency layer, rebuilt only when pyproject.toml or uv.lock change
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --system-site-packages --python /usr/bin/python3 $VIRTUAL_ENV && \
    uv sync --frozen --active

# openai/human-eval-infilling, the official FIM evaluation, from its repository: read_problems reads <package>/../data,
# outside the Python package, so the checkout itself goes on sys.path (a .pth file; its setup.py declares a console
# script uv rejects as invalid). Its README's "enable execution" step uncomments execution.py:84
# `exec(check_program, exec_globals)`; data.py:13 names the random-span file HumanEval-RandomSpan.jsonl.gz, the repository
# ships HumanEval-RandomSpanInfilling.jsonl.gz. Its dependencies (fire, tqdm, numpy) are in the image.
RUN git clone https://github.com/openai/human-eval-infilling /opt/human-eval-infilling && \
    git -C /opt/human-eval-infilling checkout 88062ff9859c875d04db115b698ed4b0f0395170 && \
    sed -i 's/^#\( *exec(check_program, exec_globals)\)/\1/' /opt/human-eval-infilling/human_eval_infilling/execution.py && \
    ln -s HumanEval-RandomSpanInfilling.jsonl.gz /opt/human-eval-infilling/data/HumanEval-RandomSpan.jsonl.gz && \
    echo /opt/human-eval-infilling > "$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/human_eval_infilling.pth"

# nuprl/MultiPL-E's evaluator, the official C++ judge (nanochat/eval/code_eval.py runs its scripts from the checkout at
# $MULTIPL_E): evaluation/src/main.py (eval_cpp.py: g++ -std=c++17, the image's g++) and pass_k.py. Their dependencies
# (tqdm, numpy) are in the image.
ENV MULTIPL_E=/opt/MultiPL-E
RUN git clone https://github.com/nuprl/MultiPL-E $MULTIPL_E && \
    git -C $MULTIPL_E checkout 3025a531af7450e7df8b96fe0440e9804480bbad

# TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0: the NGC base image sets it to 1, which runs every fp32 cuBLAS matmul in TF32
# (10-bit mantissa); the precision policy keeps fp32 as fp32.
# OMP_NUM_THREADS=1: the pipeline and the evaluation parallelize by process (pools of os.cpu_count() workers, rayon), so
# one OpenMP thread per process keeps the threads at the cores instead of multiplying them.
# HF_XET_HIGH_PERFORMANCE: saturate network and CPUs when pulling raw corpora from the Hugging Face Hub. The compiler and
# kernel caches (Inductor, Triton, quack, FlashAttention-4 CuTe, cargo's crates) and the user cache directory (evalplus's
# data releases and ground truth: appdirs' XDG_CACHE_HOME) live in the mounted checkout's artifacts/cache, so every run
# after the first reuses them instead of redoing them in each fresh container.
ENV HF_HOME=/workspace/nanochat/artifacts/cache/huggingface \
    HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    TORCHINDUCTOR_CACHE_DIR=/workspace/nanochat/artifacts/cache/inductor \
    TRITON_CACHE_DIR=/workspace/nanochat/artifacts/cache/triton QUACK_CACHE_DIR=/workspace/nanochat/artifacts/cache/quack \
    FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/nanochat/artifacts/cache/flash_attn \
    CARGO_HOME=/workspace/nanochat/artifacts/cache/cargo XDG_CACHE_HOME=/workspace/nanochat/artifacts/cache \
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
