# syntax=docker/dockerfile:1
# nanochat image (sm_90): NGC PyTorch (torch, Triton, CUDA, nvcc, numpy, nvidia-ml-py; outside uv.lock) with uv.lock's
# packages in a venv over it; base images pinned by manifest digest, git dependencies by commit. No code or data baked in:
# scripts/container.sh mounts the checkout at WORKDIR, and the entrypoint (paths.py) creates artifacts/ before every
# command.

FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv

FROM nvcr.io/nvidia/pytorch:26.09-py3@sha256:6e8ccc607fc2a51e3741667b86316a0889418ba8b78ed0ba96e8d282b648a7a6
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
WORKDIR /workspace/nanochat

# Rust toolchain (pinned, minimal profile) for the crates, built on first use; CARGO_HOME in artifacts/cache (below).
# rustup-init 1.29.2 for the build's architecture (GH200 arm64, H100 amd64), verified against its published SHA-256
ARG TARGETARCH
ENV RUSTUP_HOME=/usr/local/rustup PATH=/usr/local/cargo/bin:$PATH
RUN case $TARGETARCH in \
      arm64) T=aarch64-unknown-linux-gnu S=a9c168dbea81a84671c6b7a5c1795cb0eb699020110d107ddfca427bd797e697 ;; \
      amd64) T=x86_64-unknown-linux-gnu S=5be191e8c70a5007074254cb74d78dbfdd25cd5a61c27de7b8e5410122ec24f4 ;; \
    esac && \
    curl -sSfo /tmp/rustup-init https://static.rust-lang.org/rustup/archive/1.29.2/$T/rustup-init && \
    echo "$S  /tmp/rustup-init" | sha256sum -c - && chmod +x /tmp/rustup-init && \
    CARGO_HOME=/usr/local/cargo /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain 1.99.1 && \
    rm /tmp/rustup-init
# the image's `flash_attn` (FA2 with an FA4 beta) would hide the locked flash_attn.cute
RUN python3 -m pip uninstall -y flash-attn flash-attn-4

# dependency layer, rebuilt only when pyproject.toml or uv.lock change
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --system-site-packages --python /usr/bin/python3 $VIRTUAL_ENV && \
    uv sync --frozen --active

# evalplus (pyproject.toml's pin) untrusted_check shares progress, stat and details with the sample's process through
# lock-guarded multiprocessing objects; a sample killed at its time-out while it holds the lock deadlocks the parent's
# read (evalplus/eval/__init__.py:274). The child is their only writer: they take no lock, and the parent joins the
# child after terminate/kill, so it reads them only once the writer has exited.
RUN <<'EOF' python
import compileall, pathlib, sysconfig
path = pathlib.Path(sysconfig.get_path("purelib")) / "evalplus" / "eval" / "__init__.py"
with open(path, newline="") as f:
    source = f.read()
for old, new in (
    ('progress = Value("i", 0)\r\n', 'progress = Value("i", 0, lock=False)\r\n'),
    ('stat = Value("i", _UNKNOWN)\r\n', 'stat = Value("i", _UNKNOWN, lock=False)\r\n'),
    ('details = Array("b", [False for _ in range(len(inputs))])\r\n',
     'details = Array("b", [False for _ in range(len(inputs))], lock=False)\r\n'),
    ("    if p.is_alive():\r\n        p.kill()\r\n        time.sleep(0.1)\r\n",
     "    if p.is_alive():\r\n        p.kill()\r\n        time.sleep(0.1)\r\n    p.join()\r\n"),
):
    assert source.count(old) == 1, old
    source = source.replace(old, new)
with open(path, "w", newline="") as f:
    f.write(source)
assert compileall.compile_file(str(path), quiet=1)
EOF

# openai/human-eval-infilling: the checkout on sys.path (.pth; read_problems reads <package>/../data; its requirements
# are in pyproject.toml), execution enabled (its README's step) and HumanEval-RandomSpan.jsonl.gz, the name
# data.py:13 reads, linked to the shipped file
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
# compiler, kernel and user caches in the mounted artifacts/cache (paths.py CACHE), reused across containers
ENV XDG_CACHE_HOME=/workspace/nanochat/artifacts/cache
ENV HF_HOME=$XDG_CACHE_HOME/huggingface HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TORCHINDUCTOR_CACHE_DIR=$XDG_CACHE_HOME/inductor TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton \
    QUACK_CACHE_DIR=$XDG_CACHE_HOME/quack FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
    FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=$XDG_CACHE_HOME/flash_attn CARGO_HOME=$XDG_CACHE_HOME/cargo \
    OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

# NGC's entrypoint sources entrypoint.d before executing the command: the artifact and cache directories (paths.py)
RUN echo 'python /workspace/nanochat/paths.py || exit' > /opt/nvidia/entrypoint.d/90-nanochat-paths.sh

# LiveCodeBench/LiveCodeBench's lcb_runner (eval/livecodebench.py): the checkout on sys.path (.pth; its pyproject.toml
# requires vllm, torch and API clients, which the evaluation path does not import; that path's datasets, numpy, pandas
# and tqdm are already installed)
RUN git clone https://github.com/LiveCodeBench/LiveCodeBench /opt/LiveCodeBench && \
    git -C /opt/LiveCodeBench checkout 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24 && \
    echo /opt/LiveCodeBench > "$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/lcb_runner.pth"
