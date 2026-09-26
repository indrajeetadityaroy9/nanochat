# nanochat

![scaling laws](dev/scaling_laws_jan26.png)

nanochat is a minimal experimental harness for LLM training research on NVIDIA GPUs, from one GPU to multi-node clusters. It covers tokenization, pretraining, supervised finetuning, reinforcement learning, evaluation, and inference in one small, hackable codebase. Models are configured by a single complexity dial: `--depth`, the number of transformer layers. Width, head count, batch size, learning rates, weight decay, and training horizon are all derived from it, so sweeping depth yields a miniseries of compute-optimal models (GPT-2 capability is around d24–d26).

## Time-to-GPT-2 Leaderboard

The primary benchmark is "time to GPT-2": the wall-clock training time needed on an 8XH100 node to exceed the GPT-2 (1.6B) DCLM CORE score of 0.256525. [runs/speedrun.sh](runs/speedrun.sh) always reflects the reference recipe.

| # | time | val_bpb | CORE | Description | Date | Commit | Contributors |
|---|-------------|---------|------|-------------|------|--------|--------------|
| 0 | 168 hours | - | 0.2565 | Original OpenAI GPT-2 checkpoint | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 baseline, slightly overtrained | Jan 29 2026 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 slightly undertrained **+fp8** | Feb 2 2026 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | bump total batch size to 1M tokens | Feb 5 2026 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | change dataset to NVIDIA ClimbMix | Mar 4 2026 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch [round 1](https://x.com/karpathy/status/2031135152349524125) | Mar 9 2026 | 6ed7d1d | @karpathy |
| 6 | 1.65 | 0.71800 | 0.2626 | autoresearch round 2 | Mar 14 2026 | a825e63 | @karpathy |

See [dev/LEADERBOARD.md](dev/LEADERBOARD.md) for how time is measured and how to interpret results.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra gpu    # CUDA (A100/H100/etc.)
uv sync --extra cpu    # (or) CPU-only / MPS
source .venv/bin/activate
```

`uv sync --extra gpu --group dev` adds pytest, matplotlib, ipykernel, and python-dotenv.

Datasets (raw corpora, compiled token shards, task data, the CORE bundle) live under `$NANOCHAT_BASE_DIR/data`; put it on fast local NVMe. Outputs (tokenizer, checkpoints, eval results) live under `$NANOCHAT_BASE_DIR`, default `~/.cache/nanochat`. For NGC/Docker, see [Containers](#containers-ngc--docker).

## Reference pipeline

[runs/speedrun.sh](runs/speedrun.sh) runs the full pipeline; on an 8XH100 node it takes ~1.5 hours:

```bash
python -m nanochat.tokenizer                      # BPE tokenizer, vocab 32768
python -m nanochat.data.compile --max-files=170   # fetch, tokenize + pack once into Megatron .bin/.idx shards
torchrun --standalone --nproc_per_node=gpu -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8
torchrun --standalone --nproc_per_node=gpu -m scripts.base_eval -- --device-batch-size=16   # CORE, bpb, samples
torchrun --standalone --nproc_per_node=gpu -m scripts.chat_sft
torchrun --standalone --nproc_per_node=gpu -m scripts.chat_eval -- -i sft                  # ChatCORE
```

Optional stage: `scripts.chat_rl` (GRPO-style RL on GSM8K, evaluate with `chat_eval -i rl`).

- `--nproc_per_node=gpu` starts one rank per visible GPU; any number of GPUs works (the run scripts read `NPROC_PER_NODE`). The total batch size is fixed and gradient accumulation makes up the difference, so results are ~identical on 1 or 64 GPUs; an automatic batch size is rounded to a whole number of micro-batches across the ranks.
- On a single GPU, `torchrun` can also be omitted. A100 nodes work, just slower.
- With less than 80GB of VRAM, reduce `--device-batch-size` (32 → 16, 8, 4, ...).

## Data

All dataset and data-loading code lives in [nanochat/data](nanochat/data). Pretraining corpora are tokenized and packed once into immutable binary shards on local disk, which training memory-maps in place.

```
HF Hub parquet, pinned commits          sources.py    raw corpus registry, files fetched on demand
  ▼
drop eval-contaminated docs, tokenize,  compile.py    once, CPU-parallel, deterministic, resumable
BOS-aligned best-fit pack               decontam.py   13-gram overlap with CORE / chat-eval items
  ▼
Megatron .bin/.idx shards + index.json  shards.py     on local NVMe, under <base_dir>/data/compiled
  ▼
mmap'd in place                         stream.py     elastic deterministic order, DataLoader workers, pinned memory
  ▼
GPU
```

Post-training data sits alongside it: `tasks/` (SFT, RL and chat-eval task datasets, pinned HF parquet), `sft.py` (the SFT conversation packing loader) and `eval_bundle.py` (CORE data). Every file is downloaded on first use through `storage.py`, under a per-file lock, so the ranks on a node download it once.

### Corpora

Pretraining corpora are registered in `DATASETS` in [nanochat/data/sources.py](nanochat/data/sources.py), each pinned to a commit. `--dataset` selects one for tokenizer training, compilation and pretraining; it is recorded in the checkpoint, so `base_eval` measures bpb on the same data:

```bash
python -m nanochat.tokenizer --dataset=dclm_100b
python -m nanochat.data.compile --dataset=dclm_100b --max-files=20
torchrun --standalone --nproc_per_node=gpu -m scripts.base_train -- --dataset=dclm_100b --depth=12
```

| `--dataset` | HuggingFace repo | Released | License | Files |
|---|---|---|---|---|
| `climbmix` (default) | karpathy/climbmix-400b-shuffle | 2025 | MIT card, CC-BY-NC-4.0 upstream | 6,543 × ~250M chars |
| `smol_pdfedu_dclm_fwedu` | HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled | 2026 | ODC-BY | 100 × ~1B tokens |
| `smol_pdf_dclm_fwedu` | HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled | 2026 | ODC-BY | 100 × ~1B tokens |
| `dclm_100b` | HuggingFaceFW/dclm_100BT-shuffled | 2026 | ODC-BY | 100 × ~1B tokens |
| `fineweb_edu_100b` | HuggingFaceFW/fineweb_edu_100BT-shuffled | 2026 | ODC-BY | 100 × ~1B tokens |
| `finepdfs_edu_100b` | HuggingFaceFW/finepdfs_edu_100BT-shuffled | 2026 | ODC-BY | 100 × ~1B tokens |
| `txt360_v2_web` | IFM/TxT360-v2 (`web-high-medium`) | 2026 | CC-BY-4.0 | 828 × ~2GB |

`--max-files` counts raw files, whose sizes differ between corpora, so size it by the tokens a run needs. The tokenizer lives in `$NANOCHAT_BASE_DIR/tokenizer` and is retrained per corpus; use a separate `NANOCHAT_BASE_DIR` per corpus to keep several side by side.

To add a corpus, add a `DatasetSpec(repo, train_files, val_files, revision)` entry:

- Parquet files with one document per row and raw text in a `text` column (only that column is read).
- `val_files` selects the held-out file(s); they are excluded from train.
- `revision` pinned to a commit sha. Gated repos need `HF_TOKEN` (or `hf auth login`) and accepted terms.

### Compiled shards

`python -m nanochat.data.compile` tokenizes each raw file once and packs documents into rows of `seq_len + 1` tokens that each start with a document's BOS: the longest buffered document that fits goes next, and when none fits the shortest is cropped to fill the row, so there is no padding. The buffer holds `--buffer-docs` documents (default 16000): a fuller buffer finds more exact fits, which on ClimbMix halves the tokens lost to cropping (22% at 1000, 12% at 16000, against an 11% floor set by documents longer than a row, of which only the first row is kept). Files are compiled in parallel (`--workers`, default all CPUs), the output is identical for any number of workers, and an interrupted compile resumes where it stopped.

Shards use Megatron-LM's indexed dataset format (`.bin`/`.idx`, one packed row per sequence), the format Megatron-LM and NeMo pretrain from: `megatron.core`'s `IndexedDataset` reads them back row for row (checked in [tests/test_data_megatron_compat.py](tests/test_data_megatron_compat.py)). nanochat reads them with its own small mmap reader rather than importing `megatron.core`, which pulls in the whole Megatron framework and triton. An `index.json` per split records the tokenizer fingerprint, sequence length, source and a blake2b digest of each shard's rows, which a resumed run checks. Compiled data is keyed by corpus, sequence length and tokenizer fingerprint (`$NANOCHAT_BASE_DIR/data/compiled/<dataset>-T<seq_len>-<fingerprint>/`), so changing any of them needs a recompile. NeMo Curator's `MegatronTokenizerWriter` is not used: it needs a Hugging Face `AutoTokenizer` and writes whole documents, while nanochat trains its own tokenizer and trains on BOS-aligned packed rows.

### Decontamination

Before tokenization, compile drops every document that shares a 13-word sequence (`--decontam-ngram`, 0 disables) with an item of the evaluation sets nanochat reports: every CORE task in the eval bundle's `core.yaml`, and the ARC-Easy, ARC-Challenge, MMLU and GSM8K test splits and HumanEval used by `chat_eval` ([nanochat/data/decontam.py](nanochat/data/decontam.py)). Words are lowercased alphanumeric runs, so formatting differences do not hide a match. The corpora are not decontaminated by their publishers (Nemotron-CC, the bulk of ClimbMix, explicitly is not), and without this step benchmark text seen in pretraining inflates CORE and ChatCORE. The eval-set digest is part of the compile settings, so changing the eval sets forces a recompile, and `index.json` records how many documents were dropped.

### Loading

- Each epoch permutes the shard order and shuffles rows within blocks of 16 shards. The global row order depends only on the seed and the data, never on the number of GPUs, nodes or workers, and every optimizer step covers the same rows at any GPU count. The checkpoint stores only the rows consumed, so a run resumes exactly, even on a different number of GPUs.
- `--data-workers` DataLoader processes per rank gather rows from the mmap'd shards into pinned memory, and batches are copied to the GPU asynchronously. There is no tokenization or packing on the training nodes.

## Containers (NGC / Docker)

[docker/Dockerfile](docker/Dockerfile) builds on an NGC PyTorch image by default and keeps NVIDIA's PyTorch, CUDA, NCCL and cuDNN as built. nanochat's other dependencies go on top ([docker/ngc_requirements.py](docker/ngc_requirements.py)): packages the image lacks at their `uv.lock` version, NVIDIA's own builds (every package with a local version label, such as `torch 2.9.0a0+145a3a7`) pinned as installed, and the image's other packages kept unless nanochat needs a newer one, which then moves to its `uv.lock` version. `nvcr.io/nvidia/pytorch:25.10-py3` ships PyTorch 2.9 (the release nanochat pins) with CUDA 13.0, so the host needs a CUDA 13-capable driver; `--build-arg BASE_IMAGE=ubuntu:24.04 --build-arg TORCH=lock` instead installs the exact locked environment with PyTorch 2.9.1 CUDA 12.8 wheels (`--build-arg EXTRA=cpu` for CPU-only data preparation nodes).

```bash
docker build -f docker/Dockerfile -t nanochat .
docker compose -f docker/compose.yaml run --rm prepare  # train tokenizer, fetch raw corpus, compile
docker compose -f docker/compose.yaml run --rm train    # train on every GPU
```

[docker/compose.yaml](docker/compose.yaml) wires this up; configure it with environment variables: `DATASET`, `NUM_FILES`, `SEQ_LEN`, `TRAIN_ARGS`, `HF_TOKEN`, `WANDB_API_KEY`. Mount node-local NVMe as `DATA_HOST`; it holds raw corpora and compiled shards. The GPU count is never fixed: `NPROC_PER_NODE` defaults to one rank per visible GPU. For multi-node runs, every node reads the tokenizer and compiled shards locally: give each node the same `RUNS_HOST` and `DATA_HOST` contents (a shared filesystem, or a copy made after `prepare`), then start `train` on every node with torchrun rendezvous flags in `TORCHRUN_ARGS` (e.g. `--nnodes=4 --node-rank=0 --rdzv-endpoint=head:29500`).

Without compose, run the image with `--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864`, an output volume at `/runs` and the data volume at `/runs/data`: DataLoader workers hand batches to the trainer through shared memory, which Docker otherwise limits to 64MB, and NCCL needs it too.

## Experiments

For quick iteration (~5 min pretraining runs), train a 12-layer model:

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=gpu -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

This logs to wandb (run name "d12"), runs the CORE metric only on the last step, and skips sampling and intermediate checkpoints. Metrics to compare runs on:

1. `val_bpb` (validation loss in vocab-size-invariant bits per byte) against `step`, `total_training_time`, and `total_training_flops`.
2. `core_metric` (the DCLM CORE score).
3. VRAM utilization, `train/mfu` (model FLOPS utilization), `train/tok_per_sec` (throughput).

Changes must be principled enough to hold across all depths, not just the one tested. Two sweep drivers check this:

- [runs/miniseries.sh](runs/miniseries.sh): trains d12–d26 at the default data:param ratio and writes a results CSV.
- [runs/scaling_laws.sh](runs/scaling_laws.sh): trains depths 10–20 at fixed FLOP budgets (1e18–1e19) and writes a resumable CSV, analyzed in [dev/scaling_analysis.ipynb](dev/scaling_analysis.ipynb).

Experiment history, including negative results, is in [dev/LOG.md](dev/LOG.md).

## Precision / dtype

nanochat does not use `torch.amp.autocast`. Precision is managed explicitly through a single global `COMPUTE_DTYPE` (defined in `nanochat/common.py`), auto-detected from the hardware:

| Hardware | Default dtype | Why |
|----------|--------------|-----|
| CUDA SM 80+ (A100, H100, ...) | `bfloat16` | Native bf16 tensor cores |
| CUDA SM < 80 (V100, T4, ...) | `float32` | No bf16; fp16 available via `NANOCHAT_DTYPE=float16` (uses GradScaler) |
| CPU / MPS | `float32` | Safe default. On recent macOS, MPS also runs `NANOCHAT_DTYPE=bfloat16` fine (~25% less memory, similar speed) |

Override the default with the `NANOCHAT_DTYPE` environment variable:

```bash
NANOCHAT_DTYPE=float32 python -m scripts.base_eval --eval=sample           # force fp32
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=gpu -m scripts.base_train  # force bf16
```

Model weights are stored in fp32 (for optimizer precision), and the custom `Linear` layer casts them to `COMPUTE_DTYPE` during the forward pass. Embeddings are stored directly in `COMPUTE_DTYPE` to save memory. This gives the mixed-precision benefit of autocast with explicit control over what runs in which precision.

`float16` training automatically enables a `GradScaler` in `base_train.py` and `chat_sft.py`; RL does not support it yet. Inference in fp16 works everywhere. `--fp8` (CUDA only) converts eligible linear layers to FP8 matmuls with tensorwise scaling (`nanochat/fp8.py`).
