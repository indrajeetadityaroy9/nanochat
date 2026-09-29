# nanochat

![scaling laws](dev/scaling_laws_jan26.png)

nanochat is a minimal experimental harness for LLM training research on NVIDIA GPUs, from one GPU to multi-node clusters. It covers tokenization, pretraining, evaluation, and sampling of base models in one small, hackable codebase. Models are configured by a single complexity dial: `--depth`, the number of transformer layers. Width, head count, batch size, learning rates, weight decay, and training horizon are all derived from it, so sweeping depth yields a miniseries of compute-optimal models (GPT-2 capability is around d24–d26).

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
uv sync    # Linux, Python 3.12, NVIDIA Hopper or Blackwell: PyTorch's CUDA 13 wheels and FlashAttention-4
source .venv/bin/activate
```

`uv sync --group dev` adds python-dotenv.

Datasets (raw corpora, compiled token rows, task data, the CORE bundle) live under `$NANOCHAT_BASE_DIR/data`; put it on fast local NVMe. Outputs (tokenizer, checkpoints, eval results) live under `$NANOCHAT_BASE_DIR`, default `~/.cache/nanochat`. For NGC/Docker, see [Containers](#containers-ngc--docker).

## Reference pipeline

[runs/speedrun.sh](runs/speedrun.sh) runs the full pipeline; on an 8XH100 node it takes ~1.5 hours:

```bash
python -m nanochat.data.pretrain.fetch --dataset=climbmix --max-files=170   # the only download: 170 raw files + val
python -m nanochat.tokenizer                              # BPE tokenizer, vocab 32768, on the fetched files
python -m nanochat.data.pretrain.compile --max-files=170   # tokenize + pack once into a file of token rows per split
torchrun --standalone --nproc_per_node=gpu -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8
torchrun --standalone --nproc_per_node=gpu -m scripts.base_eval -- --device-batch-size=16   # CORE, bpb, samples
```

nanochat trains base models for next-token prediction: the tokenizer's only special token is `<|bos|>`, which begins every document.

- `--nproc_per_node=gpu` starts one rank per visible GPU; any number of GPUs works (the run scripts read `NPROC_PER_NODE`). The total batch size is fixed and gradient accumulation makes up the difference, so results are ~identical on 1 or 64 GPUs; an automatic batch size is rounded to a whole number of micro-batches across the ranks.
- On a single GPU, `torchrun` can also be omitted. A100 nodes work, just slower.
- With less than 80GB of VRAM, reduce `--device-batch-size` (32 → 16, 8, 4, ...).

## Data

All dataset and data-loading code lives in [nanochat/data](nanochat/data), one package per stage: `pretrain/` and `eval/`. Pretraining data is fetched once into local files, the only network step; the tokenizer, compile and training read only those files. Corpora are tokenized and packed once into one file of token rows per split, which training memory-maps in place (modules of `pretrain/`):

```
HF Hub, Software Heritage, pinned       fetch.py      the only network step: resumable, deterministic, tmux-safe
  ▼
raw/<corpus>/: parquet + manifest.json  sources.py    general-domain corpora as published, code in one schema
  ▼
drop eval-contaminated docs and train   compile.py    once, CPU-parallel, deterministic
copies of val docs, tokenize, split     decontam.py   13-gram overlap with CORE / benchmark items
long docs, lossless best-fit pack
  ▼
{train,val}.bin: packed token rows                    under <base_dir>/data/compiled, on local NVMe
  ▼
numpy.memmap, weighted mixture          stream.py     elastic deterministic order, pinned memory
  ▼
GPU
```

Decontamination reads the benchmarks in `eval/`: the CORE bundle (`core.py`) and the ARC, MMLU, GSM8K and HumanEval test sets, as the tasks of [task.py](nanochat/data/task.py), each reading one split of a pinned HF dataset repo. These files are small and downloaded on first use through `storage.fetch`, which checks for each file under a per-file lock, so the ranks on a node download each once. (huggingface_hub's own lock is not enough: in 2.0.0 every concurrent caller re-downloads the file and first deletes the copy another has just completed.)

### Corpora

Pretraining corpora are registered in `DATASETS` in [sources.py](nanochat/data/pretrain/sources.py), each pinned to a commit. `--dataset` takes a corpus or a weighted mixture of corpora (below) for tokenizer training and pretraining; it is recorded in the checkpoint, so `base_eval` measures bpb on the same data, per corpus:

```bash
python -m nanochat.data.pretrain.fetch --dataset=opc_fineweb_math --max-files=20
python -m nanochat.tokenizer --dataset=opc_fineweb_math
python -m nanochat.data.pretrain.compile --dataset=opc_fineweb_math --max-files=20
torchrun --standalone --nproc_per_node=gpu -m scripts.base_train -- --dataset=opc_fineweb_math --depth=12
```

| Domain | `--dataset` | Source | Released | License | On the Hub |
|---|---|---|---|---|---|
| Code | `stack_edu` | HuggingFaceTB/stack-edu metadata; text from Software Heritage | 2025 | per file (`license_type`) | 167M files, 15 languages |
| Code | `stack_v3` | HuggingFaceCode/stack-v3-train | 2026 | ODC-BY; per file | 8,192 parts, 173M repositories |
| General | `climbmix` (default) | karpathy/climbmix-400b-shuffle | 2025 | MIT card, CC-BY-NC-4.0 upstream | 6,543 × ~250M chars |
| Code-related web | `opc_fineweb_code` | OpenCoder-LLM/opc-fineweb-code-corpus | 2024 | MIT | 510 × ~198k documents |
| Math-related web | `opc_fineweb_math` | OpenCoder-LLM/opc-fineweb-math-corpus | 2024 | ODC-BY | 37 × ~142k documents |

`--max-files` counts local files, whose sizes differ between corpora, so size it by the tokens a run needs: compile reports every file's tokens. The tokenizer lives in `$NANOCHAT_BASE_DIR/tokenizer`; train one on the intended distribution and keep it for every run being compared (see Mixtures).

To add a general-domain corpus, add a `TextCorpus(repo, revision, train_files, val_files)` entry: parquet files with one document per row and its raw text in a `text` column (only that column is read); `val_files` selects the held-out file(s), which are excluded from train; `revision` is pinned to a commit sha. Gated repos need `HF_TOKEN` (or `hf auth login`) and accepted terms.

### Fetching

`python -m nanochat.data.pretrain.fetch --dataset=<corpus> --max-files=N` ([fetch.py](nanochat/data/pretrain/fetch.py)) is the only step that downloads pretraining data. It writes `raw/<corpus>/manifest.json` once, the files of each split in order, then writes every val file and the first N train files that are missing, in parallel: a text corpus's file is downloaded as it is, a code file is materialized by its source ([code/](nanochat/data/pretrain/code)). A file appears only once it is complete, so an interrupted fetch is simply run again, and a larger N adds only the missing files. The tokenizer and compile read a split's files in manifest order up to the first missing one and fail with the fetch command when a corpus has not been fetched far enough.

Run long fetches in tmux so they survive a dropped session (from the repository root; data under `~/nanochat-runs/data`, logs under `~/nanochat-runs/logs`):

```bash
docker build -f docker/Dockerfile --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:26.08-py3 -t nanochat .
mkdir -p ~/nanochat-runs/logs
tmux new-session -d -s fetch-stack_edu -c "$PWD" "bash -o pipefail -c 'docker run --rm --ipc=host \
  --user $(id -u):$(id -g) -e HOME=/tmp --name fetch-stack_edu -v $HOME/.cache/huggingface/token:/tmp/hftoken:ro \
  -e HF_TOKEN_PATH=/tmp/hftoken -e NANOCHAT_BASE_DIR=/runs -v $HOME/nanochat-runs:/runs \
  -v $PWD:/workspace/nanochat:ro -w /workspace/nanochat nanochat \
  python -m nanochat.data.pretrain.fetch --dataset=stack_edu --max-files=8 2>&1 | tee -a $HOME/nanochat-runs/logs/fetch-stack_edu.log'"
tmux attach -t fetch-stack_edu   # watch; detach with Ctrl-b d. `docker stop fetch-stack_edu` stops the fetch
```

### Code corpora

Fetch materializes the two code corpora into zstd parquet files of one row per source file, in 1,024-document row groups: the columns every code file has (text, language, document_id, repository, path, license_type, detected_licenses), then its source's own metadata. Their two sources live in [code/](nanochat/data/pretrain/code).

- `stack_edu` ([code/swh.py](nanochat/data/pretrain/code/swh.py)) publishes metadata only. Each file's text comes from Software Heritage's public S3 bucket by blob ID, fetched anonymously with [obstore](https://developmentseed.org/obstore/) (whose client retries throttled and failed requests), a process per shard and a thread pool in each, and is stored as the file's original bytes decoded with its `src_encoding`. A blob absent from the bucket is dropped; a request that still fails after the retries stops the fetch, which a rerun resumes, so an outage never leaves a shard short of blobs. The metadata is grouped by language, so the DuckDB-built manifest holds out for val the whole repositories first in md5(repository) order that fit in one shard of text, the one val shard, and orders the other files by md5 of their blob ID before cutting ~250 MB shards: every shard mixes all languages.
- Software Heritage holds each file's original bytes, and fetch stores them as they are, decoded with the file's `src_encoding`: no copyright-header, PII or credential rewriting. Such rewriting removed code: Pygments classes preprocessor directives and PHP's `<?php` as comments, so removing a leading copyright comment block also removed the `#include`/`#import` lines after it (on 20,000 sampled RefineCode files, 94.5% of Objective-C and 11.6% of C++ files lost code), and `name@2x.png` asset names became `<EMAIL>`.
- `stack_v3` ([code/stack_v3.py](nanochat/data/pretrain/code/stack_v3.py)) carries the (PII-redacted) text inline, one repository per row, in 8,192 hash-partitioned parts. The manifest orders the parts by md5 of their path: the first part's first row group is val (whole repositories), and every row group of the others is a train file (92,162), read from the Hub by its byte range and flattened to one row per source file, repositories kept whole and in order for later repository-aware work. Building the manifest reads every part's footer, which the Hub rate-limits (the client waits and retries), so it takes several minutes. It is used as published: its publisher near-deduplicated the whole corpus (MinHash, Jaccard ≥ 0.7) and applied StarCoder2's quality filters, so nanochat does not deduplicate it again ([dev/LOG.md](dev/LOG.md) has the measurement).
- Identity: `document_id` is the source's own: the Software Heritage blob ID, the sha1 of the file's bytes; Stack v3's `content_id`, the sha1 of the bytes before redaction.

### General-domain corpora

The general-domain corpora are the pinned Hub parquet files, kept as they are; training reads their `text` column. Measured with a vocab-32768 tokenizer trained on ClimbMix, at seq 2048 (`climbmix` 10 train files and val, `opc_fineweb_math` 1 and val):

| | `climbmix` | `opc_fineweb_math` |
|---|---|---|
| per train file | 85k documents, 252M chars, 54M tokens | 142k documents, 107M tokens |
| documents longer than a row (segmented) | 3.0% | 6.4% |
| dropped as eval-contaminated | 0.07% | 0.26% |
| train documents dropped as exact copies of val documents | 5.6 per file | 0 |

- Val is held out by file. ClimbMix repeats some val documents exactly in its train files, mostly boilerplate pages; compile drops those train copies (Decontamination). Near-duplicates remain: about 50 val documents per ClimbMix train file share at least half their 13-word sequences with it.
- Exact duplicates within ClimbMix: 0.019% of documents.

### Mixtures

`--dataset` takes weights, e.g. `climbmix:0.7,stack_edu:0.1,stack_v3:0.1,opc_fineweb_math:0.1`. Each corpus is compiled on its own, and training interleaves their rows by a deterministic low-discrepancy schedule (global row k goes to the corpus whose cumulative-weight interval contains frac(k·φ)), so every prefix of a run matches the weights and new weights need no rebuild. base_eval reports bpb per corpus.

```bash
MIX=climbmix:0.7,stack_edu:0.1,stack_v3:0.1,opc_fineweb_math:0.1
python -m nanochat.tokenizer --dataset=climbmix,stack_edu,stack_v3,opc_fineweb_math   # once, then kept for every run being compared
for c in climbmix stack_edu stack_v3 opc_fineweb_math; do python -m nanochat.data.pretrain.compile --dataset=$c --max-files=8; done
torchrun --standalone --nproc_per_node=gpu -m scripts.base_train -- --dataset=$MIX --depth=12
```

The tokenizer takes an equal share of `--max-chars` from each corpus and writes the sample's composition and the bytes per token of held-out prose and code, by language, to `tokenizer_stats.json`. Train it once on the intended distribution and keep it for all runs being compared, so a mixture ablation does not also change the tokenizer.

### Compiled rows

`python -m nanochat.data.pretrain.compile` compiles val, then train, and tokenizes each raw file once. A document longer than a row is first split losslessly into consecutive pieces of at most a row, each led by the document's BOS and cut after the last line-ending token that fits (or at the row limit inside a line longer than a row); the pieces of each file are then packed into rows of `seq_len + 1` tokens by best fit over the whole file: the longest remaining piece that fits goes next, and when none fits the shortest fills the row and its rest, again led by the BOS, goes back among the pieces. So every row starts with a BOS, there is no padding, and no token is discarded but those of each file's unfinished last row: on 10 ClimbMix files this gives 262,087 rows where cropping long documents and discarding crop remainders gave 230,122 (12.2% of tokens lost). Keeping long documents' tails matches best-fit packing (Ding et al. 2024) and Megatron-Core; repeating the BOS on continuation pieces does not (Megatron continues a document at the start of the next sample without one), but best fit places a piece anywhere in a row, where the BOS delimits it from the document before it. Row groups, not files, are the unit of parallel work: each task reads, decontaminates, tokenizes and segments one row group, so every CPU is busy however many files a split has (`ProcessPoolExecutor`, so a worker's error, including a Rust panic, stops compile with it). The parent packs each file once its row groups are done and writes the rows in raw file order, so the output does not depend on the number of CPUs. Per file and split, compile writes `<split>.json` next to the compiled file: documents, documents dropped as eval-contaminated or as copies of val documents, segmented documents and their pieces, tokens entering packing, and rows.

A compiled split is one flat file of token rows, `uint16` while the vocabulary fits, at `$NANOCHAT_BASE_DIR/data/compiled/<dataset>-T<seq_len>-<fingerprint>/{train,val}.bin`. It is keyed by corpus, sequence length and tokenizer fingerprint, so changing any of them needs a recompile, and it appears only once compile has finished. `{train,val}.json` beside it records the settings that produced it (corpus, sequence length, tokenizer fingerprint, decontamination n-gram size, the sha256 of the eval n-gram set and the number of val documents train was checked against) and per-file statistics. Training reads it in place with `numpy.memmap`.

### Decontamination

Before tokenization, compile drops every document that shares a 13-word sequence (GPT-3's criterion) with an item of the evaluation sets nanochat reports or keeps clean: every CORE task in the eval bundle's `core.yaml`, and the ARC-Easy, ARC-Challenge, MMLU and GSM8K test splits and HumanEval ([nanochat/data/pretrain/decontam.py](nanochat/data/pretrain/decontam.py)). Words are lowercased alphanumeric runs, so formatting differences do not hide a match; a document's n-gram hashes are looked up exactly, in sorted order, in the sorted eval n-gram hashes. The corpora are not decontaminated by their publishers (Nemotron-CC, the bulk of ClimbMix, explicitly is not), and without this step benchmark text seen in pretraining inflates CORE. An eval item shorter than 13 words cannot match; 98% of the 20,435 such items are bigbench_qa_wikidata's one-line facts, which web text states legitimately. In train, compile also drops every document whose exact text (blake2b hash) is a document of the corpus's val split, so val bpb measures held-out text. Compile prints how many documents it dropped for each reason.

### Loading

- Each epoch of a corpus visits every row once, in a permutation seeded by `--data-seed` and the epoch; validation and `base_eval` read the rows in stored order. A mixture interleaves the corpora by their weights as above. The global row order depends only on the seed, the weights and the data, never on the number of GPUs or nodes, and every optimizer step covers the same rows at any GPU count. The checkpoint stores only the rows consumed, so a run resumes exactly, even on a different number of GPUs.
- The training process gathers each micro-batch from the memory-mapped file into pinned memory and copies it to the GPU asynchronously: on a GB10, 0.09 ms per 16-row micro-batch with the file in the page cache and 0.2 ms from NVMe. There is no tokenization or packing during training.

## Containers (NGC / Docker)

[docker/Dockerfile](docker/Dockerfile) builds on NVIDIA's NGC PyTorch image (`nvcr.io/nvidia/pytorch:26.08-py3`: Python 3.12, PyTorch 2.14, CUDA 13.4, cuDNN 9.25, NCCL 2.30) the way NVIDIA's own stacks (Megatron-LM, Megatron-Bridge, NeMo Automodel, NeMo-RL) do: the image's Python stays as NVIDIA built it, and `uv.lock` is installed into a virtual environment (`/opt/venv`, created with `--system-site-packages`) that sees the image's packages. The environment takes torch, everything only torch needs (Triton, the CUDA libraries) and numpy from the image (`uv export --prune`) and every other package, FlashAttention-4 among them, at its locked version; the image's `torchrun` is pointed at the environment's Python. The host needs a CUDA 13-capable driver.

```bash
docker build -f docker/Dockerfile -t nanochat .
docker compose -f docker/compose.yaml run --rm prepare  # fetch the raw corpus, train the tokenizer, compile
docker compose -f docker/compose.yaml run --rm train    # train on every GPU
```

[docker/compose.yaml](docker/compose.yaml) wires this up; configure it with environment variables: `DATASET`, `NUM_FILES`, `SEQ_LEN`, `TRAIN_ARGS`, `HF_TOKEN`, `WANDB_API_KEY`. Mount node-local NVMe as `DATA_HOST`; it holds raw corpora and compiled data. The GPU count is never fixed: `NPROC_PER_NODE` defaults to one rank per visible GPU. For multi-node runs, every node reads the tokenizer and compiled data locally: give each node the same `RUNS_HOST` and `DATA_HOST` contents (a shared filesystem, or a copy made after `prepare`), then start `train` on every node with torchrun rendezvous flags in `TORCHRUN_ARGS` (e.g. `--nnodes=4 --node-rank=0 --rdzv-endpoint=head:29500`).

Without compose, run the image with `--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864`, an output volume at `/runs` and the data volume at `/runs/data`: NCCL uses shared memory, which Docker otherwise limits to 64MB.

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
- [runs/scaling_laws.sh](runs/scaling_laws.sh): trains depths 10–20 at fixed FLOP budgets (1e18–1e19) and writes a resumable CSV.

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

`float16` training automatically enables a `GradScaler` in `base_train.py`. Inference in fp16 works everywhere. `--fp8` (CUDA only) converts eligible linear layers to FP8 matmuls with tensorwise scaling (`nanochat/fp8.py`).
