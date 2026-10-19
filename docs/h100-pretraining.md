# Pretraining a code model on one H100

This document describes the production configuration of this fork and how to reproduce it: a Python and C++
code-generation base model, pretrained once on every unique token of its data on a single NVIDIA H100 80 GB SXM5
(sm_90). The code is the specification. Each section names the file that defines a setting, and every number below is
read from that file or measured in `bench/h100.txt`.

## Model

`nanochat/model/gpt.py` is nanochat (karpathy/nanochat 92d63d4) plus the measured wins of its experiment log, with
Qwen3's QK norm with learnable gains.

| | d24 (production) |
|---|---|
| Shape | 24 layers × 768 (`gpt_config`: width 30 × depth, rounded up to whole 128-dim heads), 6 heads × 128, MHA |
| Attention | per-head RMS norm → fp32 per-dim gain → RoPE over the full head; windows SSSL (S = 512), last layer full; FlashAttention-4 varlen with document masking |
| MLP | ReLU², hidden 4C |
| Residual | `x = resid_lambda·x + x0_lambda·x0` per layer; value embeddings on alternating layers (12 of 24) behind a `2·sigmoid` gate |
| Head | untied `lm_head`, plain cross entropy, no logit softcap |
| Vocabulary | 2^15 ids (SA-BPE, BOS + 3 FIM sentinels) |
| Parameters | 195.1M scaling parameters (blocks + `lm_head`), plus 13 embedding tables (token + 12 value) of 32,768 × 768 |
| Init | matrices and tables uniform with std C^-½; `c_o`, `c_proj`, `lm_head` and the gates zero; lambdas 1 / 0; gains 1 |
| Precision | fp32 parameters, gradients and optimizer state; bf16 activations and GEMM inputs |

## Data and stages

The data pipeline (`nanochat/data/`) prepares the corpora, trains the tokenizer, compiles the documents and packs them
into rows. Its output is published as the private Hugging Face dataset `iaroy/nanochat-data`: the tokenizer, the
compiled documents and the packed 8K and 32K rows.

- **Mixture** (`config.MIXTURE`): code:text:math = 87:10:3, following DeepSeek-Coder's code share. The code group is
  8 Python and C++ corpora; text is ClimbMix; math is two math corpora.
- **Objective:** next token, with the packs' fill-in-the-middle pieces read as FIM examples (DeepSeek-Coder §3.1).
- **Stages** (`config.FILE_STAGE`, `config.REPO_STAGE`), following Qwen2.5-Coder §3.2:
  - File level: 8,192-token rows of every corpus, RoPE base 10^4.
  - Repository level: the last 300/5,500 of the steps, 32,768-token rows of the repository corpora, RoPE base 10^6.
- **Token budget** (`train/stages.py max_ratio`): the most tokens per scaling parameter at which neither stage reads a
  document twice. For d24 that is 490.80, i.e. 95.75B tokens.

## Regime

| Setting | Value | Source |
|---|---|---|
| Batch | 1,179,648 tokens (12 micro-batches of 98,304) | DeepSeek LLM laws, B = 0.2920·C^0.3271 (`train/schedule.py`) |
| Peak lr | 9.3795e-4, every group | DeepSeek LLM, η = 0.3118·C^-0.125 |
| Steps | 81,171; repository stage from step 76,743 | `schedule.plan` |
| Warmup | 40 steps, linear | 2/(1−β2) (Ma & Yarats, arXiv 1910.04209) |
| Decay | constant, then (1−√) cooldown over the last 20% | Hägele et al., arXiv 2405.18392, via `torch.optim.lr_scheduler.LambdaLR` |
| Optimizers | `torch.optim.AdamW` (fused) on tables and vectors; `torch.optim.Muon` with `match_rms_adamw` on matrices | Moonlight, arXiv 2502.16982 §2.2 (`train/optim.py`) |
| AdamW | β = (0.9, 0.95), weight decay 0.1 (tables, `lm_head`, Muon matrices), 0 on vectors | DeepSeek LLM §2.3 |
| Gradient clipping | global L2 norm 1.0, `torch.nn.utils.clip_grad_norm_` | DeepSeek LLM §2.3, inherited by DeepSeek-Coder §3.4 |

**Why clip.** A paired A/B at d8, 20 tokens per parameter, compared clipping at 1.0 with no clipping (`bench/h100.txt`).
Both arms used the same seed and data order, and training is deterministic, so the clipping is the only difference
between them. With clipping:

- Mean validation bits per byte over all 11 corpora fell from 0.88119 to 0.87806.
- Every corpus scored lower: Python 0.74504 → 0.74209, C++ 0.68317 → 0.67994.

## H100 implementation

The model's ops live in `nanochat/ops/`. Each is a `torch.library` op, so `torch.compile(fullgraph=True)` traces one
graph per stage.

| Op | Implementation |
|---|---|
| Projections, MLP, LM-head GEMMs | QuACK SM90 GEMMs with a per-role tile table measured at M = 98,304 (`quack_gemm.TABLE`, `bench/gemmsweep.py` → `gemmtable.py`); weight gradients use cuBLAS `torch.mm` or serial split-K, whichever is most accurate within 1% of the fastest |
| ReLU² MLP | QuACK fused `gemm_act` / `gemm_dact` |
| Cross entropy | QuACK, from bf16 logits read once; the loss and the logit gradient in one pass, the gradient written over the logits |
| Attention | FlashAttention-4 varlen, deterministic backward |
| q/k norm + gain + RoPE | `qk_prologue.cu`: one pass per direction; the backward is persistent and reduces the gain gradient in a fixed order (row math shared with the decoder through `qk_row.cuh`) |
| Embeddings | `embedding.cu`: bf16 lookup; the gradient sums each token's rows in a fixed order (stable sort, partials of 32 rows, ordered sums) |
| Decoding | `inference/decode.cu`: batched native decoder for the code benchmarks |

Measured on the d24 step with real data (`bench/profile_step.py 24 30 12`):

- 3,764 ms per step, 313.4K tok/s.
- 99% of the step's time is GPU kernels.
- Peak memory 69.5 GiB.
- Kernel time by category: GEMM 51%, FA4 26%, Inductor 14%, other 7%, cross entropy 1.5%.

## Reproducibility

- **Bitwise-deterministic training.** Every gradient reduction runs in a fixed order: FA4's deterministic backward, and
  the q/k prologue and embedding kernels, which use no atomics.
  - `verify/determinism.py` repeats a d24 step 6 times. 0 of 220 parameter gradients differ between repeats; with
    PyTorch's default kernels, 212 differ.
  - A run killed and resumed from its checkpoint reproduces the uninterrupted run's losses bit for bit
    (`verify/smoke.sh`).
  - Cost: +1.5% per micro-batch, against +8% for `torch.use_deterministic_algorithms`.
- **Pinned environment.**
  - `docker/Dockerfile` pins every base image by digest (NGC PyTorch 26.09, uv, Go) and every git dependency by commit;
    `uv.lock` pins the Python packages.
  - The image keeps fp32 matmuls in fp32 (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`).
- **Run identity.** `runs/h100.sh` freezes a copy of the repository for the run. At every launch it records the image
  id and a hash of that copy.
- **Data order.** Each corpus's rows are permuted once by the seed (NumPy `RandomState`). The data state in a
  checkpoint is the number of rows consumed.

## Reproducing the run

On a fresh Lambda `gpu_1x_h100_sxm5` (Ubuntu with Docker and the NVIDIA driver):

```bash
git clone https://github.com/indrajeetadityaroy9/nanochat ~/projects/nanochat && cd ~/projects/nanochat
HF_TOKEN=<read token> runs/setup.sh iaroy/nanochat-data   # CDI spec, docker group, image, tokenizer + packed data
# log in again (docker group), then:
verify/all.sh                                             # every check below
tmux new -d -s h100 'cd ~/projects/nanochat && runs/h100.sh'
```

What `runs/h100.sh` does:

1. Derives the ratio and the tag `d24-r490.805-s42`.
2. Trains both stages, saving a checkpoint every 812 steps (1%).
3. Relaunches after a failure, as long as the failed attempt saved a newer checkpoint.
4. Runs `scripts/base_eval.py` on the final checkpoint: per-corpus validation bits per byte, reference-solution bits
   per byte, EvalPlus HumanEval+/MBPP+, MultiPL-E C++ and HumanEval infilling, each scored by its official evaluator.

**Expected cost:** about 85 h at the measured 313.4K tok/s, about $365 at $4.29/h. This is an estimate: the
repository stage is 5.5% of the steps and was not re-profiled with the final code.

Training metrics go to `~/nanochat-runs/metrics/<tag>.jsonl` (`train/metrics.py`); runner logs to
`~/nanochat-runs/logs/h100/`. Every command runs in the image through `runs/container.sh`.

## Verification

`verify/all.sh` runs these checks on the H100. Figures are from the run on the released code.

| Check | Result |
|---|---|
| `ops_check`: each op vs fp64, next to plain bf16 PyTorch | q/k prologue outputs and input gradients rel 1.7e-3 (bf16 PyTorch 3.9e-3), gain gradients rel 1.9e-7 (bf16 PyTorch 3.8e-3); embedding lookup exact, gradient rel 1.2e-7 and bitwise repeatable |
| `model_check`: compiled d8 vs an independent fp64 reference (random init, perturbed) | loss rel 1.8e-5; no gradient worse than bf16 eager |
| `model_check_data`: the data pipeline's rows at both stages | loss rel 3.5e-5 / 5.8e-6 (8K), 1.2e-6 (32K) |
| `determinism`: d24 step repeated 6 times | 0 of 220 gradients differ |
| `decoder_check`: native decoder vs eager greedy decoding | trained d8: 32/32 identical; random d24: tokens differ only where eager's top two logits are within 0.013 |
| `smoke.sh`: d8 both stages, killed after the switch and resumed | losses on the overlapping steps bitwise equal |

The fp64 reference (`verify/reference.py`) restates the window and value-embedding rules itself rather than importing
them, so a bug in the model's rules shows up as a mismatch.

## Measured design decisions

Each entry has its measurement in `bench/h100.txt`:

- **Micro-batch 98,304** rather than 65,536: 1% faster per token, peak 69.5 GiB of the 80 GB.
- **FlashAttention-4** rather than cuDNN attention (Transformer Engine): faster forward and backward in all four
  measured window/segment configurations.
- **CUDA q/k prologue** rather than Inductor's fusion: the step went from 298.2K to 305.8K tok/s; the backward later went
  from 168 ms to 93 ms per step.
- **Value-embedding gate as an fp32 broadcast-sum** rather than fp32 GEMMs on CUDA cores: 0.88 vs 1.20 ms per layer,
  error unchanged (5e-8 vs fp64). bf16 would be faster but has gradient error 3e-3.
- **Custom deterministic kernels** rather than `torch.use_deterministic_algorithms`: +1.5% vs +8% per micro-batch, both
  bitwise reproducible.

## Layout

- `nanochat/config.py`: context, vocabulary, mixture, micro-batch, stages, run tag.
- `nanochat/model/gpt.py`: the model, its scaling parameters and FLOPs per token.
- `nanochat/ops/`: the ops above, CUDA sources included.
- `nanochat/train/`: `optim.py`, `schedule.py` (plan and lr multiplier), `stages.py` (token budget, rows per stage),
  `checkpoint.py`, `metrics.py`.
- `nanochat/eval/`: bits per byte and the code benchmarks.
- `nanochat/inference/`: the native decoder.
- `nanochat/data/`: the data pipeline, including the Rust tokenizer, repository and packing crate in `bpe/`.
- `scripts/`: `base_train.py`, `base_eval.py`.
- `verify/`: the checks above; `all.sh` runs them.
- `bench/`: the benchmarks, the GEMM sweep record and `h100.txt`, the measurement record the code cites.
- `runs/`: `setup.sh` (fresh host), `container.sh` (run a command in the image), `h100.sh` (the production run).
- `docker/Dockerfile`: the image.
