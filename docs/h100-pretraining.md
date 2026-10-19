# Pretraining a code model on one H100

A Python and C++ code-generation base model pretrained once on every unique token of its data on one NVIDIA H100 80 GB
SXM5 (sm_90). The code is the specification; each section names the file that defines a setting, and every measured
number is in `benchmarks/h100.txt`.

## Model

`nanochat/model/gpt.py`: karpathy/nanochat 92d63d4 plus the wins of its experiment log, with Qwen3's QK norm and
learnable gains. Each layer is a `Block` of `Attention` and `MLP`, each applied to the RMS-normed stream and added back.

| | d24 (production) |
|---|---|
| Shape | 24 layers × 768 (`gpt_config`: width 30 × depth rounded up to whole 128-dim heads), 6 heads × 128, MHA |
| Attention | per-head RMS norm → learned per-dim gain → RoPE over the full head, base 10^6; SSSL windows (S = 512), last layer full; FlashAttention-4 varlen with document masking |
| MLP | ReLU², hidden 4C |
| Residual | `x = resid_lambda·x + x0_lambda·x0` per layer; value embeddings on alternating layers (12 of 24) behind a `2·sigmoid` gate |
| Head | untied `lm_head`, plain cross entropy |
| Vocabulary | 2^15 ids (SA-BPE, BOS + 3 FIM sentinels) |
| Parameters | 195.1M scaling parameters (blocks + `lm_head`), plus 13 embedding tables (token + 12 value) of 32,768 × 768 |
| Init | stream-reading matrices and tables uniform with std C^-½; `c_o`, `c_proj`, `lm_head`, gates, depth queries zero; resid lambdas and gains 1, x0 lambdas 0 |
| Precision | fp32 parameters, gradients and optimizer state; bf16 activations and GEMM inputs |

`Mechanism` (in `gpt.py`) holds the switches the scale-up study uses, baseline first: value embeddings, x0, resid, and
two candidate additions, Block Attention Residuals (`attn_res`) and p-RoPE with the partial key offset (`key_offset`).
`GPTConfig` and the trainer's `Variant` inherit it.

## Data and stages

The data pipeline (`nanochat/data/`) prepares the corpora, trains the tokenizer, compiles the documents and packs them
into rows; its output is the Hugging Face dataset `iaroy/nanochat-data`.

- **Mixture** (`config.MIXTURE`): code:text:math = 87:10:3 (DeepSeek-Coder's code share); code is 8 Python and C++
  corpora, text ClimbMix, math two corpora.
- **Objective:** next token, with the packs' fill-in-the-middle pieces read as FIM examples (DeepSeek-Coder §3.1).
- **Stages** (`config.FILE_STAGE`, `config.REPO_STAGE`; Qwen2.5-Coder §3.2): 8,192-token rows of every corpus, then the
  last 300/5,500 of the steps on 32,768-token rows of the repository corpora. One RoPE base, 10^6 (`config.ROPE_BASE`:
  Qwen2.5-Coder's repository-stage base, above Men et al.'s lower bound for 32K, arXiv 2405.14591; Llama 3 trains at a
  large base from the start).
- **Token budget** (`train/stages.py max_ratio`): the most tokens per scaling parameter at which no stage reads a
  document twice: 490.80 at d24, 95.75B tokens.

## Regime

| Setting | Value | Source |
|---|---|---|
| Batch | 1,179,648 tokens (12 micro-batches of 98,304) | DeepSeek LLM laws, B = 0.2920·C^0.3271 (`train/schedule.py`) |
| Peak lr | 9.3795e-4, every group | DeepSeek LLM, η = 0.3118·C^-0.125 |
| Steps | 81,171; repository stage from step 76,743 | `schedule.plan` |
| Warmup | 40 steps, linear | 2/(1−β2) (Ma & Yarats, arXiv 1910.04209) |
| Decay | linear to zero | Bergsma et al., arXiv 2502.15938 (the gain over WSD grows with tokens per parameter) |
| Optimizers | AdamW (fused) on tables and vectors; Muon with `match_rms_adamw` on matrices | Moonlight, arXiv 2502.16982 §2.2 (`train/optim.py`) |
| AdamW | β = (0.9, 0.95) | DeepSeek LLM §2.3 |
| Weight decay | 1/(η τ T), τ = 1.084·TPP^−0.527 of the T steps (0.317 at d24), on tables, `lm_head` and Muon matrices, 0 on vectors | Power Lines, arXiv 2505.13738; studied against 0.1 (DeepSeek LLM §2.3) |
| Gradient clipping | none | studied against global norm 1.0 (DeepSeek LLM §2.3) |
| Repository stage | continues the decay | DeepSeek-Coder §3.6; studied against a re-warmed schedule (SmolLM3) |

With a fixed threshold of 1.0, the clipped fraction of steps ranged from 0.2% to 97% across the configurations of an
earlier d8 screen, and the baseline's median gradient norm sat at the threshold: hence clipping is studied.

## H100 implementation

Ops in `nanochat/ops/`, CUDA sources in `nanochat/csrc/` (built on first use by `native.cuda_library`). Each is a
`torch.library` op, so `torch.compile(fullgraph=True)` traces one graph per stage. Inductor runs with
coordinate-descent tuning (`runs/container.sh`).

| Op | Implementation |
|---|---|
| Projection, MLP, LM-head GEMMs | QuACK SM90 GEMMs, per-role tiles measured at M = 98,304 on real operands (`quack_gemm.TABLE`, `benchmarks/gemmsweep.py` → `gemmtable.py`); weight gradients by cuBLAS or serial split-K, the most accurate within 1% of the fastest |
| ReLU² MLP | QuACK `gemm_act` / `gemm_dact` |
| Cross entropy | QuACK, bf16 logits read once, the logit gradient written over them |
| Attention | FlashAttention-4 varlen, its default backward |
| q/k norm + gain + RoPE | `csrc/qk_prologue.cu`: one pass per direction; gain gradient without atomics (row math shared with the decoder, `csrc/qk_row.cuh`) |
| Block Attention Residuals (candidate) | `csrc/depth_attention.cu`: the paper's two-phase computation (§4.2, Algorithm 1); +12% step time at d24 |
| Embeddings | `F.embedding`, bf16 cast fused by Inductor; backward adds rows with atomics |
| Decoding | `csrc/decode.cu` (`inference/decoder.py`): batched native decoder for the code benchmarks |

The d24 step on real rows (`benchmarks/profile_step.py 24 12 12 file '{}'`): 3,712 ms, 317.8K tok/s, 100% GPU-busy,
peak 69.6 GiB; GEMM 53%, FA4 25%, Inductor 13.5%. A fused q/k/v GEMM measured 3,729 ms and was not adopted.

## Reproducibility

- **Fastest kernels.** FA4's and the embedding backward reduce with atomics, so repeated steps differ in the last bits;
  fixed-order reductions cost 1.1% per step.
- **Resume.** A relaunched run continues from its newest checkpoint (weights, optimizer state, data position), written
  to a temporary file and renamed, so a kill mid-save leaves the previous one.
- **Environment.** `docker/Dockerfile` pins base images by digest and git dependencies by commit; `uv.lock` the Python
  packages; fp32 matmuls stay fp32 (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`).
- **Run identity.** A run's tag (`config.run_tag`) is its depth, ratio and seeds plus a hash of those, its changes from
  the baseline and the git object ids of the code its result depends on (`config.CODE`: `nanochat/`, `scripts/`,
  `docker/`, `pyproject.toml`, `uv.lock`); uncommitted changes there refuse to run. A run never resumes, appends to or
  counts the artifacts of other code or another configuration, and a finished study run (its evaluation exists) is
  skipped. `runs/h100.sh` also logs the commit and the image id at launch.
- **Data order.** Each corpus's rows are permuted once by the data seed (NumPy `RandomState`).

## Reproducing the run

On a fresh Lambda `gpu_1x_h100_sxm5` (Ubuntu with Docker and the NVIDIA driver):

```bash
git clone https://github.com/indrajeetadityaroy9/nanochat ~/projects/nanochat-next && cd ~/projects/nanochat-next
HF_TOKEN=<read token> runs/setup.sh iaroy/nanochat-data   # CDI spec, docker group, image, tokenizer + packed data
# log in again (docker group), then:
experiments/run.sh regime scale_up final
# apply results/studies/final.md to the baseline code, then:
tmux new -d -s h100 'cd ~/projects/nanochat-next && runs/h100.sh'
```

`runs/h100.sh` derives the ratio and the tag (`d24-r490.805-s0-d0-<hash>`), trains both stages with a checkpoint every
1% of the steps, keeping the model before the repository stage, then runs `scripts/base_eval.py` and writes
`~/nanochat-runs/results/<tag>.md` (`scripts/report.py`). Rerunning it on the same code continues a stopped run.
`base_eval` reports validation bits per byte per corpus at 8K and at 32K per context bucket (≤8K, >8K), the mixture
bits per byte per group (`loss_eval.mixture_bpb`), reference-solution bits per byte, and EvalPlus HumanEval+/MBPP+,
MultiPL-E C++ and HumanEval infilling, each generated and scored by its official tools.

**Expected cost:** 81,171 steps at 3.712 s, about 84 h, about $360 at $4.29/h.

`runs/release.sh <tag> <repo>` uploads the checkpoint, metrics and evaluation to a Hugging Face model repository;
`hf download <repo> --local-dir ~/nanochat-runs` restores them where `load_model`, `base_eval` and `base_train` read
them.

## Appendix studies

`experiments/studies.py`: each study is a set of runs, a decision rule fixed before they run, and a report.
`experiments/run.sh <study> ...` trains and evaluates every run (`experiments/train_eval.py`) and writes the study's
report; `final` writes `results/studies/final.md`, the production run's changes from the baseline. Metric: validation
bits per byte of the code mixture at 8K, and for the repository-stage schedule of the repository mixture at 32K.

Adopted without a study, on published evidence: linear decay to zero, one RoPE base, SSSL windows (Gemma 3, arXiv
2503.19786 §5.2), learned QK-norm gains (Qwen3), the x0 and residual scalars (nanochat LOG 2026-01-11), and the peak
lr of DeepSeek LLM's law: an lr-vs-horizon fit at affordable d24 horizons (≤ 2B tokens) extrapolates 38× to
production's tokens with an interval too wide to override it.

| Study | Scale | Design | Decides |
|---|---|---|---|
| `regime` | d8, 100 tokens per parameter | 2³ full factorial × 2 seed pairs (16 runs), seed pair as a block | weight decay τ-law vs 0.1; no clipping vs 1.0; repository stage continue vs rewarm |
| `scale_up` | d24, 0.5B tokens (the first pair's selected run also 1B), 2 seed pairs (9 runs) | the regime decision vs each candidate at equal H100 time | value embeddings removed; Block Attention Residuals; partial key offset |

Decision rules:

- **Regime** (effects by OLS, SEs from the replicates, Holm across the three tests): a hand-set value (weight decay
  0.1, clip 1.0) replaces the baseline's only if significantly better at 8K; rewarm is adopted only if significantly
  better at 32K and not significantly worse at 8K.
- **Scale-up:** a candidate is adopted if on both seed pairs it beats the selected configuration at equal H100 time
  (the selected run's metric moved to the candidate's wall-clock along the first pair's slope per doubling of tokens).
  Removing value embeddings saves ~5% of the step and 302M table parameters; Block Attention Residuals (arXiv
  2603.15031; at 194M activated parameters val loss 1.931 → 1.909) replace the resid and x0 scalars at +12% step time;
  the partial key offset (modded-nanogpt record #49) costs ~0. The native decoder runs the baseline mechanism; an
  adopted addition needs it in `csrc/decode.cu` before production's code-generation evaluation.

Outputs: `results/studies/<study>.md` and `.csv`, `final.md`; the runs' records (`metrics/<tag>.jsonl`,
`evals/<tag>.json`) stay the source.

## Measured design decisions

Each in `benchmarks/h100.txt`:

- **Micro-batch 98,304** rather than 65,536: 1% faster per token, peak 69.6 GiB of 80 GB.
- **FlashAttention-4** rather than cuDNN attention: faster forward and backward in every measured window/segment case.
- **CUDA q/k prologue** rather than Inductor's fusion: 298.2K → 305.8K tok/s.
- **Value-embedding gate as an fp32 broadcast-sum** rather than fp32 GEMMs: 0.88 vs 1.20 ms per layer, same error.
- **Inductor coordinate-descent tuning:** Inductor's kernels 533 → 500 ms per d24 step.
- **Split q/k/v GEMMs** rather than one fused GEMM: 3,712 vs 3,729 ms per d24 step.

## Layout

- `nanochat/config.py`: vocabulary, mixture, micro-batch, stages, RoPE base, run tag.
- `nanochat/model/gpt.py`: `Mechanism`, the model, scaling parameters and FLOPs per token.
- `nanochat/ops/`, `nanochat/csrc/`: the ops above and the decoder.
- `nanochat/train/`: `optim.py`, `schedule.py` (plan, lr multiplier), `stages.py` (token budget, rows per stage),
  `checkpoint.py`, `metrics.py`.
- `nanochat/eval/`: bits and bytes per context bucket and mixture bits per byte (`loss_eval.py`), code benchmarks.
- `nanochat/inference/`: the native decoder.
- `nanochat/data/`: the data pipeline, including the Rust tokenizer, repository and packing crate in `bpe/`.
- `scripts/`: `base_train.py`, `base_eval.py`, `report.py`.
- `runs/`: `setup.sh`, `container.sh`, `h100.sh`, `release.sh`.
- `experiments/`: `studies.py` (runs, rules, reports), `train_eval.py`, `run.sh`.
- `results/`: the tables the paper reports.
- `benchmarks/`: `profile_step.py`, the GEMM sweep (`gemmsweep.py`, `gemmtable.py`, its record) and `h100.txt`.
- `docker/Dockerfile`: the image.
