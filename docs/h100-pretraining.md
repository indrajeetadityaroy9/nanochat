# Pretraining a code model on one H100

This document describes the production configuration of this fork and how to reproduce it: a Python and C++
code-generation base model, pretrained once on every unique token of its data on a single NVIDIA H100 80 GB SXM5
(sm_90). The code is the specification. Each section names the file that defines a setting, and every number below is
read from that file or measured in `benchmarks/h100.txt`.

## Model

`nanochat/model/gpt.py` is nanochat (karpathy/nanochat 92d63d4) plus the measured wins of its experiment log, with
Qwen3's QK norm with learnable gains. As in DeepSeek-V3's `model.py` and Qwen3's Hugging Face implementation, each
layer is a `Block` of an `Attention` and an `MLP` module, each applied to the RMS-normed stream and added back to it;
`GPT` stacks the blocks between the embeddings and `lm_head`.

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

**Why clip.** A paired A/B at d8, 20 tokens per parameter, compared clipping at 1.0 with no clipping
(`benchmarks/h100.txt`). Both arms used the same seed and data order, and the training of
that comparison was deterministic, so the clipping is the only difference between them. With clipping:

- Mean validation bits per byte over all 11 corpora fell from 0.88119 to 0.87806.
- Every corpus scored lower: Python 0.74504 → 0.74209, C++ 0.68317 → 0.67994.

## H100 implementation

The model's ops live in `nanochat/ops/` and their CUDA sources in `nanochat/csrc/` (built on first use by
`native.cuda_library`). Each op is a `torch.library` op, so `torch.compile(fullgraph=True)` traces one graph per stage.

| Op | Implementation |
|---|---|
| Projections, MLP, LM-head GEMMs | QuACK SM90 GEMMs with a per-role tile table measured at M = 98,304 (`quack_gemm.TABLE`, `benchmarks/gemmsweep.py` → `gemmtable.py`); weight gradients use cuBLAS `torch.mm` or serial split-K, whichever is most accurate within 1% of the fastest |
| ReLU² MLP | QuACK fused `gemm_act` / `gemm_dact` |
| Cross entropy | QuACK, from bf16 logits read once; the loss and the logit gradient in one pass, the gradient written over the logits |
| Attention | FlashAttention-4 varlen, its default (fastest) backward |
| q/k norm + gain + RoPE | `csrc/qk_prologue.cu`: one pass per direction; the backward is persistent and reduces the gain gradient without atomics (row math shared with the decoder through `csrc/qk_row.cuh`) |
| Embeddings | `F.embedding` with the bf16 cast fused by Inductor; its backward adds rows with atomics |
| Decoding | `csrc/decode.cu` (wrapper `inference/decoder.py`): batched native decoder for the code benchmarks |

Measured on the d24 step with real data (`benchmarks/profile_step.py 24 12 12 file`), before the switch to the fastest
(atomic) gradient reductions, which take the step from 3,764 to 3,724 ms in `benchmarks/h100.txt`:

- 3,764 ms per step, 313.4K tok/s.
- 99% of the step's time is GPU kernels.
- Peak memory 69.5 GiB.
- Kernel time by category: GEMM 51%, FA4 26%, Inductor 14%, other 7%, cross entropy 1.5%.

## Reproducibility

- **Fastest kernels over reproducibility.** Gradients are reduced by the fastest kernels, FA4's and Inductor's embedding
  backward with atomics, so repeated steps differ in the last bits. Fixed-order reductions (FA4's deterministic
  backward and a sorted embedding backward) made the d24 step bitwise reproducible at +1.1% per step (3,764 ms against
  3,724 ms, `benchmarks/h100.txt`).
- **Resume.** A run killed and resumed from its checkpoint continues it (`tests/smoke.sh`).
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
tests/all.sh                                              # every check below
tmux new -d -s h100 'cd ~/projects/nanochat && runs/h100.sh'
```

What `runs/h100.sh` does:

1. Derives the ratio and the tag `d24-r490.805-s42` (`python -m nanochat.train.schedule`: depth, ratio, seed).
2. Trains both stages, saving a checkpoint every 812 steps (1%) and keeping (`--keep`) the model before the repository
   stage, at the configuration its weights trained in (8K, RoPE base 10^4).
3. Relaunches after a failure, as long as the failed attempt saved a newer checkpoint.
4. Runs `scripts/base_eval.py` on the final checkpoint:
   - validation bits and bytes per corpus at 8K, and at 32K per context bucket (≤8K, 8–16K, >16K tokens of context);
   - one mixture bits per byte per group (`loss_eval.mixture_bpb`: val rows drawn as training draws them);
   - reference-solution bits per byte;
   - EvalPlus HumanEval+/MBPP+, MultiPL-E C++ and HumanEval infilling, each generated as its official generator does
     and scored by its official evaluator.
5. Writes its tables to `~/nanochat-runs/results/<tag>.md` (`scripts/report.py`).

**Expected cost:** about 85 h at the measured 313.4K tok/s, about $365 at $4.29/h. This is an estimate: the
repository stage is 5.5% of the steps and was not re-profiled with the final code.

Training metrics go to `~/nanochat-runs/metrics/<tag>.jsonl` (`train/metrics.py`); runner logs to
`~/nanochat-runs/logs/h100/`. Every command runs in the image through `runs/container.sh`.

**Publishing the model:** `runs/release.sh <tag> <repo>` uploads the run's checkpoint, metrics and evaluation to a
Hugging Face model repository in the experiment-directory layout; `hf download <repo> --local-dir ~/nanochat-runs`
puts them back where `load_model`, `base_eval` and a resumed `base_train` read them.

## Verification

`tests/all.sh` runs these checks on the H100, all on the data pipeline's rows. Figures are from the run on the released
code.

| Check | Result |
|---|---|
| `model_check`: compiled d8 vs an independent fp64 reference on the first micro-batch of each stage (random init, perturbed) | loss rel 3.5e-5 / 5.8e-6 (8K), 1.2e-6 (32K); no gradient worse than bf16 eager; the same for each switch off |
| `model_check`, deletion equivalence: each switch off vs the canonical model with that component at its identity | losses bitwise equal; gradients equal within one bf16 rounding |
| `decoder_check` (in `smoke.sh`): native decoder vs eager greedy decoding on the smoke's trained model | 32/32 identical, and with a stop rule and more requests than slots 96/96 |
| `smoke.sh`: d8 both stages, killed after the switch and resumed | the relaunch resumes at the switch and trains to the end; the kept switch checkpoint survives |

The fp64 reference (`tests/reference.py`) restates the window and value-embedding rules itself rather than importing
them, so a bug in the model's rules shows up as a mismatch.

## Appendix studies

`experiments/studies.py` defines the paper's appendix studies: each a set of arms, a decision rule fixed before its
arms ran, and an analysis. `experiments/run.sh <study> ...` trains and evaluates every arm without a record
(`experiments/arm.py`; an arm is identified by its configuration, so a run shared by studies trains once), then writes
the study's tables. The metric of every study is the code group's mixture bits per byte of the 8K val rows.

| Study | Scale | Design | Question |
|---|---|---|---|
| `noise` | d8, 100 TPP | canonical model, 4 init × 2 data seeds | run-to-run SD; ANOVA variance components |
| `screen` | d8, 100 TPP | 2^(7−3) resolution-IV fraction (`pydoe.fracfact`), common random numbers | which components earn their place: value embeddings, x0, resid, QK gains, windows; WSD vs D2Z; fixed vs τ-law weight decay |
| `long_context` | d8, 100 TPP | 2 × 2 | the repo stage's schedule (continue / re-warm) × RoPE base (switch / one base), at 32K per context bucket and at 8K |
| `confirm` | d24, 0.5B tokens (the base also 1B) | screen verdict vs canonical; two candidate additions on the verdict; 2 seed pairs | the pruned architecture at the target scale; Block Attention Residuals and p-RoPE + partial key offset at equal H100 time |
| `lr_horizon` | d24, 0.5/1/2B tokens | peak lr × {0.5, 1, 2, 4} of the law's | the optimal lr per horizon, extrapolated to the production run |
| `optimizer` | d24, 0.5B tokens | AdamW at the same lr grid | Muon+AdamW vs AdamW, each at its fitted optimum |
| `batch` | d24 | gradient noise scale at the kept checkpoints | the critical batch's trend against the production batch |

Decision rules:

- **Screen** (main effects by OLS on the ±1 design; each tested one-sided against the noise SD with the noise arms'
  degrees of freedom, Holm across the seven factors, and against Lenth's margin of error):
  - a component that costs parameters stays only if it significantly helps;
  - the windows, which save compute, go only if they significantly hurt;
  - a regime alternative replaces the current choice only if it is significantly better.
- **Confirm:** the architecture verdict stands unless the pruned model loses on both seed pairs. A candidate addition is
  adopted if on both seed pairs it beats the base at equal H100 time: the base's metric at the tokens it trains in the
  candidate's wall-clock (the arms' median tok/s), interpolated in log tokens between its 0.5B and 1B runs.
  - Block Attention Residuals (Chen et al., arXiv 2603.15031): the residual stream as softmax attention over depth,
    8 blocks of sublayer outputs plus the embedding, a zero-initialized query per sublayer; at 194M activated
    parameters val loss 1.931 → 1.909. It replaces the resid and x0 scalars.
  - p-RoPE with the partial key offset on the full-context layers (modded-nanogpt record #49): RoPE on the
    higher-frequency half of the rotation pairs, and the other half of each key's dims taken from the previous token
    of its document, so one layer can match a token and read what followed it.
  - The native decoder (`csrc/decode.cu`) runs the canonical mechanism; an adopted candidate needs it there before
    production's code-generation evaluation.
- **lr horizon:** the extrapolated lr replaces the law's only if its 95% interval excludes it. Each horizon's optimum is
  the vertex of a quadratic in ln lr with its delta-method SE; the line of ln η* on ln D is fitted by WLS over the
  horizons with an interior optimum.

With the WSD schedule and fixed weight decay, an lr's horizons share one constant-lr phase, so each longer horizon
continues the shorter one's checkpoint kept at its cooldown start: the same run, for 2.3B rather than 3.5B
tokens per lr. The noise SD is between seeds; the screen's contrasts at shared seeds carry the divergence of a changed
component, which it bounds.

Outputs, in `results/studies/`: `<study>.md` (the appendix table) and `<study>.csv`, and three tidy tables for figures:
`arms.csv` (one row per study arm: every Variant field, the plan, the optimizer hyperparameters, the train summary,
the group bits per byte), `eval.csv` (bits, bytes and rows per arm, evaluation, corpus and context bucket) and
`curves.parquet` (one row per arm and step; a branch's preceded by its parent's steps).

## Measured design decisions

Each entry has its measurement in `benchmarks/h100.txt`:

- **Micro-batch 98,304** rather than 65,536: 1% faster per token, peak 69.5 GiB of the 80 GB.
- **FlashAttention-4** rather than cuDNN attention (Transformer Engine): faster forward and backward in all four
  measured window/segment configurations.
- **CUDA q/k prologue** rather than Inductor's fusion: the step went from 298.2K to 305.8K tok/s; the backward later went
  from 168 ms to 93 ms per step.
- **Value-embedding gate as an fp32 broadcast-sum** rather than fp32 GEMMs on CUDA cores: 0.88 vs 1.20 ms per layer,
  error unchanged (5e-8 vs fp64). bf16 would be faster but has gradient error 3e-3.

## Layout

- `nanochat/config.py`: context, vocabulary, mixture, micro-batch, stages, run tag.
- `nanochat/model/gpt.py`: the model (`Attention`, `MLP`, `Block`, `GPT`), its scaling parameters and FLOPs per token.
- `nanochat/ops/`: the ops above; `nanochat/csrc/`: their CUDA sources and the decoder's.
- `nanochat/train/`: `optim.py`, `schedule.py` (plan and lr multiplier), `stages.py` (token budget, rows per stage),
  `checkpoint.py`, `metrics.py`.
- `nanochat/eval/`: bits and bytes per context bucket and the mixture bits per byte (`loss_eval.py`), the code
  benchmarks.
- `nanochat/inference/`: the native decoder.
- `nanochat/data/`: the data pipeline, including the Rust tokenizer, repository and packing crate in `bpe/`.
- `scripts/`: `base_train.py`, `base_eval.py`, `report.py` (results tables).
- `runs/`: `setup.sh` (fresh host), `container.sh` (run a command in the image), `h100.sh` (the production run),
  `release.sh` (publish a run).
- `experiments/`: the appendix studies (`studies.py`: arms, decision rules, analyses, tidy tables; `arm.py`: one arm;
  `noise_scale.py`: the gradient noise scale of a checkpoint; `run.sh`).
- `results/`: the tables the paper reports.
- `tests/`: the checks above; `all.sh` runs them.
- `benchmarks/`: the benchmarks, the GEMM sweep record and `h100.txt`, the measurement record the code cites.
- `docker/Dockerfile`: the image.
