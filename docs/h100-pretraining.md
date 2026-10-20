# Pretraining a code model on one H100

A Python and C++ code-generation base model pretrained on one NVIDIA H100 80 GB SXM5 (sm_90) at the most tokens its data
holds without any stage reading a document twice. The code is the specification; each section names the file that
defines a setting. Measurements and ablations, with the code that runs them, are in `ablations/` (`ablations.md`); the
pre-launch runs' records are under `artifacts/` by tag and published with the model (`runs/release.sh`).

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
| Init | stream-reading matrices and tables uniform with std C^-½; `c_o`, `c_proj`, `lm_head`, value-embedding gates zero; resid lambdas and gains 1, x0 lambdas 0 |
| Precision | fp32 parameters, gradients and optimizer state; bf16 activations and GEMM inputs |

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
| Weight decay | λ = 1/(η τ T), the EMA timescale τ = 1.084·TPP^−0.527 of the T steps (τ = 0.0414, λ = 0.317 at d24), on tables, `lm_head` and Muon matrices, 0 on vectors | Power Lines, arXiv 2505.13738 (DeepSeek LLM: 0.1, §2.3) |
| Gradient clipping | none; the global gradient norm is logged every step | (DeepSeek LLM: 1.0, §2.3) |
| Repository stage | continues the decay | DeepSeek-Coder §3.6 |

No clipping: a fixed threshold is scale-dependent (at 1.0 the clipped fraction of steps ranged from 0.2% to 97% across
the configurations of an earlier d8 screen). The one direct measurement, a single-seed d8 A/B in the previous regime
(`ablations/ablations.md` §3), had clipping at 1.0 lower on all 11 corpora by 0.003 bits per byte on average, under the
paired comparison's spread of per-pair differences (sd 0.0065 on 8K code); at the production batch and lr the gradient
norm stays below 1.9 after warmup (d24 check below). Clipping is the regime choice the evidence leaves least settled; the
pre-registered screen tests it (`ablations/ablations.md` §1).

## H100 implementation

Ops in `nanochat/ops/`, CUDA sources in `nanochat/csrc/` (built on first use by `native.cuda_library`). Each is a
`torch.library` op, so `torch.compile(fullgraph=True)` traces one graph per stage. Inductor runs with
coordinate-descent tuning (`runs/container.sh`).

| Op | Implementation |
|---|---|
| Projection, MLP, LM-head GEMMs | forward and dX: QuACK SM90 GEMMs autotuned by QuACK (`quack.gemm_interface`, `tuned=True`); fp32 weight gradients: cuBLAS `torch.mm`, faster and more accurate than QuACK's tiles on their small outputs (`ablations/ablations.md`) |
| ReLU² MLP | QuACK `gemm_act` / `gemm_dact` |
| Cross entropy | QuACK, bf16 logits read once, the logit gradient written over them |
| Attention | FlashAttention-4 varlen, its default backward |
| q/k norm + gain + RoPE | `csrc/qk_prologue.cu`: one pass per direction; gain gradient without atomics (row math shared with the decoder, `csrc/qk_row.cuh`) |
| Embeddings | `F.embedding`, bf16 cast fused by Inductor; backward adds rows with atomics |
| Decoding | `csrc/decode.cu` (`inference/decoder.py`): batched native decoder for the code benchmarks |

The d24 step on real rows (`ablations/profile_step.py 24 12 12 file`): 3,705–3,713 ms, 318K tok/s, 100% GPU-busy,
peak 69.6 GiB; GEMM 53%, FA4 25%, Inductor 13%; the repository stage's (`24 12 3 repo`) 4,016 ms. A fused q/k/v GEMM
measured 3,729 ms and was not adopted.

## Reproducibility

- **Fastest kernels.** FA4's and the embedding backward reduce with atomics, so repeated steps differ in the last bits;
  FA4's deterministic backward costs 1–3% of a d24 micro-batch's forward and backward. Inductor's coordinate-descent
  tuning picks kernel configurations by timing, so a rebuilt compiler cache can change reduction orders: the same code
  from two caches agreed bitwise on the first steps and then differed by ~10⁻⁶ relative (d8); from one cache, the
  pre-refactor and refactored code agreed bitwise.
- **Resume.** A relaunched run continues from its newest checkpoint (weights, optimizer state, data position), written
  to a temporary file and renamed, so a kill mid-save leaves the previous one.
- **Environment.** `docker/Dockerfile` pins base images by digest and git dependencies by commit; `uv.lock` the Python
  packages; fp32 matmuls stay fp32 (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`).
- **Run identity.** A run's tag (`config.run_tag`) is its depth, ratio and seeds plus a hash of those and the git
  object ids of the code its result depends on (`config.CODE`: `nanochat/`, `scripts/`, `docker/`, `pyproject.toml`,
  `uv.lock`, `runs/container.sh`); uncommitted changes there refuse to run. The data is the dataset revision
  `config.DATASET_REVISION` (`runs/setup.sh` downloads it), so it is part of the hash too. A run never resumes,
  appends to or counts the artifacts of other code or another configuration. `runs/h100.sh` also logs the commit and
  the image id at launch.
- **Data order.** Each corpus's rows are permuted once by the data seed (NumPy `RandomState`).

## Reproducing the run

On a fresh Lambda `gpu_1x_h100_sxm5` (Ubuntu with Docker and the NVIDIA driver):

```bash
git clone https://github.com/indrajeetadityaroy9/nanochat nanochat && cd nanochat
HF_TOKEN=<read token> runs/setup.sh   # CDI spec, docker group, image, artifacts/, tokenizer + data
# log in again (docker group), then from the checkout:
tmux new -d -s h100 runs/h100.sh
```

Everything a run reads or writes lives in the checkout (`nanochat/paths.py`): `artifacts/` (gitignored, created by
`runs/setup.sh`) holds `data/`, `tokenizer/`, `checkpoints/` (`<tag>.pt`, the resume checkpoint, and
`<tag>-step<NNNNNN>.pt`, the model-only snapshots), `metrics/<tag>.jsonl`, `base_eval/` and `probes/`
(`<tag>[-step<NNNNNN>].json`), `logs/` (`<script>-<tag>.log`) and `cache/`, plus `evals/<tag>.json`, the pre-launch
runs' loss evaluations; `results/` (tracked) holds the tables.

`runs/h100.sh` derives the ratio and the tag (`d24-r490.805-s0-d0-<hash>`), trains both stages with a checkpoint every
1% of the steps and a model-only snapshot (weights, no optimizer state, ~2.1 GB) every 10% and at the repository
stage's start, then runs `scripts/base_eval.py` (greedy and system decoding), the probe set
(`nanochat/eval/probes.py`) and writes `results/<tag>.md` (`scripts/report.py`). Rerunning it on the
same code continues a stopped run. `runs/curves.sh <tag>` then evaluates every snapshot the same way (`--step`) and
tables them with the final model in `results/<tag>-curves.md`: when pass@1 and each probe's robustness emerge.
`base_eval` reports validation bits per byte per corpus at 8K and at 32K per context bucket (≤8K, >8K), the mixture
bits per byte per group (`loss_eval.mixture_bpb`), reference-solution bits per byte, and EvalPlus HumanEval+/MBPP+,
MultiPL-E C++ and HumanEval infilling, each generated and scored by its official tools.

**Expected cost:** 76,743 file-level steps at 3.709 s and 4,428 repository-stage steps at 4.016 s, about 84 h, about
$360 at $4.29/h.

`runs/release.sh <tag> <repo>` uploads the checkpoint and snapshots, every run record under `artifacts/` (this run's and
the pre-launch runs') and `results/` to a Hugging Face model repository; `runs/container.sh hf download <repo>
--local-dir /workspace/nanochat` (the checkout inside the container) restores them where `load_model`, `base_eval` and
`base_train` read them.

## Pre-launch verification

Commit `60138de` on the H100, on the packed rows; records under `artifacts/` by tag.

| Check | Result |
|---|---|
| Resume: d8, 20 tokens per parameter (`d8-r20-s0-d0-d69671850bea`), killed with `docker kill` at step 991 and relaunched | continued from its step-900 checkpoint, switched to the 32K stage at step 2,824 (`repo_start`), finished at step 2,987; the step-2,824 checkpoint kept |
| Production path: `base_train` CLI, `base_eval` (loss, reference solutions, EvalPlus, MultiPL-E, infilling), `report` | all complete |
| Run identity | the production tag stable across calls; an untracked file under `nanochat/` refused |

**Training record**, d8 at 20 tokens per parameter (2,987 steps of 98,304 tokens, peak lr 2.73e-3): mean training loss
per window of steps, one record per step (a resumed run's repeated steps counted once), against runs of the previous
regime (WSD, RoPE base 10^4 → 10^6 at the repository stage; weight decay 0.1 as recorded by `d8-r20-s42`, whose clip
was 1.0; the `-g1`/`-ginf` records predate the weight-decay field). Seeds differ (0 vs 42), so data order differs.

| Run | 0–40 | 40–250 | 250–500 | 500–1000 | 1000–1500 | 1500–2000 | 2000–2500 | 2500–2824 | 2824–2987 (32K) |
|---|---|---|---|---|---|---|---|---|---|
| This configuration (seed 0) | 8.742 | 5.485 | 3.660 | 2.652 | 2.347 | 2.207 | 2.094 | 2.025 | 1.701 |
| Previous regime, clip 1.0 (`d8-r20-s42`) | 8.733 | 5.362 | 3.634 | 2.703 | 2.352 | 2.217 | 2.136 | 2.032 | 1.785 |
| Previous regime, clip 1.0 (`d8-r20-s42-g1`) | 8.738 | 5.347 | 3.553 | 2.623 | 2.332 | 2.209 | 2.130 | 2.026 | 1.808 |
| Previous regime, no clip (`d8-r20-s42-ginf`) | 8.740 | 5.455 | 3.726 | 2.671 | 2.351 | 2.220 | 2.140 | 2.035 | 1.803 |

No non-finite loss or gradient; median gradient norm over steps 1,000–2,824 1.49 (0.77–0.86 for the previous runs,
whose weight decay is 0.1 against this configuration's 0.548 at this scale); throughput 2.3M tokens/s.

**Held-out bits per byte** of the same runs (`artifacts/base_eval/<tag>.json`; the `-g1`/`-ginf` files in the earlier
evaluator's format), unpaired:

| Validation split (8K rows) | This configuration (seed 0) | Previous, clip 1.0 (`-g1`) | Previous, no clip (`-ginf`) |
|---|---|---|---|
| stack_edu_python | 0.7355 | 0.7421 | 0.7450 |
| ultradata_code_l2_python | 0.5551 | 0.5581 | 0.5611 |
| ultradata_stack_v3_l2_python | 0.5818 | 0.5858 | 0.5885 |
| kaggle_notebooks | 0.7773 | 0.7793 | 0.7824 |
| codeforces_cots | 0.9246 | 0.9261 | 0.9277 |
| stack_edu_cpp | 0.6729 | 0.6799 | 0.6832 |
| ultradata_code_l2_cpp | 0.6954 | 0.7011 | 0.7040 |
| ultradata_stack_v3_l2_cpp | 0.6551 | 0.6633 | 0.6662 |
| climbmix (text) | 1.3615 | 1.3468 | 1.3503 |
| opc_fineweb_math | 1.3966 | 1.3848 | 1.3884 |
| ultradata_math_l2 | 1.3013 | 1.2915 | 1.2963 |
| **Reference solutions given the prompt** | | | |
| HumanEval+ | 0.6635 | 0.6604 | 0.6726 |
| MBPP+ | 0.7778 | 0.7798 | 0.7740 |
| HumanEval-X C++ | 0.6281 | 0.6368 | 0.6441 |

Lower on every code corpus, higher on text and math (+0.010 to +0.015); on the benchmarks' reference solutions within
the spread of the two previous runs (0.012 apart on HumanEval+ from clipping alone) except HumanEval-X C++, lower.
This configuration's mixtures: 8K code 0.6354, all 0.7408; 32K repository 0.6200. pass@1 at this scale is about 0
(HumanEval+ 0, MBPP+ base 0.0026, single-line infilling SPM 0.036): the check is that the evaluation runs.

**Paired comparison against the standard regime**, d8 at 20 tokens per parameter, three seed pairs (seed = data seed
∈ {0, 1, 2}; same initialization and data order within a pair). This configuration's arm is `60138de`'s baseline
(`d8-r20-s{0,1,2}-d{0,1,2}-{d69671850bea, ca426bc7cec9, ccd433c30eb2}`, tags reproduced from `60138de`'s `run_tag`);
the standard regime is commit `c00d2f6`'s default (WSD, weight decay 0.1, clip 1.0, RoPE base 10^4 → 10^6), trained in
its own worktree (`d8-r20-s{0,1,2}`). Both arms are evaluated by `60138de`'s `loss_evaluation` on the same held-out
rows (`artifacts/evals/<tag>.json`; training records `artifacts/metrics/<tag>.jsonl`). Held-out bits per byte, new
minus standard (paired t-test, df = 2; 95% CI from the t distribution):

| Metric | new s0 | std s0 | new s1 | std s1 | new s2 | std s2 | Mean difference | 95% CI | p |
|---|---|---|---|---|---|---|---|---|---|
| 8K code | 0.63539 | 0.63839 | 0.63003 | 0.64245 | 0.63796 | 0.63782 | −0.00509 | [−0.02133, +0.01115] | 0.31 |
| 8K text | 1.36145 | 1.35218 | 1.35838 | 1.35141 | 1.36027 | 1.35338 | +0.00771 | [+0.00436, +0.01107] | 0.010 |
| 8K math | 1.34961 | 1.34378 | 1.34382 | 1.34143 | 1.34818 | 1.34086 | +0.00518 | [−0.00110, +0.01146] | 0.071 |
| 8K all | 0.74078 | 0.74210 | 0.73566 | 0.74540 | 0.74280 | 0.74166 | −0.00331 | [−0.01748, +0.01086] | 0.42 |
| 32K repository | 0.62002 | 0.62182 | 0.61461 | 0.62665 | 0.62284 | 0.62128 | −0.00410 | [−0.02169, +0.01350] | 0.42 |
| HumanEval+ solutions | 0.66348 | 0.66029 | 0.64647 | 0.66510 | 0.66011 | 0.66790 | −0.00775 | [−0.03486, +0.01937] | 0.34 |
| MBPP+ solutions | 0.77777 | 0.79011 | 0.78449 | 0.79356 | 0.78620 | 0.80870 | −0.01464 | [−0.03204, +0.00277] | 0.069 |
| HumanEval-X C++ solutions | 0.62815 | 0.63128 | 0.62961 | 0.64343 | 0.62856 | 0.63893 | −0.00911 | [−0.02265, +0.00443] | 0.10 |

Code and the benchmarks' reference solutions are equal or lower under this configuration (MBPP+ and C++ solutions
lower on every seed), none significantly higher; general text is significantly higher (+0.0077, 0.6% relative), math
higher at p = 0.07. Training: no non-finite loss or gradient in either arm; maximum gradient norm 4.39 / 5.43 / 5.13
without clipping against 7.85 / 10.00 / 6.16 (measured before the standard regime's clip); training loss over the last
5% of steps (repository stage) 1.702 / 1.654 / 1.710 against 1.739 / 1.729 / 1.736.

**Production settings at d24** (`d24-r490.805-s0-d0-c7c5fbbe4b31`, `60138de`): the production run's batch (1,179,648
tokens) and peak lr (9.3795e-4), cut to 0.5B tokens (423 steps), file-level stage; weight decay 3.814, the τ law at
this 423-step horizon (production: 0.317). Per tenth of the run:

| Steps | Mean loss | Gradient norm, median | Gradient norm, max | lr at the end | tok/s |
|---|---|---|---|---|---|
| 0–42 | 8.269 | 1.864 | 4.717 | 9.331e-4 | 316,018 |
| 43–85 | 5.704 | 1.010 | 1.820 | 8.277e-4 | 315,689 |
| 86–128 | 4.936 | 1.096 | 1.265 | 7.224e-4 | 315,894 |
| 129–170 | 4.384 | 1.188 | 1.472 | 6.196e-4 | 315,041 |
| 171–212 | 3.898 | 1.204 | 1.403 | 5.167e-4 | 315,497 |
| 213–254 | 3.279 | 1.345 | 1.684 | 4.139e-4 | 315,811 |
| 255–296 | 2.785 | 1.203 | 1.381 | 3.110e-4 | 315,217 |
| 297–338 | 2.514 | 0.961 | 1.346 | 2.082e-4 | 315,058 |
| 339–380 | 2.368 | 0.650 | 0.882 | 1.053e-4 | 314,567 |
| 381–422 | 2.287 | 0.282 | 0.384 | 2.449e-6 | 314,405 |

No non-finite loss or gradient; the gradient norm exceeds 1.9 only during warmup. Held-out bits per byte at 8K
(`artifacts/evals/<tag>.json`): code 0.7485, text 1.4555, math 1.4454, all 0.8512: worse than the d8 run of 2,987
steps (code 0.6354), a model 13× smaller on fewer tokens. Per optimizer step d24 is lower at every step after the
first (2.274 vs 3.332 over steps 400–420); per token it is higher (3.508 vs 2.024 over 250–270M tokens, both in the
file-level stage): at the production batch 0.5B tokens is 423 optimizer steps. A proxy cut from the production plan
measures early optimization speed, so `schedule.plan` sets every run's batch and lr at its own compute (DeepSeek LLM
sets its scaling runs' by the same laws).

## Decoding and syntax analysis

The paper's Evaluate & Break section rests on a **pre-registered probe set**: 25 failure hypotheses fixed before the
production run from the literature and this pipeline's design, each a newly written input (no benchmark text, not in
the training data) with a programmatic check (`nanochat/eval/probes.py`, `python -m nanochat.eval.probes --model-tag
TAG` → `artifacts/probes/<tag>.json`, `.md`). Mitigations live at inference; pretraining is unchanged. Every probe runs
under each configuration that applies:

| Configuration | Decoding |
|---|---|
| `greedy` | left-to-right argmax, the literature's pass@1 setting |
| `healed` | greedy after token healing: the prompt's last token removed, the first generated token constrained to start with its bytes (Guidance; Lundberg 2023) |
| `fim`, `fim_spm` | the prompt as fill-in-the-middle, PSM / SPM, the format whose character-level random spans trained mid-token boundaries (`bpe/pack.rs:145-151`) |
| `constrained` | greedy under the Python 3.12 syntax constraint (Python probes) |
| `system` | healed, constrained greedy, then post-generation checks with seeded resampling until a candidate passes |

| Area | Probes (hypothesis) |
|---|---|
| Decoding / prompting | D1 prompt ends after `"""` + newline (boundary unlike training's `"""\n` chunk); D2 ends mid-identifier (`import num`); D3 ends in partial indentation; D4 ends at `def f(x):` without newline; D5 C++ prompt ends at `{` (MultiPL-E strips the newline); D6 greedy repetition loop; D7 generation runs past the function; D8 FIM PSM vs SPM disagree; D9 FIM middle never ends; D10 FIM with nothing to fill; D11 the literal text `<fim_prefix>` in code |
| Model / training | M1 cross-file use < 8K tokens back (training's `# path` headers, dependencies first); M2 ~24K tokens back (the 32K stage); M3 input > 32,768 tokens (must refuse); M4 dependency order reversed (helper only in the suffix); M5 2-space indentation; M6 tabs; M7 Python 2 input (training saw Python 3 only); M8 non-ASCII identifiers and strings; M9 C++ templates |
| Software system | S1 missing import (undefined name); S2 return value contradicts `-> str`; S3 wrong answer the prompt's examples catch; S4 C++ compile error; S5 unbalanced brackets in a long expression |

**Decoder** (`inference/decoder.py`, `csrc/decode.cu`): batched continuous decoding with one captured CUDA graph per
step. A `Request` carries the prompt, budget, stop rule, `temperature` (0 = argmax, ties to the lowest id), `seed` (the
sample at position *i* depends only on seed and *i*, not on slot or batch) and an optional `Constraint` (`mask()` over
the vocabulary before each token, `push(token)` after). `heal()` and `AllOf()` build constraints; each step takes the
masks of all syntax-constrained slots in one batched call (`step_masks`); a prompt whose length plus budget exceeds the
context raises `ValueError`.

**Constrained decoding: syntax, not types.** Type-constrained decoding (Mündler et al., PLDI 2025, arXiv 2504.09246)
cut TypeScript compilation errors by 75.3% (HumanEval) and 52.1% (MBPP). Its artifact
(`eth-sri/type-constrained-code-generation`) implements TypeScript only (`typesafe_llm/parser/parser_ts.py`), as do its
10 forks (9 identical, the tenth at upstream's last commit); ChopChop (Nagy et al., POPL 2026,
`timothytmzhou/chopchop`) ships TypeScript and e-graph constrainers; Monitor-Guided Decoding (arXiv 2306.10763,
`microsoft/monitors4codegen`) constrains member access and was evaluated on Java, C# and Rust. Biagiola et al. (arXiv
2606.21619) re-ran the TypeScript setup over 6 models and 10 seeds: unconstrained decoding was significantly more
functionally correct (Gemma-2 2B HumanEval 35.7% vs 32.0%; Qwen-2.5 32B 82.3% vs 72.3%), because the incomplete
constrainer pushes the model into low-probability text; a complete (syntax) constrainer instead gained 53% exact match
at 2B and 2.5% at 32B. A type constrainer for dynamically typed Python is incomplete by construction; Python 3 syntax
can be constrained completely, and the training data holds only files that parse as Python 3 (`prepare.py:201-208`),
so the model's distribution lies inside the constrainer's language (the alignment Biagiola et al. restore by
fine-tuning). Types are checked after generation instead.

**`PythonSyntax`** (`inference/syntax_constraint.py`, the Rust crate `inference/syntax/`):
`PythonSyntax(tokenizer).constraint(prefix, *, suffix="")`; `masks(constraints)` computes a batch of constraints' next
masks in parallel (rayon) without the GIL. Token *t* is allowed iff `prefix + generated + t` can still become a valid
Python 3.12 module, the text ending anywhere (mid-token, mid-indentation, mid-UTF-8 character); BOS iff
`ast.parse(prefix + generated + suffix)` succeeds; FIM sentinels never; a prefix that cannot become valid Python 3.12
raises `ValueError`. Layers: CPython 3.12.3's tokenizer (`Parser/tokenizer.c`) as a character-level state machine
(nested f-strings, continuations; `lexer.rs`); INDENT/DEDENT and the tab-consistency check as `tokenizer.c` computes
them; an LALR(1) grammar (`python.lark`) that accepts every CPython program, its tables built by Lark at load (strict
mode: no conflicts) and read by the crate (`parser.rs`), the soft keywords `match`, `case`, `type` read both ways until
one fails; per lexer state, the vocabulary's events over its byte trie, computed once and shared by threads
(`vocab.rs`). Where it is looser than CPython (targets parsed as expressions, argument order, escapes and nesting
limits unchecked), it accepts rather than rejects. Rejected alternatives: SynCode (one-regex f-strings reject valid 3.12
code; pins `transformers`, needs a Hugging Face tokenizer), llguidance (no INDENT/DEDENT, issue #107), Melcer et al.
(arXiv 2402.17988, no released library).

| PythonSyntax check | Result |
|---|---|
| Completeness: 1,051,013 validation tokens, 1,200 documents (`stack_edu_python`, `ultradata_stack_v3_l2_python`, `ultradata_code_l2_python`), the tokenizer's ids, a mask per token | 0 false rejections |
| CPython 3.12.3's grammar tests (20 files, 208,989 tokens); 167 standard-library files (1,149,695 tokens) | 0 false rejections |
| 213 edge cases against `ast.parse` | 0 mismatches |
| Soundness: 881 token substitutions that break `ast.parse` | all rejected: 492 at the substitution, 383 later (median 2 tokens), 6 at BOS |
| 3,000 random walks of 500 allowed tokens | no empty mask |
| Rust port against the Python reference it replaced | bitwise-identical masks at every step: the 1,200 documents (1,052,213 masks), the 20 grammar test files, the 213 edge cases (whole prefix, token by token, FIM split), 823 substitutions |
| `mask()` latency (Rust) | one mask: median 0.22 ms (Python reference: 1.36 ms); 64 constraints' masks: 1.6 ms in parallel (26 CPUs) vs 11.6 ms one by one, 3.5 ms with the BOS candidates' `ast.parse` |

C++ is not constrained (its training files were not parse-filtered); it is checked after generation by
`g++ -std=c++17 -fsyntax-only`, the MultiPL-E judge's compiler.

**Post-generation checks** (`eval/verify.py`): Python — `ast.parse` and a static type checker limited to definite
failures (undefined names, unknown attributes, call arity and argument types, return type against the annotation);
C++ — `g++ -fsyntax-only`; both — the prompt's own examples (docstring `>>>` lines, MBPP+ `assert` lines) in a
subprocess with a timeout, never the hidden tests.

**Not used:**
- *Speculative decoding* (e.g. DSpark, arXiv 2607.05147): lossless by construction (greedy outputs identical), so no
  failure mode to probe; its gains target large-model serving, while this decoder is CPU-bound in constrained mode and
  a drafter would need its own training run.
- *Multi-token prediction*: reported to hurt models of this size on code benchmarks (Gloeckle et al., arXiv
  2404.19737).
- *Self-correction prompting*: a base model without instruction tuning; reasoning and repair come with SFT and RL.

**First probe run** (pipeline check, `d8-r20-s0-d0-d69671850bea`, 14.7M scaling parameters, commit `60138de`): 114
probe × configuration cells, no crashes, 16 pass, 154 s of generation. The context guard refuses M3 in every
configuration; healing turns D5 (C++ `{`) from a run to the budget into a pass; `system` passes S5 (unbalanced
brackets) on its 9th candidate where every single-shot configuration runs to the budget; FIM with an empty suffix
rarely ends by itself (empty suffixes are rare in training: AST middles never end a piece). Pass rates at this scale
are near zero; the probe set's results are reported on the production model.

## Appendix ablation

The paired comparison above tests this configuration's regime as a whole against the standard regime, and the d24
check the production batch and lr. The pre-registered ablations (attention windows, gradient clipping, MLP activation,
matrix optimizer, weight decay, value embeddings, residual scalars, the repository stage; the peak lr) at d8 with 100
tokens per parameter and d12 with 20, each against the choice of DeepSeek-Coder and the code models it is compared with or
against deleting the component, their designs, decision rules and results are in `ablations/ablations.md`; only
changes confirmed at d16 reach production.

Adopted without a study, on published evidence and the pre-launch measurements: linear decay to zero continuing
through the repository stage, one RoPE base, learned QK-norm gains (Qwen3). Not studied, by expected value against
GPU cost:

- **Block Attention Residuals not adopted.** +12% d24 step time (about 10 hours of production; measured before
  `60138de`, record not kept) for a published 1.1% loss gain at 194M parameters (arXiv 2603.15031); the equal-time
  margin is below what a few seed pairs detect, and adopting it would need a native decode path.
- **Partial key offset not adopted.** A speedrun result (modded-nanogpt record #49, 124M, short training) with a small
  expected effect at 490 tokens per parameter and no native decode path.
- **Repository-stage rewarm not adopted.** Qwen2.5-Coder and DeepSeek-Coder continue the decay into the repository
  stage; no run here isolates a rewarm.

## Measured design decisions

Each in `ablations/ablations.md` §2:

- **Micro-batch 98,304** rather than 65,536: 1% faster per token, peak 69.6 GiB of 80 GB.
- **FlashAttention-4** rather than cuDNN attention: faster forward and backward in every measured window/segment case.
- **CUDA q/k prologue** rather than Inductor's fusion: 298.2K → 305.8K tok/s.
- **Value-embedding gate as an fp32 broadcast-sum** rather than fp32 GEMMs: 0.88 vs 1.20 ms per layer, same error.
- **Inductor coordinate-descent tuning:** Inductor's kernels 533 → 500 ms per d24 step.
- **Split q/k/v GEMMs** rather than one fused GEMM: 3,712 vs 3,729 ms per d24 step.
- **Runs one at a time:** one d8 run saturates the GPU (2.60M tok/s); two concurrent runs total 2.34M, four 2.57M.

## Layout

- `nanochat/config.py`: vocabulary, mixture, micro-batch, stages, RoPE base, run tag.
- `nanochat/model/gpt.py`: the model, scaling parameters and FLOPs per token.
- `nanochat/ops/`, `nanochat/csrc/`: the ops above and the decoder.
- `nanochat/train/`: `optim.py`, `schedule.py` (plan, lr multiplier), `stages.py` (token budget, rows per stage),
  `checkpoint.py`, `metrics.py`.
- `nanochat/eval/`: bits and bytes per context bucket and mixture bits per byte (`loss_eval.py`), code benchmarks
  (`code_eval.py`, `fim_eval.py`), post-generation checks (`verify.py`), the probe set (`probes.py`).
- `nanochat/inference/`: the native decoder (`decoder.py`), the Python 3.12 syntax constraint
  (`syntax_constraint.py`, its Rust crate and grammar in `syntax/`).
- `nanochat/data/`: the data pipeline, including the Rust tokenizer, repository and packing crate in `bpe/`.
- `scripts/`: `base_train.py`, `base_eval.py`, `report.py`.
- `runs/`: `setup.sh`, `container.sh`, `h100.sh`, `curves.sh`, `release.sh`.
- `results/`: the tables the paper reports.
- `ablations/`: the ablation designs, runner and analysis (`study.py`, `report.py`, `swiglu.py`), the step profiler
  (`profile_step.py`) and `ablations.md`.
- `docker/Dockerfile`: the image.
