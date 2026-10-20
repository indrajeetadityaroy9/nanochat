# Pretraining a code model on one H100

A Python and C++ code-generation base model pretrained on one NVIDIA H100 80 GB SXM5 (sm_90) at the most tokens its data
holds without any stage reading a document twice. The code is the specification; each section names the file that
defines a setting. Measurements and ablations, with the code that runs them, are in `ablations/` (`ablations.md`). The
pre-launch runs' numbers are stated here and in `ablations.md` (their raw records were not kept); the experiments'
results are in `results/`.

## Model

`nanochat/model/gpt.py`: karpathy/nanochat 92d63d4 plus the wins of its experiment log, with Qwen3's QK norm and
learnable gains. Each layer is a `Block` of `Attention` and `MLP`, each applied to the RMS-normed stream and added back.

| | d24 (production) |
|---|---|
| Shape | 24 layers × 768 (`gpt_config`: width 30 × depth rounded up to whole 128-dim heads), 6 heads × 128, MHA |
| Attention | per-head RMS norm → learned per-dim gain → RoPE over the full head, base 10^6; SSSL windows (S = 512), last layer full; FlashAttention-4 varlen with document masking |
| MLP | ReLU², hidden 4C |
| Residual | `x = resid_lambda·x + x0_lambda·x0` per layer; value embeddings on alternating layers (12 of 24) behind a `2·sigmoid` gate |
| Head | untied `lm_head`, zero init, plain cross entropy; no logit softcap or z-loss (QK norm bounds the attention logits) |
| Vocabulary | 2^15 ids (SA-BPE, BOS + 3 FIM sentinels) |
| Parameters | 195,096,576 scaling parameters (blocks 169,930,752 + `lm_head` 25,165,824), what the token budget counts; plus the token table (25,165,824) and 12 value-embedding tables (301,989,888, 1.55× the scaling parameters), lookups without FLOPs: 522,252,336 in all |
| FLOPs per token | 1.557 GFLOP non-embedding: DeepSeek LLM's M (Eq. 2, 72·n_layer·d² + 12·n_layer·d·l_seq, l_seq each layer's window at 8K: 1.020 matrices + 0.538 attention), plus 0.151 for `lm_head` |
| Init | stream-reading matrices and tables uniform with std C^-½; `c_o`, `c_proj`, `lm_head`, value-embedding gates zero; resid lambdas and gains 1, x0 lambdas 0 |
| Precision | fp32 parameters, gradients and optimizer state; bf16 activations and GEMM inputs |

## Data and stages

The data pipeline (`nanochat/data/`) prepares the corpora, trains the tokenizer, compiles the documents and packs them
into rows; its output is the Hugging Face dataset `iaroy/nanochat-data`.

- **Mixture** (`config.MIXTURE`): code:text:math = 87:10:3 (DeepSeek-Coder's code share); code is 8 Python and C++
  corpora, text ClimbMix, math two corpora.
- **Objective:** next token, with the packs' fill-in-the-middle pieces read as FIM examples (DeepSeek-Coder §3.1): half
  the pieces of the repository corpora's train splits, in PSM or SPM order; a middle is a random character span (10%)
  or AST-FIM's syntactic unit (Gong et al., arXiv 2506.00204), half of those widened to whole lines, the boundaries of
  line-level infilling (`data/pack.py`).
- **Stages** (`config.FILE_STAGE`, `config.REPO_STAGE`; Qwen2.5-Coder §3.2): 8,192-token rows of every corpus, then the
  last 300/5,500 of the steps on 32,768-token rows of the repository corpora. One RoPE base, 10^6 (`config.ROPE_BASE`:
  Qwen2.5-Coder's repository-stage base, above Men et al.'s lower bound for 32K, arXiv 2405.14591; Llama 3 trains at a
  large base from the start).
- **Token budget** (`train/stages.py max_ratio`): the most tokens per scaling parameter at which no stage reads a
  document twice: 490.77 at d24, 95.75B tokens (dataset revision `1e711aa`).

## Regime

| Setting | Value | Source |
|---|---|---|
| Batch | 1,179,648 tokens (12 micro-batches of 98,304) | DeepSeek LLM laws, B = 0.2920·C^0.3271, C = M·D (`train/schedule.py`) |
| Peak lr | 2.3887e-3, every group: 2.5467 × the law's 9.3796e-4 | DeepSeek LLM, η = 0.3118·C^-0.125, times the multiplier `ablations/lr.py` measured (`scripts/production.py` `LR_SCALE`; Appendix ablation) |
| Steps | 81,166 (95.75B tokens, 490.77 per scaling parameter); repository stage from step 76,739 | `schedule.plan` |
| Warmup | 40 steps, linear (47M tokens, 0.05% of the run) | 2/(1−β2) (Ma & Yarats, arXiv 1910.04209) |
| Decay | linear to zero | Bergsma et al., arXiv 2502.15938 (the gain over WSD grows with tokens per parameter) |
| Optimizers | AdamW (fused) on tables and vectors; Muon with `match_rms_adamw` on matrices | Moonlight, arXiv 2502.16982 §2.2 (`train/optim.py`) |
| AdamW | β = (0.9, 0.95) | DeepSeek LLM §2.3 |
| Weight decay | λ = 1/(η τ T), the EMA timescale τ = 1.084·TPP^−0.527 of the T steps (τ = 0.0414, λ = 0.1246 at d24's peak lr: 1/(ηλ) = 3,360 steps, 4.0B tokens), on tables, `lm_head` and Muon matrices, 0 on vectors | Power Lines, arXiv 2505.13738 (DeepSeek LLM: 0.1, §2.3) |
| Gradient clipping | global norm 1.0 on the raw gradient, before both optimizers; the norm is logged every step | DeepSeek LLM §2.3 (`base_train.MAX_NORM`) |
| Repository stage | continues the decay: 5.46% of the peak lr at its first step, 0 at the last, over 4,427 steps (5.22B tokens) | DeepSeek-Coder §3.6 |

Clipping at 1.0, decided on evidence rather than an experiment (`ablations/ablations.md` §1, Not tested). It rescales
a step's raw gradient before Muon's momentum and AdamW's moments; both updates are invariant to a gradient scale that
varies slowly (Adam's m/√v, Newton–Schulz orthogonalization), so clipping changes training only where the norm jumps,
by down-weighting that step's gradient. That makes a fixed threshold's scale-dependence harmless (at 1.0 the clipped
fraction of steps ranged from 0.2% to 97% across the configurations of an earlier d8 screen), including the norm's
late rise under a decaying lr and weight decay (Defazio, arXiv 2506.02285; d8 here: medians 0.46, 1.09 and 1.52 over
the run's thirds, §3). The one direct measurement, a single-seed d8 A/B in the previous regime (`ablations/ablations.md`
§3), had clipping lower on all 11 corpora by 0.003 bits per byte on average, under the paired comparison's spread of
per-pair differences (sd 0.0065 on 8K code); at the production batch and lr the norm stays below 1.9 after warmup (d24
check below), so outside spikes a clip scales the gradient by at least 0.53. nanochat deleted its clipping because its
norms never reached 1.0, leaving it inactive (karpathy/nanochat LOG 2026-01-08); its cost there was the norm's
all-reduce, here one device's norm the log computes anyway. Clipping the orthogonalized update instead, as Megatron-LM
and DeepSpeed have done for Muon, rescales the step itself (Megatron-LM PR 5395, DeepSpeed issue 8439); it is not
done here.

## H100 implementation

Ops in `nanochat/ops/`, CUDA sources in `nanochat/csrc/` (built on first use by `native.cuda_library`). Each is a
`torch.library` op, so `torch.compile(fullgraph=True)` traces one graph per stage. Inductor's coordinate-descent tuning
is off (`runs/container.sh`).

| Op | Implementation |
|---|---|
| Projection, MLP, LM-head GEMMs | forward and dX: QuACK SM90 GEMMs autotuned by QuACK (`quack.gemm_interface`, `tuned=True`); fp32 weight gradients: cuBLAS `torch.mm`, faster and more accurate than QuACK's tiles on their small outputs (`ablations/ablations.md`) |
| ReLU² MLP | QuACK `gemm_act` / `gemm_dact` |
| Cross entropy | QuACK, bf16 logits read once, the logit gradient written over them |
| Attention | FlashAttention-4 varlen, its default backward |
| q/k norm + gain + RoPE | `csrc/qk_prologue.cu`: one pass per direction; gain gradient without atomics (row math shared with the decoder, `csrc/qk_row.cuh`) |
| Embeddings | `F.embedding`, bf16 cast fused by Inductor; backward adds rows with atomics |
| Decoding | `csrc/decode.cu` (`inference/decoder.py`): batched native decoder for the code benchmarks |

The d24 step on real rows (12 micro-batches of 12 rows, `torch.profiler`): 3,741 ms, 315K tok/s, 100% GPU-busy, peak
69.6 GiB; GEMM 53%, FA4 25%, Inductor 14%; the repository stage's (12 of 3 rows) 4,016 ms with coordinate-descent
tuning (0.8% faster at the file level; not used, `ablations/ablations.md` §2). A fused q/k/v GEMM measured 3,729 ms
against 3,712 ms and was not adopted. At 3,741 ms the step computes 539 TFLOP/s by M plus `lm_head` (54.5% of the
H100's 989 dense bf16), 369 TFLOP/s (37.3%) counting the weight matrices alone.

## Reproducibility

- **Fastest kernels.** FA4's and the embedding backward reduce with atomics, so repeated steps differ in the last bits;
  FA4's deterministic backward costs 1–3% of a d24 micro-batch's forward and backward. The forward is deterministic:
  with Inductor's coordinate-descent tuning off, a compiled loss evaluation is bitwise equal across processes.
- **Resume.** A relaunched run continues from its newest checkpoint (weights, optimizer state, data position), written
  to a temporary file and renamed, so a kill mid-save leaves the previous one.
- **Environment.** `docker/Dockerfile` pins base images by digest and git dependencies by commit; `uv.lock` the Python
  packages; fp32 matmuls stay fp32 (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`).
- **Run identity.** A run's tag (`config.run_tag`) is its depth, ratio and seeds plus a hash of those, its variant
  (an ablation run's fields; production's lr scale) and the dataset revision `config.DATASET_REVISION`
  (`python -m nanochat.paths` downloads it), so runs of different settings or data never share artifacts. Each run's
  metrics record its full configuration (model, plan, seeds, lr, weight decay, clip threshold) and the dataset revision.
- **Data order.** Each corpus's rows are permuted once by the data seed (NumPy `RandomState`).

## Reproducing the run

On a fresh Lambda `gpu_1x_h100_sxm5` (Ubuntu with Docker and the NVIDIA driver):

```bash
git clone https://github.com/indrajeetadityaroy9/nanochat nanochat && cd nanochat
HF_TOKEN=<read token> runs/setup.sh   # CDI spec, docker group, image, artifacts/, tokenizer + data (nanochat/paths.py)
# log in again (docker group), then from the checkout:
tmux new -d -s h100 runs/h100.sh
```

Everything a run reads or writes lives in the checkout (`nanochat/paths.py`): `artifacts/` (gitignored, created by
`runs/setup.sh`) holds `data/`, `tokenizer/`, `checkpoints/` (`<tag>.pt`, the resume checkpoint, and
`<tag>-step<NNNNNN>.pt`, the model-only snapshots), `metrics/<tag>.jsonl`, `base_eval/<tag>[-step<NNNNNN>].json`,
`logs/` (`production-<tag>.log`, `<tag>.log` per ablation run) and `cache/`; `results/` (tracked) holds the data the
paper's tables and figures are drawn from.

`runs/h100.sh` runs `scripts/production.py`: it derives the ratio and the tag (`d24-r490.774-s0-d0-<hash>`), trains both
stages with a checkpoint every 1% of the steps and a model-only snapshot (weights, no optimizer state, ~2.1 GB) every
10% and at the repository stage's start, then evaluates the final model and every snapshot with `scripts/base_eval.py`
(when pass@1 emerges) and exports the run's tracked metrics (`train/metrics.py export`): `results/<tag>.jsonl` (its
records and every evaluation) and `results/<tag>-curves.jsonl` (its per-step records). Rerunning continues a stopped
run from its checkpoint and evaluates every model again. `base_eval` reports validation bits and bytes per corpus at 8K
and at 32K per context bucket (≤8K, >8K), reference-solution bits and bytes, and EvalPlus HumanEval+/MBPP+, MultiPL-E
C++ and HumanEval infilling (Benchmarks, below), each scored by its official tools.

**Expected cost:** 76,739 file-level steps at 3.741 s and 4,427 repository-stage steps at about 4.05 s (4,016 ms with
coordinate-descent tuning, plus its 0.8%), about 85 h, about $365 at $4.29/h.

`runs/release.sh <tag> <repo>` uploads the checkpoint and snapshots, the run records under `artifacts/` (metrics and
evaluations) and `results/` to a Hugging Face model repository; `runs/container.sh hf download <repo>
--local-dir /workspace/nanochat` (the checkout inside the container) restores them where `load_model`, `base_eval` and
`base_train` read them.

## Pre-launch verification

Commit `60138de` on the H100, on the packed rows; records under `artifacts/` by tag.

| Check | Result |
|---|---|
| Resume: d8, 20 tokens per parameter (`d8-r20-s0-d0-d69671850bea`), killed with `docker kill` at step 991 and relaunched | continued from its step-900 checkpoint, switched to the 32K stage at step 2,824 (`repo_start`), finished at step 2,987; the step-2,824 checkpoint kept |
| Production path (2026-10-20): `base_train` CLI with `--lr-scale` (d8, 74 steps), `base_eval` (loss, reference solutions, EvalPlus, MultiPL-E, infilling; greedy and filtered; d8 in 46 minutes), the harness's train-evaluate-export | all complete; the recorded lr is the scale times the law's; the loss evaluation matches the pre-consolidation code to 7 decimals in bits per byte |
| Run identity | the production tag stable across calls; tags of the same fields differ only by dataset revision and lr scale |

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

**Benchmarks** (`eval/code_eval.py`, `scripts/base_eval.py`). Each benchmark is decoded two ways:

| Mode | Decoding | Score |
|---|---|---|
| `greedy` | left-to-right argmax, one completion per problem | pass@1, the literature's greedy setting |
| `filtered` | left-to-right prompts healed (Guidance; Lundberg 2023), Python and FIM middles under the syntax constraint (FIM: given the prefix and suffix, no healing); the greedy candidate, then seeded samples until one has no post-generation diagnostic and passes the prompt's own examples, at most 10 (`verify.ATTEMPTS`); the accepted candidate, else the greedy one | filtered pass@1 (Codex's filtered pass@k, Chen et al. arXiv 2107.03374 §3.5; AlphaCode §4.5): a system measurement, not comparable to pass@1 tables |

- **Comparability.** MultiPL-E pass@1 here is greedy, not MultiPL-E's sampled estimate; HumanEval infilling is
  greedy, not Bavarian et al.'s (temperature 0.2, 100–200 samples). The prompts' examples overlap the scored base tests
  (MBPP+: 373 of 378 prompt asserts use a base input; HumanEval+: 55 of 164 tasks share a doctest input), so the
  filtered mode favours base over plus pass@1.
- **Infilling** (human-eval-infilling, Bavarian et al.): each middle is cut to the reference middle's line count, a
  single-line middle to its first line (the benchmark's standard post-processing for pass@1: arXiv 2410.03103; arXiv
  2505.18789 Fig. 4), and decoding stops there; random-span middles end at BOS. Both orders, PSM and SPM, are scored
  greedy; the filtered mode runs SPM.
- **Contamination.** The training corpora were not decontaminated against the benchmarks: a benchmark problem in the
  training data (HumanEval and MBPP are public on GitHub) inflates its score.

**Decoder** (`inference/decoder.py`, `csrc/decode.cu`): batched continuous decoding of up to 256 slots, each step one
captured CUDA graph. A slot runs its prompt through the same kernels as its generation (a prompt token per step), so
prompts and generations share steps; a prompt's K and V are kept in a paged prompt store and a later request with the
same prompt (each search attempt after the first) starts at its last token. The GEMV runs a one-warp block per 16-row
weight tile and 16-slot group, weights streamed through a register ring and inputs copied by `cp.async`; every sum's
order depends on the shapes and the slot's position alone, so a request's tokens do not depend on the batch, the store
or timing. A `Request` carries the prompt, budget, stop rule, `temperature` (0 = argmax, ties to the lowest id), `seed`
(the sample at position *i* depends only on seed and *i*) and an optional constraint (`heal()`'s first token, the
Python syntax constraint, `AllOf()` of both), whose masks each step takes for all constrained slots in one batched call
(`step_masks`); a prompt whose length plus budget exceeds the context raises `ValueError`.

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

**Post-generation checks** (`eval/verify.py`): Python — the prompt plus completion parses (a parser overflow counts as
a failure), the EvalPlus entry function exists and returns a value, and a static type checker limited to definite
failures (undefined names, unknown attributes, call arity, return type against the annotation); then the prompt's own
examples (docstring `>>>` lines, MBPP+ `assert` lines) run by value to the last one (an `exit()` fails) in a subprocess
with a timeout. C++ — `g++ -fsyntax-only` (any error or nonzero exit). The examples are the prompt's, not the scored
test suite, though they overlap its base tests (Benchmarks).

**Not used:**
- *Speculative decoding* (e.g. DSpark, arXiv 2607.05147): lossless by construction (greedy outputs identical), so no
  failure mode to probe; its gains target large-model serving, while this decoder is CPU-bound in constrained mode and
  a drafter would need its own training run.
- *Multi-token prediction*: reported to hurt models of this size on code benchmarks (Gloeckle et al., arXiv
  2404.19737).
- *Self-correction prompting*: a base model without instruction tuning; reasoning and repair come with SFT and RL.

## Appendix ablation

The paired comparison above tests this configuration's regime as a whole against the standard regime, and the d24
check the production batch and lr. Each decision below was taken without a study of its own, on published evidence and
the pre-launch measurements. The two whose result would change production have a pre-registered ablation,
`ablations/<file>.py` (method, power, decision criterion and results in `ablations/ablations.md`):

- **The peak lr of DeepSeek LLM's law** (`lr.py`: × 1, × 2, × 4 at d24 over 5 tokens per parameter, times the
  optimum's horizon trend measured at d8, × 1 to × 8 over 5 and over 200, its power-law exponent carried to
  production's horizon range and corrected for production's larger batch growth by the square-root batch rule). The
  transfer rules disagreed on that trend: over d8's range Complete^(d)P's SDE rule (arXiv 2512.22382) predicts × 0.354,
  Step-Law (arXiv 2503.04715) × 4.92. Measured (`results/lr.json`): optima × 2.0235 (d24 over 5), × 3.4669 and
  × 3.1565 (d8 over 5 and 200), so a d8 ratio of 0.9105: the optimum relative to the law barely moves with the horizon
  (the law's own C^−0.125 carries the trend), against both rules. Production's multiplier × 2.5467 (2.3887e-3) is
  adopted; the best arm beat the law by 0.0044 to 0.125 bits per byte, no run diverged, and the optima agree within 7%
  with nanochat's hand-tuned Muon lr at the same widths (`ablations/ablations.md` §3).
- **Value embeddings kept** (`value_embeds.py`: none, at d16). They cost 4.7% of the d24 step (measured before
  `60138de`, record not kept); MoVE (arXiv 2601.22887) measures −0.016 bits per byte at d12 and −0.010 at d20.

The others rest on their evidence alone (`ablations/ablations.md` §1, Not tested): gradient clipping at 1.0 (Regime);
one RoPE base; ReLU² (nanochat LOG 2026-02-05); linear decay to zero continuing through the repository stage, as
Qwen2.5-Coder and DeepSeek-Coder; learned QK-norm gains (Qwen3) and the x0 and residual scalars (nanochat LOG
2026-01-11); SSSL windows (Gemma 3's near-lossless local windows); Block Attention Residuals not adopted (+12% d24 step
time, measured before `60138de`, record not kept, for a published 1.1% loss gain at 194M parameters, arXiv
2603.15031); the partial key offset not adopted (a speedrun result, modded-nanogpt record #49, 124M); either of the
last two would need a native decode path.

## Measured design decisions

Each in `ablations/ablations.md` §2:

- **Micro-batch 98,304** rather than 65,536: 1% faster per token, peak 69.6 GiB of 80 GB.
- **FlashAttention-4** rather than cuDNN attention: faster forward and backward in every measured window/segment case.
- **CUDA q/k prologue** rather than Inductor's fusion: 298.2K → 305.8K tok/s.
- **Value-embedding gate as an fp32 broadcast-sum** rather than fp32 GEMMs: 0.88 vs 1.20 ms per layer, same error.
- **No Inductor coordinate-descent tuning:** it chooses kernels by timing, so two processes' compiled losses differ;
  off, losses are bitwise equal across processes, for 0.8% of the d24 step.
- **Split q/k/v GEMMs** rather than one fused GEMM: 3,712 vs 3,729 ms per d24 step.
- **Runs one at a time:** one d8 run saturates the GPU (2.60M tok/s); two concurrent runs total 2.34M, four 2.57M.

## Layout

- `nanochat/config.py`: vocabulary, mixture, micro-batch, stages, RoPE base, run tag.
- `nanochat/model/gpt.py`: the model, scaling parameters and FLOPs per token.
- `nanochat/ops/`, `nanochat/csrc/`: the ops above and the decoder.
- `nanochat/train/`: `optim.py`, `schedule.py` (plan, lr multiplier), `stages.py` (token budget, rows per stage),
  `checkpoint.py`, `metrics.py`.
- `nanochat/train/metrics.py` also holds the evaluation's metrics: bits and bytes per context bucket, mixture bits per
  byte, reference-solution bits, the evaluation record and the tracked-metrics export.
- `nanochat/eval/`: the code benchmarks (`code_eval.py`: EvalPlus, MultiPL-E, HumanEval infilling) and the
  post-generation checks (`verify.py`).
- `nanochat/inference/`: the native decoder (`decoder.py`), the Python 3.12 syntax constraint
  (`syntax_constraint.py`, its Rust crate and grammar in `syntax/`).
- `nanochat/data/`: the data pipeline, including the Rust tokenizer, repository and packing crate in `bpe/`.
- `scripts/`: `production.py` (the production run end to end), `base_train.py`, `base_eval.py`.
- `runs/`: `setup.sh`, `container.sh`, `h100.sh`, `release.sh`.
- `results/`: the data the paper's tables and figures are drawn from.
- `ablations/`: the ablation experiments (`lr.py`, `value_embeds.py`), their mechanism (`harness.py`, `model.py`) and
  `ablations.md`.
- `docker/Dockerfile`: the image.
