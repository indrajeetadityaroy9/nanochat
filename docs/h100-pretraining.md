# Pretraining a code model on one H100

A Python and C++ code-generation base model pretrained on one NVIDIA H100 80 GB SXM5 (sm_90) at the most tokens its data
holds without any stage reading a document twice. This is the repository's one document; the code is the
specification. Each section states its decisions, the basis of each (a measurement here, a cited result, or the law it
is derived from) and the file that defines it. Measurements are on an H100 80GB HBM3 SXM5 (700 W cap) in the image
`nanochat` (NGC 26.09), bf16, compiled as training compiles, unless stated; the pre-launch runs' raw records were not
kept, so their numbers are stated here. The ablation experiments' code is in `ablations/`, their results in
`results/`.

## Model

`nanochat/model/gpt.py`: karpathy/nanochat 92d63d4 plus the wins of its experiment log, with Qwen3's QK norm and
learnable gains. Each layer mixes x0 into the stream and looks up its value embeddings (`GPT.hidden`); its `Block` then
applies `Attention` and `MLP` to the RMS-normed stream, each added back.

| | d24 (production) |
|---|---|
| Shape | 24 layers × 768 (`gpt_config`: width 30 × depth rounded up to whole 128-dim heads), 6 heads × 128, MHA |
| Attention | per-head RMS norm → learned per-dim gain → RoPE over the full head, base 10^6; SSSL windows (S = 512), last layer full; FlashAttention-4 varlen with document masking |
| MLP | ReLU², hidden 4C |
| Residual | `x = resid_lambda·x + x0_lambda·x0` per layer; value embeddings on alternating layers (12 of 24) behind a per-head `2·sigmoid` gate of the normed stream, zero init (gate 1) |
| Head | untied `lm_head`, zero init, plain cross entropy; no logit softcap or z-loss (QK norm bounds the attention logits) |
| Vocabulary | 2^15 ids (SA-BPE, BOS + 3 FIM sentinels) |
| Parameters | 195,096,576 scaling parameters (blocks 169,930,752 + `lm_head` 25,165,824), what the token budget counts; plus the token table (25,165,824) and 12 value-embedding tables (301,989,888, 1.55× the scaling parameters), lookups without FLOPs: 522,252,336 in all |
| FLOPs per token | 1.557 GFLOP non-embedding: DeepSeek LLM's M (Eq. 2): 6 per block matrix weight (72·n_layer·d² and the value-embedding gates' 0.0003) + 12·n_layer·d·l_seq, l_seq each layer's window at 8K: 1.020 matrices + 0.538 attention; plus 0.151 for `lm_head` |
| Init | stream-reading matrices and tables uniform with std C^-½; `c_o`, `c_proj`, `lm_head`, value-embedding gates zero; resid lambdas and gains 1, x0 lambdas 0 |
| Precision | fp32 parameters, gradients and optimizer state; bf16 activations and GEMM inputs |
| Lineage | From nanochat 92d63d4: the block layout, value-embedding layers, MLP and window pattern. Not taken: its smear, backout and logit softcap, QK order (RoPE → norm → ×1.2), short window (a quarter of the context), RoPE base (10^5) and init (`wte` N(0, 0.8), `lm_head` N(0, 0.001), `c_fc` at 0.4×, ramped lambdas). The gate is nanochat e85db6b's (2·sigmoid, zero init) read over the full width rather than its first 32 channels; 92d63d4 has 3·sigmoid of 12 channels, U(0, 0.02) init. The full-width read is not ablated |

### Basis

- **Measured here: value embeddings,** the one component whose result would change production. They cost 4.7% of the
  d24 step (measured before `60138de`, record not kept; at d8 2.287M against 2.383M tokens per second without them,
  steps 10–29 of a run on real rows, 2026-10-21) and 302M parameters, for MoVE's −0.016 bits per byte at d12 and
  −0.010 at d20 (arXiv 2601.22887), a gain that shrinks with scale; no reference code model has them.
  `ablations/value_embeds.py` decides them (Ablations), not yet run.
- **Published, not tested here:** no reference code model ablates such details, and no affordable run could change
  them.
  - *ReLU² rather than SwiGLU:* nanochat measured SwiGLU worse at d12 and d24 under Muon (LOG 2026-02-05); at matched
    parameters it cost 5.1% of d8's throughput and would need a gated decode MLP (`csrc/decode.cu`).
  - *Learned QK-norm gains and the residual and x0 scalars:* 2·depth scalars and two head-width vectors per layer,
    measured by nanochat (LOG 2026-01-11) and Qwen3's QK norm.
  - *SSSL windows:* Gemma 3 (§5.2) measures local:global up to 7:1 and windows down to 512 nearly lossless in
    perplexity, and StarCoder2 trains code under a 4,096-token window; full attention costs 8.8% of d8's throughput
    and ≈ 4.4% of the d24 step.
  - *Shape:* one head narrower and deeper at d16's scaling parameters (31 × 384 against 16 × 512, MobileLLM §2.2.2)
    takes 26.7% more FLOPs per token (685M against 541M) and trains 27% slower (540K against 741K tokens per second,
    measured), so an equal-token comparison gives it more compute and the d24 analogue (35 × 640) would lengthen
    production by about as much [inference].
  - *head_dim 128, not 64:* the decoder's attention is built for head_dim 128 (`config.HEAD_DIM`, compiled into its
    kernels), so adoption rewrites it; nanochat measured 128 better (LOG 2026-01-17).
  - *MHA, not grouped-query attention:* its benefit is KV-cache memory; d24's cache for the decoder's 256 slots
    (`csrc/decode.cu` MAX_BATCH) of the longest benchmark request (1,158 tokens) is 21.8 GB, which an H100 holds.
- **Not adopted:** Block Attention Residuals (+12% d24 step time, measured before `60138de`, record not kept, for a
  published 1.1% loss gain at 194M parameters, arXiv 2603.15031) and the partial key offset (a speedrun result,
  modded-nanogpt record #49, 124M), either of which would need a native decode path; multi-token prediction, reported
  to hurt models of this size on code benchmarks (Gloeckle et al., arXiv 2404.19737).

### Against the reference code models

Where production departs from the code models it is measured against:

| Choice | Production | DeepSeek-Coder / Qwen2.5-Coder / StarCoder2 / Code Llama |
|---|---|---|
| Attention | SSSL: 3 of 4 layers a 512-token window | full attention (DeepSeek-Coder, Qwen2.5-Coder, Code Llama); StarCoder2 a 4,096-token window in a 16,384 context |
| MLP | ReLU², hidden 4C | SwiGLU (DeepSeek-Coder, Qwen2.5-Coder, Code Llama); StarCoder2 GELU |
| Value embeddings | alternating layers, gated (nanochat) | none |
| Optimizer | Muon on matrices, AdamW on tables and vectors | AdamW, β = (0.9, 0.95) (DeepSeek-Coder §3.4, Code Llama) |
| Peak lr | DeepSeek LLM's law × 2.5467 (`lr.py`), as the peak of linear decay to zero | the law's lr held for 80% of tokens, then 31.6% and 10% of it (DeepSeek LLM §2.3, the schedule DeepSeek-Coder §3.4 follows) |
| Weight decay | τ law (d8: 0.55, d24: 0.1246 at its peak lr) | 0.1 (DeepSeek LLM §2.3) |
| Gradient clipping | 1.0 | 1.0 (DeepSeek LLM §2.3) |
| Context stages | 8K file level, then the last 5.45% of steps at 32K on repositories | Qwen2.5-Coder: the same 8K → 32K repository stage; DeepSeek-Coder: 16K repository-ordered files; Code Llama: a 16K long-context stage |
| Data | code:text:math 87:10:3 | DeepSeek-Coder 87% code, 10% code-related English, 3% Chinese; Qwen2.5-Coder code:text:math 70:20:10 |
| FIM | 50% of the repository corpora's train pieces, PSM or SPM (`pack.FIM_RATE`, `SPM_SHARE`) | 50% (DeepSeek-Coder, StarCoder2); Code Llama 90% |

What those models ablate: Code Llama the RoPE base of its long-context stage (Appendix G.4, Table 18: without the
raised base, retrieval fails at 8K despite fine-tuning at 16K) and FIM's cost to left-to-right completion;
DeepSeek-Coder the FIM rate (Fig. 3); StarCoder2 RoPE against learned positions; OpenCoder its data (deduplication
level, annealing data, GitHub stars). None ablates architecture details such as gains, scalars or attention windows.

## Data and stages

The data pipeline (`nanochat/data/`) prepares the corpora, trains the tokenizer, compiles the documents and packs them
into rows; its output is the Hugging Face dataset `iaroy/nanochat-data`.

- **Mixture** (`config.MIXTURE`): code:text:math = 87:10:3 (DeepSeek-Coder's code share); code is 8 Python and C++
  corpora, text ClimbMix, math two corpora. Not tested: the code metric falls with the code share by construction, and
  the coding gain Qwen2.5-Coder reports for 70:20:10 (Table 3, 7B on 5.5T tokens; non-monotone: 100:0:0 best, 85:10:5
  worst) is a capability no small run shows.
- **Objective:** next token, with the packs' fill-in-the-middle pieces read as FIM examples (DeepSeek-Coder §3.1): half
  the pieces of the repository corpora's train splits, in PSM or SPM order; a middle is a random character span (10%)
  or AST-FIM's syntactic unit (Gong et al., arXiv 2506.00204), half of those widened to whole lines, the boundaries of
  line-level infilling (`data/pack.py`). Not tested: DeepSeek-Coder's ablation (Fig. 3) finds a 50% rate balances
  infilling and left-to-right completion, Bavarian et al. (arXiv 2207.14255) FIM nearly free for left-to-right; a test
  repacks every corpus.
- **Stages** (`config.FILE_STAGE`, `config.REPO_STAGE`; Qwen2.5-Coder §3.2): 8,192-token rows of every corpus, then the
  last 300/5,500 of the steps on 32,768-token rows of the repository corpora; a model never trained at 32K is worse at
  32K by construction (Qwen2.5-Coder §3.2). One RoPE base, 10^6 (`config.ROPE_BASE`), from the first step: it meets Men
  et al.'s lower bound for a 32K context (arXiv 2405.14591), as Llama 3 trains at 5 × 10^5 from the start (arXiv
  2407.21783 §3.2), and is Qwen2.5-Coder's repository-stage base; Code Llama's ablation (Appendix G.4) shows the base
  must be raised for long context, which production's already is. Staging (10^4 at 8K, as Code Llama and
  Qwen2.5-Coder) would add a base change for the repository stage to adapt to, with no published gain at 8K.
- **Token budget** (`train/stages.py max_ratio`): the most tokens per scaling parameter at which no stage reads a
  document twice: 490.77 at d24, 95.75B tokens (dataset revision `1e711aa`).

**Pipeline** (`nanochat/data/__init__.py`; each step `runs/container.sh python -m nanochat.data.<step>`, reading the
previous step's output under `paths.DATA`):

1. `prepare`: fetches every source (`sources.py`) at its pinned Hub revision, normalizes it into a Parquet store and
   curates it: the code group whole, text and math up to Qwen2.5-Coder's 7:2:1 shares of its horizon; exact and near
   duplicates dropped across all corpora and splits (`dedup.py`: StarCoder2's MinHash LSH), and GitHub Linguist's
   vendored and generated files; 1% of each source's groups (a repository, a codeforces problem, else a document) held
   out as val; repository files grouped into repository samples, dependencies first (DeepSeek-Coder §2.2).
2. `tokenizer`: SA-BPE merges (Chizhov et al., arXiv 2604.14053) trained on about equal byte shares of every source's
   train split, encoded by tiktoken. The vocabulary, 2^15, is not tested: a test retrains the tokenizer and repacks
   every corpus.
3. `compile`: every curated document tokenized once, independent of the sequence length (a repository sample's files
   each after a comment with its path).
4. `pack --stage file`, `pack --stage repo`: documents that do not fit a row cut between whole statements
   (tree-sitter), the FIM examples made, and each stage's rows packed Best-Fit-Decreasing, no token discarded.

## Regime

| Setting | Value | Source |
|---|---|---|
| Batch | 1,179,648 tokens (12 micro-batches of 98,304) | DeepSeek LLM laws, B = 0.2920·C^0.3271, C = M·D (`train/schedule.py`) |
| Peak lr | 2.3887e-3, every group: 2.5467 × the law's 9.3796e-4 | DeepSeek LLM, η = 0.3118·C^-0.125, times the multiplier `ablations/lr.py` measured (`scripts/production.py` `LR_SCALE`; Peak lr, below) |
| Steps | 81,166 (95.75B tokens, 490.77 per scaling parameter); repository stage from step 76,739 | `schedule.plan` |
| Warmup | 40 steps, linear (47M tokens, 0.05% of the run) | 2/(1−β2) (Ma & Yarats, arXiv 1910.04209) |
| Decay | linear to zero | Bergsma et al., arXiv 2502.15938 (the gain over WSD grows with tokens per parameter) |
| Optimizers | AdamW (fused) on tables and vectors; Muon with `match_rms_adamw` on matrices | Moonlight, arXiv 2502.16982 §2.2 (`train/optim.py`) |
| AdamW | β = (0.9, 0.95) | DeepSeek LLM §2.3 |
| Weight decay | λ = 1/(η τ T), the EMA timescale τ = 1.084·TPP^−0.527 of the T steps (τ = 0.0414, λ = 0.1246 at d24's peak lr: 1/(ηλ) = 3,360 steps, 4.0B tokens), on tables, `lm_head` and Muon matrices, 0 on vectors | Power Lines, arXiv 2505.13738 (DeepSeek LLM: 0.1, §2.3) |
| Gradient clipping | global norm 1.0 on the raw gradient, before both optimizers; the norm is logged every step | DeepSeek LLM §2.3 (`train/optim.py` `MAX_NORM`, `update`) |
| Repository stage | continues the decay: 5.46% of the peak lr at its first step, 0 at the last, over 4,427 steps (5.22B tokens) | DeepSeek-Coder §3.6 |

**Gradient clipping** at 1.0, decided on evidence rather than an experiment. It rescales a step's raw gradient before
Muon's momentum and AdamW's moments; both updates are invariant to a gradient scale that varies slowly (Adam's m/√v,
Newton–Schulz orthogonalization), so clipping changes training only where the norm jumps, by down-weighting that
step's gradient. That makes a fixed threshold's scale-dependence harmless (at 1.0 the clipped fraction of steps ranged
from 0.2% to 97% across the configurations of an earlier d8 screen), including the norm's late rise under a decaying
lr and weight decay (Defazio, arXiv 2506.02285; the norms below). Its direct measurements, single-seed d8 pairs, have
it lower: by 0.003 bits per byte on average and on all 11 corpora in the previous regime (below), by 0.0034 on code in
this one (production's arm, Ablations), under the paired comparison's spread of per-pair differences (sd 0.0065 on 8K
code) and below what an affordable experiment resolves (about 0.008, Power). At the production batch and the law's lr
the norm stays below 1.9 after warmup (Production's batch at d24), so outside spikes a clip scales the gradient by at
least 0.53. nanochat deleted its clipping because its norms never reached 1.0, leaving it inactive
(karpathy/nanochat LOG 2026-01-08); its cost there was the norm's all-reduce, here one device's norm the log computes
anyway. Clipping the orthogonalized update instead, as Megatron-LM and DeepSpeed have done for Muon, rescales the step
itself (Megatron-LM PR 5395, DeepSpeed issue 8439); it is not done here.

- *The A/B in the previous regime:* d8 at 20 tokens per scaling parameter, seed 42, both arms bitwise deterministic on
  the same data order (so they differ by the clipping alone), 2,987 steps each; val bits per byte of the final
  checkpoint on each corpus's val split, train loss mean over the last 10% of steps:

  | Arm | Val bpb mean | stack_edu_python | stack_edu_cpp | climbmix | Train | Steps with norm > 1 | Max norm |
  |---|---|---|---|---|---|---|---|
  | clip 1.0 | 0.87806 | 0.74209 | 0.67994 | 1.34683 | 1.89735 | 165 | 10.40 |
  | no clip | 0.88119 | 0.74504 | 0.68317 | 1.35029 | 1.89841 | 179 | 9.83 |

- *Gradient norms:* d8 at 20 tokens per parameter (`d8-r20-s0-d0-2449fca0cf29`), median over steps 0–298 0.464 (9% of
  steps above 1), 298–1,493 1.093 (57%), 1,493–2,987 1.522 (100%); the d24 run below, steps 0–42 1.866 (86%), 42–211
  1.145 (83%), 211–423 0.969 (47%).

**Peak lr.** DeepSeek LLM's law was fit for AdamW under a multi-step schedule, 7.5× below production's compute, and
the transfer rules disagree on how the optimum moves with the horizon: from d8 over 5 to over 200 tokens per parameter,
Complete^(d)P's SDE rule (arXiv 2512.22382) predicts the optimum's multiplier of the law changes × 0.354, Step-Law
(arXiv 2503.04715) × 4.92. At d8 at 20 the law's lr is low: × 2 lowered code bits per byte by 0.018 on both of 2 seeds
(below). `ablations/lr.py` measures the multiplier (Ablations): × 1, × 2, × 4 at d24 over 5 tokens per parameter,
times the optimum's horizon trend measured at d8 (× 1 to × 8 over 5 and over 200), its power-law exponent carried to
production's horizon range and corrected for production's larger batch growth by the square-root batch rule. Measured
(`results/lr.json`): optima × 2.0235 (d24 over 5), × 3.4669 and × 3.1565 (d8 over 5 and 200), so a d8 ratio of
0.9105: relative to the law the optimum barely moves with the horizon (the law's own C^−0.125 carries the trend),
against both rules. Production's multiplier × 2.5467 = 2.0235 · 0.9105^(ln 98.17 / ln 40.05) · √2 (2.3887e-3, weight
decay 0.1246) is adopted; the best arm beat the law by 0.0044 to 0.125 bits per byte, no run diverged, and the optima
agree within 7% with nanochat's hand-tuned Muon lr at the same widths.

- *d8 at 20* (`harness.main`, production vs × 2, seeds 0 and 1, 2026-10-20; GPU shared, so throughput is not
  comparable): code bits per byte 0.6323 and 0.6324 at the law's lr, 0.6144 and 0.6137 at × 2 (paired differences
  −0.0179, −0.0187); solutions −0.014, 32K repository −0.018, text −0.030, math −0.033. The law's lr is below the
  optimum at this horizon; `lr.py`'s grids start at × 1 where they look upward.
- *`lr.py`* (2026-10-20, dataset revision `1e711aa`, seed 0; `results/lr.json`, `lr.jsonl`, `lr-curves.jsonl`): code
  bits per byte at × 1 / 2 / 4 / 8 of the law's lr: d8 at 5 0.89491 / 0.78969 / 0.76960 / 0.81791 (× 8 the amended
  rule's extension), vertex × 3.4669; d8 at 200 0.52774 / 0.51750 / 0.51592 / 0.52431, × 3.1565; d24 at 5 0.44230 /
  0.43787 / 0.44201, × 2.0235. Horizon exponent −0.0254. No run diverged (up to × 8 at d8, 6.66e-3 at d24); each
  setting's best arm beats the law by 0.1253 (d8 at 5), 0.0118 (d8 at 200) and 0.0044 (d24 at 5). Checks: every
  derived number recomputed from the raw run records by an independent script (closed-form parabola) agrees to 6e-15;
  each run's logged lr equals plan × schedule at every step and its weight decay the τ law (lr × wd constant within a
  setting); seeds and evaluation rows are identical within a setting; the last-5% train loss ranks each setting's arms
  as code does; `torch.optim.Muon` decays by the unscaled lr and steps by lr · 0.2√max(A, B) (Moonlight's form).
  Against references: nanochat's hand-tuned Muon lr 0.02 under its √max(1, rows/cols) scaling (karpathy/nanochat
  92d63d4 `optim.py:393`) is the AdamW-matched 0.1/√C, 3.61e-3 at width 768 and 6.25e-3 at 256, against the measured
  optima 3.37e-3 (d24 at 5) and 6.47e-3 (d8 at 200); d8's optimal lr falls to 0.574 over 40× the tokens with its batch
  doubled (exponent 0.150, 0.244 net of √2 for the batch), against DeepSeek LLM's 0.125 under the same batch
  co-scaling and Bjorck et al.'s 0.32 at fixed batch and weight decay, where the τ law moves the weight decay instead
  (Wang & Aitchison, arXiv 2405.13698); the SDE rule's exponent 1/2 is too large, as arXiv 2604.27077 also finds. At
  d24 at 5's curvature (0.0089 per (ln lr)²) a 30% lr error costs 0.0006. Wall time 13:09–18:39 UTC, 37 minutes of it
  the GPU idle between the two `lr.py` invocations (a waiting script matched its own tmux session).

**The rest of the regime, not tested:**

- *Weight decay:* the τ law (Power Lines) is the regime's documented choice, tested with the regime as a whole by the
  paired comparison below.
- *Warmup:* 2/(1 − β₂) = 40 steps: 0.11% of d8's run at 490.8, 0.05% of production. Warmup's benefit is a larger
  stable peak lr (Kalra & Barkeshli, arXiv 2406.09405) by bounding early update size, which normalized matrix updates
  such as Muon's already bound (Kosson et al., arXiv 2410.23922); `lr.py`'s × 2 arm probes that margin, its per-step
  loss and gradient norms showing any early instability.
- *The repository stage's schedule:* the linear decay continues through it, as DeepSeek-Coder (§3.6) and Qwen2.5-Coder
  continue theirs; re-warming (Gupta et al., arXiv 2308.04014) is measured for continued pretraining after a finished
  schedule.
- *NorMuon:* its reported gain (arXiv 2510.05491: 6% efficiency at 124M) is about 0.002 bits per byte, below the
  experiments' resolution.

### Against the standard regime

Paired comparison, d8 at 20 tokens per parameter, three seed pairs (seed = data seed ∈ {0, 1, 2}; same initialization
and data order within a pair). This configuration's arm is `60138de`'s baseline: linear decay to zero, τ-law weight
decay, one RoPE base and no clipping, which production has since added
(`d8-r20-s{0,1,2}-d{0,1,2}-{d69671850bea, ca426bc7cec9, ccd433c30eb2}`, tags reproduced from `60138de`'s `run_tag`).
The standard regime is commit `c00d2f6`'s default (WSD, weight decay 0.1, clip 1.0, RoPE base 10^4 → 10^6), trained in
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
5% of steps (repository stage) 1.702 / 1.654 / 1.710 against 1.739 / 1.729 / 1.736. pass@1 at d8 is about 0 (seed 0:
HumanEval+ 0, MBPP+ base 0.0026, single-line infilling SPM 0.036), so the comparison rests on bits per byte.

An earlier unpaired comparison agrees: seed 0 of the baseline (2,987 steps of 98,304 tokens, peak lr 2.73e-3;
`d8-r20-s0-d0-d69671850bea`) against the previous regime's runs (WSD, RoPE base 10^4 → 10^6 at the repository stage;
weight decay 0.1 as recorded by `d8-r20-s42`, whose clip was 1.0; the `-g1`/`-ginf` records predate the weight-decay
field). Seeds differ (0 vs 42), so data order differs. Mean training loss per window of steps, one record per step (a
resumed run's repeated steps counted once):

| Run | 0–40 | 40–250 | 250–500 | 500–1000 | 1000–1500 | 1500–2000 | 2000–2500 | 2500–2824 | 2824–2987 (32K) |
|---|---|---|---|---|---|---|---|---|---|
| `60138de` (seed 0) | 8.742 | 5.485 | 3.660 | 2.652 | 2.347 | 2.207 | 2.094 | 2.025 | 1.701 |
| Previous regime, clip 1.0 (`d8-r20-s42`) | 8.733 | 5.362 | 3.634 | 2.703 | 2.352 | 2.217 | 2.136 | 2.032 | 1.785 |
| Previous regime, clip 1.0 (`d8-r20-s42-g1`) | 8.738 | 5.347 | 3.553 | 2.623 | 2.332 | 2.209 | 2.130 | 2.026 | 1.808 |
| Previous regime, no clip (`d8-r20-s42-ginf`) | 8.740 | 5.455 | 3.726 | 2.671 | 2.351 | 2.220 | 2.140 | 2.035 | 1.803 |

No non-finite loss or gradient; median gradient norm over steps 1,000–2,824 1.49 (0.77–0.86 for the previous runs,
whose weight decay is 0.1 against the baseline's 0.548 at this scale); throughput 2.3M tokens/s. Held-out bits per
byte (`artifacts/base_eval/<tag>.json`; the `-g1`/`-ginf` files in the earlier evaluator's format):

| Validation split (8K rows) | `60138de` (seed 0) | Previous, clip 1.0 (`-g1`) | Previous, no clip (`-ginf`) |
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

### Production's batch at d24

`d24-r490.805-s0-d0-c7c5fbbe4b31` (`60138de`, unclipped): the production run's batch (1,179,648 tokens) at the law's
peak lr (9.3795e-4, before the × 2.5467 above), cut to 0.5B tokens (423 steps), file-level stage; weight decay 3.814,
the τ law at this 423-step horizon (production's at this lr: 0.317). Per tenth of the run:

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

## Ablations

Pre-registered experiments, one per decision this document records without a study of its own whose result would
change production; every other decision states its basis where it is made. They share one mechanism
(`ablations/harness.py`, `ablations/model.py`) whose runs train and evaluate through the production code.

| File | Settings | Production (the recorded decision) | Alternative | Production cost if adopted |
|---|---|---|---|---|
| `lr.py` | d8 at 5 and 200, d24 at 5 (1 seed each): the d24 optimum times d8's horizon trend carried to production | DeepSeek LLM's law (§3.1) × 2.5467, adopted 2026-10-20 (Peak lr) | × 2 and × 4 (× 8 at d8): the law was fit for AdamW under a multi-step schedule, 7.5× below production's compute, and the transfer rules disagree on the optimum's horizon trend (d8 ratio: SDE rule × 0.354, Step-Law × 4.92); × 2 beat × 1 by 0.018 at d8 at 20 | none |
| `value_embeds.py` | d16 at 20 (4 seeds), d8 at 20 (4) and 490.8 (1) | value embeddings (MoVE: −0.016 at d12), absent from every reference model; 302M parameters, 1.55× the scaling parameters | none | −4.7% of the d24 step, 12 vocabulary tables fewer |

**Results.** `lr.py` ran on 2026-10-20 (`results/lr.json`, `lr.jsonl`, `lr-curves.jsonl`; Peak lr) and its
multiplier is production's. `value_embeds.py` has not run.

### Method

- **Arms.** An experiment is production and one arm per alternative at each of its settings, each on the same seeds;
  an arm differs from production only in the fields of `harness.Run`. Every run goes through production's code:
  `scripts/base_train.train` (the file-level then the repository stage, the production data order by seed, τ-law
  weight decay) and `eval/suite.py evaluate`'s likelihood metrics (the same held-out rows for every run); its model is
  `model.Candidate`, which is production's GPT unless `value_embeds` differs. Its plan (batch, steps, peak lr) is the
  production model's at the setting (`harness.plan_of`), so arms match in tokens and hyperparameters, and every run
  completes its own schedule (warmup, then linear decay to zero over its length): a longer run cut short would be
  compared before its decay, where loss curves cross and rankings flip (Wen et al., arXiv 2509.02046; Hägele et al.,
  arXiv 2405.18392). An lr arm's weight decay follows its lr by the τ law, λ = 1/(η τ T), as production's would: the
  sweep varies η at the fixed EMA timescale τ that sets the decay's effect (Wang & Aitchison, arXiv 2405.13698; Power
  Lines, arXiv 2505.13738). A run that diverges (a non-finite loss or gradient norm; its failure record) is an outcome
  of a sweep's high end (Wortsman et al., arXiv 2309.14322): it is recorded with its step and excluded from the fits
  and comparisons, and the experiment continues; any other failure stops it.
- **Settings.** No affordable run reaches production (d24 at 490.8 tokens per scaling parameter). `lr.py` measures the
  optimum on production's model at a short horizon, d24 at 5 (3,307 steps of 294,912 tokens, the longest d24 run a
  sweep affords), and the optimum's horizon trend at d8 between 5 (746 steps of 98,304 tokens; B_opt is 0.46
  micro-batches there, so the plan's floor of one) and 200 (14,937 steps of 196,608): 200 is the shortest horizon at
  which d8's law batch has grown to 2 micro-batches (B_opt 1.52, against 2.04 at 490.8; both round to 2), so the batch
  correction is the same as at 490.8 at less than half the cost (Analysis); one seed per setting, the cheap settings
  first: at d8 at 20 the arms × 1 and × 2 differed by 0.0179 and 0.0187 bits per byte on seeds 0 and 1 (Peak lr).
  `value_embeds.py` measures its decision's dependence on training length at d8 and on width at a short length and
  combines them as separable (log-linear in each): d16 (67.1M scaling parameters, 1/2.9 of d24) at Chinchilla's 20, 4
  seeds, the decision's setting; d8 at 20, 4 seeds, and at production's ratio, one seed, for the gain's trends in width
  and length: it shrinks with scale (−0.016 at d12, −0.010 at d20, MoVE), and d8's tables are 2.3× its scaling
  parameters against d24's 1.55×. Architecture rankings can change with scale (Tay et al., arXiv 2207.10551); for data
  recipes a single proxy 7× below the target picks the target's better of two in about 80% of pairs, best with
  likelihood metrics such as bits per byte (DataDecide, arXiv 2504.11393). No run combines production's width and
  length: the experiments assume the two trends separable (Bjorck et al.'s size-independent β for the lr; a log-linear
  trend for value embeddings), which only a d24 run at 490.8 could check.
- **Order.** `lr.py` first; its multiplier is production's peak lr (`scripts/production.py` `LR_SCALE`, hashed into
  production's tag). `value_embeds.py`'s arms train at the law's lr, which `lr.py` puts 2.0 to 3.5 times below each
  measured optimum: its baseline is under-tuned unless its settings take their own optima (under-tuned baselines
  inflate gains: Kaddour et al., arXiv 2307.06440; Wen et al.). A run's tag hashes its peak lr, so runs at a superseded
  lr are never reused.
- **Measures** (`harness.evaluation`, bits per byte unless stated). *code*: 8K validation of the code group at
  production's corpus weights, the decision measure. *solutions*: the HumanEval+, MBPP+ and HumanEval-X C++ reference
  solutions given their prompts, pooled (the benchmarks' pass@1 is near zero at this size). *repo_32k*: 32K
  repository rows of the code group; *repo_beyond_8k*: the same rows' positions past 8,192, the context only the
  repository stage trains. *text*, *math*; tokens per second (median over training steps). Efficiency (from the run's
  run and end records): parameters, non-embedding FLOPs per token, peak memory.
- **Analysis** (from `results/<experiment>.jsonl` and `.json`), pre-registered. *lr* (computed by `lr.py` into
  `results/lr.json`'s `analysis`, every number from the runs' records and plans): per setting, while its lowest-code
  run is at an end of the grid (a diverged run counting as worse than any evaluated one) the grid gains the next factor
  of 2 in that direction and the experiment runs again (`lr.py` repeats this until every grid brackets its optimum);
  then the vertex u* of the parabola in u = ln(multiplier) through the lowest-code run and its two neighbours (positive
  curvature by construction), or that run's own u when the neighbour above it diverged. Amended 2026-10-20, after d8 at
  5's grid (× 1, 2, 4: code 0.89491, 0.78969, 0.76960) put its lowest run at the grid's top with one quadratic over
  all runs placing the vertex inside (× 3.33): that quadratic's vertex is pulled by the far arms of an asymmetric curve
  (planted below: an optimum × 3.0 four times steeper below than above, read × 3.95 by one quadratic over × 1–8, × 3.72
  by the bracketing parabola). The measured d8 ratio r = exp(u*_d8@200 − u*_d8@5) is the optimum's change over a token
  ratio m_D,d8 = 40.0, assumed width-independent (Bjorck et al.'s size-independent horizon exponent, arXiv
  2409.19913), and the test of the transfer rules, which disagree on its direction: Complete^(d)P's AdamW SDE rule,
  η ∝ √(m_B / m_D) (Mlodozeniec et al., arXiv 2512.22382, Table 1), over the law's own fall predicts
  √(2 / 40.0) / 0.631 = 0.354 (d8's batch goes from 1 to 2 micro-batches, the law's lr falls to 0.631 of itself);
  Step-Law (Li et al., arXiv 2503.04715, η = 1.79 N^−0.713 D^0.307) predicts 40.0^0.307 / 0.631 = 4.92; both are
  reported against the measurement, with the horizon exponent ln r / ln m_D,d8. Production's multiplier of the law's
  lr is exp(u*_d24@5) · r^(ln m_D,prod / ln m_D,d8) · √(m_B,prod / m_B,d8): the exponent carried 1.24× further in
  log-horizon, to production's m_D,prod = 98.2 (every rule in play is a power law in D: the SDE rule, Bjorck et al.,
  Step-Law, the law), and production's batch growth over its range, 4× (294,912 to 1,179,648), against d8's 2×,
  carried by the square-root batch rule (Malladi et al., arXiv 2205.10287; Complete^(d)P's √m_B), so √(4 / 2). m_B and
  m_D come from the runs' recorded plans and production's (`harness.plan_of` at `max_ratio` of d24). Weight decay
  follows each run's lr by the τ law (Power Lines' fit for single-epoch pretraining), not Complete^(d)P's
  λ ∝ √(m_B / m_D). *Value embeddings*: per setting, the alternative's mean code minus production's, the two-sided
  pooled t-test's p and the difference's 95% confidence interval (scipy `ttest_ind`; pooled with equal arm sizes is
  robust to unequal variances); their gain g is code without them minus code with them. Production's gain is predicted
  as g(d16, 20) · g(d8, 490.8) / g(d8, 20) · (g(d16, 20) / g(d8, 20))^0.70 (0.70 = ln(195.1 / 67.1) / ln(67.1 / 14.7),
  log-linear in scaling parameters), or g(d16, 20) alone if g(d8, 20)'s interval includes zero. They are removed iff
  that prediction is below the compute their removal frees, 0.05 · L · ln r (L production's code, r the alternative's
  tokens per second over production's at d16; Kaplan et al., arXiv 2001.08361 §1.2: loss ∝ C^−0.050); otherwise they
  stay, their gain and its interval reported. Unpaired although arms share seeds: the six-run comparison's per-pair
  differences (sd 0.0065) show no gain from pairing over independent runs (√2 σ = 0.0048).
- **Equal tokens.** The production run's length is set by its data (no document read twice), not by time.

**Power.** The paired comparison's six runs (Against the standard regime) give a within-configuration σ of 0.0034
bits per byte on code (0.0044 on solutions, 0.0036 on 32K repository); both experiments' 4-seed settings measure it
where they run. Value embeddings at d16, 4 runs against 4 (6 df): standard error 0.71σ = 0.0024, 80% power at an
effect of 0.008, below the published 0.010 to 0.016. lr: each vertex rests on the arms' differences within one seed,
which varied by 0.0008 across 2 seeds at d8 at 20 against a 0.018 effect; at the curvature Bjorck et al.'s Llama-1
estimate implies (0.027 nats for a 2.5× lr; about 0.026 bits per byte per (ln lr)² at 3.5 bytes per token [assumed])
and per-run noise σ, a three-point fit places a vertex within about ±14% (one standard error, 0.13 in u). The d8 ratio
r combines two vertices' errors, about ±20% (0.185 in u); raised to 1.24 for production's horizon range it
contributes 0.23 in u, and with the d24 vertex production's multiplier is within about ±30% (0.26 in u). The two
rules' predictions for r differ by 14× (ln 14 = 2.6 in u), far beyond its error. `lr.py` measures the curvatures
themselves.

**Cost** (measured: whole runs, child process, compilation, both stages, loss evaluation). d8 at 5: about 3 minutes
[estimate]; d8 at 20: 234 s; d8 at 200: about 24 minutes [estimate]; d8 at 490.8: about 54 minutes (52 minutes
training, 91 s evaluation); d24 at 5: about 65 minutes (52 minutes training at 311K tokens per second, 711 s
evaluation); d16 at 20: about 35 minutes (30 minutes training at 746K tokens per second, 299 s evaluation). `lr.py`:
3 × 3 min + 4 × 24 min + 3 × 65 min ≈ 5.0 h; `value_embeds.py`: 18 runs ≈ 7.0 h; ≈ 12.0 h ($51 at $4.29/h).

Where a run's time goes (2026-10-22). Steps (file-level stage, micro-batches of 12 rows, torch.profiler): d8 (one
micro-batch) 43 ms per 98,304 tokens, GPU 93% busy (GEMM 49%, FA4 18%, cross entropy 12%, Inductor 11%); d16 (two)
264 ms per 196,608 tokens, 98% busy (GEMM 49%, FA4 25%); d24 (two) 633 ms per 196,608 tokens (311K tokens per
second), 98% busy. The loss evaluation reads 640M tokens whatever the depth (390M at 8K over 11 corpora, 250M at 32K
over 6): 8K, 32K and solutions take 52.5, 35.7 and 2.4 s at d8; 173.7, 115.5 and 9.4 s at d16; 409.8, 282.9 and
18.6 s at d24. A d8 run at 20 (`d8-r20-s0-d0-2449fca0cf29`): 9 s to its run record, 135 s training, 5.5 s at the
repository stage's switch (snapshot and recompilation at 32K), 90 s evaluation. So the runs are bound by library
kernels (QuACK GEMMs, FlashAttention-4) at ≥ 93% GPU occupancy; d8's 7% idle is the most CUDA graphs could recover
(≈ 11 minutes of `lr.py`), against capture constraints on the varlen and custom-kernel paths production does not need
at d24 (98% busy), so no kernel or graph work is spent on the experiments. Their cost lever is the number and size of
runs.

**Records.** A run's tag is production's `config.run_tag` over its fields (its peak lr and the dataset revision
hashed with them), and a rerun experiment resumes: a run with an evaluation or a divergence is not trained again, a
stopped one trains again from its start (its one checkpoint is its final model); each run is launched from its fields
(`python -m ablations.harness '<fields>'`). Per run, in production's schemas: `artifacts/metrics/<tag>.jsonl`
(`nanochat/train/metrics.py`: every step and attempt, the learning curve and gradient norms included; the run record
holds the model config, the run's fields as `variant`, parameters, FLOPs per token, the clip threshold, the plan, lr
and weight decay; the end record the peak memory; a failure record the step a run diverged at),
`artifacts/base_eval/<tag>-step<steps>.json` (`eval/suite.py evaluate`, likelihood metrics only) and
`artifacts/logs/<tag>.log` (the run's output, a failure's traceback included). Each experiment gathers every tracked
metric of its runs into `results/`, so tables and figures need no rerun and no `artifacts/`: `<experiment>.jsonl`, a
record per run (arm, tag, setting, fields; `eval/suite.py record`: its metrics records but the per-step ones: run,
stage, end, failure; its evaluation verbatim: per corpus and context bucket bits, bytes and rows at 8K and 32K, per
benchmark solution bits and bytes; the compared measures; efficiency; the step it diverged at, measures and
efficiency null then); `<experiment>-curves.jsonl`, per run its per-step train records as columns (step, tokens,
loss, lr, gradient norm before clipping, tokens per second, attempt, time; 2,987 steps of d8 at 20 take 360 KB, so all
runs about 48 MB); and `<experiment>.json`, per setting depth, tokens per parameter, seeds, production's evaluated
runs, and per alternative its fields, evaluated runs and per measure difference, p and confidence interval, null for
one run per arm; `lr.json` adds the Analysis (`analysis`: per setting runs, curvature, vertex and grid extension; m_B,
m_D, the law's fall, both predictions, the measured d8 ratio, production's multiplier and lr, null while a grid needs
extending).

### Verification of the experiments' code

2026-10-20 unless stated:

- *Arms* share production's plan at their setting: d8 at 50, 7,468 steps of 98,304 tokens (`repo_start` 7,061, peak
  lr 2.4378e-3); d8 at 490.8, 36,656 steps of 196,608 tokens (34,657, 1.8324e-3); d24 at 5, 3,307 steps of 294,912
  tokens (3,127, 1.6641e-3); the τ law sets each arm's weight decay from its lr, keeping the decay's timescale as the
  lr moves. Production's d24 plan unchanged: 81,171 steps, `repo_start` 76,743.
- *Candidate* (`model.py`, real rows, 2026-10-21): with production's fields `Candidate` equals `GPT` bitwise.
- *Every arm* (2026-10-22): `lr.py`'s at d8 over 50 and over 490.80456005542607 (`max_ratio` of d24, 20.6 s; × 0.25
  included) and at d24 over 5, `value_embeds.py`'s at d8 over 20 and 490.8 and at d16 over 20, compiled and trained
  4 steps through both stages on real rows with finite losses; weight decay at d8 over 490.8: 0.3597 (1.4387 at
  × 0.25, 0.7194 at × 0.5, 0.1798 at × 2), over 50: 0.3982, d24 over 5: 0.3914; parameters: d8 56,627,216, d24
  522,252,336, d16 218,124,320 (83,890,208 without value embeddings).
- *lr.py's plans* (`harness.plan_of`, τ-law weight decay at each arm's lr): d8 at 5, 746 steps of 98,304 tokens
  (`repo_start` 705), law lr 3.2509e-3, weight decay 0.8878 / 0.4439 / 0.2219 at × 1 / 2 / 4; d8 at 200, 14,937 steps
  of 196,608 (14,122), 2.0500e-3, 0.4916 / 0.2458 / 0.1229 / 0.0614 at × 1 / 2 / 4 / 8; d24 at 5, 3,307 steps of
  294,912 (3,127), 1.6641e-3, 0.3914 / 0.1957 / 0.0979. From the plans: m_B 4 (production) and 2 (d8), m_D 98.18
  (production) and 40.05 (d8), the law's fall 0.6306; predictions for the d8 ratio × 0.3544 (SDE rule) and × 4.9231
  (Step-Law).
- *Analysis* (planted code, `lr.py` through `harness.main` over planted records, optima × 2.4 and × 3.0 at d8 over 5
  and 200, × 1.8 at d24 over 5; before the 2026-10-20 amendment): a quadratic in u is recovered exactly (vertices
  × 2.4000, 3.0000, 1.8000; r × 1.2500, horizon exponent 0.060472, production's multiplier × 3.359307 = 1.8 ·
  1.25^(ln 98.18 / ln 40.05) · √2, lr 3.1509e-3), also with d8 over 200's × 8 diverged (its 3 other runs); an optimum
  outside its grid (× 8 at d8 over 5) is reported as extend "up" with the derived values null; a curve steeper above
  its optimum (0.02 (e^d − 1 − d), d = u − u*) biases each vertex low, × 2.240, 2.554, 1.679 (r × 1.140, multiplier
  × 2.795 against 3.359). The amended fit (synthetic records): an optimum × 2.4 on a symmetric curve read × 2.4000
  from × 1–8; × 3.0 on a curve four times steeper below read × 3.72 (one quadratic: × 3.95); a lowest run at either
  end of the grid extends it; a diverged run above the lowest gives the lowest run's multiplier.
- *Production's arm, run 0 at 20 tokens per parameter* (whole run through `harness.evaluation` and its efficiency,
  2026-10-22, 248 s, unclipped): code 0.6354, solutions 0.6964, repo_32k 0.6201, repo_beyond_8k 0.4048, text 1.3619,
  math 1.3498; 2.28M tokens per second; peak memory 14.0 GB. The same run clipped at 1.0, launched from its fields
  (`d8-r20-s0-d0-73e955e0631c`, 237 s): code 0.6320, solutions 0.6958, repo_32k 0.6158, repo_beyond_8k 0.3938, text
  1.3614, math 1.3473, 2.27M tokens per second; 75.4% of its steps clipped (one seed each, under σ's resolution).
- *Divergence* (planted failure records, `harness.main`): diverged runs are skipped on rerun, written with their step
  and null measures, and dropped from the comparisons (an arm left without runs is reported with 0 evaluated runs).
- *Export* (`d8-r20-s0-d0-73e955e0631c` through `harness.main`): its record holds the run, stage and end records, the
  evaluation verbatim (11 corpora at 8K, 6 at 32K, 3 benchmarks) and the measures; its curve 8 columns of 2,987 steps.
- *Harness end to end* (a scratch experiment through `harness.main`: d8 at 20, seed 3, production and × 2, before and
  after the harness's consolidation): both runs trained, evaluated and exported; the current harness over the earlier
  run's artifacts writes the same `.json` bytes, the same curves and the same records less the dropped `probes` field.
  Retrained, the runs agree on their first 5 steps bitwise and then drift by the backward's atomics
  (Reproducibility): last-5% train loss 1.69657 vs 1.69695 (× 1), 1.65248 vs 1.65172 (× 2); code 0.63008 vs 0.63015,
  0.61311 vs 0.61251.
- *Real data* (screen run 0, `d8-r20-s0-d0-2449fca0cf29`): the data on disk is the pinned Hub revision
  (`6744e406…`, the download's recorded commit); the run record's mixture is the 11 production corpora at their
  weights; the first training micro-batch, rebuilt as `train` reads it (data seed 0), decodes to real C++ files and
  a FIM example; the loss evaluation reads the held-out val splits (stack_edu_python's first document: a word2vec
  module) and the benchmarks' reference solutions (HumanEval+ 164, MBPP+ 378, HumanEval-X C++ 164). The trained
  model's loss on that batch is 1.869 nats against 11.705 on uniformly random tokens (ln V = 10.397).

## H100 implementation

Ops in `nanochat/ops/`, CUDA sources in `nanochat/csrc/` (built on first use by `native.cuda_library`). Each is a
`torch.library` op, so `torch.compile(fullgraph=True)` traces one graph per stage. Each choice below was measured
against its alternative with `torch.profiler` over training steps.

| Op | Implementation |
|---|---|
| Projection, MLP, LM-head GEMMs | forward and dX: QuACK SM90 GEMMs autotuned by QuACK (`quack.gemm_interface`, `tuned=True`); fp32 weight gradients: cuBLAS `torch.mm`, faster and more accurate than QuACK's tiles on their small outputs (The d24 step) |
| ReLU² MLP | QuACK `gemm_act` / `gemm_dact` |
| Cross entropy | QuACK, bf16 logits read once, the logit gradient written over them |
| Attention | FlashAttention-4 varlen, its default backward |
| q/k norm + gain + RoPE | `csrc/qk_prologue.cu`: one pass per direction; gain gradient without atomics (row math shared with the decoder, `csrc/qk_row.cuh`) |
| Embeddings | `F.embedding`, bf16 cast fused by Inductor; backward adds rows with atomics |
| Decoding | `csrc/decode.cu` (`inference/decoder.py`): batched native decoder for the code benchmarks |

Not adopted, without a measured gain at this scale (a 2026 survey of torchtitan, OLMo-core, Megatron, nanotron,
Marin, modded-nanogpt, nanochat): FP8 training, Liger and Cut Cross-Entropy losses, fused RMSNorm/RoPE libraries, DCP
checkpointing, config frameworks.

### The d24 step

On real rows (12 micro-batches of 12 rows, `torch.profiler`): 3,741 ms, 315K tok/s, 100% GPU-busy, peak 69.6 GiB;
GEMM 53%, FA4 25%, Inductor 14%; the repository stage's (12 of 3 rows) 4,016 ms, measured with Inductor's
coordinate-descent tuning (0.8% faster at the file level; off, Reproducibility). At 3,741 ms the step computes
539 TFLOP/s by M plus `lm_head` (54.5% of the H100's 989 dense bf16), 369 TFLOP/s (37.3%) counting the weight
matrices alone, about 700 TFLOP/s (71%) over the GEMMs' own time (1,966–1,977 ms a step): the rest of the step is
attention and Inductor's kernels. Production's step on 2026-10-21 (two runs): 3,700 / 3,689 ms, GEMM 1,966 ms, FA4
935 ms, peak 69.6 GiB. The configurations measured (file-level stage unless stated):

| Configuration | Step | Throughput | Notes |
|---|---|---|---|
| Per-role QuACK configs measured at M = 98,304, cuBLAS weight gradients | 3,785 ms | 311.7K tok/s | q/k prologue backward 168.4 ms; VE-gate fp32 GEMMs on CUDA cores (sm80_xmma f32 + simt sgemm) 80.6 ms |
| VE gate as fp32 broadcast-sum, persistent atomic-free q/k prologue backward | 3,724 ms | 316.8K tok/s | q/k prologue forward 75.2 ms, backward 92.7 ms |
| Production regime (d2z, τ-law weight decay, no clipping), per-role configs | 3,736 ms (step 5) | 315.7K tok/s | steps 2–7: 3,461 / 3,719 / 3,775 / 3,790 / 3,795 / 3,781 ms; peak 69.6 GiB; GPU 100% busy: GEMM 52.9%, FA4 24.9%, Inductor 14.3%, other 6.4%, cross entropy 1.5%. 1 × 12 rows: step 4 327 ms, 301.0K tok/s, peak 67.6 GiB |
| QuACK's own autotuner (`tuned=True`) for every GEMM, 2026-10-20 | 4,051 ms (step 5) | 291.2K tok/s | GEMM 2,408.1 ms (59.4%), FA4 875.0, Inductor 491.1, q/k prologue 168.1, cross entropy 57.5; peak 76.7 GiB (the autotuner's operand copies on a cold cache) |
| QuACK's own autotuner, repository stage (3 × 32,768 rows) | 4,325 ms (step 5) | 272.7K tok/s | GEMM 2,403.2 ms, FA4 997.2 ms; peak 69.4 GiB; 96% busy |
| **Final:** QuACK's autotuner for forward and dX, cuBLAS (`torch.mm`) for every fp32 weight gradient, 2026-10-20 | 3,705 / 3,713 ms (two runs) | 318.4K / 317.7K tok/s | GEMM 1,976.7 / 1,974.2 ms (53.3%), FA4 937, Inductor 496, q/k prologue 171, cross entropy 57; peak 69.6 GiB |
| **Final**, repository stage (3 × 32,768 rows) | 4,016 ms (step 5) | 293.8K tok/s | GEMM 1,974.3 ms, FA4 1,064.3 ms; peak 69.6 GiB; 96% busy |

Why the autotuner alone was slower (per GEMM on d24's real operands after two optimizer steps, CUDA-graph median of
5): the weight gradient dY^T X has a small output (768 × 768 or 768 × 3,072) and K = 98,304 tokens. QuACK's `gemm`
defaults to split-K 1, so the output's 24–96 tiles leave most of the 132 SMs idle, and its autotuner never offers
cuBLAS: c_q/c_k/c_v/c_o dW 0.544 ms (rel. error vs fp64 2.3e-4) vs cuBLAS 0.191 ms (1.8e-5), 1,152 calls per step
(+406 ms); MLP dW 0.700 / 0.751 vs 0.658 / 0.653 ms. With split-K, QuACK still trails cuBLAS (768 × 768: 0.214 vs
0.163 ms). Its forward and dX picks are within noise of the old per-role table (proj_fwd 0.174 vs 0.180 ms, mlp_dx
0.675 vs 0.706, proj_dx 0.198 vs 0.180, mlp_down 0.693 vs 0.649). The autotuner times each candidate once and caches
its pick on disk (`artifacts/cache/quack`), so a re-tune can move a role by ±20 ms per step.

### q/k prologue and micro-batch

`csrc/qk_prologue.cu` rather than Inductor's fusion of the PyTorch ops, and micro-batches of 98,304 tokens rather than
65,536 (1% faster per token, peak 69.6 GiB of 80 GB). Composite d24 (Block with QK gains, VE gate 2 sigmoid, torch
AdamW + Muon), real data, one step of 1,179,648 tokens (torch.profiler, the micro-batches and rows of each row below),
2026-10-19. GPU 99% busy in each: GEMM 54%, FA4 24%, Inductor 12%.

| q/k prologue | Rows | Step | Throughput | Notes |
|---|---|---|---|---|
| Inductor-fused PyTorch ops | 18 × 8 | 3,890 ms | 298.2K tok/s | Inductor 789 ms |
| `csrc/qk_prologue.cu` | 18 × 8 | 3,792 ms | 305.8K tok/s | Inductor 453 ms + the kernel's 228 ms in "other" |
| `csrc/qk_prologue.cu` | 12 × 12 (98,304-token micro-batches) | 3,756 ms | 308.8K tok/s | peak 69.6 GiB (8 rows: 48.9 GiB) |

### Attention

FlashAttention-4 rather than cuDNN: faster forward and backward in every measured case. Causal varlen forward +
backward on real inputs (`d8-r20-s42`: first-layer q, k, v and output gradient per window, the file stage's first fim
micro-batch, 12 × 8192, 103 segments; H 2 × D 128); FA4 (`flash_attn.cute` SM90) vs cuDNN 9.26 (Transformer Engine
2.19 DotProductAttention, thd, fused backend); GPU shared with another job. Error: max |diff| / max |FA4| of o, dq,
dk, dv.

| Window | FA4 fwd | FA4 fwd+bwd | cuDNN fwd | cuDNN fwd+bwd | o / dq / dk / dv |
|---|---|---|---|---|---|
| 512 | 0.379 ms | 1.467 ms | 1.376 ms | 4.959 ms | 4.0e-3 / 7.5e-3 / 3.4e-3 / 3.9e-4 |
| 8192 | 0.465 ms | 1.830 ms | 1.618 ms | 5.850 ms | 5.0e-3 / 6.1e-3 / 4.3e-3 / 6.6e-4 |

FA4 backward, d24, one real micro-batch, compiled forward + backward repeated 6 times (median): dq accumulated with
atomics (FA4's default) 295.6–298.3 ms; `deterministic=True` 299.9–303.9 ms.

FA4 full-context layers (window = T), d8 real rows (12 × 8192, 3,300 cu_seqlens entries), 2026-10-19, GPU shared with a
running d8 study (absolute times vary by tens of percent between runs; pairs interleaved in one process):
`causal=True` with `window_size_left = T` (FA4 resolves it to its local kernel) vs `None` (its causal kernel): outputs
and dq/dk/dv bitwise equal (file stage; repository stage 3 × 32768: o, dk, dv equal, dq max |diff| 2.4e-4, atomic
order); forward + backward 3.358–3.412 vs 3.301–3.386 ms (file), 12.38–12.49 vs 12.17–12.34 ms (repository, 2 of 3
rounds; the third disturbed) → `ops/flash_attn.py` passes `None`.

### Value-embedding gate

An fp32 broadcast-sum rather than fp32 GEMMs: forward + backward of gate and mix per layer at M = 98,304
(`torch.utils.benchmark` median), relative error vs fp64 of gate / dW:

| Implementation | Time | Error |
|---|---|---|
| fp32 `F.linear` | 1.195 ms | 1.3e-7 / 5.6e-7 |
| bf16 `F.linear` | 0.575 ms | 8.1e-4 / 3.0e-3 |
| TF32 | 1.008 ms | 8.0e-5 / 2.2e-4 |
| fp32 broadcast-sum (kept) | 0.881 ms | 5.3e-8 / 4.1e-7 |

### Fused q/k/v projection

Split rather than one QuACK GEMM on cat(c_q, c_k, c_v) weights (N = 3C, the cat's backward splitting dW), d24 12 × 12
file, 2026-10-19, GPU alone, one profiled step each. Per-GEMM timing (5 interleaved rounds of CUDA graphs ≥ 30 ms;
real operands: d8 / d24 at `init_weights(seed=0)` after two optimizer steps), fastest config of the fused GEMM vs 3 ×
the split one: C 768 forward 0.4920 vs 0.5085 ms, dX 0.4595 vs 0.4725, dW 0.4865 (QuACK 128 × 256 split-K 2) vs 0.4830
(cuBLAS); C 256 0.0778 / 0.0723 / 0.0786 vs 0.1143 / 0.1140 / 0.1281. The split projection already runs at 684 TFLOPS
(2 × 98,304 × 768² FLOP in 0.1695 ms), the fused at 707.

| Layout | Step | Peak | GEMM | Inductor | q/k prologue | Other |
|---|---|---|---|---|---|---|
| split (kept) | 3,712 ms (317.8K tok/s) | 69.6 GiB | 1,981.7 ms | 495.5 ms | 171.2 ms | |
| fused, strided q/k/v views of the (B, T, 3C) output, autograd stack of dq, dk, dv | 3,827 ms | 71.4 GiB | 1,966.8 ms | 648.1 ms (the stack 185.6) | 172.8 ms | |
| fused, prologue on the (B, T, 3, H, D) output, writing dq, dk into the (B, T, 3C) gradient | 3,729 ms | 73.1 GiB | 1,982.5 ms | 468.3 ms | 173.9 ms | v/dv copies 59.1 ms |

→ split kept: fusion saves no measurable GEMM time (sweep: ≤ 10 ms per step), less than the copies its layout needs.

### LM head and evaluation losses (d8)

d8 real rows (file stage's first fim micro-batch, 12 × 8192; H 2 × D 128), 2026-10-19, GPU shared (pairs interleaved):

- **LM-head loss**, M 98,304 × V 32,768, C 256: ours (bf16 logits GEMM, QuACK CrossEntropy, dx/dW GEMMs) 29.2 ms;
  QuACK `scaled_exp_linear_cross_entropy_fwd` at chunk 4,096 / 8,192 / 16,384 / 32,768: 45.0 / 34.4 / 31.4 / 31.7 ms,
  dx and dW relative difference to ours 2.6e-3 / 2.1e-3 → ours kept. `lm_dx` GEMM output fp32 vs bf16: 7.84–7.99 ms
  both; bf16 kept (an output dtype change of the op collides with graphs in the shared Inductor cache).
- **Per-position evaluation losses:** bf16 logits + CrossEntropy 16.2–16.3 ms, relative error vs fp64 8e-4; QuACK
  `lse_target_epi` (no logits) 14.9–15.0 ms, 5.0e-8 → `lse_target_epi`.

### Concurrency

Runs train one at a time. Concurrent d8 training processes, 2026-10-20 (file-level stage, each timing a
16-micro-batch step of 1,572,864 tokens, started together; peak 13.3 GiB each): 1 process 2,598,140 tok/s; 2 processes
1,155,960 + 1,179,686 = 2,335,646 aggregate (−10%); 4 processes 589,606 + 614,366 + 672,177 + 692,019 = 2,568,168
(−1%): one d8 run saturates the GPU (its 280–440 W power draw understates SM occupancy).

## Reproducibility

- **Fastest kernels.** FA4's and the embedding backward reduce with atomics, so repeated steps differ in the last bits;
  FA4's deterministic backward costs 1–3% of a d24 micro-batch's forward and backward (Attention). The forward is
  deterministic: Inductor's coordinate-descent tuning is off (`runs/container.sh`), and a compiled loss evaluation is
  bitwise equal across processes. On, it picks kernel configurations by timing, so two processes of the same code
  compile different kernels: d8 compiled loss differs by 6.7e-6 and per-token losses by up to 3.5e-2 between two
  baseline runs; off, both are bitwise equal across processes (2026-10-20), for 0.8% of the step (d24 12 × 12 file,
  2026-10-19, GPU alone, one profiled step each: off 3,741 ms, 315.3K tok/s, Inductor kernels 533.3 ms; on 3,711 ms,
  317.9K tok/s, 500.2 ms; GEMM and FA4 differ by run noise, ±10 ms). From two Inductor/Triton caches with the tuning
  on, the same code (d8, seed 0, real data, 2026-10-20) agrees on the first steps and then differs by ~1e-6 relative;
  over 2,987 steps the mean loss of the last 5% of steps differed by 2e-4 (1.70273 vs 1.70255).
- **Resume.** A relaunched run continues from its newest checkpoint (weights, optimizer state, data position), written
  to a temporary file and renamed, so a kill mid-save leaves the previous one; checkpoints at each tenth of the run
  bound a failure's loss to a tenth.
- **Environment.** `docker/Dockerfile` pins base images by digest and git dependencies by commit; `uv.lock` the Python
  packages; fp32 matmuls stay fp32 (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`).
- **Run identity.** A run's tag (`config.run_tag`) is its depth, ratio and seed plus a hash of those, its variant
  (an ablation run's fields; production's lr scale) and the dataset revision `config.DATASET_REVISION`
  (`python -m nanochat.paths` downloads it), so runs of different settings or data never share artifacts. Each run's
  metrics record its full configuration (model, plan, seed, lr, weight decay, clip threshold) and the dataset revision.
- **Data order.** Each corpus's rows are permuted once by the run's seed (NumPy `RandomState`; `train/stages.py`
  `stage_batches`), which also seeds its initialization.

## Reproducing the run

On a fresh Lambda `gpu_1x_h100_sxm5` (Ubuntu with Docker and the NVIDIA driver):

```bash
git clone https://github.com/indrajeetadityaroy9/nanochat nanochat && cd nanochat
HF_TOKEN=<read token> runs/setup.sh   # CDI spec, docker group, image, artifacts/, tokenizer + data (nanochat/paths.py)
# log in again (docker group), then from the checkout:
tmux new -d -s h100 'runs/container.sh python -m scripts.production 2>&1 | tee -a artifacts/logs/production.log'
```

Everything a run reads or writes lives in the checkout (`nanochat/paths.py`): `artifacts/` (gitignored, created by
`runs/setup.sh`) holds `data/`, `tokenizer/`, `checkpoints/<tag>-step<step>.pt` (each a run's full state),
`metrics/<tag>.jsonl`, `base_eval/<tag>-step<step>.json`, `logs/` (`production.log`, the launch's output; `<tag>.log`
per ablation run) and `cache/`; `results/` (tracked) holds the data the paper's tables and figures are drawn from.

`scripts/production.py` derives the ratio and the tag (`d24-r490.774-s0-<hash>`), trains both stages with a checkpoint
(the run's full state: weights, optimizer state and data position, ~5.6 GB) at each tenth of the steps and at the
repository stage's start, then evaluates every checkpoint (`evaluate`: the curve over training, and the final model)
and exports the run's tracked metrics (`eval/suite.py export`): `results/<tag>.jsonl` (its records and every
evaluation) and `results/<tag>-curves.jsonl` (its per-step records). Training and each evaluation run in a fresh
process (torch.compile's recompile limit and the CUDA caching allocator's reservations are per process). Rerunning
continues a stopped run from its newest checkpoint and evaluates every checkpoint again. The evaluation (Evaluation,
below) reports the likelihood metrics and the benchmarks' pass@1, each scored by its official tools.

**Expected cost:** 76,739 file-level steps at 3.741 s and 4,427 repository-stage steps at about 4.05 s (4,016 ms with
coordinate-descent tuning, plus its 0.8%), about 85 h, about $365 at $4.29/h.

`runs/release.sh <tag> <repo>` uploads the checkpoints, the run records under `artifacts/` (metrics and
evaluations) and `results/` to a Hugging Face model repository; `runs/container.sh hf download <repo>
--local-dir /workspace/nanochat` (the checkout inside the container) restores them where `load_model` and a resumed
production run read them.

## Pre-launch verification

### The model against its specification

Each op and the whole model against fp64 restatements written from the Model table, forward and backward (2026-10-20):
on NVIDIA GB10 (sm_121, where FA4 runs its SM120 kernels) and on a fresh H100 at `7faf11f` (this image's SM90
kernels). d24's shapes, parameter counts, value-embedding layers, windows and FLOPs and d8's init are exactly as
stated. Relative error against the restatement:

| Check | GB10 | H100 |
|---|---|---|
| GEMMs and ReLU² (fp32 weight gradients) | 3.7e-3 (4.4e-6) | 3.7e-3 (4.4e-6) |
| q/k prologue (gain gradients) | 1.7e-3 (1.5e-7) | 1.7e-3 (2.0e-7, bitwise repeatable) |
| FA4 output and gradients | 3.6e-3 | 3.6e-3 |
| FA4 window boundary: worst row, against either off-by-one (keys 512 back attended, 513 not) | 2.7e-3 against 1.0–2.3 | 2.7e-3 against 1.06–2.20 |
| FA4 without the document mask | 4.1 | 4.2 |
| Cross entropy | 1.7e-3 | 1.7e-3 |
| Whole model (4 × 256, every component perturbed to matter): per-position loss | 5.0e-3 | 5.2e-3 mean, 2.5e-2 max |
| Restatements each breaking one rule (no x0, σ gate, no value embeddings, RoPE +θ, no document mask) | 5.1e-2–1.5e-1 from the model | the model's deviation projects at most 0.001 onto each (1 would be that rule) |
| Gradients | 1.7e-2 | 2.1e-2 |
| Native decoder: greedy tokens the forward's argmax, the rest within | 499 of 512, 0.014 of its maximum logit | 502 of 512, 0.015 |

### The custom kernels in use

2026-10-20, a fresh Lambda `gpu_1x_h100_sxm5`, `7faf11f` cloned, this image; `torch.profiler` over `train` at
production's d24 plan (10 steps of 12 micro-batches, the 32K stage from step 5, synthetic rows of ~950-token
documents), the compiled evaluation losses and the native decoder:

- **Built:** `native.cuda_library` compiles `qk_prologue.cu` and `decode.cu` as sm_90 SASS, every kernel present and
  none spilling; Inductor's generated forward and backward call every `nanochat::` op.
- **Training,** per step at both contexts, the counts the architecture implies: QuACK GEMMs 2,892 (121 a micro-batch
  forward, 120 dX), cuBLAS fp32 weight gradients 1,728 and the head's 24, and 288 (12 micro-batches × 24 layers) each
  of `k_forward`, `k_backward`, `k_gain_sum`, FA4 SM90 forward and backward, `gemm_act` and `gemm_dact`; QuACK cross
  entropy 12. Of the step's 3.41 s of GPU time these take 81%, Inductor's Triton 16.5%, the optimizers and gradient
  accumulation 2.4%.
- **Throughput is not a production estimate:** the GEMMs take 1,983 ms a step, as on real rows, but FA4 570 ms against
  real rows' 935: equal-length documents carry less attention than real files of the same mean length, and far less
  than the repository stage's (342K tok/s at 8K and 341K at 32K, against 318K and 294K on real rows). Peak 76.5 GiB,
  QuACK's autotuner copying operands on a cold cache (The d24 step: 76.7 GiB; 69.6 GiB warm).
- **Evaluation:** per layer one each of `k_forward`, FA4 forward and `gemm_act` and five QuACK GEMMs, then one
  `lse_target_epi`.
- **Decoding:** every kernel `decode.cu`'s (GEMVs 69%, `k_attn` 25%, `k_norm` 6%); per step 24 `k_attn`, 50 `k_norm`
  (embedding, two per layer, final) and 97 GEMVs, the choice head on sampled and constrained steps. Healed,
  syntax-constrained requests among greedy and sampled ones keep their first token within the healing mask (5/5, fresh
  and from the prompt store).

### Operations

| Check | Result |
|---|---|
| Resume (the H100, `60138de`, packed rows): d8 at 20 tokens per parameter (`d8-r20-s0-d0-d69671850bea`), killed with `docker kill` at step 991 and relaunched | continued from its step-900 checkpoint, switched to the 32K stage at step 2,824 (`repo_start`), finished at step 2,987; the step-2,824 checkpoint kept |
| Checkpoints and lineage (2026-10-20, NVIDIA GB10; the image of this `uv.lock`; `train` on synthetic token rows, d8, 74 steps, checkpoints at 20, 40, 70 and 74) | a run killed after its step-40 checkpoint resumed there (attempt 1), switched to the 32K stage at step 70 and ended at 74; the final checkpoint reloaded reproduces the live model's loss bitwise; against the code of `results/lr.json` (`8caecb9`) the lr is bitwise equal at every step and the loss bitwise through step 5, then within 5.1e-6 relative, as an uninterrupted and a resumed run of the same code differ (4.8e-6: the backward's atomics) |
| End to end (2026-10-20, through the `base_train` and `base_eval` command lines `scripts/production.py` has since replaced with direct calls in spawned processes): training with lr scale 2.5 (d8, 74 steps), the evaluation (loss, reference solutions, EvalPlus, MultiPL-E, infilling; greedy and filtered; d8 in 46 minutes), the harness's train-evaluate-export | all complete; the recorded lr is the scale times the law's; the loss evaluation matches the pre-consolidation code to 7 decimals in bits per byte |
| Run identity (`60138de`) | the production tag stable across calls; tags of the same fields differ only by dataset revision and lr scale |

## Evaluation

Every metric is a published one, computed as its source computes it. `eval/suite.py evaluate` scores a checkpoint;
production's checkpoints take everything, the ablation runs the likelihood metrics alone.

**Likelihood metrics** (`eval/bpb.py`), in bits per byte (The Pile, Gao et al., arXiv 2101.00027: summed bits over
summed UTF-8 bytes, tokenizer-independent):

- *Held-out corpora:* each training corpus's val split at 8K and at 32K, per context bucket (≤ 8K, > 8K: Code Llama's
  loss by context length, arXiv 2308.12950 §3.3), pooled by the stage's mixture weights: the AR test loss of Bavarian
  et al. (arXiv 2207.14255 §3).
- *Reference solutions:* the bits of HumanEval+'s, MBPP+'s and HumanEval-X C++'s reference solutions given their
  prompts, the decision measure at scales where pass@1 is near zero: the task loss of Bhagia et al. (arXiv 2412.04403
  §2); the correct answer's likelihood is DataDecide's best small-scale predictor for HumanEval and MBPP (arXiv
  2504.11393 §3.3); BPB raises the signal-to-noise ratio of most benchmarks (Heineman et al., arXiv 2508.13144 §5.3)
  and tracks pass@1 across checkpoints where corpus loss does not (Xing et al., arXiv 2604.07769).

**Benchmarks** (`eval/code_eval.py`): EvalPlus HumanEval+ and MBPP+ (base and plus pass@1), MultiPL-E HumanEval and
MBPP in C++, and HumanEval single-line, multi-line and random-span infilling (PSM and SPM), each scored by its official
evaluator, by up to three generation modes:

| Mode | Decoding | Score |
|---|---|---|
| `greedy` | left-to-right argmax, one completion per problem | pass@1 as DeepSeek-Coder (arXiv 2401.14196 §4.1), Qwen2.5-Coder and EvalPlus's leaderboard report it |
| `sampled` (EvalPlus, MultiPL-E) | 20 samples per problem at temperature 0.2 from the nucleus top-p 0.95 (`code_eval.SAMPLES`, `SAMPLE_TEMPERATURE`, `TOP_P`; the decoder's `k_nucleus`) | pass@1 by Chen et al.'s unbiased estimator (Codex, arXiv 2107.03374 §2.1: top p 0.95; T = 0.2 the pass@1 optimum), as MultiPL-E (its pass@1 stable from about 20 samples) and StarCoder2 (20 samples at T = 0.2) report it, computed by each benchmark's evaluator |
| `filtered` | left-to-right prompts healed (Guidance; Lundberg 2023), Python and FIM middles under the syntax constraint (FIM: given the prefix and suffix, no healing); the greedy candidate, then seeded samples until one has no post-generation diagnostic and passes the prompt's own examples, at most 10 (`verify.ATTEMPTS`); the accepted candidate, else the greedy one | filtered pass@1 (Codex's filtered pass@k, Chen et al. arXiv 2107.03374 §3.5; AlphaCode §4.5): a system measurement, not comparable to pass@1 tables |

- **Comparability.** The sampled mode draws 20 samples per problem where Codex and MultiPL-E's paper drew 200 and
  the BigCode leaderboard 50: the same estimate, with more variance per problem. HumanEval infilling is greedy, not
  Bavarian et al.'s (temperature 0.2, 100–200 samples). The prompts' examples overlap the scored base tests (MBPP+:
  373 of 378 prompt asserts use a base input; HumanEval+: 55 of 164 tasks share a doctest input), so the filtered mode
  favours base over plus pass@1.
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
or timing. A `Request` carries the prompt, budget, stop rule, `temperature` (0 = argmax, ties to the lowest id), `top_p`
(below 1: the sample from the nucleus, Holtzman et al., ICLR 2020, §3.1; `decode.cu k_nucleus` keeps the token set
Hugging Face's `TopPLogitsWarper` keeps), `seed` (the sample at position *i* depends only on seed and *i*) and an
optional constraint (`heal()`'s first token, the Python syntax constraint, `AllOf()` of both), whose masks each step
takes for all constrained slots in one batched call (`step_masks`); a prompt whose length plus budget exceeds the
context raises `ValueError`.

- **KV cache sized to the workload** (greedy, no stop rule, 125,952 tokens per run, under `gpu.lock`, 2026-10-20): d24
  `max_len` 32,768 (the context), 32 slots: 2,379 tok/s; `max_len` 1,158 (longest HumanEval+ prompt 391 + 768 − 1),
  64 slots: 3,008 tok/s (+26%); outputs identical on all 164 HumanEval+ prompts (d8, every `max_len` tried).
- **Decoder kernels** (random-init d8 and d24, zero `lm_head` so greedy never stops, 200-token prompts + 256 tokens,
  the GPU alone, 2026-10-20). Nsight Compute on the earlier 64-slot GEMV (a warp per 16-row tile, all slots, full K):
  achieved occupancy 1.6–3.1%, DRAM throughput 0.7–3.6%; the 768-row matrices (o, proj) ran 48 warps on the whole GPU
  (65% of the d24 step), `lm_head`'s argmax 2M `atomicMax` on 64 addresses (0.6 ms a step). After (a one-warp block
  per 16-row tile and 16-slot group, register weight ring of 8 blocks, `cp.async` inputs, the norm once per step in
  `k_norm`, a warp's argmax reduced before its atomic, 256 slots), ms per step:

  | | d8, 1 | d8, 64 | d8, 256 | d24, 1 | d24, 64 | d24, 256 |
  |---|---|---|---|---|---|---|
  | before | 0.98 | 3.42 | — | 6.78 | 18.05 | — |
  | after | 0.36 | 0.57 | 1.02 | 2.78 | 3.84 | 9.94 |

  d8 checkpoint: 64 greedy HumanEval+ tasks 2.7 → 0.5 s; 144 healed, syntax-constrained, sampled requests 2.1 → 0.8 s;
  tokens identical to the 64-slot build on both and under reversed submission order. d24's tokens per second still
  rise from 128 to 256 slots, whose caches for the longest benchmark request take 21.8 GB.
- **Prompt store** (`inference/decoder.py` `prompt_store`), d8 checkpoint, 48 healed, syntax-constrained HumanEval+
  prompts × 3 rounds (128 tokens each), 2026-10-20: rounds after the first 0.39–0.40 s without the store, 0.26 s with
  it (−35%); tokens identical with and without it, and to the requests decoded alone.

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
test suite, though they overlap its base tests (Benchmarks). The type checker runs batched: on 164 programs
(HumanEval+ prompt + reference solution, half with injected errors, 26 CPUs) one basedpyright process per program took
8.46 s, one batched run 1.59 s, with identical diagnostics (85 programs, 249 diagnostics).

**Not used:**
- *Speculative decoding* (e.g. DSpark, arXiv 2607.05147): lossless by construction (greedy outputs identical), so no
  failure mode to probe; its gains target large-model serving, while this decoder is CPU-bound in constrained mode and
  a drafter would need its own training run.
- *Self-correction prompting*: a base model without instruction tuning; reasoning and repair come with SFT and RL.

## Layout

- `nanochat/config.py`: vocabulary, head width, mixture, micro-batch, stages, RoPE base, run tag.
- `nanochat/model/gpt.py`: the model, scaling parameters and FLOPs per token.
- `nanochat/ops/`, `nanochat/csrc/`: the ops above and the decoder.
- `nanochat/train/`: `optim.py` (optimizers, weight decay, gradient clip), `schedule.py` (plan, each step's stage, lr
  multiplier), `stages.py` (token budget, rows per stage, each stage's micro-batches in the run's order),
  `checkpoint.py`, `metrics.py`.
- `nanochat/eval/`: the checkpoint's evaluation and the run's export (`suite.py`), the likelihood metrics (`bpb.py`),
  the code benchmarks (`code_eval.py`: EvalPlus, MultiPL-E, HumanEval infilling) and the post-generation checks
  (`verify.py`).
- `nanochat/inference/`: the native decoder (`decoder.py`), the Python 3.12 syntax constraint
  (`syntax_constraint.py`, its Rust crate and grammar in `syntax/`).
- `nanochat/data/`: the data pipeline (Pipeline, above) and `bpe/`, its Rust crate: the SA-BPE trainer, deduplication,
  repository samples and the packer.
- `scripts/`: `production.py` (the production run end to end, a checkpoint's evaluation included), `base_train.py` (the
  training driver: setup, resume, the step loop, checkpoints and records).
- `runs/`: `setup.sh`, `container.sh`, `release.sh`.
- `results/`: the data the paper's tables and figures are drawn from.
- `ablations/`: the ablation experiments (`lr.py`, `value_embeds.py`) and their mechanism (`harness.py`, `model.py`).
- `docker/Dockerfile`: the image.
