# Ablations

Everything measured to choose this repository's architecture, training regime and H100 implementation, with the code
that measures it. Two kinds:

- **Architecture and regime** (§1): pre-registered designs whose runs train and evaluate through the production code
  (`study.py`, analysis `report.py`). Not yet run.
- **H100 implementation** (§2): kernel and throughput measurements that chose each op (`profile_step.py`), and the
  parity records of the code paths (§3).

H100 80GB HBM3 SXM5 (700 W cap), image `nanochat` (NGC 26.09), bf16, compiled as `base_train`, unless stated.

## 1. Architecture and regime (pre-registered)

### Base

Every run goes through production's code: `scripts/base_train.train` (batch and peak lr by DeepSeek LLM's laws,
linear warmup and decay to zero, τ-law weight decay, the file-level then the repository stage, the production data
order by seed) and `scripts/base_eval.loss_evaluation` (the same held-out rows for every run). A run differs from
production only in the fields of `study.Run`; its plan (batch, steps, peak lr, weight decay) is the production model's
at the run's shape (`study.plan_of`), so the arms of a comparison match in tokens and hyperparameters. What an
alternative needs lives in the main code where it is intertwined with it (`GPTConfig.window_pattern`;
`train(run=, weight_decay=, max_norm=, matrix_optimizer=)`, the plan and weight decay given by the caller) and here
when it is isolated (`swiglu.py`; `study.model_of` dropping the value embeddings or freezing the residual scalars at
their identity; `study.plan_of` scaling the lr or dropping the repository stage). GEMMs of every width and MLP run on
QuACK's autotuned kernels (`quack.gemm_interface`, `tuned=True`), weight gradients on cuBLAS.

### Code models against this architecture

Where production departs from the code models it is measured against, and what each departure is worth deciding:

| Choice | Production | DeepSeek-Coder / Qwen2.5-Coder / StarCoder2 / Code Llama |
|---|---|---|
| Attention | SSSL: 3 of 4 layers a 512-token window | full attention (DeepSeek-Coder, Qwen2.5-Coder, Code Llama); StarCoder2 a 4,096-token window in a 16,384 context |
| MLP | ReLU², hidden 4C | SwiGLU (DeepSeek-Coder, Qwen2.5-Coder, Code Llama); StarCoder2 GELU |
| Value embeddings | alternating layers, gated (nanochat) | none |
| Optimizer | Muon on matrices, AdamW on tables and vectors | AdamW, β = (0.9, 0.95) (DeepSeek-Coder §3.4, Code Llama) |
| Peak lr | DeepSeek LLM's law, as the peak of linear decay to zero | the law's lr held for 80% of tokens, then 31.6% and 10% of it (DeepSeek LLM §2.3, the schedule DeepSeek-Coder §3.4 follows) |
| Weight decay | τ law (d8: 0.55, d24: 0.32) | 0.1 (DeepSeek LLM §2.3) |
| Gradient clipping | none | 1.0 (DeepSeek LLM §2.3) |
| Context stages | 8K file level, then the last 5.45% of steps at 32K on repositories | Qwen2.5-Coder: the same 8K → 32K repository stage; DeepSeek-Coder: 16K repository-ordered files; Code Llama: a 16K long-context stage |
| Data | code:text:math 87:10:3 | DeepSeek-Coder 87% code, 10% code-related English, 3% Chinese; Qwen2.5-Coder code:text:math 70:20:10 |
| FIM | 50% PSM in the packed data | 50% (DeepSeek-Coder, StarCoder2); Code Llama 90% |

### What is tested and why

A choice is tested only if a result can change production: its alternative is adoptable (no rebuild of the data or
of the decoder beyond its MLP) or deletes a component, and its effect plausibly exceeds the design's resolution
(Power).

| Factor | Production | Alternative | Evidence for production | Why it is questioned | Production cost if adopted |
|---|---|---|---|---|---|
| Attention windows | SSSL, S = 512, last layer full | full attention in every layer | Gemma 3 §5.2 Fig. 4 (local windows near-lossless); nanochat LOG 2026-01-11 | measured on text; code reuses definitions thousands of tokens back; DeepSeek-Coder, Qwen2.5-Coder and Code Llama attend fully | ≈ +4.4% step, ≈ +3.7 h (§2: FA4 at window 8192 1.25× window 512 per layer on real rows; 18 of d24's 24 layers) |
| Gradient clipping | none | global norm 1.0 | the paired comparison (`docs/h100-pretraining.md`) | the single-seed A/B (§3): clipping at 1.0 lower on all 11 corpora, −0.003 bits per byte on average; DeepSeek LLM clips at 1.0 | none |
| MLP | ReLU², hidden 4C | SwiGLU, hidden 8C/3 (parameter-matched) | Primer (arXiv 2109.08668); nanochat LOG 2026-02-05 (SwiGLU worse at d12 and d24) | DeepSeek-Coder, Qwen2.5-Coder and Code Llama use SwiGLU | `csrc/decode.cu`'s MLP gated; step time in the run records |
| Matrix optimizer | `torch.optim.Muon` (5 Newton-Schulz steps) | NorMuon (`dion`, as OLMo-core uses it): Polar Express, neuron-wise normalization, cautious weight decay | Moonlight (arXiv 2502.16982) | nanochat trains with it (LOG 2026-01-10); NorMuon paper (arXiv 2510.05491): 6% and 15% efficiency at 124M and 350M | none |
| Peak lr (the lr design) | the law's lr | × 2^(k/2), k = −2..2 | DeepSeek LLM §3.1 | the law was fit for AdamW under the multi-step schedule; production uses it as the peak of linear decay to zero (59% of that schedule's mean lr), for Muon | none |
| Weight decay | τ law: λ = 1/(η τ T) | 0.1 | the τ law (`train/optim.py`), the paired comparison | the code models' 0.1 is 5× smaller at d8 and 3× at d24 | none |
| Value embeddings | 4 of 8 layers at d8 (12 of 24 at d24) | none | ResFormer (arXiv 2410.17897); MoVE (arXiv 2601.22887): −0.016 bits per byte at d12 | no code model uses them; they cost 4.7% of the d24 step and 12 vocabulary tables | none (decoder takes layers without them) |
| Residual scalars | per layer x = λ_resid·x + λ_x0·x0, learned | λ_resid = 1, λ_x0 = 0, frozen (a plain residual stream) | nanochat LOG 2026-01-11 (x0: −0.004 to −0.010 bits per byte, d8–d20) | no code model uses them; the resid scalars were never ablated apart from x0 | none (deletes 2·depth parameters) |
| Repository stage | the last 5.45% of steps at 32K on repositories | every step file-level at 8K (scored at 32K too) | Qwen2.5-Coder §3.2; DeepSeek-Coder §2 | no published ablation at this share; the stage's steps are 8% slower | none (≈ −0.4 h) |
| Shape | width 30·depth (d24: 24 × 768) | one head narrower at the same scaling parameters (d16: 31 × 384 against 16 × 512) | MobileLLM §2.2.2 (deep and thin better at 125M and 350M) | measured on commonsense tasks, not code; nanochat uses 64·depth | per-token speed in the run records; at d24 the same move gives one head narrower at 195.1M |

### Designs

| Design | Model | Points | Runs |
|---|---|---|---|
| screen | d8 (8 × 256, 14.7M scaling parameters; 2,987 steps of 98,304 tokens) and d12 (12 × 384, 33.8M; 6,882 steps of 98,304), 20 tokens per parameter | at each depth a 2^(8−4) fractional factorial over the eight screen factors, generators E = BCD, F = ACD, G = ABC, H = ABD (resolution IV: main effects aliased only with three-factor interactions; Box, Hunter & Hunter, *Statistics for Experimenters*, 2nd ed., Table 6.22; built by `pydoe.fracfact`), each point twice, every run its own seed | 2 × 32 |
| lr | d8 and d12, production | the peak lr × 0.5, 0.71, 1, 1.41, 2 | 2 × 5 × 2 seeds (two shared with the screen) |
| shape | the d16 count (67.1M scaling parameters), 20 tokens per parameter: 16 × 512 (production rule) and 31 × 384 | one head narrower and deeper at the count | 2 × 4 seeds |
| confirm | 16 × 512 | each change the screen and the lr design adopt, against the shape design's 16 × 512 runs | 4 per change |

**Cost.** Measured production throughput (25-step runs, 2026-10-20): d8 2.30M, d12 1.20M, d16 0.74M tokens per
second, so training takes 2.1, 9.4 and 30.2 minutes; with compilation and evaluation about 5, 13 and 36 minutes per run
[estimate]: screen ≈ 9.6 h, lr ≈ 2.7 h, shape ≈ 4.8 h, confirm ≈ 2.4 h per adopted change, ≈ 17 h ($74 at $4.29/h)
before confirmation.

Runs train one at a time, each in its own process (`python -m ablations.study screen|lr|shape|confirm`); a run's tag is
production's `config.run_tag` over its fields and the code it depends on (`config.CODE`, `study.py`, `swiglu.py`), so a
result names the code that produced it. Records, in production's schemas: `artifacts/metrics/<tag>.jsonl`
(`nanochat/train/metrics.py`: every step, every attempt; the run record holds the model, its MLP, value-embedding
layers and frozen parameters, the plan, lr, weight decay, clip threshold and matrix optimizer) and
`artifacts/base_eval/<tag>.json` (`scripts/base_eval.py`, loss evaluation only); `report.py` computes every aggregate
from them and writes its tables to `results/`.

### Metrics and decision rules (`report.py`)

- **Primary:** *code*, the 8K validation bits per byte of the code group at production's corpus weights; *solutions*,
  the bits per byte of the HumanEval+, MBPP+ and HumanEval-X C++ reference solutions given their prompts, pooled (the
  generation benchmarks' pass@1 is near zero at these sizes); *repo_32k*, the 32K repository bits per byte of the
  code group. **Secondary**, reported only: text, math, the repository bits per byte beyond the file-level context,
  each benchmark's solutions, tokens per second.
- **Rule.** At one depth a candidate is *better* iff its effect is negative with Holm-adjusted p < 0.05 (Holm across
  the design's candidates) on a primary metric, and not positive with unadjusted p < 0.05 on any. It is *adopted* iff
  it is better at the deepest depth of its design and its effect on each of those metrics is negative at every other
  depth too (the result replicates one scale down).
- **screen:** per depth, OLS of each metric on the eight factors (−1 production, +1 alternative; 23 residual degrees
  of freedom; two-factor interactions, aliased in groups, left to the residual); effect = 2 × coefficient.
- **lr:** per depth, OLS on the multiplier (treatment contrasts against 1); of the adopted multipliers, the one with the
  lowest mean code bits per byte at d12.
- **shape:** OLS on the shape (contrast against 16 × 512); 31 × 384 is adopted iff better.
- **confirm:** an adopted change is confirmed iff, on each metric it was adopted for, its four d16 runs are lower than
  the four 16 × 512 runs by a one-sided pooled t-test at p < 0.05. Only confirmed changes reach production, so every
  change has held at d8, d12 and d16.
- **Equal tokens.** Comparisons hold tokens fixed: the production run's length is set by its data (no document read
  twice), not by time. Tokens per second is reported, so each alternative's cost in production hours is explicit.

**Power.** The paired comparison's six runs (two configurations × three seeds, `docs/h100-pretraining.md`) give a
within-configuration σ of 0.0034 bits per byte on 8K code (0.0044 on solutions, 0.0036 on 32K repository); the same
seed moved the two configurations in opposite directions (correlation −0.98 across seeds), so seeds are not blocks
and every run has its own. A screen main effect at one depth has a standard error of 2σ/√32 = 0.0012: 80% power at an
effect of 0.0035 bits per byte (unadjusted, 23 df), 0.0047 for Holm's strictest step (α/8); at a loss–compute exponent
near 0.05 that is the gain of about 12% (16%) more compute. An lr contrast, 2 runs against 2, is coarser (standard
error σ = 0.0034): the sweep finds the optimum's side, not small differences. The shape contrast, 4 runs against 4
(6 df), has a standard error of σ/√2 = 0.0024 (80% power at about 0.008); confirm's one-sided test at about 0.007, if
σ at d16 is d8's. `report.py` estimates σ at each depth from the residuals.

### Not tested, and why

- **head_dim 64.** The decoder's attention is built for head_dim 128 (`csrc/qk_row.cuh` HEAD_DIM, `decode.cu`
  lanes), so adoption rewrites it; at d8 the test is 2 heads against 4, not d24's 6 against 12. nanochat measured 128
  better on text (LOG 2026-01-17).
- **Mixture.** The primary code metric falls with the code share by construction, and the coding gain Qwen2.5-Coder
  reports for 70:20:10 (Table 3, 7B on 5.5T tokens; non-monotone: 100:0:0 best, 85:10:5 worst) is a capability no
  14.7M-parameter run on 294M tokens shows; production keeps DeepSeek-Coder's 87:10:3.
- **Four shapes.** Four shapes × 2 seeds leave 3 residual degrees of freedom (80% power only near 0.017 bits per
  byte); the two-shape design spends the same runs on the one direction MobileLLM supports.
- **FIM rate and format.** DeepSeek-Coder's ablation (1.3B, Python; Fig. 3): 50% PSM balances infilling and
  left-to-right completion, 100% hurts completion; Bavarian et al. (arXiv 2207.14255) find FIM nearly free for
  left-to-right. The rate is fixed in the packed data, so a test repacks every corpus, and its benefit (infilling) is
  not a loss metric here.
- **Vocabulary (2^15).** A test retrains the tokenizer and repacks every corpus.
- **Grouped-query attention.** Its benefit is KV-cache memory; the decoder's slots are capped at 64 by its active mask
  (`csrc/decode.cu` MAX_BATCH), and d24's cache for 64 slots of the longest benchmark request (1,158 tokens) is 5.5 GB
  (2 × 24 layers × 768 × 2 bytes per token), so no decoding throughput is to be bought.
- **Horizon.** Every design trains at 20 tokens per parameter; production trains at 490.8. The lr and batch laws carry
  the compute and the τ law the tokens per parameter, but no design measures an effect at production's horizon: d8 at
  490.8 tokens per parameter takes 52 minutes of training per run (36,596 steps of 196,608 tokens), 4 seeds per arm.
- **QK norm, untied head, RoPE base, the batch law:** the published evidence and pre-launch measurements in
  `docs/h100-pretraining.md`.
- **Library alternatives without a measured gain at this scale** (a 2026 survey of torchtitan, OLMo-core, Megatron,
  nanotron, Marin, modded-nanogpt, nanochat): FP8 training, Liger and Cut Cross-Entropy losses, fused RMSNorm/RoPE
  libraries, DCP checkpointing, config frameworks. The LR schedule already follows torchtitan's (`LambdaLR` with a
  closed-form warmup and decay).

### Results

`results/{screen,lr,shape,confirm}.md` and the decisions `results/screen.json`, `results/lr.json` (written by
`report.py`); none yet.

## 2. H100 implementation

### q/k prologue

Composite d24 (Block with QK gains, VE gate 2 sigmoid, torch AdamW + Muon), real data, one step of 1,179,648 tokens
(`profile_step.py 24 MICRO_STEPS ROWS`), 2026-10-19. GPU 99% busy in each: GEMM 54%, FA4 24%, Inductor 12%.

| q/k prologue | Rows | Step | Throughput | Notes |
|---|---|---|---|---|
| Inductor-fused PyTorch ops | 18 × 8 | 3,890 ms | 298.2K tok/s | Inductor 789 ms |
| `csrc/qk_prologue.cu` | 18 × 8 | 3,792 ms | 305.8K tok/s | Inductor 453 ms + the kernel's 228 ms in "other" |
| `csrc/qk_prologue.cu` | 12 × 12 (98,304-token micro-batches) | 3,756 ms | 308.8K tok/s | peak 69.6 GiB (8 rows: 48.9 GiB) |

### Attention: FlashAttention-4 vs cuDNN

Causal varlen forward + backward on real inputs (`d8-r20-s42`: first-layer q, k, v and output gradient per window,
the file stage's first fim micro-batch, 12 × 8192, 103 segments; H 2 × D 128); FA4 (`flash_attn.cute` SM90) vs cuDNN
9.26 (Transformer Engine 2.19 DotProductAttention, thd, fused backend); GPU shared with another job. Error: max |diff|
/ max |FA4| of o, dq, dk, dv.

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

### The d24 step

Real data, 12 × 12 rows (`profile_step.py 24 12 12 file`):

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
5): the weight gradient dY^T X has a small output (768 × 768 or 768 × 3,072) and K = 98,304 tokens. QuACK's
`gemm` defaults to split-K 1, so the output's 24–96 tiles leave most of the 132 SMs idle, and its autotuner never
offers cuBLAS: c_q/c_k/c_v/c_o dW 0.544 ms (rel. error vs fp64 2.3e-4) vs cuBLAS 0.191 ms (1.8e-5), 1,152 calls per
step (+406 ms); MLP dW 0.700 / 0.751 vs 0.658 / 0.653 ms. With split-K, QuACK still trails cuBLAS (768 × 768: 0.214 vs
0.163 ms). Its forward and dX picks are within noise of the old per-role table (proj_fwd 0.174 vs 0.180 ms, mlp_dx
0.675 vs 0.706, proj_dx 0.198 vs 0.180, mlp_down 0.693 vs 0.649). The autotuner times each candidate once and caches
its pick on disk (`artifacts/cache/quack`), so a re-tune can move a role by ±20 ms per step.

Value-embedding gate alone, forward + backward of gate and mix per layer at M = 98,304 (`torch.utils.benchmark`
median), relative error vs fp64 of gate / dW:

| Implementation | Time | Error |
|---|---|---|
| fp32 `F.linear` | 1.195 ms | 1.3e-7 / 5.6e-7 |
| bf16 `F.linear` | 0.575 ms | 8.1e-4 / 3.0e-3 |
| TF32 | 1.008 ms | 8.0e-5 / 2.2e-4 |
| fp32 broadcast-sum (kept) | 0.881 ms | 5.3e-8 / 4.1e-7 |

### Inductor coordinate-descent tuning

`TORCHINDUCTOR_COORDINATE_DESCENT_TUNING` (`runs/container.sh`), d24 12 × 12 file, 2026-10-19, GPU alone, one profiled
step each: off 3,741 ms (315.3K tok/s), Inductor kernels 533.3 ms; on 3,711 ms (317.9K tok/s), Inductor kernels 500.2
ms (GEMM and FA4 differ by run noise, ±10 ms) → on.

### Fused q/k/v projection

One QuACK GEMM on cat(c_q, c_k, c_v) weights (N = 3C, the cat's backward splitting dW), d24 12 × 12 file, 2026-10-19,
GPU alone, one profiled step each. Per-GEMM timing (5 interleaved rounds of CUDA graphs ≥ 30 ms; real operands: d8 /
d24 at `init_weights(seed=0)` after two optimizer steps), fastest config of the fused GEMM vs 3 × the split one: C 768
forward 0.4920 vs 0.5085 ms, dX 0.4595 vs 0.4725, dW 0.4865 (QuACK 128 × 256 split-K 2) vs 0.4830 (cuBLAS); C 256
0.0778 / 0.0723 / 0.0786 vs 0.1143 / 0.1140 / 0.1281. The split projection already runs at 684 TFLOPS (2 × 98,304 ×
768² FLOP in 0.1695 ms), the fused at 707.

| Layout | Step | Peak | GEMM | Inductor | q/k prologue | Other |
|---|---|---|---|---|---|---|
| split (kept) | 3,712 ms (317.8K tok/s) | 69.6 GiB | 1,981.7 ms | 495.5 ms | 171.2 ms | |
| fused, strided q/k/v views of the (B, T, 3C) output, autograd stack of dq, dk, dv | 3,827 ms | 71.4 GiB | 1,966.8 ms | 648.1 ms (the stack 185.6) | 172.8 ms | |
| fused, prologue on the (B, T, 3, H, D) output, writing dq, dk into the (B, T, 3C) gradient | 3,729 ms | 73.1 GiB | 1,982.5 ms | 468.3 ms | 173.9 ms | v/dv copies 59.1 ms |

→ split kept: fusion saves no measurable GEMM time (sweep: ≤ 10 ms per step), less than the copies its layout needs.

### LM head and GEMM backends (d8)

d8 real rows (file stage's first fim micro-batch, 12 × 8192; H 2 × D 128), 2026-10-19, GPU shared (pairs interleaved):

- **LM-head loss**, M 98,304 × V 32,768, C 256: ours (bf16 logits GEMM, QuACK CrossEntropy, dx/dW GEMMs) 29.2 ms;
  QuACK `scaled_exp_linear_cross_entropy_fwd` at chunk 4,096 / 8,192 / 16,384 / 32,768: 45.0 / 34.4 / 31.4 / 31.7 ms,
  dx and dW relative difference to ours 2.6e-3 / 2.1e-3 → ours kept. `lm_dx` GEMM output fp32 vs bf16: 7.84–7.99 ms
  both; bf16 kept (an output dtype change of the op collides with graphs in the shared Inductor cache).
- **Per-position evaluation losses:** bf16 logits + CrossEntropy 16.2–16.3 ms, relative error vs fp64 8e-4; QuACK
  `lse_target_epi` (no logits) 14.9–15.0 ms, 5.0e-8 → `lse_target_epi`.

### Concurrency

Concurrent d8 training processes, 2026-10-20 (`profile_step.py 8 16 12 file`, each timing a 16-micro-batch step of
1,572,864 tokens, started together; peak 13.3 GiB each): 1 process 2,598,140 tok/s; 2 processes 1,155,960 + 1,179,686 =
2,335,646 aggregate (−10%); 4 processes 589,606 + 614,366 + 672,177 + 692,019 = 2,568,168 (−1%) → one d8 run saturates
the GPU (its 280–440 W power draw understates SM occupancy): runs train one at a time.

### Decoding and checking

- **Native decoder** (`inference/decoder.py`), greedy, no stop rule, 125,952 tokens per run, under `gpu.lock`,
  2026-10-20: d24 `max_len` 32,768 (the context), 32 slots: 2,379 tok/s; `max_len` 1,158 (longest HumanEval+ prompt
  391 + 768 − 1), 64 slots: 3,008 tok/s (+26%); outputs identical on all 164 HumanEval+ prompts (d8, every `max_len`
  tried) → the KV cache sized to the workload.
- **Post-generation checks** (`eval/verify.py`), 164 programs (HumanEval+ prompt + reference solution, half with
  injected errors), 26 CPUs: one basedpyright process per program 8.46 s; one batched run 1.59 s; identical
  diagnostics (85 programs, 249 diagnostics) → batched.

## 3. Regime and verification records

- **Gradient clipping at 1.0**, paired A/B: d8 at 20 tokens per scaling parameter, seed 42, both arms bitwise
  deterministic on the same data order (so they differ by the clipping alone), 2,987 steps each; val bits per byte of
  the final checkpoint on each corpus's val split, train loss mean over the last 10% of steps:

  | Arm | Val bpb mean | stack_edu_python | stack_edu_cpp | climbmix | Train | Steps with norm > 1 | Max norm |
  |---|---|---|---|---|---|---|---|
  | clip 1.0 | 0.87806 | 0.74209 | 0.67994 | 1.34683 | 1.89735 | 165 | 10.40 |
  | no clip | 0.88119 | 0.74504 | 0.68317 | 1.35029 | 1.89841 | 179 | 9.83 |

  Clipping is lower on all 11 corpora.
- **Compiler caches**, d8 (`base_train`, seed 0, real data), 2026-10-20: from two Inductor/Triton caches the same code
  agrees on the first steps and then differs by ~1e-6 relative (coordinate-descent tuning times kernels); over 2,987
  steps the mean loss of the last 5% of steps differed by 2e-4 (1.70273 vs 1.70255).
- **The ablations' components**, 2026-10-20, against fp64 references on real-scale inputs and real rows:
  - *SwiGLU* (`swiglu.py`, QuACK `gemm_gated`/`gemm_dgated`) vs fp64 at M 98,304, C 256: y 3.7e-3, dx 4.2e-3, dW
    3.5e-3 / 3.5e-3 / 3.9e-3 (ReLU² on the same inputs 3.4e-3 / 4.0e-3 / 3.3e-3 / 4.1e-3); against the
    gate/up-swapped reference 0.77, so the interleave is right.
  - *NorMuon* (`dion`): one step on the 52 d8 block matrices vs the documented algorithm around dion's own Polar
    Express: at weight decay 0, median 2.6e-2 (two bf16 orthogonalizations of the same input); cautious decay applied
    to dion's own update, median 9.0e-3. Polar Express vs the exact polar factor (SVD, norm-matched): median 0.11.
  - *Every screen alternative* (full windows, clip 1.0, SwiGLU, NorMuon, weight decay 0.1, no value embeddings,
    identity residual scalars, no repository stage) at d8: compiled, three optimizer steps on real rows, finite
    losses; the run records hold each alternative (value-embedding layers `[]`; frozen `resid_lambdas`, `x0_lambdas`,
    still 1 and 0 after the steps; clip 1.0; weight decay 0.1); without the repository stage the model ends at 8K and
    `loss_evaluation` scores it at 32K. Full windows change the forward on real rows (one file-stage micro-batch: 79
    of 3,300 segments are longer than 512 and hold 88,486 of its 98,304 tokens).
  - *Plans:* every screen arm of a depth shares production's (d8: 2,987 steps of 98,304 tokens, `repo_start` 2,824,
    lr 2.7337e-3, weight decay 0.548; d12: 6,882, 6,507, 2.1768e-3, 0.299); the shapes 16 × 512 (67.1M, 6,828 steps
    of 196,608) and 31 × 384 (67.5M, 6,862). Production's d24 plan unchanged: 81,171 steps, `repo_start` 76,743.
  - *Analysis* (`report.py` on planted effects, σ 0.0034, its metrics replaced): the screen recovers the planted code
    effects at d12 (clip −0.0065 for −0.006, value embeddings +0.0131 for +0.012, SwiGLU +0.0036 for +0.004, the
    null factors within ±0.002) and adopts clip alone; the lr design adopts the planted best multiplier; confirm
    confirms both at d16.
