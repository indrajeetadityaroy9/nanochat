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

Every run goes through production's code: `scripts/base_train.train` (batch and peak lr by DeepSeek LLM's laws at the
run's own compute, linear warmup and decay to zero, τ-law weight decay, the file-level then the repository stage, the
production data order by seed) and `scripts/base_eval.loss_evaluation` (the same held-out rows for every run). A run
differs from production only in the fields of `study.Run`. What an alternative needs lives in the main code where it is
intertwined with it (`GPTConfig.window_pattern`; `train(model=, max_norm=, file_stage=)`; `plan(model=)`; the q/k
prologue built per head_dim, `native.cuda_library(head_dim=)`) and here when it is isolated (`swiglu.py`). GEMMs of every
width and MLP run on QuACK's autotuned kernels (`quack.gemm_interface`, `tuned=True`).

### What is tested and why

| Choice | Production | Alternative | Evidence for production | Why it is questioned |
|---|---|---|---|---|
| Attention windows | SSSL, S = 512, last layer full | full attention in every layer | Gemma 3 §5.2 Fig. 4 (local windows near-lossless); nanochat LOG 2026-01-11 | measured on text; code reuses definitions thousands of tokens back; DeepSeek-Coder and Qwen2.5-Coder attend fully; StarCoder2 uses a 4,096-token window without a published ablation |
| head_dim | 128 | 64 (twice the heads) | nanochat LOG 2026-01-17 | measured on text; head count is one of MobileLLM's tuned dimensions (§2.2.4) |
| Gradient clipping | none | global norm 1.0 | the paired comparison (`docs/h100-pretraining.md`), no clipping within the regime | the single-seed A/B (§3): clipping at 1.0 lower on all 11 corpora, −0.003 bits per byte on average |
| Mixture, code:text:math | 87:10:3 | 70:20:10 | DeepSeek-Coder §2 (87% code) | Qwen2.5-Coder Table 3 (7B): coding 49.8 / 40.3 at 100:0:0, 43.3 / 36.2 at 85:10:5, 48.3 / 38.3 at 70:20:10; production sits next to the lowest row |
| MLP | ReLU², hidden 4C | SwiGLU, hidden 8C/3 (parameter-matched) | Primer (arXiv 2109.08668); nanochat LOG 2026-02-05 (SwiGLU worse at d12 and d24) | DeepSeek-Coder, Qwen2.5-Coder and StarCoder2 all use SwiGLU |
| Matrix optimizer | `torch.optim.Muon` (5 Newton-Schulz steps, PyTorch's coefficients) | NorMuon (`dion`, as OLMo-core uses it): Polar Express orthogonalization, neuron-wise normalization, cautious weight decay | Moonlight (arXiv 2502.16982) | nanochat, our base, trains with it (LOG 2026-01-10: cautious weight decay "solid improvements", NorMuon a small gain); NorMuon paper (arXiv 2510.05491): 6% and 15% efficiency at 124M and 350M |
| Shape | width 30·depth (d24: 24 × 768) | iso-parameter shapes from 31 × 384 to 6 × 768 | MobileLLM §2.2.2 (deep-and-thin better at 125M and 350M) | measured on commonsense tasks, not code; nanochat uses 64·depth; deeper shapes run slower per token |

### Designs

| Design | Model | Points | Runs |
|---|---|---|---|
| screen | d8 (8 × 256, 14.7M scaling parameters), 20 tokens per parameter (2,987 steps of 98,304 tokens) | 2^(6−2) fractional factorial over the first six choices, generators E = ABC, F = BCD (resolution IV: main effects aliased only with three-factor interactions; Box, Hunter & Hunter, *Statistics for Experimenters*, 2nd ed., Table 6.22; built by `pydoe.fracfact`), two seed blocks | 32 |
| shape | the d16 count (67.1M scaling parameters), 20 tokens per parameter (≈6,800 steps of 196,608 tokens): 31 × 384, 16 × 512 (production rule), 9 × 640, 6 × 768 (−2.8% to +0.8% parameters) | widths of whole 128-dim heads from one head narrower to two wider, each at the depth nearest the count | 8 |
| confirm | 16 × 512 | each factor the screen adopts at its alternative, against the shape design's 16 × 512 runs | 2 per factor |

Runs train one at a time, each in its own process (`python -m ablations.study screen|shape|confirm`); a run's tag is
production's `config.run_tag` over its fields and the code it depends on (`config.CODE`, `study.py`, `swiglu.py`), so a
result names the code that produced it. Records, in production's schemas: `artifacts/metrics/<tag>.jsonl`
(`nanochat/train/metrics.py`: every step, every attempt; the run record holds the model, MLP, clip threshold, matrix
optimizer and mixture) and `artifacts/base_eval/<tag>.json` (`scripts/base_eval.py`, loss evaluation only);
`report.py` computes every aggregate from them and writes its tables to `results/`.

### Metrics and decision rules (`report.py`)

- **Primary:** *code*, the 8K validation bits per byte of the code group at production's corpus weights; *solutions*,
  the bits per byte of the HumanEval+, MBPP+ and HumanEval-X C++ reference solutions given their prompts, pooled (the
  generation benchmarks' pass@1 is near zero at these sizes). **Secondary**, reported only: text, math, the 32K
  repository bits per byte overall and beyond the file-level context, each benchmark's solutions, tokens per second.
- **screen:** OLS of each metric on the six factors (−1 production, +1 alternative) and the seed block (24 residual
  degrees of freedom; two-factor interactions, aliased in groups, left to the residual); effect = 2 × coefficient. A
  factor is adopted iff its effect is negative with Holm-adjusted p < 0.05 (Holm across the six factors) on code or on
  solutions, and not positive with unadjusted p < 0.05 on the other.
- **shape:** OLS on the shape (contrasts against 16 × 512) and the seed block; Holm across the three contrasts; a
  shape is better by the screen's rule.
- **confirm:** an adopted factor is confirmed iff, on each metric it was adopted for, both seeds' d16 differences have
  the screen effect's sign. Only confirmed changes reach production.
- **Equal tokens.** Comparisons hold tokens fixed: the production run's length is set by its data (no document read
  twice), not by time. Tokens per second is reported, so each alternative's cost in production hours is explicit.

**Power.** The paired comparison's per-pair spread on 8K code (sd 0.0065) gives a per-run σ of about 0.0046, so a
screen main effect has a standard error of 2σ/√32 = 0.0016: 80% power at an effect of 0.0047 bits per byte
(unadjusted, 24 df), 0.0061 for Holm's strictest step (α/6).

### Not tested, and why

- **FIM rate and format.** DeepSeek-Coder's ablation (1.3B, Python; Fig. 3): 50% PSM balances infilling and
  left-to-right completion, 100% hurts completion; Bavarian et al. (arXiv 2207.14255) find FIM nearly free for
  left-to-right. The rate is fixed in the packed data, so a test repacks every corpus.
- **Vocabulary (2^15).** A test retrains the tokenizer and repacks every corpus.
- **Grouped-query attention.** Its benefit is KV-cache memory; the decoder's slots are capped at 64 by its active mask
  (`csrc/decode.cu` MAX_BATCH), and d24's cache for 64 slots of the longest benchmark request (1,158 tokens) is 5.5 GB
  (2 × 24 layers × 768 × 2 bytes per token), so no decoding throughput is to be bought.
- **Value embeddings, x0/residual scalars, QK norm, untied head, the repository stage and RoPE base, the lr and
  batch laws:** the published evidence and pre-launch measurements in `docs/h100-pretraining.md`.
- **Library alternatives without a measured gain at this scale** (a 2026 survey of torchtitan, OLMo-core, Megatron,
  nanotron, Marin, modded-nanogpt, nanochat): FP8 training, Liger and Cut Cross-Entropy losses, fused RMSNorm/RoPE
  libraries, DCP checkpointing, config frameworks. The LR schedule already follows torchtitan's (`LambdaLR` with a
  closed-form warmup and decay).

### Results

`results/screen.md`, `results/shape.md`, `results/confirm.md` and the screen's decisions `results/screen.json`
(written by `report.py`); none yet.

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
| QuACK's own autotuner (`tuned=True`), 2026-10-20 | 4,051 ms (step 5) | 291.2K tok/s | GEMM 2,408.1 ms (59.4%), FA4 875.0, Inductor 491.1, q/k prologue 168.1, cross entropy 57.5; peak 76.7 GiB |
| QuACK's own autotuner, repository stage (3 × 32,768 rows) | 4,325 ms (step 5) | 272.7K tok/s | GEMM 2,403.2 ms, FA4 997.2 ms; peak 69.4 GiB; 96% busy |

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
  - *q/k prologue at head_dim 64* (built with `-DHEAD_DIM=64`): relative error equal to head_dim 128's (1.66e-3 on
    outputs and dq/dk, ~2e-7 on gain gradients; B 2, T 8192). `decode.cu` refuses head_dim 64 at build
    (`static_assert(LANE_DIMS == 4)`).
  - *FA4 at head_dim 64* vs fp64 masked attention (varlen with a one-token document, windows 512 and full): out
    2.0e-3, dq/dk/dv 2.3–2.5e-3, as at head_dim 128.
  - *SwiGLU* (`swiglu.py`, QuACK `gemm_gated`/`gemm_dgated`) vs fp64 at M 98,304, C 256: y 3.7e-3, dx 4.2e-3, dW
    3.5e-3 / 3.5e-3 / 3.9e-3 (ReLU² on the same inputs 3.4e-3 / 4.0e-3 / 3.3e-3 / 4.1e-3); against the
    gate/up-swapped reference 0.77, so the interleave is right.
  - *NorMuon* (`dion`): one step on the 52 d8 block matrices vs the documented algorithm around dion's own Polar
    Express: at weight decay 0, median 2.6e-2 (two bf16 orthogonalizations of the same input); cautious decay applied
    to dion's own update, median 9.0e-3. Polar Express vs the exact polar factor (SVD, norm-matched): median 0.11.
  - *Every screen alternative* (full windows, head_dim 64, clip 1.0, the 70:20:10 mixture, SwiGLU, NorMuon) at d8:
    compiled, three optimizer steps on real rows, finite losses; the shapes' plans from 31 × 384 to 6 × 768.
  - *Production paths:* plans unchanged by the harness (d24: 81,171 steps of 1,179,648 tokens, `repo_start` 76,743,
    lr 9.3795e-4; d8: 2,987, 2,824, 2.7337e-3).
