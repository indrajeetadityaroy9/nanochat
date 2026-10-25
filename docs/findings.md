# Findings (working notes for the paper)

One entry per agent execution: the question, the method, the measured outputs and the conclusion, with the commits
that carry it. §1–§4 state the model, training, evaluation and hardware setup as built; §5 is the execution log,
newest last. Numbers are measurements on the GH200 (one H100 96 GB, 72 Grace cores) on the real corpus unless marked
[estimate]. Open items say so.

## 1. Model architecture (as built, `model/modeling.py`)

karpathy/nanochat 92d63d4 `gpt.py` with the wins of its experiment log and learnable QK-norm weights:

| Component | Choice | Source |
|---|---|---|
| Embedding | RMS-normed token embedding, no positional embedding | 92d63d4 `gpt.py` |
| Residual | per layer x = resid_λ·x + x0_λ·x0, x0 the normed embedding | nanochat LOG 2026-01-11 |
| Attention | MHA, head_dim 128; per-head RMS norm with a learned per-dimension weight shared across heads, then RoPE over the full head (θ = 1e6, rotation by −θ) | transformers `modeling_qwen3.py`; arXiv 2309.14322 §3.2.5 |
| Windows | SSSL: every fourth layer and the last see the full context, the others 512 keys before the query | nanochat LOG 2026-01-11; Gemma 3, arXiv 2503.19786 §5.2 |
| Value embeddings | alternating layers (the last included), added to V behind a per-head gate 2σ(·) of the normed stream, zero init | modded-nanogpt record #14; MoVE LaVE, arXiv 2601.22887 §4.1 Eq. 6 |
| MLP | ReLU², hidden 4C (SwiGLU worse at d12/d24 in nanochat's log) | nanochat LOG 2026-02-05 |
| Norms, linears | gain-free RMS norm (eps = FLT_EPSILON), bias-free linears, untied lm_head | modded-nanogpt record #8 |
| Output | plain cross entropy; optional logit cap c·tanh(z/c) (ablation `logit_softcap`) | Gemma 2; modded-nanogpt |
| Attention kernel | FlashAttention-4 (CuTe DSL) with document masking (varlen) | — |
| Vocabulary | 32,768 ids: 256 bytes, 32,508 merges, BOS, three DeepSeek-Coder FIM sentinels | `docs/corpus.md` §5 |

Shape rule (92d63d4 `scripts/base_train.py`): width = 30 × depth rounded up to whole 128-wide heads.

| Model | Layers | Hidden | Heads | Non-embedding params |
|---|---|---|---|---|
| d8 | 8 | 256 | 2 | 14.7M (tokens at 20 tokens/param: 0.294B) |
| d16 | 16 | 512 | 4 | 67.1M |
| d24 (base model) | 24 | 768 | 6 | 195.1M |

Serving without architecture change (§5.8): an out-of-tree vLLM model needs only weight and config conversions —
RoPE −θ folded into the q/k norm weights (negate their second half), HF/vLLM `sliding_window` = 513 for 512 keys
before the query, rms_norm_eps = FLT_EPSILON.

## 2. Training setup

- **Data**: `iaroy/nanochat-data` (private), shards committed at `a4ec1a08` ("nanochat corpus, curated", 2026-10-24):
  19 sources, 1,812 parquet shards, 160,061,439,381 bytes; the local copy passes `hf cache verify` (1,812 checked,
  revision `b091cb47`). Tokenized, then packed per stage (`corpus/pack.py`).
- **Stages**: file level at 8,192 tokens, then repository level at 32,768 tokens (`corpus/mixture.py`); micro-batch
  98,304 tokens. d24 plan: 69,611 steps × 1,474,560 tokens = 102.6B tokens (97.05B at 8K, 5.60B at 32K).
- **FIM** (train splits of repository code): rate 0.5 per piece; middles 90% AST (Single-Node / Aligned-Span 50/50),
  10% random character spans (arXiv 2506.00204 §3.3, §5.1); PSM/SPM 50/50 (arXiv 2207.14255); a line-initial AST
  middle takes its indentation with probability 0.5 (§5.3).
- **Optimizer**: Muon on matrices (batched Newton–Schulz) and AdamW groups, β = (0.9, 0.95), gradient clip 1.0;
  weight decay by the τ law, τ* = 1.084·TPP^−0.527 (Power Lines, arXiv 2505.13738); peak learning-rate multiplier
  from the `peak_lr` ablation (2.7970).
- **Throughput metric**: MFU = executed model FLOPs (causal, per document, per window) / (seconds × 989.4 TFLOP/s);
  the nominal count (DeepSeek's M) exceeds 100% "utilization" at 32K and is kept as the plan's compute only.

## 3. Evaluation methodology

- **Benchmarks**: HumanEval+ and MBPP+ (EvalPlus `26d6d00`, greedy, 768 new tokens, EvalPlus prompts and stops,
  scored by `evalplus.evaluate`); MultiPL-E HumanEval/MBPP C++ (dataset pin `3a9e8d2`, T = 0.2, top-p 0.95, 20
  samples, MultiPL-E's judge with `--max-workers 1`, `pass_k.py`); HumanEval infilling single-line / multi-line /
  random-span in PSM and SPM (openai/human-eval-infilling, `check_correctness`); code bits per byte at 8K and 32K on
  packed val splits with the training forward; parse rates.
- **FIM protocol**: middles scored as generated (official); the reference-line cut is a separate record
  (`greedy_reference_lines`).
- **Decoding**: NativeDecoder (`model/kernels/decode.cu`: continuous batching, prefix store, batched prefill, CUDA
  graphs) for every pass; syntax-constrained ("filtered") passes with a Rust/LR mask over Python 3.12 syntax and a
  verifier search (pyright, examples under `reliability_guard`, g++ `-fsyntax-only`).
- **Pipeline**: scoring runs on CPU pools while the GPU runs the next pass; decodes merged per mode (`a41415e`).
- **Standard-protocol cross-check**: the unmodified EvalPlus and MultiPL-E CLIs through vLLM on an exported checkpoint
  (`scripts/standard_eval.sh`).
- **Statistics** (`metrics.py`): pass@k (Codex estimator), Miller CLT, clustered and paired errors (statsmodels),
  Wilson intervals; CrossCodeEval exact match, edit similarity (difflib), identifier match, line exact match.

## 4. Hardware and measured throughput

GH200 module: GPU power limit fixed at 700 W by the host (not changeable; default 900 W); under training the SM
clock averages 1,647 MHz against 1,980 MHz in short benchmarks, so sustained GEMMs run ~14% below isolated speed.

| Model | 8K tokens/s (MFU) | 32K tokens/s (MFU) |
|---|---|---|
| d8 | 2.81M | 2.37M |
| d16 | 803,750 (0.391) | 638,405 |
| d24 | 338,039 (0.450) | 273,536 (0.465) |

d24 step at 8K (4,355 ms GPU, busy 99.4%): GEMM 60.5% (~680 TFLOP/s), FlashAttention-4 18.9% (34.7% at 32K; the
window-512 layers at 283–336 TFLOP/s against 511–600 for full attention), memory-bound kernels 19.8%, optimizer 0.6%.
Base-model pre-training ≈ 85 h at these rates.

Kernel commits since, each against its parent in one session (d24, 8K / 32K tok/s): post-attention norm in the GEMM
epilogues `373b549` 338,039 → 341,919 (+1.15%) / +0.5%; value-embedding gate `eaf14d0` 344,768 → 350,185 (+1.57%) /
277,238 → 280,201 (+1.07%), step 4,290 → 4,223 ms; weight gradients into the fp32 `.grad` and one bf16 weight copy
`f499be7` 344,925 → 347,634 (+0.79%) / 275,613 → 277,383 (+0.64%); FA4 window layers with a persistent forward and bf16
dQ accumulation `0e627ff` 344,967 → 348,218 (+0.94%) / 276,538 → 278,524 (+0.72%). Master measured as a whole (`w1`,
`eeb0c53`; the later model and train commits are bitwise or SASS-identical): 356,259 tok/s (MFU 0.473) at 8K and
282,256 (0.483) at 32K, +5.4% / +3.2% over 338,039 / 273,536; pre-training ≈ 81 h (97.1B tokens at 8K, 5.6B at 32K).

Sharing the GPU (d16 training, 150 steps): one run 808.7K tok/s; two time-sliced processes 743.9K together (−8.0%);
two MPS clients 825.0K (+2.0%). Under MPS an illegal memory access in one client kills every client (a device-side
assert does not); time-sliced processes stay isolated. So throughput is measured with the GPU alone, stable non-timing
jobs (loss runs, evaluations) share it as MPS clients within a memory budget, and new kernels' first runs stay
time-sliced.
Two operational lessons (Oct 25): a job's declared memory must cover its evaluation phase (the decoder takes
NANOCHAT_DECODER_MEMORY_FRACTION of the GPU), or co-running jobs over-commit it (two out-of-memory failures, one in
the decoder's cudaMalloc at the end of a screen run); and the compiled training forward and FA4 compile once per
distinct sequence length, so checks that run the forward prompt by prompt pad to one length (an agreement check spent
47 min compiling). A third: the profiling harness lives outside the repository, so a repository audit that removed a
convenience default (`meta_model`'s softcap) broke every scratch copy of it at start, losing one loss window.

## 5. Execution log

### 5.1 Ablation cost (AblationTime)
Question: GPU time of every planned ablation with the current code. Method: `plan_of` for each config; measured d16
and d24 throughput (8K and 32K). Outputs: per run d16 at 20 tokens/param ~33 min, d16 at 31.4 ~50 min, d8 at 200
~20 min; all ablations as planned ≈ 20 GPU-h (Muon at 20 on d16); Muon at 200 on d16 would be 13.4B tokens, ~29
GPU-h. Conclusion (user): Muon's 20 tokens/param half moves to d16, the 200 half stays d8 (`df62b22`).

### 5.2 FIM format (FimFormat)
Question: is the PSM failure a format mismatch? Outputs: training rows and evaluation prompts match (sentinel ids,
sections encoded separately, BOS ends the middle). Conclusion: not the format.

### 5.3 PSM infilling failure and fix
Question: why d16 infilled 16× worse in PSM than SPM (single-line line EM 0.022 vs 0.367). Method: per-prompt length
split; boundary statistics of 3,000 packed PSM examples vs HumanEval infilling. Outputs: not the 512-key window (PSM
0.023 vs SPM 0.353 on the 982 prompts it covers whole); 95.6% of training PSM middles started with non-whitespace
(AST nodes begin after their indentation) while every HumanEval middle starts with indentation after a line break;
the model opened new definitions or files instead. Fix: `LINE_START_RATE` 0.5 (15.6% of PSM examples line-level).
Validation (d16 at 20 tokens/param retrained on repacked data): single-line PSM line EM 0.022 → 0.175, pass@1 0.037 →
0.161, multi-line PSM line EM 0.008 → 0.045; SPM unchanged (0.367 → 0.356); code bpb unchanged (0.4392). AST-FIM's own
limitation section names the train/test span-shape mismatch (arXiv 2506.00204 §7). Commits `dc76110`, `1671b56`.
Hybrid local+global attention (HybridAttnLit, FA4Internals) is not needed for this.

### 5.4 Repacking and the packer crash
Outputs: `python -m ablations.tokenizer_split_merge repack` 176 s (new step, `df62b22`); `python -m corpus.pack`
segfaulted (core dump: stack overflow in a rayon worker on a 2 MiB std stack: rayon nesting stolen jobs while many
pack calls ran at once; compiling the tokenizer pattern needs under 128 KiB, measured by EarlierAudit); rayon workers
given 64 MiB stacks (`0de5dfb`; since `03d131e` they take the process's stack limit, 64 MiB under
`scripts/container.sh`); the corpus repack then ran 1,567 s. Profile of the
packer: 63/72 cores busy, parsing 45%, FIM encoding 25%, no duplicate parses: not worth optimizing further.

### 5.5 Syntax-constrained decoding (SyntaxDecoding)
Outputs: no false rejections (11.4M BPE and 9.2M random token steps, every real d16 output replayed, all 8,488
reference middles); masks written by Rust into pinned rows without the GIL, ends confirmed once chosen, batched
prefill, 1,024 slots; mask time per step 3.4–15.8 → 0.7–2.6 ms, filtered passes ~2× faster at ~0.9 GPU busy; pass@1
equal or higher except MBPP C++ filtered 0.101 → 0.096. Commit `55e64de`.

### 5.6 Evaluation audit against 2026 harnesses (EvalAudit)
Outputs: prompts, stops, budgets and scorers match EvalPlus, MultiPL-E and human-eval-infilling; deviations fixed in
§5.9 (FIM line cut, MultiPL-E data pin, sandboxing). Wall time of the d16 codegen half (969 s): CPU scoring with the
GPU idle 36%, likelihood 28%, generation 15% (already 85–96% SM), filtered search 22%. Conclusion: the CPU-bound part
is scoring, which an inference engine cannot fix; vLLM's value is standardness.

### 5.7 Code-model survey (CodeModelSurvey)
Outputs: Qwen2.5-Coder, Seed-Coder and OpenCoder evaluate through vLLM; DeepSeek-Coder and StarCoder2 used HF
transformers; all release safetensors + `tokenizer.json` with a standard architecture class. A nanochat derivative
with the same features is served by an out-of-tree vLLM plugin (`nanoswe-vllm`). Conclusion: native vLLM plugin, no
architecture change.

### 5.8 vLLM serving (VllmServe)
Outputs: d16 served by vLLM 0.32.0 as a native out-of-tree model (image `nanochat-vllm`); `tokenizer.json` matches
tiktoken on 108,026 held-out texts; logits vs the training forward KL 3.0–4.8e-4, top-1 98.0–99.1% (training vs itself
at a row offset: KL 1.8e-4); HumanEval+ 0.067/0.067 vs NativeDecoder 0.061/0.061 (SE 0.019), MBPP+ 0.056/0.045 vs
0.045/0.037, FIM single-line PSM/SPM line EM 0.0223/0.368 vs 0.0223/0.367. Throughput on the d16 request sets:
NativeDecoder 175 s vs vLLM 312 s (vLLM host-bound at 19% GPU); fixed-length decode (2,048 requests × 512 tokens):
d16 NativeDecoder 66.5K vs vLLM 31.0K tok/s, d24 21.8K vs 23.1K. Conclusion: NativeDecoder stays the evaluation
engine at d16; at d24 both are far below the KV-read roofline (~100–200K tok/s [estimate]) — see §5.16. Standard
protocol through the unmodified CLIs (`evalplus.codegen --backend vllm --greedy` + `evalplus.evaluate`, d16):
HumanEval+ 0.067/0.067, MBPP+ 0.053/0.042; the MultiPL-E half is deferred to the d24 checkpoints. Workarounds in the
model code only: RoPE rotated in fp32 by the model itself (−θ), `sliding_window` 513, token ids passed to every layer
for the value embeddings. Commits `3c03023`, `971b1b8`, `ac2b997`, `4214ae9`, `505dbc0`, `c38ed0c` through `ec1c841`.

### 5.9 Pipelined evaluation (EvalOverlap)
Outputs: full d16 benchmarks evaluation 1,472 → 1,064 s (−28%), GPU idle 57.4% → 34.2%, scoring time with no GPU work
483 → 8.4 s; 0 per-sample flips where the protocol is unchanged. Protocol changes: FIM scored as generated
(single-line SPM greedy pass@1 0.074 uncut vs 0.437 with the reference-line cut), MultiPL-E data at `3a9e8d2`
(identical C++ prompts/tests), example checks under `reliability_guard`. Commit `a41415e`.

### 5.10 Training-step profile (TrainProfile)
Method: the repo's `train()` on the real plan and corpus under nsys, d16 and d24, 8K and 32K. Outputs: §4 breakdown;
GPU-bound (CUDA graphs cannot help); measured and rejected — `torch.optim.Muon` −4.3% (the batched Muon stays),
max-autotune +0.05%, deterministic mode −14%, cuDNN attention slower than FA4 everywhere, QuACK split-K dW slower than
cuBLAS, FP8 via torchao (d16 rowwise −28.8%, tensorwise −9.5%; d24 tensorwise −2.8%: unfused casts), lm_head on
QuACK (−10 ms/step, cold-autotune peak 89.8 GB), the `ve_dw` rewrite (−2.6 ms/step). Kept: the qk-prologue forward per
position, 250.3 → 183.6 µs per call (+0.3% end to end, d24 200-step loss Δ last-20 +0.0035 vs seed spread 0.035),
`62b50c2`. Loss noise: d24 200 steps seed 0 vs 1 last-20 mean Δ −0.035; d16 same-seed reruns |Δ| ≈ 0.007. Training
diagnostics committed (`7e77833`): per-corpus training loss, logsumexp max and mean square, the largest QK-norm gain,
per-group update RMS and parameter norm every 10th step (from optimizer state; every step costs 0.50% of d24 tok/s);
+0.4 GiB peak at d24; the interval held-out evaluation was removed (it added ~2.1 GiB and 0.24% of step time). Audit:
the spike and clipped flags were removed (arbitrary 1.1x / 3x / 100-step thresholds; derivable from loss, grad_norm).

### 5.11 Attention kernels (AttnKernel, AttnKernel2, AttnOwner) — in progress
Outputs: every FA4 forward/backward tile and scheduling variant tried is slower than the defaults; static persistent
schedules lose to the hardware scheduler on varlen tiles. Per micro-batch at 8K: window layers 1.50 ms (fwd 285, bwd
389 TFLOP/s), full layers 3.13 ms; backward pre/post 0.30 ms per call (20% of a window layer); real segments: median
746 tokens. Hand CUDA kernels against FA4 on real d24 segments, ms per call at 8K (32K): window forward 0.369 (0.364)
vs FA4 0.413 and persistent FA4 0.359 (0.355); forward with QK-norm + RoPE fused 0.985 (1.029) vs prologue + forward
0.546 (0.556), each K block being transformed ~5 times; window backward 1.193 (1.294) vs FA4 0.771 (0.828); a prologue
backward reading FA4's fp32 dq accumulator 0.823 vs 0.405 for the dq post-process plus the plain prologue backward, so
the combined prologue + attention op is −0.90% at d24 8K. PDL in the qk prologue +0.02%. Closed: hand attention
kernels and QK-norm + RoPE inside attention; FA3 not pursued (user). Committed (`0e627ff`): the persistent FA4 window
forward and bf16 dQ accumulation in the window layers only (at most 5 partial sums per dQ row there; full layers keep
fp32 accumulation as FA3 does: 64–256 sums at 8K–32K). d24 exclusive A/B (25 steps, 8 warm-up; tok/s median, event
mean): 8K base 344,967 / 344,686, persistent forward alone 346,074 / 345,801 (+0.32%), both 348,218 / 347,862
(+0.94% / +0.92%); 32K 276,538 / 276,013 → 278,524 / 277,647 (+0.72% / +0.59%). The both-arm's wall rate (328,934)
carries one 4 s step: FA4's preprocess compiling its >256-segment variant (`BIN_BATCH_SEARCH_THRESH`) once into the
JIT disk cache. Window dq vs FA4's fp32 accumulation: norm rel 2.45e-3, median rel 0 (full layers 1.1e-6); 200-step
loss last-20 3.4113 vs 3.4024 (+0.009; seed spread 0.035).
Architecture levers ranked (GQA, more local layers, sinks, gated attention); the user approved one arm, GQA at a 2:1
query:KV ratio, screened at d16 (4 query, 2 KV heads; seed 0) against the optimizer screen's MHA baselines: last-200
train loss 1.3358 vs 1.3244 (s0) / 1.3168 (s1); code bpb 8K/32K 0.4435/0.4254 vs 0.4394/0.4212 (s1 0.4380/0.4196);
all bpb 0.5263/0.5132 vs 0.5219/0.5087; HumanEval+ 0.067/0.061 vs 0.067/0.055, MBPP+ 0.090/0.071 vs 0.074/0.061
(within noise); peak 28.7 vs 32.0 GiB. Loss and bpb worse beyond the seed spread: rejected, not committed.

### 5.12 Norm folded into GEMMs, fused QKV (NormFusion) — B1 committed, C3 open
Method: norm(x)Wᵀ = diag(rstd)·xWᵀ for the gain-free RMS norm, rstd applied as a row scale in QuACK epilogues; fused
ops have lower error vs fp64 than the bf16 path (y 2.6e-3 vs 3.2e-3). Outputs: `373b549`, d24 8K 341,919 vs 338,039
tok/s (+1.15%), 32K +0.5%, 200-step loss last-20 Δ −0.0071, peak 65.6 vs 69.3 GB. The rstd-scaled dact epilogue costs
+200 µs per layer and micro-batch over plain QuACK gemm_dact (~72 ms/step; §5.19). Norm statistics from the producing
epilogue (B2) dropped by analysis: the remaining Inductor norm/mix (231 ms/step: norm1 + mix backward 155 ms, forward
58 ms) already runs at ~3.6 TB/s, and deferring norm1 into the down-projection epilogue moves a scaling pass elsewhere
(net ≈ 0). Fused QKV (C3, one (M, 3C) projection, q/k/v as strided views): the first prototype's +0.89% at d24 8K was
measured against the combined prologue + attention op (−0.90%, §5.11), so roughly neutral against master. Port onto
master's attention path (AttnOwner, landing by C3Lander): one stacked bf16 q/k/v weight per layer, one `qkv_proj`
op (the separate `linear` op deleted), one `attention` op holding the q/k prologue, the value-embedding mix and FA4
whose backward writes every q/k/v gradient into one buffer; against master's ops on real d24 8K segments, the output,
dk, dv, the k-norm gradient and every value-embedding gradient are bitwise equal (window and full layers, with and
without value embeddings), dq and the q-norm gradient differ by 1e-6 to 2e-5 (inside master's own run-to-run spread
from FA4's atomic adds); a 2-step warm-up gave 355,980 vs 359,063 tok/s (a hint, not the A/B). The rebased candidate
(`a3dc544` + C3) passed its smoke. Open: the d24 A/B (queued) and the 200-step loss, at the base recipe's lr (×2.797)
against the postact check's master curve (§5.15: every earlier loss check ran at ×1).

### 5.13 Value embeddings and cross entropy (VeCeKernel) — gate committed
Method (TileKernels' Engram gate and weight-gradient reduce, in CUDA): the backward visits rows in token order (one sort
of the input ids per micro-batch, shared by every layer); a run of equal tokens reads its table row once and adds its
summed gradient straight into the table's `.grad` (no zeroed 100 MB gradient, no autograd add); dz in bf16, the
gate-weight gradient one cuBLAS GEMM. Outputs (`eaf14d0`): d24 8K 344,768 → 350,185 tok/s (+1.57%), 32K 277,238 →
280,201 (+1.07%); ve_gate 161.9 → 92.6 ms/step; 200-step loss last-20 +0.0095 (seed spread 0.035). Not pursued (return
per GPU-minute below the cost of validating them): QuACK's scaled-exp linear cross entropy, lm_head path 399 → 379
ms/step (+0.46%) and peak −5 GiB (Cut Cross-Entropy, arXiv 2411.09009, is at best as fast as torch.compile; its gain
is memory); a two-rows-per-warp gate forward 199 → 168 µs per call (identical output); a fused clip-grad norm and
optimizer pass (~13 ms/step).

### 5.14 Fused FP8 (FP8Fused) — closed, negative
Outputs: cuTile 1.7.0 runs on sm_90 (FP8 e4m3 GEMM correct, rel. err 6.8e-4 vs the bf16 product); a naive cuTile
GEMM reaches 361–666 TFLOP/s vs cuBLAS FP8 856–1,334 on the d24 shapes (FP8 GEMMs alone 1.3–2.0× bf16). Transformer
Engine LayerNormMLP with delayed scaling is −21% for one isolated d24 MLP forward+backward, but end to end at d24 8K
(40 steps) every TE variant is slower than the bf16 QuACK baseline (340,283 tok/s, 69.3 GB): MLP delayed 327,018
(−3.9%, 77.1 GB), MLP current 306,165 (−10.0%), MLP + q/k/v/o delayed 323,765 (−4.9%, 79.3 GB), block scaling
1×128/128×128 ~−14% (84.9 GB); loss within noise in all. torchao float8 was already negative (§5.10). Causes: not the
power cap (the TE runs held higher SM clocks, 1,681–1,750 vs 1,521 MHz at the same ~660 W); more memory traffic (DRAM
utilization 57% → 64–69%: separate cast passes and the transposed FP8 copies Hopper's FP8 weight-gradient GEMMs need,
which at K = 768 outweigh the faster GEMMs); +8–10 GB from TE saving both the bf16 fc1 output and FP8 transposes;
likely lost Inductor fusion around TE's graph breaks [inference]. Conclusion: FP8 does not pay at this model size.

### 5.15 Optimizer screen (OptScreen, OptAdopt) — screen at the wrong lr; NorMuon adopted (`646a986`)
d16 at 20 tokens/param, 1 seed each (user-approved), each arm paired with base seed 0 (same init and data order).
Outputs, end of run (bpb code 8K / 32K, all 8K / 32K; HumanEval+ and MBPP+ pass@1 base/plus):
base s0 0.4394 / 0.4212, 0.5219 / 0.5087, 0.067/0.055, 0.074/0.061; base s1 0.4380 / 0.4196, 0.5205 / 0.5071,
0.079/0.073, 0.085/0.063; NorMuon (arXiv 2510.05491) 0.4385 / 0.4205, 0.5210 / 0.5079, 0.073/0.061, 0.085/0.063;
Polar Express (arXiv 2505.16932) 0.4385 / 0.4199, 0.5211 / 0.5076, 0.091/0.085, 0.087/0.058. Training loss, last 200
steps: base s0 1.3244, s1 1.3168, NorMuon 1.3223, Polar Express 1.3217; paired Δ vs s0 over the last 50% of steps
−0.0029 (NorMuon) and −0.0019 (Polar Express), negative in all 23 hundred-step windows. Both arms beat base s0 on
every bpb measure (text and math too) by 0.0003–0.0013, with gradient norms comparable (max 4.45 / 4.46 vs 4.17) and
no divergence. Statistics: the screen's "paired noise" (0.0006; NorMuon's stopped run vs its same-configuration
rerun over 3,450 common steps: window sd 0.0005, max 0.0010) measures kernel nondeterminism only; an arm at the same
seed shares init and batches but its trajectory diverges, so the relevant noise is the seed-to-seed spread of the
treatment effect, which one seed cannot estimate (base s1 − s0: bpb −0.0014 to −0.0027, paired training-loss Δ
−0.0052). The arms' gains are about half the seed difference; significance would need ~8 paired seeds per arm. The
adoption rests instead on the literature (NorMuon: −11% steps vs Muon at 1.1B; modded-nanogpt measured NorMuon on
top of Polar Express, PR #144 over #134: −15 steps over 20 runs; Polar Express −10 steps), a consistent direction on
every measure, no sign of harm, and a small cost (d16 tok/s: base 828.8K, NorMuon 827.5K, −0.15%; Polar Express
825.9K, −0.35%). Their interaction is untested here, so the stack runs at seed 1, paired with base s1 (interaction
check and second-seed replication in one run; pre-registered: stack if its paired Δ over the last 50% is negative
and code bpb not worse at 8K and 32K, else NorMuon alone). Polar Express's first end-of-run evaluation ran out of GPU
memory beside co-running jobs and was rerun from its checkpoint. Cautious weight decay (arXiv 2510.12402): lower loss
mid-run (−0.022 near 50%), worse at the end (last-200 +0.0108, code bpb +0.0023 / +0.0043): closed (the paper finds
the best λ unchanged under CWD); 818.5K tok/s (−1.25%).
Superseded (Oct 25, 22:30): the screen ran at the wrong learning rate. Every run has learning_rate_scale 1.0 (d16
peak 1.669e-3), while the base recipe trains at the law × 2.797 (4.667e-3 at the same plan): the run app's default
was the law's lr, and only `pretrain.yaml` applied the multiplier. `peak_lr.analysis` puts every measured optimum at
×2.0–3.5 (d8 at 5 tokens/param ×3.47, curvature 0.071 bpb per (ln m)²; d8 at 200 ×3.16, 0.010; d24 at 5 ×2.02,
0.009), so the screen's baselines sat ~0.005–0.05 bpb [inference] from their optimum, 2–20× the effects measured.
Polar Express's update is larger than torch's Newton–Schulz at the same lr (CPU, bf16, its coefficients against
torch's: update RMS ×1.02–1.25 on 1024×1024 and 1024×4096, noise and σ ∝ i^-0.5 or i^-1; NS 0.77–0.97 of an exactly
orthogonal update, PE 0.95–0.99): at ×1 that step alone is worth ~0.002–0.004 bpb by the curvature, its whole
−0.0019, and at ×2.797 it costs ≤ 0.0004. NorMuon preserves the update norm, so its −0.0029 is no step effect. The
stack at seed 1 (×1) lost to base s1: paired Δ +0.0008 over the last 50% (22/22 windows positive), last 10%
+0.0006, code bpb 0.4381 / 0.4196 vs 0.4380 / 0.4196; additivity predicted −0.0048, so either the arms interact
(+0.0056) or the treatment effect swings ~0.004 between seeds. Literature: Polar Express wins at every swept lr with
the optimum unchanged (arXiv 2505.16932 v5, GPT-2 Small and Large); NorMuon keeps Muon's update norm by design, and
no published comparison tunes it and Muon separately (its paper shares one lr; tuned speedrun: −15 of 2,330 steps,
modded-nanogpt PR #144; Aurora's ablation −0.014 val loss, arXiv 2606.27715); its gain comes from the rectangular
MLP matrices, none on square QKVO (DGA-Muon on NanoChat, arXiv 2610.06578; modded-nanogpt record 2025-12-11:
W_O's orientation fix left loss at 3.2780 → 3.2779). Both implementations match their references (PolarExpress's
coefficients, safety factor and normalization; NorMuon's per-row second moment and norm-preserving rescale; rows
are the right neurons here, `ve_gate` one row per head). Design (user: research first, then base vs stack): both
arms at ×2.797, seeds 0 and 1, each seed's arms one concurrent pair, parse evaluation only (HumanEval+/MBPP+ near
0.07 at d16 inform nothing); adopt the stack iff its paired Δ over the last 50% is negative at both seeds and the
seed-mean code bpb at 8K and 32K is not worse, otherwise plain Muon. The kernel loss checks ran at ×1 too (the
harness's default; C3's reference l_cpb included), so the pending postact and C3 checks run at ×2.797 against one
shared base curve.
Superseded (user): NorMuon alone, adopted on the literature (upstream karpathy/nanochat dev/LOG.md 2026-01-10:
Polar Express in the noise, NorMuon a very small improvement, kept) and one d24 stability check; the 2-seed A/B was
cancelled and Polar Express dropped. d24 (200 steps, ×2.797, seed 0, 8K, alone) vs master + postact: last-20 loss
2.2946 vs 2.2967, max grad norm after step 40 1.153 vs 1.387, no loss spike, 358,929 vs 359,229 tok/s (−0.08%),
peak equal (77.5 GiB); landed as Muon's only path. The ×1 stack at seed 1 vs base s1: paired Δ +0.0008 over the
last 50% (22/22 windows positive), code bpb 0.4381 / 0.4196 vs 0.4380 / 0.4196.

### 5.16 Decoder at d24 (DecoderD24, DecoderOwner) — committed (`4385748`), FP8 KV
Target ≥ 2× NativeDecoder's 21.8K tok/s on the fixed-length d24 benchmark, accuracy within noise. Profile (1,024
slots, positions 256–320): GEMV ~22 of a ~35 ms step (each 16-slot group re-read the weights from L2), attention ~7.8
ms. Changes: a TMA-fed WGMMA GEMM over the slots keeping every epilogue (greedy and sampled choice, syntax masks, the
BOS alternative), an e4m3 KV cache with one fp32 scale per position and head, attention streaming each slot's window
through shared memory with the scores on tensor cores (f16, exact for e4m3 keys and the bf16 q), PDL; the GEMV and
mma.sync GEMMs, the split attention and the ≤ 16-slot graphs deleted. d24 random init, 1,024 slots, prompt 256, 512
steps: main 34.8 ms/step, 29.4K tok/s (prefill 5.89 s) → bf16 KV + WGMMA 16.5 ms, 62.1K (1.68 s) → FP8 KV 10.85 ms,
94.4K (1.39 s) → scores on tensor cores 9.67 ms, 105.9K (1.32 s), 3.6×. The second audit pass changed the tree:
attention runs of 64 keys instead of 128 (−5.5% step, measured), one 64×64 WGMMA tile shape (as fast as the size
dispatch, so the large-tile path is deleted), which also restores batch-independent tokens; the final tree's numbers
come from its own benchmark. FP8 KV costs ≤ 1–2 points of pass@1 with per-tensor scales (vLLM, 2026). Final tree,
fixed-length benchmark (2,048 requests × 512 tokens, 1,048,576 output tokens): d24 main 21,809 → 94,778 tok/s (4.35×),
d16 66,559 → 189,307 (2.84×). Batch independence: 64/64 greedy and 64/64 sampled requests identical alone and batched.
Agreement with the training forward's greedy choice: final 0.9916 (137 of 16,261 tokens differ, largest top-2 logit
gap 0.56, 11 above 0.25) against main's 0.9971 (12 of 4,096, largest 0.125): FP8 KV rounding flips near-ties ~3× as
often. Committed (`4385748`; d24 step 8.27 ms, 123.9K tok/s at 1,024 slots, prefill 1.03 s, one slot 1.29 ms): the
121 extra mismatches are all near-ties (top-2 gap < 0.57, median 0.0625; 48 of 72 first divergences pick the
forward's second choice, none BOS); full d16 evaluation 862 s vs 1,064 s, per-problem changes few and two-sided
(MBPP+ greedy +4/−5 base, +2/−7 plus; HumanEval+ greedy unchanged; FIM line EM within 0.002), every path (sampled,
syntax-constrained, FIM) run end to end; FP8 KV kept.

### 5.17 DeepSeek's open kernel libraries as guidance (user-provided repositories)
Sources: DeepGEMM, FlashMLA (Hopper deep-dive at commit `ba89a34`; the 2026.09.30 release dropped Hopper), DeepEP,
TileLang. Transferable: DeepGEMM's SM90 BF16 GEMM and dense weight-gradient kernels with D = C + A·B accumulation
(benchmarked against QuACK/cuBLAS on d24 shapes, and used to write weight gradients straight into the fp32 `.grad`:
GemmDeep), its SM90 GEMM that computes pre-norm statistics inside the GEMM (reference for folding RMSNorm into GEMMs:
NormFusion) and programmatic dependent launch; its README's requirement that FP8 casts be fused into the producing
kernels, which is why unfused FP8 lost here (§5.14); FlashMLA's fused Q-norm + RoPE + attention + O-RoPE + cast kernel
(Blackwell), evidence that QK-norm and RoPE fuse into attention without losing TFLOP/s (AttnKernel); FlashMLA's Hopper
FP8 KV decode — 1×128 tile scales, RoPE dimensions kept in bf16, dequantization-bound on Hopper (no direct fp8→bf16:
~50 CUDA-core cycles per token against ~34 MMA cycles), relieved by 2-CTA clusters sharing dequantized K/V through
distributed shared memory (DecoderD24); TileLang's SM90 producer-consumer pipelines, persistent tile scheduler,
variable-length block-causal attention and TMA gather/scatter (AttnKernel's window kernel, VeCeKernel's
value-embedding gather/scatter). Not applicable: DeepEP (expert-parallel communication; one GPU, dense model).
TileKernels (DeepSeek's TileLang kernels used in its training): the Engram gate — a persistent kernel fusing two
RMSNorms, the gate and the mix, in place, saving only per-token gate values — and its weight-gradient reduction (fp32
partials per persistent block, one reduce accumulating into `.grad`, no atomics) are the designs for the value-
embedding gate and its backward (VeCeKernel); the mHC norm kernels (partial sums of squares from the producing
kernel, a small reduce finishing the RMS statistics forward and backward) the design for removing the norm pass
(NormFusion); programmatic dependent launch on every kernel. Implemented in CUDA C++ (user: all implementations in
CUDA), the TileLang sources serving as reference.
The user made DeepGEMM and TileKernels the reference for all kernel work (Oct 25; clones at DeepGEMM `5081f85`,
TileKernels `c1cf914`): DeepGemmStudy owns the GEMM area (§5.19); each TileKernels design is ported only where the
op's measured bandwidth leaves headroom or the design removes bytes.
TileKernels outcome (TileKernelsStudy, AttnKernel2): the qk prologue already runs at 3.36 TB/s forward and 3.3 TB/s
backward (~7% headroom: not ported); designs that remove bytes, recorded as follow-ups of the GEMM work: the q/k norm
+ RoPE in the fused-QKV GEMM's epilogue (no separate prologue pass; 302 MB per layer and micro-batch, ≤ 66 ms/step)
and the next layer's residual mix + norm in the down projection's epilogue (≤ 58 ms/step); FA4's dpsum in the o_proj
dX epilogue not pursued (it replaces part of a 178 µs/call preprocess).

### 5.18 Earlier executions (summary)
- **Metrics** (MetricsValidation): edit similarity by difflib as cceval (rapidfuzz, thefuzz and torchmetrics disagree
  on 46–71% of real strings); bpb bucket sums kept manual (torchmetrics SumMetric differs).
- **TokEval** (TokEvalDeps): needs tree-sitter-language-pack 1.22.1 (0.13.0 changes 35/135 values).
- **MultiPL-E scoring** (EvalPipeline): 64 concurrent `main.py` × 63 threads oversubscribed g++ and scored compile
  timeouts as SyntaxError; `--max-workers 1` per file: HumanEval C++ OK 70 → 175 of 3,220, pass@1 0.022 → 0.054;
  FIM scoring through `check_correctness` in a fork pool 211 → 33 s, 0 flips (`9d61a1b`).
- **Reference validation** (ReferenceModelEval): deepseek-coder-1.3b-base through DeepSeek's harness HumanEval 0.305
  (paper 0.348; transformers 5 drops BOS), MBPP 0.466 (paper 0.462); our EvalPlus scorer agrees on 163/164 problems.
- **d16 reference** (RefModelTrain): code bpb 0.439, HumanEval+ 0.061, MBPP+ 0.071/0.050 at 20 tokens/param.

### 5.19 GEMMs (GemmDeep, DeepGemmStudy, GemmOwner) — C1+C2, postact committed; split-K dW rejected
Outputs: DeepGEMM on the d24 shapes (TFLOP/s, sequential bench): forward q/k/v/o QuACK 828, torch 705, DeepGEMM 710;
up-projection without epilogue QuACK 652, torch 543, DeepGEMM 779 — a lead an interleaved rerun attributes to drift
(per-shape interleaved table queued); dW C×C torch.mm fp32 662, DeepGEMM with fp32 accumulation 295–315, DeepGEMM
through cuBLASLt 643. DeepGEMM's dW is slow here because its persistent scheduler has no split-K (a 768×768 fp32
output in 128-row tiles gives at most 36 tiles for 132 SMs), its split-K kernel (`bmk_bnk_mn`) needs K-major operands
(activations stored transposed), and on the 4C dW shapes (590–598) it neither prefetches across k-blocks nor overlaps
the epilogue with the next tile: not adopted as a library. C1, dW added into the fp32 `.grad` by cuBLAS addmm with
β = 1 (Megatron-LM's gradient-accumulation fusion), and C2, one bf16 copy of each q/k/v/o/up/down weight refreshed per
weight change instead of 12 casts per layer and micro-batch (`f499be7`): d24 8K 344,925 → 347,634 tok/s (+0.79%), 32K
275,613 → 277,383 (+0.64%), peak 61.6 → 62.0 GiB; numerically neutral. `b59a22b` removes the casts C2 made no-ops.
The hand sm_90a relu² GEMMs (DeepGEMM's producer/consumer layout, consumer warpgroups alternating whole tiles) are
bitwise equal to QuACK but slower: the backward 2.78 ms vs QuACK's 2.45 ms without the postact write, the up
projection 3.02 vs 1.64 ms (`gemm_norm_act`), the 2-CTA multicast variants 6–7 ms: not landed. `b59a22b` confirmed
bitwise (23/23 tensors). The MLP postact saved in the forward with QuACK's dact epilogue (not bitwise: the down
projection's weight gradient uses the forward's relu² of the fp32 pre-activation, rounded once, instead of relu²
recomputed from the bf16 pre-activation), d24 exclusive A/B (`w1`; 25 steps / 8 warm-up at 8K, 12 / 6 at 32K): master
356,259 → 361,324 tok/s (+1.42%, MFU 0.473 → 0.480), 32K 282,256 → 285,118 (+1.01%); peak 66.4 → 80.9 GB at 8K,
66.2 → 80.7 at 32K. 200-step loss at the base recipe's lr (×2.797, seed 0, same-seed master a3dc544): last-100 mean
3.0041 → 3.0014 (−0.0027; same-seed noise 0.005 at ×1), last-20 2.2984 → 2.2967; 354,550 → 359,229 tok/s over
the 189 timed steps (+1.32%), peak 68.7 → 83.2 GB (master with NorMuon). Committed. Split-K dW (`w1`, exclusive,
7 interleaved rounds, ms per call; today's cuBLAS addmm β = 1 → best split-K): C×C 0.194 → 0.184 (QuACK `gemm_add`
split_k auto; run spread 9–20%), 4C×C 0.734 → 0.751 (QuACK auto), C×4C 0.738 → 0.744 (QuACK split_k 4; DeepGEMM's
k-grouped G = 4 0.765 with the partial sum): only the four C×C dW per layer gain, −0.04 ms per layer and micro-batch,
inside the run spread: rejected. Lead (not split-K): DeepGEMM's cuBLASLt path with C = D (`bf16_gemm_tn(c=acc)`) beats
`torch.addmm` on all three dW shapes, 0.170 / 0.713 / 0.702 against 0.194 / 0.734 / 0.738 ms (−0.15 ms per layer and
micro-batch, about −55 ms of the 4.16 s step [estimate]).

### 5.20 Code audit (user specification; one auditor per file) — in progress
Specification: every parameter, literal, branch and dependency justified by mathematics, the data, model or hardware
structure, a library contract or an explicit experimental setting; arbitrary values removed at their source (not
renamed), experimental settings kept; one implementation per mechanism; behavior verified (SASS diffs, byte-identical
outputs, identical plans and analyses). Commits so far: `eeb0c53` native.py (the arch_specific flag; SASS identical for
every kernel); `b59a22b` linear.py, modeling.py (casts C2 made no-ops, a guard, a wrapper); `141e44a` run.py, conf/
(restated defaults; every experiment's plans and run keys byte-identical); `e8f3c85` ablations/ (swept scales read from
the records; analyses identical, peak_lr 2.7970369); `9b094a8` setup.sh (the SM clock lock queried from the GPU, the
same command here), .gitignore; `9d40cb1` ve_gate.cu, ve_gate.py, cross_entropy.py (unreachable guards removed; launch
sizes cited to their measurements; SASS identical); `03d131e`, `c1a7d7b`, `2b32aef` corpus/ and corpus/crate/ (the
rayon stack literal replaced by the process's stack limit; curate's in-flight row groups bounded by memory instead of
4 × cores; tokenize's per-byte worker peak measured, 8.18, instead of an estimate of 10; unused defaults; corrected
comments: RECURSION limits fissix's 2to3, not `ast.parse`) — curate, tokenize and pack on a sandbox of all 18
sources (2,062 files) byte-identical, real shard 0 of six splits packs equal to the packs on disk, the tokenizer
sample identical on the whole corpus; one intended change, the tokenizer ablation's fidelity check over a whole val
row group instead of 200 documents (its recorded counts grow); `142eae3` makes logit_softcap's learning_rate_scale
`${lr_multiplier:}` (the 2.7972 copy had drifted from the analysis' 2.79704; no softcap run exists), and `269e671`
makes it every run's default in `conf/run.yaml` (user; §5.15), pretrain and softcap jobs byte-identical, the others
exactly ×2.797 in lr. EarlierAudit: `7794e2f` vLLM serving (a second model implementation, `vllm_model.py` and its
image, removed: vLLM's Transformers backend runs `modeling_nanochat.py` at the same speed and logits; export
byte-identical but the unused pair template; transformers logits bitwise on 6-, 16- and 1,803-token prompts; the GPU
vLLM check queued), `b483f1e` eval and metrics (engine/generator protocols, dead error-bar helpers, unread keys; 77,332
cached judged items re-scored with 0 outcome differences), `865610e` syntax constraint (an unused mode, the crate's
copy of `Reemit`, unused parameters; 47,166 recorded and fuzzed token streams, 8.38M steps, replayed identical; LTO
kept, 13% faster). OptAdopt: `51c3dec` train/ (the spike and clipped flags, arbitrary thresholds over quantities the
records already hold, removed; the update-statistics interval of 10 kept by measurement, every step −0.50% at d24;
step 0's diagnostics bitwise equal to master's; −38 lines). Per-file tables filed: RunAudit, EarlierAudit,
CorpusAudit, VeCeKernel, OptAdopt (but `optim.py`'s algorithms, with the optimizer decision); the attention files
(C3Lander), the decoder (DecoderOwner) and postact's GEMM files (GemmOwner) land with their commits. Integration
check queued: a d8 run end to end through `run.py` at master, at ×1 (reference code bpb 2.7011–2.7013) and at the
new default. Noted, unchanged: `stack_edu_python`'s published row groups
(~64,920 rows) are below the 65,536 the current curate writes, so an older version wrote them — tokens do not depend
on row grouping (three shards re-cut at 122,880 rows tokenize identically).
