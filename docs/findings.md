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
segfaulted (core dump: stack overflow in a rayon worker compiling the tokenizer regex on a 2 MiB std stack, nested
work stealing); rayon pool built with 64 MiB stacks (`0de5dfb`); the corpus repack then ran 1,567 s. Profile of the
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
engine at d16; at d24 both are far below the KV-read roofline (~100–200K tok/s [estimate]) — see §5.16. Commits
`3c03023`, `971b1b8`, `ac2b997`, `4214ae9`, `505dbc0`, `c38ed0c`.

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
diagnostics (per-corpus loss, interval held-out loss, update RMS, spike flags) implemented; +2.5 GiB peak at d24 being
isolated before commit.

### 5.11 Attention kernels (AttnKernel) — in progress
Outputs so far: every FA4 forward/backward tile and scheduling variant tried is slower than the defaults; static
persistent schedules lose to the hardware scheduler on varlen tiles; a persistent window-layer forward 0.422 → 0.373
ms per call (~0.3%, d24 confirmation running). Per micro-batch at 8K: window layers 1.50 ms (fwd 285, bwd 389 TFLOP/s),
full layers 3.13 ms; backward pre/post 0.30 ms per call (20% of a window layer); real segments: median 746 tokens.
Architecture levers ranked (GQA, more local layers, sinks, gated attention); the user approved one arm, GQA at a 2:1
query:KV ratio (d16 4:2, d24 6:3), screened against the optimizer screen's baselines.

### 5.12 Norm folded into GEMMs (NormFusion) — in progress
Method: norm(x)Wᵀ = diag(rstd)·xWᵀ for the gain-free RMS norm, rstd applied as a row scale in QuACK epilogues; fused
ops have lower error vs fp64 than the bf16 path (y 2.6e-3 vs 3.2e-3). Outputs so far: d24 8K 342,010 tok/s (200
steps) vs baseline ~338K; 32K 276,001 vs 274,554; loss check pending.

### 5.13 Value embeddings and cross entropy (VeCeKernel) — in progress

### 5.14 Fused FP8 (FP8Fused) — closing
Outputs: cuTile 1.7.0 runs on sm_90 (FP8 e4m3 GEMM correct, rel. err 6.8e-4 vs the bf16 product); a naive cuTile
GEMM reaches 434 TFLOP/s vs cuBLAS FP8 1,020 on the d24 up-projection (FP8 GEMMs alone 1.3–2.0× bf16). Transformer
Engine LayerNormMLP with delayed scaling is −21% for one isolated d24 MLP forward+backward (current scaling −1%, block
scaling slower), but end to end at d24 8K (40 steps) every TE variant is slower than the bf16 QuACK baseline
(340,283 tok/s, 69.3 GB): MLP delayed 327,018 (−3.9%, 77.1 GB), MLP current 306,165 (−10.0%), all linears delayed
323,765 (−4.9%, 79.3 GB); torchao float8 was already negative (§5.10). Conclusion: FP8 does not pay at this model
size (K = 768: quantization cost comparable to the GEMM time saved); one last check caches the FP8 weights across the
15 gradient-accumulation micro-batches (TE `is_first_microbatch`).

### 5.15 Optimizer screen (OptScreen) — in progress
d16 at 20 tokens/param, 1 seed each (user-approved): baselines seeds 0 and 1, NorMuon (arXiv 2510.05491), cautious
weight decay (arXiv 2510.12402), Polar Express (arXiv 2505.16932), and all three stacked; default-off knobs.

### 5.16 Decoder at d24 (DecoderD24) — in progress
Target ≥ 2× NativeDecoder's 21.8K tok/s on the fixed-length d24 benchmark, accuracy within noise.

### 5.17 Earlier executions (summary)
- **Metrics** (MetricsValidation): edit similarity by difflib as cceval (rapidfuzz, thefuzz and torchmetrics disagree
  on 46–71% of real strings); bpb bucket sums kept manual (torchmetrics SumMetric differs).
- **TokEval** (TokEvalDeps): needs tree-sitter-language-pack 1.22.1 (0.13.0 changes 35/135 values).
- **MultiPL-E scoring** (EvalPipeline): 64 concurrent `main.py` × 63 threads oversubscribed g++ and scored compile
  timeouts as SyntaxError; `--max-workers 1` per file: HumanEval C++ OK 70 → 175 of 3,220, pass@1 0.022 → 0.054;
  FIM scoring through `check_correctness` in a fork pool 211 → 33 s, 0 flips (`9d61a1b`).
- **Reference validation** (ReferenceModelEval): deepseek-coder-1.3b-base through DeepSeek's harness HumanEval 0.305
  (paper 0.348; transformers 5 drops BOS), MBPP 0.466 (paper 0.462); our EvalPlus scorer agrees on 163/164 problems.
- **d16 reference** (RefModelTrain): code bpb 0.439, HumanEval+ 0.061, MBPP+ 0.071/0.050 at 20 tokens/param.
