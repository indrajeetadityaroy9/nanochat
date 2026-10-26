# Architecture and GPU implementation

The base model (`model/modeling.py`), its fixed dimensions (`model/config.py`), the training ops (`model/ops/`), their
CUDA kernels (`model/kernels/`) and the native build (`native.py`). The code is the specification; each section states
what it does, the basis for each choice and its measured effect. Markers: [arithmetic] derived from code numbers,
[inference] reasoning not measured, [estimate], [unverified] a figure without a record or a paper value its text lacks.

**Code**

| File | Section |
|---|---|
| `model/modeling.py` | §1 (components, `init_weights`, counts), §2, §4 |
| `model/config.py` | §1 (`VOCAB_SIZE`, `HEAD_DIM`, `ROPE_THETA`), §2 |
| `model/ops/linear.py` | §3 (projection and MLP GEMMs, post-attention norm, `CONFIGS`), §4 |
| `model/ops/flash_attn.py` | §3 (the `attention` op, FA4 subclasses), §4 |
| `model/ops/cross_entropy.py` | §3 (`lm_loss`, `lm_losses`, the fused cap), §4 |
| `model/ops/qk_prologue.py`, `model/kernels/qk_prologue.cu`, `model/kernels/qk_row.cuh` | §3, §4 |
| `model/ops/ve_gate.py`, `model/kernels/ve_gate.cu` | §3, §4 |
| `model/kernels/fa4_rows.cu` | §3 |
| `model/kernels/common.cuh` | §3 (shared vector access, fixed-order warp sum, occupancy) |
| `model/ops/__init__.py` | (empty package marker) |
| `native.py` | §3 (build and loading of the kernels and of the Rust crates) |

Names follow Hugging Face's (`ModelConfig`, `num_hidden_layers`, `hidden_size`, `CausalLM`, `DecoderLayer`,
`self_attn`, `q/k/v/o_proj`, `up/down_proj`); the decoder (`model/kernels/decode.cu`) is `docs/evaluation.md` §6.

## 1. Model

`CausalLM` is karpathy/nanochat 92d63d4 `nanochat/gpt.py` with the wins of its experiment log (`dev/LOG.md`, "LOG")
and Qwen3's QK-Norm (arXiv 2505.09388 §2) with learnable per-dimension weights (transformers `modeling_qwen3.py`).
`CausalLM.hidden`: x0 = norm(bf16(E[ids])); before layer i, x ← λr_i·x + λ0_i·x0; `DecoderLayer`:
h = x + o_proj(Attn(norm(x))), x = h + down(relu(up(norm(h)))²); then the final norm, `lm_head` and the capped cross
entropy (`CausalLM.forward`).

| Component | Definition (code) |
|---|---|
| Embedding | token table 32,768 × C (`VOCAB_SIZE` = 2^15: 256 bytes, 32,508 merges, BOS and 3 FIM sentinels, `docs/corpus.md` §5), RMS-normed, no positional embedding |
| Residual | per layer x = `resid_lambdas[i]`·x + `x0_lambdas[i]`·x0, x0 the normed embedding (LOG 2026-01-11) |
| Norms, linears | gain-free RMS norm, eps = FLT_EPSILON, statistics in fp32 (`norm`); bias-free linears; untied `lm_head` |
| Attention | MHA, H = C / 128 heads of `HEAD_DIM` = 128; q and k: per-head RMS norm, × a learned per-dim gain shared by the heads (`q_norm_weight`, `k_norm_weight`), then RoPE over the full head, rotation by −θ, θ = `ROPE_THETA` = 10^6 (92d63d4 `gpt.py:57-65`); causal within each attention segment, one per packed piece of a document (FA4 varlen over `cu_seqlens`, `docs/corpus.md` §8–§9; ProLong, arXiv 2410.02660 App. B.2) |
| Windows | `layer_windows`: layers i ≡ 3 (mod 4) and the last attend the whole context, the others the 512 keys before the query (513 with itself); d24: 6 full, 18 windowed |
| Value embeddings | `has_ve`: alternating layers, the last included (12 of 24): v ← v + 2σ(n W_g^T)_head · bf16(E_i[ids]), n the normed stream, W_g (H × C) `ve_gate`, E_i a per-layer V × C table |
| MLP | ReLU², hidden 4C |
| Output | z = x W^T capped c·tanh(z/c), c = `ModelConfig.final_logit_softcapping` (`run.py` `RunConfig`: 15.0; `None` uncapped), in the training loss, the evaluation losses and the decoder's sampling; no z-loss |
| Precision | fp32 parameters and gradients; bf16 activations and GEMM operands (bf16 weight copies `weight_bf16`, `qkv_bf16`, refreshed after every update by `refresh_gemm_weights`); fp32 accumulation (§4) |

**Shape rule** (`model_config`, 92d63d4 `scripts/base_train.py:51-52, 131-134`): C = 30 × `num_hidden_layers` rounded
up to whole 128-wide heads; 30 = MobileLLM-350M's 960 / 32 (arXiv 2402.14905 Table 9). d24: 720 → 768.
`max_position_embeddings` is the stage's context; `set_stage` recomputes the windows and the rotary tables (computed in
fp64, rounded once to fp32: `init_buffers`), the weights unchanged.

**Initialization** (`init_weights`): q/k/v_proj, up_proj, the token table and the value-embedding tables uniform on
±√(3/C) (std C^−½, "Efficient BackProp" §4.6); zero: o_proj and down_proj (each layer starts as the identity,
modded-nanogpt record #5), `lm_head` (record #8), the gates' W_g (gate 1), `x0_lambdas`; one: `resid_lambdas` and the
QK-norm weights. Each random tensor draws from a generator seeded by seed·2^32 + crc32(its name).

**Counts** (`num_non_embedding_params`: decoder layers + `lm_head`, what the token budget scales with,
`docs/training.md` §2; `num_matmul_params`: every 2-D weight but the tables, PaLM's N; M:
`non_embedding_flops_per_token`, DeepSeek LLM's 72·n_layer·C² + 12·n_layer·C·l_seq, arXiv 2401.02954 §3.2 Eq. 2,
windowed layers counted as full; nominal: `metrics.model_flops_per_token`, 6N + 12·C·Σ_layers min(window, context),
defined in `docs/evaluation.md` §5). Recomputed on CPU from `model/modeling.py` on the meta device [arithmetic]:

| Model | Layers | C | Heads | VE tables | Non-embedding | Matmul N | All | M at 8K (32K) | Nominal at 8K (32K) |
|---|---|---|---|---|---|---|---|---|---|
| d8 | 8 | 256 | 2 | 4 | 14,684,160 | 14,682,112 | 56,627,216 | 239,075,328 (843,055,104) | 147,861,504 (298,856,448) |
| d16 | 16 | 512 | 4 | 8 | 67,129,344 | 67,125,248 | 218,124,320 | 1,107,296,256 (3,523,215,360) | 641,826,816 (1,245,806,592) |
| d24 (base) | 24 | 768 | 6 | 12 | 195,096,576 | 195,090,432 | 522,252,336 | 2,831,155,200 (8,266,973,184) | 1,708,462,080 (3,067,416,576) |

At d24 the 6,144 non-embedding parameters outside N are the QK-norm gains; "all" adds the token table and 12
value-embedding tables (25,165,824 each) and the 48 lambdas [arithmetic].

## 2. Design basis

**Lineage.** Taken from 92d63d4: the block layout, value-embedding layers, ReLU² MLP, SSSL pattern and the cap at 15.
Not taken: its smear and backout, its QK order (RoPE, then norm, then ×1.2), its short window (a quarter of the
context rounded up to 128), its RoPE base (10^5) and its init. The gate is nanochat e85db6b's (2·sigmoid, zero init)
read over the full width rather than its first 32 channels (92d63d4 reads 12 channels with 3·sigmoid); the full-width
read is not ablated.

- **Value embeddings, kept** (from the literature): per-layer token-indexed tables are modded-nanogpt record #14's and
  MoVE's LaVE baseline, V = V + g ⊙ M_t with g = 2σ(x W_g) (arXiv 2601.22887 §4.1), on half the layers; LaVE ×1 gains
  0.016, 0.010 and 0.010 validation bpb at D12, D20 and D32 (§4.2 Table 1). nanochat found alternating layers best
  and the gate helpful (LOG 2026-01-17). Not ResFormer's first-layer value reuse, a different mechanism. None of the
  reference code models has them.
- **Learnable QK-norm gains, kept** (not ablated): per-head qk-layernorm with head-dim learnable parameters is arXiv
  2309.14322's default, slightly better than across the model dimension (§3.2.5, Fig. E.8); nGPT finds fixed scales
  "only a slight degradation" (arXiv 2410.01131 §3.3; App. A.9: "negligible degradation (≤0.3%) in validation loss";
  Table 5's 0.11% [unverified: table body not in the extracted text]). Resolving a ~0.0006 bpb effect would need ~250
  seeds per arm against the 2-seed margin of `docs/experiments.md` §1 [estimate]. The gains train under AdamW without
  decay (`docs/training.md` §3).
- **Output-logit cap c = 15, adopted on the literature:** QK norm bounds the attention logits, not these: it "does not
  solve the problem of logit shift in the final softmax" (Chameleon, arXiv 2405.09818 §2.3). Output-logit divergence
  occurs "in models with no weight decay regardless of scale … Weight decay also mitigates this instability for the
  larger models we test" (arXiv 2309.14322 §3.1.2); `lm_head` is weight-decayed (`docs/training.md` §3) and no
  `peak_lr` run diverged up to ×8 (`docs/training.md` §5), so no z-loss. Effect sizes: Gemma 2 caps the final layer
  (arXiv 2408.00118 §2); modded-nanogpt record #9 added a cap at 30 (val loss 3.2844 → 3.2791, one run each) and
  record #18 lowered it to 15 (steps to the 3.28 target 1,490 → 1,390; 80 runs, p = 0.0001); nanochat at d24 found
  caps from 5 to 30 about equal except 5 (much worse), 20 best by ~1e-3 val loss (LOG 2026-03-02), while 92d63d4
  `gpt.py:511-515` keeps 15. Its cost: 0.58% of the 8K step (§3). The designed ablation and its detection power:
  `docs/experiments.md` §3.
- **RoPE base 10^6 from the first step:** above the 32K context's lower bound of arXiv 2405.14591 (Table 2: 6.4e5
  [unverified: caption only in the extracted text]); large from the start as Llama 3 (500,000, arXiv 2407.21783 §3.2);
  base 10k against 500k at 8K changes Gemma 2B's perplexity 4.4627 → 4.4485 (Wiki) and 6.4429 → 6.4593 (FlanV2), under
  0.33% and of mixed sign (arXiv 2410.06205 Table 2). Qwen2.5-Coder instead raises 10^4 to 10^6 at its repository
  stage (arXiv 2409.12186 §3.2.2).
- **ReLU², not SwiGLU:** nanochat measured SwiGLU at matched parameters and FLOPs worse at d12 and d24 on steps,
  wall clock and FLOPs (LOG 2026-02-05); a gated MLP would also need a gated decode MLP (`model/kernels/decode.cu`
  hard-codes relu², `EPI_RELU2`).
- **SSSL windows of 512:** nanochat's SSSL pattern with the last layer full (LOG 2026-01-11); Gemma 3 finds the
  local:global ratio of minimal impact up to 7:1 and the window reducible "significantly without impacting
  perplexity" (arXiv 2503.19786 §5.2, Fig. 3–4; down to 512 [unverified: figure only]). StarCoder2's long-context
  stage uses a 4,096-token window in a 16,384-token context (arXiv 2402.19173 §6.3).
- **Shape:** MobileLLM's deep-and-thin ratio (arXiv 2402.14905). A deeper, thinner alternative at about d24's
  non-embedding parameters, 35 × 640 (193,070,080), costs M 3,234,201,600 per token at 8K (+14.2%) and nominal
  1,826,833,920 (+6.9%) [arithmetic], so an equal-token comparison gives it more compute; its 5 heads are not among
  the head counts `ve_gate.cu` compiles (2, 4, 6).
- **head_dim 128:** compiled into the kernels (`HEAD_DIM`, `native.py`); nanochat found 128 better than 64 (LOG
  2026-01-17).
- **MHA, not grouped-query attention:** GQA's benefit is KV-cache memory; its d16 screen is `docs/experiments.md` §3.
  The decoder's e4m3 cache (`docs/evaluation.md` §6: 38,016 bytes per position at d24) is 44.0 MB for 1,158 positions
  (a 391-token prompt plus 768 new tokens [unverified]) and 45.1 GB for `decode.cu`'s 1,024 slots (`MAX_BATCH`)
  [arithmetic], which the GPU holds.
- **Not adopted** (Block Attention Residuals, the partial key offset, multi-token prediction): `docs/experiments.md` §3.

**Against the reference code models** (training-side rows: `docs/training.md` §1, §3; data and FIM: `docs/corpus.md`
§7–§8):

| Choice | The base model | DeepSeek-Coder / Qwen2.5-Coder / StarCoder2 / Code Llama |
|---|---|---|
| Attention | MHA; SSSL, 3 of 4 layers a 512-key window | full attention (DeepSeek-Coder, Qwen2.5-Coder, Code Llama) [inference: no window in their texts]; StarCoder2 a 4,096-token window in its 16,384-token long-context stage (arXiv 2402.19173 §6.3) |
| MLP | ReLU², hidden 4C | SwiGLU (DeepSeek-Coder, arXiv 2401.14196 Table 2); Qwen2.5-Coder and Code Llama SwiGLU, StarCoder2 GELU [unverified: not in the papers' extracted text] |
| Value embeddings | alternating layers, gated | none |
| RoPE base | 10^6 in both stages | Qwen2.5-Coder 10^4 → 10^6 at 32K (arXiv 2409.12186 §3.2.2); Code Llama 10^4 → 10^6 for its 16,384-token long-context fine-tuning (arXiv 2308.12950 §2.4) |
| Output logits | capped, c = 15 | uncapped [unverified] |

What those models ablate: Code Llama its long-context RoPE base period (arXiv 2308.12950 App. G.4, Table 18),
DeepSeek-Coder the FIM rate (arXiv 2401.14196 Fig. 3); none ablates gains, residual scalars or attention windows
[inference: from their texts].

## 3. GPU implementation

**Build** (`native.py`). `cuda_library(name)` builds `model/kernels/<name>.cu` with one nvcc command for the device's
architecture-specific target (sm_90a: wgmma, setmaxnreg) and `-DHEAD_DIM=128`, into `NATIVE/<name>-sm_90a-<sha256 of
the command, the file and every header>.so` (written to a temporary file and renamed, so concurrent first builds never
share a file), loaded by ctypes once per process. `crate_module` does the same for the Rust crates through cargo
(`corpus/crate`, `docs/corpus.md` §6; `inference/syntax`, `docs/evaluation.md` §7). The kernels follow ThunderKittens'
layout: shared headers (`common.cuh`: 16-byte vector loads and stores, a fixed-order butterfly warp sum, resident-block
counts for persistent grids; `qk_row.cuh`: the q/k row math (RMS norm, gain, RoPE) shared by `qk_prologue.cu` and
the decoder's `decode.cu`) and one self-contained file per kernel with its C entry points. Each op is a
`torch.library` custom op with a fake implementation, so `torch.compile(fullgraph=True)` traces one graph per stage
(`train/loop.py`).

| Op | Implementation |
|---|---|
| q/k/v projection | `linear.py` `qkv_proj`: one (M, 3C) QuACK GEMM against the stacked bf16 [Wq; Wk; Wv]; q, k, v its column blocks, read by attention as strided views; dX one GEMM with K = 3C; dW three cuBLAS `addmm` (fp32 out, β = 1) straight into each weight's `.grad` |
| Attention | `flash_attn.py` `attention`: one op holding the q/k prologue, the value-embedding mix, FlashAttention-4 (CuTe DSL) varlen over the micro-batch's real segments only (`segmented`) and `o_proj`; its backward writes every q/k/v gradient into one (B, T, 3C) buffer. Window layers: `PersistentForwardSm90` with `PersistentVarlenScheduler` (one CTA per SM, the output tile staged apart from Q) and `BackwardSm90Bf16dQ` (dQ reduce-added in bf16 by `cp.reduce.async.bulk` into a zeroed bf16 dq: no fp32 accumulator, fp32 zero fill or postprocess; masks only on boundary tiles). Full layers (`fa4_window_left` → None when the window spans T): FA4's causal kernels, fp32 dQ |
| o_proj, post-attention norm | `linear.py` `residual_rms`: h = residual + O W_o^T and each row's squared sum in the GEMM epilogue (QuACK `sq_reduce_mod`, `rms_final_reduce`), so rstd = (mean(h²) + eps)^−½ without a norm pass; backward `residual_rms_bwd`: dO with FA4's dpsum = rowsum(dO·O) per head summed in the dX epilogue (`dx_dpsum`, `DPSUM_CONFIG`: one 128-column tile per head) |
| FA4 row statistics | `fa4_rows.cu`: dpsum and lse·log2 e at FA4's padded rows (+inf lse, 0 dpsum on padding) in place of FA4's backward preprocess (`precomputed_dpsum`), which read O and dO again |
| MLP | `linear.py` `relu2_mlp`: the norm as a row scale, since norm(h)Wᵀ = diag(rstd)·hWᵀ for a gain-free norm: rstd and relu² in the up projection's epilogue (`gemm_norm_act`, pre-activation and activation both stored for the backward), the residual as the down projection's C (`gemm_add`); backward: relu²'s derivative × rstd and the norm's row sum in one epilogue (`relu2_dact_rstd`), the norm's backward in the up projection's dX epilogue (`norm_bwd_apply`); the normed stream is never stored |
| Tiles | `CONFIGS`: a fixed QuACK SM90 configuration per call and width (C 256, 512, 768) at M = 98,304, each the lowest mean of three runs of QuACK's own autotuning benchmark; QuACK's `default_config` elsewhere |
| q/k prologue | `qk_prologue.cu`: norm → gain → RoPE in fp32, one bf16 rounding per output, one pass per direction; forward one thread group per position, its gains and cos/sin loaded once for the H heads, two heads in flight; gain gradient without atomics (persistent blocks, per-block partials summed in block order) |
| Value-embedding gate and mix | `ve_gate.cu` (`ve_gate.py`): one warp per row, the gate logits in fp32, the mix rounded once; backward in token order (one `torch.sort` of `input_ids` per micro-batch, shared by the layers through CSE), 16 positions per warp, a run of equal tokens summed in registers and added into the table's `.grad` with one 16-byte atomic per lane and chunk (no zeroed gradient, no autograd add); the gate weight's gradient one cuBLAS GEMM from bf16 dz; compiled for H = 2, 4, 6 |
| Cross entropy, training | `cross_entropy.py` `lm_loss`: QuACK's scaled-exp linear cross entropy per `LCE_CHUNK` = 8,192 rows: the logits GEMM's epilogue (N tile 128, pingpong) stores bf16 rescaled exponentials with each row's tile maxima, sums and the label's fp32 logit; `lce_glue` forms the loss from the fp32 accumulators; dX (bf16) and dW (fp32, accumulated across chunks) GEMMs rescale on the A-operand load. No logits tensor and no softmax pass. The cap in the same pipeline: tanh on the fp32 accumulator, the tile maximum capped likewise (tanh is increasing), 1 − tanh² folded into the stored exponentials, the label's entry fixed by one Triton kernel (`cap_label_fix`) |
| Cross entropy, evaluation | `lm_losses`: uncapped, QuACK's `lse_target_epi` (logsumexp and label logit from the fp32 accumulators, no logits stored); capped, per 8,192 rows (`CHUNK`) an fp32-output GEMM and the capped logsumexp in one Inductor-fused kernel (`capped_losses`) |
| Embedding | `F.embedding` of the token table, the bf16 cast fused by Inductor; PyTorch's backward accumulates rows with atomics |
| Optimizer | Muon's Newton–Schulz batched over same-shape matrices and NorMuon: `docs/training.md` §3 |

**Landed changes** (d24 tok/s, each against its parent commit in one exclusive A/B: GH200 at 700 W, SM clock locked,
`docs/training.md` §7; 25 steps after 8 warm-up, median, an arm run twice by its mean; 32K arms of `6299e84` 12 after
6). Rates of different rows come from different measurement windows and do not chain. Loss checks: 200 steps against
the d24 seed spread (seed 0 vs 1, last-20 mean Δ 0.035), or 10 steps at the base lr multiplier ×2.797 against the
parent tree's run-to-run spread (§4).

| Change | Commit | 8K tok/s | 32K tok/s | Memory, other | Numerics |
|---|---|---|---|---|---|
| FA4 on the real segments (the loader had padded `cu_seqlens` to 2,796 segments for ~94 real) + Muon's Newton–Schulz batched by shape | `a81410c` | step 3,709 → 3,511 ms (−5.3%); d16 266 → 239.5 ms (−10%) | | | losses bitwise; gradients within run-to-run spread; momentum bitwise, parameters at bf16 GEMM rounding |
| Fused value-embedding gate and mix | `7ad594f` | step time −0.5% | | | ~15% of elements one bf16 ulp from the PyTorch expressions (fp32 reduction order) |
| q/k prologue per position | `62b50c2` | +0.3%; forward 250.3 → 183.6 µs per call | | | 200-step Δ +0.0035 |
| Post-attention norm in GEMM epilogues | `373b549` | 338,039 → 341,919 (+1.15%) | +0.5% | peak 64.5 → 61.1 GiB; GPU time 4,355 → 4,296 ms per step [unverified] | 200-step Δ −0.0071; y vs fp64 2.6e-3 (bf16 path 3.2e-3) |
| Token-ordered value-embedding backward | `eaf14d0` | 344,768 → 350,185 (+1.57%) | 277,238 → 280,201 (+1.07%) | ve_gate 161.9 → 92.6 ms per step; peak unchanged | 200-step Δ +0.0095 |
| dW accumulated into the fp32 `.grad` by cuBLAS `addmm` (β = 1, Megatron-LM's gradient-accumulation fusion) + one bf16 copy per weight per update | `f499be7` | 344,925 → 347,634 (+0.79%) | 275,613 → 277,383 (+0.64%) | peak 61.6 → 62.0 GiB | neutral |
| FA4 window layers: persistent forward, bf16 dQ | `0e627ff` | 344,967 → 348,218 (+0.94%; forward alone +0.32%) | 276,538 → 278,524 (+0.72%) | | window dQ vs fp32 accumulation norm rel 2.45e-3; 200-step Δ +0.009 |
| MLP activation saved by the forward | `6299e84` | 356,259 → 361,324 (+1.42%) | 282,256 → 285,118 (+1.01%) | peak 61.8 → 75.3 GiB (32K 61.7 → 75.2) | not bitwise (§4); 200-step at ×2.797 last-100 Δ −0.0027 (same-seed noise 0.005) |
| Fused q/k/v projection, one attention op | `70025d8` | 356,431 → 359,180 (+0.77%) | 283,357 → 285,776 (+0.85%) | peak +1.7 GiB warm (75.1 → 76.8) | out, dk, dv, k-gain and value-embedding gradients bitwise; dq, q-gain 1e-6–2e-5; 10 steps bitwise through step 3, then max Δ 1.6e-5 (parent against itself 1.3e-5) |
| Scaled-exp linear cross entropy | `067e60b` | 362,988 → 365,466 (+0.68%) | 286,619 → 287,476 (+0.30%) | peak 76.99 → 71.44 GiB; 26.6–27.1 → 24.2 ms per micro-batch | closer to fp64 (§4); 10 steps within spread (largest 6.4e-4) |
| FA4's dpsum from o_proj's dX epilogue, `fa4_rows.cu` | `db1e90e` | 363,087 → 364,399 (+0.36%) | 286,588 → 287,097 (+0.18%) | 165.1 + 87.4 → 172.2 + 7.5 µs per call | dpsum vs fp64 4.70e-8 (FA4's 5.17e-8); dv bitwise; 10 steps within 1.8e-5 |
| Fixed QuACK tiles (`CONFIGS`) | `fb86efc` | 362,972 → 364,278 (+0.36%) | 286,555 → 287,482 (+0.32%) | peak unchanged; cold first step 119 → 35 s | every op output bitwise at C 768 and 512 |
| Output-logit cap fused into the linear cross entropy | `e24333d` | cost +1.56 ms per micro-batch (25.30 → 26.86 ms), 0.58% of the 8K step | not measured | op peak 0.80 GiB capped or not; 10-step peak 82.67 → 71.11 GiB, a cold start 71.11 | §4; 10 steps within max(spread, 1e-3), largest 8.0e-4 |

Notes on the rows:
- **Fixed tiles.** QuACK's autotuner (`tuned=True`) timed every candidate at a shape's first call, so the choice was a
  per-process timing result: two cold d24 runs picked different configurations for 2 of 9 GEMM keys, and the shared
  cache's q/k/v forward pick was 6.6% slower than the best. QuACK's `default_config` would cost 1.1% / 5.4% / 26% of
  these GEMMs' time at d24 / d16 / d8; a single benchmark run's winner is up to 2.1% slower than the three-run best
  (`linear.py` `CONFIGS`). With `e24333d` no training GEMM autotunes, and a cold start peaks at the warm 71.11 GiB.
- **Norm in epilogues.** The rstd-scaled relu² derivative epilogue costs ~200 µs per layer and micro-batch over QuACK's
  plain `gemm_dact` (~72 ms per step); the Inductor norm and mix kernels that remain (norm before attention, the
  residual mix) run at ~3.6 TB/s.
- **Linear cross entropy.** On a real d24 micro-batch, chunks of 8,192 to 65,536 rows time alike (24.1–24.4 ms per
  call) while the op's peak grows (0.82 → 4.59 GiB); a 98,304-row chunk faults in QuACK 0.6.5; N tile 128 with pingpong
  24.2 ms against 256 without 25.5 (`cross_entropy.py` `LCE_CHUNK`, `TILE_N1`).
- **The cap.** The removed capped path (bf16 capped logits, a logits-sized derivative buffer, a cross-entropy pass, an
  autotuned dX GEMM) cost +9.4 ms and +6 GiB over uncapped (28.04 → 37.44 ms, 6.14 → 12.14 GiB). Over the run, 0.58%
  of the 8K stage's 73.8 h is ~26 min [arithmetic, `docs/training.md` §7]; the cap's 32K cost is not measured.
- **dpsum.** `DPSUM_CONFIG` is the fastest QuACK SM90 configuration with tile_n = 128 on the d24 shape (172.2 µs;
  cluster (2, 1) 172.8, tile_m 192 212.4, cooperative 256 268.5); `fa4_rows` takes 7.5 µs.
- **Value-embedding backward.** RUN = 16 positions per warp (isolated backward 222 / 203 / 251 µs at RUN 8 / 16 / 32),
  2 blocks of 256 threads per SM (`ve_gate.cu`).
- **Window dQ in bf16.** A window dQ element sums at most 5 bf16 partials (one per 128-key block of the 512-key
  window); full layers would sum T/128 (64 at 8K, 256 at 32K) and keep FA4's fp32 accumulator and FA4's own launch
  and tile shapes (the persistent and tile variants measured for them: `docs/experiments.md` §5).
- **Design sources.** The gate kernel and its token-ordered backward follow TileKernels' Engram gate and
  weight-gradient reduce; the norm in GEMM epilogues follows TileKernels' mHC norm kernels and DeepGEMM's in-GEMM
  pre-norm statistics; dW into `.grad` follows DeepGEMM's D = C + A·B and Megatron-LM's gradient-accumulation fusion.

On the GH200 the GEMMs run at the 700 W cap's ceiling; the step profile, the latest rates and the memory peak are
`docs/training.md` §7. Rejected or not pursued kernels and systems changes (FP8, hand
attention kernels, QK-norm + RoPE in the projection epilogue, the residual mix in the down projection, window-attention
tile skipping, split-K and DeepGEMM dW, a global cuBLASLt workspace, a single 3C × C q/k/v dW, cuDNN attention,
PyTorch's deterministic mode, max-autotune, Inductor's PDL, Liger and Cut Cross-Entropy): `docs/experiments.md` §5.

## 4. Numerics and verification

**Precision.** Parameters, gradients and optimizer state are fp32; activations and GEMM operands bf16 with fp32
accumulation and one rounding per output; fp32 matmuls stay fp32 (TF32 override off, `docs/training.md` §8). The q/k
prologue and the value-embedding mix compute in fp32 and round once; the rotary tables are computed in fp64. The
training loss comes from the logits GEMM's fp32 accumulators and its gradients from the bf16 rescaled exponentials (no
bf16 logits); dW is fp32 throughout.
Two choices are deliberately not bitwise to their predecessors: the saved MLP activation (the down projection's dW uses
relu² of the fp32 pre-activation rounded once, not relu² of the bf16 pre-activation) and the gate kernel's fp32
reduction order.

**Non-bitwise atomics.** FA4's backward (fp32 dQ atomics in full layers, bf16 reduce-adds in window layers), PyTorch's
token-table backward and the value-embedding table's 16-byte atomics accumulate in no fixed order, so gradients are not
bitwise across runs and step losses differ between processes from step 3 (two of five 4-step runs at `fb86efc`, the GEMM
configurations identical). Everything else is fixed by construction: GEMM tiles (`CONFIGS`, `DPSUM_CONFIG`,
`cross_entropy.py`'s explicit tiles, `lm_losses` at QuACK's untuned default), the QK-gain gradient's summation order
(`qkp_backward_blocks`: half the device's resident blocks, partials summed in block order), the warp sums
(`common.cuh` `warp_sum`) and Inductor's coordinate-descent tuning off (`docs/training.md` §8). GEMM configurations do
not change the bits of plain GEMMs (split-K 1, checked across the autotuner's picks); reduce epilogues' partial sums
follow tile_n. PyTorch's deterministic mode is rejected for its cost (`docs/experiments.md` §5).

**Op checks against fp64.** Each landing since `70025d8` carries an op check (against an fp64 reference, or bitwise
against its parent's op) and a 10-step run at ×2.797 against the parent tree's run-to-run spread (the landing bar:
`docs/experiments.md` §1); results in the §3 table. The linear cross entropy and the fused cap against fp64 on real
captures at step 200 (relative error, previous path → landed):

| Quantity | `067e60b` uncapped (bf16-logit path → scaled-exp) | `e24333d` capped (removed capped path → fused) |
|---|---|---|
| Mean loss | 2.8e-5 → 3.3e-8 | 1.7e-4 → 2.3e-8 |
| Per-position loss | 3.3e-3 → 5.8e-7 | 4.2e-3 → 9.5e-6 |
| Logsumexp | | 7.8e-4 → 2.5e-6 |
| dx | 5.6e-3 → 2.6e-3 | 5.5e-3 → 3.1e-3 |
| dW | 2.2e-3 → 1.0e-3 | 2.2e-3 → 8.7e-4 |

The fused cap's 10-step difference (8.0e-4 at step 9) exceeds the old capped path's run-to-run ≤ 7e-6, as the uncapped
landing's 6.4e-4 did: the more accurate gradient diverging from the less accurate one [inference]; the uncapped path is
bitwise `067e60b`'s.

**The model against fp64.** A plain-PyTorch restatement of the model in fp64 (4 layers, C 768, 2 × 2,048 tokens; its
fp32 run 2e-6 from it): per-parameter gradient relative error median 1.17e-2, max 1.96e-2 at `70025d8`, against
1.18e-2 / 1.95e-2 at `646a986` (the model code of its parent `cf0aff5`), the two trees 3.75e-3 apart: bf16 rounding,
not a defect.

## References

| arXiv | Title |
|---|---|
| 2308.12950 | Code Llama: Open Foundation Models for Code |
| 2309.14322 | Small-scale proxies for large-scale Transformer training instabilities |
| 2401.02954 | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism |
| 2401.14196 | DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence |
| 2402.14905 | MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases |
| 2402.19173 | StarCoder 2 and The Stack v2: The Next Generation |
| 2405.09818 | Chameleon: Mixed-Modal Early-Fusion Foundation Models |
| 2405.14591 | Base of RoPE Bounds Context Length |
| 2407.21783 | The Llama 3 Herd of Models |
| 2408.00118 | Gemma 2: Improving Open Language Models at a Practical Size |
| 2409.12186 | Qwen2.5-Coder Technical Report |
| 2410.01131 | nGPT: Normalized Transformer with Representation Learning on the Hypersphere |
| 2410.02660 | How to Train Long-Context Language Models (Effectively) |
| 2410.06205 | Round and Round We Go! What makes Rotary Positional Encodings useful? |
| 2503.19786 | Gemma 3 Technical Report |
| 2505.09388 | Qwen3 Technical Report |
| 2601.22887 | MoVE: Mixture of Value Embeddings -- A New Axis for Scaling Parametric Memory in Autoregressive Models |

Repositories, at the commit, file or record cited: `karpathy/nanochat` (92d63d4 `nanochat/gpt.py`,
`scripts/base_train.py`; e85db6b `nanochat/gpt.py`; `dev/LOG.md` 2026-01-11, 2026-01-17, 2026-02-05, 2026-03-02);
`KellerJordan/modded-nanogpt` (records #5, #8, #9, #14, #18); `Dao-AILab/flash-attention` (FlashAttention-4,
`flash_attn.cute`); `Dao-AILab/quack` 0.6.5; `deepseek-ai/DeepGEMM`, `deepseek-ai/TileKernels` (design sources);
`HazyResearch/ThunderKittens` (kernel layout); `NVIDIA/Megatron-LM` (gradient-accumulation fusion);
`huggingface/transformers` (`modeling_qwen3.py`). LeCun et al., "Efficient BackProp" (1998) §4.6.
