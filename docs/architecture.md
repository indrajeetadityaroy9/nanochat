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
| `model/__init__.py`, `model/ops/__init__.py` | (empty package markers) |
| `native.py` | §3 (build and loading of the kernels and of the Rust crates) |

Names follow Hugging Face's (`ModelConfig`, `num_hidden_layers`, `hidden_size`, `CausalLM`, `DecoderLayer`,
`self_attn`, `q/k/v/o_proj`, `up/down_proj`); the decoder (`model/kernels/decode.cu`) is `docs/evaluation.md` §6.

## 1. Model

`CausalLM` is a pre-norm decoder with a final norm before the head (arXiv 2002.04745 §3.1) and Qwen3's QK-Norm
(arXiv 2505.09388 §2) with learnable per-dimension weights (transformers `modeling_qwen3.py`).
`CausalLM.hidden`: x0 = norm(E[ids]); before layer i, x ← λr_i·x + λ0_i·x0; `DecoderLayer`:
h = x + o_proj(Attn(norm(x))), x = h + down(relu(up(norm(h)))²); then the final norm, `lm_head` and the capped cross
entropy (`CausalLM.forward`). The eager code computes x0 = bf16(norm(bf16(E[ids]))) and the mix in bf16; under
`torch.compile` (training, `train/loop.py`; bits-per-byte, `eval/bpb.py`; Inductor output read at d16) x0 is the norm
of the fp32 row, kept in fp32, and each layer's mix is one fp32 fma(λr_i, x, λ0_i·x0) whose bf16 rounding is the
stored stream while the layer's norm reads the unrounded value (§3 Embedding); `decode.cu` computes the same.

| Component | Definition (code) |
|---|---|
| Embedding | token table 32,768 × C (`VOCAB_SIZE` = 2^15: 256 bytes, 32,508 merges, BOS and 3 FIM sentinels, `docs/corpus.md` §5), RMS-normed, no positional embedding |
| Residual | per layer x = `resid_lambdas[i]`·x + `x0_lambdas[i]`·x0, x0 the normed embedding (learned depth weights initialized to the plain residual stack, after DenseFormer, arXiv 2402.02622 §3) |
| Norms, linears | gain-free RMS norm, eps = FLT_EPSILON, statistics in fp32 (`norm`); bias-free linears; untied `lm_head` |
| Attention | MHA, H = C / 128 heads of `HEAD_DIM` = 128; q and k: per-head RMS norm, × a learned per-dim gain shared by the heads (`q_norm_weight`, `k_norm_weight`), then RoPE over the full head (arXiv 2104.09864 §3.2.2), rotation by −θ, θ = `ROPE_THETA` = 10^6; causal within each attention segment, one per packed piece of a document (FA4 varlen over `cu_seqlens`, `docs/corpus.md` §8–§9; document masks improved both long- and short-context results in ProLong's continued long-context training, arXiv 2410.02660 App. B.2) |
| Windows | `layer_windows`: layers i ≡ 3 (mod 4) and the last attend the whole context, the others the 512 keys before the query (513 with itself); d24: 6 full, 18 windowed |
| Value embeddings | `has_ve`: alternating layers, the last included (12 of 24): v ← v + 2σ(n W_g^T)_head · bf16(E_i[ids]), n the normed stream, W_g (H × C) `ve_gate`, E_i a per-layer V × C table |
| MLP | ReLU², hidden 4C |
| Output | z = x W^T capped c·tanh(z/c), c = `ModelConfig.final_logit_softcapping` = `FINAL_LOGIT_SOFTCAPPING` = 15 (`meta_model`), in the training loss, the evaluation losses and the decoder's sampling; no z-loss |
| Precision | fp32 parameters and gradients; bf16 activations (compiled: x0 and the residual mix before its norm in fp32, above) and GEMM operands (bf16 weight copies `weight_bf16`, `qkv_bf16`, refreshed after every update by `refresh_gemm_weights`); fp32 accumulation (§4) |

**Shape rule** (`meta_model`): C = 30 × `NUM_HIDDEN_LAYERS` (24) rounded up to whole 128-wide heads; 30 =
MobileLLM-350M's 960 / 32 (arXiv 2402.14905 Table 9). d24: 720 → 768. The code builds d24 only; the proxies d8 and
d16 (`docs/experiments.md` §1) are the same rule at 8 and 16 layers, run with multi-depth code since removed (GEMM
tiles and kernel instances for their widths). `set_stage` sets `max_position_embeddings` to the stage's context and
recomputes the windows and the rotary tables (fp64, rounded once to fp32: `init_buffers`); weights are unchanged.

**Initialization** (`init_weights`): q/k/v_proj, up_proj, the token table and the value-embedding tables uniform on
±√(3/C) (std C^−½, "Efficient BackProp" §4.6); zero: o_proj and down_proj (each layer starts as the identity) and
`lm_head`, as Fixup's rule (arXiv 1901.09321 §3), the gates' W_g (gate 1), `x0_lambdas`; one: `resid_lambdas` and the
QK-norm weights. Each random tensor draws from a generator seeded by seed·2^32 + crc32(its name).

**Counts** [arithmetic; recomputed on CPU from `model/modeling.py` on the meta device, d8 and d16 by the shape rule].
Non-embedding: `num_non_embedding_params`, decoder layers + `lm_head`, what the token budget scales with
(`docs/training.md` §2). Matmul N: `num_matmul_params`, every 2-D weight but the tables, PaLM's N. M:
`non_embedding_flops_per_token`, DeepSeek LLM's 72·n_layer·C² + 12·n_layer·C·l_seq (arXiv 2401.02954 §3.2 Eq. 2), the
count its batch and lr laws were fit with (C = MD; its Table 3 evaluates M at l_seq 4,096); windowed layers count as
full, on the assumption that a model with near-lossless local windows (arXiv 2503.19786 §5.2) shares its
full-attention counterpart's hyperparameters [inference]. Nominal: `metrics.model_flops_per_token`, 6N + 12·C·
Σ_layers min(window + 1, context) (`docs/evaluation.md` §5).

| Model | Layers | C | Heads | VE tables | Non-embedding | Matmul N | All | M at 8K (32K) | Nominal at 8K (32K) |
|---|---|---|---|---|---|---|---|---|---|
| d8 | 8 | 256 | 2 | 4 | 14,684,160 | 14,682,112 | 56,627,216 | 239,075,328 (843,055,104) | 147,879,936 (298,874,880) |
| d16 | 16 | 512 | 4 | 8 | 67,129,344 | 67,125,248 | 218,124,320 | 1,107,296,256 (3,523,215,360) | 641,900,544 (1,245,880,320) |
| d24 (base) | 24 | 768 | 6 | 12 | 195,096,576 | 195,090,432 | 522,252,336 | 2,831,155,200 (8,266,973,184) | 1,708,627,968 (3,067,582,464) |

At d24 the 6,144 non-embedding parameters outside N are the QK-norm gains; "all" adds the token table and 12
value-embedding tables (25,165,824 each) and the 48 lambdas [arithmetic]. The d24 run records carry the same values
(`metrics.jsonl` `run` record: `num_parameters`, `flops_per_token`, `model_flops_per_token`; `stage` records the 32K
nominal; `train/loop.py`).

## 2. Design basis

- **Value embeddings, kept** (from the literature): per-layer token-indexed tables are MoVE's LaVE baseline,
  V = V + g ⊙ M_t with g = 2σ(x W_g), zero-initialized and reading the full width (arXiv 2601.22887 §4.1), on half
  the layers; LaVE ×1 gains 0.016, 0.010 and 0.010 validation bpb at D12, D20 and D32 (§4.2 Table 1). The alternating
  placement, the gate and its full-width read are not ablated here. None of the reference code models has them.
- **Learnable QK-norm gains, kept** (not ablated): per-head qk-layernorm with head-dim learnable parameters is arXiv
  2309.14322's default, slightly better than across the model dimension (§3.2.5, Fig. E.8). In nGPT, a normalized
  Transformer (0.5B parameters, 1K context, ~52B tokens), a fixed s_qk = 1 costs +0.11% validation loss (arXiv
  2410.01131 App. A.9, Table 5). At the d16 proxy's code bits per byte (0.42724, τ×1 seed 0 of
  `results/muon_wd_timescale_20tpp.jsonl`) +0.11% is ≈ 0.0005, which the shared test's margin z·σ·√(2/n)
  (`docs/experiments.md` §1) reaches at n ≈ 400 seeds per arm [estimate]. The gains train under AdamW without
  decay (`docs/training.md` §3).
- **Output-logit cap c = 15, adopted on the literature:** QK norm bounds the attention logits, not these: it "does not
  solve the problem of logit shift in the final softmax" (Chameleon, arXiv 2405.09818 §2.3). Output-logit divergence
  occurs "in models with no weight decay regardless of scale … Weight decay also mitigates this instability for the
  larger models we test" (arXiv 2309.14322 §3.1.2); `lm_head` is weight-decayed (`docs/training.md` §3) and no
  `peak_lr` run of the earlier recipe, uncapped, diverged up to ×8 (`docs/training.md` §5), so no z-loss. The cap is
  Gemma 2's soft_cap·tanh(logits/soft_cap) (final layer 30, arXiv 2408.00118 §2), after logit clipping C·tanh(u)
  (arXiv 1611.09940 App. A.2, C = 10). c = 15 has no paper and is not ablated here. Its cost: 0.58% of the 8K step
  (§3). The designed ablation and its detection power: `docs/experiments.md` §3.
- **RoPE base 10^6 from the first step:** above the 32K context's lower bound of arXiv 2405.14591 (Table 2: 6.4e5);
  large from the start as Llama 3 (500,000, arXiv 2407.21783 §3.2); base 10k against 500k at 8K changes Gemma 2B's
  perplexity 4.4627 → 4.4485 (Wiki) and 6.4429 → 6.4593 (FlanV2), under 0.33% and of mixed sign (arXiv 2410.06205
  Table 2).
- **ReLU², not SwiGLU:** squared ReLU is Primer's most effective modification (arXiv 2109.08668 §3, Fig. 5); against
  it, SwiGLU beats ReLU at matched parameters and compute (arXiv 2002.05202 Table 1); not ablated here. A gated MLP
  would also need a gated decode MLP (`model/kernels/decode.cu` hard-codes relu², `EPI_RELU2`).
- **SSSL windows of 512:** sliding-window layers (arXiv 2004.05150 §3.1) with every fourth and the last layer full;
  Gemma 3 finds the local:global ratio of minimal impact up to 7:1 and the window reducible "significantly without
  impacting perplexity" (arXiv 2503.19786 §5.2: Fig. 3, 2B and 9B models; Fig. 4, 2B, windows 512–4,096 at 1:1 and
  3:1).
- **Shape:** the shape rule's MobileLLM ratio (§1). A deeper, thinner alternative at about d24's
  non-embedding parameters, 35 × 640 (193,070,080), costs M 3,234,201,600 per token at 8K (+14.2%) and nominal
  1,827,033,600 (+6.9%) [arithmetic], so an equal-token comparison gives it more compute; its 5 heads would need
  their own `ve_gate.cu` instance and GEMM tiles (the code is d24's: H = 6, C = 768).
- **head_dim 128:** compiled into the kernels (`HEAD_DIM`, `native.py`); the head size is set independently of the
  number of heads (arXiv 2002.07028 Theorem 1); 128 itself has no paper.
- **MHA, not grouped-query attention:** GQA's benefit is KV-cache memory; its d16 screen is `docs/experiments.md` §3.
  The decoder's e4m3 cache (`docs/evaluation.md` §6: 38,016 bytes per position at d24) is 44.0 MB for 1,158 positions
  (a slot's `max_length`, len(prompt) + max_new_tokens − 1: a 391-token prompt [unverified] and 768 new tokens) and
  45.1 GB for `decode.cu`'s 1,024 slots (`MAX_BATCH`) [arithmetic], which the GPU holds.
- **Not adopted:** smear and backout (no paper; provenance: `docs/lineage.md`); Block Attention Residuals, the partial
  key offset and multi-token prediction (`docs/experiments.md` §3).

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

**Build** (`native.py`). `cuda_library(name)` builds `model/kernels/<name>.cu` with one nvcc command for the GH200's
architecture-specific target (sm_90a: `decode.cu`'s wgmma) and `-DHEAD_DIM=128` into a shared library named by the
hash of the command, the file and its headers, loaded by ctypes once per process; `crate_module` does the same for
the Rust crates through cargo (`corpus/crate`, `docs/corpus.md` §6; `inference/syntax`, `docs/evaluation.md` §7).
`common.cuh` holds 16-byte vector access, a fixed-order butterfly warp sum and resident-block counts for persistent
grids; `qk_row.cuh` the q/k row math (RMS norm, gain, RoPE) shared by `qk_prologue.cu` and `decode.cu`. Each op is a
`torch.library` custom op with a fake implementation, so `torch.compile(fullgraph=True)` traces one graph per stage
(`train/loop.py`); attention is one op because FA4's own `autograd.Function` is not traceable.

| Op | Implementation |
|---|---|
| q/k/v projection | `linear.py` `qkv_proj`: one (M, 3C) QuACK GEMM against the stacked bf16 [Wq; Wk; Wv]; q, k, v its column blocks, read by attention as strided views; dX one GEMM with K = 3C; dW three cuBLAS `addmm` (fp32 out, β = 1) straight into each weight's `.grad` |
| Attention | `flash_attn.py` `attention`: one op holding the q/k prologue, the value-embedding mix, FlashAttention-4 (CuTe DSL) varlen over the micro-batch's real segments only (`segmented`) and `o_proj`; its backward writes every q/k/v gradient into one (B, T, 3C) buffer. Window layers: `PersistentForwardSm90` with `PersistentVarlenScheduler` (one CTA per SM, the output tile staged apart from Q, since FA4's sO is a view of sQ that a persistent CTA's next Q load would overwrite [inference]) and `BackwardSm90Bf16dQ` (dQ reduce-added in bf16 by `cp.reduce.async.bulk` into a zeroed bf16 dq: no fp32 accumulator, fp32 zero fill or postprocess; masks only on boundary tiles). Full layers (`fa4_window_left` → None when the window spans T, so plain causal attention never runs FA4's local kernel): FA4's causal kernels, fp32 dQ |
| o_proj, post-attention norm | `linear.py` `residual_rms`: h = residual + O W_o^T and each row's squared sum in the GEMM epilogue (QuACK `sq_reduce_mod`, `rms_final_reduce`), so rstd = (mean(h²) + eps)^−½ without a norm pass; backward `residual_rms_bwd`: dO with FA4's dpsum = rowsum(dO·O) per head summed in the dX epilogue (`dx_dpsum`, `DPSUM_CONFIG`: one 128-column tile per head) |
| FA4 row statistics | `fa4_rows.cu`: dpsum and lse·log2 e at FA4's padded rows (+inf lse, 0 dpsum on padding) in place of FA4's backward preprocess (`precomputed_dpsum`), which read O and dO again; one block of m threads per FA4 backward m-block |
| MLP | `linear.py` `relu2_mlp`: the norm as a row scale, since norm(h)Wᵀ = diag(rstd)·hWᵀ for a gain-free norm: rstd and relu² in the up projection's epilogue (`gemm_norm_act`, pre-activation and activation both stored for the backward), the residual as the down projection's C (`gemm_add`); backward: relu²'s derivative × rstd and the norm's row sum in one epilogue (`relu2_dact_rstd`), the norm's backward in the up projection's dX epilogue (`norm_bwd_apply`); the normed stream is never stored |
| Tiles | `CONFIGS`: a fixed QuACK SM90 configuration per call at d24's C = 768 and M = 98,304, each the lowest mean of three runs of QuACK's own autotuning benchmark |
| q/k prologue | `qk_prologue.cu`: norm → gain → RoPE in fp32, one bf16 rounding per output, one pass per direction; forward one thread group per position, its gains and cos/sin loaded once for the H heads, two heads in flight (H = 6 even); gain gradient without atomics (persistent blocks, per-block partials summed in block order) |
| Value-embedding gate and mix | `ve_gate.cu` (`ve_gate.py`): one warp per row, the gate logits in fp32, the mix rounded once; backward in token order (one `torch.sort` of `input_ids` per micro-batch, shared by the layers through CSE), 16 positions per warp, a run of equal tokens summed in registers and added into the table's `.grad` with one 16-byte atomic per 4 fp32 (two per lane and 8-wide chunk; the zeroed table gradient allocated once per step, when `.grad` is None, then accumulated into; no autograd add); the gate weight's gradient one cuBLAS GEMM from bf16 dz; compiled for d24's H = 6 |
| Cross entropy, training | `cross_entropy.py` `lm_loss`: QuACK's scaled-exp linear cross entropy per `CHUNK` = 8,192 rows: the logits GEMM's epilogue (N tile 128, pingpong) stores bf16 rescaled exponentials with each row's tile maxima, sums and the label's fp32 logit; `lce_glue` forms the loss from the fp32 accumulators; dX (bf16) and dW (fp32, accumulated across chunks) GEMMs rescale on the A-operand load, their N tile over C QuACK's rule at C = 768 (`TILE_N2` = 192). No logits tensor and no softmax pass. The cap in the same pipeline: tanh on the fp32 accumulator, the tile maximum capped likewise (tanh is increasing), 1 − tanh² folded into the stored exponentials, the label's entry fixed by one Triton kernel (`cap_label_fix`) |
| Cross entropy, evaluation | `lm_losses`: per `CHUNK` rows an fp32-output GEMM and the capped logsumexp in one Inductor-fused kernel (`capped_losses`); QuACK's logsumexp epilogue has no tanh |
| Embedding | `F.embedding` of the token table; Inductor fuses the gather, the bf16 cast and the norm without rounding in between (`emulate_precision_casts` off), so the compiled x0 is the norm of the fp32 rows, as `decode.cu` `k_norm` computes it (d8, 98,304 random ids, compiled against eager outputs: 114 of 25,165,824 elements differ from eager bf16(norm(E[ids])), 6,312,306 from eager norm(bf16(E[ids]))); the compiled backward is an fp32 `atomic_add` scatter into a zero table (§4) |
| Optimizer | Muon's Newton–Schulz batched over same-shape matrices and NorMuon: `docs/training.md` §3 |

**Landed changes** (d24 tok/s as each change's record states them: an exclusive A/B on the GH200 at 700 W against
the master tree the change landed on, not always its immediate predecessor; the SM clock is recorded as locked for the
FA4-segments/batched-Newton–Schulz and value-embedding-gate changes only (`docs/training.md` §7); mostly 25 timed
steps after 8 warm-up at 8K and 12 after 6 at 32K, median, an arm run twice by its mean; the fp32 `.grad`
accumulation 40 steps, 29 timed; the post-attention norm's 8K rate a 200-step run against an earlier base run).
Rates of different rows come from different measurement windows and do not chain. Loss checks: 200 steps against the
d24 seed spread (seed 0 vs 1, last-20 mean Δ 0.035), or 10 steps at the earlier recipe's base multiplier ×2.797
against the parent tree's run-to-run spread (§4).

| Change | 8K tok/s | 32K tok/s | Memory, other | Numerics |
|---|---|---|---|---|
| FA4 on the real segments (the loader had padded `cu_seqlens` to 2,796 segments for ~94 real) + Muon's Newton–Schulz batched by shape | step 3,709 → 3,511 ms (−5.3%); d16 266 → 239.5 ms (−10%) | | | losses bitwise; gradients within run-to-run spread; momentum bitwise, parameters at bf16 GEMM rounding |
| Fused value-embedding gate and mix | step time −0.5% | | | ~15% of elements one bf16 ulp from the PyTorch expressions (fp32 reduction order) |
| q/k prologue per position | +0.3%; forward 250.3 → 183.6 µs per call | | | 200-step Δ +0.0035 |
| Post-attention norm in GEMM epilogues | 338,039 → 341,919 (+1.15%) | 274,783 → 276,661 (+0.68%) | peak 64.5 → 61.1 GiB; nsys GPU time 4,355 → 4,296 ms per step | 200-step Δ −0.0071; y vs fp64 2.6e-3 (bf16 path 3.2e-3) |
| Token-ordered value-embedding backward | 344,768 → 350,185 (+1.57%) | 277,238 → 280,201 (+1.07%) | ve_gate 161.9 → 92.6 ms per step; peak unchanged | 200-step Δ +0.0095 |
| dW accumulated into the fp32 `.grad` by cuBLAS `addmm` (β = 1) + one bf16 copy per weight per update | 344,925 → 347,634 (+0.79%) | 275,613 → 277,383 (+0.64%) | peak 61.6 → 62.0 GiB | neutral |
| FA4 window layers: persistent forward, bf16 dQ | 344,967 → 348,218 (+0.94%; forward alone +0.32%) | 276,538 → 278,524 (+0.72%) | | window dQ vs fp32 accumulation norm rel 2.45e-3; 200-step Δ +0.009 |
| MLP activation saved by the forward | 356,259 → 361,324 (+1.42%) | 282,256 → 285,118 (+1.01%) | peak 61.8 → 75.3 GiB (32K 61.7 → 75.2) | not bitwise (§4); 200-step at ×2.797 last-100 Δ −0.0027 (same-seed noise 0.005) |
| Fused q/k/v projection, one attention op | 356,431 → 359,180 (+0.77%) | 283,357 → 285,776 (+0.85%) | peak +1.7 GiB warm (75.1 → 76.8) | out, dk, dv, k-gain and value-embedding gradients bitwise; dq, q-gain 1e-6–2e-5; 10 steps bitwise through step 3, then max Δ 1.6e-5 (parent against itself 1.3e-5) |
| Scaled-exp linear cross entropy | 362,988 → 365,466 (+0.68%) | 286,619 → 287,476 (+0.30%) | peak 76.99 → 71.44 GiB; 26.6–27.1 → 24.2 ms per micro-batch | closer to fp64 (§4); 10 steps within spread (largest 6.4e-4) |
| FA4's dpsum from o_proj's dX epilogue, `fa4_rows.cu` | 363,087 → 364,399 (+0.36%) | 286,588 → 287,097 (+0.18%) | 165.1 + 87.4 → 172.2 + 7.5 µs per call | dpsum vs fp64 4.70e-8 (FA4's 5.17e-8); dv bitwise; 10 steps within 1.8e-5 |
| Fixed QuACK tiles (`CONFIGS`) | 362,972 → 364,278 (+0.36%) | 286,555 → 287,482 (+0.32%) | peak unchanged; cold first step 119 → 35 s | every op output bitwise at C 768 and 512 |
| Output-logit cap fused into the linear cross entropy | cost +1.56 ms per micro-batch (25.30 → 26.86 ms), 0.58% of the 8K step | not measured | op peak 0.80 GiB capped or not; 10-step peak 82.67 → 71.11 GiB, a cold start 71.11 | §4; 10 steps within max(spread, 1e-3), largest 8.0e-4 |

Notes on the rows:
- **Fixed tiles.** QuACK's autotuner (`tuned=True`) timed every candidate at a shape's first call, so the choice was a
  per-process timing result: two cold d24 runs picked different configurations for 2 of 9 GEMM keys, and the shared
  cache's q/k/v forward pick was 6.6% slower than the best. QuACK's `default_config` would cost 1.1% / 5.4% / 26% of
  these GEMMs' time at d24 / d16 / d8; a single benchmark run's winner is up to 2.1% slower than the three-run best,
  and a cold first d24 step spent ~80 s autotuning. With the fused cap no training GEMM autotunes, and a cold start
  peaks at the warm 71.11 GiB.
- **Norm in epilogues.** The rstd-scaled relu² derivative epilogue costs ~200 µs per layer and micro-batch over QuACK's
  plain `gemm_dact` (~72 ms per step); the Inductor norm and mix kernels that remain (norm before attention, the
  residual mix) run at ~3.6 TB/s.
- **Linear cross entropy.** On a real d24 micro-batch, chunks of 8,192 to 65,536 rows time alike (24.1–24.4 ms per
  call) while the op's peak grows (0.82 → 4.59 GiB), so `CHUNK` is the smallest; a 98,304-row chunk faults in QuACK
  0.6.5. N tile 128 with pingpong takes 24.2 ms against 256 without 25.5 (`TILE_N1`); QuACK's own rule for this tile
  (`_pick_tile_n1`: 256 at V = 2^15, no pingpong) times +5.8% at C = 768, +3.7% at 512 and within 0.3% at 256
  against 128 with pingpong (QuACK 0.6.5's `scaled_exp_linear_cross_entropy_fwd`, uncapped, 98,304 rows, median of
  20 calls). The strip GEMMs' N tile follows QuACK's rule (192 if C % 192 = 0, else 256): on a 98,304-row micro-batch
  with the cap (GH200, exclusive, median of 5 alternated rounds of 20 calls) the op at C = 512 (d16) takes 19.06 ms
  with 256 against 23.60 ms with a fixed 192 (−19.2%; 192 leaves 3 N tiles, which the dW GEMM's cluster_N = 2 pads to
  4 [inference]), at C = 256 (d8) 15.71 against 15.05 ms (+4.4%; one 256-wide N tile gives the dX GEMM 64 CTAs per
  8,192-row chunk on 132 SMs, 192 gives 128 [inference]); at C = 768 the rule gives 192 (`TILE_N2`). Loss, dX and dW
  are bitwise equal at every width whichever tile runs, and the uncapped op was bitwise QuACK 0.6.5's
  `scaled_exp_linear_cross_entropy_fwd` at the same `tile_n1`.
- **The cap.** The previous capped path (bf16 capped logits, a logits-sized derivative buffer, a cross-entropy pass,
  an autotuned dX GEMM) cost +9.4 ms and +6 GiB over uncapped (28.04 → 37.44 ms, 6.14 → 12.14 GiB). Over the run,
  0.58% of the 8K stage's 74.0 h is ~26 min [arithmetic, `docs/training.md` §7]; the cap's 32K cost is not measured.
- **dpsum.** QuACK's autotuner does not pin tile_n, so `DPSUM_CONFIG` is fixed: the fastest QuACK SM90 configuration
  with tile_n = 128 on the d24 shape (172.2 µs; cluster (2, 1) 172.8, tile_m 192 212.4 / 218.3, cooperative 256
  268.5 / 270.4); `fa4_rows` took 7.5 µs when it landed (256-thread blocks; the current launch is not timed alone).
- **Value-embedding backward.** RUN = 16 positions per warp (isolated backward 222 / 203 / 251 µs at RUN 8 / 16 / 32),
  2 blocks of 256 threads per SM (`ve_gate.cu`).
- **q/k prologue forward.** Two heads' loads in flight per thread group: `k_forward` 250.3 → 183.6 µs per call at d24
  8K against one head per group (`qk_prologue.cu`).
- **Window layers.** FA4's SM90 forward launches one CTA per (segment m-block, head), and a 512-key window's m-block of
  128 queries covers ~5 key blocks, so each CTA's launch, Q load and output store are exposed; the persistent forward
  amortizes them. Full causal layers keep FA4's launch: their m-blocks' work differs up to T / 128-fold, which a static
  stride balances worse than the hardware's CTA scheduling (measured on the d24 shapes). A window dQ element sums at
  most 5 bf16 partials (one per 128-key block of the 512-key window); full layers would sum T/128 (64 at 8K, 256 at
  32K) and keep FA4's fp32 accumulator and FA4's own launch and tile shapes, the fastest timed on the d24 shapes (the
  persistent and tile variants measured for them: `docs/experiments.md` §5).
- **Real segments only.** FA4's varlen forward launches CTAs for every segment it is given (`flash_fwd_sm90.py`
  `SingleTileVarlenScheduler`), so padding segments cost time without changing outputs, as over-padding `cu_seqlens`
  did in McCormick, "Optimizing Training with FlashAttention varlen" (2026); the op passes the first `segments` + 1
  offsets (`flash_attn.py` `segmented`).
- **Design sources.** The gate kernel follows TileKernels' Engram gate kernels, whose weight gradient (per-persistent-
  block partials summed by `engram_grad_w_reduce`) has the QK-gain gradient's scheme; the norm in GEMM epilogues
  follows TileKernels' mHC norm kernels (split squared-sum partials reduced to the RMS norm) and DeepGEMM's in-GEMM
  pre-norm statistics (`sm90_tf32_hc_prenorm_gemm`: row squared sums inside the GEMM); dW into `.grad` follows
  DeepGEMM's D = C + A·B and Megatron-LM's gradient-accumulation fusion (`wgrad_gemm_accum_fp32` into `main_grad`).

On the GH200 the GEMMs run at the 700 W cap's ceiling; the step profile, the latest rates and the memory peak are
`docs/training.md` §7. Rejected or not pursued kernels and systems changes: `docs/experiments.md` §5.

## 4. Numerics and verification

**Precision.** Parameters, gradients and optimizer state are fp32; activations and GEMM operands bf16 with fp32
accumulation and one rounding per output, except the compiled forward's x0 and pre-norm residual mix, which stay fp32
(§1); fp32 matmuls stay fp32 (TF32 override off, `docs/training.md` §8). The q/k prologue and the value-embedding mix
compute in fp32 and round once; the rotary tables are computed in fp64. The training loss comes from the logits GEMM's
fp32 accumulators and its gradients from the bf16 rescaled exponentials (no bf16 logits); dW is fp32 throughout.
Two choices are deliberately not bitwise to their predecessors: the saved MLP activation (the down projection's dW uses
relu² of the fp32 pre-activation rounded once, not relu² of the bf16 pre-activation) and the gate kernel's fp32
reduction order.

**Non-bitwise atomics.** FA4's backward (fp32 dQ bulk reduce-adds in full layers, bf16 reduce-adds in window layers),
the token-table backward (compiled: Inductor decomposes `embedding_dense_backward` into an accumulating index_put,
lowered to an fp32 `atomic_add` scatter) and the value-embedding table's 16-byte atomics accumulate in no fixed order,
so gradients are not bitwise across runs and step losses differ between processes from step 3 (two of five 4-step d24
runs with the fixed tiles, the GEMM configurations identical). Everything else is fixed by construction: GEMM tiles
(`CONFIGS`, `DPSUM_CONFIG`, `cross_entropy.py`'s explicit tiles, `lm_losses`' fp32-output `torch.mm`), the QK-gain
gradient's summation order (`qkp_backward_blocks`: half the device's resident blocks, partials summed in block order),
the warp sums (`common.cuh` `warp_sum`) and Inductor's coordinate-descent tuning off (`docs/training.md` §8). GEMM
configurations do not change the bits of plain GEMMs (split-K 1, checked across the autotuner's picks); reduce
epilogues' partial sums follow tile_n. PyTorch's deterministic mode is rejected for its cost (`docs/experiments.md`
§5).

**Op checks against fp64.** Each landing since the fused q/k/v projection carries an op check (against an fp64
reference, or bitwise against its parent's op) and a 10-step d24 8K run at the earlier recipe's ×2.797 compared with
two runs of the parent tree (the linear cross entropy and the fused cap: every step within max(the two parent runs'
difference, 1e-3); the landing rule: `docs/experiments.md` §1); results in the §3 table. The linear cross entropy and
the fused cap against fp64 on real captures at step 200 (relative error, previous path → landed; the changes'
records):

| Quantity | Linear cross entropy, uncapped (bf16-logit path → scaled-exp) | Fused cap, capped (previous capped path → fused) |
|---|---|---|
| Mean loss | 2.8e-5 → 3.3e-8 | 1.7e-4 → 2.3e-8 |
| Per-position loss | 3.3e-3 → 5.8e-7 | 4.2e-3 → 9.5e-6 |
| Logsumexp (valid positions) | 1.3e-3 → 7.1e-7 | 7.8e-4 → 2.5e-6 |
| dx | 5.6e-3 → 2.6e-3 | 5.5e-3 → 3.1e-3 |
| dW | 2.2e-3 → 1.0e-3 | 2.2e-3 → 8.7e-4 |

The fused cap's largest 10-step difference (8.0e-4 at step 9) exceeds the previous capped path's run-to-run ≤ 7e-6;
the uncapped landing's was 6.4e-4 (step 8; its parents' spread not recorded): the more accurate gradient diverging
from the less accurate one [inference]. The uncapped path, which the uncapped records measure, stayed bitwise the
scaled-exp landing's.

**The model against fp64.** A plain-PyTorch restatement of the model in fp64 (4 layers, C 768, 2 × 2,048 tokens; its
fp32 run 2e-6 from it; recorded with the fixed-tiles change): per-parameter gradient relative error median 1.17e-2,
max 1.96e-2 with the fused q/k/v projection, against 1.18e-2 / 1.95e-2 for the model code before it, each tree
bitwise run to run, the two trees a median 3.75e-3 apart: bf16 rounding, not a defect.

## References

| arXiv | Title |
|---|---|
| 1611.09940 | Neural Combinatorial Optimization with Reinforcement Learning |
| 1901.09321 | Fixup Initialization: Residual Learning Without Normalization |
| 2002.04745 | On Layer Normalization in the Transformer Architecture |
| 2002.05202 | GLU Variants Improve Transformer |
| 2002.07028 | Low-Rank Bottleneck in Multi-head Attention Models |
| 2004.05150 | Longformer: The Long-Document Transformer |
| 2104.09864 | RoFormer: Enhanced Transformer with Rotary Position Embedding |
| 2109.08668 | Primer: Searching for Efficient Transformers for Language Modeling |
| 2308.12950 | Code Llama: Open Foundation Models for Code |
| 2309.14322 | Small-scale proxies for large-scale Transformer training instabilities |
| 2401.02954 | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism |
| 2401.14196 | DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence |
| 2402.02622 | DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging |
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

Repositories, at the version or file cited: `Dao-AILab/flash-attention` (FlashAttention-4 4.0.0b36,
`flash_attn.cute`); `Dao-AILab/quack` 0.6.5; `deepseek-ai/DeepGEMM`, `deepseek-ai/TileKernels` (design sources);
`NVIDIA/Megatron-LM` (gradient-accumulation fusion); `huggingface/transformers` (`modeling_qwen3.py`). LeCun et al.,
"Efficient BackProp" (1998) §4.6.
