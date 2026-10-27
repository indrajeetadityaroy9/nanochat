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

`CausalLM` is karpathy/nanochat 92d63d4 `nanochat/gpt.py` with the wins of its experiment log (`dev/LOG.md`, "LOG")
and Qwen3's QK-Norm (arXiv 2505.09388 §2) with learnable per-dimension weights (transformers `modeling_qwen3.py`).
`CausalLM.hidden`: x0 = norm(E[ids]); before layer i, x ← λr_i·x + λ0_i·x0; `DecoderLayer`:
h = x + o_proj(Attn(norm(x))), x = h + down(relu(up(norm(h)))²); then the final norm, `lm_head` and the capped cross
entropy (`CausalLM.forward`).
The eager code computes x0 = bf16(norm(bf16(E[ids]))) and the mix in bf16; `torch.compile`, which runs training
(`train/loop.py`) and the bits-per-byte losses (`eval/bpb.py`), computes otherwise (Inductor output read at `ca18db5`,
d16): Inductor drops the embedding's bf16 cast (`emulate_precision_casts` off), so x0 is the RMS norm of the fp32 row,
kept in fp32 and never rounded; each layer's mix is one fp32 fma(λr_i, x, λ0_i·x0) whose bf16 rounding is the stored
stream while the layer's norm reads the unrounded value (§3 Embedding; `decode.cu` computes the same).

| Component | Definition (code) |
|---|---|
| Embedding | token table 32,768 × C (`VOCAB_SIZE` = 2^15: 256 bytes, 32,508 merges, BOS and 3 FIM sentinels, `docs/corpus.md` §5), RMS-normed, no positional embedding |
| Residual | per layer x = `resid_lambdas[i]`·x + `x0_lambdas[i]`·x0, x0 the normed embedding (LOG 2026-01-11) |
| Norms, linears | gain-free RMS norm, eps = FLT_EPSILON, statistics in fp32 (`norm`); bias-free linears; untied `lm_head` |
| Attention | MHA, H = C / 128 heads of `HEAD_DIM` = 128; q and k: per-head RMS norm, × a learned per-dim gain shared by the heads (`q_norm_weight`, `k_norm_weight`), then RoPE over the full head, rotation by −θ, θ = `ROPE_THETA` = 10^6 (92d63d4 `gpt.py:57-65`); causal within each attention segment, one per packed piece of a document (FA4 varlen over `cu_seqlens`, `docs/corpus.md` §8–§9; document masks improved both long- and short-context results in ProLong's continued long-context training, arXiv 2410.02660 App. B.2) |
| Windows | `layer_windows`: layers i ≡ 3 (mod 4) and the last attend the whole context, the others the 512 keys before the query (513 with itself); d24: 6 full, 18 windowed |
| Value embeddings | `has_ve`: alternating layers, the last included (12 of 24): v ← v + 2σ(n W_g^T)_head · bf16(E_i[ids]), n the normed stream, W_g (H × C) `ve_gate`, E_i a per-layer V × C table |
| MLP | ReLU², hidden 4C |
| Output | z = x W^T capped c·tanh(z/c), c = `ModelConfig.final_logit_softcapping` = `FINAL_LOGIT_SOFTCAPPING` = 15 (`meta_model`), in the training loss, the evaluation losses and the decoder's sampling; no z-loss |
| Precision | fp32 parameters and gradients; bf16 activations (compiled: x0 and the residual mix before its norm in fp32, above) and GEMM operands (bf16 weight copies `weight_bf16`, `qkv_bf16`, refreshed after every update by `refresh_gemm_weights`); fp32 accumulation (§4) |

**Shape rule** (`meta_model`, 92d63d4 `scripts/base_train.py:51-52, 131-134`): C = 30 × `NUM_HIDDEN_LAYERS` (24)
rounded up to whole 128-wide heads; 30 = MobileLLM-350M's 960 / 32 (arXiv 2402.14905 Table 9). d24: 720 → 768.
The code builds d24 only; d8 and d16, the proxies of `docs/experiments.md` §1, are the same rule at 8 and 16 layers
(built by `meta_model(num_hidden_layers=…)` at `67192ab`, with GEMM tiles and kernel instances for their widths).
`max_position_embeddings` is the stage's context; `set_stage` recomputes the windows and the rotary tables (computed in
fp64, rounded once to fp32: `init_buffers`), the weights unchanged.

**Initialization** (`init_weights`): q/k/v_proj, up_proj, the token table and the value-embedding tables uniform on
±√(3/C) (std C^−½, "Efficient BackProp" §4.6); zero: o_proj and down_proj (each layer starts as the identity,
modded-nanogpt record #5), `lm_head` (record #8), the gates' W_g (gate 1), `x0_lambdas`; one: `resid_lambdas` and the
QK-norm weights. Each random tensor draws from a generator seeded by seed·2^32 + crc32(its name).

**Counts** (`num_non_embedding_params`: decoder layers + `lm_head`, what the token budget scales with,
`docs/training.md` §2; `num_matmul_params`: every 2-D weight but the tables, PaLM's N; M:
`non_embedding_flops_per_token`, DeepSeek LLM's 72·n_layer·C² + 12·n_layer·C·l_seq, arXiv 2401.02954 §3.2 Eq. 2, the
count its batch and lr laws were fit with (C = MD; its Table 3 evaluates M at l_seq 4,096); windowed layers counted as
full, on the assumption that a model with near-lossless local windows (arXiv 2503.19786 §5.2) shares its
full-attention counterpart's hyperparameters [inference]; nominal: `metrics.model_flops_per_token`, 6N + 12·C·
Σ_layers min(window + 1, context), each layer's most keys a query attends, itself included; defined in
`docs/evaluation.md` §5). Recomputed on CPU from `model/modeling.py` on the meta device (d8 and d16 by the same shape
rule through `ModelConfig`; `meta_model(num_hidden_layers=…)` at `67192ab`) [arithmetic]:

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

**Lineage.** Taken from 92d63d4: the block layout, value-embedding layers, ReLU² MLP, SSSL pattern and the cap at 15.
Not taken: its smear and backout, its QK order (RoPE, then norm, then ×1.2), its short window (a quarter of the
context rounded up to 128), its RoPE base (10^5) and its init. The gate is nanochat e85db6b's (2·sigmoid, zero init)
read over the full width rather than its first 32 channels (92d63d4 reads 12 channels with 3·sigmoid); the full-width
read is not ablated.

- **Value embeddings, kept** (from the literature): per-layer token-indexed tables are modded-nanogpt record #14's and
  MoVE's LaVE baseline, V = V + g ⊙ M_t with g = 2σ(x W_g) (arXiv 2601.22887 §4.1), on half the layers; LaVE ×1 gains
  0.016, 0.010 and 0.010 validation bpb at D12, D20 and D32 (§4.2 Table 1). nanochat found alternating layers best
  and the gate helpful (LOG 2026-01-17). None of the reference code models has them.
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
  earlier-recipe `peak_lr` run, uncapped, diverged up to ×8 (`docs/training.md` §5), so no z-loss. The cap is Gemma
  2's soft_cap·tanh(logits/soft_cap) (final layer 30, arXiv 2408.00118 §2). Effect sizes: modded-nanogpt record #9
  added it at 30 (val loss 3.2844 → 3.2791 between the record's last two incremental runs, one run each) and record
  #18 lowered it to 15 (steps to the 3.28 target 1,490 → 1,390; 80 runs at 1,390 steps, mean 3.2791, one-sided
  p = 0.0001 for a mean below 3.28); nanochat at d24 found caps from 5 to 30 about equal except 5 (much worse), 20
  best by ~1e-3 val loss (LOG 2026-03-02), while 92d63d4 `gpt.py:511-515` keeps 15.
  Its cost: 0.58% of the 8K step (§3). The designed ablation and its detection power: `docs/experiments.md` §3.
- **RoPE base 10^6 from the first step:** above the 32K context's lower bound of arXiv 2405.14591 (Table 2: 6.4e5);
  large from the start as Llama 3 (500,000, arXiv 2407.21783 §3.2); base 10k against 500k at 8K changes Gemma 2B's
  perplexity 4.4627 → 4.4485 (Wiki) and 6.4429 → 6.4593 (FlanV2), under 0.33% and of mixed sign (arXiv 2410.06205
  Table 2).
- **ReLU², not SwiGLU:** nanochat measured SwiGLU at matched parameters and FLOPs worse at d12 and d24 on steps,
  wall clock and FLOPs (LOG 2026-02-05); a gated MLP would also need a gated decode MLP (`model/kernels/decode.cu`
  hard-codes relu², `EPI_RELU2`).
- **SSSL windows of 512:** nanochat's SSSL pattern with the last layer full (LOG 2026-01-11); Gemma 3 finds the
  local:global ratio of minimal impact up to 7:1 and the window reducible "significantly without impacting
  perplexity" (arXiv 2503.19786 §5.2, Fig. 3–4: 2B and 9B models; windows 512–4,096 at 1:1 and 3:1).
- **Shape:** the shape rule's MobileLLM ratio (§1). A deeper, thinner alternative at about d24's
  non-embedding parameters, 35 × 640 (193,070,080), costs M 3,234,201,600 per token at 8K (+14.2%) and nominal
  1,827,033,600 (+6.9%) [arithmetic], so an equal-token comparison gives it more compute; its 5 heads would need
  their own `ve_gate.cu` instance and GEMM tiles (the code is d24's: H = 6, C = 768).
- **head_dim 128:** compiled into the kernels (`HEAD_DIM`, `native.py`); nanochat found 128 better than 64 (LOG
  2026-01-17).
- **MHA, not grouped-query attention:** GQA's benefit is KV-cache memory; its d16 screen is `docs/experiments.md` §3.
  The decoder's e4m3 cache (`docs/evaluation.md` §6: 38,016 bytes per position at d24) is 44.0 MB for 1,158 positions
  (a slot's `max_length`, len(prompt) + max_new_tokens − 1: a 391-token prompt [unverified] and 768 new tokens) and
  45.1 GB for `decode.cu`'s 1,024 slots (`MAX_BATCH`) [arithmetic], which the GPU holds.
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

**Build** (`native.py`). `cuda_library(name)` builds `model/kernels/<name>.cu` with one nvcc command for the GH200's
architecture-specific target (sm_90a: `decode.cu`'s wgmma) and `-DHEAD_DIM=128`, into `NATIVE/<name>-<sha256 of
the command, the file and every header>.so` (written to a temporary file unique to the build and renamed, so
concurrent first builds, from any process or container, never share a file), loaded by ctypes once per process.
`crate_module` does the same for the Rust crates through cargo (`corpus/crate`, `docs/corpus.md` §6; `inference/syntax`,
`docs/evaluation.md` §7). The kernels share headers (`common.cuh`: 16-byte vector loads
and stores, a fixed-order butterfly warp sum, resident-block counts for persistent grids; `qk_row.cuh`: the q/k row
math (RMS norm, gain, RoPE) shared by `qk_prologue.cu` and the decoder's `decode.cu`); each kernel is one
self-contained file with its C entry points. Each op is a
`torch.library` custom op with a fake implementation, so `torch.compile(fullgraph=True)` traces one graph per stage
(`train/loop.py`).

| Op | Implementation |
|---|---|
| q/k/v projection | `linear.py` `qkv_proj`: one (M, 3C) QuACK GEMM against the stacked bf16 [Wq; Wk; Wv]; q, k, v its column blocks, read by attention as strided views; dX one GEMM with K = 3C; dW three cuBLAS `addmm` (fp32 out, β = 1) straight into each weight's `.grad` |
| Attention | `flash_attn.py` `attention`: one op holding the q/k prologue, the value-embedding mix, FlashAttention-4 (CuTe DSL) varlen over the micro-batch's real segments only (`segmented`) and `o_proj`; its backward writes every q/k/v gradient into one (B, T, 3C) buffer. Window layers: `PersistentForwardSm90` with `PersistentVarlenScheduler` (one CTA per SM, the output tile staged apart from Q) and `BackwardSm90Bf16dQ` (dQ reduce-added in bf16 by `cp.reduce.async.bulk` into a zeroed bf16 dq: no fp32 accumulator, fp32 zero fill or postprocess; masks only on boundary tiles). Full layers (`fa4_window_left` → None when the window spans T): FA4's causal kernels, fp32 dQ |
| o_proj, post-attention norm | `linear.py` `residual_rms`: h = residual + O W_o^T and each row's squared sum in the GEMM epilogue (QuACK `sq_reduce_mod`, `rms_final_reduce`), so rstd = (mean(h²) + eps)^−½ without a norm pass; backward `residual_rms_bwd`: dO with FA4's dpsum = rowsum(dO·O) per head summed in the dX epilogue (`dx_dpsum`, `DPSUM_CONFIG`: one 128-column tile per head) |
| FA4 row statistics | `fa4_rows.cu`: dpsum and lse·log2 e at FA4's padded rows (+inf lse, 0 dpsum on padding) in place of FA4's backward preprocess (`precomputed_dpsum`), which read O and dO again; one block of m threads per FA4 backward m-block |
| MLP | `linear.py` `relu2_mlp`: the norm as a row scale, since norm(h)Wᵀ = diag(rstd)·hWᵀ for a gain-free norm: rstd and relu² in the up projection's epilogue (`gemm_norm_act`, pre-activation and activation both stored for the backward), the residual as the down projection's C (`gemm_add`); backward: relu²'s derivative × rstd and the norm's row sum in one epilogue (`relu2_dact_rstd`), the norm's backward in the up projection's dX epilogue (`norm_bwd_apply`); the normed stream is never stored |
| Tiles | `CONFIGS`: a fixed QuACK SM90 configuration per call at d24's C = 768 and M = 98,304, each the lowest mean of three runs of QuACK's own autotuning benchmark |
| q/k prologue | `qk_prologue.cu`: norm → gain → RoPE in fp32, one bf16 rounding per output, one pass per direction; forward one thread group per position, its gains and cos/sin loaded once for the H heads, two heads in flight (H = 6 even); gain gradient without atomics (persistent blocks, per-block partials summed in block order) |
| Value-embedding gate and mix | `ve_gate.cu` (`ve_gate.py`): one warp per row, the gate logits in fp32, the mix rounded once; backward in token order (one `torch.sort` of `input_ids` per micro-batch, shared by the layers through CSE), 16 positions per warp, a run of equal tokens summed in registers and added into the table's `.grad` with one 16-byte atomic per 4 fp32 (two per lane and 8-wide chunk; the zeroed table gradient allocated once per step, when `.grad` is None, then accumulated into; no autograd add); the gate weight's gradient one cuBLAS GEMM from bf16 dz; compiled for d24's H = 6 |
| Cross entropy, training | `cross_entropy.py` `lm_loss`: QuACK's scaled-exp linear cross entropy per `CHUNK` = 8,192 rows: the logits GEMM's epilogue (N tile 128, pingpong) stores bf16 rescaled exponentials with each row's tile maxima, sums and the label's fp32 logit; `lce_glue` forms the loss from the fp32 accumulators; dX (bf16) and dW (fp32, accumulated across chunks) GEMMs rescale on the A-operand load, their N tile over C QuACK's rule at C = 768 (`TILE_N2` = 192). No logits tensor and no softmax pass. The cap in the same pipeline: tanh on the fp32 accumulator, the tile maximum capped likewise (tanh is increasing), 1 − tanh² folded into the stored exponentials, the label's entry fixed by one Triton kernel (`cap_label_fix`) |
| Cross entropy, evaluation | `lm_losses`: per `CHUNK` rows an fp32-output GEMM and the capped logsumexp in one Inductor-fused kernel (`capped_losses`) |
| Embedding | `F.embedding` of the token table; Inductor fuses the gather, the bf16 cast and the norm without rounding in between (`torch._inductor.config.emulate_precision_casts` off), so the compiled x0 is the norm of the fp32 rows, as `decode.cu` `k_norm` computes it (d8, 98,304 random ids at `8be5e90`: 114 of 25,165,824 elements differ from eager bf16(norm(E[ids])), 6,312,306 from eager norm(bf16(E[ids]))); the backward, compiled: Inductor decomposes `embedding_dense_backward` into `aten._unsafe_masked_index_put_accumulate` on an fp32 zero table (`torch/_decomp/decompositions.py`, registered in `torch/_inductor/decomposition.py`), lowered to `_unsafe_index_put(..., accumulate=True)`, a scatter with `scatter_mode="atomic_add"` (`torch/_inductor/lowering.py`): rows accumulated with fp32 atomics (§4) |
| Optimizer | Muon's Newton–Schulz batched over same-shape matrices and NorMuon: `docs/training.md` §3 |

**Landed changes** (d24 tok/s as each commit's message records them: an exclusive A/B on the GH200 at 700 W against
the master tree the change landed on, the base the message names, not always the commit's git parent; the messages
record the SM clock as locked for `a81410c` and `7ad594f` only, the others do not state it (`docs/training.md` §7);
mostly 25 timed steps after 8 warm-up at 8K and 12 after 6 at 32K, median, an arm run twice by its mean; `f499be7` 40
steps, 29 timed; `373b549`'s 8K rate a 200-step run against an earlier base run, not one window). Rates of different
rows come from different measurement windows and do not chain. Loss checks: 200 steps against the d24 seed spread
(seed 0 vs 1, last-20 mean Δ 0.035), or 10 steps at the earlier recipe's base multiplier ×2.797 against the parent
tree's run-to-run spread (§4).

| Change | Commit | 8K tok/s | 32K tok/s | Memory, other | Numerics |
|---|---|---|---|---|---|
| FA4 on the real segments (the loader had padded `cu_seqlens` to 2,796 segments for ~94 real) + Muon's Newton–Schulz batched by shape | `a81410c` | step 3,709 → 3,511 ms (−5.3%); d16 266 → 239.5 ms (−10%) | | | losses bitwise; gradients within run-to-run spread; momentum bitwise, parameters at bf16 GEMM rounding |
| Fused value-embedding gate and mix | `7ad594f` | step time −0.5% | | | ~15% of elements one bf16 ulp from the PyTorch expressions (fp32 reduction order) |
| q/k prologue per position | `62b50c2` | +0.3%; forward 250.3 → 183.6 µs per call | | | 200-step Δ +0.0035 |
| Post-attention norm in GEMM epilogues | `373b549` | 338,039 → 341,919 (+1.15%) | 274,783 → 276,661 (+0.68%) | peak 64.5 → 61.1 GiB; nsys GPU time 4,355 → 4,296 ms per step | 200-step Δ −0.0071; y vs fp64 2.6e-3 (bf16 path 3.2e-3) |
| Token-ordered value-embedding backward | `eaf14d0` | 344,768 → 350,185 (+1.57%) | 277,238 → 280,201 (+1.07%) | ve_gate 161.9 → 92.6 ms per step; peak unchanged | 200-step Δ +0.0095 |
| dW accumulated into the fp32 `.grad` by cuBLAS `addmm` (β = 1) + one bf16 copy per weight per update | `f499be7` | 344,925 → 347,634 (+0.79%) | 275,613 → 277,383 (+0.64%) | peak 61.6 → 62.0 GiB | neutral |
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
  these GEMMs' time at d24 / d16 / d8; a single benchmark run's winner is up to 2.1% slower than the three-run best,
  and a cold first d24 step spent ~80 s autotuning (`linear.py` `CONFIGS`, `fb86efc`). With `e24333d` no training
  GEMM autotunes, and a cold start peaks at the warm 71.11 GiB.
- **Norm in epilogues.** The rstd-scaled relu² derivative epilogue costs ~200 µs per layer and micro-batch over QuACK's
  plain `gemm_dact` (~72 ms per step); the Inductor norm and mix kernels that remain (norm before attention, the
  residual mix) run at ~3.6 TB/s.
- **Linear cross entropy.** On a real d24 micro-batch, chunks of 8,192 to 65,536 rows time alike (24.1–24.4 ms per
  call) while the op's peak grows (0.82 → 4.59 GiB); a 98,304-row chunk faults in QuACK 0.6.5; N tile 128 with pingpong
  24.2 ms against 256 without 25.5 (`cross_entropy.py` `CHUNK`, `TILE_N1`); QuACK's own rule for this tile
  (`_pick_tile_n1`: 256 at V = 2^15, no pingpong) times +5.8% at C = 768, +3.7% at 512 and within 0.3% at 256
  against 128 with pingpong (QuACK 0.6.5's `scaled_exp_linear_cross_entropy_fwd`, uncapped, 98,304 rows, median of
  20 calls). The strip GEMMs' N tile is QuACK's rule (`strip_tile_n` at `67192ab`); on a 98,304-row micro-batch with
  the cap (GH200, exclusive, median of 5 alternated rounds of 20 calls) the op at C = 512 (d16) takes 19.06 ms with
  256 against 23.60 ms with a fixed 192 (−19.2%; 192 leaves 3 N tiles, which the dW GEMM's cluster_N = 2 pads to 4
  [inference]), at C = 256 (d8) 15.71 against 15.05 ms (+4.4%; one 256-wide N tile gives the dX GEMM 64 CTAs per
  8,192-row chunk on 132 SMs, 192 gives 128 [inference]), and at C = 768 (d24) the rule gives 192, the code's
  `TILE_N2`. Loss, dX and dW are bitwise equal at every width whichever tile runs, and the uncapped op (removed with
  the uncapped option) was bitwise QuACK 0.6.5's `scaled_exp_linear_cross_entropy_fwd` at the same `tile_n1`.
- **The cap.** The removed capped path (bf16 capped logits, a logits-sized derivative buffer, a cross-entropy pass, an
  autotuned dX GEMM) cost +9.4 ms and +6 GiB over uncapped (28.04 → 37.44 ms, 6.14 → 12.14 GiB). Over the run, 0.58%
  of the 8K stage's 74.0 h is ~26 min [arithmetic, `docs/training.md` §7]; the cap's 32K cost is not measured.
- **dpsum.** QuACK's autotuner does not pin tile_n, so `DPSUM_CONFIG` is fixed: the fastest QuACK SM90 configuration
  with tile_n = 128 on the d24 shape (172.2 µs; cluster (2, 1) 172.8, tile_m 192 212.4 / 218.3, cooperative 256
  268.5 / 270.4); `fa4_rows` took 7.5 µs at `db1e90e` (256-thread blocks; the current launch is not timed alone).
- **Value-embedding backward.** RUN = 16 positions per warp (isolated backward 222 / 203 / 251 µs at RUN 8 / 16 / 32),
  2 blocks of 256 threads per SM (`ve_gate.cu`).
- **q/k prologue forward.** Two heads' loads in flight per thread group: `k_forward` 250.3 → 183.6 µs per call at d24
  8K against one head per group (`qk_prologue.cu`, `62b50c2`).
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
`docs/training.md` §7. Rejected or not pursued kernels and systems changes (FP8, hand
attention kernels, QK-norm + RoPE in the projection epilogue, the residual mix in the down projection, window-attention
tile skipping, split-K and DeepGEMM dW, a global cuBLASLt workspace, a single 3C × C q/k/v dW, cuDNN attention,
PyTorch's deterministic mode, max-autotune, Inductor's PDL, Liger and Cut Cross-Entropy): `docs/experiments.md` §5.

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
lowered to an fp32 `atomic_add` scatter; torch 2.14 `torch/_decomp/decompositions.py`, `torch/_inductor/lowering.py`)
and the value-embedding table's 16-byte atomics accumulate in no fixed order, so gradients are not bitwise across
runs and step losses differ between processes from step 3 (two of five 4-step runs at `fb86efc`, the GEMM
configurations identical). Everything else is fixed by construction: GEMM tiles (`CONFIGS`, `DPSUM_CONFIG`,
`cross_entropy.py`'s explicit tiles, `lm_losses`' fp32-output `torch.mm`), the QK-gain gradient's summation order
(`qkp_backward_blocks`: half the device's resident blocks, partials summed in block order), the warp sums
(`common.cuh` `warp_sum`) and Inductor's coordinate-descent tuning off (`docs/training.md` §8). GEMM configurations do
not change the bits of plain GEMMs (split-K 1, checked across the autotuner's picks); reduce epilogues' partial sums
follow tile_n. PyTorch's deterministic mode is rejected for its cost (`docs/experiments.md` §5).

**Op checks against fp64.** Each landing since `70025d8` carries an op check (against an fp64 reference, or bitwise
against its parent's op) and a 10-step d24 8K run at the earlier recipe's ×2.797 compared with two runs of the parent
tree (`067e60b` and `e24333d`: every step within max(the two parent runs' difference, 1e-3); the landing rule:
`docs/experiments.md` §1); results in the §3 table. The linear cross entropy and the fused cap against fp64 on real
captures at step 200 (relative error, previous path → landed; the commits' messages):

| Quantity | `067e60b` uncapped (bf16-logit path → scaled-exp) | `e24333d` capped (removed capped path → fused) |
|---|---|---|
| Mean loss | 2.8e-5 → 3.3e-8 | 1.7e-4 → 2.3e-8 |
| Per-position loss | 3.3e-3 → 5.8e-7 | 4.2e-3 → 9.5e-6 |
| Logsumexp (valid positions) | 1.3e-3 → 7.1e-7 | 7.8e-4 → 2.5e-6 |
| dx | 5.6e-3 → 2.6e-3 | 5.5e-3 → 3.1e-3 |
| dW | 2.2e-3 → 1.0e-3 | 2.2e-3 → 8.7e-4 |

The fused cap's largest 10-step difference (8.0e-4 at step 9) exceeds the old capped path's run-to-run ≤ 7e-6; the
uncapped landing's was 6.4e-4 (step 8; its parents' spread not recorded): the more accurate gradient diverging from
the less accurate one [inference]; the uncapped path (since removed) was bitwise `067e60b`'s.

**The model against fp64.** A plain-PyTorch restatement of the model in fp64 (4 layers, C 768, 2 × 2,048 tokens; its
fp32 run 2e-6 from it; recorded in `fb86efc`'s message): per-parameter gradient relative error median 1.17e-2, max
1.96e-2 at `70025d8`, against 1.18e-2 / 1.95e-2 at `646a986` (the model code of its parent `cf0aff5`), each tree
bitwise run to run, the two trees a median 3.75e-3 apart: bf16 rounding, not a defect.

## 5. Implementation reference

Per-module detail behind the code's docstrings (paper-facing facts are in §1–§4 and not repeated).

**`model/modeling.py`**
- `init_weights`: the zero-initialized gate weight gives gate 1; `resid_lambdas` = 1 follows LOG 2026-01-11, the
  QK-norm gains = 1 transformers' `Qwen3RMSNorm`.
- `GemmLinear.weight_bf16`, `Attention.qkv_bf16` (the stacked [Wq; Wk; Wv] for `qkv_proj`): non-persistent buffers
  refreshed by `refresh_gemm_weights` after init, after a load (a `load_state_dict` post-hook) and after every
  optimizer step; copied in place because a compiled graph holds them; reallocated only when a load assigned weights
  on another device (`train/checkpoint.py` `load_model`'s meta model). dW accumulates into each fp32 `.grad`.
- `Attention.forward`: the value-embedding arguments are (n, `input_ids`, table, W_g, `torch.sort(input_ids)`); the
  sort runs once per micro-batch, every layer's identical sort merged by AOTAutograd's common-subexpression
  elimination (`torch/_functorch/compile_utils.py` `fx_graph_cse`).
- `hidden`: `cu_seqlens` come from `corpus/loader.py` `PretrainingBatches`, fixed in length per loader, unused entries
  empty segments; `segments` is a 0-dim int32 host tensor (no device synchronization).
- `num_matmul_params`: norm gains and scalars are elementwise, so excluded with the embedding tables.

**`model/ops/linear.py`**
- Weight gradients: each op takes its weights' fp32 `.grad` (allocated as zeros by `train/loop.py`, zeroed after each
  step by `train/optim.py` `update`) and adds dW in place, so micro-batches accumulate in the GEMM; autograd sees no
  gradient for those weights. The ops are `torch.library` custom ops so fullgraph `torch.compile` traces them;
  QuACK's fused epilogues come from `quack.gemm_interface` and `quack.epilogue`.
- `qkv_proj`: q, k, v are the column blocks [0, C), [C, 2C), [2C, 3C); `qkv_proj_bwd` dX = dY [Wq; Wk; Wv] (one GEMM,
  K = 3C), each dW = dY_blockᵀ X on the block's strided view of dY.
- `residual_rms`: QuACK's `gemm_rms` epilogue and finalize (`quack/gemm_interface.py` `_gemm_rms_call`,
  `_rms_finalize`) via `sq_reduce_mod` and `rms_final_reduce`: per-tile squared sums, finalized to
  rstd = (mean(h²) + `EPS`)^−½ (`EPS`: the one RMS-norm epsilon, also `modeling.norm`'s); called inside the attention
  op. `residual_rms_bwd`: x = O (FA4's output, heads of `HEAD_DIM`); dpsum is the sum over each head's dims of dx·x.
- `relu2_mlp`: `gemm_norm_act` stores the scaled pre-activation and the activation (dW_down's operand is the saved
  activation); the residual is the down projection's C (`gemm_add`'s body). Its backward owns the norm's backward
  (rstd gets no gradient of its own): with P = h W_upᵀ, preact = rstd·P, dP = rstd·dpreact,
  d rstd = Σ(dpreact·preact) / rstd and ∂rstd/∂h = −rstd³ h / C, giving the docstring's dh; the row sum is a reduce
  of the dact epilogue (`relu2_dact_rstd`), the rest the dX GEMM's epilogue (`norm_bwd_apply`, C = dy, h a tile load).
- `sm90`: the point of QuACK's SM90 space (`quack/gemm_config.py` `_get_sm90_configs`) with the given tile, cluster
  and operand order; the space sets pingpong (tile_m 128 with tile_n up to 208, and 192 × 128) or cooperative (tile_m
  256, and 128 × 224 or 128 × 256) and QuACK's static persistent scheduler (`is_dynamic_persistent=False`; the
  `down_dx` GEMM of `relu2_mlp_bwd` passes `dynamic_scheduler=True`, the default of QuACK's `gemm_dact`).
  `CONFIGS` holds d24's calls (C = 768) only; the d8 and d16 entries are at `67192ab`.
  M = `corpus/mixture.py` `MICRO_BATCH_TOKENS`. The benchmark is `quack/gemm_runtime/autotune.py` (L2-cold
  CUDA-graph medians of every eligible candidate, exclusive GPU). An explicit config bypasses
  QuACK's tuner whatever `tuned` says (`quack/epilogue/frontend.py`: the tuner runs only for `config=None`). `gemm`
  calls `gemm_tuned.fn` because QuACK's `gemm`/`gemm_add` take no config. `DPSUM_CONFIG` is the fastest tile_n = 128
  configuration on 98,304 × 768 × 768.

**`model/ops/cross_entropy.py`**
- `lm_loss` follows `quack/linear_cross_entropy.py` `scaled_exp_linear_cross_entropy_fwd` (its calls and tiles, except
  `TILE_N1` with pingpong where QuACK picks 256 without, §3 notes), per `CHUNK` rows. The logits GEMM's epilogue
  stores E = bf16(2^(z·log2 e − k)·(1 − tanh²)), k = rne(log2 e · the largest capped logit z in the row's
  `TILE_N1`-wide tile), with each tile's sum of the plain exponentials and the label's capped logit z[t] in fp32 (the
  recorded logsumexp is loss + z[t], `lm_loss`'s last output). `lce_glue` forms
  logsumexp − z[t] from the fp32 accumulators, fixes E at the label and writes each row's power-of-two rescalings.
  dx = (1/n)·dz W (bf16) and dW = (1/n)·dzᵀ x (fp32, accumulated across chunks, `add_to_output`) are GEMMs on E whose
  A-operand load applies the rescalings: dz = v·(u·E) row-wise, u exact powers of two, v fp32.
- Epilogue: `capped_scaled_exp_epi`, QuACK's `scaled_exp_target_epi` (`quack/epilogue/scaled_exp.py`) on the capped
  logits; its `sum_exp` sums the plain exponentials while D carries 1 − tanh², so the gradient GEMMs read
  dz = (softmax − onehot)(1 − tanh²); `cap_label_fix` applies the derivative to the label entry `lce_glue` rewrote.
- Constants: `CHUNK` the least-memory size among those equally fast on a d24 micro-batch, also the evaluation's rows
  per fp32 GEMM (1 GiB of logits); `TILE_N1` the fastest admissible tile (§3 notes); `TILE_N2` QuACK's tile_n2 rule
  (`linear_cross_entropy.py`: 192 if C % 192 = 0, else 256) at C = 768 (192; 256 at C = 256 and 512).
- `lm_loss` is opaque to `torch.compile` (CuTe DSL GEMMs, Triton kernels, gradients computed in the forward); labels
  int64, dx in x's dtype; `lm_loss_grad` multiplies the saved gradients by dloss, dx's product in fp32 rounded once.
  `step_shares` weights sum the micro-batch means to the mean over every valid label of the step.
- `lm_losses`: QuACK's `lse_target_epi` (logsumexp and label logit from the fp32 accumulators) has no tanh epilogue,
  hence the fp32 GEMM and `capped_losses`.

**`model/ops/flash_attn.py`**
- One op because FA4's own `autograd.Function` is not traceable by `torch.compile`. The backward writes every
  projection gradient into one (B, T, 3C) buffer: FA4's dv into the v block (read there by the value embeddings'
  backward; the mix passes dv through to v), the prologue's dq, dk into the q and k blocks.
- `attention` arguments: norm weights (`HEAD_DIM`,) fp32; cos/sin (T_table ≥ T, `HEAD_DIM`/2) fp32; `o_weight` o_proj's
  bf16 copy (C, C); value-embedding inputs n, `input_ids`, table, W_g and the sort's `ids`, `order`. Outputs: rstd is
  non-differentiable because `relu2_mlp`, its one consumer, differentiates through it; out (B, T, H, `HEAD_DIM`), lse
  (H, B·T) fp32; mixed v and gate empty without value embeddings (likewise dn, dw_gate in `attention_bwd`).
- `PersistentVarlenScheduler`: FA4's SM90 forward launches one CTA per (segment m-block, head)
  (`flash_fwd_sm90.py` `FlashAttentionForwardSm90.__call__`, `SingleTileVarlenScheduler`); the subclass reuses
  `VarlenDecoder.decode`. `PersistentForwardSm90` doubles sQ's allocation (the rest of FA4's `SharedStorageQKV`
  unchanged; FA4's `Q_in_regs` layout, which `interface.py` never selects, is not offered) because FA4's sO is a view
  of sQ (`flash_fwd_sm90.py`), which a persistent CTA's next Q load would overwrite [inference].
- `BackwardSm90Bf16dQ`: replaces FA4's fp32 `dq_accum` (zero-filled, converted by `flash_bwd_postprocess.py`): the
  consumers stage dQ × softmax scale as bf16 rows and the dQ warp reduce-adds each row into dq (PTX ISA
  `cp.reduce.async.bulk .add.noftz.bf16`). `mma_one_m_block` restates FA4's (`mma_dkv_is_rs`,
  Q_stage = dO_stage = PdS_stage, no score_mod); FA4's `mma` masks every tile, `_needs_mask` only boundary tiles. Tile
  shapes and stages are FA4's (`_tile_size_fwd_sm90`, `_tile_size_bwd_sm90`).
- `precomputed_dpsum` replaces FA4's `_bwd_preprocess` (`interface.py`, `flash_bwd_preprocess.py`), which reads O and
  dO for dpsum. `fa4_window_left` → None also avoids plain causal attention running on FA4's local kernel
  (`interface.py` `_resolve_causal_local_window`). `heads` views have token stride 3C.
- FA4 version: 4.0.0b36 (`pyproject.toml`). Its changes over 4.0.0b35 are SM100-only (the backward's accumulate-tile
  order, a generic hdim-256 backward, a tile-N override knob, an unused host config module); the SM90 forward and
  backward kernels, the tile scheduler, `_bwd_preprocess`, `_tile_size_bwd_sm90`, `_resolve_causal_local_window` and
  the JIT cache the subclasses and hooks use are unchanged, and the SM90 path of `flash_bwd_postprocess.py` is
  unchanged in effect.
- CUTLASS DSL 4.8.1 warns `UserWarning: Argument aux_data (position N) cannot be converted to a JitArgument` when it
  compiles a kernel whose `__call__` takes FA4's `AuxData`: position 17 for `PersistentForwardSm90`
  (`FlashAttentionForwardSm90.__call__`), 21 for `BackwardSm90Bf16dQ` (`FlashAttentionBackwardSm90.__call__`),
  counting `self` as 1; Python prints the first only (one warning location, `base_dsl/dsl.py:2300`), the forward's in
  `artifacts/runs/tokens_per_param=5.0/run.log`. This code passes `AuxData(None, None)`, as FA4's own interface does
  without auxiliary tensors (`interface.py`): a NamedTuple the DSL cannot make a runtime argument, so it holds it as a
  compile-time constant, which the kernels' `const_expr(aux_data.tensors is not None)` branches expect: no effect on
  the kernels. 4.0.0b35's signatures take the same `AuxData`; 4.0.0b36 and CUTLASS DSL 4.8.1 are PyPI's newest
  releases of either.

**`model/ops/qk_prologue.py`, `model/kernels/qk_prologue.cu`, `model/kernels/qk_row.cuh`**
- Called inside the attention op; norm → gain → RoPE is transformers' `modeling_qwen3.py` `Qwen3Attention.forward`
  order; eps is `F.rms_norm`'s fp32 epsilon. dq, dk are the q, k blocks of the projection's gradient; the forward's
  outputs and the incoming gradients are contiguous.
- Layout: token stride 3C = 3·H·`HEAD_DIM`; position of token t = t mod T; the first half of the blocks takes q, the
  second k. Forward: a `QK_LANES`-thread group per position, its gains and cos/sin loaded once for the H heads, two
  heads' loads in flight (H = 6); block size from the occupancy calculator (a multiple of `WARP`, so rows never
  straddle warps). Row sums reduce by shuffles within a row's lanes.
- Gain gradient: persistent blocks stride over rows accumulating dm·n in registers, reduce their warps in order into a
  per-block partial (`part`, fp32 (2·`qkp_backward_blocks()`, `HEAD_DIM`)), and `k_gain_sum` adds them in block order.

**`model/ops/ve_gate.py`, `model/kernels/ve_gate.cu`**
- v is the fused projection's v block and FA4's dv sits in place in the projection's gradient (token stride 3C);
  the mixed v's gradient is v's own. The op owns the lookup, so autograd receives None for the table. The gate
  weight's gradient dzᵀ n is a cuBLAS GEMM from bf16 dz, fp32 out. Sigmoid's derivative is written as torch's
  `sigmoid_backward` writes it.
- Layout: n, out, dn contiguous (rows, C), C = H·`HEAD_DIM`; v and dout rows `v_stride`/`dout_stride` apart (C, or 3C
  for the fused projection); idx int32; table, dtable (V, C) fp32; gate (rows, H) fp32; dz (rows, H) bf16. A row is
  one warp; lane l holds the `VEC`-wide chunks l, l + `WARP`, …, each inside one head. H = 6 (d24) is a compile-time
  constant (per-head arrays in registers); the d8 and d16 instances (H = 2, 4) are at `67192ab`.
- Backward: rows in token order, `RUN` positions per warp; a run of equal tokens reads its table row once, sums its
  dve in registers and adds the sum with one 16-byte `atomicAdd` per 4 fp32 (two per lane and `VEC` = 8 chunk; PTX
  `atom.global.add.v4.f32`, its result unused, assembled to `REDG.E.ADD.F32x4` for sm_90a);
  dtable is accumulated into.

**`model/kernels/fa4_rows.cu`**
- Inputs: dpsum (rows, H) fp32 (the epilogue's per-(row, 128-column tile) partials), lse (H, `lse_stride`) fp32, cu
  (segments + 1) int32; outputs (H, padded) fp32. One block of m threads per backward m-block (padded is a multiple of
  m), one thread per padded row, which finds its segment by binary search over start_b. Each segment's rows past its
  length up to a multiple of m, which the segment's last backward m-block loads, get dpsum 0 and lse_log2 +inf (the
  preprocess's lse = +inf), so their P is 0; rows between start_b + round_up(len_b, m) and start_{b+1} are never read.
  A real row whose lse is −inf gets lse_log2 0.

**`native.py`**
- The same nvcc command line builds each kernel (one invocation per kernel, on first use); the toolchain is the
  image's; sm_90a per the CUDA C++ Programming Guide, "Feature Availability". `corpus_crate` exposes `train`
  (`corpus/train_tokenizer.py`), `topological_sort` (`corpus/sources.py`), `Grams` (`corpus/decontaminate.py`) and
  `pack` (`corpus/pack.py`).

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
`KellerJordan/modded-nanogpt` (records #5, #8, #9, #14, #18); `Dao-AILab/flash-attention` (FlashAttention-4 4.0.0b36,
`flash_attn.cute`); `Dao-AILab/quack` 0.6.5; `deepseek-ai/DeepGEMM`, `deepseek-ai/TileKernels` (design sources);
`NVIDIA/Megatron-LM` (gradient-accumulation fusion); `huggingface/transformers` (`modeling_qwen3.py`). LeCun et al.,
"Efficient BackProp" (1998) §4.6.
