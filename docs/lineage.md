# Literature behind the upstream ideas

The ideas this repository took from karpathy/nanochat ("nanochat"; "LOG" is its `dev/LOG.md` at `92d63d4`) and
KellerJordan/modded-nanogpt ("modded"; record #N is row N of its README track-1 table), each with the papers to cite
for it. Upstream commits, records and LOG entries are provenance only, never evidence. Classes: ADOPTED, ADAPTED,
PARALLEL (the same idea reached on another route), EVIDENCE (an upstream measurement bears on a choice), NOT TAKEN
(a choice against it in these docs). "None" marks an idea no paper states. Markers as in the other docs.

## Summary

| Idea | Here | Cite as | Section |
|---|---|---|---|
| Pre-norm residual block with a final norm before the head | ADOPTED | 2002.04745, 1809.10853 | §1 |
| RMS normalization without a learnable gain | ADAPTED | 1910.07467, 2402.00838, 2501.00656 | §1 |
| RMS-normed token embedding, no positional embedding | ADOPTED | 2211.05100, 2110.02861 | §1 |
| Bias-free linear layers | ADOPTED | 2204.02311, 2402.00838 | §1 |
| Untied token embedding and output head | ADOPTED | 1608.05859, 2402.14905 | §1 |
| Rotary position embedding (RoPE), half-split layout | ADOPTED | 2104.09864 | §1 |
| RoPE base 10^6 from the first step | ADAPTED | 2405.14591, 2407.21783, 2410.06205 | §1 |
| QK-norm with learned per-dimension gains, fixed 1/√d scale | ADAPTED | 2302.05442, 2309.14322, 2410.01131 | §1 |
| Multi-head attention with all KV heads (not GQA/MQA) | ADOPTED | 1706.03762, 1911.02150, 2305.13245 | §1 |
| Head dimension 128 | ADOPTED | 2002.07028, 2001.08361 | §1 |
| Local–global attention: SSSL, last layer global, 512-key windows | ADAPTED | 2005.14165, 2004.05150, 2503.19786 | §1 |
| Document masking in packed sequences (varlen attention) | PARALLEL | 2402.13991, 2407.21783, 2410.02660 | §1 |
| Gated attention (output gate) | NOT TAKEN | 2505.06708 | §1 |
| Multi-token prediction | NOT TAKEN | 2404.19737 | §1 |
| ReLU² MLP, hidden 4C, not SwiGLU | ADOPTED | 2109.08668, 2002.05202 | §1 |
| Per-layer residual scalars and the x0 shortcut | ADAPTED | 2402.02622, 2410.17897 | §1 |
| Gated value embeddings on alternating layers | ADAPTED | 2601.22887 | §1 |
| Smear, backout, partial key offset | NOT TAKEN | none | §1 |
| Output-logit soft cap c·tanh(z/c), c = 15 | ADOPTED | 1611.09940, 2408.00118, 2405.09818, 2309.14322 | §1 |
| Uniform fan-in initialization, std C^−½ | ADAPTED | LeCun, Bottou, Orr & Müller, "Efficient BackProp" (1998, §4.6, Eqs. 15–16) | §1 |
| Zero-initialized residual-branch outputs and output head | ADOPTED | 1901.09321, 2203.03466 | §1 |
| Shape rule: width = depth × aspect, rounded up to whole 128-wide heads | ADAPTED | 2402.14905, 2001.08361 | §1 |
| Muon: orthogonalized momentum for hidden matrices | ADAPTED | 2409.20325, 2502.16982 | §2 |
| Newton–Schulz orthogonalization: quintic (3.4445, −4.7750, 2.0315), 5 bf16 steps, Gram form | ADOPTED | 2409.20325, 2505.16932 | §2 |
| Nesterov momentum 0.95 in EMA ("lerp") form | ADOPTED | 2502.16982 | §2 |
| Muon/AdamW parameter split | ADAPTED | 2409.20325, 2502.16982 | §2 |
| Muon step size per shape, matched to AdamW's update RMS | ADAPTED | 2502.16982 | §2 |
| NorMuon: neuron-wise second-moment normalization after orthogonalization | ADAPTED | 2510.05491 | §2 |
| Polar Express coefficients | NOT TAKEN | 2505.16932 | §2 |
| Cautious weight decay | NOT TAKEN | 2510.12402 | §2 |
| Decoupled weight decay (AdamW) and weight decay in Muon | ADOPTED | 1711.05101, 1412.6980, 2502.16982, 2401.02954 | §2 |
| Weight decay tied to the training horizon (the EMA timescale) | PARALLEL | 2505.13738, 2405.13698 | §2 |
| Per-group learning rates (AdamW groups, residual scalars) | NOT TAKEN | 2502.16982 | §2 |
| Batch size and peak learning rate from compute scaling laws | PARALLEL | 2401.02954, 2509.02046 | §2 |
| Global gradient-norm clipping at 1.0 | ADOPTED | 1211.5063, 2401.02954 | §2 |
| Linear warmup over 2/(1 − β₂) steps | ADOPTED | 1910.04209, 1706.02677 | §2 |
| Linear decay to zero, not a trapezoid with a floor | ADAPTED | 2502.15938, 2405.18392 | §2 |
| Each group's lr as its peak times one schedule multiplier | ADAPTED | none | §2 |
| Hyperparameter transfer from a small reference model | NOT TAKEN | 2203.03466, 2512.22382 | §2 |
| Training horizon from tokens per parameter | ADAPTED | 2203.15556 | §2 |
| Scaling-parameter count: decoder layers + lm_head | ADOPTED | 2001.08361, 2406.19146 | §2 |
| FLOPs per token: 6N plus attention over the attended keys | ADAPTED | 2204.02311, 2001.08361, 2401.02954 | §2 |
| Small-proxy tuning gains shrink with scale | EVIDENCE | 2509.02046, 2512.05620 | §2 |
| Byte-level BPE | ADOPTED | 1508.07909 | §3 |
| SA-BPE merge score in an incremental BPE trainer | ADAPTED | 2604.14053 | §3 |
| Vocabulary of 2^15 ids, special tokens inside it | ADOPTED | 2402.01035, 2402.19173, 2604.14053 | §3 |
| Pre-tokenization by a split regex, no line break after a punctuation run | ADAPTED | 2402.01035, 2608.18062 | §3 |
| Digit runs of one or two (`\p{N}{1,2}`) | ADOPTED | 2302.13971, 2402.14903 | §3 |
| BOS as the only document token: delimiter, end-of-document target, generation stop | ADAPTED | none | §3 |
| Tokenizer comparison by compression ratio | NOT TAKEN | 2402.18376 | §3 |
| BOS-aligned rows with best-fit-decreasing packing | ADAPTED | 2404.10830 | §3 |
| ClimbMix as the web-text source | ADOPTED | 2504.13161 | §3 |
| Bits per byte, zero-byte targets excluded | ADAPTED | 2101.00027 | §3 |
| Validation budget of 80 × 2^19 tokens | ADAPTED | none | §3 |
| Execution-based pass@k in a sandbox | PARALLEL | 2107.03374 | §3 |
| Truncated sampling: nucleus (top-p), not top-k | PARALLEL | 1904.09751 | §3 |
| FlashAttention kernels for training attention | ADAPTED | 2205.14135, 2407.08608, 2603.05451 | §4 |
| Sliding windows inside the attention kernel | ADAPTED | 2205.14135, 2412.05496 | §4 |
| Fixed-size `cu_seqlens`, with only the real segments passed to the kernel | ADAPTED | 2307.08691 | §4 |
| Mixed precision: fp32 master weights, bf16 GEMM operands with fp32 accumulation | ADAPTED | 1710.03740, 1905.12322 | §4 |
| Normalization statistics and the loss in fp32 | ADAPTED | 1710.03740, 2407.08608 | §4 |
| FP8 training | NOT TAKEN | 2209.05433, 2412.19437 | §4 |
| Fused linear cross entropy with the logit cap inside | PARALLEL | 2411.09009, 2410.10989 | §4 |
| Fused linear + ReLU² GEMM; activations stored for the backward | PARALLEL | 2104.04473, 2205.05198 | §4 |
| Fused QK-norm + RoPE kernel | PARALLEL | 2410.10989 | §4 |
| Merged q/k/v projection (one GEMM) | PARALLEL | none | §4 |
| Newton–Schulz batched over same-shape matrices | PARALLEL | 2504.05295, 2608.11612 | §4 |
| `torch.compile` of the whole model, static shapes, one graph | ADOPTED | PyTorch 2 (Ansel et al., ASPLOS 2024, not on arXiv, Abstract, §5, §5.1) | §4 |
| Fused AdamW step | ADOPTED | none | §4 |
| Meta-device construction, then explicit initialization | ADOPTED | 2304.11277 | §4 |
| Garbage-collector control | ADAPTED | none | §4 |
| Gradient accumulation weighted by valid tokens | ADAPTED | 2410.02660 | §4 |
| Overlapping host work with the device step | ADAPTED | none | §4 |
| Model FLOPs utilization (MFU) | ADAPTED | 2204.02311 | §4 |

## 1. Architecture

### Pre-norm residual block with a final norm before the head
- **Idea.** Normalize each sublayer's input, x + F(norm(x)), and the stream once more before the output layer. With
  the norm inside the residual branch the gradients at initialization are well scaled, so the warm-up stage that
  post-norm needs can be removed.
- **Papers.** arXiv 2002.04745 (Xiong et al., 2020, §3.3, Theorem 1, §4): the Pre-LN Transformer, with a final layer
  norm before the prediction; Post-LN's last-layer gradient is O(d√(ln d)) independent of depth L, Pre-LN's is scaled
  down by √L; Pre-LN trains without the warm-up stage; arXiv 1809.10853 (Baevski & Auli, 2018, §4.1): layer norm
  before the self-attention and FFN blocks "leads to more effective training".
- **Here.** ADOPTED: `model/modeling.py` `DecoderLayer.forward`, `CausalLM.hidden`; `docs/architecture.md` §1; the
  post-attention norm's statistics come from the o_proj GEMM epilogue (§3).
- **Upstream.** nanochat (initial commit); modded-nanogpt from its first script (`4a452a2`).

### RMS normalization without a learnable gain
- **Idea.** RMSNorm keeps LayerNorm's re-scaling and drops its re-centering, which it hypothesizes is dispensable, at
  lower cost; the non-parametric form also drops the gain.
- **Papers.** arXiv 1910.07467 (Zhang & Sennrich, 2019, Abstract, Eq. 4): RMSNorm matches LayerNorm with 7%–64% less
  running time (its form keeps a gain g); arXiv 2402.00838 (Groeneveld et al., 2024, §2.1): OLMo's non-parametric
  layer norm, without adaptive gain or bias, "the safest option" and the fastest; arXiv 2501.00656 (OLMo 2, 2024,
  §3.3.1): ablations "show no difference" between the non-parametric layer norm and RMSNorm.
- **Here.** ADAPTED: `model/modeling.py` `norm`; `docs/architecture.md` §1; eps FLT_EPSILON with fp32 statistics
  (upstream: bf16, torch's default eps).
- **Upstream.** nanochat (initial commit); modded-nanogpt first script (`4a452a2`), `F.rms_norm` from record #5.

### RMS-normed token embedding, no positional embedding
- **Idea.** x0 = norm(E[ids]): a norm right after the embedding lookup keeps the stream's scale independent of the
  table's; position enters only through RoPE.
- **Papers.** arXiv 2211.05100 (BLOOM, 2022, §3.2.3 "Embedding LayerNorm"): a layer norm right after the embedding
  "significantly improved training stability" at 104B in float16, though it "penalizes zero-shot generalization";
  bfloat16 may remove the need; arXiv 2110.02861 (Dettmers et al., 2021, §2.3): the stable embedding layer (layer norm
  on the embedding) avoids extreme gradient variation from the non-uniform token distribution.
- **Here.** ADOPTED: `model/modeling.py` `CausalLM.hidden`; `docs/architecture.md` §1; both papers use LayerNorm, here a
  gain-free RMS norm.
- **Upstream.** nanochat (initial commit); modded-nanogpt record #8.

### Bias-free linear layers
- **Idea.** Omit the additive bias in every projection.
- **Papers.** arXiv 2204.02311 (PaLM, 2022, §2 "No Biases"): "We found this to result in increased training stability
  for large models"; arXiv 2402.00838 (OLMo, 2024, §2.1): no bias terms, "to improve training stability".
- **Here.** ADOPTED: `model/modeling.py` `Attention`, `GemmLinear`, `CausalLM.lm_head`; `docs/architecture.md` §1.
- **Upstream.** nanochat (initial commit).

### Untied token embedding and output head
- **Idea.** Separate input and output matrices give the head capacity independent of the input table, at the cost of
  a second V × C matrix.
- **Papers.** arXiv 1608.05859 (Press & Wolf, 2016, Abstract, §1): tying the input and output embeddings improves the
  perplexity of several language models (the counter-position); arXiv 2402.14905 (MobileLLM, 2024, §2.2.3, Table 1):
  at 125M, sharing saves 16M parameters (11.8%) for a 0.2-point drop in average zero-shot accuracy.
- **Here.** ADOPTED: `model/modeling.py` `CausalLM.embed_tokens`, `CausalLM.lm_head`; `docs/architecture.md` §1; a
  design choice, the published evidence being mixed.
- **Upstream.** nanochat (initial commit); modded-nanogpt record #8.

### Rotary position embedding (RoPE), half-split layout
- **Idea.** Rotate each 2-D pair of q and k by t·θ_i with θ_i = base^(−2i/d): the q·k product depends only on the
  relative position and decays with distance.
- **Papers.** arXiv 2104.09864 (Su et al., 2021, §3.2.2 Eq. 15, §3.3, §3.4.3): RoPE, its frequencies and long-term
  decay.
- **Here.** ADOPTED: `model/kernels/qk_row.cuh` `qk_rotate`, `model/modeling.py` `CausalLM.init_buffers`;
  `docs/architecture.md` §1; pairs (i, i + d/2) rotated by −θ, a convention; base: next entry.
- **Upstream.** nanochat (initial commit); modded-nanogpt record #2.

### RoPE base 10^6 from the first step
- **Idea.** A larger base lowers the lowest frequencies, so attention can still discriminate distant tokens; the base
  needed grows with the context length.
- **Papers.** arXiv 2405.14591 (Men et al., 2024, §4.3 Table 2): the base's lower bound is 8.4e4 for an 8K context and
  6.4e5 for 32K; arXiv 2407.21783 (Llama 3, 2024, §3.2): base 500,000 in pre-training; arXiv 2410.06205 (Barbero et al.,
  2024, §6.1 Table 2): Gemma 2B at 8K, base 10k against 500k: perplexity 4.4627 → 4.4485 (Wiki), 6.4429 → 6.4593
  (FlanV2).
- **Here.** ADAPTED: `model/config.py` `ROPE_THETA`; `docs/architecture.md` §2; 10^6 in both stages (upstream 10^5).
- **Upstream.** nanochat initial commit (10^4), `6ed7d1d` (10^5); modded-nanogpt 10^4.

### QK-norm with learned per-dimension gains, fixed 1/√d scale
- **Idea.** Normalize q and k per head so attention logits cannot grow with parameter norms (near one-hot attention);
  a learned gain then sets the logit scale, so the softmax keeps 1/√d and needs no hand-set temperature.
- **Papers.** arXiv 2302.05442 (ViT-22B, 2023, §2, App. B): at ~8B, attention logits grow "over 50000 in magnitude"
  and training diverges; softmax[LN(XW_Q)LN(XW_K)ᵀ/√d] fixes it; arXiv 2309.14322 (Wortsman et al., 2023, §3.1.1,
  §3.2.5, Fig. E.8): the logit growth reproduces at small scale under high learning rates and qk-layernorm lowers the
  learning-rate sensitivity; per-head qk-layernorm with head-dim parameters shared across heads beats normalizing
  across the model dimension; arXiv 2410.01131 (nGPT, 2024, App. A.9 Table 5): at 0.5B and 1K context, fixing s_qk = 1
  instead of a learned vector costs +0.11% validation loss, a single learned scalar +0.22%.
- **Here.** ADAPTED: `model/modeling.py` `Attention.q_norm_weight`, `k_norm_weight`, `model/kernels/qk_prologue.cu`,
  `model/ops/flash_attn.py` `attention` (scale 1/√128); `docs/architecture.md` §1, §2; norm → gain → RoPE (upstream:
  RoPE → gain-free norm → ×1.2 on q and k).
- **Upstream.** nanochat initial commit (gain-free), `6ed7d1d` (×1.15), `a825e63` (×1.2); modded-nanogpt record #5
  (gain-free), record #20 (scale 0.12).

### Multi-head attention with all KV heads (not GQA/MQA)
- **Idea.** One key/value head per query head; MQA and GQA share KV heads to cut decode memory bandwidth and KV-cache
  size at some quality cost.
- **Papers.** arXiv 1706.03762 (Vaswani et al., 2017, §3.2.2): multi-head attention; arXiv 1911.02150 (Shazeer, 2019,
  Abstract): MQA decodes much faster "with only minor quality degradation"; arXiv 2305.13245 (Ainslie et al., 2023,
  Abstract): uptrained GQA reaches quality close to MHA at speed comparable to MQA.
- **Here.** ADOPTED: `model/modeling.py` `Attention` (C × C q/k/v); `docs/architecture.md` §2 (KV-cache arithmetic),
  `docs/experiments.md` §3 (d16 GQA screen, rejected).
- **Upstream.** nanochat (initial commit; supports GQA, unused).

### Head dimension 128
- **Idea.** Each head's attention has rank at most its head size, so the head size should not shrink as heads are
  added.
- **Papers.** arXiv 2002.07028 (Bhojanapalli et al., 2020, Abstract, Theorem 1): the coupling of head size to the
  number of heads causes a low-rank bottleneck; the head size should be set independently of the number of heads;
  arXiv 2001.08361 (Kaplan et al., 2020, §3.1, Fig. 5): at fixed non-embedding parameters the loss depends only weakly
  on shape, n_heads included.
- **Here.** ADOPTED: `model/config.py` `HEAD_DIM`; `docs/architecture.md` §2; no paper selects 128, which is also the
  kernels' compile-time head width.
- **Upstream.** nanochat LOG 2026-01-17 (128 better than 64); modded-nanogpt record #5 (6 heads of 128 at 768).

### Local–global attention: SSSL, last layer global, 512-key windows
- **Idea.** Most layers attend a fixed window of recent keys (linear cost, small KV cache); interleaved full-context
  layers carry long-range information, and stacked windows widen the receptive field.
- **Papers.** arXiv 2005.14165 (GPT-3, 2020, §2.1): "alternating dense and locally banded sparse attention patterns";
  arXiv 2004.05150 (Longformer, 2020, §3.1): sliding-window attention, receptive field l × w after l layers; arXiv
  2503.19786 (Gemma 3, 2025, §2; §5.2 Figs. 3–4): 5 local layers of 1,024 tokens per global one; changing the
  local:global ratio has "minimal impact on perplexity", and the window "can be reduced significantly without
  impacting perplexity".
- **Here.** ADAPTED: `model/modeling.py` `layer_windows`; `docs/architecture.md` §1, §2; the short window a fixed 512
  keys at 8K and 32K (upstream a quarter of the context, rounded up to 128).
- **Upstream.** nanochat LOG 2026-01-11 (SSSL, last layer full); modded-nanogpt record #20 (long-short windows, last
  layer long).

### Document masking in packed sequences (varlen attention)
- **Idea.** In a row packed with several documents, causal attention lets tokens attend to unrelated earlier
  documents; restricting each token to its own document removes that context, and varlen offsets (`cu_seqlens`) let
  the kernel skip the masked blocks.
- **Papers.** arXiv 2402.13991 (Zhao et al., 2024, §3.2 Table 1): 1.3B models on SlimPajama; intra-document causal
  masking gives average perplexity 8.410 against 9.293 for concatenation at 2K, 8.079 against 9.065 at 8K; arXiv
  2407.21783 (Llama 3, 2024, §3.2): the document mask had "limited impact" in standard pre-training and mattered in
  continued pre-training on very long sequences; arXiv 2410.02660 (Gao et al., 2024, ProLong, App. B.2 Table 20):
  document masks improve both long- and short-context results in continued long-context training.
- **Here.** PARALLEL: `model/ops/flash_attn.py` `attention`, `corpus/loader.py` `PretrainingBatches`;
  `docs/architecture.md` §1, `docs/corpus.md` §8–§9; FA4 varlen, one segment per packed piece of a document.
- **Upstream.** nanochat LOG 2026-01-13 (d16 at 2K: val bpb 0.85427 → 0.85407, judged not worth the code, not
  merged); modded-nanogpt records #12 (FlexAttention), #29 (FA3 varlen).

### Gated attention (output gate)
- **Idea.** A per-head sigmoid gate on the attention output adds non-linearity and query-dependent sparsity.
- **Papers.** arXiv 2505.06708 (Qiu et al., 2025, §1, Table 1): a head-specific sigmoid gate after SDPA gives up to
  0.2 lower perplexity and 2 MMLU points (15B MoE models, 400B tokens) and removes the attention sink.
- **Here.** NOT TAKEN: `docs/experiments.md` §3 (listed as not run).
- **Upstream.** nanochat LOG 2026-01-17 (no improvement, +1 GB); modded-nanogpt record #28 (sparse attention gate).

### Multi-token prediction
- **Idea.** Extra heads predict tokens t+2…t+n as an auxiliary loss.
- **Papers.** arXiv 2404.19737 (Gloeckle et al., 2024, §3.1 Fig. 3): on code (0.3B–13B; MBPP, HumanEval), n-token
  models "are worse than the baseline for small model sizes, but outperform the baseline at scale".
- **Here.** NOT TAKEN: `docs/experiments.md` §3.
- **Upstream.** nanochat LOG 2026-01-12 (d12: more memory, per step equal or slightly worse); modded-nanogpt record #53.

### ReLU² MLP, hidden 4C, not SwiGLU
- **Idea.** relu(x)² keeps ReLU's sparsity with a growing positive side; it equals ReGLU with tied matrices, so it
  captures much of the GLU gain without a third matrix.
- **Papers.** arXiv 2109.08668 (So et al., 2021, Primer, §3, Fig. 5): squared ReLU is Primer's most effective
  modification, better than GELU, Swish and the GLU variants in their T5 C4 runs; arXiv 2002.05202 (Shazeer, 2020,
  Table 1): at matched parameters and compute, SwiGLU 1.636 against ReLU 1.677 log-perplexity (T5 span filling on C4),
  the counter-evidence.
- **Here.** ADOPTED: `model/ops/linear.py` `relu2_mlp`; `docs/architecture.md` §1, §2 ("ReLU², not SwiGLU" rests on
  the upstream LOG result).
- **Upstream.** nanochat LOG 2026-02-05 (SwiGLU at matched parameters and FLOPs worse at d12 and d24); modded-nanogpt
  record #5.

### Per-layer residual scalars and the x0 shortcut
- **Idea.** Before layer i, x ← λr_i·x + λ0_i·x0 with x0 the normed embedding; λr = 1, λ0 = 0 at init, so training
  starts from the plain residual stack and learns how much token-level information to re-inject at depth.
- **Papers.** arXiv 2402.02622 (Pagliardini et al., 2024, DenseFormer, §3, Fig. 5): learned weighted averages over
  the embedding and all earlier block outputs, initialized to the identity (the plain Transformer), reach the
  perplexity of deeper Transformers; arXiv 2410.17897 (Zhou et al., 2024, ResFormer, Abstract): hidden-state residuals
  "fail to adequately preserve initial token-level information in deeper layers".
- **Here.** ADAPTED: `model/modeling.py` `CausalLM.resid_lambdas`, `x0_lambdas`, `hidden`, `init_weights`;
  `docs/architecture.md` §1; init 1 / 0 (92d63d4: decaying inits), one shared lr.
- **Upstream.** nanochat LOG 2026-01-11 (d8–d20: −0.0103 … −0.0039 bpb); modded-nanogpt record #9 (README: "partially
  following" arXiv 2410.17897).

### Gated value embeddings on alternating layers
- **Idea.** Per-layer token-indexed tables added to the attention values, v ← v + 2σ(x W_g)·E_l[ids] per head: many
  parameters at almost no FLOPs; zero-initialized W_g starts the gate at 1, which it can raise or lower within (0, 2).
- **Papers.** arXiv 2601.22887 (MoVE, 2026, §3.2 Eq. 3; §4.1 Eq. 6; §4.2 Table 1): LaVE, "adapted from the
  modded-nanoGPT repository", V + g ⊙ M_t with g = 2σ(x W_g) read from the full hidden width, on half the layers
  (L−1, L−3, …); the 2σ gate is 1.0 at z = 0, "a neutral identity mixing"; LaVE ×1 lowers val bpb by 0.016, 0.010 and
  0.010 at D12, D20 and D32 (nanochat models).
- **Here.** ADAPTED: `model/modeling.py` `has_ve`, `CausalLM.value_embeds`, `Attention.ve_gate`,
  `model/kernels/ve_gate.cu`; `docs/architecture.md` §1, §2; the gate reads the full width (nanochat `e85db6b`: 32
  channels).
- **Upstream.** nanochat LOG 2026-01-17 (alternating best, gate helps), `e85db6b`; modded-nanogpt records #14 (tables),
  #55 (2σ zero-init gate).

### Smear, backout, partial key offset
- **Idea.** Speedrun stream edits: mix each token's embedding with its predecessor's (smear), subtract a mid-layer
  residual before the head (backout), shift keys one position on part of the head dimensions (partial key offset).
- **Papers.** None: speedrun techniques without a paper; the upstream results are unreviewed observations.
- **Here.** NOT TAKEN: `docs/architecture.md` §2 (smear, backout), `docs/experiments.md` §3 (partial key offset).
- **Upstream.** nanochat LOG 2026-01-16 (smear negligible, backout no improvement), later adopted (`a825e63`);
  modded-nanogpt records #34 (smear), #40 (backout), #49 (partial key offset).

### Output-logit soft cap c·tanh(z/c), c = 15
- **Idea.** Bound the output logits smoothly to (−c, c) so they cannot drift; QK-norm bounds only the attention logits.
- **Papers.** arXiv 1611.09940 (Bello et al., 2016, §5.1; App. A.2 Eq. 16): logit clipping softmax(C tanh(u)), C = 10,
  "helps with exploration"; arXiv 2408.00118 (Gemma 2, 2024, §2): soft_cap·tanh(logits/soft_cap), 50 in attention and
  30 on the final layer; arXiv 2405.09818 (Chameleon, 2024, §2.3): QK-norm "does not solve the problem of logit shift
  in the final softmax"; arXiv 2309.14322 (Wortsman et al., 2023, §3.1.2): output-logit divergence occurs without
  weight decay, mitigated by z-loss or weight decay.
- **Here.** ADOPTED: `model/modeling.py` `FINAL_LOGIT_SOFTCAPPING`, `model/ops/cross_entropy.py` `lm_loss`,
  `lm_losses`; `docs/architecture.md` §2, `docs/experiments.md` §3 (designed ablation, not run); c = 15 has no paper.
- **Upstream.** nanochat initial commit (15), LOG 2026-03-02 (d24: 5 much worse, 5–30 about equal, 20 best by ~1e-3),
  `6ed7d1d` (back to 15); modded-nanogpt records #9 (30), #18 (15).

### Uniform fan-in initialization, std C^−½
- **Idea.** Draw each weight with mean 0 and standard deviation m^−½ (m the fan-in) so pre-activations keep unit
  variance; uniform on ±√(3/m) has that standard deviation.
- **Papers.** LeCun, Bottou, Orr & Müller, "Efficient BackProp" (1998, §4.6, Eqs. 15–16): weights drawn from a
  distribution (e.g. uniform) with mean zero and standard deviation m^−½.
- **Here.** ADAPTED: `model/modeling.py` `CausalLM.init_weights`; `docs/architecture.md` §1; upstream's bound for
  q/k/v and the value tables, applied also to up_proj and the token table (92d63d4: ×0.4, N(0, 0.8)).
- **Upstream.** nanochat `48abd7d` (uniform, "to avoid outliers"); modded-nanogpt record #19 (std 0.5·C^−½).

### Zero-initialized residual-branch outputs and output head
- **Idea.** Zeroing the last layer of each residual branch starts every block as the identity; a zero head starts
  from uniform predictions.
- **Papers.** arXiv 1901.09321 (Zhang et al., 2019, Fixup, §3): "Initialize the classification layer and the last
  layer of each residual branch to 0"; arXiv 2203.03466 (Yang et al., 2022, App. D.2): a zero output layer makes the
  optimal hyperparameters of small and large widths "match more closely", without harming performance.
- **Here.** ADOPTED: `model/modeling.py` `CausalLM.init_weights` (zero `o_proj`, `down_proj`, `lm_head`);
  `docs/architecture.md` §1.
- **Upstream.** nanochat initial commit (zero head later N(0, 0.001), `48abd7d`); modded-nanogpt records #5
  (projections), #8 (head).

### Shape rule: width = depth × aspect, rounded up to whole 128-wide heads
- **Idea.** Fixing the width-to-depth ratio lets one integer (layers) set the model; sub-billion models do better
  deep and thin.
- **Papers.** arXiv 2402.14905 (MobileLLM, 2024, §2.2.2, Fig. 4; Table 9): at ~125M and ~350M "deeper and thinner
  models outperform their shallower and wider counterparts"; MobileLLM-350M is 32 layers × 960 (ratio 30); arXiv
  2001.08361 (Kaplan et al., 2020, §3.1, Fig. 5): at fixed non-embedding parameters the aspect ratio can vary by a
  factor of 40 while only slightly impacting performance.
- **Here.** ADAPTED: `model/modeling.py` `ASPECT_RATIO`, `meta_model`; `docs/architecture.md` §1, §2; aspect 30
  (upstream 64), rounding rule unchanged.
- **Upstream.** nanochat LOG 2026-01-17 (aspect 128 worse than 64: "slightly thinner and longer").

## 2. Optimizer, schedule and scaling

### Muon: orthogonalized momentum for hidden matrices
- **Idea.** Replace the SGD-momentum update of each 2-D hidden matrix by its nearest semi-orthogonal matrix UVᵀ, which
  is steepest descent under the spectral norm; raw momentum updates are nearly low-rank, and orthogonalizing them
  raises the scale of their rare directions.
- **Papers.** Jordan et al. (2024, "Muon: An optimizer for hidden layers in neural networks", blog, non-archival,
  §Definition, §"Why is it good to orthogonalize the update?"): the algorithm and the low-rank observation;
  arXiv 2409.20325 (Bernstein, 2024, Props. 4–5): UVᵀ is the closest semi-orthogonal matrix in Frobenius norm, and
  Shampoo without accumulation is steepest descent under the max spectral norm over layers; arXiv 2502.16982 (Liu,
  2025, §2.2, §3.2): with weight decay and RMS-matched steps, Muon needs about 52% of AdamW's training FLOPs under
  compute-optimal training.
- **Here.** ADAPTED: `train/optim.py` `Muon`; `docs/training.md` §3; built on `torch.optim.Muon`, with NorMuon,
  Moonlight's step size, one lr for all groups and the τ-law weight decay.
- **Upstream.** modded-nanogpt records #3 and #4 (class `Muon`); nanochat; the code here derives from
  `torch.optim.Muon`, which credits KellerJordan/Muon, not from nanochat's optimizer.

### Newton–Schulz orthogonalization: quintic (3.4445, −4.7750, 2.0315), 5 bf16 steps, Gram form
- **Idea.** Approximate UVᵀ without an SVD by iterating an odd quintic X ← aX + (bA + cA²)X, A = XXᵀ, from
  X₀ = G/‖G‖_F: each step applies the polynomial to every singular value with matrix products only, which run in
  bf16; the coefficients maximize the slope at 0 subject to the iterates staying in [0.7, 1.3], so 5 steps suffice.
- **Papers.** Jordan et al. (2024, blog, non-archival, §Definition, §"Tuning the coefficients"): the coefficients, the
  Gram-form code, and "it suffices to run the NS iteration for only 5 steps"; arXiv 2409.20325 (Bernstein, 2024,
  App. A item 4, Eqs. 36–37): the Newton–Schulz iteration for UVᵀ and its odd-polynomial family, classical for fixed
  coefficients (Kovarik 1970; Björck & Bowie 1971); arXiv 2505.16932 (Amsel, 2025, §1.2): Jordan's polynomial does
  not converge but plateaus at error ≈ 0.3, reached rapidly, and beats plain Newton–Schulz below about 10 iterations.
- **Here.** ADOPTED: `train/optim.py` `Muon.step`; `docs/training.md` §3; PyTorch's defaults (coefficients, 5 steps,
  eps 1e-7), batched over same-shape matrices with `baddbmm`.
- **Upstream.** modded-nanogpt record #3 (coefficients, 5 bf16 steps) and record #11 (Gram form); nanochat, which
  replaced the coefficients with Polar Express (LOG 2026-01-10).

### Nesterov momentum 0.95 in EMA ("lerp") form
- **Idea.** Orthogonalize the Nesterov look-ahead of the momentum buffer; keeping the buffer as an EMA,
  buf ← β·buf + (1 − β)·g, only rescales classical momentum by (1 − β), which orthogonalization removes.
- **Papers.** Sutskever et al. (2013, ICML, non-arXiv, §2): Nesterov's accelerated gradient as a momentum method;
  Jordan et al. (2024, blog, non-archival, §"Empirical considerations"; §"Proving that NS iteration orthogonalizes the
  update"): Nesterov-style momentum "works a bit better than normal SGD-momentum in every case we have tested", and
  Ortho(cG) = Ortho(G); arXiv 2502.16982 (Liu, 2025, §2.2 "Other Hyper-parameters"): no consistent gain from tuning
  momentum, 0.95 used.
- **Here.** ADOPTED: `train/optim.py` `Muon.step` (`torch._foreach_lerp_`); `docs/training.md` §3; `momentum` 0.95,
  PyTorch's default, held constant.
- **Upstream.** modded-nanogpt records #3 and #4 (Nesterov, 0.95) and #15 (lerp form); nanochat.

### Muon/AdamW parameter split
- **Idea.** Muon only for 2-D hidden matrices; embeddings, the output head and 1-D parameters on AdamW, because these
  layers call for different norms (linear layers RMS→RMS, a rescaled spectral norm; embeddings ℓ1→RMS).
- **Papers.** Jordan et al. (2024, blog, non-archival, §Definition): scalar and vector parameters and the input and
  output layers "should be optimized by a standard method such as AdamW"; arXiv 2409.20325 (Bernstein, 2024,
  Epilogue): the RMS→RMS operator norm for linear layers and ℓ1→RMS for embeddings; arXiv 2502.16982 (Liu, 2025,
  §2.2 "Matching update RMS of AdamW"): AdamW for "RMSNorm, LM head, and embedding parameters".
- **Here.** ADAPTED: `train/optim.py` `optimizers`; `docs/training.md` §3; value-embedding tables on AdamW, `ve_gate` on
  Muon, one lr for all groups, 1-D parameters undecayed (contrary to arXiv 2502.16982 App. D on RMSNorm gains).
- **Upstream.** modded-nanogpt records #3, #8 (separate Adams for the untied embedding and head) and #9 (split by
  ndim); nanochat (value-embedding tables on AdamW).

### Muon step size per shape, matched to AdamW's update RMS
- **Idea.** An orthogonal A×B update has RMS √(1/max(A, B)), so its effective step depends on shape; scaling it by
  0.2·√max(A, B) gives every matrix AdamW's typical update RMS, so Muon and AdamW share one lr and one weight decay.
- **Papers.** arXiv 2502.16982 (Liu, 2025, §2.2 Lemma 1, Eq. 4): the update RMS and the 0.2·√max(A, B) adjustment,
  which lets matrix and non-matrix parameters share "learning rate η, weight decay λ".
- **Here.** ADAPTED: `train/optim.py` `optimizers` (`adjust_lr_fn="match_rms_adamw"`); `docs/training.md` §3;
  Moonlight's rule replaces upstream's √max(1, rows/cols).
- **Upstream.** modded-nanogpt record #4 (√max(rows, cols)), then √max(1, rows/cols) (`4706781`); nanochat.

### NorMuon: neuron-wise second-moment normalization after orthogonalization
- **Idea.** Orthogonalized updates have uniform singular values but uneven per-neuron norms; dividing each row of the
  update by the root of an EMA of its mean square, then restoring the update's scale, adds per-neuron adaptive rates.
- **Papers.** arXiv 2510.05491 (Li, 2025, §3.1 Alg. 1; §4.1.1; §4.1.2 Table 1; §4.1.4 Table 2): the method, β₂ = 0.95;
  Adam's final loss reached in 21.74% / 13.91% fewer steps at 1.1B / 5.4B (Muon 10.43% / 6.08%); +2.9% step time over
  AdamW at 5.4B.
- **Here.** ADAPTED: `train/optim.py` `Muon.step` (`NORMUON_BETA2`, `NORMUON_EPS`); `docs/training.md` §3; always per
  row, rescaled to the orthogonalized update's Frobenius norm; adopted on the paper (a one-seed d16 screen does not
  resolve it).
- **Upstream.** modded-nanogpt record #41 (PR #144), axis fix record #48, β₂ 0.9 record #78; nanochat LOG 2026-01-10
  ("a very small improvement", kept).

### Polar Express coefficients
- **Idea.** Replace the fixed quintic by a per-iteration sequence of minimax-optimal odd quintics, each optimal for the
  current singular-value interval, so the iteration is fast early and still converges; a safety factor guards bf16.
- **Papers.** arXiv 2505.16932 (Amsel, 2025, §1.3, §4.1 Thm. 4.1, §4.4, Alg. 1; §5.2 Figs. 1, 6): the method; inside
  Muon on GPT-2 Small and Large (1B FineWeb tokens), lower validation loss than Jordan's and You's coefficients at
  every learning rate.
- **Here.** NOT TAKEN: `train/optim.py` `Muon.step` (PyTorch's fixed coefficients); `docs/experiments.md` §4; one-seed
  d16 screen, its gain attributable to its larger update [inference], −0.35% tok/s.
- **Upstream.** modded-nanogpt record #38 (PR #134); nanochat LOG 2026-01-10 ("No dramatic/noticeable difference").

### Cautious weight decay
- **Idea.** Apply decoupled decay only to coordinates where the update and the parameter share a sign; the method then
  optimizes the unregularized loss and, on the stationary manifold, slides toward locally Pareto-minimal magnitudes.
- **Papers.** arXiv 2510.12402 (Chen, 2025, §3.1–3.2; §4 Figs. 1, 4, 5): the stability and sliding-mode analysis; lower
  final loss for AdamW, Lion and Muon at 338M–2B on C4 at 20 TPP and for OLMo-1B on 100B tokens, with the optimal λ
  essentially unchanged (Fig. 1).
- **Here.** NOT TAKEN: `train/optim.py` `Muon.step` (decay on every coordinate); `docs/experiments.md` §4; one-seed d16
  screen, worse at the end.
- **Upstream.** modded-nanogpt records #43 (PR #154) and #50 (Adam); nanochat LOG 2026-01-10 ("Solid improvements";
  it cites arXiv 2411.16085, an update mask, not this decay mask).

### Decoupled weight decay (AdamW) and weight decay in Muon
- **Idea.** Shrink weights by η_t·λ·θ apart from the adaptive step, so the decay is not divided by Adam's second
  moment; Muon needs the same decay at scale to keep weight and output RMS bounded.
- **Papers.** arXiv 1711.05101 (Loshchilov, 2017, §2 Alg. 2, Prop. 2): decoupled decay, scaled by the schedule
  multiplier, differs from L2 regularization for adaptive methods; arXiv 1412.6980 (Kingma, 2014, Alg. 1): Adam;
  arXiv 2502.16982 (Liu, 2025, §2.2 "Weight Decay", Fig. 2): without decay Muon's weight and output RMS grow past
  bf16's high-precision range and its early lead fades (800M, 100B tokens); arXiv 2401.02954 (DeepSeek-AI, 2024,
  §2.3): AdamW β = (0.9, 0.95).
- **Here.** ADOPTED: `train/optim.py` `optimizers`, `BETAS`; `docs/training.md` §3; p ← p·(1 − lr·λ) in both
  optimizers, λ from the τ law.
- **Upstream.** modded-nanogpt track-2 record #3 (Muon decay); nanochat (citing arXiv 1711.05101).

### Weight decay tied to the training horizon (the EMA timescale)
- **Idea.** With decoupled decay the weights are an EMA of recent updates over τ_iter = 1/(ηλ) steps; choosing λ so
  that this timescale is a set fraction of the run ties the decay to the training horizon.
- **Papers.** arXiv 2505.13738 (Bergsma, 2025, Finding 3; App. E.3.1; App. B): in single-epoch pre-training the
  optimal timescale, as a fraction of the run, falls as a power law in tokens per parameter (R² = 0.975; fit
  c_τ = 1.084, m_τ = −0.527; TPP 20–1280), fit on AdamW, the EMA view applying to other decoupled-decay optimizers;
  arXiv 2405.13698 (Wang, 2024, §3.1 Eq. 5, §4): AdamW as an EMA with 1/τ_iter = ηλ, and an optimal timescale in
  epochs roughly constant across model and dataset size (multi-epoch).
- **Here.** PARALLEL: `train/optim.py` `tau_weight_decay`; `docs/training.md` §3; one law for the AdamW tables and Muon
  alike (Muon's timescale runs: `docs/experiments.md` §4).
- **Upstream.** nanochat LOG 2026-01-10 (λ ∝ 1/depth²), then λ = λ_ref·√(B/B_ref)·(D_ref/D) after arXiv 2405.13698
  (`aeff095`), Muon only.

### Per-group learning rates (AdamW groups, residual scalars)
- **Idea.** Separate lrs, betas, eps and decay per AdamW group (token table, head, value tables, scalars, scaled by
  (d_model/768)^−½), the residual and x0 scalars at 0.005 and 0.5 × √(B/2¹⁹); the alternative is one lr and one λ.
- **Papers.** arXiv 2502.16982 (Liu, 2025, §2.2 "Matching update RMS of AdamW"): with RMS-matched Muon, "learning rate
  η, weight decay λ" are shared among matrix and non-matrix parameters; upstream's per-group values are empirical
  tuning.
- **Here.** NOT TAKEN: `train/optim.py` `optimizers`; `docs/training.md` §3, `docs/experiments.md` §4; `adamw_group_lr`
  designed, not run; `scalar_learning_rates` one d16 run, not adopted.
- **Upstream.** nanochat LOG 2026-01-11 ("Per-Layer Residual Scalars") and 2026-01-19..22 (optimizer sweep).

### Batch size and peak learning rate from compute scaling laws
- **Idea.** The loss-optimal batch grows and the optimal lr falls with training compute; read both off fitted power
  laws at the run's compute C = M·D.
- **Papers.** arXiv 2401.02954 (DeepSeek-AI, 2024, §3.1 Eq. 1): η_opt = 0.3118·C^−0.1250, B_opt = 0.2920·C^0.3271,
  fit on 1e17–2e19 FLOPs under a multi-step schedule; arXiv 2509.02046 (Wen, 2025, App. C.9 Table 85): Muon's loss is
  flat across a 2× batch (130M at 8× Chinchilla: batch 128 and 256 both 3.240, 512 3.257).
- **Here.** PARALLEL: `train/schedule.py` `plan`; `docs/training.md` §2, §5; B in whole micro-batches and not ablated;
  the law's lr times a multiplier measured at d24.
- **Upstream.** nanochat LOG 2026-02-05 ("Auto Batch Size Scaling": B = 2¹⁹·(D/D_ref)^0.383 rounded to a power of 2,
  citing arXiv 2505.13738).

### Global gradient-norm clipping at 1.0
- **Idea.** Rescale the whole gradient to norm 1 when it exceeds it, bounding the step on rare exploding-gradient
  events without changing ordinary steps.
- **Papers.** arXiv 1211.5063 (Pascanu, 2012, §3.2 Alg. 1): norm clipping against exploding gradients;
  arXiv 2401.02954 (DeepSeek-AI, 2024, §2.3): gradient clipping set to 1.0.
- **Here.** ADOPTED: `train/optim.py` `update` (`MAX_GRAD_NORM`); `docs/training.md` §3; one global norm over both
  optimizers' gradients; upstream's removal not followed.
- **Upstream.** nanochat clipped at 1.0, then removed it (LOG 2026-01-08: no benefit at d12/d20, norm never above 1.0,
  ~2% MFU); modded-nanogpt removed it before record #2 (`9805cab`).

### Linear warmup over 2/(1 − β₂) steps
- **Idea.** Ramp the lr linearly at the start, when Adam's second-moment estimate is unreliable; an untuned ramp over
  2/(1 − β₂) steps (40 at β₂ = 0.95) performs like a rectified (RAdam) schedule.
- **Papers.** arXiv 1910.04209 (Ma, 2019, §4.3 Eq. 14): "linear warmup over τ = 2·(1−β2)^−1 iterations" as an untuned
  rule of thumb; arXiv 1706.02677 (Goyal, 2017, §2.2): gradual linear warmup.
- **Here.** ADOPTED: `train/schedule.py` `WARMUP_STEPS`, `lr_lambda`; `docs/training.md` §4; step s at (s + 1)/40.
- **Upstream.** nanochat ((s + 1)/W form; 40 steps since `6ed7d1d`, tuned at d12).

### Linear decay to zero, not a trapezoid with a floor
- **Idea.** Decay the lr linearly from the peak to zero over the whole run: a vanishing lr averages more updates late
  in training, which matters more as tokens per parameter grow; the upstream alternative is a constant phase, a
  cooldown (warmup–stable–decay) and a nonzero floor.
- **Papers.** arXiv 2502.15938 (Bergsma, 2025, §4.2 Figs. 6, 8): linear decay to zero beats 10× decay by a gap that
  grows with TPP (111M–1.7B) and is "around 0.84% better than the optimal WSD" (610M, 80 TPP); arXiv 2405.18392
  (Hägele, 2024, Abstract): a constant lr with a cooldown scales "predictably and reliably similar to cosine".
- **Here.** ADAPTED: `train/schedule.py` `lr_lambda`; `docs/training.md` §4; decay from step 40 to 1/(steps − 40) of
  peak at the last step, through the repository stage.
- **Upstream.** modded-nanogpt trapezoid (`93fb178`, citing arXiv 2405.18392), floors 0.1 / 0.05 / 0.15 (records #19,
  #26, #72); nanochat warmdown to 0 over the last 20%, later over 65% to a 5% floor.

### Each group's lr as its peak times one schedule multiplier
- **Idea.** Store each group's peak lr and multiply it by one function of the step, so every group follows one
  schedule and keeps its ratio to the others.
- **Papers.** None: engineering detail.
- **Here.** ADAPTED: `train/loop.py` `LambdaLR`; `docs/training.md` §4; PyTorch's `LambdaLR` over `lr_lambda`, one per
  optimizer.
- **Upstream.** nanochat (`initial_lr` × multiplier).

### Hyperparameter transfer from a small reference model
- **Idea.** Tune at a small reference size and carry lr, batch and decay to larger models by scaling rules; the optima
  transfer only under a parameterization that makes them size-independent.
- **Papers.** arXiv 2203.03466 (Yang, 2022, Fig. 1, Table 3): under μP the optimal lr is stable with width, under
  standard parameterization it is not; arXiv 2512.22382 (Mlodozeniec, 2025, Title, Abstract): transfer across
  modules, width, depth, batch and training duration.
- **Here.** NOT TAKEN: `ablations/peak_lr.py` `lr_multiplier`; `docs/training.md` §5; standard parameterization, the lr
  multiplier measured at d24 itself.
- **Upstream.** nanochat LOG 2026-02-05 (d12 reference, "muP style").

### Training horizon from tokens per parameter
- **Idea.** Set the token budget D = r·N for a ratio r and steps = D // B; at the compute-optimal frontier parameters
  and tokens grow in equal proportion, about 20 tokens per parameter.
- **Papers.** arXiv 2203.15556 (Hoffmann, 2022, Abstract; §3.4 Table 3): tokens doubled with model size; 1B parameters
  ↔ 20.2B tokens.
- **Here.** ADAPTED: `train/schedule.py` `plan`; `docs/training.md` §2; r is data-bound (`train/stages.py`
  `max_tokens_per_param`, ≈528), not compute-optimal.
- **Upstream.** nanochat (ratio 20, later 8, 4, 10.5 and 12; LOG 2026-01-17, 2026-01-27).

### Scaling-parameter count: decoder layers + lm_head
- **Idea.** Count the parameters that scale compute, the decoder layers and the output head; lookup tables (token and
  value embeddings) cost no matmul FLOPs and are excluded.
- **Papers.** arXiv 2001.08361 (Kaplan, 2020, §1.3, §2.1): N excludes "all vocabulary and positional embeddings";
  arXiv 2406.19146 (Porian, 2024, §3.2): leaving out the head under-counts FLOPs by roughly 10% (large) to 90% (small
  models), so counting it is "significantly more accurate".
- **Here.** ADOPTED: `model/modeling.py` `num_non_embedding_params`; `docs/architecture.md` §1, `docs/training.md` §2;
  the layers' 1-D gains and `ve_gate` counted.
- **Upstream.** nanochat LOG 2026-01-27 (transformer matrices + lm_head).

### FLOPs per token: 6N plus attention over the attended keys
- **Idea.** Training costs 6 FLOPs per matmul weight per token (2 forward, 4 backward) plus attention's QKᵀ and PV,
  12·L·H·Q·T for context T; a plan read off a fitted law must count FLOPs as the law was fit.
- **Papers.** arXiv 2204.02311 (Chowdhery, 2022, App. B): 6N + 12LHQT FLOPs per token and MFU; arXiv 2001.08361
  (Kaplan, 2020, §2.1 Eq. 2.2, Table 1): C_forward ≈ 2N + 2·n_layer·n_ctx·d_attn; arXiv 2401.02954 (DeepSeek-AI, 2024,
  §3.2 Eq. 2): M = 72·n_layer·d_model² + 12·n_layer·d_model·l_seq, attention included, the scale its laws use.
- **Here.** ADAPTED: `metrics.py` `model_flops_per_token`, `executed_model_flops`; `docs/evaluation.md` §5,
  `docs/training.md` §2; per layer T = min(window + 1, context); the plan uses M (`model/modeling.py`
  `non_embedding_flops_per_token`, every layer at full context).
- **Upstream.** nanochat LOG 2026-01-11 (`estimate_flops` summing min(window, T) per layer, N over every linear weight).

### Small-proxy tuning gains shrink with scale
- **Idea.** Gains tuned on a small proxy shrink, or reverse, at larger scale, so small-proxy hyperparameter tuning is
  distrusted.
- **Papers.** arXiv 2509.02046 (Wen, 2025, §1 Fig. 1): an under-tuned AdamW baseline inflates speedups (up to 2× from
  the lr alone at 100M), and optimizer speedups fall from 1.3–1.4× below 1B to 1.1× at 1.2B; arXiv 2512.05620 (Qiu,
  2025, Abstract): with μP and 1/width weight decay, Muon keeps about 1.4× over AdamW from 190M to 1.4B, and the
  speedup vanishes under incorrect scaling.
- **Here.** EVIDENCE: `docs/experiments.md` §1 ("Proxy settings").
- **Upstream.** nanochat LOG 2026-01-19..22 (~320 runs; gains ~0.002 bpb at d12, ~0.0007 at d20; d12-tuned values
  "actively hurt" at d20).

## 3. Tokenizer, data and evaluation

### Byte-level BPE
- **Idea.** BPE builds a subword vocabulary by repeatedly merging the most frequent adjacent symbol pair, so any string
  is encoded without unknown symbols; run on UTF-8 bytes, its base vocabulary is the 256 byte values.
- **Papers.** arXiv 1508.07909 (Sennrich, 2015, §3.2): BPE for word segmentation, "iteratively count all symbol pairs
  and replace each occurrence of the most frequent pair"; final vocabulary = initial vocabulary + merges, the merges
  the only hyperparameter. Radford et al., 2019, "Language Models are Unsupervised Multitask Learners" (OpenAI
  report, non-archival, §2.2): byte-level BPE needs a base vocabulary of 256 against over 130,000 for code points.
- **Here.** ADOPTED: `corpus/crate/train.rs` `train`, `corpus/tokenizer.py` `Tokenizer`; `docs/corpus.md` §5.
- **Upstream.** nanochat (rustbpe trainer over bytes, "in the style of GPT-4").

### SA-BPE merge score in an incremental BPE trainer
- **Idea.** Merge the pair of highest F·log R (F its frequency, R the number of repositories holding it) instead of
  highest F, so pairs frequent inside one source do not become rarely seen, under-trained tokens.
- **Papers.** arXiv 2604.14053 (Chizhov, 2026, §4.1 "Priority criterion", Table 3): F·log R gives compression 3.643
  and 72 verified under-trained tokens against BPE's (F) 3.623 and 640, of 646 tested.
- **Here.** ADAPTED: `corpus/crate/train.rs` `train`, `Word::merge_pair`, `impl Ord for Job`; `corpus/tokenizer.py`
  `Tokenizer.enc`; `docs/corpus.md` §5; R counts authorship units, ties go to higher F then the smaller pair, a merge
  whose bytes are already a token is skipped, encoding is tiktoken's.
- **Upstream.** nanochat (rustbpe's incremental trainer, its smaller-pair tie-break; trained by rustbpe, encoded by
  tiktoken).

### Vocabulary of 2^15 ids, special tokens inside it
- **Idea.** A larger vocabulary compresses more but adds embedding and output parameters and rarer, less-trained
  tokens; for code, sizes above 32K have shown no gain.
- **Papers.** arXiv 2402.01035 (Dagan, 2024, §3.2, Table 3): an NL-pretrained 1.5B model given GPT-4-style tokenizers
  of 32K–256K, vocabulary size vs HumanEval pass@1 Pearson r = −0.13, p = 0.87; arXiv 2402.19173 (StarCoder2, 2024,
  §6.2): "increasing the vocabulary size to 100K did not improve performance" (kept 49,152); arXiv 2604.14053
  (Chizhov, 2026, App. B, Table 5): SA-BPE's code models use 32,768.
- **Here.** ADOPTED: `model/config.py` `VOCAB_SIZE`, `corpus/train_tokenizer.py` (32,508 merges = 2^15 − 256 − 4
  specials); `docs/corpus.md` §5; the value itself is a design choice.
- **Upstream.** nanochat (65,536 → 32,768 = 2^15, `scripts/tok_train.py`; specials counted inside the size).

### Pre-tokenization by a split regex, no line break after a punctuation run
- **Idea.** A regex cuts text into chunks (letters, digits, punctuation, whitespace) and merges never cross a chunk,
  so vocabulary slots are not spent on variants such as "dog." / "dog!"; cl100k's punctuation clause also absorbs
  following line breaks (tokens like `"):\n"`), which this pattern drops.
- **Papers.** Radford et al., 2019 (OpenAI report, non-archival, §2.2): BPE is prevented "from merging across
  character categories", against "dog. dog! dog?"; arXiv 2402.01035 (Dagan, 2024, §2.4, Fig. 3): the GPT-2 and GPT-4
  regexes clause by clause; arXiv 2608.18062 (Meister, 2026, App. F.3, fn. 16): over 18 byte-level math+code models,
  vocabulary tokens holding a line break vs MBPP pass@1 Spearman ρ = −0.717 (p = 8.1e-4); its companion post
  "Tokenizer Code Ablations" (blog, non-archival, "Differing one regex clause"): deleting the trailing `[\r\n/]*`
  from the punctuation clause, same corpus, three seeds, MBPP pass@1 mean 0.086 → 0.230.
- **Here.** ADAPTED: `corpus/tokenizer.py` `SPLIT_PATTERN`; `docs/corpus.md` §5, `docs/experiments.md` §2; nanochat's
  pattern without `[\r\n]*` after a punctuation run (the fused pattern is the ablation's positive control).
- **Upstream.** nanochat (`nanochat/tokenizer.py` `SPLIT_PATTERN`, cl100k's with `\p{N}{1,2}`).

### Digit runs of one or two (`\p{N}{1,2}`)
- **Idea.** Numbers are chunked into runs of at most k digits before BPE, giving a fixed, regular set of number
  tokens; small k makes arithmetic more uniform, larger k compresses more.
- **Papers.** arXiv 2302.13971 (LLaMA, 2023, §2.1): "we split all numbers into individual digits" (k = 1); arXiv
  2402.14903 (Singh, 2024, §1, Table 1): PaLM, LLaMA 1–2 and Mistral use single digits, GPT-3.5/4 left-to-right chunks
  of up to three; right-to-left chunking raises their addition accuracy by up to 20%. No paper tests k = 2.
- **Here.** ADOPTED: `corpus/tokenizer.py` `SPLIT_PATTERN`; `docs/corpus.md` §5; k = 2 is a design choice.
- **Upstream.** nanochat LOG 2026-01-13 "Number Token Split Pattern" (d12, one run per k = 1, 2, 3).

### BOS as the only document token: delimiter, end-of-document target, generation stop
- **Idea.** One special token starts every document; the token after a document's last token is the next BOS, so
  the model learns to end documents and a sampled BOS ends a generation. Special ids follow the mergeable ranks.
- **Papers.** None: convention; arXiv 2401.14196 (DeepSeek-Coder, 2024, §3.1.2) only for the FIM sentinels' names.
- **Here.** ADAPTED: `corpus/tokenizer.py` `BOS`, `SPECIAL_TOKENS`, `corpus/loader.py` `PretrainingBatches.assemble`,
  `inference/decoder.py` `NativeDecoder`, `inference/export_hf.py` `export`; `docs/corpus.md` §5, §9,
  `docs/evaluation.md` §6, `docs/training.md` §10; BOS is a target only after a piece that ends its document or a FIM
  middle, three FIM sentinels follow it, no chat tokens.
- **Upstream.** nanochat (`nanochat/tokenizer.py` `SPECIAL_TOKENS`: BOS and eight chat tokens; `nanochat/engine.py`).

### Tokenizer comparison by compression ratio
- **Idea.** Ranking tokenizers by bytes per token assumes fewer tokens means a better model.
- **Papers.** arXiv 2402.18376 (Schmidt, 2024, §5.3): within a vocabulary-construction method corpus token count
  varies little while pre-tokenization and segmentation change accuracy; it is not inherently beneficial to use fewer
  tokens. Meister 2026 companion post (blog, non-archival, "Code bits per byte"): code bits per byte shows no
  association with MBPP.
- **Here.** NOT TAKEN: `metrics.py` `parse_outcome`; `docs/experiments.md` §2, `docs/evaluation.md` §3; arms compared
  by line-break entries, depth invariance, merged-token reads and the parse measure.
- **Upstream.** nanochat (`scripts/tok_eval.py`: compression against GPT-2 and GPT-4).

### BOS-aligned rows with best-fit-decreasing packing
- **Idea.** Concatenate-and-chunk packing cuts documents at row boundaries, so targets lose their document's start;
  packing whole documents (or minimal pieces) into rows by bin packing removes those truncations at almost no
  padding cost.
- **Papers.** arXiv 2404.10830 (Ding, 2024, §3.1, Table 2, Abstract): Best-Fit-Decreasing over document chunks of at
  most the context length adds ≤ 0.01% training sequences over concatenation; +16.8% context following, +9.2% program
  synthesis, up to 58.3% less closed-domain hallucination.
- **Here.** ADAPTED: `corpus/crate/pack.rs` `best_fit_decreasing`, `corpus/loader.py` `PretrainingBatches.assemble`;
  `docs/corpus.md` §8, §9; documents cut at statement or line boundaries, nothing cropped, a BOS before every piece.
- **Upstream.** nanochat LOG 2026-01-13 "BOS-Aligned Dataloader with Bin Packing" (best fit, crop when none fits);
  modded-nanogpt record #26.

### ClimbMix as the web-text source
- **Idea.** Cluster a web corpus in embedding space, prune clusters and search cluster mixture weights with proxy
  models; the resulting mix trains better models per token than single-filter web datasets.
- **Papers.** arXiv 2504.13161 (Diao, 2025, §6, Fig. 1): Nemotron-ClimbMix, 400B tokens mixed by CLIMB from
  Nemotron-CC and smollm-corpus; a 1B model trained on it from scratch outperforms ones trained on Nemotron-CC,
  SmolLM, DCLM-baseline and FineWeb-Edu at equal token budget.
- **Here.** ADOPTED: `corpus/sources.py` `SOURCES`, `corpus/mixture.py` `FILE_LEVEL_STAGE`, `REPO_LEVEL_STAGE`;
  `docs/corpus.md` §1, §7; the 10% text group, read in part.
- **Upstream.** nanochat LOG 2026-03-04 "Dataset upgrade: FineWeb-EDU 100B → ClimbMix 400B"; dataset
  `karpathy/climbmix-400b-shuffle` (`nvidia/Nemotron-ClimbMix` decoded from GPT-2 tokens and shuffled).

### Bits per byte, zero-byte targets excluded
- **Idea.** Report summed nats over ln 2 times summed UTF-8 bytes of the targets, which compares models across
  tokenizers; targets that are not text (BOS, padding, FIM sentinels; 0 bytes) contribute neither nats nor bytes.
- **Papers.** arXiv 2101.00027 (The Pile, 2020, §3.1): bpb = (L_T / L_B)·ℓ / ln 2, L_B in UTF-8 bytes, preferred "due
  to its invariance to different tokenization schemes". The zero-byte exclusion has no paper.
- **Here.** ADAPTED: `eval/bpb.py` `sums`, `corpus/tokenizer.py` `Tokenizer.token_lengths`, `metrics.py` `bpb`,
  `mixture_bpb`; `docs/evaluation.md` §2, §5; fp64 sums per context bucket, byte-offset bucket and row.
- **Upstream.** nanochat (`nanochat/loss_eval.py` `evaluate_bpb`; `scripts/tok_train.py` token bytes, 0 for specials).

### Validation budget of 80 × 2^19 tokens
- **Idea.** A fixed number of validation tokens per evaluation, enough that the estimate's noise is below the
  differences of interest.
- **Papers.** None: design choice inherited from nanochat; its adequacy is the reported clustered standard errors
  (`docs/evaluation.md` §5).
- **Here.** ADAPTED: `eval/bpb.py` `VAL_TOKENS`, `sample_rows`; `docs/evaluation.md` §2; per corpus, rounded up to
  whole micro-batches, rows drawn systematically.
- **Upstream.** nanochat LOG 2026-03-04 "Dataset upgrade" (40 → 80 batches "for more stable validation loss
  estimates").

### Execution-based pass@k in a sandbox
- **Idea.** Score generated code by running its tests in an isolated process and estimate pass@k without bias from n
  samples per problem.
- **Papers.** arXiv 2107.03374 (Chen, 2021, §2.1, §2.3): functional correctness, the unbiased pass@k estimator, a
  sandbox for executing generated programs.
- **Here.** PARALLEL: `metrics.py` `pass_at_k`, `eval/verify.py` `RUN_EXAMPLES`; `docs/evaluation.md` §4, §5;
  EvalPlus, MultiPL-E and infilling evaluators, EvalPlus's `reliability_guard`.
- **Upstream.** nanochat (`tasks/humaneval.py`, `nanochat/execution.py`).

### Truncated sampling: nucleus (top-p), not top-k
- **Idea.** Sample at a temperature from a truncated distribution; the nucleus is the smallest set of tokens holding
  probability p, which adapts to the distribution's shape where a fixed k does not.
- **Papers.** arXiv 1904.09751 (Holtzman, 2019, §3.1, §3.2): nucleus sampling and the difficulty of choosing k.
- **Here.** PARALLEL: `inference/decoder.py` `NativeDecoder`, `model/kernels/decode.cu` `k_nucleus`;
  `docs/evaluation.md` §4, §6.
- **Upstream.** nanochat (`nanochat/engine.py` `sample_next_token`: temperature and top-k).

## 4. Systems and numerics

### FlashAttention kernels for training attention
- **Idea.** Exact attention computed tile by tile in on-chip SRAM with an online softmax, never materializing the N×N
  score matrix, so HBM traffic, the bottleneck, falls; later versions add Hopper asynchrony (FA3) and Blackwell
  pipelining (FA4).
- **Papers.** arXiv 2205.14135 (Dao et al., 2022, Abstract, §3.2 Thm. 2): tiling and recomputation give exact attention
  with Θ(N²d²M⁻¹) HBM accesses against Θ(Nd + N²) for standard attention; arXiv 2407.08608 (Shah et al., 2024,
  Abstract, §3.1): the Hopper kernel (warp-specialized producers/consumers, pingpong softmax–GEMM overlap), FP16
  forward 1.5–2.0× over FA2, up to 740 TFLOP/s (75%) on H100; arXiv 2603.05451 (Zadouri et al., 2026, Abstract, §3.3,
  §4): FA4, written in CuTe-DSL, LPT scheduling for causal and varlen attention, benchmarked on Blackwell GPUs only.
- **Here.** ADAPTED: `model/ops/flash_attn.py` `attention`; `docs/architecture.md` §3; FA4 (CuTe DSL) varlen in place
  of upstream's FA3, no SDPA fallback (`docs/experiments.md` §5, "FA3 not pursued").
- **Upstream.** nanochat LOG 2026-01-11 "Flash Attention 3 Integration", LOG 2026-01-16 (SDPA fallback); modded-nanogpt
  records #12 (FlexAttention), #29 (FA3).

### Sliding windows inside the attention kernel
- **Idea.** A local mask leaves most score tiles fully masked; a kernel that skips them cuts IO and compute in
  proportion to the tiles kept, and applies the mask only on partially masked tiles.
- **Papers.** arXiv 2205.14135 (Dao et al., 2022, §3.3 Prop. 4): block-sparse FlashAttention skips zero blocks,
  Θ(Nd + N²d²M⁻¹s) HBM accesses for a fraction s of nonzero blocks; arXiv 2412.05496 (Dong et al., 2024, §4.2,
  Fig. 3): a precomputed BlockMask skips fully masked blocks of a sliding-window mask and evaluates the mask only on
  partial blocks (≈15% faster for causal masks).
- **Here.** ADAPTED: `model/modeling.py` `layer_windows`, `model/ops/flash_attn.py` `fa4_window_left`;
  `docs/architecture.md` §1, §3; a token-exact 512-key window in FA4's kernel, not a block-aligned one
  (`docs/experiments.md` §3).
- **Upstream.** nanochat LOG 2026-01-11 "Sliding Window Attention"; modded-nanogpt record #16 (block sliding window).

### Fixed-size `cu_seqlens`, with only the real segments passed to the kernel
- **Idea.** A static-shape compiler recompiles for every new input shape, so varlen offsets are padded to a fixed
  length; the kernel launches work for every listed segment, so passing only the real ones removes the padding's cost.
- **Papers.** PyTorch 2 (Ansel et al., ASPLOS 2024, not on arXiv, §5): static-shape compilers "must recompile if any
  input shape changes"; arXiv 2307.08691 (Dao, 2023, §3.2): the forward is parallelized over batch, heads and row
  blocks, one thread block each, so every listed sequence gets work.
- **Here.** ADAPTED: `corpus/loader.py` `PretrainingBatches` (fixed `num_segments + 1` offsets),
  `model/ops/flash_attn.py` `segmented`; `docs/architecture.md` §3; FA4 gets the batch's exact segment count, not a
  tighter static bound.
- **Upstream.** nanochat LOG 2026-01-13 "Varlen Attention (Negative Result)" (`cu_seqlens` padded against
  recompiles); modded-nanogpt records #29, #78 (tighter `max_num_docs`).

### Mixed precision: fp32 master weights, bf16 GEMM operands with fp32 accumulation
- **Idea.** Keep fp32 weights for the update and run forward and backward GEMMs on a 16-bit copy with fp32
  accumulation; bf16 has fp32's exponent range, so it trains without loss scaling.
- **Papers.** arXiv 1710.03740 (Micikevicius et al., 2017, §3.1, Fig. 1): an FP16 copy of FP32 master weights is used
  in forward and backward; updating FP16 weights directly lost 80% relative accuracy on a Mandarin speech model;
  arXiv 1905.12322 (Kalamkar et al., 2019, Abstract, §3, Fig. 1): BF16 GEMM inputs, FP32 outputs and an FP32 weight
  copy match FP32 in the same iterations with no hyperparameter changes and no loss scaling.
- **Here.** ADAPTED: `model/modeling.py` `GemmLinear.weight_bf16`, `Attention.qkv_bf16`,
  `CausalLM.refresh_gemm_weights`; `docs/architecture.md` §4; persistent bf16 copies refreshed after each update
  instead of a cast per forward.
- **Upstream.** nanochat LOG 2026-03-04; modded-nanogpt record #10.

### Normalization statistics and the loss in fp32
- **Idea.** Large reductions (norm statistics, softmax normalizers, cross entropy) lose accuracy in 16 bits; doing
  their arithmetic in fp32 costs nothing in memory-bound layers.
- **Papers.** arXiv 1710.03740 (Micikevicius et al., 2017, §3.3): large reductions "should be carried out in FP32"
  without slowing these bandwidth-bound layers; arXiv 2407.08608 (Shah et al., 2024, §4.3): with softmax
  intermediates in FP32, FlashAttention in FP16 has 1.7× lower RMSE than standard attention against FP64.
- **Here.** ADAPTED: `model/modeling.py` `norm`, `model/ops/cross_entropy.py` `lm_loss`; `docs/architecture.md` §4;
  norm statistics in fp32 (upstream's norm runs in bf16), the loss from the logits GEMM's fp32 accumulators.
- **Upstream.** nanochat (fp32 logits for the cap and the loss); LOG 2026-03-04.

### FP8 training
- **Idea.** Run GEMMs in 8-bit floating point with scaling factors and fp32 accumulation for up to twice the
  tensor-core throughput; the casts and transposed copies are extra memory passes.
- **Papers.** arXiv 2209.05433 (Micikevicius et al., 2022, §3, §4.1 Table 4): E4M3 for weights and activations, E5M2
  for gradients, per-tensor scaling; GPT models of 126M–175B (175B at 75% of training) match 16-bit perplexity
  within run-to-run noise; arXiv 2412.19437 (DeepSeek-AI, 2024, §3.3, App. B.1, §3.5.2): 1×128 / 128×128 scaling
  with FP32 accumulation, loss within 0.25% of BF16 at ~16B and ~230B; the backward reads, dequantizes, transposes and
  re-quantizes activations through HBM.
- **Here.** NOT TAKEN: `model/ops/linear.py` (bf16 GEMMs); `docs/experiments.md` §5; torchao and Transformer Engine
  FP8 MLP GEMMs were slower at d24 [unverified: no record].
- **Upstream.** nanochat LOG 2026-01-13 "FP8 Training for lm_head", LOG 2026-02-02 (torchao); modded-nanogpt
  records #19, #84, #89, #90.

### Fused linear cross entropy with the logit cap inside
- **Idea.** Compute the loss and its gradients from the classifier GEMM tile by tile with an online log-sum-exp, so
  the N×V logits, the largest activation of a large-vocabulary model, are never stored.
- **Papers.** arXiv 2411.09009 (Wijmans et al., 2024, Abstract, §4, §5.1 Table 1): indexed matmul plus
  linear-log-sum-exp in SRAM; for Gemma 2 2B (8,192 tokens, V = 256,000) the loss's memory falls from 24 GB to 1 MB
  and the classifier head's from 28 GB to ~1 GB; arXiv 2410.10989 (Hsu et al., 2024, §3.2 "FusedLinearCrossEntropy"):
  the chunked alternative, logits and gradients computed per chunk.
- **Here.** PARALLEL: `model/ops/cross_entropy.py` `lm_loss`; `docs/architecture.md` §3; QuACK's scaled-exp linear
  cross entropy per 8,192-row chunk, tanh applied on the fp32 accumulator.
- **Upstream.** modded-nanogpt records #60, #67, #75, #79 (fused softcapped cross entropy on a stored logits tensor).

### Fused linear + ReLU² GEMM; activations stored for the backward
- **Idea.** Apply the activation in the GEMM epilogue while the tile is in registers, removing a memory-bound pass;
  store the activations that are costly to recompute.
- **Papers.** arXiv 2104.04473 (Narayanan et al., 2021, §4.2, §5.8): fused bias + GeLU and other element-wise kernels
  raise GPT-3 175B throughput by 19% (113 → 135 TFLOP/s per GPU); PyTorch 2 (Ansel et al., ASPLOS 2024, not on
  arXiv, §6.4 Table 4): matmul templates with pointwise epilogue fusion; arXiv 2205.05198 (Korthikanti et al., 2022,
  §5): selective recomputation stores what is expensive to recompute and recomputes the large, cheap parts.
- **Here.** PARALLEL: `model/ops/linear.py` `relu2_mlp`; `docs/architecture.md` §3; QuACK `gemm_norm_act` with the
  norm folded in as a row scale, pre-activation and activation both stored (memory spent for speed).
- **Upstream.** modded-nanogpt record #59 (Triton linear + ReLU², pre-activation stored).

### Fused QK-norm + RoPE kernel
- **Idea.** Compute the per-head RMS norm, gain and rotary embedding of q and k in one memory-bound pass.
- **Papers.** arXiv 2410.10989 (Hsu et al., 2024, §3.2 "RMSNorm", "RoPE"): normalization and scaling fused into one
  kernel, q and k rotation into another; no paper fuses norm and RoPE together.
- **Here.** PARALLEL: `model/kernels/qk_prologue.cu`, `model/kernels/qk_row.cuh`; `docs/architecture.md` §3; CUDA,
  fp32 math with one bf16 rounding, shared bitwise with the decoder.
- **Upstream.** modded-nanogpt record #90.

### Merged q/k/v projection (one GEMM)
- **Idea.** Stack W_q, W_k, W_v so the projection is one larger GEMM reading the input once.
- **Papers.** None: an implementation choice; its value here is this repository's own measurement.
- **Here.** PARALLEL: `model/ops/linear.py` `qkv_proj`; `docs/architecture.md` §3; separate fp32 parameters, one
  stacked bf16 copy, dW as three GEMMs.
- **Upstream.** modded-nanogpt record #20.

### Newton–Schulz batched over same-shape matrices
- **Idea.** Stack Muon's matrices of one shape and run the orthogonalization as one batched GEMM chain with identical
  math, so a deep model launches a few large kernels instead of many small ones.
- **Papers.** arXiv 2504.05295 (Ahn et al., 2025, §4.1): a batch dimension "grouping matrices of identical shape for
  efficient batched computation"; arXiv 2608.11612 (Amsel et al., 2026, §7): all same-shape matrices orthogonalized
  as one batch, which makes the communication rounds O(1) in depth.
- **Here.** PARALLEL: `train/optim.py` `Muon.step`; `docs/training.md` §3; one GPU, so the gain is kernel efficiency,
  not communication.
- **Upstream.** nanochat; modded-nanogpt records #15, #20, #32.

### `torch.compile` of the whole model, static shapes, one graph
- **Idea.** Capture the program into one graph (TorchDynamo) and generate fused kernels (TorchInductor); static
  shapes avoid dynamic-shape reasoning, and custom ops trace through their meta implementations.
- **Papers.** PyTorch 2 (Ansel et al., ASPLOS 2024, not on arXiv, Abstract, §5, §5.1): 1.41× training
  geometric-mean speedup over eager on an A100 across 180+ models; static shapes recompile on any shape change;
  user-defined meta functions for custom ops.
- **Here.** ADOPTED: `train/loop.py` `train`; `docs/architecture.md` §3; with `fullgraph=True`, recompiled at the 32K
  stage, every kernel a custom op with a fake implementation.
- **Upstream.** nanochat.

### Fused AdamW step
- **Idea.** Run the element-wise AdamW update for all tensors in a few kernels instead of one per parameter.
- **Papers.** None: a PyTorch feature (`torch.optim.AdamW(fused=True)`).
- **Here.** ADOPTED: `train/optim.py` `optimizers`; `docs/training.md` §3.
- **Upstream.** nanochat; modded-nanogpt.

### Meta-device construction, then explicit initialization
- **Idea.** Build the module with shape-only tensors so it can be planned and counted before memory is committed, then
  allocate storage and set every value in one initialization pass.
- **Papers.** arXiv 2304.11277 (Zhao et al., 2023, §3.1): deferred initialization allocates parameters on a "fake"
  device, records the initialization operations and replays them on the GPU, building a model "without allocating
  any GPU memory blocks".
- **Here.** ADOPTED: `model/modeling.py` `meta_model`, `CausalLM.init_weights`, `train/loop.py` `train`;
  `docs/architecture.md` §1; an explicit `init_weights` (per-tensor seeded) instead of recorded-op replay.
- **Upstream.** nanochat; LOG 2026-01-11 "Per-Layer Residual Scalars" ("Meta Device Footgun"), LOG 2026-02-05.

### Garbage-collector control
- **Idea.** Freeze the long-lived objects and disable automatic collection after the first step, so no collector scan
  pauses the training loop.
- **Papers.** None: an engineering detail.
- **Here.** ADAPTED: `train/loop.py` `train`; `docs/training.md` §8; collection only after each save.
- **Upstream.** nanochat; LOG 2026-02-16.

### Gradient accumulation weighted by valid tokens
- **Idea.** A step is several micro-batches whose gradients add up; weighting each by its share of the step's valid
  labels makes the loss the mean over all valid tokens, not a mean of micro-batch means.
- **Papers.** arXiv 2410.02660 (Gao et al., 2024, App. A.3 "Token-averaged loss"): averaging over sequences instead of
  valid tokens "skews the optimization" (SFT, data-parallel); they average over all valid tokens.
- **Here.** ADAPTED: `model/ops/cross_entropy.py` `step_shares`, `train/loop.py` `train`; `docs/training.md` §6;
  valid-label shares instead of 1/(accumulation steps).
- **Upstream.** nanochat.

### Overlapping host work with the device step
- **Idea.** Read the next micro-batches and copy values through pinned memory without blocking while the GPU runs the
  current step, so the device never waits on the host.
- **Papers.** None: standard engineering.
- **Here.** ADAPTED: `train/loop.py` `train`, `corpus/loader.py` `PretrainingBatches`; `docs/training.md` §6; step
  values also leave through pinned buffers and are logged one step late.
- **Upstream.** nanochat.

### Model FLOPs utilization (MFU)
- **Idea.** Report achieved model FLOPs (tokens per second times an analytic count, excluding rematerialization) over
  the hardware's peak, so systems compare independent of implementation.
- **Papers.** arXiv 2204.02311 (Chowdhery et al., 2022, §4.1, Table 3, App. B): MFU, observed throughput over the
  theoretical maximum at peak FLOPs, counting forward and backward but not rematerialization; PaLM 540B 46.2% (HFU
  57.8%).
- **Here.** ADAPTED: `metrics.py` `mfu`, `executed_model_flops`, `PEAK_FLOPS`; `docs/evaluation.md` §5,
  `docs/training.md` §7; executed per-segment FLOPs over 989.4 TFLOP/s (H100 SXM / GH200 dense BF16).
- **Upstream.** nanochat.

## References

| arXiv | First author, year | Title |
|---|---|---|
| 1211.5063 | Pascanu, 2012 | On the difficulty of training Recurrent Neural Networks |
| 1412.6980 | Kingma, 2014 | Adam: A Method for Stochastic Optimization |
| 1508.07909 | Sennrich, 2015 | Neural Machine Translation of Rare Words with Subword Units |
| 1608.05859 | Press, 2016 | Using the Output Embedding to Improve Language Models |
| 1611.09940 | Bello, 2016 | Neural Combinatorial Optimization with Reinforcement Learning |
| 1706.02677 | Goyal, 2017 | Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour |
| 1706.03762 | Vaswani, 2017 | Attention Is All You Need |
| 1710.03740 | Micikevicius, 2017 | Mixed Precision Training |
| 1711.05101 | Loshchilov, 2017 | Decoupled Weight Decay Regularization |
| 1809.10853 | Baevski, 2018 | Adaptive Input Representations for Neural Language Modeling |
| 1901.09321 | Zhang, 2019 | Fixup Initialization: Residual Learning Without Normalization |
| 1904.09751 | Holtzman, 2019 | The Curious Case of Neural Text Degeneration |
| 1905.12322 | Kalamkar, 2019 | A Study of BFLOAT16 for Deep Learning Training |
| 1910.04209 | Ma, 2019 | On the adequacy of untuned warmup for adaptive optimization |
| 1910.07467 | Zhang, 2019 | Root Mean Square Layer Normalization |
| 1911.02150 | Shazeer, 2019 | Fast Transformer Decoding: One Write-Head is All You Need |
| 2001.08361 | Kaplan, 2020 | Scaling Laws for Neural Language Models |
| 2002.04745 | Xiong, 2020 | On Layer Normalization in the Transformer Architecture |
| 2002.05202 | Shazeer, 2020 | GLU Variants Improve Transformer |
| 2002.07028 | Bhojanapalli, 2020 | Low-Rank Bottleneck in Multi-head Attention Models |
| 2004.05150 | Beltagy, 2020 | Longformer: The Long-Document Transformer |
| 2005.14165 | Brown, 2020 | Language Models are Few-Shot Learners |
| 2101.00027 | Gao, 2020 | The Pile: An 800GB Dataset of Diverse Text for Language Modeling |
| 2104.04473 | Narayanan, 2021 | Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM |
| 2104.09864 | Su, 2021 | RoFormer: Enhanced Transformer with Rotary Position Embedding |
| 2107.03374 | Chen, 2021 | Evaluating Large Language Models Trained on Code |
| 2109.08668 | So, 2021 | Primer: Searching for Efficient Transformers for Language Modeling |
| 2110.02861 | Dettmers, 2021 | 8-bit Optimizers via Block-wise Quantization |
| 2203.03466 | Yang, 2022 | Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer |
| 2203.15556 | Hoffmann, 2022 | Training Compute-Optimal Large Language Models |
| 2204.02311 | Chowdhery, 2022 | PaLM: Scaling Language Modeling with Pathways |
| 2205.05198 | Korthikanti, 2022 | Reducing Activation Recomputation in Large Transformer Models |
| 2205.14135 | Dao, 2022 | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness |
| 2209.05433 | Micikevicius, 2022 | FP8 Formats for Deep Learning |
| 2211.05100 | BigScience Workshop, 2022 | BLOOM: A 176B-Parameter Open-Access Multilingual Language Model |
| 2302.05442 | Dehghani, 2023 | Scaling Vision Transformers to 22 Billion Parameters |
| 2302.13971 | Touvron, 2023 | LLaMA: Open and Efficient Foundation Language Models |
| 2304.11277 | Zhao, 2023 | PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel |
| 2305.13245 | Ainslie, 2023 | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints |
| 2307.08691 | Dao, 2023 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning |
| 2309.14322 | Wortsman, 2023 | Small-scale proxies for large-scale Transformer training instabilities |
| 2401.02954 | DeepSeek-AI, 2024 | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism |
| 2401.14196 | Guo, 2024 | DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence |
| 2402.00838 | Groeneveld, 2024 | OLMo: Accelerating the Science of Language Models |
| 2402.01035 | Dagan, 2024 | Getting the most out of your tokenizer for pre-training and domain adaptation |
| 2402.02622 | Pagliardini, 2024 | DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging |
| 2402.13991 | Zhao, 2024 | Analysing The Impact of Sequence Composition on Language Model Pre-Training |
| 2402.14903 | Singh, 2024 | Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs |
| 2402.14905 | Liu, 2024 | MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases |
| 2402.18376 | Schmidt, 2024 | Tokenization Is More Than Compression |
| 2402.19173 | Lozhkov, 2024 | StarCoder 2 and The Stack v2: The Next Generation |
| 2404.10830 | Ding, 2024 | Fewer Truncations Improve Language Modeling |
| 2404.19737 | Gloeckle, 2024 | Better & Faster Large Language Models via Multi-token Prediction |
| 2405.09818 | Chameleon Team, 2024 | Chameleon: Mixed-Modal Early-Fusion Foundation Models |
| 2405.13698 | Wang, 2024 | How to set AdamW's weight decay as you scale model and dataset size |
| 2405.14591 | Men, 2024 | Base of RoPE Bounds Context Length |
| 2405.18392 | Hägele, 2024 | Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations |
| 2406.19146 | Porian, 2024 | Resolving Discrepancies in Compute-Optimal Scaling of Language Models |
| 2407.08608 | Shah, 2024 | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision |
| 2407.21783 | Grattafiori, 2024 | The Llama 3 Herd of Models |
| 2408.00118 | Gemma Team, 2024 | Gemma 2: Improving Open Language Models at a Practical Size |
| 2409.20325 | Bernstein, 2024 | Old Optimizer, New Norm: An Anthology |
| 2410.01131 | Loshchilov, 2024 | nGPT: Normalized Transformer with Representation Learning on the Hypersphere |
| 2410.02660 | Gao, 2024 | How to Train Long-Context Language Models (Effectively) |
| 2410.06205 | Barbero, 2024 | Round and Round We Go! What makes Rotary Positional Encodings useful? |
| 2410.10989 | Hsu, 2024 | Liger Kernel: Efficient Triton Kernels for LLM Training |
| 2410.17897 | Zhou, 2024 | Value Residual Learning |
| 2411.09009 | Wijmans, 2024 | Cut Your Losses in Large-Vocabulary Language Models |
| 2411.16085 | Liang, 2024 | Cautious Optimizers: Improving Training with One Line of Code |
| 2412.05496 | Dong, 2024 | Flex Attention: A Programming Model for Generating Optimized Attention Kernels |
| 2412.19437 | DeepSeek-AI, 2024 | DeepSeek-V3 Technical Report |
| 2501.00656 | Team OLMo, 2024 | 2 OLMo 2 Furious |
| 2502.15938 | Bergsma, 2025 | Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs |
| 2502.16982 | Liu, 2025 | Muon is Scalable for LLM Training |
| 2503.19786 | Gemma Team, 2025 | Gemma 3 Technical Report |
| 2504.05295 | Ahn, 2025 | Dion: Distributed Orthonormalized Updates |
| 2504.13161 | Diao, 2025 | Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training |
| 2505.06708 | Qiu, 2025 | Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free |
| 2505.13738 | Bergsma, 2025 | Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training |
| 2505.16932 | Amsel, 2025 | The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm |
| 2509.02046 | Wen, 2025 | Fantastic Pretraining Optimizers and Where to Find Them |
| 2510.05491 | Li, 2025 | NorMuon: Making Muon more efficient and scalable |
| 2510.12402 | Chen, 2025 | Cautious Weight Decay |
| 2512.05620 | Qiu, 2025 | Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales |
| 2512.22382 | Mlodozeniec, 2025 | Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration |
| 2601.22887 | Li, 2026 | MoVE: Mixture of Value Embeddings -- A New Axis for Scaling Parametric Memory in Autoregressive Models |
| 2603.05451 | Zadouri, 2026 | FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling |
| 2604.14053 | Chizhov, 2026 | From Where Words Come: Efficient Regularization of Code Tokenizers Through Source Attribution |
| 2608.11612 | Amsel, 2026 | Dion3: Full-Stack Orthogonal Updates |
| 2608.18062 | Meister, 2026 | TokEval: A Tokenizer Evaluation Suite |

Non-arXiv sources:
- Ansel et al., 2024. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph
  Compilation. ASPLOS '24, DOI 10.1145/3620665.3640366.
- Björck & Bowie, 1971. An iterative algorithm for computing the best estimate of an orthogonal matrix. SIAM J. Numer.
  Anal. 8:358–364 [unverified: cited through arXiv 2409.20325].
- Kovarik, 1970. Some iterative methods for improving orthonormality. SIAM J. Numer. Anal. [unverified: cited
  through arXiv 2409.20325].
- LeCun, Bottou, Orr & Müller, 1998. Efficient BackProp. Neural Networks: Tricks of the Trade, Springer.
- Sutskever, Martens, Dahl & Hinton, 2013. On the importance of initialization and momentum in deep learning. ICML,
  PMLR 28.
- Non-archival:
  - Jordan, Jin, Boza, You, Cesista, Newhouse & Bernstein, 2024, "Muon: An optimizer for hidden layers in neural
    networks", https://kellerjordan.github.io/posts/muon/ (blog)
  - Radford et al., 2019, "Language Models are Unsupervised Multitask Learners" (OpenAI report)
  - Meister, 2026, "Tokenizer Code Ablations" (TokEval companion blog post)
  - NVIDIA H100 Tensor Core GPU datasheet (peak dense BF16 FLOP/s)
- Software and data (provenance): karpathy/nanochat (`92d63d4`, `dev/LOG.md`); KellerJordan/modded-nanogpt
  (records by number); karpathy/rustbpe; openai/tiktoken; PyTorch (`torch.optim.Muon`, `torch.compile`); QuACK;
  datasets `karpathy/climbmix-400b-shuffle` and `nvidia/Nemotron-ClimbMix`.
