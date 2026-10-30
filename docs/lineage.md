# Literature behind the upstream ideas

The ideas this repository took from karpathy/nanochat ("nanochat") and KellerJordan/modded-nanogpt
("modded-nanogpt"), each with the papers to cite for it. The upstream code bases are provenance only, never evidence.
Classes: ADOPTED, ADAPTED, PARALLEL (the same idea reached on another route), EVIDENCE (an upstream measurement bears
on a choice), NOT TAKEN (a choice against it in these docs). "None" marks an idea no paper states. Markers as in the
other docs. Each entry: the idea, the papers with the decisive fact, and where it lives here.

## Summary

| Idea | Here | Upstream | Cite as | Section |
|---|---|---|---|---|
| Pre-norm residual block with a final norm before the head | ADOPTED | both | 2002.04745, 1809.10853 | §1 |
| RMS normalization without a learnable gain | ADAPTED | both | 1910.07467, 2402.00838, 2501.00656 | §1 |
| RMS-normed token embedding, no positional embedding | ADOPTED | both | 2211.05100, 2110.02861 | §1 |
| Bias-free linear layers | ADOPTED | nanochat | 2204.02311, 2402.00838 | §1 |
| Untied token embedding and output head | ADOPTED | both | 1608.05859, 2402.14905 | §1 |
| Rotary position embedding (RoPE), half-split layout | ADOPTED | both | 2104.09864 | §1 |
| RoPE base 10^6 from the first step | ADAPTED | nanochat | 2405.14591, 2407.21783, 2410.06205 | §1 |
| QK-norm with learned per-dimension gains, fixed 1/√d scale | ADAPTED | both | 2302.05442, 2309.14322, 2410.01131 | §1 |
| Multi-head attention with all KV heads (not GQA/MQA) | ADOPTED | nanochat | 1706.03762, 1911.02150, 2305.13245 | §1 |
| Head dimension 128 | ADOPTED | both | 2002.07028 | §1 |
| Local–global attention: SSSL, last layer global, 512-key windows | ADAPTED | both | 2005.14165, 2004.05150, 2503.19786 | §1 |
| Document masking in packed sequences (varlen attention) | PARALLEL | both | 2402.13991, 2407.21783, 2410.02660 | §1 |
| Gated attention (output gate) | NOT TAKEN | both | 2505.06708 | §1 |
| Multi-token prediction | NOT TAKEN | both | 2404.19737 | §1 |
| ReLU² MLP, hidden 4C, not SwiGLU | ADOPTED | both | 2109.08668, 2002.05202 | §1 |
| Per-layer residual scalars and the x0 shortcut | ADAPTED | both | 2402.02622 | §1 |
| Gated value embeddings on alternating layers | ADAPTED | both | 2601.22887 | §1 |
| Smear, backout, partial key offset | NOT TAKEN | both | none | §1 |
| Output-logit soft cap c·tanh(z/c), c = 15 | ADOPTED | both | 1611.09940, 2408.00118, 2405.09818 | §1 |
| Uniform fan-in initialization, std C^−½ | ADAPTED | both | LeCun et al., 1998 | §1 |
| Zero-initialized residual-branch outputs and output head | ADOPTED | both | 1901.09321, 2203.03466 | §1 |
| Shape rule: width = depth × aspect, rounded up to whole 128-wide heads | ADAPTED | nanochat | 2402.14905, 2001.08361 | §1 |
| Muon: orthogonalized momentum for hidden matrices | ADAPTED | both | 2409.20325, 2502.16982, Muon blog (Jordan et al., 2024) | §2 |
| Newton–Schulz orthogonalization: quintic (3.4445, −4.7750, 2.0315), 5 bf16 steps, Gram form | ADOPTED | both | 2409.20325, Muon blog (Jordan et al., 2024), Kovarik, 1970, Björck & Bowie, 1971 | §2 |
| Nesterov momentum 0.95 in EMA ("lerp") form | ADOPTED | both | 2502.16982, Muon blog (Jordan et al., 2024), Sutskever et al., 2013 | §2 |
| Muon/AdamW parameter split | ADAPTED | both | 2409.20325, 2502.16982, Muon blog (Jordan et al., 2024) | §2 |
| Muon step size per shape, matched to AdamW's update RMS | ADAPTED | both | 2502.16982 | §2 |
| NorMuon: neuron-wise second-moment normalization after orthogonalization | ADAPTED | both | 2510.05491 | §2 |
| Polar Express coefficients | NOT TAKEN | both | 2505.16932 | §2 |
| Cautious weight decay | NOT TAKEN | both | 2510.12402 | §2 |
| Decoupled weight decay (AdamW) and weight decay in Muon | ADOPTED | both | 1711.05101, 1412.6980, 2502.16982 | §2 |
| Weight decay tied to the training horizon (the EMA timescale) | PARALLEL | nanochat | 2505.13738, 2405.13698 | §2 |
| Per-group learning rates (AdamW groups, residual scalars) | NOT TAKEN | nanochat | 2502.16982 | §2 |
| Batch size and peak learning rate from compute scaling laws | PARALLEL | nanochat | 2401.02954 | §2 |
| Global gradient-norm clipping at 1.0 | ADOPTED | both | 1211.5063, 2401.02954 | §2 |
| Linear warmup over 2/(1 − β₂) steps | ADOPTED | nanochat | 1910.04209 | §2 |
| Linear decay to zero, not a trapezoid with a floor | ADAPTED | both | 2502.15938, 2405.18392 | §2 |
| Each group's lr as its peak times one schedule multiplier | ADAPTED | nanochat | none | §2 |
| Hyperparameter transfer from a small reference model | NOT TAKEN | nanochat | 2203.03466, 2512.22382 | §2 |
| Training horizon from tokens per parameter | ADAPTED | nanochat | 2203.15556 | §2 |
| Scaling-parameter count: decoder layers + lm_head | ADOPTED | nanochat | 2001.08361, 2406.19146 | §2 |
| FLOPs per token: 6N plus attention over the attended keys | ADAPTED | nanochat | 2204.02311, 2401.02954 | §2 |
| Small-proxy tuning gains shrink with scale | EVIDENCE | nanochat | 2509.02046, 2512.05620 | §2 |
| Byte-level BPE | ADOPTED | nanochat | 1508.07909, Radford et al., 2019 | §3 |
| SA-BPE merge score in an incremental BPE trainer | ADAPTED | nanochat | 2604.14053 | §3 |
| Vocabulary of 2^15 ids, special tokens inside it | ADOPTED | nanochat | 2402.01035, 2402.19173 | §3 |
| Pre-tokenization by a split regex, no line break after a punctuation run | ADAPTED | nanochat | 2608.18062, Radford et al., 2019 | §3 |
| Digit runs of one or two (`\p{N}{1,2}`) | ADOPTED | nanochat | 2302.13971, 2402.14903 | §3 |
| BOS as the only document token: delimiter, end-of-document target, generation stop | ADAPTED | nanochat | none | §3 |
| Tokenizer comparison by compression ratio | NOT TAKEN | nanochat | 2402.18376, Meister, 2026 (post) | §3 |
| BOS-aligned rows with best-fit-decreasing packing | ADAPTED | both | 2404.10830 | §3 |
| ClimbMix as the web-text source | ADOPTED | nanochat | 2504.13161 | §3 |
| Bits per byte, zero-byte targets excluded | ADAPTED | nanochat | 2101.00027 | §3 |
| Validation budget of 80 × 2^19 tokens | ADAPTED | nanochat | none | §3 |
| Execution-based pass@k in a sandbox | PARALLEL | nanochat | 2107.03374 | §3 |
| Truncated sampling: nucleus (top-p), not top-k | PARALLEL | nanochat | 1904.09751 | §3 |
| FlashAttention kernels for training attention | ADAPTED | both | 2205.14135, 2407.08608, 2603.05451 | §4 |
| Sliding windows inside the attention kernel | ADAPTED | both | 2205.14135, 2412.05496 | §4 |
| Fixed-size `cu_seqlens`, with only the real segments passed to the kernel | ADAPTED | both | 2307.08691, PyTorch 2 (Ansel et al., 2024) | §4 |
| Mixed precision: fp32 master weights, bf16 GEMM operands with fp32 accumulation | ADAPTED | both | 1710.03740, 1905.12322 | §4 |
| Normalization statistics and the loss in fp32 | ADAPTED | nanochat | 1710.03740 | §4 |
| FP8 training | NOT TAKEN | both | 2209.05433, 2412.19437 | §4 |
| Fused linear cross entropy with the logit cap inside | PARALLEL | modded-nanogpt | 2411.09009, 2410.10989 | §4 |
| Fused linear + ReLU² GEMM; activations stored for the backward | PARALLEL | modded-nanogpt | 2104.04473, PyTorch 2 (Ansel et al., 2024) | §4 |
| Fused QK-norm + RoPE kernel | PARALLEL | modded-nanogpt | 2410.10989 | §4 |
| Merged q/k/v projection (one GEMM) | PARALLEL | modded-nanogpt | none | §4 |
| Newton–Schulz batched over same-shape matrices | PARALLEL | both | 2608.11612 | §4 |
| `torch.compile` of the whole model, static shapes, one graph | ADOPTED | nanochat | PyTorch 2 (Ansel et al., 2024) | §4 |
| Fused AdamW step | ADOPTED | both | none | §4 |
| Meta-device construction, then explicit initialization | ADOPTED | nanochat | 2304.11277 | §4 |
| Garbage-collector control | ADAPTED | nanochat | none | §4 |
| Gradient accumulation weighted by valid tokens | ADAPTED | nanochat | 2410.02660 | §4 |
| Overlapping host work with the device step | ADAPTED | nanochat | none | §4 |
| Model FLOPs utilization (MFU) | ADAPTED | nanochat | 2204.02311 | §4 |

## 1. Architecture

**Pre-norm residual block with a final norm before the head.** x + F(norm(x)) per sublayer, and one more norm before
the head. *Papers:* arXiv 2002.04745 (Xiong, §3.1 Table 1, §3.3 Thm. 1, §4): at initialization Pre-LN's last-layer
gradient is scaled down by √L, and Pre-LN trains without the warm-up stage; arXiv 1809.10853 (Baevski, §4.1): norm
before the blocks "leads to more effective training". *Here* (ADOPTED): `model/modeling.py` `DecoderLayer.forward`,
`CausalLM.hidden`; `docs/architecture.md` §1.

**RMS normalization without a learnable gain.** Keep LayerNorm's re-scaling, drop re-centering and the gain.
*Papers:* arXiv 1910.07467 (Zhang, Abstract, Eq. 4): matches LayerNorm with 7%–64% less running time (keeps a gain);
arXiv 2402.00838 (OLMo, §2.1): non-parametric layer norm, "the safest option" and the fastest; arXiv 2501.00656
(OLMo 2, §3.3.1): "no difference" between it and RMSNorm. *Here* (ADAPTED): `model/modeling.py` `norm`;
`docs/architecture.md` §1; eps FLT_EPSILON, fp32 statistics (upstream: bf16, torch's default eps).

**RMS-normed token embedding, no positional embedding.** x0 = norm(E[ids]); position enters only through RoPE.
*Papers:* arXiv 2211.05100 (BLOOM, §3.2.3): embedding LayerNorm "significantly improved training stability" at 104B
in float16, though it "penalizes zero-shot generalization"; arXiv 2110.02861 (Dettmers, §2.3): the stable embedding
layer avoids extreme gradient variation. *Here* (ADOPTED): `model/modeling.py` `CausalLM.hidden`;
`docs/architecture.md` §1; a gain-free RMS norm where both papers use LayerNorm.

**Bias-free linear layers.** No additive bias in any projection. *Papers:* arXiv 2204.02311 (PaLM, §2 "No Biases"):
"increased training stability for large models"; arXiv 2402.00838 (OLMo, §2.1): "to improve training stability".
*Here* (ADOPTED): `model/modeling.py` `Attention`, `GemmLinear`, `CausalLM.lm_head`; `docs/architecture.md` §1.

**Untied token embedding and output head.** A second V × C matrix gives the head capacity independent of the input
table. *Papers:* arXiv 1608.05859 (Press, Abstract, §1): tying improves perplexity (the counter-position); arXiv
2402.14905 (MobileLLM, §2.2.3, Table 1): at 125M sharing saves 16M parameters (11.8%) for a 0.2-point accuracy drop.
*Here* (ADOPTED): `model/modeling.py` `CausalLM.embed_tokens`, `CausalLM.lm_head`; `docs/architecture.md` §1; a design
choice on mixed evidence.

**Rotary position embedding (RoPE), half-split layout.** Rotate 2-D pairs of q and k by t·θ_i, θ_i = base^(−2i/d).
*Papers:* arXiv 2104.09864 (Su, §3.2.2 Eq. 15, §3.3, §3.4.3): RoPE, relative position, long-term decay. *Here*
(ADOPTED): `model/kernels/qk_row.cuh` `qk_rotate`, `model/modeling.py` `CausalLM.init_buffers`;
`docs/architecture.md` §1; pairs (i, i + d/2) rotated by −θ, a convention.

**RoPE base 10^6 from the first step.** A larger base keeps distant tokens distinguishable; the base needed grows with
context. *Papers:* arXiv 2405.14591 (Men, §4.3 Table 2): lower bound 8.4e4 at 8K, 6.4e5 at 32K; arXiv 2407.21783
(Llama 3, §3.2): base 500,000; arXiv 2410.06205 (Barbero, §6.1 Table 2): Gemma 2B at 8K, base 10k vs 500k within
0.33% perplexity, mixed sign. *Here* (ADAPTED): `model/config.py` `ROPE_THETA`; `docs/architecture.md` §2; 10^6 in
both stages (upstream 10^5).

**QK-norm with learned per-dimension gains, fixed 1/√d scale.** Normalize q and k per head so attention logits cannot
grow with parameter norms; a learned gain sets the logit scale. *Papers:* arXiv 2302.05442 (ViT-22B, §2, App. B):
logits "over 50000 in magnitude" and divergence at ~8B, fixed by LN on q and k; arXiv 2309.14322 (Wortsman, §3.1.1,
§3.2.5, Fig. E.8): reproduces at small scale under high lr; per-head qk-layernorm beats model-dimension norm; arXiv
2410.01131 (nGPT, App. A.9 Table 5): fixed s_qk = 1 instead of learned costs +0.11% validation loss. *Here*
(ADAPTED): `model/modeling.py` `Attention.q_norm_weight`, `k_norm_weight`, `model/kernels/qk_prologue.cu`,
`model/ops/flash_attn.py` `attention` (scale 1/√128); `docs/architecture.md` §1, §2; norm → gain → RoPE (upstream:
RoPE → gain-free norm → ×1.2).

**Multi-head attention with all KV heads (not GQA/MQA).** One KV head per query head; MQA/GQA trade quality for decode
bandwidth. *Papers:* arXiv 1706.03762 (Vaswani, §3.2.2); arXiv 1911.02150 (Shazeer, Abstract): "only minor quality
degradation"; arXiv 2305.13245 (Ainslie, Abstract): GQA near MHA quality at MQA speed. *Here* (ADOPTED):
`model/modeling.py` `Attention`; `docs/architecture.md` §2 (KV-cache arithmetic), `docs/experiments.md` §3 (d16 GQA
screen, rejected).

**Head dimension 128.** Head rank is at most head size, so it should not shrink as heads are added. *Papers:* arXiv
2002.07028 (Bhojanapalli, Abstract, Thm. 1): the low-rank bottleneck; set head size independently of head count.
*Here* (ADOPTED): `model/config.py` `HEAD_DIM`; `docs/architecture.md` §2; no paper selects 128, also the kernels'
compile-time head width.

**Local–global attention: SSSL, last layer global, 512-key windows.** Windowed layers interleaved with full-context
ones. *Papers:* arXiv 2005.14165 (GPT-3, §2.1): "alternating dense and locally banded sparse attention patterns";
arXiv 2004.05150 (Longformer, §3.1): receptive field l × w; arXiv 2503.19786 (Gemma 3, §2; §5.2 Figs. 3–4): 5 local
layers of 1,024 per global; the ratio has "minimal impact on perplexity", and the window "can be reduced
significantly". *Here* (ADAPTED): `model/modeling.py` `layer_windows`; `docs/architecture.md` §1, §2; fixed 512 keys
at 8K and 32K (upstream a quarter of the context, rounded up to 128).

**Document masking in packed sequences (varlen attention).** Restrict each token to its own document in a packed row.
*Papers:* arXiv 2402.13991 (Zhao, §3.2 Table 1): 1.3B, perplexity 8.410 vs 9.172 (2K) and 8.079 vs 9.065 (8K) against
concatenation; arXiv 2407.21783 (Llama 3, §3.2): "limited impact" in standard pre-training; arXiv 2410.02660
(ProLong, App. B.2 Table 20): helps long and short context in continued training. *Here* (PARALLEL):
`model/ops/flash_attn.py` `attention`, `corpus/loader.py` `PretrainingBatches`; `docs/architecture.md` §1,
`docs/corpus.md` §8–§9; one FA4 varlen segment per packed piece.

**Gated attention (output gate).** A per-head sigmoid gate on the attention output. *Papers:* arXiv 2505.06708 (Qiu,
§1, Table 1): up to 0.2 lower perplexity and 2 MMLU points (15B MoE, 400B tokens), no attention sink. *Here* (NOT
TAKEN): `docs/experiments.md` §3 (not run).

**Multi-token prediction.** Auxiliary heads predict t+2…t+n. *Papers:* arXiv 2404.19737 (Gloeckle, §3.1 Fig. 3): on
code, "worse than the baseline for small model sizes, but outperform the baseline at scale". *Here* (NOT TAKEN):
`docs/experiments.md` §3.

**ReLU² MLP, hidden 4C, not SwiGLU.** relu(x)², ReGLU with tied matrices, without a third matrix. *Papers:* arXiv
2109.08668 (Primer, §3, Fig. 5): Primer's most effective modification, beating GELU, Swish and GLU variants; arXiv
2002.05202 (Shazeer, Table 1): SwiGLU 1.636 vs ReLU 1.677 log-perplexity at matched compute (counter-evidence).
*Here* (ADOPTED): `model/ops/linear.py` `relu2_mlp`; `docs/architecture.md` §1, §2 (not ablated here).

**Per-layer residual scalars and the x0 shortcut.** x ← λr_i·x + λ0_i·x0, init λr = 1, λ0 = 0. *Papers:* arXiv
2402.02622 (DenseFormer, Abstract, §3): learned identity-initialized averages over earlier outputs reach the perplexity
of much deeper models. *Here* (ADAPTED): `model/modeling.py` `CausalLM.resid_lambdas`, `x0_lambdas`, `hidden`,
`init_weights`; `docs/architecture.md` §1; init 1 / 0 (upstream: decaying inits), one shared lr.

**Gated value embeddings on alternating layers.** v ← v + 2σ(x W_g)·E_l[ids] per head, W_g zero so the gate starts at
1. *Papers:* arXiv 2601.22887 (MoVE, §3.2 Eq. 3; §4.1 Eq. 6; §4.2 Table 1): LaVE, "adapted from the modded-nanoGPT
repository", on layers L−1, L−3, …; val bpb −0.016, −0.010, −0.010 at D12, D20, D32. *Here* (ADAPTED):
`model/modeling.py` `has_ve`, `CausalLM.value_embeds`, `Attention.ve_gate`, `model/kernels/ve_gate.cu`;
`docs/architecture.md` §1, §2; the gate reads the full width (nanochat: 32 channels).

**Smear, backout, partial key offset.** Speedrun stream edits: mix each embedding with its predecessor's, subtract a
mid-layer residual before the head, shift keys one position on part of the head dims. *Papers:* None; unreviewed
upstream observations. *Here* (NOT TAKEN): `docs/architecture.md` §2 (smear, backout), `docs/experiments.md` §3
(partial key offset).

**Output-logit soft cap c·tanh(z/c), c = 15.** Bound output logits to (−c, c); QK-norm bounds only attention logits.
*Papers:* arXiv 1611.09940 (Bello, §5.1; App. A.2 Eq. 16): C tanh(u), C = 10; arXiv 2408.00118 (Gemma 2, §2): 50 in
attention, 30 on the final layer; arXiv 2405.09818 (Chameleon, §2.3): QK-norm "does not solve the problem of logit
shift in the final softmax". *Here* (ADOPTED): `model/modeling.py` `FINAL_LOGIT_SOFTCAPPING`,
`model/ops/cross_entropy.py` `lm_loss`, `lm_losses`; `docs/architecture.md` §2, `docs/experiments.md` §3 (designed
ablation, not run); c = 15 has no paper.

**Uniform fan-in initialization, std C^−½.** Uniform on ±√(3/m) has std m^−½, keeping unit pre-activation variance.
*Papers:* LeCun et al., "Efficient BackProp" (1998, §4.6, Eqs. 15–16). *Here* (ADAPTED): `model/modeling.py`
`CausalLM.init_weights`; `docs/architecture.md` §1; also applied to up_proj and the token table (upstream: ×0.4,
N(0, 0.8)).

**Zero-initialized residual-branch outputs and output head.** Every block starts as the identity, the head at
uniform predictions. *Papers:* arXiv 1901.09321 (Fixup, §3): "Initialize the classification layer and the last layer
of each residual branch to 0"; arXiv 2203.03466 (Yang, App. D.2): a zero output layer makes small- and large-width
optima "match more closely". *Here* (ADOPTED): `model/modeling.py` `CausalLM.init_weights` (`o_proj`, `down_proj`,
`lm_head`); `docs/architecture.md` §1.

**Shape rule: width = depth × aspect, rounded up to whole 128-wide heads.** One integer sets the model. *Papers:*
arXiv 2402.14905 (MobileLLM, §2.2.2, Fig. 4; Table 9): "deeper and thinner models outperform" at 125M/350M; 350M is
32 × 960 (ratio 30); arXiv 2001.08361 (Kaplan, §3.1, Fig. 5): aspect can vary 40× with slight effect. *Here*
(ADAPTED): `model/modeling.py` `ASPECT_RATIO`, `meta_model`; `docs/architecture.md` §1, §2; aspect 30 (upstream 64).

## 2. Optimizer, schedule and scaling

**Muon: orthogonalized momentum for hidden matrices.** Replace each hidden matrix's momentum update by UVᵀ.
*Papers:* Jordan et al. (2024, Muon blog, §Definition, §"Why is it good to orthogonalize the update?"): the algorithm
and the near-low-rank updates; arXiv 2409.20325 (Bernstein, Props. 4–5): UVᵀ is the closest semi-orthogonal matrix
and steepest descent under the spectral norm; arXiv 2502.16982 (Liu, §2.2, §3.2): Muon at LLM scale. *Here*
(ADAPTED): `train/optim.py` `Muon`; `docs/training.md` §3; `torch.optim.Muon` with NorMuon added.

**Newton–Schulz orthogonalization: quintic (3.4445, −4.7750, 2.0315), 5 bf16 steps, Gram form.** X ← aX + (bA +
cA²)X, A = XXᵀ, from X₀ = G/‖G‖_F. *Papers:* Jordan et al. (2024, blog, §Definition, §"Tuning the coefficients"):
the coefficients and "only 5 steps"; arXiv 2409.20325 (Bernstein, App. A item 4, Eqs. 36–37): the iteration,
classical for fixed coefficients (Kovarik 1970; Björck & Bowie 1971). *Here* (ADOPTED): `train/optim.py`
`Muon.step`; `docs/training.md` §3; PyTorch's defaults (eps 1e-7).

**Nesterov momentum 0.95 in EMA ("lerp") form.** buf ← β·buf + (1 − β)·g; the (1 − β) scale vanishes under
orthogonalization. *Papers:* Sutskever et al. (2013, ICML, §2); Jordan et al. (2024, blog, §"Empirical
considerations"): Nesterov "works a bit better ... in every case we have tested", Ortho(cG) = Ortho(G); arXiv
2502.16982 (Liu, §2.2): no consistent gain from tuning momentum. *Here* (ADOPTED): `train/optim.py` `Muon.step`;
`docs/training.md` §3; 0.95 constant.

**Muon/AdamW parameter split.** Muon for 2-D hidden matrices; embeddings, head and 1-D parameters on AdamW.
*Papers:* Jordan et al. (2024, blog, §Definition): these "should be optimized by a standard method such as AdamW";
arXiv 2409.20325 (Bernstein, Epilogue): RMS→RMS for linear layers, ℓ1→RMS for embeddings; arXiv 2502.16982 (Liu,
§2.2): AdamW for "RMSNorm, LM head, and embedding parameters". *Here* (ADAPTED): `train/optim.py` `optimizers`;
`docs/training.md` §3; value tables on AdamW, one lr for every group.

**Muon step size per shape, matched to AdamW's update RMS.** Scale the orthogonal update by 0.2·√max(A, B).
*Papers:* arXiv 2502.16982 (Liu, §2.2 Lemma 1, Eq. 4): lets matrix and non-matrix parameters share "learning rate η,
weight decay λ". *Here* (ADAPTED): `train/optim.py` `optimizers` (`adjust_lr_fn="match_rms_adamw"`);
`docs/training.md` §3; replaces upstream's √max(1, rows/cols).

**NorMuon: neuron-wise second-moment normalization after orthogonalization.** Divide each update row by the root of
an EMA of its mean square, then restore the norm. *Papers:* arXiv 2510.05491 (Li, §3.1 Alg. 1; §4.1.1; §4.1.2
Table 1): β₂ = 0.95; Adam's final loss in 21.74% / 13.91% fewer steps at 1.1B / 5.4B (Muon 10.43% / 6.08%). *Here*
(ADAPTED): `train/optim.py` `Muon.step` (`NORMUON_BETA2`, `NORMUON_EPS`); `docs/training.md` §3; always per row
(upstream: along the larger dimension).

**Polar Express coefficients.** Per-iteration minimax-optimal quintics. *Papers:* arXiv 2505.16932 (Amsel, §4.1
Thm. 4.1, Alg. 1; §5.2 Figs. 1, 6): lower GPT-2 validation loss than fixed coefficients at every lr. *Here* (NOT
TAKEN): `train/optim.py` `Muon.step`; `docs/experiments.md` §4; dropped after a one-seed d16 screen.

**Cautious weight decay.** Decay only coordinates where update and parameter share a sign. *Papers:* arXiv
2510.12402 (Chen, §3.1–3.2; §4 Figs. 1, 4): lower final loss for AdamW, Lion and Muon at 338M–2B, 20 TPP, optimal λ
unchanged. *Here* (NOT TAKEN): `train/optim.py` `Muon.step`; `docs/experiments.md` §4; worse at the end of a one-seed
d16 screen.

**Decoupled weight decay (AdamW) and weight decay in Muon.** p ← p·(1 − lr·λ) apart from the adaptive step.
*Papers:* arXiv 1711.05101 (Loshchilov, §2 Alg. 2, Prop. 2): decoupled decay differs from L2 for adaptive methods;
arXiv 1412.6980 (Kingma, Alg. 1): Adam; arXiv 2502.16982 (Liu, §2.2, Fig. 2): without decay Muon's weight and output
RMS outgrow bf16's precise range and its lead fades. *Here* (ADOPTED): `train/optim.py` `optimizers`, `Muon.step`;
`docs/training.md` §3; λ from the τ law.

**Weight decay tied to the training horizon (the EMA timescale).** Weights average updates over τ = 1/(ηλ) steps;
set λ so τ is a fraction of the run. *Papers:* arXiv 2505.13738 (Bergsma, Finding 3; App. E.3.1; App. B): the optimal
fraction is a power law in TPP (c_τ = 1.084, m_τ = −0.527; TPP 20–1280), fit on AdamW; arXiv 2405.13698 (Wang, §3.1
Eqs. 5–6): AdamW as an EMA. *Here* (PARALLEL): `train/optim.py` `tau_weight_decay`; `docs/training.md` §3; one law
for AdamW and Muon.

**Per-group learning rates (AdamW groups, residual scalars).** Separate lr, betas, eps and decay per group versus one
lr and λ. *Papers:* arXiv 2502.16982 (Liu, §2.2): with RMS-matched Muon "learning rate η, weight decay λ" are shared.
*Here* (NOT TAKEN): `train/optim.py` `optimizers`; `docs/training.md` §3, `docs/experiments.md` §4; one d16 run of
the scalar lrs, not adopted.

**Batch size and peak learning rate from compute scaling laws.** Read both off power laws at C = M·D. *Papers:*
arXiv 2401.02954 (DeepSeek LLM, §3.1 Eq. 1): η_opt = 0.3118·C^−0.1250, B_opt = 0.2920·C^0.3271 (1e17–2e19 FLOPs,
multi-step schedule). *Here* (PARALLEL): `train/schedule.py` `plan`; `docs/training.md` §2, §5; the law's lr times a
multiplier measured at d24.

**Global gradient-norm clipping at 1.0.** *Papers:* arXiv 1211.5063 (Pascanu, §3.2 Alg. 1): norm clipping against
exploding gradients; arXiv 2401.02954 (DeepSeek LLM, §2.3): clipping at 1.0. *Here* (ADOPTED): `train/optim.py`
`update` (`MAX_GRAD_NORM`); `docs/training.md` §3; kept where both upstreams removed it.

**Linear warmup over 2/(1 − β₂) steps.** 40 steps at β₂ = 0.95. *Papers:* arXiv 1910.04209 (Ma, §4.3 Eq. 14,
§4.4): "linear warmup over τ = 2·(1−β2)^−1 iterations", untuned, about interchangeable with RAdam. *Here* (ADOPTED):
`train/schedule.py` `WARMUP_STEPS`, `lr_lambda`; `docs/training.md` §4; step s at (s + 1)/40.

**Linear decay to zero, not a trapezoid with a floor.** *Papers:* arXiv 2502.15938 (Bergsma, §4.2 Figs. 6, 8): beats
10× decay by a gap growing with TPP and is "around 0.84% better than the optimal WSD" (610M, 80 TPP); arXiv
2405.18392 (Hägele, Abstract): constant + cooldown scales "similar to cosine". *Here* (ADAPTED): `train/schedule.py`
`lr_lambda`; `docs/training.md` §4; the last step at 1/(steps − 40) of peak.

**Each group's lr as its peak times one schedule multiplier.** *Papers:* None. *Here* (ADAPTED): `train/loop.py`
`LambdaLR`; `docs/training.md` §4.

**Hyperparameter transfer from a small reference model.** *Papers:* arXiv 2203.03466 (Yang, Fig. 1, Table 3): under
μP the optimal lr is stable with width, under standard parameterization it is not; arXiv 2512.22382 (Mlodozeniec,
Abstract): transfer across modules, width, depth, batch and duration. *Here* (NOT TAKEN): `ablations/peak_lr.py`
`lr_multiplier`; `docs/training.md` §5; the multiplier measured at d24 itself.

**Training horizon from tokens per parameter.** D = r·N, steps = D // B. *Papers:* arXiv 2203.15556 (Hoffmann,
Abstract; §3.4 Table 3): 1B parameters ↔ 20.2B tokens. *Here* (ADAPTED): `train/schedule.py` `plan`;
`docs/training.md` §2; r data-bound (`train/stages.py` `max_tokens_per_param`, ≈528), not compute-optimal.

**Scaling-parameter count: decoder layers + lm_head.** Lookup tables cost no matmul FLOPs and are excluded.
*Papers:* arXiv 2001.08361 (Kaplan, §1.3): N excludes embeddings; arXiv 2406.19146 (Porian, §3.2): omitting the head
under-counts FLOPs by ~10% (large) to 90% (small). *Here* (ADOPTED): `model/modeling.py` `num_non_embedding_params`;
`docs/architecture.md` §1, `docs/training.md` §2; 1-D gains and `ve_gate` counted.

**FLOPs per token: 6N plus attention over the attended keys.** A plan read off a law must count FLOPs as the law was
fit. *Papers:* arXiv 2204.02311 (PaLM, App. B): 6N + 12LHQT; arXiv 2401.02954 (DeepSeek LLM, §3.2 Eq. 2):
M = 72·n_layer·d_model² + 12·n_layer·d_model·l_seq. *Here* (ADAPTED): `metrics.py` `model_flops_per_token`,
`executed_model_flops`; `docs/evaluation.md` §5, `docs/training.md` §2; T = min(window + 1, context) per layer, the
plan uses M.

**Small-proxy tuning gains shrink with scale.** *Papers:* arXiv 2509.02046 (Wen, §1 Fig. 1): optimizer speedups fall
from 1.3–1.4× below 1B to 1.1× at 1.2B; arXiv 2512.05620 (Qiu, Abstract): with μP and 1/width decay Muon keeps ~1.4×
from 190M to 1.4B, vanishing under incorrect scaling. *Here* (EVIDENCE): `docs/experiments.md` §1 ("Proxy
settings").

## 3. Tokenizer, data and evaluation

**Byte-level BPE.** *Papers:* arXiv 1508.07909 (Sennrich, §3.2): iteratively merge the most frequent pair; Radford
et al. (2019, OpenAI report, §2.2): byte-level base vocabulary of 256. *Here* (ADOPTED): `corpus/crate/train.rs`
`train`, `corpus/tokenizer.py` `Tokenizer`; `docs/corpus.md` §5.

**SA-BPE merge score in an incremental BPE trainer.** Merge the pair of highest F·log R (R repositories holding it).
*Papers:* arXiv 2604.14053 (Chizhov, §4.1; §5 Table 3): under-trained tokens 640 → 72 of 646 tested, compression
3.643 vs BPE's 3.623. *Here* (ADAPTED): `corpus/crate/train.rs` `score`, `impl Ord for Job`; `docs/corpus.md` §5; R
counts authorship units; a merge whose bytes are already a token is skipped.

**Vocabulary of 2^15 ids, special tokens inside it.** *Papers:* arXiv 2402.01035 (Dagan, §3.2, Table 3): 32K–256K
vocabulary vs HumanEval pass@1 r = −0.13, p = 0.87; arXiv 2402.19173 (StarCoder2, §6.2): 100K "did not improve
performance". *Here* (ADOPTED): `model/config.py` `VOCAB_SIZE`, `corpus/train_tokenizer.py` `main`;
`docs/corpus.md` §5.

**Pre-tokenization by a split regex, no line break after a punctuation run.** *Papers:* Radford et al. (2019, §2.2):
no merges "across character categories"; arXiv 2608.18062 (Meister, App. F.3, fn. 16): over 18 models, line-break
tokens vs MBPP pass@1 Spearman ρ = −0.717 (p = 8.1e-4); its companion post ("Differing one regex clause"): dropping
the trailing `[\r\n/]*` lifts MBPP pass@1 0.086 → 0.230 (three seeds). *Here* (ADAPTED): `corpus/tokenizer.py`
`SPLIT_PATTERN`; `docs/corpus.md` §5, `docs/experiments.md` §2; nanochat's pattern without `[\r\n]*` there.

**Digit runs of one or two (`\p{N}{1,2}`).** *Papers:* arXiv 2302.13971 (LLaMA, §2.1): single digits; arXiv
2402.14903 (Singh, §1, Table 1): right-to-left 3-digit chunking raises addition accuracy up to 20%; neither tests
k = 2. *Here* (ADOPTED): `corpus/tokenizer.py` `SPLIT_PATTERN`; `docs/corpus.md` §5.

**BOS as the only document token: delimiter, end-of-document target, generation stop.** *Papers:* None. *Here*
(ADAPTED): `corpus/tokenizer.py` `SPECIAL_TOKENS`, `corpus/loader.py` `PretrainingBatches.assemble`,
`inference/decoder.py` `NativeDecoder`; `docs/corpus.md` §5, §9, `docs/evaluation.md` §6; BOS is a target only after a
piece that ends its document or a FIM middle; three FIM sentinels replace the chat tokens.

**Tokenizer comparison by compression ratio.** *Papers:* arXiv 2402.18376 (Schmidt, §5.3): fewer tokens is not
inherently beneficial; Meister 2026 companion post ("Code bits per byte"): no association with MBPP. *Here* (NOT
TAKEN): `metrics.py` `parse_outcome`; `docs/experiments.md` §2, `docs/evaluation.md` §3.

**BOS-aligned rows with best-fit-decreasing packing.** *Papers:* arXiv 2404.10830 (Ding, Abstract, §3.1, Table 2):
BFD adds ≤ 0.01% sequences over concatenation and improves text and code models. *Here* (ADAPTED):
`corpus/crate/pack.rs` `best_fit_decreasing`, `corpus/loader.py` `PretrainingBatches.assemble`; `docs/corpus.md` §8,
§9; cuts at statement or line boundaries, a BOS before every piece.

**ClimbMix as the web-text source.** *Papers:* arXiv 2504.13161 (Diao, §6, Fig. 1): 400B tokens; a 1B model on it
beats Nemotron-CC, SmolLM, DCLM-baseline and FineWeb-Edu at equal tokens. *Here* (ADOPTED): `corpus/sources.py`
`SOURCES`, `corpus/mixture.py` `FILE_LEVEL_STAGE`, `REPO_LEVEL_STAGE`; `docs/corpus.md` §1, §7; the 10% text group.

**Bits per byte, zero-byte targets excluded.** *Papers:* arXiv 2101.00027 (The Pile, §3.1): bpb, "invariance to
different tokenization schemes"; the zero-byte exclusion has no paper. *Here* (ADAPTED): `eval/bpb.py` `sums`,
`corpus/tokenizer.py` `Tokenizer.token_lengths`, `metrics.py` `bpb`, `mixture_bpb`; `docs/evaluation.md` §2, §5.

**Validation budget of 80 × 2^19 tokens.** *Papers:* None (nanochat's size). *Here* (ADAPTED): `eval/bpb.py`
`VAL_TOKENS`, `sample_rows`; `docs/evaluation.md` §2; per corpus, whole micro-batches, rows drawn systematically.

**Execution-based pass@k in a sandbox.** *Papers:* arXiv 2107.03374 (Chen, §2.1, §2.3): functional correctness, the
unbiased estimator, a sandbox. *Here* (PARALLEL): `metrics.py` `pass_cell`, `eval/code_eval.py` `evalplus`,
`multipl_e`, the official tools' judges (`scripts/official_eval.sh`); `docs/evaluation.md` §4, §5.

**Truncated sampling: nucleus (top-p), not top-k.** *Papers:* arXiv 1904.09751 (Holtzman, §3.1, §3.2). *Here*
(PARALLEL): `inference/decoder.py` `NativeDecoder`, `model/kernels/decode.cu` `k_nucleus`; `docs/evaluation.md` §4,
§6.

## 4. Systems and numerics

**FlashAttention kernels for training attention.** *Papers:* arXiv 2205.14135 (Dao, Abstract, §3.2 Thm. 2): exact
tiled attention with far fewer HBM accesses; arXiv 2407.08608 (Shah, Abstract, §3.1): Hopper warp specialization and
pingpong; arXiv 2603.05451 (Zadouri, Abstract, §4): FA4 in CuTe-DSL. *Here* (ADAPTED): `model/ops/flash_attn.py`
`attention`; `docs/architecture.md` §3; FA4's SM90 varlen kernels instead of FA3, no SDPA fallback
(`docs/experiments.md` §5).

**Sliding windows inside the attention kernel.** *Papers:* arXiv 2205.14135 (Dao, §3.3 Prop. 4): block-sparse IO
scales with nonzero blocks; arXiv 2412.05496 (Dong, §4.2, Fig. 3): BlockMask skips fully masked blocks. *Here*
(ADAPTED): `model/modeling.py` `layer_windows`, `model/ops/flash_attn.py` `fa4_window_left`; `docs/architecture.md`
§1, §3; token-exact, not block-aligned (`docs/experiments.md` §3).

**Fixed-size `cu_seqlens`, with only the real segments passed to the kernel.** *Papers:* PyTorch 2 (Ansel, ASPLOS
2024, §5): static-shape compilers "must recompile if any input shape changes"; arXiv 2307.08691 (Dao, §3.2): one
thread block per sequence and row block, so every listed segment gets work. *Here* (ADAPTED): `corpus/loader.py`
`PretrainingBatches`, `model/ops/flash_attn.py` `segmented`; `docs/architecture.md` §3.

**Mixed precision: fp32 master weights, bf16 GEMM operands with fp32 accumulation.** *Papers:* arXiv 1710.03740
(Micikevicius, §3.1): FP32 master weights; arXiv 1905.12322 (Kalamkar, Abstract, §3): BF16 matches FP32 without loss
scaling. *Here* (ADAPTED): `model/modeling.py` `GemmLinear.weight_bf16`, `Attention.qkv_bf16`,
`CausalLM.refresh_gemm_weights`; `docs/architecture.md` §4; persistent bf16 copies refreshed after each update.

**Normalization statistics and the loss in fp32.** *Papers:* arXiv 1710.03740 (Micikevicius, §3.3): large
reductions "should be carried out in FP32". *Here* (ADAPTED): `model/modeling.py` `norm`,
`model/ops/cross_entropy.py` `lm_loss`; `docs/architecture.md` §1, §4; upstream's norm runs in bf16.

**FP8 training.** *Papers:* arXiv 2209.05433 (Micikevicius, §3, §4.1 Table 4): E4M3/E5M2 match 16-bit perplexity at
126M–175B; arXiv 2412.19437 (DeepSeek-V3, §3.3, App. B.1, §3.5.2): fine-grained scaling, loss within 0.25% of BF16;
the backward re-quantizes activations through HBM. *Here* (NOT TAKEN): `model/ops/linear.py`;
`docs/experiments.md` §5; torchao and Transformer Engine FP8 MLP GEMMs slower at d24 [unverified: no record].

**Fused linear cross entropy with the logit cap inside.** *Papers:* arXiv 2411.09009 (Wijmans, Abstract, §4):
log-sum-exp inside the classifier GEMM; arXiv 2410.10989 (Liger, §3.2): the chunked alternative. *Here* (PARALLEL):
`model/ops/cross_entropy.py` `lm_loss`; `docs/architecture.md` §3; QuACK per 8,192-row chunk, tanh on the fp32
accumulator.

**Fused linear + ReLU² GEMM; activations stored for the backward.** *Papers:* arXiv 2104.04473 (Narayanan, §4.2,
§5.8): element-wise fusion +19% GPT-3 175B throughput; PyTorch 2 (Ansel, §6.4 Table 4): epilogue fusion. *Here*
(PARALLEL): `model/ops/linear.py` `relu2_mlp`; `docs/architecture.md` §3; QuACK `gemm_norm_act`, norm as a row scale.

**Fused QK-norm + RoPE kernel.** *Papers:* arXiv 2410.10989 (Liger, §3.2 "RMSNorm", "RoPE"). *Here* (PARALLEL):
`model/kernels/qk_prologue.cu`, `model/kernels/qk_row.cuh`; `docs/architecture.md` §3; row math shared with the
decoder.

**Merged q/k/v projection (one GEMM).** *Papers:* None; the measurement is this repository's. *Here* (PARALLEL):
`model/ops/linear.py` `qkv_proj`; `docs/architecture.md` §3.

**Newton–Schulz batched over same-shape matrices.** *Papers:* arXiv 2608.11612 (Amsel, §7): same-shape matrices as
one batch. *Here* (PARALLEL): `train/optim.py` `Muon.step`; `docs/training.md` §3; on one GPU the gain is kernel
efficiency, not communication.

**`torch.compile` of the whole model, static shapes, one graph.** *Papers:* PyTorch 2 (Ansel, Abstract, §5, §5.1):
1.41× geometric-mean training speedup on A100 over 180+ models; meta functions for custom ops. *Here* (ADOPTED):
`train/loop.py` `train`; `docs/architecture.md` §3; `fullgraph=True`, recompiled at the 32K stage.

**Fused AdamW step.** *Papers:* None (`torch.optim.AdamW(fused=True)`). *Here* (ADOPTED): `train/optim.py`
`optimizers`; `docs/training.md` §3.

**Meta-device construction, then explicit initialization.** *Papers:* arXiv 2304.11277 (FSDP, §3.1): deferred
initialization on a "fake" device. *Here* (ADOPTED): `model/modeling.py` `meta_model`, `CausalLM.init_weights`,
`train/loop.py` `train`; `docs/architecture.md` §1; per-tensor seeded `init_weights`, not op replay.

**Garbage-collector control.** *Papers:* None. *Here* (ADAPTED): `train/loop.py` `train`; `docs/training.md` §8;
collection only after each save.

**Gradient accumulation weighted by valid tokens.** *Papers:* arXiv 2410.02660 (Gao, App. A.3): averaging over
sequences "skews the optimization". *Here* (ADAPTED): `model/ops/cross_entropy.py` `step_shares`, `train/loop.py`
`train`; `docs/training.md` §6; valid-label shares instead of 1/(accumulation steps).

**Overlapping host work with the device step.** *Papers:* None. *Here* (ADAPTED): `train/loop.py` `train`,
`corpus/loader.py` `PretrainingBatches`; `docs/training.md` §6; step values logged one step late via pinned buffers.

**Model FLOPs utilization (MFU).** *Papers:* arXiv 2204.02311 (PaLM, §4.1, App. B): observed over peak throughput,
no rematerialization. *Here* (ADAPTED): `metrics.py` `mfu`, `executed_model_flops`, `PEAK_FLOPS`;
`docs/evaluation.md` §5, `docs/training.md` §7; over 989.4 TFLOP/s (H100 SXM / GH200 dense BF16).

## References

| arXiv | First author, year | Title |
|---|---|---|
| 1211.5063 | Pascanu, 2012 | On the difficulty of training Recurrent Neural Networks |
| 1412.6980 | Kingma, 2014 | Adam: A Method for Stochastic Optimization |
| 1508.07909 | Sennrich, 2015 | Neural Machine Translation of Rare Words with Subword Units |
| 1608.05859 | Press, 2016 | Using the Output Embedding to Improve Language Models |
| 1611.09940 | Bello, 2016 | Neural Combinatorial Optimization with Reinforcement Learning |
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
| 2411.09009 | Wijmans, 2024 | Cut Your Losses in Large-Vocabulary Language Models |
| 2412.05496 | Dong, 2024 | Flex Attention: A Programming Model for Generating Optimized Attention Kernels |
| 2412.19437 | DeepSeek-AI, 2024 | DeepSeek-V3 Technical Report |
| 2501.00656 | Team OLMo, 2024 | 2 OLMo 2 Furious |
| 2502.15938 | Bergsma, 2025 | Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs |
| 2502.16982 | Liu, 2025 | Muon is Scalable for LLM Training |
| 2503.19786 | Gemma Team, 2025 | Gemma 3 Technical Report |
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
- Software and data (provenance): karpathy/nanochat; KellerJordan/modded-nanogpt; karpathy/rustbpe; openai/tiktoken;
  PyTorch (`torch.optim.Muon`, `torch.compile`); QuACK; datasets `karpathy/climbmix-400b-shuffle` and
  `nvidia/Nemotron-ClimbMix`.
