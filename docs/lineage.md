# Literature behind the upstream ideas

Each idea taken or considered from karpathy/nanochat ("nanochat"; its `dev/LOG.md` at `92d63d4` is "LOG") and
KellerJordan/modded-nanogpt ("modded"; "record #N" is row N of its README track-1 table), with the papers that state
or establish it. Commits, records and LOG entries are provenance only, never evidence. Classes: ADOPTED, ADAPTED,
EVIDENCE (an upstream measurement bears on a choice), PARALLEL (same idea, own route), NOT TAKEN. Markers:
[arithmetic] derived from numbers in the code or a paper, [inference] reasoning not measured, [unverified].

## Summary

| Idea | Here | Cite as (arXiv ids) | Section |
|---|---|---|---|
| Pre-norm residual block with a final norm | ADOPTED | 1809.10853, 2002.04745 | §1 |
| RMS normalization without a learnable gain | ADAPTED | 1910.07467, 2402.00838, 2501.00656 | §1 |
| Normalization-free alternative (DyT) | NOT TAKEN | 2503.10622 | §1 |
| RMS-normed token embedding, no positional embedding | ADOPTED | 2211.05100, 2110.02861 | §1 |
| Bias-free linear layers | ADOPTED | 2204.02311 | §1 |
| Untied token embedding and output head | ADOPTED | no paper (design choice; mixed: 1608.05859, 2010.12821, 2402.14905) | §1 |
| RoPE, half-split layout | ADAPTED | 2104.09864 | §1 |
| RoPE base 10^6 from the first step | ADAPTED | 2405.14591, 2407.21783, 2410.06205 (2309.16039) | §1 |
| Partial or truncated RoPE | NOT TAKEN | 2410.06205 | §1 |
| QK-norm with learnable per-dimension gains | ADAPTED | 2302.05442, 2309.14322, 2505.09388, 2410.01131 (2010.04245) | §1 |
| Attention temperature 1/√d, not a hand-set scale | NOT TAKEN | 1706.03762, 2010.04245 | §1 |
| Multi-head attention with all KV heads | ADOPTED | 1706.03762, 2305.13245, 1911.02150 | §1 |
| Head dimension 128 | ADOPTED | no paper for 128 (2002.07028, 2001.08361) | §1 |
| Local–global attention (SSSL, 512-key windows) | ADAPTED | 2004.05150, 2005.14165, 2408.00118, 2503.19786 | §1 |
| Document masking in packed sequences (varlen) | PARALLEL | 2402.13991, 2407.21783, 2410.02660 | §1 |
| Gated attention | NOT TAKEN | 2505.06708 | §1 |
| Multi-token prediction | NOT TAKEN | 2404.19737 | §1 |
| ReLU² activation, hidden 4C | ADOPTED | 2109.08668, 1706.03762 | §1 |
| SwiGLU instead of ReLU² | EVIDENCE | 2109.08668 vs 2002.05202 (conflict); needs own ablation | §1 |
| Mixture of experts | NOT TAKEN | 1701.06538 | §1 |
| Per-layer residual scalars and the x0 shortcut | ADAPTED | 2402.02622, 2410.17897 | §1 |
| Value embeddings on alternating layers | ADAPTED | 2601.22887, 2410.17897 | §1 |
| Value-embedding gate 2σ(x W_g), zero init | ADAPTED | 2601.22887 | §1 |
| Bigram hash embeddings (Engram) | NOT TAKEN | 2601.07372 | §1 |
| Smear, backout, U-net skip, partial key offset | NOT TAKEN | no paper (software) | §1 |
| Output-logit soft cap, c = 15 | ADOPTED | 2408.00118, 1611.09940, 2405.09818, 2309.14322 | §1 |
| Uniform fan-in initialization | ADAPTED | LeCun et al. 1998 (non-arXiv) | §1 |
| Zero-initialized residual-branch outputs | ADOPTED | 1901.09321, 2003.04887, 2002.10444 | §1 |
| Zero-initialized output head | ADOPTED | 1901.09321, 2203.03466 | §1 |
| Depth-scaled norm outputs | NOT TAKEN | 2502.05795 | §1 |
| Shape rule: width = depth × aspect | ADAPTED | 2402.14905, 2001.08361 | §1 |
| Muon | ADAPTED | Jordan et al. 2024 (blog); 2409.20325, 2502.16982, 2509.02046, 2512.05620 | §2 |
| Newton–Schulz iteration for the polar factor | ADOPTED | Björck & Bowie 1971, Kovarik 1970; 2409.20325 | §2 |
| Tuned quintic coefficients, 5 bf16 steps | ADOPTED | Jordan et al. 2024 (blog); 2505.16932, 2502.16982 | §2 |
| Gram-form Newton–Schulz step | ADOPTED | 2608.11612 (no originating paper) | §2 |
| Nesterov momentum 0.95, EMA form | ADOPTED | Sutskever et al. 2013; 2502.16982 | §2 |
| Muon/AdamW parameter split | ADAPTED | Jordan et al. 2024 (blog); 2409.20325, 2502.16982 | §2 |
| Muon step size per shape | ADOPTED (RMS matching); NOT TAKEN (upstream's) | 2502.16982 (2310.17813) | §2 |
| NorMuon | EVIDENCE | 2510.05491 | §2 |
| Polar Express coefficients | NOT TAKEN | 2505.16932 | §2 |
| Cautious weight decay | NOT TAKEN | 2510.12402 (not 2411.16085) | §2 |
| Decoupled weight decay; AdamW; decay in Muon | ADOPTED | 1711.05101, 1412.6980, 2401.02954, 2502.16982 | §2 |
| Weight decay tied to the training horizon | PARALLEL | 2405.13698, 2505.13738 | §2 |
| Weight-decay schedule | NOT TAKEN | no paper (2405.13698 for η_t·λ) | §2 |
| Muon momentum warmup and schedule | NOT TAKEN | no paper (Sutskever et al. 2013 precedent) | §2 |
| Per-group AdamW learning rates and width scaling | NOT TAKEN | no paper (2502.16982 for one lr; 2203.03466) | §2 |
| Learning rates of the residual and x0 scalars | NOT TAKEN | no paper | §2 |
| Learning rate ∝ √(batch size) | NOT TAKEN | 2205.10287, 1404.5997; 2505.13738 | §2 |
| Batch size and peak lr from scaling laws | ADOPTED (DeepSeek's); NOT TAKEN (upstream's) | 2401.02954; 2505.13738 | §2 |
| Batch-size schedule | NOT TAKEN | 1711.00489 | §2 |
| Gradient-norm clipping at 1.0 | ADOPTED | 1211.5063, 2401.02954 | §2 |
| Linear warmup over 2/(1 − β₂) steps | ADOPTED | 1910.04209, 1706.02677 | §2 |
| Trapezoid / WSD schedule, nonzero final lr | NOT TAKEN | 2405.18392, 2404.06395; 2502.15938 | §2 |
| Linear decay to zero | ADOPTED | 2502.15938 | §2 |
| Schedule as a multiplier of each group's lr | ADAPTED | no paper (software) | §2 |
| Hyperparameter transfer from a reference model | NOT TAKEN | 2203.03466, 2512.22382 | §2 |
| Training horizon from tokens per parameter | ADAPTED | 2203.15556; 2505.13738, 2502.15938 | §2 |
| Compute-optimal TPP from upstream's sweeps | NOT TAKEN | 2203.15556, 2406.12907, 2406.19146 | §2 |
| Scaling-parameter count (layers + lm_head) | ADOPTED | 2001.08361, 2406.19146, 2406.12907 | §2 |
| FLOPs per token (plan and model FLOPs) | ADAPTED | 2204.02311, 2001.08361, 2401.02954 | §2 |
| Upstream optimizer sweep: gains shrink with scale | EVIDENCE | 2509.02046, 2512.05620 | §2 |
| Record methodology (n runs, t-test) | EVIDENCE | no paper (software) | §2 |
| Byte-level BPE | ADOPTED | 1508.07909; Radford et al. 2019 | §3 |
| Vocabulary of 2^15 ids | ADOPTED | 2407.13623, 2402.01035, 2402.19173 | §3 |
| Pre-tokenization by a split regex | ADAPTED | 2402.01035; Radford et al. 2019 | §3 |
| Digit runs of one or two | ADOPTED; EVIDENCE | 2302.13971, 2402.14903 (no paper for k = 2) | §3 |
| Punctuation run fused with line breaks | NOT TAKEN | 2608.18062 | §3 |
| Rust BPE trainer, tiktoken encoder | ADOPTED | no paper (software) | §3 |
| Incremental BPE trainer with SA-BPE's score | ADAPTED | 2604.14053 | §3 |
| Merge tie-break | ADOPTED | no paper | §3 |
| BOS as the only delimiter; special-id layout | ADAPTED | no paper (2401.14196 for FIM names) | §3 |
| Chat special tokens | NOT TAKEN | not needed | §3 |
| Tokenizer training sample | NOT TAKEN | 2405.05417, 2402.01035 | §3 |
| Tokenizer compression evaluation | NOT TAKEN | 2608.18062 (2402.18376) | §3 |
| BOS-aligned rows with best-fit packing | ADAPTED | 2404.10830 | §3 |
| BOS as end-of-document target and stop | ADAPTED | no paper | §3 |
| ClimbMix as the web-text source | ADOPTED | 2504.13161 | §3 |
| Data-source negatives against FineWeb-Edu | EVIDENCE | 2406.17557, 2504.13161 | §3 |
| Validation split = last parquet shard | NOT TAKEN | not needed | §3 |
| Bytes per token, 0 for special tokens | ADOPTED | 2101.00027 | §3 |
| Bits per byte, zero-byte targets excluded | ADAPTED | 2101.00027 | §3 |
| Validation budget of 80 × 2^19 tokens | ADAPTED | no paper for the value (2411.00640) | §3 |
| CORE metric | NOT TAKEN | 2406.11794 | §3 |
| HumanEval with upstream's sandbox | NOT TAKEN | 2107.03374 | §3 |
| Temperature + top-k sampling | NOT TAKEN | 1904.09751 | §3 |
| FlashAttention kernels | NOT TAKEN (FA3); FA4 used | 2205.14135, 2407.08608, 2603.05451, 2307.08691 | §4 |
| Sliding windows inside the attention kernel | ADAPTED | 2205.14135, 2412.05496 | §4 |
| Fixed-size `cu_seqlens`, real segments only | ADAPTED | 2307.08691, 2603.05451; Ansel et al. 2024 | §4 |
| Mixed precision: fp32 master, bf16 GEMMs | ADAPTED | 1710.03740, 1905.12322 | §4 |
| fp16 with loss scaling | NOT TAKEN | 1905.12322, 1710.03740 | §4 |
| Reductions, softmax, loss statistics in fp32 | ADAPTED | 1710.03740, 2407.08608 | §4 |
| FP8 training | NOT TAKEN | 2209.05433, 2412.19437 (2310.18313) | §4 |
| Fused linear cross entropy with the cap inside | PARALLEL | 2411.09009, 2410.10989 | §4 |
| Fused GEMM epilogues; stored activations | PARALLEL | 2104.04473, 2205.05198, 1604.06174, 2410.10989; Ansel et al. 2024 | §4 |
| Fused QK-norm + RoPE kernel | PARALLEL | 2410.10989 | §4 |
| Merged q/k/v projection | PARALLEL | no paper | §4 |
| Newton–Schulz batched over same-shape matrices | PARALLEL | 2504.05295, 2608.11612 | §4 |
| torch.compile, static shapes, one graph | ADOPTED | Ansel et al. 2024 (ASPLOS, non-arXiv) | §4 |
| CUDA graphs | NOT TAKEN (training); ADOPTED (decoder) | Ansel et al. 2024; 2608.11612 | §4 |
| Fused and compiled optimizer steps | ADAPTED | no paper (software) | §4 |
| Meta-device construction, seeded initialization | ADOPTED | 2304.11277 | §4 |
| Garbage-collector control | ADAPTED | no paper | §4 |
| Gradient accumulation, per-token weighting | ADAPTED | 2410.02660 | §4 |
| Overlapping host work with the device step | ADAPTED | no paper | §4 |
| Allocator `expandable_segments` | NOT TAKEN | not needed | §4 |
| FlashAttention-3 with an SDPA fallback | NOT TAKEN | not needed | §4 |
| Model FLOPs utilization and the peak | ADAPTED | 2204.02311 | §4 |

## 1. Architecture

### Pre-norm residual block with a final norm before the head
- **Idea.** Normalize the input of each sublayer, x + F(norm(x)), and normalize the stream once more before the
  output layer. With the norm inside the residual branch the gradients at initialization are well behaved, so the
  model trains without the long learning-rate warm-up post-norm needs.
- **Literature.** arXiv 1809.10853 (Baevski et al., 2018, §4.1): applies layer normalization "before the
  self-attention and FFN blocks instead of after, as we find it leads to more effective training". arXiv 2002.04745
  (Xiong et al., 2020, §3.3, Theorem 1, §4): defines the Pre-LN Transformer including its final layer normalization;
  proves the last-layer gradient of Post-LN is O(d√ln d) independent of L, while Pre-LN's are scaled down by √L;
  Pre-LN trains without warm-up to comparable results in less time.
- **Upstream.** modded `4a452a2` (first script: pre-norm, `rmsnorm` before `lm_head`); nanochat `3a5e0bc`
  (`gpt.py` `Block.forward`). Neither cites a paper.
- **Here.** ADOPTED; `model/modeling.py` `DecoderLayer.forward`, `CausalLM.hidden` (final `norm`);
  `docs/architecture.md` §1. The post-attention norm's statistics come from the o_proj GEMM epilogue (kernel-level
  only; `docs/architecture.md` §3).
- **Cite as.** arXiv 1809.10853 §4.1 (pre-norm placement) and arXiv 2002.04745 (the Pre-LN Transformer with final
  norm, its gradient analysis and warm-up result).

### RMS normalization without a learnable gain
- **Idea.** RMSNorm drops LayerNorm's re-centering and keeps only re-scaling by the root mean square, hypothesizing
  that re-scaling invariance is what makes LayerNorm work; it is cheaper. Here the norm also has no gain (no affine
  parameters), the "non-parametric" form.
- **Literature.** arXiv 1910.07467 (Zhang & Sennrich, 2019, §4 Eq. 4; Abstract): RMSNorm, comparable quality to
  LayerNorm with 7%–64% less running time; the paper's RMSNorm has a gain g (initialized to 1). arXiv 2402.00838
  (Groeneveld et al., 2024, OLMo, §2.1): "non-parametric formulation of layer norm … no 'adaptive gain' (or bias)",
  chosen as "the safest option and … the fastest". arXiv 2501.00656 (OLMo 2, 2024, §3.3.1): "Our ablations show no
  difference between" the non-parametric layer norm and RMSNorm.
- **Upstream.** nanochat `3a5e0bc` ("Purely functional rmsnorm with no learnable params"); modded `4a452a2`,
  `F.rms_norm` from `1d954e9` (record #5). Evidence: LOG 2026-01-29 (Hyperball/MuonH branch): learnable RMSNorm
  scales "still worse".
- **Here.** ADAPTED; `model/modeling.py` `norm` (`F.rms_norm`, eps `EPS` = FLT_EPSILON, fp32 statistics);
  `docs/architecture.md` §1.
- **Cite as.** arXiv 1910.07467 (RMSNorm); arXiv 2402.00838 §2.1 for the gain-free (non-parametric) choice, with
  arXiv 2501.00656 §3.3.1 for its parity with parametric RMSNorm. The eps value is a numerical choice (no paper).

### Normalization-free alternative (DyT)
- **Idea.** Replace every normalization by DyT(x) = γ·tanh(αx) + β.
- **Literature.** arXiv 2503.10622 (Zhu et al., 2025, §4 Table 4, Fig. 6): LLaMA 7B–70B on 200B tokens with DyT
  match RMSNorm's loss; App. C and "Limitations": DyT "offers no speedup over models with normalization layers when
  properly compiled/optimized".
- **Upstream.** LOG 2026-05-05 (`dc54a1a`): no DyT variant beat RMSNorm at d12, even per step; ~10% lower tok/s.
- **Here.** NOT TAKEN (RMSNorm kept); not discussed in the docs.
- **Cite as.** arXiv 2503.10622 if the alternative is mentioned (parity at 7B–70B, no compiled speedup). The nanochat
  d12 result is an unreviewed observation (cite the repository as software, or run an own ablation).

### RMS-normed token embedding, no positional embedding
- **Idea.** x0 = norm(E[ids]): normalizing right after the embedding lookup keeps the stream's scale independent of
  the table's; position enters only through RoPE in attention.
- **Literature.** arXiv 2211.05100 (BLOOM, 2022, §3.2.3 "Embedding LayerNorm"): a layer norm right after the
  embedding (bitsandbytes' StableEmbedding) "significantly improved training stability" at 104B in float16, but
  "penalizes zero-shot generalization"; "bfloat16 [may alleviate] the need". arXiv 2110.02861 (Dettmers et al.,
  2021, §2.3): the stable embedding layer (layer norm on the embedding output) reduces gradient variance from the
  non-uniform token distribution.
- **Upstream.** modded `5fe33da`, record #8 ("added RMSNorm after embed"); nanochat `3a5e0bc` ("norm after token
  embedding", "no positional embeddings").
- **Here.** ADOPTED; `CausalLM.hidden` (`x0 = norm(F.embedding(...))`); `docs/architecture.md` §1 table "Embedding".
- **Cite as.** arXiv 2211.05100 §3.2.3 and arXiv 2110.02861 §2.3 for a norm after the embedding (both LayerNorm, both
  a stability motive). No paper measures a gain-free RMS norm here; the choice is upstream's.

### Bias-free linear layers
- **Idea.** Omit additive biases in every projection; reported to improve large-model training stability.
- **Literature.** arXiv 2204.02311 (PaLM, 2022, §2 "No Biases"): "We found this to result in increased training
  stability for large models". arXiv 2302.05442 (ViT-22B, 2023, §2): "omitted biases" as one of three
  stability/efficiency changes; arXiv 2402.00838 (OLMo, §2.1): "No biases … to improve training stability".
- **Upstream.** nanochat `3a5e0bc` ("no bias in linear layers"); modded from its first script.
- **Here.** ADOPTED; `nn.Linear(..., bias=False)`, `GemmLinear`; `docs/architecture.md` §1.
- **Cite as.** arXiv 2204.02311 §2.

### Untied token embedding and output head
- **Idea.** Separate input and output matrices, giving the head capacity independent of the input table.
- **Literature.** arXiv 1608.05859 (Press & Wolf, 2016, Abstract, §1): tying input and output embeddings improves
  perplexity of several LMs (the counter-position). arXiv 2010.12821 (Chung et al., 2020, Abstract): decoupled
  embeddings give "increased modeling flexibility" and, with the input parameters reallocated, better NLU results at
  equal size (multilingual encoders). arXiv 2402.14905 (MobileLLM, 2024, §2.2.3 Table 1): at 125M, sharing saves 16M
  parameters (~11.8%) for a 0.2-point average-accuracy drop.
- **Upstream.** modded `5fe33da`, record #8 ("untied embed and head weights"; later retied, #51, and untied at 2/3
  of training, #53); nanochat `3a5e0bc` ("untied weights for token embedding and lm_head").
- **Here.** ADOPTED; `CausalLM.embed_tokens`, `CausalLM.lm_head`; `docs/architecture.md` §1.
- **Cite as.** No paper establishes untying as better for decoder LMs at this scale; the evidence is mixed (arXiv
  1608.05859 for tying; arXiv 2010.12821 and arXiv 2402.14905 Table 1 for a small cost of tying). Frame as a design
  choice inherited from nanochat/modded (record #8 is a speedrun observation, not citable as evidence).

### Rotary position embedding (RoPE), half-split layout
- **Idea.** Rotate each 2-D pair of q and k by angle t·θ_i with θ_i = base^(−2i/d); the q·k product then depends
  only on relative position, with a long-term decay of the inner product.
- **Literature.** arXiv 2104.09864 (Su et al., 2021, §3.2.2 Eq. 15 and θ_i = 10000^(−2(i−1)/d); §3.3 long-term
  decay; §3.4.3).
- **Upstream.** modded `3e48a54`, record #2; nanochat `3a5e0bc` (`apply_rotary_emb`), `3616a02` (comment: rotates
  by −θ).
- **Here.** ADAPTED (base: next entry); `model/kernels/qk_row.cuh`, `CausalLM.init_buffers`;
  `docs/architecture.md` §1. The −θ sign and the half-split pairing (dims i, i + d/2) are conventions; the
  relative-position property is unchanged [inference].
- **Cite as.** arXiv 2104.09864. No paper for the sign/layout convention (none needed).

### RoPE base 10^6 from the first step
- **Idea.** Larger bases lower the lowest frequencies, so the model can resolve and attend over longer distances; a
  lower bound on the base grows with the context length.
- **Literature.** arXiv 2405.14591 (Men et al., 2024, §4.3 Table 2): lower bound of the base 6.4e5 for a 32K context
  (8.4e4 at 8K). arXiv 2309.16039 (Xiong et al., 2023, §4.1, "RoPE ABF"): raising the base frequency from 10,000 to
  500,000 reduces RoPE's decay on distant tokens for long-context continual pretraining. arXiv 2407.21783 (Llama 3,
  2024, §3.2): base 500,000, "effective for context lengths up to 32,768" (citing Xiong et al.). arXiv 2410.06205
  (Barbero et al., 2024, §6.1 Table 2): Gemma 2B at 8K, base 10k vs 500k: 4.4627 vs 4.4485 (Wiki), 6.4429 vs 6.4593
  (FlanV2).
- **Upstream.** nanochat `3a5e0bc` (10^4), `6ed7d1d` (10^5); modded 10^4.
- **Here.** ADAPTED; `model/config.py` `ROPE_THETA` = 1,000,000; `docs/architecture.md` §2.
- **Cite as.** arXiv 2405.14591 §4.3 Table 2 (bound for 32K), arXiv 2407.21783 §3.2 (large base from the start),
  arXiv 2410.06205 §6.1 Table 2 (small effect at 8K).

### Partial or truncated RoPE
- **Idea.** Rotate only part of each head; the unrotated dimensions act as position-free semantic channels.
- **Literature.** arXiv 2410.06205 (§6.1 Table 2): 0.75-RoPE 4.4414 / 6.4422 (best), 0.25-RoPE 4.5302 / 6.5111
  (worse than RoPE 10k's 4.4627 / 6.4429), Gemma 2B at 8K.
- **Upstream.** modded record #17 (`2d019fa`, half the dims, base 1024); LOG 2026-01-16 ("Half-truncated RoPE: No
  improvement"); LOG 2026-03-24 (`4e1694c`, a quarter of each head: "Slightly worse").
- **Here.** NOT TAKEN (full-head RoPE); not discussed.
- **Cite as.** arXiv 2410.06205 §6.1 Table 2 if mentioned (its 0.25-RoPE is worse, consistent with the upstream
  observation; its 0.75-RoPE is better, so the literature does not support "full head is best").

### QK-norm with learnable per-dimension gains shared by heads
- **Idea.** Normalize q and k per head before the dot product so attention logits cannot grow with parameter norms
  (attention-entropy collapse); a learned gain restores the logit scale.
- **Literature.** arXiv 2010.04245 (Henry et al., 2020, Abstract, §3): QKNorm, ℓ2-normalized q and k scaled by a
  learnable parameter instead of 1/√d (+0.928 BLEU, low-resource MT). arXiv 2302.05442 (Dehghani et al., 2023,
  ViT-22B, §2, App. B, Fig. 1): divergence at ~8B from attention logits "over 50000 in magnitude", one-hot
  attention; LayerNorm on q and k fixes it. arXiv 2309.14322 (Wortsman et al., 2023, §3.1.1, Fig. 1–2): reproduces
  attention-logit growth in small models at high LR; qk-layernorm lowers LR sensitivity (1.2B trains at LR 0.3);
  §3.2.5, Fig. E.8: per-head qk-layernorm with parameters shared across heads beats normalizing across the model
  dimension. arXiv 2405.09818 (Chameleon, 2024, §2.3), arXiv 2501.00656 (OLMo 2, §3.3.2), arXiv 2505.09388 (Qwen3,
  §2): QK-norm adopted for stable training at scale. arXiv 2410.01131 (nGPT, 2024, App. A.9 Table 5): fixing
  s_qk = 1 instead of a learned vector costs +0.11% validation loss; a single scalar +0.22% (0.5B, 1K context).
- **Upstream.** modded `1d954e9`/`7e1f20b`, record #5 (gain-free, after then before RoPE; record #20's README cites
  ViT-22B and Chameleon); nanochat `3a5e0bc` (RoPE, then gain-free norm), `6ed7d1d` ×1.15, `a825e63` ×1.2.
- **Here.** ADAPTED; `Attention.q_norm_weight`/`k_norm_weight`, `model/ops/qk_prologue.py`,
  `model/kernels/qk_prologue.cu` (norm → gain → RoPE); `docs/architecture.md` §1, §2 "Learnable QK-norm gains". The
  per-dimension gains follow Qwen3 (`transformers` `modeling_qwen3.py`); the norm → gain → RoPE order has no paper
  basis [inference: with a gain the order matters; without one it does not]. The fused kernel: §4, Fused QK-norm +
  RoPE kernel.
- **Cite as.** arXiv 2302.05442 §2 (QK-norm), arXiv 2309.14322 §3.1.1 and §3.2.5 (mechanism at small scale;
  per-head shared parameters), arXiv 2505.09388 §2 (adoption); arXiv 2410.01131 App. A.9 Table 5 for learned vs
  fixed gain.

### Attention temperature: fixed 1/√d, not a hand-set scale
- **Idea.** After QK-norm the logits are bounded by the gain; a fixed extra scale (modded 0.12, nanochat ×1.2 on q
  and k) sharpens attention. Here the scale stays 1/√d and the learned gains carry the temperature.
- **Literature.** arXiv 1706.03762 (Vaswani et al., 2017, §3.2.1): scaling by 1/√d_k "to counteract" softmax
  saturation. arXiv 2010.04245 (Abstract, §3): a learnable scale replaces 1/√d after normalization.
- **Upstream.** modded record #20 (`ebdeb37`, `attn_scale = 0.12`; README: a learned per-layer scale or channel-wise
  gains gave the same 20-step gain); nanochat `6ed7d1d`, `a825e63`.
- **Here.** NOT TAKEN; `model/ops/flash_attn.py` softmax scale 1/√128; `docs/architecture.md` §2 "Lineage".
- **Cite as.** arXiv 1706.03762 §3.2.1 for 1/√d; arXiv 2010.04245 for learnable scaling of normalized logits. The
  0.12 value has no paper (speedrun tuning).

### Multi-head attention with all KV heads (not GQA/MQA)
- **Idea.** One key/value head per query head. MQA/GQA share KV heads to cut decode memory bandwidth and KV-cache
  size at some quality cost.
- **Literature.** arXiv 1706.03762 (§3.2.2): multi-head attention. arXiv 1911.02150 (Shazeer, 2019, Abstract): MQA
  decodes much faster with "only minor quality degradation". arXiv 2305.13245 (Ainslie et al., 2023, Abstract,
  Table 1): GQA reaches quality close to MHA at speed close to MQA.
- **Upstream.** nanochat `3a5e0bc` (n_kv_head = n_head; supports GQA, unused).
- **Here.** ADOPTED; `Attention` (C × C q/k/v); `docs/architecture.md` §2 (KV-cache arithmetic); the d16 GQA screen
  in `docs/experiments.md` §3.
- **Cite as.** arXiv 1706.03762 (MHA); arXiv 2305.13245 and arXiv 1911.02150 for the alternative and its rationale
  (decode memory). The GQA screen is this repository's own evidence.

### Head dimension 128
- **Idea.** Fewer, wider heads: each head's attention has rank ≤ head size, so small heads limit what one head can
  represent.
- **Literature.** arXiv 2002.07028 (Bhojanapalli et al., 2020, Abstract, §3 Theorem 1): the head-size/number-of-
  heads coupling causes a low-rank bottleneck; head size should not shrink as heads are added. arXiv 1706.03762
  (§6.2, Table 3 rows A–B): reducing d_k hurts; too many heads hurts. arXiv 2001.08361 (Kaplan et al., 2020, §3.1,
  Fig. 5): at fixed N, loss depends only weakly on n_heads (hence head size).
- **Upstream.** modded `1d954e9`, record #5 ("head dim 128 suggested by @Grad62304977"); nanochat `3a5e0bc`,
  `ccf4b7f`; LOG 2026-01-17 ("Head dim definitely prefers to be 128 instead of 64").
- **Here.** ADOPTED; `model/config.py` `HEAD_DIM` (compiled into the kernels); `docs/architecture.md` §2.
- **Cite as.** No paper selects 128. arXiv 2002.07028 for why larger heads help, arXiv 2001.08361 §3.1 Fig. 5 for the
  weak dependence; the value is a design choice (also the kernels' tile size). The LOG result is not citable.

### Local–global attention: SSSL pattern, last layer global, 512-key windows
- **Idea.** Most layers attend a fixed window of recent keys (linear cost, small KV cache); a few full-context layers
  carry long-range information; stacked windows still widen the receptive field.
- **Literature.** arXiv 1904.10509 (Child et al., 2019): sparse/local attention patterns. arXiv 2004.05150 (Beltagy
  et al., 2020, Longformer, §3.1): sliding-window attention; stacked layers give a receptive field l × w. arXiv
  2005.14165 (Brown et al., 2020, GPT-3, §2.1): "alternating dense and locally banded sparse attention patterns in the
  layers … similar to the Sparse Transformer". arXiv 2310.06825 (Mistral 7B, 2023, §2): SWA with W = 4096; reach
  W × k tokens after k layers. arXiv 2408.00118 (Gemma 2, 2024, §2): local (4096) and global (8192) attention in every
  other layer. arXiv 2503.19786 (Gemma 3, 2025, §2; §5.2 Fig. 3–4): 5 local : 1 global with a 1024-token local span;
  "minimal impact on perplexity" when changing the local:global ratio; the window "can be reduced significantly
  without impacting perplexity" (2B models).
- **Upstream.** nanochat `fbc1484` ("following GPT-3"; SSSL, last layer forced long), `b33e394`, `e7ed208`; short
  window seq/2 → seq/3 → seq/4 (`6ed7d1d`, `a825e63`); LOG 2026-01-11. modded record #20 (`ebdeb37`; README cites
  Gemma 2 and Longformer); block-aligned windows #15/#16 (not taken; §4, Sliding windows inside the attention kernel).
- **Here.** ADAPTED; `model/modeling.py` `layer_windows` (i ≡ 3 mod 4 or last: full; else 512 keys), FA4
  `window_size_left`; `docs/architecture.md` §1, §2 "SSSL windows of 512".
- **Cite as.** arXiv 2004.05150 §3.1 (sliding window), arXiv 2005.14165 §2.1 (alternating local/dense layers), arXiv
  2408.00118 §2 and arXiv 2503.19786 §2, §5.2 (interleaved local/global; ratio and window ablations). The SSSL period
  and the forced-global last layer have no paper: nanochat's design choice.

### Document masking in packed sequences (varlen attention)
- **Idea.** In a row packed with several documents, causal attention lets tokens attend to unrelated earlier
  documents. Restricting each token to its own document removes this distracting context. Given to the kernel as
  cumulative offsets (`cu_seqlens`), each document is its own causal segment and the masked cross-document blocks are
  never computed.
- **Literature.** arXiv 2402.13991 (Zhao et al., 2024, Abstract, §3.1–§3.2, Table 1): 1.3B models on SlimPajama;
  intra-document causal masking (IntraDoc) gives the lowest average perplexity, 8.410 vs 9.293 (MixChunk) at 2K and
  8.079 vs 9.065 at 8K; largest gains on GitHub; 4.0% efficiency cost in their implementation (§3.1). arXiv
  2404.10830 (Ding et al., 2024, App. C.3, Table 10): the perplexity gain (9.64 → 9.53) is attributable to the
  attention mask; downstream gains need Best-fit Packing as well. arXiv 2407.21783 (Llama 3, 2024, §3.2): the
  document mask had "limited impact during … standard pre-training" but was "important in continued pre-training on
  very long sequences". arXiv 2410.02660 (Gao et al., 2024, ProLong, §6, App. B.2 Table 20; App. A.3): document masks
  improve both long- and short-context results in continued long-context training; FlashAttention 2's varlen
  implementation computes attention per document for speed. arXiv 2412.05496 (Dong et al., 2024, §3.1, §6.3): the
  document mask as a FlexAttention `mask_mod`. arXiv 2107.02027 (Krell et al., 2021, Abstract): packing without
  cross-contamination.
- **Upstream.** modded record #12 (`ebce774`, FlexAttention `document_causal_mask`), #29 (`8386811`, PR #118, FA3
  `flash_attn_varlen_func`, fixed-size `cu_seqlens`); nanochat LOG 2026-01-13 "Varlen Attention (Negative Result)"
  (`f92efce`): d16 val bpb 0.85427 → 0.85407 at a 2K context, MFU and tok/s about equal, "not worth the code
  complexity", "similar to modded-nanogpt's approach"; not merged.
- **Here.** PARALLEL (upstream's measurement is contrary; the docs do not cite it); `model/ops/flash_attn.py`
  `attention` (FA4 varlen over `cu_seqlens`, one segment per packed piece of a document), `corpus/loader.py`
  `cu_seqlens`; `docs/architecture.md` §1, `docs/corpus.md` §8–§9. The fixed-size offsets: §4, Fixed-size
  `cu_seqlens`.
- **Cite as.** arXiv 2402.13991 §3.2 Table 1 (mechanism and the perplexity gain at 2K/8K and on code), arXiv
  2407.21783 §3.2 and arXiv 2410.02660 App. B.2 (when it matters: long context; App. A.3 for varlen as the
  implementation). Llama 3's "limited impact" at standard lengths agrees with nanochat's 2K null result [inference]
  and is its citable form; the nanochat measurement (one d16 run) is not citable evidence.

### Gated attention (output gate)
- **Idea.** A per-head sigmoid gate on the attention output adds non-linearity and query-dependent sparsity.
- **Literature.** arXiv 2505.06708 (Qiu et al., 2025, §1, Table 1): a head-specific sigmoid gate after SDPA gives up
  to 0.2 PPL and 2 MMLU points (15B MoE, 400B tokens) and removes the attention sink.
- **Upstream.** modded record #28 (`9d9dc96`, sparse attention gate on 12 channels); LOG 2026-01-17 (`1933e85`): "No
  improvement", +1 GB.
- **Here.** NOT TAKEN; `docs/experiments.md` §3 lists gated attention as not run.
- **Cite as.** arXiv 2505.06708 for the idea and its published gain; the nanochat null result is not citable.

### Multi-token prediction
- **Idea.** Extra heads predict tokens t+2…t+n as an auxiliary loss.
- **Literature.** arXiv 2404.19737 (Gloeckle et al., 2024, §3.1 Fig. 3): on code, n-token models "are worse than the
  baseline for small model sizes, but outperform the baseline at scale" (0.3B–13B; MBPP, HumanEval).
- **Upstream.** modded record #53 (`9060ff4`); LOG 2026-01-12 (`4610a83`): d12 memory 34 → 47 GB, per step same or
  slightly worse, wall clock worse.
- **Here.** NOT TAKEN; `docs/experiments.md` §3.
- **Cite as.** arXiv 2404.19737 §3.1 Fig. 3 (small models lose).

### ReLU² activation, hidden 4C
- **Idea.** relu(x)² keeps ReLU's sparsity with a smooth, growing positive side; it overlaps ReGLU when the two
  projections are equal, capturing much of the GLU gain without a third matrix.
- **Literature.** arXiv 2109.08668 (So et al., 2021, Primer, §3, Fig. 5): squared ReLU is "the most effective"
  Primer modification; better than GELU, Swish and the GLU variants in their C4 LM runs; Primer-EZ = squared ReLU +
  depthwise conv. arXiv 1706.03762 (§3.3): d_ff = 4·d_model (2048 for 512).
- **Upstream.** modded `072ee59`, record #5; code comment "https://arxiv.org/abs/2109.08668v2; ~1-2% better than
  GELU". nanochat `3a5e0bc`. Evidence: LOG 2026-03-24 (`4e1694c`): LeakyReLU(0.5)² slightly better per step, net
  worse on wall clock.
- **Here.** ADOPTED; `model/ops/linear.py` `relu2_mlp` (fused in QuACK epilogues), `decode.cu` `EPI_RELU2`;
  `docs/architecture.md` §1–§3.
- **Cite as.** arXiv 2109.08668 §3 (squared ReLU); arXiv 1706.03762 §3.3 (4× width).

### SwiGLU instead of ReLU²
- **Idea.** Gated MLP Swish(xW)⊗xV with hidden 8/3·C to match parameters and FLOPs.
- **Literature.** arXiv 2002.05202 (Shazeer, 2020, §2, Table 1): at matched parameters and compute, SwiGLU 1.636 vs
  ReLU 1.677 log-perplexity (524,288 steps); GEGLU/SwiGLU best. arXiv 2109.08668 (Fig. 5): squared ReLU better than
  these GLU variants in Primer's LM runs. arXiv 2402.14905 (MobileLLM, App. Table 10): FC→ReLU→FC to SwiGLU improves
  accuracy at 125M.
- **Upstream.** LOG 2026-02-05 (`d63b7ab`): SwiGLU at matched parameters and FLOPs worse at d12 and d24 on steps,
  wall clock and FLOPs.
- **Here.** EVIDENCE; `docs/architecture.md` §2 "ReLU², not SwiGLU" (plus the decoder's fixed `EPI_RELU2`).
- **Cite as.** The published comparisons conflict: arXiv 2002.05202 Table 1 favors SwiGLU over plain ReLU, arXiv
  2109.08668 Fig. 5 favors squared ReLU over SwiGLU. Cite Primer for the choice; the nanochat d12/d24 result is not
  citable as evidence (an own ablation, or a software citation as an unreviewed observation).

### Mixture of experts
- **Idea.** Route each token to a few expert MLPs: more parameters at constant active FLOPs.
- **Literature.** arXiv 1701.06538 (Shazeer et al., 2017, Abstract): >1000× model capacity "with only minor losses in
  computational efficiency".
- **Upstream.** LOG 2026-02-19 (`2dffdc8`): per step better, d18 MFU ~46% → ~35%, net negative on wall clock.
- **Here.** NOT TAKEN; not discussed.
- **Cite as.** arXiv 1701.06538 for MoE; the throughput-loss observation is nanochat's (not citable).

### Per-layer residual scalars and the x0 shortcut
- **Idea.** Before layer i, x ← λr_i·x + λ0_i·x0, with x0 the normed embedding; λr = 1, λ0 = 0 at init, so the
  network starts as the plain residual stack and learns how much raw token information to re-inject at depth.
- **Literature.** arXiv 2402.02622 (Pagliardini et al., 2024, DenseFormer, §3; Fig. 5): depth-weighted averaging
  over all earlier outputs including the embedding X0, initialized at α_ii = 1, others 0 (the plain Transformer);
  learned weights put large weight on the input embeddings; better perplexity at equal depth (Table 1). arXiv
  2410.17897 (Zhou et al., 2024, ResFormer, Abstract, §3.2): hidden-state residuals "fail to adequately preserve
  initial token-level information in deeper layers"; a residual from layer 1's values reaches equal loss with 16.11%
  fewer parameters and 20.3% less data. arXiv 2003.04887 (Bachlechner et al., 2020, ReZero, §1 Eq. 6): learned
  per-layer residual weights α_i (init 0); a 12-layer Transformer converges 56% faster on enwik8. arXiv 2603.15031
  (Kimi Team, 2026, "Attention Residuals", Abstract; §5.1 Table 2): learned attention over earlier layer outputs;
  Block AttnRes matches a baseline trained with 1.25× compute.
- **Upstream.** modded `e59ae87`, record #9 ("embed shortcut", λ = (1, 0); README: no paper for the shortcut, the
  learnable λ is "@Grad62304977's innovation on top of" arXiv 2410.17897); nanochat `aa530cd`, LOG 2026-01-11
  (d8–d20: −0.0103 … −0.0039 bpb); decaying inits `a825e63` (not taken).
- **Here.** ADAPTED; `CausalLM.resid_lambdas`, `x0_lambdas`, `hidden`, `init_weights` (1 / 0);
  `docs/architecture.md` §1 "Residual". Block AttnRes: `docs/experiments.md` §3. The scalars' learning rates: §2,
  Learning rates of the residual and x0 scalars.
- **Cite as.** arXiv 2402.02622 (learned, identity-initialized mixing with the embedding output) and arXiv 2410.17897
  (motivation: deep layers lose token-level information). The exact two-scalar form is modded's (record #9) and has
  no paper; the nanochat sweep is not citable.

### Value embeddings on alternating layers
- **Idea.** Per-layer token-indexed tables added to the attention values: parameters at almost no FLOPs, a direct
  token-identity path into each layer's values.
- **Literature.** arXiv 2601.22887 (MoVE, 2026, §4.1): LaVE, a baseline "adapted from the modded-nanoGPT repository"
  (per-layer memory M_t added to V with gate g = 2σ(x W_g), on half the layers L−1, L−3, …); §4.2 Table 1: LaVE ×1
  lowers val bpb by 0.016, 0.010, 0.010 at D12, D20, D32 (nanochat models). arXiv 2410.17897 (§3.2): value residual
  from the first layer, the idea modded's tables grew from.
- **Upstream.** modded `77732eb`, record #14 (code comment: "inspired by @Grad62304977's value residual
  implementation following https://arxiv.org/abs/2410.17897"); nanochat `e3f58b8`, `0b58d70`, `9a88194`, `e85db6b`
  (alternating, `has_ve`), LOG 2026-01-16/17 (alternating best; sharing and low-rank fail).
- **Here.** ADAPTED; `CausalLM.value_embeds`, `has_ve` (12 of 24 layers), `model/ops/ve_gate.py`;
  `docs/architecture.md` §1, §2. Value residual itself (modded #9) not taken.
- **Cite as.** arXiv 2601.22887 §4.1–4.2 (the mechanism as LaVE and its measured gain at nanochat scales); arXiv
  2410.17897 as the precursor. The repository (record #14, nanochat) is the origin; cite it as software.

### Value-embedding gate 2σ(x W_g), zero init
- **Idea.** A per-head gate in (0, 2) scales the table's contribution; zero-initialized W_g gives g = 1, a neutral
  start that can amplify or suppress.
- **Literature.** arXiv 2601.22887 (§3.2): "scaled sigmoid … to the range (0, 2) … initializes the gates (at z = 0)
  to 1.0 … neutral identity mixing"; §4.1: LaVE reads the full hidden width d (nanochat reads 32 channels).
- **Upstream.** modded `5a0486a`, record #55 (PR #186; 12 channels; README: loss ~3.2830 without the gates, 7 runs
  mean 3.2788); nanochat `e85db6b` (2σ, zero init, 32 channels); `6ed7d1d` (3σ, 12 channels, uniform(0, 0.02)).
- **Here.** ADAPTED; `Attention.ve_gate` (H × C, full width), `model/kernels/ve_gate.cu`; `docs/architecture.md` §1,
  §2.
- **Cite as.** arXiv 2601.22887 §3.2 and §4.1 (the 2σ zero-centered gate, full-width input). Origin record #55.

### Bigram hash embeddings (Engram)
- **Idea.** O(1) hashed n-gram lookups as a static memory module next to the Transformer.
- **Literature.** arXiv 2601.07372 (DeepSeek, 2026, Abstract): Engram-27B beats an iso-parameter, iso-FLOPs MoE
  baseline (MMLU +3.4, CMMLU +4.0).
- **Upstream.** modded record #62 (PR #201); LOG 2026-01-27/28 (`c8d93be`, reverted `74554be`): helped at d12, gone
  at d25 on wall clock, VRAM cost.
- **Here.** NOT TAKEN; not discussed.
- **Cite as.** arXiv 2601.07372 if mentioned.

### Smear, backout, U-net skip, partial key offset
- **Idea.** Speedrun-specific stream edits: mix each token embedding with its predecessor (smear), subtract a
  mid-layer residual at the end (backout), U-net encoder→decoder skips, shift keys one position on some head dims.
- **Literature.** None found.
- **Upstream.** modded #34 (`d149ec4`), #40 (`ba3e54f`), #11, #49 (`28fda1e`); LOG 2026-01-16: smear negligible,
  backout no improvement, a 25%→50% skip slightly worse; nanochat later adopted smear/backout (`a825e63`).
- **Here.** NOT TAKEN; `docs/architecture.md` §2 "Lineage", `docs/experiments.md` §3 (partial key offset).
- **Cite as.** No paper: speedrun techniques; cite the repository (records #11, #34, #40, #49) as software.

### Output-logit soft cap c·tanh(z/c), c = 15
- **Idea.** Bound the output logits smoothly to (−c, c) so they cannot drift; QK-norm bounds only attention logits.
- **Literature.** arXiv 1611.09940 (Bello et al., 2016, §5.1; App. A.2 Eq. 16): "logit clipping" softmax(C tanh(u))
  (C = 10) "helps with exploration", the origin Gemma 2 cites. arXiv 2408.00118 (Gemma 2, 2024, §2):
  soft_cap·tanh(logits/soft_cap), 50 in attention, 30 on the final layer. arXiv 2503.19786 (Gemma 3, §2) replaces
  Gemma 2's soft-capping by QK-norm (both kept here). arXiv 2405.09818 (Chameleon, §2.3): QK-norm "does not solve the
  problem of logit shift in the final softmax" (they use z-loss). arXiv 2309.14322 (§3.1.2): output-logit divergence
  occurs without weight decay; z-loss or weight decay mitigates it.
- **Upstream.** modded `e59ae87`, record #9 (c = 30, README: "following" Gemma 2); `45544f5`, record #18 (30 → 15);
  asymmetric 23σ((z+5)/7.5) record #54 (LOG 2026-01-16: slightly worse). nanochat `3a5e0bc` (15), LOG 2026-03-02
  (d24: 5 terrible, 5–30 ≈ equal, 20 best by ~1e-3), `6ed7d1d` (back to 15).
- **Here.** ADOPTED; `FINAL_LOGIT_SOFTCAPPING` = 15, `model/ops/cross_entropy.py` `lm_loss`/`lm_losses`,
  `decode.cu`; `docs/architecture.md` §2; designed ablation `docs/experiments.md` §3. The fused implementation: §4,
  Fused linear cross entropy.
- **Cite as.** arXiv 2408.00118 §2 (the cap) and arXiv 1611.09940 App. A.2 (its origin); arXiv 2405.09818 §2.3 and
  arXiv 2309.14322 §3.1.2 (why output logits need their own control). c = 15 has no paper (record #18's tuning).

### Uniform fan-in initialization, std C^−½
- **Idea.** Draw weights with mean 0 and standard deviation m^−½ (m = fan-in) so pre-activations keep unit variance;
  uniform on ±√(3/m) has that std.
- **Literature.** LeCun, Bottou, Orr & Müller, "Efficient BackProp" (1998, Neural Networks: Tricks of the Trade,
  §4.6 Eq. 15–16): weights "randomly drawn from a distribution (e.g. uniform) with mean zero and standard deviation
  σ = m^−½". Glorot & Bengio (AISTATS 2010, Eq. 1, Eq. 16): the common U[−1/√n, 1/√n] heuristic and the normalized
  (fan-in + fan-out) initialization.
- **Upstream.** nanochat `48abd7d` (uniform, s = √3·C^−½, "Uniform to avoid outliers"); modded record #19
  (`602014e`, std 0.5·C^−½). Negative: LOG 2026-03-24 orthogonal init "Did not help" (orthogonal init: arXiv
  1312.6120, Saxe et al., 2013, Abstract: depth-independent learning times in deep linear networks).
- **Here.** ADAPTED; `CausalLM.init_weights` (`bound = (3 / C) ** 0.5`; token and value tables too);
  `docs/architecture.md` §1.
- **Cite as.** LeCun et al. 1998 §4.6. "Uniform rather than normal to avoid outliers" has no paper (upstream's design
  note).

### Zero-initialized residual-branch outputs (o_proj, down_proj)
- **Idea.** Each block starts as the identity, so signal and gradients propagate through the skip path and depth does
  not amplify activations at initialization.
- **Literature.** arXiv 1901.09321 (Zhang et al., 2019, Fixup, §3 box): "Initialize the classification layer and the
  last layer of each residual branch to 0"; applied to Transformer machine translation (§4.3). arXiv 2003.04887
  (ReZero, §1 Eq. 6): x + α·F(x) with α = 0 at init; arXiv 2002.10444 (De & Smith, 2020, SkipInit, §2.2): a scalar
  on each residual branch initialized to 0; arXiv 1706.02677 (Goyal et al., 2017, §5.1): γ = 0 in each residual
  block's last BN "eases optimization".
- **Upstream.** modded `5450eda`, record #5 ("zero init suggested by @Grad62304977"); nanochat `3a5e0bc`.
- **Here.** ADOPTED; `init_weights` zero `o_proj`, `down_proj` (also the VE gates and `x0_lambdas`: identity start);
  `docs/architecture.md` §1.
- **Cite as.** arXiv 1901.09321 §3 (zero last layer of each residual branch); arXiv 2003.04887 and arXiv 2002.10444
  for the identity-at-init principle.

### Zero-initialized output head
- **Idea.** Logits are 0 at init (uniform predictions); the head learns from the first step without random-feature
  noise in the output.
- **Literature.** arXiv 1901.09321 (Fixup §3 box): zero "classification layer". arXiv 2203.03466 (Yang et al., 2022,
  μTransfer, App. D.2): output layers initialized at 0 make small and large widths' optimal hyperparameters "match
  more closely"; "not … detrimental to performance".
- **Upstream.** modded `fdc6e0b`, record #8 ("init head to zero"); nanochat `3a5e0bc` (zero; later N(0, 0.001) in
  `48abd7d`, not taken). Record #20's README: Adam's ε had to drop to 1e-10 because the zero head's early gradients
  are tiny.
- **Here.** ADOPTED; `init_weights` zero `lm_head.weight`; `docs/architecture.md` §1.
- **Cite as.** arXiv 1901.09321 §3 and arXiv 2203.03466 App. D.2.

### Depth-scaled norm outputs (1/√l)
- **Idea.** Scale each layer's LayerNorm output by 1/√l to curb Pre-LN variance growth with depth.
- **Literature.** arXiv 2502.05795 (Sun et al., 2025, Abstract, §4): LayerNorm Scaling improves 130M–7B models.
- **Upstream.** LOG 2026-03-24 (`4e1694c`): 1/√(l+1) "Did not help".
- **Here.** NOT TAKEN; not discussed.
- **Cite as.** arXiv 2502.05795 if mentioned; the nanochat null result is not citable.

### Shape rule: width = depth × aspect, rounded up to whole 128-wide heads
- **Idea.** Fix the width-to-depth ratio so one integer (layers) sets the model; sub-billion models do better deep and
  thin.
- **Literature.** arXiv 2402.14905 (MobileLLM, §2.2.2, Fig. 4; Table 9): at ~125M and ~350M "deeper and thinner
  models outperform their shallower and wider counterparts"; MobileLLM-350M is 32 layers × 960 (ratio 30). arXiv
  2001.08361 (Kaplan et al., §3.1, Fig. 5): at fixed N the loss depends only weakly on shape (aspect ratio over a
  factor of 40 within a few percent), the counter-position for larger models.
- **Upstream.** nanochat `3a5e0bc` (depth × 64), `ccf4b7f`, `cf5c9e5`, `f41dd3c`; LOG 2026-01-17 (aspect 128 worse
  than 64 at 1e18 FLOPs: "slightly thinner and longer").
- **Here.** ADAPTED; `model/modeling.py` `ASPECT_RATIO` = 960 // 32 = 30, `meta_model`; `docs/architecture.md` §1, §2
  "Shape".
- **Cite as.** arXiv 2402.14905 §2.2.2 and Table 9 (ratio and deep-thin evidence); arXiv 2001.08361 §3.1 for the
  weak shape dependence. The rounding rule is nanochat's.

## 2. Optimizer, schedule and scaling

### Muon: orthogonalized momentum for hidden matrices
- **Idea.** Take the SGD-momentum update of each 2-D hidden weight matrix and replace it by its nearest
  semi-orthogonal matrix UVᵀ (the polar factor). Orthogonalizing the update is steepest descent under the spectral
  norm, the norm in which a linear layer's loss admits a quadratic upper bound; empirically, the raw momentum updates
  of transformer matrices are nearly low-rank, and orthogonalization raises the scale of their rare directions.
- **Literature.**
  - Jordan et al., 2024, "Muon: An optimizer for hidden layers in neural networks", blog post (non-archival
    original; §Definition, §"Why is it good to orthogonalize the update?"): defines Muon as NS-orthogonalized
    SGD-momentum; reports 1.35× faster NanoGPT speedrunning and a 1.5B model at GPT-2 XL HellaSwag level in 10 vs
    13.3 8×H100-hours for AdamW (§Results).
  - arXiv 2409.20325 (Bernstein & Newhouse, 2024, Story II, Props. 4–6): the closest semi-orthogonal matrix to G in
    Frobenius norm is UVᵀ (Prop. 4); Shampoo without accumulation is steepest descent under the max spectral norm
    over layers (Prop. 5); the square loss of a linear predictor is bounded by a quadratic in the spectral norm of the
    weight perturbation (Prop. 6). This is the theory the blog cites as Muon's motivation.
  - arXiv 2502.16982 (Liu et al., 2025, Fig. 1, §2.2, §3.2): with weight decay and per-shape update scaling, Muon is
    ~2× more compute-efficient than AdamW under compute-optimal training ("approximately 52% of the training FLOPs"),
    scaled to a 15.29B-total / 2.24B-active MoE trained on 5.7T tokens (§3.3).
  - arXiv 2509.02046 (Wen et al., 2025, §1 Fig. 1): with every optimizer tuned, matrix-preconditioned optimizers
    (Muon, Soap, Kron) beat scalar ones, but speedups over AdamW shrink from 1.3–1.4× below 1B to 1.1× at 1.2B.
  - arXiv 2512.05620 (Qiu et al., 2025, Abstract): with μP learning-rate scaling and independent weight decay
    ∝ 1/width, Muon, SOAP and Shampoo keep ~1.4× over AdamW from 190M to 1.4B; the speedup vanishes under incorrect
    scaling.
- **Upstream.** modded record #3 (`c26c2a1` 2024-10-04, "new optimizer", no log) and record #4 (`b356a1f`, class
  `Muon`); nanochat `3a5e0bc` `nanochat/muon.py` ("Muon optimizer from Keller et al."), `92d63d4` `nanochat/optim.py`
  (cites the blog). Both reference the blog post.
- **Here.** ADAPTED (lineage through `torch.optim.Muon`, whose `_muon.py` credits KellerJordan/Muon and the blog, not
  through nanochat's code); `train/optim.py` `Muon(torch.optim.Muon)`, `optimizers`; `docs/training.md` §3. Differs:
  NorMuon normalization, Moonlight step size, one lr for all groups, τ-law weight decay, constant momentum.
- **Cite as.** Jordan et al. 2024 (blog, non-archival) for the algorithm; arXiv 2409.20325 for the
  steepest-descent-under-the-spectral-norm rationale; arXiv 2502.16982 for its use at scale with weight decay and RMS
  matching. For the size of the advantage at small scale, arXiv 2509.02046 (shrinking speedup) and arXiv 2512.05620
  (preserved under correct transfer) together.

### Newton–Schulz iteration for the polar factor
- **Idea.** Compute UVᵀ without an SVD by iterating an odd matrix polynomial X ← p(XXᵀ)X from X₀ = G/‖G‖_F: each step
  applies the scalar polynomial to every singular value, so if p's iterates map (0, 1] to ≈1 the result approaches
  the polar factor using only matrix products, which run in bf16 on GPUs.
- **Literature.**
  - arXiv 2409.20325 (Bernstein & Newhouse, 2024, App. A item 4, Eq. 36): the cubic Newton–Schulz iteration
    X ← 1.5X − 0.5XXᵀX for UVᵀ, adapted from Higham (2008) Eq. 5.22, converging for singular values in (0, √3); the
    authors note the fixed-coefficient iteration is classical (Kovarik, 1970; Björck & Bowie, 1971).
  - arXiv 2505.16932 (Amsel et al., 2025, §1.2): Newton–Schulz [Higham 2008, Ch. 8] converges super-exponentially
    near the polar factor but slowly at first; the degree-5 form (15x − 10x³ + 3x⁵)/8.
  - Björck & Bowie, 1971, SIAM J. Numer. Anal. 8:358–364; Kovarik, 1970, SIAM J. Numer. Anal.; Higham, 2008,
    *Functions of Matrices*, SIAM (non-arXiv; content [unverified], cited via the two papers above and the blog).
  - Jordan et al., 2024 blog (§"Eliminating alternatives to NS iteration"): SVD is too slow; coupled Newton
    iterations need fp32; NS runs stably in bf16.
- **Upstream.** modded `c26c2a1` (record #3) `zeroth_power_via_newtonschulz5`.
- **Here.** ADOPTED (torch's `_zeropower_via_newtonschulz` semantics, re-implemented batched); `train/optim.py`
  `Muon.step` (`ns_steps`, Frobenius normalization clamped at `eps`, tall matrices transposed); `docs/training.md` §3.
  The batching: §4, Newton–Schulz batched over same-shape matrices.
- **Cite as.** Björck & Bowie 1971 and Kovarik 1970 as the classical iteration [unverified: bibliographic sources],
  with arXiv 2409.20325 App. A for its use as an orthogonalizer of gradient updates.

### Tuned quintic coefficients (3.4445, −4.7750, 2.0315), 5 bf16 steps
- **Idea.** Choose the quintic's coefficients to maximize the slope at 0 (fast growth of small singular values)
  subject to the iterates staying in [0.7, 1.3]; an update that is only approximately orthogonal (error ≈ 0.3) trains
  as well, so 5 steps suffice.
- **Literature.**
  - Jordan et al., 2024 blog (non-archival original; §"Tuning the coefficients"): the objective, the coefficients
    (found by "an ad-hoc gradient based approach") and "it suffices to run the NS iteration for only 5 steps".
  - arXiv 2505.16932 (Amsel et al., 2025, §1.2): Jordan's polynomial does not converge to polar(M) but plateaus at an
    error ≈ 0.3, reached rapidly; it beats plain Newton–Schulz at few iterations.
  - arXiv 2502.16982 (Liu et al., 2025, §2.2 "Other Hyper-parameters"): 10 NS steps orthogonalize more accurately
    than 5 but do not train better; 5 kept.
- **Upstream.** modded `c26c2a1` (record #3); nanochat `3a5e0bc` `muon.py:21` (replaced by Polar Express in
  `2c4473d`).
- **Here.** ADOPTED as PyTorch defaults (`DEFAULT_A/B/C`, `DEFAULT_NS_STEPS` = 5, `EPS` = 1e-7); `train/optim.py`
  `Muon.step` (`group["ns_coefficients"]`, `.bfloat16()`); `docs/training.md` §3 ("PyTorch's coefficients").
- **Cite as.** Jordan et al. 2024 (blog, non-archival) for the coefficients; arXiv 2505.16932 §1.2 for their
  characterization (fast, non-convergent, error ≈ 0.3); arXiv 2502.16982 §2.2 for 5 steps sufficing at scale.

### Gram-form Newton–Schulz step (A = XXᵀ, B = bA + cA², X ← aX + BX)
- **Idea.** Factor the quintic step through the Gram matrix so each step costs 4nm² + 2m³ instead of 6nm² FLOPs for
  an n×m matrix (m ≤ n): an algebraic rearrangement, numerically equivalent.
- **Literature.** arXiv 2608.11612 (Amsel et al., 2026, §2.2 Alg. 1) gives this form as "Standard Newton-Schulz"
  (A ← XXᵀ, B ← b_tA + c_tA², X ← a_tX + BX) with its cost T(4mn² + 2n³) FLOPs for n ≤ m, without attribution;
  arXiv 2505.16932 Alg. 1 uses the same form. The Jordan et al. 2024 blog (§"Muon Contributors") credits the variant
  to Bernstein, You and Cesista and states the FLOP count.
- **Upstream.** modded `3b3d650` 2024-11-10 (first in record #11's logs, "adapted from suggestion by @leloykun,
  @YouJiacheng, and @jsbz").
- **Here.** ADOPTED (via torch's `addmm` form; batched `baddbmm` here); `train/optim.py` `Muon.step`.
- **Cite as.** arXiv 2608.11612 §2.2 Alg. 1 for the form and its FLOP count; it is an implementation identity with no
  originating paper (the blog's contributors note, non-archival, for credit).

### Nesterov momentum 0.95 in EMA ("lerp") form
- **Idea.** Nesterov's look-ahead applied to the momentum buffer before orthogonalization; the buffer is kept as an
  EMA, buf ← β·buf + (1 − β)·g, which only rescales classical momentum by (1 − β), a factor the orthogonalization
  removes (Ortho(cG) = Ortho(G)).
- **Literature.**
  - Sutskever et al., 2013, ICML, PMLR 28 (non-arXiv; §2): Nesterov's accelerated gradient as a momentum method that
    evaluates the gradient at the look-ahead point; the effective-momentum analysis of NAG vs classical momentum.
    Nesterov, 1983 (original; non-arXiv, [unverified]).
  - Jordan et al., 2024 blog (§"Empirical considerations"): Nesterov-style momentum "works a bit better than normal
    SGD-momentum in every case we have tested"; §"Proving that NS iteration orthogonalizes": Ortho(cG) = Ortho(G).
  - arXiv 2502.16982 (Liu et al., 2025, §2.2 "Other Hyper-parameters"): no consistent gain from tuning momentum;
    0.95 used, as in Jordan et al.
- **Upstream.** modded `c26c2a1`/`b356a1f` (records #3, #4: `momentum=0.95, nesterov=True`); EMA form `8011fe7`
  (record #15); nanochat `3a5e0bc` `muon.py:80-81`.
- **Here.** ADOPTED via torch defaults; `train/optim.py` `Muon.step` (`torch._foreach_lerp_`, `torch._foreach_lerp`;
  `momentum` = 0.95); `docs/training.md` §3.
- **Cite as.** Sutskever et al. 2013 (and Nesterov 1983) for Nesterov momentum; arXiv 2502.16982 §2.2 for 0.95 being
  insensitive in Muon. The EMA form needs no citation (scale-invariance of the orthogonalization).

### Muon/AdamW parameter split
- **Idea.** Muon only for 2-D hidden matrices; embeddings, the output head and all 1-D parameters (gains, scalars) on
  AdamW, because their natural norms differ (embedding: ℓ1→RMS operator norm; hidden linear: RMS→RMS, a rescaled
  spectral norm), and AdamW empirically works better for input and output layers.
- **Literature.**
  - Jordan et al., 2024 blog (§Definition; §"Empirical considerations"; non-archival original): "AdamW should be used
    for the embedding and final classifier head layers"; scalars and vectors on AdamW.
  - arXiv 2409.20325 (Bernstein & Newhouse, 2024, Epilogue after Prop. 8): linear layers get the RMS→RMS operator
    norm (a rescaled spectral norm), embedding layers the ℓ1→RMS norm, so they call for different steepest-descent
    updates.
  - arXiv 2502.16982 (Liu et al., 2025, §2.2 "Matching update RMS of AdamW"): AdamW handles "RMSNorm, LM head, and
    embedding parameters"; App. D: weight decay on RMSNorm gains "crucial" for their stability (not done here: 1-D
    parameters have no decay).
  - arXiv 2505.16932 (Amsel et al., 2025, §5.2): same split in their GPT-2 experiments, following modded-nanogpt.
- **Upstream.** modded `c26c2a1` (record #3), `5fe33da` (untied separate Adams, record #8), `e59ae87` (ndim split,
  record #9); nanochat `3a5e0bc`, `e3f58b8`/`0b58d70`/`9a88194` (value-embedding tables on AdamW), `92d63d4` `gpt.py`
  `setup_optimizer`.
- **Here.** ADAPTED; `train/optim.py` `optimizers` (tables `embed_tokens`, `lm_head`, value-embedding tables and
  every 1-D parameter on fused AdamW; every other 2-D parameter, including `ve_gate`, on Muon); one lr and one λ (0 on
  1-D); `docs/training.md` §3 table.
- **Cite as.** Jordan et al. 2024 (blog) for the split; arXiv 2409.20325 for the norm-based reason; arXiv 2502.16982
  §2.2 for the same split at scale. The weight-decay exemption of 1-D parameters is the nanoGPT convention (software)
  and is contrary to arXiv 2502.16982 App. D's advice for RMSNorm gains.

### Muon step size per shape: RMS matching vs √max(1, rows/cols)
- **Idea.** An orthogonal m×n update has RMS 1/√max(m, n), so its effective step depends on shape. Two fixes: scale
  by √max(1, fan_out/fan_in), which makes the update's spectral norm ∝ √(fan_out/fan_in) (the spectral condition for
  feature learning); or scale by 0.2·√max(m, n), which gives every matrix the update RMS of AdamW (≈0.2) so all groups
  share one lr and λ.
- **Literature.**
  - arXiv 2502.16982 (Liu et al., 2025, §2.2 Lemma 1, Eq. 4 "Matching update RMS of AdamW", §3.1, App. A Table 8):
    update RMS √(1/max(A, B)); the 0.2·√max(A, B) adjustment lets Muon "directly reuse the learning rate and weight
    decay tuned for AdamW".
  - arXiv 2310.17813 (Yang, Simon & Bernstein, 2023, Condition 1): ‖ΔW_ℓ‖* = Θ(√(n_ℓ/n_ℓ₋₁)) for feature learning.
    That Keller Jordan's √max(1, rows/cols) implements this for an orthogonal update is [inference].
- **Upstream.** Shape scale: modded `b356a1f` (√max(rows, cols), record #4), `4706781` (√max(1, rows/cols)); nanochat
  `3a5e0bc` `muon.py:83`; LOG 2026-02-03 (PR #492, flipped multiplier "slightly worse", not adopted).
- **Here.** Moonlight's rule ADOPTED; upstream's shape scale NOT TAKEN; `train/optim.py` `optimizers`
  (`adjust_lr_fn="match_rms_adamw"`), `Muon.step` (`_adjust_lr`); `docs/training.md` §3.
- **Cite as.** arXiv 2502.16982 §2.2 Eq. 4 for the step size used here. For upstream's rule: Jordan et al. 2024 /
  modded-nanogpt as software, with arXiv 2310.17813 Condition 1 as its theoretical counterpart [inference].

### NorMuon: neuron-wise second-moment normalization after orthogonalization
- **Idea.** Orthogonalized updates have uniform singular values but very uneven per-neuron (row) norms; keep an EMA of
  each row's mean square of O, divide the row by its square root, then restore O's Frobenius norm, combining Muon's
  conditioning with Adam-mini-style per-neuron adaptive rates.
- **Literature.** arXiv 2510.05491 (Li et al., 2025, §1 Fig. 1; §3.1 Alg. 1 lines 7–9; §4.1.1; Table 1; Abstract):
  per-neuron norms of Muon updates vary widely; NorMuon reaches AdamW's final loss in 21.74% / 13.91% fewer steps at
  1.1B / 5.4B (Muon: 10.43% / 6.08%), "11.31% improvement over Muon" at 1.1B; β₂ = 0.95; +2.9% step time over AdamW
  (§4.1.4); normalizing also m ≤ n matrices helps (§4.1.3).
- **Upstream.** modded record #41 (`9f0d8ca`, PR #144; on top of Polar Express; 20 runs; README: −15 steps), axis fix
  record #48 (`b72cabd`, PR #168), β₂ 0.95 → 0.9 record #78 (`5845e94`, PR #246); nanochat `2c4473d` (LOG
  2026-01-10: "Led to a very small improvement, kept and enabled by default"); both reference arXiv 2510.05491.
- **Here.** EVIDENCE (adopted on the paper and upstream results; algorithm from arXiv 2510.05491 and
  zichongli5/NorMuon, not from upstream code); `train/optim.py` `Muon.step` lines 59–63, `NORMUON_BETA2` = 0.95,
  `NORMUON_EPS` = 1e-10 (`646a986`); `docs/training.md` §3 "NorMuon's adoption". Differs from PR #144: always per
  row, √v + eps, torch Newton–Schulz rather than Polar Express, β₂ 0.95 (upstream now 0.9).
- **Cite as.** arXiv 2510.05491 (Alg. 1 for the method; Table 1 and Abstract for the gains). The upstream records and
  the LOG are not citable as evidence; this repository's one-seed d16 screen does not resolve the effect
  (`docs/training.md` §3), so the adoption rests on the paper.

### Polar Express coefficients
- **Idea.** Replace the fixed quintic with a per-iteration sequence of minimax-optimal odd quintics (each optimal for
  the current singular-value interval), so the iteration is fast early and converges asymptotically, with a safety
  factor for bf16.
- **Literature.** arXiv 2505.16932 (Amsel et al., 2025, §1.3, Thm. 4.1, Alg. 1; §5.2 Figs. 1, 6, 7): worst-case
  optimal polynomial compositions; inside Muon on GPT-2 Small/Large (1B FineWeb tokens) lower validation loss than
  Jordan's and You's coefficients at every learning rate.
- **Upstream.** modded record #38 (`9350f4d`, PR #134); nanochat `2c4473d`, LOG 2026-01-10 ("No dramatic/noticeable
  difference in training, but keeping the new Polar Express as default"); both cite the paper.
- **Here.** NOT TAKEN; `docs/experiments.md` §4 "Polar Express" (d16 one-seed screen: within noise, −0.35% tok/s;
  dropped).
- **Cite as.** arXiv 2505.16932 for the method and its GPT-2 gains. Upstream's null result (LOG) agrees with this
  repository's but is not citable evidence; the repository's screen is the reported measurement.

### Cautious weight decay
- **Idea.** Apply decoupled weight decay only to coordinates where the update and the parameter have the same sign
  (I(u ⊙ x ≥ 0)); decay then never opposes the update, the method optimizes the unregularized loss, and on the
  stationary manifold it slides toward locally Pareto-minimal parameter magnitudes.
- **Literature.**
  - arXiv 2510.12402 (Chen et al., 2025, Alg. 1; §3.1–3.2; §4 Fig. 1, Table 2, Fig. 5): Lyapunov analysis (unbiased,
    sliding mode); lower final loss for AdamW, Lion and Muon at 338M–2B on C4 at 20 TPP and OLMo-1B at 100 TPP, with
    the optimal λ essentially unchanged (Fig. 1).
  - arXiv 2411.16085 (Liang et al., 2024, Alg. 1): "cautious optimizers" mask the *update* where it disagrees in sign
    with the gradient (u·g > 0), a different mask; it is the paper nanochat's LOG cites.
- **Upstream.** modded record #43 (`29aefcf`, PR #154; README: 10 runs, p = 0.0018; cites arXiv 2510.12402), Adam
  version record #50 (`49465cc`, PR #172); nanochat `2c4473d`, LOG 2026-01-10 ("Solid improvements, especially the
  cautious version was better than standard wd"; cites arXiv 2411.16085).
- **Here.** NOT TAKEN; `docs/experiments.md` §4 "Cautious weight decay" (d16 one-seed screen: worse at the end).
- **Cite as.** arXiv 2510.12402 for the weight-decay mask (the mechanism both upstreams implement, `g·p ≥ 0` with g
  the update). arXiv 2411.16085 only as the related update-masking method; it is the wrong citation for the decay
  mask. Upstream's positive results are contrary but uncitable; the repository's negative screen is one seed.

### Decoupled weight decay; AdamW and its hyperparameters; weight decay in Muon
- **Idea.** Shrink weights by η_t·λ·θ separately from the adaptive gradient step, so decay is not divided by Adam's
  second moment and λ decouples from the learning rate; Muon needs the same decay at scale to keep weight and output
  RMS bounded.
- **Literature.**
  - arXiv 1412.6980 (Kingma & Ba, 2014, §2 Alg. 1; §2.1): Adam; its step is invariant to rescaling the gradient.
  - arXiv 1711.05101 (Loshchilov & Hutter, 2017, §2 Alg. 2 line 12, Prop. 2): decoupled decay (AdamW), scaled by the
    schedule multiplier η_t; L2 and decoupled decay differ for adaptive methods.
  - arXiv 2502.16982 (Liu et al., 2025, §2.2 "Weight Decay", Fig. 2): without decay Muon's weights and outputs grow
    past bf16's precise range and its early lead fades at 800M / 100B tokens; AdamW-style decay fixes it.
  - arXiv 2401.02954 (DeepSeek-AI, 2024, §2.3): AdamW β = (0.9, 0.95), weight decay 0.1 (the betas used here).
- **Upstream.** modded track 2 record #3 (`d39af59`, Muon decay `p *= 1 − lr·wd`); Adam β₁ 0.8 (`759b8fc`) and eps
  1e-10 (record #20 `ebdeb37`); nanochat `92d63d4` `nanochat/optim.py` (cites arXiv 1711.05101), per-group betas.
- **Here.** ADOPTED (decoupled, lr-coupled decay in both optimizers); upstream's β₁ 0.8 / eps 1e-10 NOT TAKEN;
  `train/optim.py` `BETAS`, `optimizers`, `Muon.step` (`1 − lr·weight_decay`); `docs/training.md` §3.
- **Cite as.** arXiv 1711.05101 (decoupled decay), arXiv 1412.6980 (Adam), arXiv 2401.02954 §2.3 (β values), arXiv
  2502.16982 §2.2 (decay in Muon).

### Weight decay tied to the training horizon (the EMA timescale)
- **Idea.** With decoupled decay the weights are an EMA of recent updates with timescale τ_iter = 1/(ηλ) steps; the
  right λ keeps that timescale a fixed fraction of training (or a law in tokens per parameter), so λ must fall as the
  run lengthens.
- **Literature.**
  - arXiv 1711.05101 (Loshchilov & Hutter, 2017, §2 Abstract point; App. B.1 Eq. 15): optimal decay depends on the
    number of updates; normalized decay λ = λ_norm·√(b/(BT)).
  - arXiv 2405.13698 (Wang & Aitchison, 2024, §3.1 Eq. 5; §4, Figs. 1–2): AdamW as an EMA with 1/τ_iter = η_t·λ; the
    optimal τ_epoch = τ_iter/M is roughly constant across model and dataset size (multi-epoch settings).
  - arXiv 2505.13738 (Bergsma et al., 2025, Finding 3, Fig. 1 left; App. E.3.1): in single-epoch LLM pre-training
    the optimal normalized timescale τ = 1/(ηλS) is not constant but a power law in tokens per parameter (≈1.0 at 1 TPP
    to 0.01 at 1000 TPP; R² = 0.975; exponent percentiles (−0.529, −0.507)); App. E.3.1 quotes c_τ = 1.084,
    m_τ = −0.527 (the same appendix also states m_τ = −0.520 [extracted text has both]); Finding 1: τ_opt is stable
    across batch size.
- **Upstream.** nanochat `2c4473d` (λ ∝ 1/depth²; LOG 2026-01-10 "Weight Decay Scaling Experiments"), `aeff095`
  (λ = λ_ref·√(B/B_ref)·(D_ref/D), "We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698"), Muon
  only.
- **Here.** PARALLEL (same EMA idea, a different law); `train/optim.py` `tau_weight_decay`, `TAU_C`, `TAU_E`
  (`3bfa7c5`), applied to AdamW tables and Muon alike; `docs/training.md` §3; Muon-specific timescale runs
  `docs/experiments.md` §4.
- **Cite as.** arXiv 2405.13698 for the EMA view; arXiv 2505.13738 Finding 3 / App. E.3.1 for the TPP law used here.
  Its fit is on AdamW; applying it to Muon is an assumption (`docs/training.md` §3 gives the supporting literature).

### Weight-decay schedule (λ decayed to zero, linearly or by cosine)
- **Idea.** Upstream multiplies Muon's λ by its own schedule (linear, later cosine, to 0) on top of the lr-coupled
  decay η_t·λ, so the effective per-step decay falls faster than the learning rate late in training.
- **Literature.** No paper found that schedules λ itself. The closest statements treat η_t·λ as a time-varying EMA
  coefficient: arXiv 2405.13698 §3.1 Eq. 5 (1/τ_iter;t = η_t λ) and arXiv 2505.13738 App. E.4 (the EMA view under
  learning-rate schedules, following arXiv 2502.15938). arXiv 1711.05101 Alg. 2 scales λ by the schedule multiplier
  η_t, which the lr-coupled decay here already does.
- **Upstream.** nanochat `2c4473d` (linear), `6ed7d1d` (cosine); LOG 2026-01-10 ("modded-nanogpt has the same
  schedule").
- **Here.** NOT TAKEN (undiscussed): λ is constant; the effective decay lr·λ follows the lr schedule
  (`train/optim.py` `Muon.step`, `torch.optim.AdamW`).
- **Cite as.** No paper: an upstream design choice. If the paper discusses it, cite arXiv 2405.13698 §3.1 for why
  η_t·λ is the quantity that matters, and state that λ is held constant here.

### Muon momentum warmup and schedule (0.85 → 0.95; later → 0.97, down to 0.90 in the warmdown)
- **Idea.** Start Muon's momentum low and raise it over the first few hundred steps, when gradients change fastest;
  later variants also lower it during the learning-rate decay.
- **Literature.** No paper for Muon's schedule. Prior art for increasing momentum: Sutskever et al., 2013 (ICML,
  non-arXiv, §3 Eq. 5: μ_t = min(1 − 2^(−1−log₂(⌊t/250⌋+1)), μ_max), motivated by Nesterov 1983). arXiv 2502.16982
  §2.2 found no consistent gain from tuning momentum (constant 0.95).
- **Upstream.** modded record #9 (`e59ae87`; README: 3.2972 → 3.2844 between consecutive incremental runs, one run
  each, the step also adding other changes); nanochat `3a5e0bc` (300 steps), `6ed7d1d`, `a825e63`.
- **Here.** NOT TAKEN (undiscussed; removed with nanochat's MuonAdamW in `a8a6971`): `momentum` constant 0.95.
- **Cite as.** No paper for the Muon schedule (an upstream heuristic; a single unreplicated record step). Sutskever et
  al. 2013 §3 is the citable precedent for momentum schedules in general.

### Per-group AdamW learning rates, betas and width scaling
- **Idea.** Separate lrs, betas, eps and decay for the token table, the head, the value-embedding tables and the
  scalars, all × (d_model/768)^−½, tuned empirically at d12.
- **Literature.** No paper for this recipe. Principled per-layer rules exist: arXiv 2203.03466 (Yang et al., 2022,
  Table 3): under μP, Adam's lr is Θ(1) for input weights and ∝ 1/fan_in for output and hidden weights (not
  1/√d_model for all). The basis for one shared lr here is arXiv 2502.16982 §2.2 ("learning rate η, weight decay λ"
  shared among matrix and non-matrix parameters).
- **Upstream.** nanochat `3a5e0bc` (`dmodel_lr_scale`, betas (0.8, 0.95), eps 1e-10), `6ed7d1d`, `85b3e95` (β₁ of x0
  0.8 → 0.96 from "320 experiments"), `a825e63`; `92d63d4` `gpt.py:432-445` ("tuned for 768 dim model").
- **Here.** NOT TAKEN (designed as `adamw_group_lr`, not run); `docs/experiments.md` §4; one lr, `docs/training.md`
  §3.
- **Cite as.** No paper for upstream's recipe (empirical tuning). Cite arXiv 2502.16982 §2.2 for the shared lr used
  here; arXiv 2203.03466 Table 3 if the paper discusses principled per-layer rates.

### Learning rates of the residual and x0 scalars
- **Idea.** Give the per-layer residual and x0 mixing scalars their own lrs (0.5 and 0.005, × √(B/2¹⁹)), much larger
  / smaller than the shared lr.
- **Literature.** None found.
- **Upstream.** nanochat `aa530cd` (LOG 2026-01-11 "Per-Layer Residual Scalars", "Cherry-picked … from
  modded-nanogpt"), `7312ec9`; `92d63d4` `gpt.py:442-443`.
- **Here.** NOT TAKEN (tested: one d16 run, branch `scalar-learning-rates` `e6e3837`, not adopted);
  `docs/experiments.md` §4; `results/scalar_learning_rates.json`.
- **Cite as.** No paper: an upstream tuning choice; the repository's single run is the evidence reported.

### Learning rate ∝ √(batch size)
- **Idea.** For adaptive optimizers, keep the SDE of training fixed across batch sizes by scaling η with √κ when the
  batch grows by κ (and adjusting 1 − β and ε accordingly); the √ keeps the update noise variance constant.
- **Literature.**
  - arXiv 1404.5997 (Krizhevsky, 2014, §4.2): "Theory suggests" √k for batch ×k "to keep the variance in the
    gradient expectation constant", though linear k worked best for his SGD runs.
  - arXiv 2205.10287 (Malladi et al., 2022, §5 Def. 5.2, Thm. 5.3): the square-root scaling rule for Adam, proven as
    a weak approximation of the same SDE (the radical is lost in the extracted text; the section title and §1 state
    it).
  - arXiv 2505.13738 (Bergsma et al., 2025, Finding 2, Table 1): with AdamW at 610M / 20 TPP, adjust λ (λ_opt ∝ B,
    Finding 1), not η, as the batch changes (contrary evidence for √B scaling of η in this regime).
- **Upstream.** nanochat `ccf4b7f` (`batch_lr_scale = batch_ratio ** 0.5`), `f41dd3c`, `aeff095` ("AdamW: sqrt
  scaling is standard"; for Muon "not studied carefully, assumption!").
- **Here.** NOT TAKEN; lr comes from the DeepSeek LLM compute law × a measured multiplier (`train/schedule.py`
  `plan`; `docs/training.md` §2, §5).
- **Cite as.** arXiv 2205.10287 §5 (with arXiv 1404.5997 §4.2 as the origin of the √k heuristic) if the rule is
  mentioned; arXiv 2505.13738 Findings 1–2 for the alternative (scale λ with B).

### Batch size and peak learning rate from scaling laws
- **Idea.** The loss-optimal batch grows with training scale; fit B_opt (and η_opt) as power laws and read them off
  for the run. Upstream fits B in tokens D; here both B and η come from compute C = M·D.
- **Literature.**
  - arXiv 2401.02954 (DeepSeek-AI, 2024, §3.1 Eq. 1): η_opt = 0.3118·C^−0.1250, B_opt = 0.2920·C^0.3271, fit on
    1e17–2e19 FLOPs with a multi-step schedule, near-optimal = within 0.25% of the best loss.
  - arXiv 2505.13738 (Bergsma et al., 2025, Finding 5, Fig. 1 middle; App. F.5 Table 5): B_opt and B_crit are power
    laws in D, not C or L; tuned-λ fit B_opt = 0.0306·D^0.383 (R² = 0.984; exponent percentiles 0.367–0.391).
  - arXiv 2001.08361 (Kaplan et al., 2020, §5.1): the critical batch size B_crit(L), the time/compute trade-off.
- **Upstream.** nanochat `f41dd3c`, `aeff095` (B = 2¹⁹·(D/D_ref)^0.383 rounded to a power of 2; comment cites arXiv
  2505.13738); LOG 2026-02-05 "Auto Batch Size Scaling".
- **Here.** Upstream's rule NOT TAKEN; DeepSeek LLM's laws ADOPTED, B rounded to whole micro-batches;
  `train/schedule.py` `plan`; `docs/training.md` §2 (batch not ablated), §5 (multiplier measured at d24).
- **Cite as.** arXiv 2401.02954 §3.1 Eq. 1 for both laws as used here; arXiv 2505.13738 Finding 5 as the D-based
  alternative upstream uses.

### Batch-size schedule (ramp)
- **Idea.** Increase the batch during training instead of (or along with) decaying the learning rate: a larger batch
  reduces gradient noise as a smaller lr does, with fewer optimizer steps.
- **Literature.**
  - arXiv 1711.00489 (Smith et al., 2017, Abstract): increasing the batch gives the same learning curves as decaying
    the lr, for SGD, momentum, Nesterov and Adam, with fewer parameter updates.
  - arXiv 2502.16982 (Liu et al., 2025, §3.3 "Pretraining"): Moonlight doubles its batch from 2048 to 4096 sequences
    after 200B tokens.
- **Upstream.** modded record #46 (`25500ab`, PR #163); nanochat LOG 2026-01-17 ("Batch size schedule | Abandoned"),
  LOG 2026-02-05 ("batch size ramp … Not merged").
- **Here.** NOT TAKEN (undiscussed): fixed batch (`train/schedule.py` `plan`).
- **Cite as.** arXiv 1711.00489 if the paper mentions it; no upstream evidence is citable.

### Global gradient-norm clipping at 1.0 (and upstream's removal)
- **Idea.** Rescale the whole gradient to norm ≤ 1 when it exceeds it, bounding the step on rare exploding-gradient
  events without changing ordinary steps.
- **Literature.**
  - arXiv 1211.5063 (Pascanu et al., 2012, §3.2 Alg. 1): norm clipping against exploding gradients.
  - arXiv 2401.02954 (DeepSeek-AI, 2024, §2.3): "gradient clipping … set to 1.0" (the value used here).
  - arXiv 1412.6980 (Kingma & Ba, §2.1): Adam's step is invariant to the gradient's scale, so a clip only matters
    when the scale changes fast [inference from the invariance].
- **Upstream.** Removal: modded `9805cab` 2024-06-01 (between records #1 and #2); nanochat `061f83c` (LOG 2026-01-08
  "exp_grad_clip": no benefit at d12/d20, norm never > 1.0, ~2% MFU cost; "modded-nanogpt does not appear to clip").
  nanochat's clip before that: `3a5e0bc` (1.0).
- **Here.** ADOPTED from DeepSeek LLM (upstream's removal NOT TAKEN); `train/optim.py` `MAX_GRAD_NORM`, `update`
  (`e4c3587`, whose message reports a d8 A/B 0.87806 vs 0.88119 bpb favouring the clip); `docs/training.md` §3.
- **Cite as.** arXiv 1211.5063 for the mechanism; arXiv 2401.02954 §2.3 for the value. The no-clip evidence (LOG,
  `9805cab`) is an unreviewed upstream observation; the repository's d8 A/B is one run each and also not paper-grade.

### Linear warmup over 2/(1 − β₂) steps
- **Idea.** Ramp the lr linearly at the start because Adam's second-moment estimate is unreliable (high-variance) for
  roughly its averaging horizon 1/(1 − β₂); a linear ramp over 2/(1 − β₂) steps (40 at β₂ = 0.95) is an untuned rule
  of thumb.
- **Literature.**
  - arXiv 1706.02677 (Goyal et al., 2017, §2.2 "Gradual warmup"): linear ramp of the lr at the start of large-batch
    training.
  - arXiv 1908.03265 (Liu et al., 2019, Abstract): warmup as variance reduction of the adaptive lr early in training.
  - arXiv 1910.04209 (Ma & Yarats, 2019, §4.3 Eq. 14): "linear warmup over τ = 2·(1−β2)^−1 iterations", matching
    RAdam without tuning.
  - arXiv 2406.19146 (Porian et al., 2024, §3.3): over-long warmup biases small-scale scaling fits.
- **Upstream.** nanochat `3a5e0bc` ((s+1)/W form), `6ed7d1d` (default 40 steps, "All tuning was done on d12"; d24
  speedrun keeps it); no paper referenced.
- **Here.** ADOPTED (form and value; basis arXiv 1910.04209); `train/schedule.py` `WARMUP_STEPS`, `lr_lambda`;
  `docs/training.md` §4.
- **Cite as.** arXiv 1910.04209 §4.3 Eq. 14 (with arXiv 1706.02677 §2.2 for linear warmup itself). Upstream's 40 is an
  empirical d12 tuning, not citable; that it coincides with 2/(1 − 0.95) is the stated basis here.

### Trapezoid / warmup–stable–decay schedule and a nonzero final lr
- **Idea.** Hold the lr constant and anneal it in a final cooldown; the loss drops sharply during the decay, and the
  constant phase makes runs of different lengths share a prefix. Upstream also stops the decay at a floor.
- **Literature.**
  - arXiv 2405.18392 (Hägele et al., 2024, Abstract; §3.2 Figs. 4–5; §3.3): constant lr + cooldown scales like
    cosine; (1 − sqrt) cooldown shape and longer cooldowns help.
  - arXiv 2404.06395 (Hu et al., 2024, §4.2 Eq. 1; §4.4): the warmup–stable–decay (WSD) schedule and the loss drop
    during decay.
  - arXiv 2502.15938 (Bergsma et al., 2025, §4.2 Fig. 8): linear decay to zero is ~0.84% better than the optimal WSD
    (610M, 80 TPP).
- **Upstream.** modded `93fb178` 2024-06-05 ("constant then warmdown schedule from https://arxiv.org/abs/2405.18392"),
  `b356a1f`; floors 0.1 (record #19), 0.05 (record #26), 0.15 (record #72); nanochat `3a5e0bc` (warmdown 0.2,
  final 0), `ae0bf52`, `3c3a3d7`, `6ed7d1d` (warmdown 0.65, final 0.05).
- **Here.** NOT TAKEN (linear decay to zero instead); `train/schedule.py` `lr_lambda`; `docs/training.md` §4.
- **Cite as.** arXiv 2405.18392 (what modded-nanogpt cites) and arXiv 2404.06395 for WSD; arXiv 2502.15938 for why it
  is not used here. No paper fixes upstream's floor values (5–15% of peak); they are tuned constants.

### Linear decay to zero
- **Idea.** Decay linearly from the peak to zero over the whole run: under the EMA view of AdamW, a vanishing lr
  averages over more updates late in training, which matters more as tokens per parameter (gradient noise) grow.
- **Literature.** arXiv 2502.15938 (Bergsma et al., 2025, Abstract; §4.2 Figs. 6, 8; App. C.7): D2Z beats 10× decay
  and WSD; the gap to 10× grows with TPP (111M–1.7B); a 610M model at 80 TPP with D2Z beats 200 TPP with 10× decay.
- **Upstream.** nanochat `3a5e0bc` had a linear warmdown to 0 over the last 20% (`final_lr_frac` 0.0); later
  versions kept a 5% floor.
- **Here.** ADOPTED (from the paper, not upstream): decay from step 40 to 1/(steps − 40) of peak at the last step;
  `train/schedule.py` `lr_lambda`; `docs/training.md` §4.
- **Cite as.** arXiv 2502.15938 §4.2.

### The schedule as a multiplier of each group's initial lr
- **Idea.** Store each group's peak lr and multiply it by a schedule function of the step; with one shared function
  every group keeps its ratio to the others.
- **Literature.** None (implementation pattern; arXiv 1711.05101 Alg. 2's "SetScheduleMultiplier(t)" is the same
  abstraction).
- **Upstream.** nanochat `3a5e0bc` (`group["initial_lr"]` × multiplier).
- **Here.** ADAPTED; `train/loop.py` `LambdaLR(…, lr_lambda)` (`e4c3587`, bit-identical to the manual multiplier).
- **Cite as.** No paper needed: software.

### Hyperparameter transfer from a reference model (d12, "μP style")
- **Idea.** Tune at a small reference depth and carry lr, batch and decay to larger models by scaling rules, so that
  only the reference is swept.
- **Literature.**
  - arXiv 2203.03466 (Yang et al., 2022, Table 3): μP makes optimal hyperparameters width-independent; standard
    parameterization does not.
  - arXiv 2512.22382 (Mlodozeniec et al., 2025, Abstract; Table 1): transfer across width, depth, batch size and
    training duration (Complete(d)P).
  - arXiv 2512.05620 (Qiu et al., 2025, Abstract): with the right transfer, Muon's gain over AdamW persists to 1.4B.
- **Upstream.** nanochat `aeff095` ("transfered to higher depths (muP style)"), LOG 2026-02-05.
- **Here.** NOT TAKEN: standard parameterization; the lr multiplier is measured at d24 itself; `docs/training.md` §5.
- **Cite as.** arXiv 2203.03466 and arXiv 2512.22382 for why small-model tuning transfers only under a matching
  parameterization; upstream's d12 rule is not a μP parameterization [inference: it scales lr by √B only].

### Training horizon from tokens per parameter
- **Idea.** Set tokens D = r·N for a target ratio r and steps = D // B; at the compute-optimal frontier, parameters
  and tokens grow in equal proportion, ≈20 tokens per parameter.
- **Literature.**
  - arXiv 2203.15556 (Hoffmann et al., 2022, §3.4 Table 2; Table 3): equal-proportion scaling; 1B parameters ↔ 20.2B
    tokens.
  - arXiv 2505.13738 (Bergsma et al., 2025, Finding 3) and arXiv 2502.15938 (§4.2 Fig. 6): hyperparameters (decay
    timescale, schedule gap) depend on TPP far beyond 20, up to 1000+ TPP.
- **Upstream.** nanochat `3a5e0bc` (ratio 20), `ccf4b7f` (8), `413e91a` (4), `c8d93be` (10.5), `1cd94d7` (12);
  `base_train.py:58` comment "(Chinchilla=20)".
- **Here.** ADAPTED: same arithmetic, but r is data-bound (`train/stages.py` `max_tokens_per_param`, about 526 TPP,
  the most whole micro-batches with no stage overdraw), not compute-optimal; `train/schedule.py` `plan`;
  `docs/training.md` §2.
- **Cite as.** arXiv 2203.15556 Table 3 for the compute-optimal reference ratio; the ratio used is a data-supply
  choice (no paper); arXiv 2505.13738 / arXiv 2502.15938 for TPP-dependent hyperparameters in the over-trained
  regime.

### Compute-optimal tokens per parameter from upstream's scaling sweeps (~10.5 → 12)
- **Idea.** Re-fit the compute-optimal ratio for the specific architecture and parameter-count convention; it depends
  on what N counts.
- **Literature.**
  - arXiv 2203.15556 (Hoffmann et al., 2022, §3.4): ~20 with total parameters (embeddings included, App. F).
  - arXiv 2406.12907 (Pearce & Song, 2024, Abstract): Kaplan's N ∝ C^0.73 vs Chinchilla's C^0.50 is largely due to
    counting non-embedding parameters at small scale; recommends total parameters and compute.
  - arXiv 2406.19146 (Porian et al., 2024, §3.2–3.5): head FLOPs, warmup length and optimizer tuning (β₂ at small
    batch) explain the discrepancy.
- **Upstream.** nanochat `c8d93be`, `1cd94d7`; LOG 2026-01-27 (re-run, "~10.5"), 2026-01-17.
- **Here.** NOT TAKEN (deliberately over-trained, about 526 TPP).
- **Cite as.** arXiv 2203.15556 with arXiv 2406.12907 / arXiv 2406.19146 for why the optimal ratio depends on the
  count convention. Upstream's ratios are unreviewed fits on their own setup and are not citable.

### The scaling-parameter count: N = decoder layers + lm_head (embedding tables excluded)
- **Idea.** The tokens-per-parameter budget scales with what scales compute: the per-token matmul work of the blocks
  and the output head; lookup tables (token and value embeddings) cost no matmul FLOPs and are excluded.
- **Literature.**
  - arXiv 2001.08361 (Kaplan et al., 2020, §1.1, §2.1 Eq. 2.1, Table 1; §3.2 Fig. 6): N excludes "all vocabulary and
    positional embeddings" (and the de-embedding row is outside N's total); trends are cleaner in non-embedding N.
  - arXiv 2406.19146 (Porian et al., 2024, §3.2): not counting the head under-approximates FLOPs by ~10% (large) to
    ~90% (small models); counting it is "significantly more accurate".
  - arXiv 2401.02954 (DeepSeek-AI, 2024, §3.2 Eq. 2, Table 3, App. Fig. 6): non-embedding FLOPs/token M (attention
    included, vocabulary excluded) fits better than 6N₁ (non-embedding) or 6N₂ (with 6·n_vocab·d_model).
  - arXiv 2406.12907 (Pearce & Song, 2024, Abstract): much of the Kaplan (N ∝ C^0.73) vs Chinchilla (C^0.50) gap
    comes from "Kaplan counting non-embedding rather than total parameters, combined with their analysis being
    performed at small scale"; total parameters recommended for compute-optimal fits.
- **Upstream.** nanochat `ccf4b7f` (`num_scaling_params`, "same as Chinchilla paper", comment "per Kaplan et al."),
  `c8d93be` (transformer matrices + lm_head); LOG 2026-01-27 (Kaplan-style counts give the most stable exponents);
  `92d63d4` `gpt.py:396-397` cites arXiv 2203.15556 and arXiv 2001.08361.
- **Here.** ADOPTED; `model/modeling.py` `num_non_embedding_params` (decoder-layer parameters, including 1-D gains
  and `ve_gate`, plus `lm_head`); `docs/architecture.md` §1 "Counts", `docs/training.md` §2.
- **Cite as.** arXiv 2001.08361 §2.1 for non-embedding N, arXiv 2406.19146 §3.2 for counting the untied head; arXiv
  2406.12907 as the caution that the count choice shifts fitted exponents (total parameters the alternative).
  Counting `lm_head` but not the value tables is nanochat's convention (no paper).

### FLOPs per token: 6N + attention over attended keys (plan and model FLOPs)
- **Idea.** Training costs 6 FLOPs per matmul parameter per token (2 forward, 4 backward) plus attention's QKᵀ and PV,
  12·L·H·Q·T per token for context T; with sliding windows each layer's T is its attended keys. A plan derived from a
  fitted law needs the FLOP definition the law was fit with.
- **Literature.**
  - arXiv 2001.08361 (Kaplan et al., 2020, §2.1 Eq. 2.2, Table 1): C_forward ≈ 2N + 2·n_layer·n_ctx·d_attn, hence
    C ≈ 6N per token for training; N excludes embeddings.
  - arXiv 2204.02311 (Chowdhery et al., 2022, App. B): "an N-parameter decoder-only model requires 6N matmul FLOPs per
    token … dense self-attention add 6LH(2QT) FLOPs per token", i.e. 6N + 12LHQT; MFU is defined in its §4.1 (see §4,
    Model FLOPs utilization).
  - arXiv 2203.15556 (Hoffmann et al., 2022, App. F, Table A4): an exact count including embedding and softmax FLOPs
    differs little from 6ND.
  - arXiv 2401.02954 (DeepSeek-AI, 2024, §3.2 Eq. 2): non-embedding FLOPs/token M = 72·n_layer·d_model² +
    12·n_layer·d_model·l_seq, used in place of 6N for the scaling laws.
- **Upstream.** nanochat `3a5e0bc` `estimate_flops` (6N + 12·l·h·q·t; cites PaLM and Chinchilla in
  `gpt.py:319-339`), `fbc1484` (per-window attention sum; LOG 2026-01-11), `92d63d4` (`num_matmul_params`; `gpt.py`
  comments cite arXiv 2204.02311 and a Medium post by D. Bahdanau [non-archival]).
- **Here.** ADAPTED. Model FLOPs: `metrics.py` `model_flops_per_token` (N = `model/modeling.py` `num_matmul_params`,
  the decoder and `lm_head` matmul weights; keys min(window + 1, context), counting the query's own key) and
  `executed_model_flops` (per packed segment, the MFU numerator). The plan: DeepSeek's M, `model/modeling.py`
  `non_embedding_flops_per_token` (windowed layers counted as full; `ba01bb5`, `b065ad7`); upstream's per-window
  count is not used for the plan. `docs/architecture.md` §1, `docs/evaluation.md` §5, `docs/training.md` §2, §7.
- **Cite as.** arXiv 2204.02311 App. B (6N + 12LHQT) and arXiv 2001.08361 §2.1 (6N); arXiv 2401.02954 §3.2 Eq. 2 for
  M (the law's own definition). Replacing T by min(window + 1, T) applies PaLM's T to each layer's attended span: no
  paper states it.

### Upstream optimizer sweep: tuned gains shrink with scale
- **Idea.** Improvements tuned on a small proxy shrink, or reverse, at larger scale, so small-proxy tuning is
  distrusted.
- **Literature.**
  - arXiv 2509.02046 (Wen et al., 2025, §1 Fig. 1 top left and bottom left): an under-tuned AdamW baseline inflates
    speedups (up to 2× from lr alone at 100M); optimizer speedups decay from 1.3–1.4× to 1.1× at 1.2B.
  - arXiv 2512.05620 (Qiu et al., 2025, Abstract): the decay is partly an artefact of incorrect hyperparameter
    scaling.
- **Upstream.** nanochat LOG 2026-01-19..22 (`85b3e95`; ~320 runs; gains ~0.002 bpb at d12 → ~0.0007 at d20;
  d12-tuned values "actively hurt" at d20).
- **Here.** EVIDENCE; `docs/experiments.md` §1 "Proxy settings".
- **Cite as.** arXiv 2509.02046 §1 Fig. 1 (with arXiv 2512.05620 as the caveat). The LOG sweep is not citable as
  evidence; it could be cited as software (karpathy/nanochat) labelled an unreviewed observation.

### Record methodology: n runs and a one-sided t-test of the mean loss
- **Idea.** A record is accepted when the mean final validation loss over n seeds is below the target with a
  one-sided one-sample t-test (e.g. 80 runs, p = 0.0001 for record #18).
- **Literature.** None specific (a standard one-sample t-test). For paired comparisons of models on the same
  evaluation items, arXiv 2411.00640 (Miller, 2024, §4.2) is the statistical treatment used elsewhere here.
- **Upstream.** modded records #18 (`45544f5`), #26 (`72f449c`), #43 (`29aefcf`), #41 (`9f0d8ca`, 20 logs).
- **Here.** EVIDENCE (records cited as support in `docs/architecture.md` §2 and `docs/training.md` §3); this
  repository's own tests (`docs/experiments.md` §1) are different.
- **Cite as.** No paper: the records are unreviewed; cite modded-nanogpt as software with the record number when a
  record's result is mentioned, and do not present it as published evidence.

## 3. Tokenizer, data and evaluation

Document masking across packed documents: see §1, Document masking in packed sequences. Model FLOPs per token: see
§2, FLOPs per token. MFU: see §4, Model FLOPs utilization.

### Byte-level BPE
- **Idea.** BPE builds a subword vocabulary by repeatedly merging the most frequent adjacent pair of symbols, so any
  string can be encoded with no unknown symbols, and the number of merges is the only hyperparameter. Running BPE on
  UTF-8 bytes rather than Unicode code points shrinks the base vocabulary to 256 symbols and still lets the model
  assign a probability to any Unicode string.
- **Literature.** arXiv 1508.07909 (Sennrich et al., 2015, §3.2): adapts byte pair encoding (Gage, 1994, a
  compression algorithm) to word segmentation; "iteratively count all symbol pairs and replace each occurrence of the
  most frequent pair"; final vocabulary = initial vocabulary + merges; open vocabulary (§3.2, §4). Radford et al.,
  2019, "Language Models are Unsupervised Multitask Learners" (OpenAI report, non-archival, §2.2 "Input
  Representation"): a byte-level BPE needs a base vocabulary of 256, against more than 130,000 for code-point BPE;
  the model can be evaluated "on any dataset regardless of pre-processing, tokenization, or vocab size". arXiv
  2302.13971 (Touvron et al., 2023, §2.1 "Tokenizer"): BPE with a fallback "to bytes to decompose unknown UTF-8
  characters", byte coverage by another route (SentencePiece).
- **Upstream.** nanochat `3a5e0bc` (2025-10-13, "BPE Tokenizer in the style of GPT-4", rustbpe trainer over bytes);
  upstream cites no paper.
- **Here.** ADOPTED; `corpus/crate/train.rs` `train` (256 byte tokens + 32,508 merges), `corpus/tokenizer.py`
  `Tokenizer`; `docs/corpus.md` §5.
- **Cite as.** Sennrich et al. 2015 (arXiv 1508.07909, ACL 2016) for BPE; Radford et al. 2019 (GPT-2 report,
  non-archival) for byte-level BPE with 256 base symbols. Gage (1994, C Users Journal) only through Sennrich
  [unverified].

### Vocabulary of 2^15 ids, special tokens counted inside it
- **Idea.** A vocabulary size trades shorter sequences (better compression) against more embedding/output parameters
  and rarer, less-trained tokens; at fixed compute the loss-optimal vocabulary grows with model size, so a small model
  wants a small vocabulary.
- **Literature.** arXiv 2407.13623 (Tao et al., 2024, §3, Fig. 3; §5, Table 1): for each FLOP budget there is an
  optimal vocabulary size, rising with the budget; Table 1's smallest row (N_nv = 3B, compute-optimal data) predicts
  V_opt = 37K–43K across the three approaches. Its models span 33M–3B; d24's 195M non-embedding parameters lie below
  the table, so a predicted optimum ≤ ~40K is an extrapolation [inference]. arXiv 2402.01035 (Dagan et al., 2024,
  §3.2, Table 3): a 1.5B code model given GPT-4-style tokenizers of 32K, 64K, 128K, 256K: HumanEval pass@1 vs
  vocabulary size, Pearson r = −0.13, p = 0.87 (no significant effect); §2.5 and App. B: vocabulary parameters weigh
  more in small models. arXiv 2402.19173 (Lozhkov et al., 2024, §6.2): "increasing the vocabulary size to 100K did
  not improve performance" (StarCoder2 keeps 49,152). arXiv 2604.14053 (Chizhov et al., 2026, App. B, Table 5):
  SA-BPE's code models use 32,768.
- **Upstream.** nanochat `ccf4b7f` (2026-01-07, 65,536 → 32,768 "= 2^15", `scripts/tok_train.py:19`); specials
  subtracted from the size since `3a5e0bc`. No paper cited upstream; the LOG gives no ablation of the size.
- **Here.** ADOPTED; `model/config.py` `VOCAB_SIZE` (comment cites nanochat `92d63d4` `scripts/tok_train.py:19`),
  `corpus/train_tokenizer.py` (32,508 merges = 2^15 − 256 − 4 specials); `docs/corpus.md` §5,
  `docs/architecture.md` §1.
- **Cite as.** Tao et al. 2024 (arXiv 2407.13623) for compute-dependent optimal vocabulary and the small-model
  direction; Dagan et al. 2024 (arXiv 2402.01035 §3.2) and StarCoder2 (arXiv 2402.19173 §6.2) for code performance
  being insensitive to vocabulary size above 32K. The value 2^15 itself is a design choice (no paper fixes it).

### Pre-tokenization by a split regex (GPT-4 / cl100k pattern)
- **Idea.** Before BPE, a regular expression cuts text into chunks (letters, digits, punctuation, whitespace) and
  merges never cross a chunk, so the vocabulary is not spent on variants like "dog." / "dog!" and tokens stay
  compositional.
- **Literature.** GPT-2 report (Radford et al., 2019, non-archival, §2.2): "we prevent BPE from merging across
  character categories for any byte sequence", with an exception for spaces; the stated reason is the sub-optimal
  allocation of vocabulary slots to "dog. dog! dog?". arXiv 2402.01035 (Dagan et al., 2024, §2.4, Fig. 3; §3,
  Table 2; §5): Fig. 3 prints the GPT-2 and GPT-4 regexes clause by clause (the only paper source for cl100k's pattern
  found); GPT-4's pattern limits digit runs to three; at 32K, GPT-4-regex and Punct models match Llama's tokenizer on
  HumanEval/MBPP; "skipping pre-tokenization (Identity) can maximize compression at significant cost" to performance
  (§5).
- **Upstream.** nanochat `3a5e0bc` (`SPLIT_PATTERN`, "in the style of GPT-4"); comment at `92d63d4`
  `nanochat/tokenizer.py:23-26`. No paper cited.
- **Here.** ADAPTED; `corpus/tokenizer.py` `SPLIT_PATTERN` = nanochat's without `[\r\n]*` after a punctuation run;
  `docs/corpus.md` §5; arms in `docs/experiments.md` §2.
- **Cite as.** Radford et al. 2019 (non-archival) for pre-tokenization by character category; Dagan et al. 2024
  (arXiv 2402.01035 §2.4, Fig. 3, Table 2) for the GPT-4 pattern and evidence that the pre-tokenizer changes
  compression and code performance. GPT-4's report does not describe the regex; cl100k is otherwise citable only as
  software (openai/tiktoken).

### Digit runs of one or two (`\p{N}{1,2}`)
- **Idea.** Numbers are chunked into runs of at most k digits before BPE, so number tokens are a fixed, regular set
  instead of idiosyncratic merges; small k makes arithmetic more uniform, larger k compresses more.
- **Literature.** arXiv 2302.13971 (Touvron et al., 2023, §2.1 "Tokenizer"): "we split all numbers into individual
  digits" (k = 1). arXiv 2402.14903 (Singh & Strouse, 2024, Table 1; §3.1): PaLM, LLaMA 1–2 and Mistral use single
  digits; GPT-3.5/GPT-4 (cl100k) use left-to-right chunks of up to three; forcing right-to-left 3-digit chunking
  raises GPT-3.5/4 addition accuracy by up to 20 points. The study varies inference-time formatting of frontier
  models; it trains no models and does not test k = 2. arXiv 2402.01035 (Dagan et al., 2024, §2.4): unconstrained
  number tokens force learning "arithmetic for every token independently"; digit splitting can generalize basic
  arithmetic (citing earlier work).
- **Upstream.** nanochat `3a5e0bc` (k = 2 from the start); LOG 2026-01-13 "Number Token Split Pattern" (`64b48d0`):
  d12, 32K vocabulary, val_bpb 0.969 (k = 1), **0.965** (k = 2), 0.972 (k = 3), one run each; comment `2398541`.
- **Here.** ADOPTED (clause kept in `corpus/tokenizer.py` `SPLIT_PATTERN`); EVIDENCE: the LOG table is the only
  measurement behind k = 2 and is not cited in the docs (`docs/corpus.md` §5, `docs/experiments.md` §2).
- **Cite as.** LLaMA (arXiv 2302.13971 §2.1) and Singh & Strouse (arXiv 2402.14903 Table 1) for digit chunking as a
  design axis (k = 1 vs k ≤ 3). No paper measures k = 2: it is a design choice; nanochat's LOG measurement (one d12
  run per arm, 0.004 bpb) is an unreviewed observation, citable only as software, not as evidence.

### Punctuation run fused with following line breaks (`?[^\s\p{L}\p{N}]++[\r\n]*`)
- **Idea.** cl100k's punctuation clause absorbs trailing line breaks, producing tokens like `"):\n"`; in code the line
  break and indentation then share tokens with punctuation, which the model must emit exactly.
- **Literature.** arXiv 2608.18062 (Meister, 2026, App. F.3, "Code-structure metrics", fn. 16): over 18 byte-level
  math+code models, the count of vocabulary tokens containing a line break correlates with MBPP pass@1 at Spearman
  ρ = −0.717 (p = 8.1e-4). TokEval companion post (Meister, 2026, "Tokenizer Code Ablations", non-archival,
  §"Differing one regex clause"): deleting the trailing `[\r\n/]*` from the punctuation clause, same corpus, three
  seeds: MBPP pass@1 0.086 → 0.230, seed ranges disjoint.
- **Upstream.** nanochat `3a5e0bc` (inherited from cl100k); still in `92d63d4` `SPLIT_PATTERN`.
- **Here.** NOT TAKEN; removed in `dd9f945`; `docs/corpus.md` §5; `docs/experiments.md` §2 (the `sa-bpe-fused` arm =
  nanochat's pattern, a positive control).
- **Cite as.** Meister 2026 (arXiv 2608.18062 App. F.3) for the correlation; the controlled one-clause result is in
  the non-archival companion post only (cite as a blog, marked).

### Train with a Rust BPE trainer, encode with tiktoken
- **Idea.** Separate the (slow, one-off) training of merges from encoding; encoding uses tiktoken's regex + rank
  lookup over the mergeable ranks.
- **Literature.** None found.
- **Upstream.** nanochat `3a5e0bc` (`RustBPETokenizer`: "train with rustbpe, inference with tiktoken").
- **Here.** ADOPTED (architecture); `corpus/tokenizer.py` `Tokenizer.enc` (`tiktoken.Encoding`), own pyo3 crate
  `corpus/crate/train.rs`; `docs/corpus.md` §5, §10.
- **Cite as.** No paper: an engineering choice; cite openai/tiktoken and karpathy/rustbpe as software if needed.

### Incremental BPE trainer (rustbpe design) with SA-BPE's merge score
- **Idea.** Count each unique pre-token chunk once with its frequency, index pairs → chunks, and after each merge
  update pair counts only for affected chunks, refreshing a lazy max-heap; SA-BPE replaces the frequency criterion F
  by F·log R (R = number of repositories holding the pair) to stop merges that are frequent only inside one source
  and so become under-trained tokens.
- **Literature.** arXiv 2604.14053 (Chizhov et al., 2026, §4.1 "Priority criterion"; Table 3): criteria F·L,
  F·log(R+1), F·log R (zeroes single-repository pairs) and combinations; Table 3 compares compression and verified
  under-trained tokens (as used in `docs/corpus.md` §5: F·log R 3.643 and 72 vs F's 640). arXiv 2405.05417 (Land &
  Bartolo, 2024): the under-trained-token problem SA-BPE targets.
- **Upstream.** nanochat `3a5e0bc` `rustbpe/src/lib.rs` ("Core incremental BPE training", "Lazy refresh"), moved to
  karpathy/rustbpe (`aa42f40`).
- **Here.** ADAPTED; `corpus/crate/train.rs` `train`, `Word::merge_pair` (score F·ln R, exact R, merges whose bytes
  already form a token skipped); `docs/corpus.md` §5.
- **Cite as.** Chizhov et al. 2026 (arXiv 2604.14053 §4.1, Table 3) for the merge criterion. The incremental
  pair-count algorithm has no paper: cite karpathy/rustbpe as software.

### Merge tie-break: on equal score, the smaller pair
- **Idea.** A total order on merge candidates makes training deterministic and independent of thread count.
- **Literature.** None.
- **Upstream.** nanochat `3a5e0bc` `impl Ord for MergeJob` ("tie-break to ascending pair order (deterministic)").
- **Here.** ADOPTED inside an extended order (score, then F, then pair); `corpus/crate/train.rs:75-77`;
  `docs/corpus.md` §5.
- **Cite as.** No paper: implementation detail.

### BOS as the only document delimiter; special ids after the mergeable ranks
- **Idea.** One special token marks a document start; ids for specials are appended after the BPE ranks so the
  mergeable vocabulary is contiguous.
- **Literature.** None establishes the layout. The FIM sentinels added here follow DeepSeek-Coder's names (arXiv
  2401.14196 §3.1.2 writes `<|fim_start|>`, `<|fim_hole|>`, `<|fim_end|>`; the released tokenizer's are used).
- **Upstream.** nanochat `3a5e0bc` (`SPECIAL_TOKENS = ["<|bos|>", …8 chat tokens]`, ids `len(mergeable_ranks) + i`).
- **Here.** ADAPTED; `corpus/tokenizer.py` `BOS`, `SPECIAL_TOKENS` (BOS id 32,764, three FIM sentinels; chat tokens
  removed in `e5c15b1`); `docs/corpus.md` §5.
- **Cite as.** No paper: convention; DeepSeek-Coder (arXiv 2401.14196 §3.1.2) only for the FIM sentinel names.

### Chat special tokens
- **Idea.** Eight tokens delimiting user/assistant/tool turns for fine-tuning.
- **Literature.** None relevant to pretraining.
- **Upstream.** nanochat `3a5e0bc` `SPECIAL_TOKENS`.
- **Here.** NOT TAKEN (pretraining only; `e5c15b1`); post-training not implemented (`docs/training.md` §10).
- **Cite as.** Not needed.

### Tokenizer training sample (`--max-chars` 2e9 of the first shards, `--doc-cap` 10,000)
- **Idea.** The text used to fit merges should resemble the training mixture, since merge frequencies follow it.
- **Literature.** arXiv 2405.05417 (Land & Bartolo, 2024, §5): tokenizer training data should be "representative of
  model training data". arXiv 2402.01035 (Dagan et al., 2024, §2.3): a subset's compression varies by 5–6 points
  across tokenizer training mixes, higher as the mix holds more of it.
- **Upstream.** nanochat `3a5e0bc` (`--max_chars` 10e9, doc cap 10,000), `8630d32` (2e9).
- **Here.** NOT TAKEN; `corpus/train_tokenizer.py` `sample`, `SAMPLE_BYTES` = 13.38e9 bytes drawn per source in
  proportion to the file-stage mixture, whole documents; `docs/corpus.md` §5.
- **Cite as.** Land & Bartolo 2024 (arXiv 2405.05417 §5) and Dagan et al. 2024 (arXiv 2402.01035 §2.3) for drawing
  the sample in mixture proportions; the sample size is a design choice.

### Tokenizer compression evaluation against GPT-2/GPT-4 (`scripts/tok_eval.py`)
- **Idea.** Compare tokenizers by compression ratio (bytes per token).
- **Literature.** arXiv 2608.18062 (Meister, 2026): TokEval, intrinsic measures beyond compression and their
  correlation with downstream scores (App. F); arXiv 2402.18376 (Schmidt et al., 2024, Abstract): PathPiece, which
  segments text into the fewest tokens for a vocabulary, does not improve downstream performance; fewer tokens is not
  the goal by itself.
- **Upstream.** nanochat `scripts/tok_eval.py` (gpt2, cl100k_base, ours).
- **Here.** NOT TAKEN; replaced by TokEval-style intrinsic measures; `docs/evaluation.md` §5 (definitions),
  `docs/experiments.md` §2 (results).
- **Cite as.** Meister 2026 (arXiv 2608.18062) for the intrinsic measures used instead.

### BOS-aligned rows with best-fit packing
- **Idea.** Concatenate-and-chunk packing cuts documents at row boundaries, so many targets are predicted without
  their document's start and with misleading context; packing whole documents (or minimal segments) into rows by bin
  packing removes those truncations at almost no padding cost.
- **Literature.** arXiv 2404.10830 (Ding et al., 2024, §3, §3.1, Table 2): segment each document into chunks of at
  most L, pack chunks by Best-Fit-Decreasing; ≤ 0.01% more sequences than concatenation (0.0024% RefinedWeb at 2K,
  0.00063% at 8K, 0.0028% Stack at 2K); Abstract: up to +16.8% context following, +9.2% program synthesis, up to
  58.3% less closed-domain hallucination; §2: truncation analysis.
- **Upstream.** nanochat `43c29dd` (2026-01-13, `_bos_bestfit`: every row starts at BOS, largest doc that fits, crop
  when none fits); LOG 2026-01-13 "BOS-Aligned Dataloader with Bin Packing" (T = 2048: greedy-crop 39.4% cropped,
  BestFit-Crop 34.6%, FFD 99.7% utilization, 23.0% crop, 0.3% pad, 22.9% theoretical minimum). modded record #26
  (`72f449c`, "Align batch starts with EoS"; README: 20 runs, mean 3.2791, p = 0.0049). Neither cites a paper.
- **Here.** ADAPTED; `corpus/crate/pack.rs` `best_fit_decreasing` (comment cites arXiv 2404.10830 §3.1),
  `corpus/loader.py` `PretrainingBatches.assemble` (BOS before every piece; documents cut at statement/line
  boundaries, BFD with padding, nothing cropped); `docs/corpus.md` §8, §9.
- **Cite as.** Ding et al. 2024 (arXiv 2404.10830 §3.1, Table 2) for Best-Fit-Decreasing packing, its negligible
  padding and its downstream gains. nanochat's crop/FFD percentages and record #26's t-test are unreviewed
  observations (software/record citation only).

### BOS as the end-of-document target and the generation stop
- **Idea.** The token after a document's last token is the next BOS, so the model learns to end documents; at
  inference a sampled BOS ends the continuation.
- **Literature.** None found that states it for a BOS-only scheme.
- **Upstream.** nanochat `3a5e0bc` `engine.py` (sampled BOS ends a row); implicit in `43c29dd`'s rows.
- **Here.** ADAPTED; `corpus/loader.py` `assemble` (BOS is a target only after a piece that ends its document or a
  FIM example, else `IGNORE_INDEX`); `inference/decoder.py` (FIM middle ends with BOS), `inference/export_hf.py`
  (`eos_token_id` = BOS); `docs/corpus.md` §9, `docs/evaluation.md` §6, §8.
- **Cite as.** No paper: convention.

### ClimbMix as the web-text source
- **Idea.** Cluster a large web corpus in embedding space, prune low-quality clusters and search cluster mixture
  weights with proxy models; the resulting 400B-token mix trains better models per token than single-filter web
  datasets.
- **Literature.** arXiv 2504.13161 (Diao et al., 2025, §2; §6, Fig. 1): CLIMB clustering + iterative mixture search;
  Nemotron-ClimbMix (400B tokens, from Nemotron-CC + smollm-corpus, 20 clusters in ClimbLab); a 1B model trained on
  it "significantly outperform[s]" models trained on Nemotron-CC, SmolLM, DCLM-baseline and FineWeb-Edu at equal
  token budget (§6; Fig. 1's values not transcribed).
- **Upstream.** nanochat `324e69c` (2026-03-04, FineWeb-EDU → `karpathy/climbmix-400b-shuffle`); LOG 2026-03-04
  (GPT-2 speedrun 2 h 46 min → 2 h 1 min; d26 → d24; links `nvidia/Nemotron-ClimbMix`);
  `dev/repackage_data_reference.py` (`92d63d4:51-60`: the source is stored as GPT-2 tokens, decoded with tiktoken,
  shuffled, repackaged).
- **Here.** ADOPTED (as the 10% text group, read in part); `corpus/sources.py:84`, `corpus/mixture.py:93,99`;
  `docs/corpus.md` §1, §7 (the 10% share's basis is DeepSeek-Coder's, owned there).
- **Cite as.** Diao et al. 2025 (arXiv 2504.13161 §6) for Nemotron-ClimbMix and its per-token advantage over
  FineWeb-Edu/DCLM-baseline; cite the dataset `karpathy/climbmix-400b-shuffle` (a re-packaging of
  `nvidia/Nemotron-ClimbMix`) as the artifact. nanochat's speedrun gain is unreviewed.

### Data-source negatives against FineWeb-Edu (Olmo 3 mix, FinePDFs/DCLM/FineWeb-Edu mixture, FineWeb)
- **Idea.** Educationally filtered web text (FineWeb-Edu) is a strong baseline that several alternative mixtures did
  not beat at nanochat scale until ClimbMix.
- **Literature.** arXiv 2406.17557 (Penedo et al., 2024, §4, Fig. 10–11): FineWeb-Edu beats FineWeb and other open
  web datasets on the aggregate (about +2%) and reaches Matrix's MMLU with ~10× fewer tokens (33.6% at 38B tokens).
  arXiv 2504.13161 (§6), previous entry: ClimbMix > FineWeb-Edu, DCLM-baseline.
- **Upstream.** LOG 2026-01-15 (Olmo 3 mix, d16 CORE 15.5 → 13.8); 2026-02-17 (FinePDFs/DCLM/FineWeb-Edu mixture,
  d26 0.2602 → 0.2549, d18 0.199 → 0.192; FineWeb, d26 0.2602 → 0.2241); commits `747ed44`, `f5fe792`, `48804bf`.
- **Here.** EVIDENCE (applicable, unused); bears on the text group's source (`docs/corpus.md` §1, §7).
- **Cite as.** Penedo et al. 2024 (arXiv 2406.17557 §4) for FineWeb-Edu > FineWeb; Diao et al. 2025 (arXiv
  2504.13161 §6) for ClimbMix > FineWeb-Edu. nanochat's CORE numbers are single runs and not citable as evidence; the
  claim here would need an own ablation.

### Validation split = last parquet shard
- **Idea.** Hold out data by position in a pre-shuffled shard list.
- **Literature.** None needed.
- **Upstream.** nanochat `92d63d4` `nanochat/dataset.py:70,75`.
- **Here.** NOT TAKEN; curated per-source val splits (`docs/corpus.md` §3).
- **Cite as.** Not needed.

### Bytes per token (`token_bytes`), 0 for special tokens
- **Idea.** To normalize loss by text length independent of the tokenizer, each target token is weighted by the
  number of UTF-8 bytes it decodes to; tokens that are not text (specials) carry 0 bytes.
- **Literature.** arXiv 2101.00027 (Gao et al., 2020, §3.1): bits per UTF-8 byte, bpb = (L_T / L_B)·ℓ / ln 2 with L_B
  the length in UTF-8 bytes; preferred "due to its invariance to different tokenization schemes".
- **Upstream.** nanochat `3a5e0bc` `scripts/tok_train.py` (0 for specials); `2ce972a` (raw token bytes, not a decode
  round trip; 193/32,768 tokens had been wrong, bpb ~0.05% higher).
- **Here.** ADOPTED; `corpus/tokenizer.py` `Tokenizer.token_lengths` (byte lengths of the mergeable tokens, 0 for
  `SPECIAL_TOKENS`); `docs/evaluation.md` §2.
- **Cite as.** Gao et al. 2020 (arXiv 2101.00027 §3.1) for counting UTF-8 bytes; the 0 for specials is a convention
  (this repository's and nanochat's), no paper.

### Bits per byte with zero-byte and ignored targets excluded
- **Idea.** Report Σ nats / (ln 2 · Σ bytes) over the targets, which compares models with different tokenizers;
  targets that are not text (BOS, sentinels) or are masked contribute neither nats nor bytes.
- **Literature.** arXiv 2101.00027 (Gao et al., 2020, §3.1, Table 2): definition and tokenizer invariance of bpb.
  arXiv 2407.13623 (Tao et al., 2024, §2.2, Eq. 4) normalizes loss across vocabularies differently (unigram-normalized
  loss) for the same comparability problem.
- **Upstream.** nanochat `3a5e0bc` `nanochat/loss_eval.py` `evaluate_bpb` ("a tokenization vocab size-independent
  metric"; specials and ignore-index targets masked); upstream cites no paper.
- **Here.** ADAPTED; `eval/bpb.py` `sums` (fp64, per context and byte-offset bucket, per row), `metrics.py` `bpb`,
  `mixture_bpb`; docstring cites nanochat `92d63d4`; `docs/evaluation.md` §2, §5.
- **Cite as.** Gao et al. 2020 (arXiv 2101.00027 §3.1) for bits per byte and its tokenizer invariance. The exclusion
  of zero-byte targets has no paper: a convention consistent with the definition (non-text targets have no bytes)
  [inference]; cite nanochat as software if attribution is wanted.

### Validation budget of 80 × 2^19 tokens
- **Idea.** A fixed number of validation tokens per evaluation, large enough that the loss estimate's noise is below
  the differences of interest.
- **Literature.** No paper fixes the value. arXiv 2411.00640 (Miller, 2024, §2.2, Eq. 4; §5; App. C): clustered
  standard errors and sample-size formulas for choosing an evaluation's size by power; the method behind this
  repository's cluster standard errors, not the 80 × 2^19 figure.
- **Upstream.** nanochat `3a5e0bc` (20 × 524,288), `6079f78` (40×), `324e69c` (80×; LOG 2026-03-04: "doubled … for
  more stable validation loss estimates").
- **Here.** ADAPTED; `eval/bpb.py` `VAL_TOKENS` (per corpus, rounded up to whole micro-batches, rows drawn
  systematically, `sample_rows`); `docs/evaluation.md` §2, §5.
- **Cite as.** No paper for the value: a design choice inherited from nanochat (software citation). For its adequacy
  report the repository's standard errors (arXiv 2411.00640 §2.2) rather than the inheritance.

### CORE metric (DCLM's 22-task centered accuracy)
- **Idea.** Average of rescaled accuracies over low-variance tasks, 0 = chance, 1 = perfect, for small-scale signal.
- **Literature.** arXiv 2406.11794 (Li et al., 2024, §3.5 "Evaluation"): Core centered accuracy over 22 tasks.
- **Upstream.** nanochat `nanochat/core_eval.py` (cites arXiv 2406.11794), `scripts/base_eval.py`.
- **Here.** NOT TAKEN; code-specific evaluation instead (`docs/evaluation.md` §1–§4).
- **Cite as.** DCLM (arXiv 2406.11794 §3.5) only if the paper explains why CORE is not reported.

### HumanEval task with upstream's execution sandbox
- **Idea.** Functional correctness of generated code by executing tests in a sandbox.
- **Literature.** arXiv 2107.03374 (Chen et al., 2021, §2.1 pass@k; §2.3 sandbox).
- **Upstream.** nanochat `tasks/humaneval.py`, `nanochat/execution.py`.
- **Here.** NOT TAKEN; EvalPlus/MultiPL-E/infilling with evalplus's `reliability_guard` (`eval/verify.py:141-148`);
  `docs/evaluation.md` §4, §5.
- **Cite as.** Chen et al. 2021 (arXiv 2107.03374 §2.1, §2.3) for pass@k and sandboxed execution.

### Temperature + top-k sampling
- **Idea.** Truncate the sampling distribution to the k most likely tokens.
- **Literature.** arXiv 1904.09751 (Holtzman et al., 2019, §3.1–3.2): nucleus (top-p) sampling, truncating to the
  smallest set holding mass p, against top-k's fixed k.
- **Upstream.** nanochat `92d63d4` `nanochat/engine.py:141` `sample_next_token(temperature, top_k)`.
- **Here.** NOT TAKEN; `NativeDecoder` + `model/kernels/decode.cu` (temperature, top-p); `docs/evaluation.md` §6.
- **Cite as.** Holtzman et al. (arXiv 1904.09751 §3.1) for the nucleus sampling used.

## 4. Systems and numerics

Document masking through varlen attention: see §1, Document masking in packed sequences. The fused output-logit cap's
modelling basis: see §1, Output-logit soft cap.

### FlashAttention kernels for training attention
- **Idea.** Exact attention computed tile by tile in on-chip SRAM with an online softmax, never materializing the N×N
  score matrix. This cuts HBM traffic, which is the bottleneck. Later versions repartition the work across thread
  blocks and warps (FA2), add asynchrony and warp specialization on Hopper (FA3), and pipeline around Blackwell's slow
  exponential and shared-memory units (FA4).
- **Literature.**
  - arXiv 2205.14135 (Dao et al., 2022, §3.1, Thm. 2, Abstract): tiling plus recomputation. Θ(N²d²M⁻¹) HBM accesses
    against Θ(Nd + N²) for standard attention; 3× end-to-end speedup on GPT-2 (seq. 1K).
  - arXiv 2307.08691 (Dao, 2023, §1, §3.2): fewer non-matmul FLOPs and parallelism over the sequence length; ~2× over
    FA1, up to 73% of A100 peak forward, 225 TFLOP/s end-to-end on GPT training.
  - arXiv 2407.08608 (Shah et al., 2024, Abstract, §3.1–§3.2): the Hopper kernel (warp-specialized
    producer/consumer, pingpong, GEMM–softmax overlap). FP16 forward 1.5–2.0× over FA2, up to 740 TFLOP/s on H100;
    FA2 reached only 35% utilization on H100.
  - arXiv 2603.05451 (Zadouri et al., 2026, Abstract, §3.3, §4): FA4. Written entirely in CuTe-DSL (Python, JIT,
    20–30× faster compiles than FA3's C++ templates); LPT tile scheduling for causal and varlen; up to 1,613 TFLOP/s
    (71%) on B200. Every benchmark in the paper is on Blackwell.
- **Upstream.** nanochat `2ff7d51` and LOG 2026-01-11: FA3 through the `kernels` hub, "+9% tok/sec" at d12; FA3 "2x
  faster than FA2" at batch 32, seq 2048. `8203efa` and LOG 2026-01-16 added an SDPA fallback. modded record #12
  (FlexAttention, `ebce774`) and #29 (FA3, PR #118, `8386811`): the README says FA3 has "greater SM utilization on
  Hopper GPUs than Flash Attention v2". Neither upstream cites a paper.
- **Here.** NOT TAKEN for upstream's FA3 integration; the same family is used through FA4. `model/ops/flash_attn.py`
  `attention` runs FA4 (CuTe DSL, `flash-attn-4==4.0.0b36`) varlen on the GH200's SM90 kernels, with a persistent
  forward and bf16 dQ in the window layers (`docs/architecture.md` §3, `0e627ff`). "FA3 not pursued" and the rejected
  hand kernels: `docs/experiments.md` §5.
- **Cite as.** arXiv 2205.14135 for IO-aware exact attention; arXiv 2407.08608 for the Hopper algorithm; arXiv
  2603.05451 for the implementation used (CuTe-DSL FA4, its scheduler). Its Hopper (SM90) path has no evaluation in
  the paper, so speed claims on the GH200 rest on this repository's measurements. arXiv 2307.08691 when crediting the
  sequence-parallel partitioning and the dQ atomics.

### Sliding windows inside the attention kernel; block-sparse windows
- **Idea.** A local (sliding-window) mask makes most score tiles fully masked. A kernel that skips those tiles cuts IO
  and compute in proportion to the fraction of tiles kept. Separating fully-visible from partially-masked tiles also
  avoids evaluating the mask on most tiles.
- **Literature.**
  - arXiv 2205.14135 (Dao et al., 2022, §3.3, Prop. 4): block-sparse FlashAttention skips zero blocks;
    Θ(Nd + N²d²M⁻¹s) HBM accesses, s the fraction of nonzero blocks.
  - arXiv 2412.05496 (Dong et al., 2024, §3.1, §4.2, Fig. 3): `mask_mod` expresses sliding-window and document masks.
    A precomputed BlockMask skips fully masked blocks and applies the mask only to partial blocks ("approximately a
    15% performance improvement for common patterns such as causal masks").
- **Upstream.** nanochat LOG 2026-01-11 ("FA3 supports sliding window via `window_size=(left, 0)`"). modded #16
  (`0690f2c`, block sliding window with separate full/partial block masks); the window was first rounded to 128-token
  blocks in #15's script (`8011fe7`).
- **Here.** ADAPTED. FA4's `window_size_left` with a token-exact 512-key window (`model/modeling.py`
  `layer_windows`; `flash_attn.py` `fa4_window_left` gives FA4's causal kernel when the window spans T;
  `docs/architecture.md` §1, §3). Block-aligned windows were NOT TAKEN (`docs/experiments.md` §3). Window-tile
  skipping and the measured per-tile fixed cost: `docs/experiments.md` §5 (`window-attn` `2bf0e12`).
- **Cite as.** arXiv 2205.14135 §3.3 and arXiv 2412.05496 §4.2 for block-sparse execution of window masks. The
  modelling basis of the window: §1, Local–global attention.

### Fixed-size `cu_seqlens` for static compilation, and passing only the real segments
- **Idea.** A static-shape compiler recompiles for every new input shape, so varlen offsets are padded to a fixed
  length. The kernel then launches work for every listed segment, so the padding segments cost time without changing
  outputs. Passing only the real segments removes that cost.
- **Literature.**
  - PyTorch 2, ASPLOS '24 (Ansel et al., 2024, §5): static-shape compilers "must recompile if any input shape
    changes". Not on arXiv; peer-reviewed (DOI 10.1145/3620665.3640366).
  - arXiv 2307.08691 (Dao, 2023, §3.2): one thread block per row block of each (sequence, head) in the forward pass,
    the mechanism by which every listed segment launches work.
  - arXiv 2603.05451 (Zadouri et al., 2026, §3.3): the varlen kernel reads the per-batch lengths at runtime and
    schedules over them.
- **Upstream.** nanochat LOG 2026-01-13: variable-length `cu_seqlens` "caused torch.compile recompilation
  (25s/iter!) - fixed by padding to fixed size". modded #29 (`max_num_docs` padding) and #78 (PR #246, `5845e94`, by
  Chris McCormick): tightening `max_num_docs` gained −0.498 s over 4 runs (86.74 → 86.24 s). McCormick's blog post
  "Optimizing Training with FlashAttention varlen" (2026) is non-archival.
- **Here.** ADAPTED. The loader keeps a fixed `num_segments + 1` offsets, and the op passes the first `segments` + 1
  (`flash_attn.py` `segmented`, a 0-dim host count). Landed in `a81410c`, whose −5.3% d24 8K step also includes the
  batched Newton–Schulz, so the two cannot be separated (`docs/architecture.md` §3 "Real segments only").
- **Cite as.** arXiv 2307.08691 §3.2 or arXiv 2603.05451 §3.3 for why padding segments cost kernel work; PyTorch 2 §5
  for why the shape is fixed. No paper measures the cost of padded `cu_seqlens`: an engineering observation
  (modded-nanogpt record #78 and McCormick's post, citable only as software or web sources). The 5.3% here is a
  combined effect.

### Mixed precision: fp32 master weights, bf16 GEMM operands with fp32 accumulation, explicit dtypes
- **Idea.** Keep an fp32 master copy of the weights for the update, and run forward and backward on a 16-bit copy with
  fp32 accumulation. Small updates survive in fp32, and the GEMMs run on 16-bit tensor cores at half the storage and
  bandwidth. bf16 has fp32's exponent range, so it trains without loss scaling or hyperparameter changes.
- **Literature.**
  - arXiv 1710.03740 (Micikevicius et al., 2017, §3.1, Fig. 1, §3.3): "an FP16 copy of the master weights is used in
    the forward and backward pass". Updating FP16 weights without the FP32 master copy gave 80% relative accuracy loss
    on the Mandarin speech model. Dot products accumulate in FP32.
  - arXiv 1905.12322 (Kalamkar et al., 2019, Abstract, §3, Fig. 1): BF16 GEMM inputs with FP32 outputs, and an FP32
    weight copy converted to BF16 for the forward pass. Matches FP32 "in the same number of iterations and with no
    changes to hyper-parameters".
- **Upstream.** nanochat `1076f97` and LOG 2026-03-04 "Remove autocast, explicit dtype management"
  (`COMPUTE_DTYPE`; `Linear` casts the weight per forward). modded record #10 "Bfloat16 activations" (`6671ed1`,
  `CastedLinear`; README 8.2 → 7.8 min). Neither cites a paper.
- **Here.** ADAPTED. fp32 parameters, gradients and optimizer state. Persistent bf16 GEMM copies
  (`GemmLinear.weight_bf16`, `Attention.qkv_bf16`) are refreshed in place after every update
  (`CausalLM.refresh_gemm_weights`), not cast per forward; there is no autocast (`docs/architecture.md` §1
  "Precision", §4).
- **Cite as.** arXiv 1710.03740 §3.1 and arXiv 1905.12322 §3 for the fp32-master/bf16-compute scheme. The refreshed
  persistent copy is the "FP16 copy of the master weights" of arXiv 1710.03740 §3.1, Fig. 1 (bf16 here). Explicit
  dtype management instead of autocast is a code-structure choice and needs no citation.

### fp16 with loss scaling (GradScaler)
- **Idea.** FP16's narrow exponent range flushes small gradients to zero. Scaling the loss shifts the gradients into
  range. bf16 keeps fp32's range and needs no scaling.
- **Literature.**
  - arXiv 1710.03740 (Micikevicius et al., 2017, §3.2): loss scaling. Constant factors of 8 to 32K were used, and
    some networks diverge without it.
  - arXiv 1905.12322 (Kalamkar et al., 2019, §3): BF16's range represents small gradients "without applying
    complicated loss scaling methods".
- **Upstream.** nanochat `1076f97` and LOG 2026-03-04 added fp16 with `torch.amp.GradScaler`.
- **Here.** NOT TAKEN; bf16 only (`docs/architecture.md` §4).
- **Cite as.** arXiv 1905.12322 §3 for why bf16 needs no loss scaling; arXiv 1710.03740 §3.2 for what loss scaling
  solves.

### Reductions, softmax and loss statistics in fp32
- **Idea.** Large sums (normalization statistics, softmax normalizers, cross-entropy) lose accuracy in 16 bits. Doing
  the arithmetic in fp32 costs nothing in memory-bound layers.
- **Literature.**
  - arXiv 1710.03740 (Micikevicius et al., 2017, §3.3): "Large reductions … should be carried out in FP32", for
    batch-norm statistics and softmax. This does not slow these memory-bandwidth-bound layers.
  - arXiv 2407.08608 (Shah et al., 2024, §4.3): FlashAttention keeps the softmax intermediates in FP32. In FP16 this
    gives 1.7× lower RMSE than standard attention against an FP64 reference.
- **Upstream.** nanochat LOG 2026-03-04 (`F.rms_norm`, `F.cross_entropy` and FA "handle their own dtypes"). modded
  #37 "Compute cross entropy in BF16 during training" (PR #133). The #60 record's fused kernels "use float32 for
  internal computations".
- **Here.** ADAPTED. Norm statistics in fp32 (`model/modeling.py` `norm`; `linear.py` `residual_rms` from GEMM
  epilogues). The loss comes from the logits GEMM's fp32 accumulators (`model/ops/cross_entropy.py` `lm_loss`). The
  q/k prologue and the value-embedding mix compute in fp32 and round once (`docs/architecture.md` §4). modded #37's
  bf16 cross entropy was NOT TAKEN.
- **Cite as.** arXiv 1710.03740 §3.3 for the rule; arXiv 2407.08608 §4.3 for the measured accuracy benefit of fp32
  softmax statistics.

### FP8 training
- **Idea.** Run GEMMs in 8-bit floating point (E4M3 for weights and activations, E5M2 or E4M3 for gradients), with
  per-tensor or finer scaling factors and fp32 accumulation. This gives ~2× tensor-core throughput and less memory.
  The scaling and casts are extra memory passes, and Hopper's FP8 weight-gradient GEMMs need transposed copies.
- **Literature.**
  - arXiv 2209.05433 (Micikevicius et al., 2022, §2, §3, §4.1 Table 4): the E4M3/E5M2 formats and per-tensor
    scaling. GPT models from 126M to 175B match bf16 perplexity (126M: 19.14 bf16 against 19.24 FP8; 175B: 6.65
    against 6.68, within run-to-run noise).
  - arXiv 2310.18313 (Peng et al., 2023, §1): FP8 gradients and optimizer states as well. 29–39% less memory;
    GPT-175B trains 37% faster than with Transformer Engine on H100.
  - arXiv 2412.19437 (DeepSeek-AI, 2024, §3.3; App. B.1; §3.5.2): 1×128 / 128×128 fine-grained scaling with
    higher-precision accumulation; relative loss error below 0.25% against BF16 at 16B and 230B MoE scale. §3.5.2
    describes the separate cast and the "dequantized, transposed, re-quantized" HBM round trips of the backward pass,
    the overheads this repository measured.
- **Upstream.** nanochat LOG 2026-01-13 "FP8 Training for lm_head" (d12, ~1% faster, +2 GB; "following
  modded-nanogpt"; negative). LOG 2026-02-02 "FP8 Training with torchao" (`6079f78`): d26 tensorwise 1.17× tok/s,
  rowwise 0.90×, d12 slower, "~5%" capability-matched at d24. modded #19 (FP8 head, `602014e`), #84 (PR #306), #89
  (PR #342), #90 (PR #344). nanochat cites torchtitan's 25–28%, not a paper.
- **Here.** NOT TAKEN. Training is bf16. FP8 MLP GEMMs: torchao d24 tensorwise −2.8%; Transformer Engine −3.9% to
  −10% at d24 8K. The causes measured were DRAM traffic from separate casts and transposed FP8 copies at K = 768
  (`docs/experiments.md` §5, marked [unverified: no record]).
- **Cite as.** arXiv 2209.05433 (formats, accuracy parity) and arXiv 2412.19437 §3.3 and App. B.1 (fine-grained
  recipe, <0.25%) for FP8 as an option; arXiv 2412.19437 §3.5.2 for the cast/transpose memory overhead. No paper
  establishes that FP8 fails to pay at ~200M parameters with K = 768: that verdict is this repository's measurement.
  nanochat's "larger models benefit more" is an uncitable upstream observation, consistent with it.

### Fused linear cross entropy (no logits tensor) with the logit cap inside
- **Idea.** Compute the loss and its gradients from the classifier GEMM tile by tile with an online log-sum-exp. The
  N×V logits are never materialized, which removes the largest activation of a large-vocabulary model and its HBM
  round trips.
- **Literature.**
  - arXiv 2411.09009 (Wijmans et al., 2024, §4.1–§4.3, §5.1 Table 1): Cut Cross-Entropy, an indexed matmul plus a
    linear-log-sum-exp in SRAM. On Gemma 2 2B (8,192 tokens, V = 256K, A100), the loss's memory drops from 24 GB to
    1 MB and loss + gradient from 28 GB to 1.16 GB (Baseline, row 5, against CCE, row 1). CCE's loss + gradient takes
    145 ms against torch.compile's (row 4) 143 ms, +1.4% [arithmetic]; the §5.1 text states 6%, 2 ms.
  - arXiv 2410.10989 (Hsu et al., 2024, §3.2 "FusedLinearCrossEntropy"): chunked head projection with in-place
    gradient replacement.
- **Upstream.** modded #60 (PR #199, fused softcapped cross entropy, Triton), #67 (PR #207), #75 (PR #235), #79
  (PR #251, "Fuse CE fwd and bwd to avoid recalc on softcap sigmoid"). These read a materialized logits tensor. No
  paper is cited.
- **Here.** PARALLEL, without materialized logits. `model/ops/cross_entropy.py` `lm_loss` uses QuACK's scaled-exp
  linear cross entropy per 8,192-row chunk, with tanh applied on the fp32 accumulator and 1 − tanh² folded into the
  stored exponentials. `067e60b`: +0.68% at 8K, peak 76.99 → 71.44 GiB. `e24333d`: the cap costs 0.58% of the 8K
  step (`docs/architecture.md` §3–§4). Liger and CCE themselves were not adopted (`docs/experiments.md` §5).
- **Cite as.** arXiv 2411.09009 for loss computation without materialized logits (and its memory argument); arXiv
  2410.10989 as the chunked alternative. The capped variant in the GEMM epilogue has no paper; it is QuACK's kernel
  (software) plus this repository's modification.

### Fused GEMM epilogues (linear + ReLU²) and storing versus recomputing activations
- **Idea.** Apply elementwise work (activation, norm scale, residual add) in the GEMM epilogue while the tile is in
  registers, removing memory-bound passes. For the backward pass, store the activations that are expensive to
  recompute and recompute the cheap, large ones.
- **Literature.**
  - arXiv 2104.04473 (Narayanan et al., 2021, §4.2, §5.8): fused bias + GeLU and bias + dropout + add, and fused
    scale-mask-softmax. Throughput +19% on GPT-3 175B (113 → 135 TFLOP/s per GPU) and +11% on 530B.
  - PyTorch 2, ASPLOS '24 (Ansel et al., 2024, §6.4, Table 4): Inductor's matmul templates with pointwise epilogue
    fusion. Removing them costs 0.04 of the 1.45× training geomean; removing fusion costs 0.18.
  - arXiv 2410.10989 (Hsu et al., 2024, §2.2): operation fusion avoids the HBM↔SRAM transfers that leave the SMs
    idle.
  - arXiv 2205.05198 (Korthikanti et al., 2022, §5): selective recomputation stores what is expensive to recompute
    and recomputes the large but cheap parts.
  - arXiv 1604.06174 (Chen et al., 2016, Abstract): the memory–compute trade-off of recomputation, O(√n) memory for
    one extra forward pass.
- **Upstream.** modded #59 (PR #197, `647d498`): a Triton fused linear + ReLU² that stores the pre-activation.
- **Here.** PARALLEL. `model/ops/linear.py` `relu2_mlp` uses QuACK `gemm_norm_act` with the norm's rstd as a row
  scale, the residual as the down projection's C, and the backward's dact × rstd in one epilogue. The forward stores
  both pre-activation and activation (`6299e84`: +1.42% at 8K, peak 61.8 → 75.3 GiB). The norm statistics come from
  o_proj's epilogue (`residual_rms`, `373b549`, +1.15%) (`docs/architecture.md` §3).
- **Cite as.** arXiv 2104.04473 §4.2/§5.8 and PyTorch 2 §6.4 for epilogue/operator fusion as a throughput method.
  Cite arXiv 2205.05198 §5 (with arXiv 1604.06174) as the published framing of trading memory for recomputation; the
  saved MLP activation is the opposite choice on that axis (memory spent, compute saved). The norm-as-row-scale fold
  (norm(h)Wᵀ = diag(rstd)·hWᵀ) is this repository's design (no paper found).

### Fused QK-norm + RoPE kernel
- **Idea.** Compute the per-head RMS norm, gain and rotary embedding of q and k in one memory-bound pass instead of
  several.
- **Literature.** arXiv 2410.10989 (Hsu et al., 2024, §3.2 "RMSNorm", "RoPE"): fused RMSNorm (normalization and
  scaling in one kernel) and a single fused RoPE kernel for q and k. These are separate kernels, not norm and RoPE
  together.
- **Upstream.** modded #90 (PR #344, `9346b0a`, Triton `qk_norm_rope_pad_forward/backward`).
- **Here.** PARALLEL. `model/kernels/qk_prologue.cu` and `qk_row.cuh` do norm → gain → RoPE in fp32 with one bf16
  rounding, shared bitwise with the decoder (`62b50c2`: forward 250.3 → 183.6 µs per call). Fusing it into the
  projection epilogue or into attention was rejected (`docs/experiments.md` §5). The modelling choice: §1, QK-norm.
- **Cite as.** arXiv 2410.10989 §3.2 for fused RMSNorm and RoPE kernels as practice. The combined norm–gain–RoPE
  prologue has no paper; it is a kernel design choice (modded-nanogpt #90 is the upstream analogue, citable only as
  software).

### Merged q/k/v projection (one GEMM)
- **Idea.** Concatenate W_q, W_k, W_v so the projection is one larger GEMM with better utilization and one read of
  the input.
- **Literature.** None found that states it as a contribution.
- **Upstream.** modded record #20 (`ebdeb37`, `qkv_w` (3, d, d)). nanochat LOG 2026-01-18: a fused QKVO linear had
  "~zero impact".
- **Here.** PARALLEL. `model/ops/linear.py` `qkv_proj`: one (M, 3C) QuACK GEMM against the stacked bf16 `qkv_bf16`,
  separate fp32 parameters, dX one GEMM, dW three `addmm` into `.grad` (`70025d8`: +0.77% / +0.85% at 8K / 32K;
  `docs/architecture.md` §3). The single 3C×C dW variant was rejected (`docs/experiments.md` §5).
- **Cite as.** No paper: a standard implementation choice; report the repository's own measurement.

### Newton–Schulz batched over same-shape matrices
- **Idea.** Muon orthogonalizes every hidden matrix each step. Matrices of one shape can be stacked and iterated as one
  batched GEMM chain with identical math, so a deep model launches a few large kernels instead of many small ones
  (better GPU occupancy; distributed, O(1) communication rounds).
- **Literature.**
  - arXiv 2504.05295 (Ahn et al., 2025, §4.1): "the batch dimension L grouping matrices of identical shape for
    efficient batched computation".
  - arXiv 2608.11612 (Amsel et al., 2026, §7): "megabatching": all matrices of the same shape orthogonalized as a
    single batch. Its stated purpose is cutting communication rounds to O(1), independent of depth; the
    orthogonalization math is unchanged.
  - arXiv 2510.05491 §3.2 and arXiv 2502.16982 §2.3 distribute orthogonalization across devices, a different
    (multi-GPU) concern.
- **Upstream.** modded #15 (size groups, `8011fe7`), #20 ("batched Muon", 3-D batched NS, `ebdeb37`), #32 (PR #125,
  stacked grads, `a96fbbd`); nanochat `6bb9240` (2026-01-15, per-shape stacked groups), `41bb2ea`. No paper is
  cited.
- **Here.** PARALLEL. `train/optim.py` `Muon.step` does a per-shape `torch.stack` and a `baddbmm` chain (`shapes`;
  `a81410c`); `docs/training.md` §3. With FA4 real segments the d24 step gained 5.3% (`docs/architecture.md` §3);
  the unbatched `torch.optim.Muon` measured −4.3% (`docs/experiments.md` §5, [unverified: no record]).
- **Cite as.** arXiv 2504.05295 §4.1 and arXiv 2608.11612 §7 for same-shape batching of the orthogonalization;
  arXiv 2608.11612 motivates it by communication, while here (one GPU) the gain is kernel efficiency [inference],
  measured by this repository's own A/B (`a81410c`), not upstream's. The iteration itself: §2, Newton–Schulz
  iteration.

### torch.compile of the whole model with static shapes and one graph
- **Idea.** Capture the Python program into one graph (TorchDynamo) and generate fused kernels (TorchInductor).
  Compiling for static shapes avoids guards and dynamic-shape reasoning, and custom ops trace through their meta
  ("fake") implementations.
- **Literature.** PyTorch 2, ASPLOS '24 (Ansel et al., 2024, Abstract, §5, §5.1, §6.4 Table 4): 1.41×
  training geometric-mean speedup over eager on an A100 across 180+ models. Static-shape compilers recompile per shape
  (§5). Meta functions propagate shapes, with "a mechanism for defining your own meta functions for custom ops"
  (§5.1). Fusion is the largest single source of speedup (Table 4). Not on arXiv; peer-reviewed.
- **Upstream.** nanochat `3a5e0bc` `torch.compile(model, dynamic=False)`. modded compiles from its first script.
- **Here.** ADOPTED (+ `fullgraph=True`). `train/loop.py:65,144` recompiles at the 32K stage switch. Every kernel is a
  `torch.library` custom op with a fake implementation (`docs/architecture.md` §3). max-autotune +0.05% was not
  adopted (`docs/experiments.md` §5).
- **Cite as.** Ansel et al., ASPLOS 2024 (non-arXiv, archival) for torch.compile, static-shape specialization and
  custom-op meta functions.

### CUDA graphs
- **Idea.** Record a sequence of kernel launches once and replay it, removing per-launch CPU and driver overhead. This
  matters when launches, not GPU work, bound the step.
- **Literature.**
  - PyTorch 2, ASPLOS '24 (Ansel et al., 2024, §4.7, §6.4 Table 4): with `mode="reduce-overhead"` Inductor uses CUDA
    Graphs. Removing them costs 0.08 of the 1.45× training geomean on A100.
  - arXiv 2608.11612 (Amsel et al., 2026, §6 "Kernel launch minimization"): CUDA-graph capture and replay to remove
    launch overhead "on smaller-scale models".
- **Upstream.** None.
- **Here.** NOT TAKEN for training: "CUDA graphs cannot help a GPU-bound step" (`docs/experiments.md` §5, no record).
  ADOPTED in the decoder, where each decode step is captured graphs (`docs/evaluation.md` §6).
- **Cite as.** Ansel et al. §4.7 for the mechanism. The training-side verdict is this repository's own observation.

### Fused and compiled optimizer steps
- **Idea.** Run the elementwise optimizer update for many tensors in one or a few kernels instead of one per
  parameter.
- **Literature.** None found that establishes it as a method; it is a framework feature
  (`torch.optim.AdamW(fused=True)`).
- **Upstream.** nanochat `22a71aa` ("fuse adamw into a single torch compiled kernel … about 1.7X faster, but … not
  making a major dent") and `41bb2ea` (combined MuonAdamW). modded `b7b8313` (`fused=True`).
- **Here.** ADAPTED in part. `train/optim.py` uses `torch.optim.AdamW(..., fused=True)`; Muon runs eagerly with
  `_foreach` ops and batched Newton–Schulz. A fused clip + optimizer pass (~13 ms/step [estimate]) was not pursued
  (`docs/experiments.md` §5).
- **Cite as.** No paper. If needed, cite PyTorch (software) for the fused AdamW implementation.

### Meta-device construction, then per-tensor seeded initialization
- **Idea.** Build the module with shape-only tensors (no storage), so the model can be planned, counted or sharded
  before memory is committed. Then materialize the storage and run the initialization, each tensor drawn from its own
  generator seeded by its name.
- **Literature.** arXiv 2304.11277 (Zhao et al., 2023, §3.1): "deferred initialization". Parameters are allocated on a
  "fake" device, the initialization operations are recorded, and they are replayed when the tensor moves to a GPU.
  This creates a model instance "without allocating any GPU memory blocks". No paper for the per-tensor seeding.
- **Upstream.** nanochat `3a5e0bc` (meta init), `aa530cd`, LOG 2026-01-11 ("Meta Device Footgun"), `f41dd3c` and LOG
  2026-02-05 (`build_model_meta(depth)`).
- **Here.** ADOPTED. `model/modeling.py` `meta_model` and `CausalLM.__init__` ("Shapes only (meta device)"). Values
  are set only by `CausalLM.init_weights` (called from `train/loop.py`; generator seed `seed`·2^32 + crc32(name) per
  tensor), with no replay. The app process plans and counts on the meta device without initializing CUDA
  (`docs/training.md` §11 `run.py`; `docs/architecture.md` §1 counts).
- **Cite as.** arXiv 2304.11277 §3.1 for deferred initialization on a fake/meta device. The variant here is an
  explicit `init_weights` rather than recorded-op replay; describe it as such. The per-tensor seeding is an
  implementation detail (no citation needed).

### Garbage-collector control
- **Idea.** Python's cyclic GC periodically scans every tracked object. After the first step the long-lived objects are
  frozen and automatic collection is disabled, so no scan pauses the training loop.
- **Literature.** None.
- **Upstream.** nanochat `07c4dd4` ("manually control the over-active garbage collector, save a small few minutes")
  and LOG 2026-02-16 ("~500ms pauses … same as base pretraining"; collection every 5,000 steps).
- **Here.** ADAPTED. `train/loop.py:182-186` (`gc.collect`, `gc.freeze`, `gc.disable` after the first step) with
  `gc.collect()` after each checkpoint or weights save (`:130,134`); `docs/training.md` §8 cites `07c4dd4`.
- **Cite as.** No paper: an engineering choice. Cite karpathy/nanochat (software) or nothing; the CPython `gc`
  documentation describes `gc.freeze`.

### Gradient accumulation with per-token loss weighting; dW accumulated in the GEMM
- **Idea.** A step is several micro-batches whose gradients add up. Weighting each micro-batch by its share of valid
  labels makes the step's loss the mean over every valid token, not a mean of micro-batch means. Accumulating dW
  directly into the fp32 gradient buffer (β = 1) avoids a separate add.
- **Literature.** arXiv 2410.02660 (Gao et al., 2024, App. A.3 "Token-averaged loss"): averaging the loss over
  sequences instead of valid tokens "skews the optimization"; they average over all valid tokens. For fusing the
  accumulation into the weight-gradient GEMM no paper was found: it is Megatron-LM's `gradient_accumulation_fusion`
  (`wgrad_gemm_accum_fp32` into `main_grad`), a software feature.
- **Upstream.** nanochat `3a5e0bc` (`loss / grad_accum_steps` per micro-batch).
- **Here.** ADAPTED. `model/ops/cross_entropy.py` `step_shares` (valid-label shares; `docs/training.md` §6). dW goes
  into the fp32 `.grad` by cuBLAS `addmm` (β = 1) in `f499be7` (+0.79% / +0.64% at 8K / 32K;
  `docs/architecture.md` §3, design source NVIDIA/Megatron-LM).
- **Cite as.** arXiv 2410.02660 App. A.3 for token-averaged rather than sequence-averaged loss. Gradient accumulation
  itself needs no citation. The fused accumulation has no paper; cite NVIDIA/Megatron-LM as software.

### Overlapping host work with the device step
- **Idea.** Prepare the next micro-batches and copy values through pinned memory without blocking while the GPU runs
  the current step, so the device never waits on the host.
- **Literature.** None found as a method statement.
- **Upstream.** nanochat `3a5e0bc` (next batch fetched during forward/backward). modded #33 (PR #127, asynchronous
  fetching and indexing of batches).
- **Here.** ADAPTED. The next step's micro-batches are read during the current step (`train/loop.py`). Step values go
  to pinned host buffers without blocking and are logged after the next step is launched (`docs/training.md` §6).
- **Cite as.** No paper: standard engineering.

### CUDA caching-allocator `expandable_segments`
- **Idea.** Let the caching allocator grow segments instead of allocating fixed blocks, reducing fragmentation.
- **Literature.** None (PyTorch documentation only).
- **Upstream.** nanochat `3a5e0bc` (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`), renamed in `2874eda`.
- **Here.** NOT TAKEN (removed at `19d5ba3`); the docs do not discuss it.
- **Cite as.** No paper. Not needed.

### FlashAttention-3 with an SDPA fallback
- **Idea.** Choose a fast kernel on the hardware that supports it and fall back to PyTorch's
  `scaled_dot_product_attention` elsewhere.
- **Literature.** None (portability engineering).
- **Upstream.** nanochat `8203efa` and LOG 2026-01-16.
- **Here.** NOT TAKEN. FA4 only, no fallback (`docs/experiments.md` §5 "FA3 not pursued").
- **Cite as.** No paper. Not needed.

### Model FLOPs utilization (MFU) and the peak
- **Idea.** Report achieved model FLOPs (tokens per second times the analytic count, excluding rematerialization)
  over the hardware's peak, so systems are compared independent of implementation.
- **Literature.** arXiv 2204.02311 (Chowdhery et al., 2022, §4.1, Table 3; App. B): MFU defined against hardware
  FLOPs utilization; "the 'theoretical maximum' throughput only accounts for the required operations … and not
  rematerialization"; PaLM 540B 46.2% MFU with attention (45.7% without), HFU 57.8%.
- **Upstream.** nanochat `3a5e0bc` (`base_train.py` `promised_flops_per_sec_h100 = 989e12`), `2955650`
  (`get_peak_flops`, device detection).
- **Here.** ADAPTED. `metrics.py` `mfu` over the executed per-segment model FLOPs (`executed_model_flops`; the count:
  §2, FLOPs per token) against `PEAK_FLOPS` = 989.4e12 from NVIDIA's H100 SXM datasheet (dense BF16), not nanochat's
  constant. The definition is in `docs/evaluation.md` §5 and the values in `docs/training.md` §7.
- **Cite as.** arXiv 2204.02311 §4.1 and App. B for MFU; the NVIDIA H100 datasheet (non-archival) for the peak.

**Related citations for this repository's own numerics** (no upstream lineage). Non-bitwise backward from atomics:
arXiv 2307.08691 §3.2 ("We use atomic adds … to update dQ"); arXiv 2603.05451 §3.2.4: the backward's inter-CTA
reductions make dQ nondeterministic, and FA4's deterministic mode serializes them with a semaphore at a performance
cost (`docs/architecture.md` §4). Checks against an fp64 reference: arXiv 2407.08608 §4.3 and arXiv 2603.05451
§3.1.3 Table 2 measure kernel error against FP64, the practice of `docs/architecture.md` §4. Software exp2 emulation
on FMA units: arXiv 2603.05451 §3.1.3, the degree-3 polynomial matches MUFU.EX2 within 1 bf16 ULP on 99% of inputs;
tried in the capped loss and rejected (+2.9 to +3.3 ms; `docs/experiments.md` §5).

## 5. Ideas without a paper

Whole ideas (entry in parentheses):
- Untied embedding and head (§1): a design choice; the published evidence is mixed.
- Smear, backout, U-net skip, partial key offset (§1): speedrun techniques; software citation (modded records #11,
  #34, #40, #49).
- Weight-decay schedule (§2): an upstream design choice; λ is constant here.
- Muon momentum warmup and schedule (§2): an upstream heuristic; Sutskever et al. 2013 is the general precedent.
- Per-group AdamW learning rates and width scaling (§2): empirical tuning; software citation.
- Learning rates of the residual and x0 scalars (§2): upstream tuning; this repository's single run is the evidence.
- Schedule as a multiplier of each group's lr (§2): software pattern.
- Record methodology (§2): records are unreviewed; software citation with the record number.
- Rust BPE trainer, tiktoken encoder (§3): engineering; software citation (openai/tiktoken, karpathy/rustbpe).
- Merge tie-break (§3): implementation detail.
- BOS as the only delimiter; special-id layout (§3): convention.
- Chat special tokens (§3), validation split = last shard (§3): not needed.
- BOS as end-of-document target and stop (§3): convention.
- Validation budget of 80 × 2^19 tokens (§3): a design choice; its adequacy is reported as standard errors.
- Merged q/k/v projection (§4): implementation choice; this repository's measurement.
- Fused and compiled optimizer steps (§4): framework feature; software citation (PyTorch).
- Garbage-collector control (§4), overlapping host work (§4): engineering.
- Allocator `expandable_segments` (§4), FlashAttention-3 with an SDPA fallback (§4): not needed.

Parts of cited ideas without a paper:
- RMSNorm's eps value (§1, RMS normalization): a numerical choice.
- A gain-free RMS norm after the embedding (§1, RMS-normed token embedding): upstream's choice.
- RoPE's −θ sign and half-split pairing (§1, RoPE): conventions.
- The norm → gain → RoPE order (§1, QK-norm); the combined norm–gain–RoPE kernel (§4, Fused QK-norm + RoPE): design.
- Attention scale 0.12 (§1, Attention temperature): speedrun tuning, not taken.
- Head dimension 128 (§1): a design choice and the kernels' tile size.
- The SSSL period and the forced-global last layer (§1, Local–global attention): nanochat's design choice.
- The two-scalar x0 form (§1, Per-layer residual scalars): modded record #9, software citation.
- c = 15 (§1, Output-logit soft cap): record #18's tuning; the designed ablation is `docs/experiments.md` §3.
- Uniform rather than normal initialization (§1, Uniform fan-in initialization): upstream's design note.
- The shape rule's rounding to whole heads (§1, Shape rule): nanochat's.
- No weight decay on 1-D parameters (§2, Muon/AdamW split): the nanoGPT convention, contrary to arXiv 2502.16982
  App. D.
- Upstream's √max(1, rows/cols) step size (§2, Muon step size): software, with arXiv 2310.17813 as counterpart
  [inference].
- WSD floor values of 5–15% (§2, Trapezoid / WSD): tuned constants.
- The data-bound tokens-per-parameter ratio (§2, Training horizon): a data-supply choice.
- Counting `lm_head` but not the value tables (§2, Scaling-parameter count): nanochat's convention.
- Attended keys min(window + 1, T) in the FLOP count (§2, FLOPs per token): PaLM's T applied per layer.
- 2^15 as the vocabulary size (§3, Vocabulary): a design choice.
- Digit runs of k = 2 (§3, Digit runs): a design choice.
- The incremental pair-count algorithm (§3, Incremental BPE trainer): software citation (karpathy/rustbpe).
- The tokenizer sample size (§3, Tokenizer training sample): a design choice.
- 0 bytes for specials and excluded zero-byte targets (§3, Bytes per token; Bits per byte): convention.
- The cost of padded `cu_seqlens` (§4, Fixed-size `cu_seqlens`): engineering observation (record #78, a blog post).
- The capped loss in the GEMM epilogue (§4, Fused linear cross entropy): QuACK (software) plus a modification here.
- The norm-as-row-scale fold (§4, Fused GEMM epilogues): this repository's design.
- dW accumulated in the GEMM (§4, Gradient accumulation): NVIDIA/Megatron-LM, software citation.
- Per-tensor seeded initialization (§4, Meta-device construction): implementation detail.
- FP8 not paying at ~200M parameters with K = 768 (§4, FP8 training): this repository's measurement.

## 6. Upstream measurements are not evidence

Each upstream measurement that bears on a choice here, with the published evidence that can replace it. The first six
are the EVIDENCE entries. Upstream measurements behind ideas NOT TAKEN stay in their entries and support nothing.

| Upstream measurement | Bears on | Replacement | Entry |
|---|---|---|---|
| LOG 2026-02-05: SwiGLU worse at d12, d24 | ReLU² over SwiGLU | arXiv 2109.08668 §3 Fig. 5, against arXiv 2002.05202 Table 1 (conflict): needs our own ablation | §1, SwiGLU |
| Records #41, #48; LOG 2026-01-10: NorMuon | NorMuon adopted | arXiv 2510.05491 Table 1, Abstract | §2, NorMuon |
| LOG 2026-01-13: digit runs k = 1, 2, 3 | k = 2 | none measures k = 2: needs our own ablation (or state a design choice) | §3, Digit runs |
| LOG 2026-01-15, 2026-02-17: data-source CORE | text-group source | arXiv 2406.17557 §4, arXiv 2504.13161 §6; at this scale needs our own ablation | §3, Data-source negatives |
| LOG 2026-01-19..22: ~320-run sweep | distrust of small-proxy tuning | arXiv 2509.02046 §1 Fig. 1, with arXiv 2512.05620 | §2, Upstream optimizer sweep |
| Records' n-run t-tests | records cited as support | no paper: cite modded as software, not as evidence | §2, Record methodology |
| LOG 2026-01-13: varlen null at 2K (d16) | document masking used | arXiv 2407.21783 §3.2 (limited impact at standard lengths); arXiv 2402.13991 Table 1 (gain) | §1, Document masking |
| LOG 2026-01-29: learnable RMSNorm gains worse | gain-free norm | arXiv 2501.00656 §3.3.1 (parity) | §1, RMS normalization |
| Record #8: untied embed and head | untied head | none: design choice, or our own ablation | §1, Untied |
| LOG 2026-01-17: head dim 128 over 64 | `HEAD_DIM` = 128 | arXiv 2002.07028 (direction only); the value needs our own ablation or a design-choice statement | §1, Head dimension |
| LOG 2026-01-11: SSSL and window length | windows | arXiv 2503.19786 §5.2 (ratio and window ablations) | §1, Local–global attention |
| LOG 2026-01-11: x0 shortcut sweep d8–d20 | residual scalars | arXiv 2402.02622, arXiv 2410.17897 | §1, Per-layer residual scalars |
| LOG 2026-01-16/17: alternating VE; record #55: gate | value embeddings, gate | arXiv 2601.22887 §4.2 Table 1, §3.2 | §1, Value embeddings; gate |
| Records #9, #18; LOG 2026-03-02: cap value | c = 15 | arXiv 2408.00118 §2 (a cap, at 30); c = 15 needs our own ablation (designed, `docs/experiments.md` §3) | §1, Output-logit soft cap |
| LOG 2026-01-17: aspect 64 over 128 | shape rule | arXiv 2402.14905 §2.2.2, Table 9 | §1, Shape rule |
| LOG 2026-01-08, modded `9805cab`: no gain from clipping | clipping kept (contrary) | arXiv 2401.02954 §2.3 for the value; a gain from clipping needs our own ablation (the d8 A/B is one run each) | §2, Gradient-norm clipping |
| LOG 2026-01-10: Polar Express no difference | not taken | arXiv 2505.16932 (positive) against this repository's screen | §2, Polar Express |
| Records #43, #50; LOG 2026-01-10: cautious decay better | not taken (contrary) | arXiv 2510.12402 (positive) against this repository's one-seed screen | §2, Cautious weight decay |
| nanochat `6ed7d1d`: warmup 40 tuned at d12 | 40 warmup steps | arXiv 1910.04209 §4.3 Eq. 14 | §2, Linear warmup |
| LOG 2026-01-13, record #26: BOS-aligned packing | packing | arXiv 2404.10830 §3.1, Table 2 | §3, BOS-aligned rows |
| LOG 2026-03-04: ClimbMix speedrun gain | text-group source | arXiv 2504.13161 §6 | §3, ClimbMix |
| LOG 2026-03-04: validation budget doubled | `VAL_TOKENS` | this repository's standard errors (arXiv 2411.00640 §2.2) | §3, Validation budget |
| LOG 2026-01-13, 2026-02-02: FP8 | FP8 not taken | arXiv 2209.05433, arXiv 2412.19437 for the option; the verdict is this repository's measurement | §4, FP8 training |
| LOG 2026-01-11: FA3 +9% tok/s | attention kernel | arXiv 2407.08608 (FA3 over FA2); GH200 speeds are this repository's measurements | §4, FlashAttention kernels |
| Record #78: tighter `max_num_docs` | real segments only | no paper: this repository's combined measurement (`a81410c`) | §4, Fixed-size `cu_seqlens` |

## References

| arXiv | First author, year | Title |
|---|---|---|
| 1211.5063 | Pascanu, 2012 | On the difficulty of training Recurrent Neural Networks |
| 1312.6120 | Saxe, 2013 | Exact solutions to the nonlinear dynamics of learning in deep linear neural networks |
| 1404.5997 | Krizhevsky, 2014 | One weird trick for parallelizing convolutional neural networks |
| 1412.6980 | Kingma, 2014 | Adam: A Method for Stochastic Optimization |
| 1508.07909 | Sennrich, 2015 | Neural Machine Translation of Rare Words with Subword Units |
| 1604.06174 | Chen, 2016 | Training Deep Nets with Sublinear Memory Cost |
| 1608.05859 | Press, 2016 | Using the Output Embedding to Improve Language Models |
| 1611.09940 | Bello, 2016 | Neural Combinatorial Optimization with Reinforcement Learning |
| 1701.06538 | Shazeer, 2017 | Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer |
| 1706.02677 | Goyal, 2017 | Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour |
| 1706.03762 | Vaswani, 2017 | Attention Is All You Need |
| 1710.03740 | Micikevicius, 2017 | Mixed Precision Training |
| 1711.00489 | Smith, 2017 | Don't Decay the Learning Rate, Increase the Batch Size |
| 1711.05101 | Loshchilov, 2017 | Decoupled Weight Decay Regularization |
| 1809.10853 | Baevski, 2018 | Adaptive Input Representations for Neural Language Modeling |
| 1901.09321 | Zhang, 2019 | Fixup Initialization: Residual Learning Without Normalization |
| 1904.09751 | Holtzman, 2019 | The Curious Case of Neural Text Degeneration |
| 1904.10509 | Child, 2019 | Generating Long Sequences with Sparse Transformers |
| 1905.12322 | Kalamkar, 2019 | A Study of BFLOAT16 for Deep Learning Training |
| 1908.03265 | Liu, 2019 | On the Variance of the Adaptive Learning Rate and Beyond |
| 1910.04209 | Ma, 2019 | On the adequacy of untuned warmup for adaptive optimization |
| 1910.07467 | Zhang, 2019 | Root Mean Square Layer Normalization |
| 1911.02150 | Shazeer, 2019 | Fast Transformer Decoding: One Write-Head is All You Need |
| 2001.08361 | Kaplan, 2020 | Scaling Laws for Neural Language Models |
| 2002.04745 | Xiong, 2020 | On Layer Normalization in the Transformer Architecture |
| 2002.05202 | Shazeer, 2020 | GLU Variants Improve Transformer |
| 2002.07028 | Bhojanapalli, 2020 | Low-Rank Bottleneck in Multi-head Attention Models |
| 2002.10444 | De, 2020 | Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks |
| 2003.04887 | Bachlechner, 2020 | ReZero is All You Need: Fast Convergence at Large Depth |
| 2004.05150 | Beltagy, 2020 | Longformer: The Long-Document Transformer |
| 2005.14165 | Brown, 2020 | Language Models are Few-Shot Learners |
| 2010.04245 | Henry, 2020 | Query-Key Normalization for Transformers |
| 2010.12821 | Chung, 2020 | Rethinking embedding coupling in pre-trained language models |
| 2101.00027 | Gao, 2020 | The Pile: An 800GB Dataset of Diverse Text for Language Modeling |
| 2104.04473 | Narayanan, 2021 | Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM |
| 2104.09864 | Su, 2021 | RoFormer: Enhanced Transformer with Rotary Position Embedding |
| 2107.02027 | Krell, 2021 | Efficient Sequence Packing without Cross-contamination: Accelerating Large Language Models without Impacting Performance |
| 2107.03374 | Chen, 2021 | Evaluating Large Language Models Trained on Code |
| 2109.08668 | So, 2021 | Primer: Searching for Efficient Transformers for Language Modeling |
| 2110.02861 | Dettmers, 2021 | 8-bit Optimizers via Block-wise Quantization |
| 2203.03466 | Yang, 2022 | Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer |
| 2203.15556 | Hoffmann, 2022 | Training Compute-Optimal Large Language Models |
| 2204.02311 | Chowdhery, 2022 | PaLM: Scaling Language Modeling with Pathways |
| 2205.05198 | Korthikanti, 2022 | Reducing Activation Recomputation in Large Transformer Models |
| 2205.10287 | Malladi, 2022 | On the SDEs and Scaling Rules for Adaptive Gradient Algorithms |
| 2205.14135 | Dao, 2022 | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness |
| 2209.05433 | Micikevicius, 2022 | FP8 Formats for Deep Learning |
| 2211.05100 | BigScience Workshop, 2022 | BLOOM: A 176B-Parameter Open-Access Multilingual Language Model |
| 2302.05442 | Dehghani, 2023 | Scaling Vision Transformers to 22 Billion Parameters |
| 2302.13971 | Touvron, 2023 | LLaMA: Open and Efficient Foundation Language Models |
| 2304.11277 | Zhao, 2023 | PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel |
| 2305.13245 | Ainslie, 2023 | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints |
| 2307.08691 | Dao, 2023 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning |
| 2309.14322 | Wortsman, 2023 | Small-scale proxies for large-scale Transformer training instabilities |
| 2309.16039 | Xiong, 2023 | Effective Long-Context Scaling of Foundation Models |
| 2310.06825 | Jiang, 2023 | Mistral 7B |
| 2310.17813 | Yang, 2023 | A Spectral Condition for Feature Learning |
| 2310.18313 | Peng, 2023 | FP8-LM: Training FP8 Large Language Models |
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
| 2404.06395 | Hu, 2024 | MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies |
| 2404.10830 | Ding, 2024 | Fewer Truncations Improve Language Modeling |
| 2404.19737 | Gloeckle, 2024 | Better & Faster Large Language Models via Multi-token Prediction |
| 2405.05417 | Land, 2024 | Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models |
| 2405.09818 | Chameleon Team, 2024 | Chameleon: Mixed-Modal Early-Fusion Foundation Models |
| 2405.13698 | Wang, 2024 | How to set AdamW's weight decay as you scale model and dataset size |
| 2405.14591 | Men, 2024 | Base of RoPE Bounds Context Length |
| 2405.18392 | Hägele, 2024 | Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations |
| 2406.11794 | Li, 2024 | DataComp-LM: In search of the next generation of training sets for language models |
| 2406.12907 | Pearce, 2024 | Reconciling Kaplan and Chinchilla Scaling Laws |
| 2406.17557 | Penedo, 2024 | The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale |
| 2406.19146 | Porian, 2024 | Resolving Discrepancies in Compute-Optimal Scaling of Language Models |
| 2407.08608 | Shah, 2024 | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision |
| 2407.13623 | Tao, 2024 | Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies |
| 2407.21783 | Grattafiori, 2024 | The Llama 3 Herd of Models |
| 2408.00118 | Gemma Team, 2024 | Gemma 2: Improving Open Language Models at a Practical Size |
| 2409.20325 | Bernstein, 2024 | Old Optimizer, New Norm: An Anthology |
| 2410.01131 | Loshchilov, 2024 | nGPT: Normalized Transformer with Representation Learning on the Hypersphere |
| 2410.02660 | Gao, 2024 | How to Train Long-Context Language Models (Effectively) |
| 2410.06205 | Barbero, 2024 | Round and Round We Go! What makes Rotary Positional Encodings useful? |
| 2410.10989 | Hsu, 2024 | Liger Kernel: Efficient Triton Kernels for LLM Training |
| 2410.17897 | Zhou, 2024 | Value Residual Learning |
| 2411.00640 | Miller, 2024 | Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations |
| 2411.09009 | Wijmans, 2024 | Cut Your Losses in Large-Vocabulary Language Models |
| 2411.16085 | Liang, 2024 | Cautious Optimizers: Improving Training with One Line of Code |
| 2412.05496 | Dong, 2024 | Flex Attention: A Programming Model for Generating Optimized Attention Kernels |
| 2412.19437 | DeepSeek-AI, 2024 | DeepSeek-V3 Technical Report |
| 2501.00656 | Team OLMo, 2024 | 2 OLMo 2 Furious |
| 2502.05795 | Sun, 2025 | The Curse of Depth in Large Language Models |
| 2502.15938 | Bergsma, 2025 | Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs |
| 2502.16982 | Liu, 2025 | Muon is Scalable for LLM Training |
| 2503.10622 | Zhu, 2025 | Transformers without Normalization |
| 2503.19786 | Gemma Team, 2025 | Gemma 3 Technical Report |
| 2504.05295 | Ahn, 2025 | Dion: Distributed Orthonormalized Updates |
| 2504.13161 | Diao, 2025 | Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training |
| 2505.06708 | Qiu, 2025 | Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free |
| 2505.09388 | Yang, 2025 | Qwen3 Technical Report |
| 2505.13738 | Bergsma, 2025 | Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training |
| 2505.16932 | Amsel, 2025 | The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm |
| 2509.02046 | Wen, 2025 | Fantastic Pretraining Optimizers and Where to Find Them |
| 2510.05491 | Li, 2025 | NorMuon: Making Muon more efficient and scalable |
| 2510.12402 | Chen, 2025 | Cautious Weight Decay |
| 2512.05620 | Qiu, 2025 | Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales |
| 2512.22382 | Mlodozeniec, 2025 | Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration |
| 2601.07372 | Cheng, 2026 | Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models |
| 2601.22887 | Li, 2026 | MoVE: Mixture of Value Embeddings -- A New Axis for Scaling Parametric Memory in Autoregressive Models |
| 2603.05451 | Zadouri, 2026 | FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling |
| 2603.15031 | Kimi Team, 2026 | Attention Residuals |
| 2604.14053 | Chizhov, 2026 | From Where Words Come: Efficient Regularization of Code Tokenizers Through Source Attribution |
| 2608.11612 | Amsel, 2026 | Dion3: Full-Stack Orthogonal Updates |
| 2608.18062 | Meister, 2026 | TokEval: A Tokenizer Evaluation Suite |

Non-arXiv sources:
- Ansel et al., 2024. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph
  Compilation. ASPLOS '24, DOI 10.1145/3620665.3640366.
- Björck & Bowie, 1971. An iterative algorithm for computing the best estimate of an orthogonal matrix. SIAM J. Numer.
  Anal. 8:358–364 [unverified: cited through arXiv 2409.20325].
- Gage, 1994. A New Algorithm for Data Compression. C Users J. 12(2):23–38 [unverified: cited through arXiv
  1508.07909].
- Glorot & Bengio, 2010. Understanding the difficulty of training deep feedforward neural networks. AISTATS.
- Higham, 2008. Functions of Matrices. SIAM [unverified: cited through arXiv 2409.20325, arXiv 2505.16932].
- Kovarik, 1970. Some iterative methods for improving orthonormality. SIAM J. Numer. Anal. [unverified: cited
  through arXiv 2409.20325].
- LeCun, Bottou, Orr & Müller, 1998. Efficient BackProp. Neural Networks: Tricks of the Trade, Springer.
- Nesterov, 1983 [unverified: cited through Sutskever et al. 2013].
- Sutskever, Martens, Dahl & Hinton, 2013. On the importance of initialization and momentum in deep learning. ICML,
  PMLR 28.
- Non-archival: Jordan, Jin, Boza, You, Cesista, Newhouse & Bernstein, 2024, "Muon: An optimizer for hidden layers in
  neural networks", https://kellerjordan.github.io/posts/muon/ (blog); Radford et al., 2019, "Language Models are
  Unsupervised Multitask Learners" (OpenAI report); Meister, 2026, "Tokenizer Code Ablations" (TokEval companion
  blog post); McCormick, 2026, "Optimizing Training with FlashAttention varlen" (blog post); NVIDIA H100 Tensor Core
  GPU datasheet (peak dense BF16 FLOP/s).
- Software and data: karpathy/nanochat (`92d63d4`, `dev/LOG.md`); KellerJordan/modded-nanogpt (records by number);
  karpathy/rustbpe; openai/tiktoken; PyTorch (`torch.optim.Muon`, fused AdamW, `torch.compile`); zichongli5/NorMuon;
  QuACK; FlashAttention-4 (`flash-attn-4`); NVIDIA/Megatron-LM; Hugging Face `transformers` (`modeling_qwen3.py`);
  datasets `karpathy/climbmix-400b-shuffle` and `nvidia/Nemotron-ClimbMix`.
