# Pre-training a code base model on one GH200

A Python and C++ code-generation base model pre-trained on one NVIDIA Hopper GPU (sm_90: the GH200's, or an H100 80 GB
SXM5) at the most tokens its data holds without any stage reading a document twice. The code is the specification;
this document states each decision, its basis (a cited result, a measurement, or the law it is derived from) and the
file that defines it. Two companion documents own their parts, and this one links to them:

- `docs/corpus.md`: the corpus (sources, curation, mixture, FIM, packing, the tokenizer's data).
- `docs/tokenizer-ablation.md`: the tokenizer ablation (arms, measures, intrinsic results, the runs' decision rule).

Numbers come from the code, from `results/`, or are derived from them [arithmetic]. Measurements whose records were not
kept (the pre-launch measurements of 2026-10-19 to 2026-10-22, mostly on an H100 80GB HBM3 SXM5 at its 700 W cap) are
marked [unverified]. Terminology follows DeepSeek-Coder: the base model, its pre-training, an ablation's baseline.

## Model

`model/gpt.py`: karpathy/nanochat 92d63d4 plus the wins of its experiment log ("LOG", `dev/LOG.md`), with
Qwen3's QK norm and learnable gains. Each layer mixes x0 into the stream and looks up its value embeddings
(`GPT.hidden`); its `Block` then applies `Attention` and `MLP` to the RMS-normed stream, each added back.

| | d24 (the base model) |
|---|---|
| Shape | 24 layers × 768 (`gpt_config`: width 30 × depth rounded up to whole 128-dim heads; MobileLLM-350M's 32 × 960, arXiv 2402.14905 Table 9), 6 heads × 128, MHA |
| Attention | per-head RMS norm → learned per-dim gain → RoPE over the full head, base 10^6 (`model/config.py` `ROPE_BASE`); SSSL windows (S = 512), last layer full; FlashAttention-4 varlen with document masking |
| MLP | ReLU², hidden 4C |
| Residual | `x = resid_lambda·x + x0_lambda·x0` per layer; value embeddings on alternating layers (12 of 24) behind a per-head `2·sigmoid` gate of the normed stream, zero init (gate 1): modded-nanogpt record #14's per-layer tables, MoVE's LaVE (arXiv 2601.22887 §4.1 Eq. 6) |
| Head | untied `lm_head`, zero init (modded-nanogpt record #8), plain cross entropy, no z-loss; an output-logit softcap only as the D5 ablation's arm (`GPTConfig.softcap`) |
| Vocabulary | 2^15 ids (`model/config.py` `VOCAB_SIZE`; BOS and 3 FIM sentinels; `docs/tokenizer-ablation.md`) |
| Parameters | 195,096,576 scaling parameters (blocks + `lm_head`, `scaling_params`), what the token budget counts; 522,252,336 in all with the token table and 12 value-embedding tables (25,165,824 each) |
| FLOPs per token | 2.831 GFLOP non-embedding: DeepSeek LLM's M (arXiv 2401.02954 §3.2 Eq. 2), 72·n_layer·d² + 12·n_layer·d·l_seq at the full 8K context (`non_embedding_flops_per_token`) |
| Init | stream-reading matrices and tables uniform with std C^-½; `c_o`, `c_proj` (modded-nanogpt record #5), `lm_head`, value-embedding gates zero; resid lambdas and gains 1, x0 lambdas 0 |
| Precision | fp32 parameters, gradients and optimizer state; bf16 activations and GEMM inputs |
| Lineage | From nanochat 92d63d4: the block layout, value-embedding layers, MLP and window pattern. Not taken: its smear, backout and logit softcap, QK order (RoPE → norm → ×1.2), short window (a quarter of the context), RoPE base (10^5) and init. The gate is nanochat e85db6b's (2·sigmoid, zero init) read over the full width rather than its first 32 channels; the full-width read is not ablated |

### Basis

- **Value embeddings, kept** (decided from the literature): MoVE's LaVE gains 0.016, 0.010 and 0.010 at D12, D20
  and D32 (arXiv 2601.22887 Table 1), ≈ −0.014 extrapolated to d24 in MoVE's coordinates [inference], against 4.7% of
  the d24 step [unverified]. No reference code model has them.
- **Learnable QK gains, kept** (D4): fixing the normalized q/k scale costs 0.11% of loss (nGPT, arXiv 2410.01131
  App. A.9 Table 5); per-head QK norm with learnable parameters in arXiv 2309.14322 §3.2.5.
- **Published, not tested here** (no reference code model ablates such details):
  - *ReLU² rather than SwiGLU:* nanochat measured SwiGLU worse at d12 and d24 under Muon (LOG 2026-02-05); a gated
    MLP would also need a gated decode MLP (`model/kernels/decode.cu`).
  - *SSSL windows:* Gemma 3 (arXiv 2503.19786 §5.2) measures local:global up to 7:1 and windows down to 512 nearly
    lossless in perplexity; StarCoder2 trains code under a 4,096-token window.
  - *Shape:* MobileLLM's deep-and-thin shape at d24's scaling parameters (35 × 640) takes 9.2% more FLOPs per token
    [unverified], so an equal-token comparison would give it more compute.
  - *head_dim 128:* compiled into the decoder's kernels (`model/config.py` `HEAD_DIM`); nanochat measured 128 better
    than 64 (LOG 2026-01-17).
  - *MHA, not grouped-query attention:* GQA's benefit is KV-cache memory; d24's cache for the decoder's 256 slots
    (`model/kernels/decode.cu` `MAX_BATCH`) of the longest benchmark request (391 prompt + 768 new tokens) is 21.9 GB
    [arithmetic: 2 · 24 · 768 · 2 bytes · 1,158 · 256], which the GPU holds.
  - *RoPE base 10^6 from the first step* (D9): above arXiv 2405.14591's lower bound for a 32K context (Table 2:
    6.4e5), large from the start as Llama 3 (arXiv 2407.21783 §3.2); Barbero et al. (arXiv 2410.06205 Table 2) find
    10k against 500k at 8K within 0.3%, of mixed sign.
- **Not adopted:** Block Attention Residuals (a published 1.1% loss gain at 194M parameters, arXiv 2603.15031; +12% d24
  step time [unverified]), the partial key offset (modded-nanogpt record #49), either of which would need a native
  decode path; multi-token prediction, reported to hurt models of this size on code benchmarks (arXiv 2404.19737).

### Against the reference code models

| Choice | The base model | DeepSeek-Coder / Qwen2.5-Coder / StarCoder2 / Code Llama |
|---|---|---|
| Attention | SSSL: 3 of 4 layers a 512-token window | full attention (DeepSeek-Coder, Qwen2.5-Coder, Code Llama); StarCoder2 a 4,096-token window in a 16,384 context |
| MLP | ReLU², hidden 4C | SwiGLU (DeepSeek-Coder, Qwen2.5-Coder, Code Llama); StarCoder2 GELU |
| Value embeddings | alternating layers, gated | none |
| Optimizer | Muon on matrices, AdamW on tables and vectors | AdamW, β = (0.9, 0.95) (DeepSeek-Coder §3.4, Code Llama) |
| Peak lr | DeepSeek LLM's law × `ablations/peak_lr.py`'s multiplier, the peak of linear decay to zero | the law's lr held for 80% of tokens, then 31.6% and 10% of it (DeepSeek LLM §2.3, the schedule DeepSeek-Coder §3.4 follows) |
| Weight decay | τ law (Regime) | 0.1 (DeepSeek LLM §2.3) |
| Gradient clipping | 1.0 | 1.0 (DeepSeek LLM §2.3) |
| Context stages | 8K file level, then the last 5.45% of steps at 32K on repositories | Qwen2.5-Coder: the same 8K → 32K repository stage; DeepSeek-Coder: 16K repository-ordered files; Code Llama: a 16K long-context stage |
| Data | code:text:math 87:10:3 | DeepSeek-Coder 87% code, 10% code-related English, 3% Chinese; Qwen2.5-Coder code:text:math 70:20:10 |
| FIM | rate 0.5 per piece of the repository corpora's train documents, PSM/SPM 50/50, AST-FIM middles (`pack.FIM_RATE`, `SPM_SHARE`; `docs/corpus.md` §6) | DeepSeek-Coder 0.5, PSM only (arXiv 2401.14196 §3.1.2); StarCoder2 ≈25% of chunks; Code Llama 90% |

What those models ablate: Code Llama the RoPE base of its long-context stage (Appendix G.4, Table 18) and FIM's cost
to left-to-right completion; DeepSeek-Coder the FIM rate (Fig. 3); StarCoder2 RoPE against learned positions; OpenCoder
(arXiv 2411.04905) its data. None ablates architecture details such as gains, scalars or attention windows.

## Data and stages

The corpus, its mixture and its two stages are specified in `docs/corpus.md`; the code is `corpus/`
(`mixture.py` the stages and shares, `pack.py` the rows and FIM, `loader.py` the micro-batches). In short:
code:text:math 87:10:3 in both stages; an 8,192-token file stage, then a 32,768-token repository stage over the last
`mixture.REPO_SHARE` (5.45%) of steps (Qwen2.5-Coder, arXiv 2409.12186 §3.2); 102.6B tokens in all (`mixture.STEPS` ×
`STEP_TOKENS`), what the curated corpus holds without any stage reading a document twice. A run's ratio defaults to its
data's `train/stages.py` `max_ratio`. The tokenizer is decided by `docs/tokenizer-ablation.md`.

## Regime

| Setting | Value | Source |
|---|---|---|
| Batch | 1,474,560 tokens (15 micro-batches of 98,304) at 102.6B tokens | DeepSeek LLM's law B = 0.2920·C^0.3271, C = M·D (`train/schedule.py` `plan`), rounded to whole micro-batches |
| Steps | 69,580 (525.89 tokens per scaling parameter); repository stage from step 65,785 | `schedule.plan` (D11: `round(ratio · params)`) |
| Peak lr | the law's η = 0.3118·C^-0.125 = 8.6295e-4 times `ablations/peak_lr.py`'s multiplier: × 2.7972 = 2.4139e-3 at 102.6B tokens | DeepSeek LLM §3.1 Eq. 1; `conf/experiment/pretrain.yaml`'s `lr_scale` resolves `peak_lr.py`'s `analysis` at the corpus's `max_ratio` (Peak lr) |
| Warmup | 40 steps, linear | 2/(1 − β₂) (arXiv 1910.04209, for Adam); nanochat 92d63d4's 40 steps at d24 |
| Decay | linear to zero; the last step at 1/(69,580 − 40) = 1.44e-5 of peak (D10: exactly 0 is not required) | arXiv 2502.15938 §4 |
| Optimizers | AdamW (fused) on tables, `lm_head` and vectors; Muon (`torch.optim.Muon`'s update, Newton–Schulz batched by shape) with `match_rms_adamw` on the other matrices, one lr for every group | Moonlight, arXiv 2502.16982 §2.2 (`train/optim.py`) |
| AdamW | β = (0.9, 0.95) | DeepSeek LLM §2.3 |
| Weight decay | λ = 1/(η τ T), the EMA timescale τ = 1.084·TPP^−0.527 of the T steps (λ = 0.1492 at the base model's peak lr), on tables, `lm_head` and Muon matrices, 0 on vectors | Power Lines, arXiv 2505.13738 App. E.3.1 (`optim.tau_weight_decay`) |
| Gradient clipping | global norm 1.0 on the accumulated gradient, before both optimizers; the norm is logged every step | DeepSeek LLM §2.3 (D3; `optim.MAX_NORM`, `update`) |
| Repository stage | continues the decay: 5.46% of peak at its first step, 0 at the last, over 3,795 steps | D2: decay to zero (arXiv 2502.15938); DeepSeek-Coder §3.6 and DeepSeek-V3 instead hold their final lr |

Batch, steps, lr and weight decay at 102.6B tokens are computed from the code (`schedule.plan` at d24); the run's own
values come from `max_ratio` on the adopted tokenizer's packs.

**Gradient clipping** at 1.0 (D3). Adam's update is invariant to the gradient's scale and Muon's Newton–Schulz divides
its input by its norm, so under a slowly varying scale the clip damps only spikes. The failure Megatron-LM PR 5395
describes for a pre-Muon clip needs gradient norms far above any recorded here; DeepSpeed issue 8439 clips after
orthogonalization, which this code does not.

**Warmup.** 40 steps, the value nanochat 92d63d4 uses at d24 and 2/(1 − β₂) for AdamW's β₂ (arXiv 1910.04209); the
lr ablation's 11 runs (`results/peak_lr.jsonl`, up to × 8 at d8) show no lr-dependent warmup instability. No paper fixes a
warmup for Muon.

### Peak lr

DeepSeek LLM's law was fit for AdamW at compute below the base model's (§3.1), and the transfer rules disagree on how
the optimum moves with the horizon: from d8 at 5 to 200 tokens per parameter, Complete^(d)P's SDE rule
(arXiv 2512.22382 Table 1) predicts the optimum's multiplier of the law changes × 0.354, Step-Law (arXiv 2503.04715,
fit on 1,912 samples from 7 dense model sizes) × 4.92 (`results/peak_lr.json`). `ablations/peak_lr.py` therefore measures the
multiplier: × 1, × 2, × 4 at d24 over 5 tokens per parameter, times the optimum's horizon trend measured at d8 (× 1
to × 8 over 5 and over 200), net of d8's batch growth by the square-root batch rule (arXiv 2205.10287 Def. 5.2), its
exponent carried to the base model's horizon, and the base model's batch growth by the same rule. Each setting's
optimum is the vertex of the 3-point local parabola in ln(multiplier) through its lowest-code run and that run's two
neighbours (`lr.fit`; arXiv 2409.19913 §3.1 fits quadratics in log lr). The horizon trend is assumed width-independent
[inference: arXiv 2409.19913 §4 finds its exponent size-independent only from 760M parameters].

Code bits per byte at × 1 / 2 / 4 / 8 of the law's lr, seed 0 (`results/peak_lr.jsonl`, dataset revision `1e711aa`, the
removed data pipeline's data):

| Setting | Plan | × 1 | × 2 | × 4 | × 8 | Vertex (`lr.json`) |
|---|---|---|---|---|---|---|
| d8 at 5 | 746 steps of 98,304 tokens | 0.89491 | 0.78969 | 0.76960 | 0.81791 | × 3.4669 |
| d8 at 200 | 14,937 steps of 196,608 | 0.52774 | 0.51750 | 0.51592 | 0.52431 | × 3.1565 |
| d24 at 5 | 3,307 steps of 294,912 | 0.44230 | 0.43787 | 0.44201 | | × 2.0235 |

No run diverged. Each setting's best arm beats the law by 0.1253, 0.0118 and 0.0044 bits per byte [arithmetic]. The
measured d8 ratio is × 0.9105 (horizon exponent −0.0254, −0.1193 net of the batch): relative to the law the optimum
barely moves with the horizon, against both rules. At d8 at 200, d8's law batch is 2 micro-batches (B_opt 2.04 under
Eq. 2's M; it rounds to 2 from 77.7 tokens per parameter on [arithmetic, `schedule.plan`]).

The vertices are multipliers of the law's lr the runs recorded, which predates D1 (DeepSeek LLM Eq. 2's M); the runs'
batches and steps are unchanged under D1, so `peak_lr.analysis` re-bases each vertex onto the current law's lr. Under D1 the
base model's multiplier is × 2.7972 at 102.6B tokens (fixed in `conf/experiment/logit_softcap.yaml`). The pretrain
experiment computes its own when it launches (`run.py` `lr_multiplier`, from `results/peak_lr.jsonl` at the adopted tokenizer's
plan), so no copy is stored. A d8 at 20 comparison of × 2 against × 1 (seeds 0 and 1, 2026-10-20) has no record in
`results/` [unrecorded].

## Ablations

One pre-registered experiment per decision whose outcome the literature does not settle. Every run is a
`run.py` `RunConfig` (Hydra, hydra-core 1.3.7: `RunConfig`'s defaults are the base model's recipe,
`conf/experiment/<name>.yaml` sweeps fields against them), trained by `train/loop.py` and evaluated by
`eval/suite.py evaluate`, the base model's code paths. No experiment runs without the user's approval.

| Experiment (`conf/experiment/`) | Decision | Arms | Analysis | State |
|---|---|---|---|---|
| `peak_lr` (`ablations/peak_lr.py`, 4ade225's records) | the base model's peak lr | Peak lr | `peak_lr.analysis` | done, `results/peak_lr.jsonl`; the multiplier computed at launch |
| `tokenizer_split_merge.yaml` | split pattern and merge score | 4 arms at d16, each at its `max_ratio`, 2 seeds | `ablations.tokenizer.decide` | intrinsic part and data done; runs wait (`docs/tokenizer-ablation.md`) |
| `logit_softcap.yaml` | D5 output-logit softcap | cap 15 vs none, d16 at 20, 4 seeds, lr × 2.7972 | `ablations.recipe.logit_softcap` | waits |
| `muon_wd_timescale_20tpp.yaml`, `muon_wd_timescale_200tpp.yaml` | D7 τ law on Muon | Muon's τ × ½, × 1, × 2, d8 at 20 and at 200, 2 seeds each | `ablations.recipe.muon_wd_timescale`, `muon_wd_timescale_law` | waits; the 200 half needs the corpus's packs |
| `adamw_group_lr.yaml` | D6 AdamW groups' lr | × ¼, × 1, × 4 at d16 on the sa-bpe arm's data, 2 seeds; × 1 is the tokenizer ablation's sa-bpe runs | `ablations.recipe.adamw_group_lr` | waits for the tokenizer runs |
| `pretrain.yaml` | the base model's pre-training | d24 at `max_ratio`, seed 0, every benchmark | none | waits for every ablation above |

Each recipe ablation's rule (its contrasts, margins and adoption condition) is in `ablations/recipe.py`'s docstring;
the tokenizer ablation's in `docs/tokenizer-ablation.md`.

### Method

- **Arms.** An arm differs from the experiment's baseline only in its yaml's fields. Its plan (batch, steps, peak lr)
  is the base model's at its depth and ratio (`run.base_plan`, `plan_of` at its `lr_scale`), so arms match in
  hyperparameters and, unless an arm sets its own ratio (the tokenizer ablation's equal bytes), in tokens; every run
  completes its own schedule (warmup, then linear decay to zero over its length). Weight decay follows each run's lr
  by the τ law, so an lr sweep varies η at the fixed EMA timescale (arXiv 2405.13698; Power Lines). A run that
  diverges (a non-finite loss or gradient norm) is recorded with its step and excluded from the comparisons
  (arXiv 2309.14322 §3.1); any other failure stops the sweep.
- **Settings.** No affordable run reaches the base model (d24 at 525.9 tokens per scaling parameter). For data
  recipes, a single proxy 7× below the target picks the target's better of two in about 80% of pairs (DataDecide,
  arXiv 2504.11393 Abstract, §3.1), and a continuous correct-answer likelihood makes code benchmarks predictable
  where accuracy is not (its §3.4, Fig. 6). Architecture rankings can change with scale (arXiv 2207.10551).
- **Measures** (`metrics.measures`, bits per byte unless stated): *code*, 8K validation of the code
  group at its corpus weights, the decision measure; *solutions*, the HumanEval+, MBPP+ and HumanEval-X C++ reference
  solutions given their prompts, pooled; *repo_32k* and *repo_beyond_8k*, the 32K repository rows and their targets
  beyond 8,192; *text*, *math*; *unparsed*, *indentation*, *syntax*, the shares of HumanEval+ and MBPP+ problems whose
  healed greedy completion does not compile (Parse measure, Evaluation); *greedy*, their base pass@1; *tok_per_s*.
- **Power.** σ = 0.0034 bits per byte on 8K code (`ablations/contrast.py` `CODE_SIGMA`): the pooled seed standard
  deviation of six d8 runs at 20 tokens per parameter, two configurations × data seeds 0–2, on the removed pipeline's
  data (records not in `results/` [unverified]), carried to d16 and to the corpus [inference]. Two runs against two
  resolve 1.96·σ·√(2/2) = 0.0067; four against four 0.0047 [arithmetic].
- **Cost** [unverified, H100 SXM5 runs of 2026-10-20 to 2026-10-22: whole runs with compilation and loss evaluation]:
  d8 at 20 about 4 minutes, d16 at 20 about 35 minutes, d24 at 5 about 65 minutes.

### Records

A run's key (`run.key`) is its `RunConfig` fields that differ from the defaults, sorted, `k=v` joined by `,`
(e.g. `depth=16,seed=1,tokenizer=sa-bpe`; `defaults` if none), so a run two experiments share is one run. Per run, in
`artifacts/runs/<key>/`: `.hydra/` (the composed config), `metrics.jsonl` (`metrics.py`: every step and attempt;
the run record holds the model config, the `RunConfig` as `config`, the tokenizer's `data_key`, parameters, FLOPs per
token, the clip threshold, the plan, lr and weight decay per group; the end record the peak memory; a failure record
the step a run diverged at), `step<N>.pt`, `eval-step<N>.json` (`eval/suite.py evaluate`) and `run.log`. A rerun
resumes: a run with a final evaluation or a divergence is skipped, a stopped one resumes from its newest checkpoint,
and a run whose recorded plan or data key differs stops the sweep ("recipe changed") until its directory is deleted.
Training and each checkpoint's evaluation run in fresh spawned processes. When the sweep ends, `ablations/records.py`'s
`Records` callback writes `results/<experiment>.jsonl` (a record per run: its evaluation verbatim, `config`, measures,
efficiency), `results/<experiment>-curves.jsonl` (per run its per-step train records as columns) and, with an analysis,
`results/<experiment>.json` = {"analysis": ...}, so tables and figures need no rerun. `results/peak_lr.jsonl`,
`peak_lr-curves.jsonl` and `peak_lr.json` were written by 4ade225's `ablations/harness.py` in its earlier format (hashed tags, the
run's fields as `fields`).

## GPU implementation

Ops in `model/ops/`, CUDA sources in `model/kernels/` (built on first use by `native.cuda_library`). Each is a
`torch.library` op, so `torch.compile(fullgraph=True)` traces one graph per stage.

| Op | Implementation |
|---|---|
| Projection, MLP, LM-head GEMMs | forward and dX: QuACK SM90 GEMMs autotuned by QuACK (`quack.gemm_interface`, `tuned=True`); fp32 weight gradients: cuBLAS `torch.mm` |
| ReLU² MLP | QuACK `gemm_act` / `gemm_dact` |
| Cross entropy | QuACK, bf16 logits read once, the logit gradient written over them; the softcap arm's capped loss fused in the same op (`model/ops/cross_entropy.py`) |
| Attention | FlashAttention-4 varlen over the micro-batch's real document segments only (`model/ops/flash_attn.py`), its default backward |
| q/k norm + gain + RoPE | `model/kernels/qk_prologue.cu`: one pass per direction; gain gradient without atomics (row math shared with the decoder, `model/kernels/qk_row.cuh`) |
| Value-embedding gate and mix | `model/kernels/ve_gate.cu` (`model/ops/ve_gate.py`): one launch forward, one pass backward, the gate's logits in fp32 |
| Muon | Newton–Schulz batched over same-shape matrices (`optim.Muon`) |
| Embeddings | `F.embedding`, bf16 cast fused by Inductor; backward adds rows with atomics |
| Decoding | `model/kernels/decode.cu` (`inference/decoder.py`): batched native decoder for the code benchmarks |

Measured choices [unverified: `torch.profiler` and CUDA-graph timings, records not kept]:
- **cuBLAS for the fp32 weight gradients:** dYᵀX has a small output (768 × 768 or 768 × 3,072) and K = 98,304; QuACK's
  split-K-1 tiles left most SMs idle (c_q/c_k/c_v/c_o dW 0.544 against 0.191 ms, relative error vs fp64 2.3e-4 against
  1.8e-5).
- **`model/kernels/qk_prologue.cu`** rather than Inductor's fusion of the PyTorch ops: the d24 step 3,792 against
  3,890 ms; and micro-batches of 98,304 tokens rather than 65,536 (1% faster per token).
- **FlashAttention-4** rather than cuDNN 9.26: forward + backward 1.47 against 4.96 ms at window 512, 1.83 against 5.85
  ms at 8,192 (d8 shapes); its full-context layers use the causal kernel (`window = None`), 1–2% faster than the local
  kernel at window T, outputs bitwise equal. FA4 on the real segments rather than the loader's padded `cu_seqlens`:
  losses bitwise, gradients within run-to-run spread.
- **Split q/k/v projections** rather than one GEMM on concatenated weights: fusion saved no measurable GEMM time and
  its layout needs copies.
- **Ours over QuACK's chunked linear cross entropy** for the LM-head loss (29.2 against ≥ 31.4 ms at d8); per-position
  evaluation losses by QuACK's `lse_target_epi` (no logits).
- **The value-embedding gate kernel** is not bitwise to the PyTorch expressions it replaced (its fp32 reduction order
  moves about 15% of elements by one bf16 ulp).
- **Rejected:** FP8 MLP GEMMs (e4m3 rowwise, d16 on real rows: mean loss over steps 250–299 +0.0048 over bf16 against
  a bf16 two-run spread of 0.0014, growing); two concurrent runs (−8% aggregate; one run saturates the GPU); Liger and
  Cut Cross-Entropy losses, fused RMSNorm/RoPE libraries, DCP checkpointing.

**The d24 step** [unverified]: on the H100 SXM5, 12 micro-batches, 3,741 ms (GEMM 53%, FA4 25%, Inductor 14%; peak
69.6 GiB); on the GH200 with the SM clock locked at 1,980 MHz and items above applied, about 3,490 ms for the same 12
micro-batches. The GEMMs run at the 700 W power cap's ceiling (703 of 989 dense bf16 TFLOP/s).

## Reproducibility

- **Fastest kernels.** FA4's and the embedding backward reduce with atomics, so repeated steps differ in the last bits;
  fixed-order reductions would cost about 1% of a step [unverified]. Python's garbage collector stays off after setup
  and collects only after a checkpoint save (`train/loop.py`). The forward is deterministic: Inductor's
  coordinate-descent tuning is off (`scripts/container.sh`), since it picks kernel configurations by timing, so two
  processes would compile different kernels.
- **Resume.** A relaunched run continues from its newest checkpoint (weights, optimizer state, data position), written
  to a temporary file and renamed, so a kill mid-save leaves the previous one.
- **Environment.** `docker/Dockerfile` pins base images by digest and git dependencies by commit; `uv.lock` the Python
  packages; fp32 matmuls stay fp32 (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`).
- **Run identity.** The run's key and its recorded configuration (Records); a run recorded under a different plan or
  data key stops its experiment.
- **Data order.** Each corpus's packed rows are permuted once by the run's seed (NumPy `RandomState`; `train/stages.py`
  `stage_batches`), which also seeds its initialization; rows go to corpora by the loader's Weyl sequence
  (`mixture.mixture_rows`).

## Reproducing the run

The pipeline splits at the tokenizer. Everything through curation is text and runs on the GB10; the curated shards go
to the private Hub dataset `iaroy/nanochat-data` (`corpus/publish.py` `DATASET`). The tokenizer ablation and decision,
compile, pack and training run on the GH200 from a pull of that dataset. Every command runs in the `nanochat` image
(`docker build -f docker/Dockerfile -t nanochat .`) from the repository's root: through `scripts/container.sh` on the
GH200, or on the GB10 as

```
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e PYTHONPATH=/workspace/nanochat -w /workspace/nanochat \
  -v $PWD:/workspace/nanochat nanochat python -m corpus.<step> ...
```

(steps that read the Hub add `-e HF_HOME=/tmp/hf -e HF_HUB_DISABLE_XET=1 -e HF_TOKEN=$(hf auth token | tail -1)`).

On the GB10, once (the dataset is constant afterwards):

1. **Prepare**: `python -m corpus.prepare` fetches and standardizes every source (`docs/corpus.md` §1);
   fetched files are kept, so a rerun downloads only what it lacks; a source read in part fetches its extent
   (`prepare.extent`).
2. **Curate**: `python -m corpus.curate`: the Python 3 check with repair, exact deduplication, the split, the
   curated shards (`CORPUS/curated/<source>/<split>/<k>.parquet`; `docs/corpus.md` §4, its invariants by construction).
3. **Publish**: `python -m corpus.publish push` writes the card, replaces the dataset's tree and squashes its
   history into one commit (the card turns the Hub's viewer off).

On a fresh Lambda `gpu_1x_gh200` (Ubuntu with Docker and the NVIDIA driver; the H100 SXM5 runs the same image):

```bash
git clone https://github.com/indrajeetadityaroy9/nanochat nanochat && cd nanochat
HF_TOKEN=<read token> scripts/setup.sh   # CDI spec, clock lock, docker group, image, artifacts/, the curated corpus
```

then, after logging in again (docker group), each through `scripts/container.sh`:

1. **Tokenizer ablation, intrinsic part and data** (`docs/tokenizer-ablation.md`): `python -m ablations.tokenizer
   vocabularies`, `intrinsic`, `data`. An arm's extent margin (`results/tokenizer_split_merge-intrinsic.json`) below 1 means a group
   read in part is too small for it: raise `mixture.SPREAD`, then prepare, curate and publish again on the GB10.
2. **Recipe ablations and tokenizer decision** (the user's approval): `python run.py experiment=<name>` for
   `muon_wd_timescale_20tpp`, `logit_softcap`, `muon_wd_timescale_200tpp`, `tokenizer_split_merge`, `adamw_group_lr` (records in `results/<name>.*`; runs in
   `artifacts/runs/<key>/`).
3. **Adopt the tokenizer**: `python -m ablations.tokenizer adopt` writes the decided vocabulary to `CORPUS/tokenizer`
   (over any provisional one); every later step reads it.
4. **Compile**: `python -m corpus.compile`, ending with the 32K stage's repository partitions
   (`compile.compile_stage`).
5. **Pack**: `python -m corpus.pack`; it stops naming any source that holds fewer rows than its draw.
6. **Smoke**: the first micro-batches of both stages (`loader.PretrainingBatches` at the stages' weights), decoded;
   each step's group composition and FIM share.
7. **Train** (the user's approval): `tmux new -d -s run 'scripts/container.sh python run.py experiment=pretrain'`.

Everything a run reads or writes lives in the checkout (`paths.py`): `artifacts/` (gitignored, created by
`scripts/setup.sh`) holds `corpus/`, `runs/<key>/` and `cache/`; `results/` (tracked) holds the data the paper's tables
and figures are drawn from.

`python run.py experiment=pretrain` (`conf/experiment/pretrain.yaml`) trains both stages with a checkpoint (the run's
full state) at each tenth of the steps and at the repository stage's start, evaluates every checkpoint with every
benchmark, and writes `results/pretrain.jsonl` and `results/pretrain-curves.jsonl`. **Expected time** [inference from the
unverified GH200 step]: 69,580 steps of 15 micro-batches at about 1.25 × 3.49 s, about 84 h, plus evaluation.

`scripts/release.sh <key> <repo>` uploads the run's checkpoints, records and `results/` to a Hugging Face model
repository; `scripts/container.sh hf download <repo> --local-dir /workspace/nanochat` restores them where `load_model`
and a resumed run read them.

### Hardware

The GH200 (one H100 GPU with 96 GB HBM3, 64 Neoverse-V2 cores, 525 GB of memory) is the host of the ablations and
the base model's pre-training; the H100 SXM5 (80 GB) is the measured fallback. What the code depends on:

- *Power.* The GPU's enforced limit is 700 W, set outside the VM (`nvidia-smi -pl 900` is accepted but not enforced),
  and the GH200 module (Grace + Hopper + memory) shares a 1,000 W budget with the CPU [unverified], so no heavy CPU job
  runs beside a measured run. `scripts/setup.sh` locks the SM clock at 1,980 MHz (speed only, not numerics).
- *Memory.* `corpus/mixture.py` `MICRO_BATCH_TOKENS` (98,304: 12 rows at 8K, 3 at 32K) is the most whole rows of both
  contexts whose d24 step fits 80 GB (peak 69.6 GiB [unverified]); on 96 GB it stays, as a larger micro-batch would
  change every run's batch rounding (`schedule.plan`).
- *Kernels.* Both GPUs are sm_90; QuACK's autotuner picks GEMM tiles by timing, per host, so steps need not be bitwise
  across hosts [inference].
- *ARM64.* `docker/Dockerfile` pins multi-arch index digests; the data order does not depend on the CPU [inference].
  `char` is unsigned on ARM64 and signed on x86, so a MultiPL-E C++ program that depends on it can score differently
  [inference].
- *Evaluation is where the CPU counts:* its pools are sized by `os.cpu_count()`.

## Pre-launch verification

**The model against its specification** [unverified: 2026-10-20 at `7faf11f`, before D1 and the kernels above; records
not kept]. Each op and the whole model against fp64 restatements written from the Model table, forward and backward,
on the GB10 (sm_121) and an H100 (SM90); relative error:

| Check | GB10 | H100 |
|---|---|---|
| GEMMs and ReLU² (fp32 weight gradients) | 3.7e-3 (4.4e-6) | 3.7e-3 (4.4e-6) |
| q/k prologue (gain gradients) | 1.7e-3 (1.5e-7) | 1.7e-3 (2.0e-7, bitwise repeatable) |
| FA4 output and gradients | 3.6e-3 | 3.6e-3 |
| FA4 window boundary: worst row, against either off-by-one | 2.7e-3 against 1.0–2.3 | 2.7e-3 against 1.06–2.20 |
| FA4 without the document mask | 4.1 | 4.2 |
| Cross entropy | 1.7e-3 | 1.7e-3 |
| Whole model (4 × 256, every component perturbed to matter): per-position loss | 5.0e-3 | 5.2e-3 mean, 2.5e-2 max |
| Restatements each breaking one rule (no x0, σ gate, no value embeddings, RoPE +θ, no document mask) | 5.1e-2–1.5e-1 | 4.4e-2–1.5e-1 |
| Gradients | 1.7e-2 | 2.1e-2 |
| Native decoder: greedy tokens the forward's argmax, the rest within | 499 of 512, 0.014 of its maximum logit | 502 of 512, 0.015 |

## Evaluation

Every metric is a published one, computed as its source computes it. `eval/suite.py evaluate` scores a checkpoint:
the base model's checkpoints take everything (`generate="benchmarks"`), the ablation runs the likelihood metrics and
the parse measure (`generate="parse"`).

**Likelihood metrics** (`eval/bpb.py`, by `metrics.bpb`), in bits per byte (The Pile, arXiv 2101.00027: summed bits
over summed UTF-8 bytes, tokenizer-independent):

- *Held-out corpora:* each training corpus's val split at 8K and at 32K, per context bucket (≤ 8K, > 8K: Code Llama's
  perplexity by context length, arXiv 2308.12950 §3.3, Fig. 4a) and per byte-offset bucket (doublings from 1 KiB to
  128 KiB of the piece before a target, so tokenizers compare at the same bytes of context, arXiv 2503.13423 §3.1,
  App. B.1), pooled by the stage's mixture weights: the AR loss of arXiv 2207.14255 §2. Targets of no bytes (BOS,
  padding, sentinels) are not scored, as karpathy/nanochat's `evaluate_bpb`. A corpus's sample is its rows up to
  80 × 2^19 tokens' worth (`bpb.VAL_TOKENS`), systematic.
- *Reference solutions:* the bits of HumanEval+'s, MBPP+'s and HumanEval-X C++'s reference solutions given their
  prompts: the task loss of arXiv 2412.04403 §2.2; a continuous correct-answer likelihood makes HumanEval and MBPP
  decisions predictable where accuracy is trivial (DataDecide, arXiv 2504.11393 §3.4, Fig. 6); BPB raises the
  signal-to-noise ratio of most benchmarks (arXiv 2508.13144 §5.3); the loss on a benchmark's problems and reference
  solutions tracks pass@1 where held-out loss does not (arXiv 2604.07769 §3.3.1). EvalPlus's solutions begin with blank
  lines, which a tokenizer joins to the prompt's last line break, so those blank lines count as prompt.

**Parse measure** (`code_eval.parse_generation`): one greedy, unconstrained completion of each HumanEval+ and MBPP+
prompt with BOS and the prompt healed at its boundary, as arXiv 2608.18062 generates (§4.3; App. C: the trailing
prompt token dropped and generation constrained until its bytes are re-emitted); the share whose prompt + completion
compiles, fails with an IndentationError (TabError included) or with another SyntaxError (the companion post's
classification; a parser overflow counts as a SyntaxError), and the completions' EvalPlus pass@1. It departs from
TokEval's protocol in its prompts (EvalPlus's 0-shot base prompts; TokEval: lm-eval-harness, MBPP 3-shot), its budget
and stops (EvalPlus's 768 tokens and stop strings; the post: 256 tokens), and compiles the raw completion.

**Benchmarks** (`eval/code_eval.py`): EvalPlus (arXiv 2305.01210) HumanEval+ and MBPP+ (base and plus pass@1),
MultiPL-E (arXiv 2208.08227) HumanEval and MBPP in C++, and HumanEval single-line, multi-line and random-span
infilling (arXiv 2207.14255; PSM and SPM), each scored by its official evaluator, by up to three generation modes:

| Mode | Decoding | Score |
|---|---|---|
| `greedy` | left-to-right argmax, one completion per problem | pass@1 as DeepSeek-Coder (arXiv 2401.14196 §4.1), StarCoder2 (arXiv 2402.19173 §7.1.1) and EvalPlus's leaderboard report it |
| `sampled` (EvalPlus, MultiPL-E) | 20 samples per problem at temperature 0.2 from the nucleus (arXiv 1904.09751) top-p 0.95 (`code_eval.SAMPLES`, `SAMPLE_TEMPERATURE`, `TOP_P`) | pass@1 by Codex's unbiased estimator (arXiv 2107.03374 §2.1, §3.2–3.3), as MultiPL-E reports it |
| `filtered` | left-to-right prompts healed (arXiv 2608.18062 App. C; token healing, Guidance 2023), Python and FIM middles under the syntax constraint; the greedy candidate, then seeded samples until one has no post-generation diagnostic and passes the prompt's own examples, at most 10 (`verify.ATTEMPTS`); the accepted candidate, else the greedy one | filtered pass@1 (arXiv 2107.03374 §3.5; AlphaCode, arXiv 2203.07814 §4.5): a system measurement, not comparable to pass@1 tables |

- **Comparability.** The sampled mode draws 20 samples per problem where Codex and MultiPL-E drew 200 and the BigCode
  leaderboard 50: the same estimate, with more variance per problem. HumanEval infilling is greedy, not arXiv
  2207.14255's sampling. The prompts' examples overlap the scored base tests, so the filtered mode favours base over
  plus pass@1. EvalPlus's verdicts are timed, so heavy CPU load can fail a slow correct solution.
- **Infilling** (human-eval-infilling): each middle is cut to the reference middle's line count, a single-line middle
  to its first line (arXiv 2505.18789 Table 1, Fig. 4); random-span middles end at BOS, uncut. Both orders are scored
  greedy; the filtered mode runs SPM.
- **Contamination.** The training corpora were not decontaminated against the benchmarks: a benchmark problem in the
  training data inflates its score.

**Decoder** (`inference/decoder.py`, `model/kernels/decode.cu`): batched continuous decoding of up to 256 slots, each
step one captured CUDA graph. A slot runs its prompt through the same kernels as its generation, so prompts and
generations share steps; a prompt's K and V are kept in a paged prompt store (pages of 64 positions) and a later
request with the same prompt starts at its last token. Every sum's order depends on the shapes and the slot's
position alone, so a request's tokens do not depend on the batch, the store or timing. A `Request` carries the
prompt, budget, stop rule,
`temperature` (0 = argmax, ties to the lowest id), `top_p` (the nucleus, arXiv 1904.09751 §3.1, as Hugging Face's
`TopPLogitsWarper` keeps it), `seed` (the sample at position *i* depends only on seed and *i*) and an optional
constraint (`heal()`, the Python syntax constraint, `AllOf()` of both), whose masks each step takes for all
constrained slots in one batched call.

**Constrained decoding: syntax, not types.** Type-constrained decoding (arXiv 2504.09246) implements TypeScript only;
arXiv 2606.21619 found an incomplete constrainer lowers functional correctness, while a complete syntax constrainer
gained (its §5.3). Python 3 syntax can be constrained completely, and every training Python file parses as Python 3
(`curate.repaired`, `docs/corpus.md` §4.2), so the model's distribution lies inside the constrainer's language.
Types are checked after generation instead.

**`PythonSyntax`** (`inference/syntax_constraint.py`, the Rust crate `inference/syntax/`):
`PythonSyntax(tokenizer).constraint(prefix, *, suffix="")`; `masks(constraints)` computes a batch of constraints' next
masks in parallel (rayon) without the GIL. Token *t* is allowed iff `prefix + generated + t` can still become a valid
Python 3.12 module, the text ending anywhere; BOS iff `ast.parse(prefix + generated + suffix)` succeeds; FIM sentinels
never; a prefix that cannot become valid Python 3.12 raises `ValueError`. Layers: CPython 3.12.3's tokenizer
(`Parser/tokenizer.c`) as a character-level state machine (`lexer.rs`); INDENT/DEDENT and the tab-consistency check;
an LALR(1) grammar (`python.lark`) built by Lark at load and read by the crate (`parser.rs`), the soft keywords read
both ways until one fails; per lexer state, the vocabulary's events over its byte trie (`vocab.rs`). Where it is looser
than CPython it accepts rather than rejects. Checks [unverified]: 0 false rejections on 1,051,013 validation tokens of
three Python corpora and on CPython 3.12.3's grammar tests and 167 standard-library files; all 881 grammar-breaking
token substitutions rejected; no empty mask on 3,000 random walks. Rejected alternatives: SynCode (one-regex f-strings
reject valid 3.12 code), llguidance (no INDENT/DEDENT, issue #107), arXiv 2402.17988 (no released library). C++ is not
constrained; it is checked after generation by `g++ -std=c++17 -fsyntax-only`, the MultiPL-E judge's compiler.

**Post-generation checks** (`eval/verify.py`): Python — the prompt plus completion parses, the EvalPlus entry function
exists and returns a value, and a static type checker (basedpyright, batched) limited to definite failures; then the
prompt's own examples (docstring `>>>` lines, MBPP+ `assert` lines) run by value in a subprocess with a timeout.
C++ — `g++ -fsyntax-only`.

**Not used:** speculative decoding (lossless by construction; its gains target large-model serving); self-correction
prompting (a base model without instruction tuning).

## Post-training

Not implemented: the repository pre-trains and evaluates a base model. For HumanEval-style Python and C++ problem
solving, supervised fine-tuning on function-completion exercises after pre-training is the step with the largest
published effect at this scale:

- *Exercises:* fine-tuning on under 180M tokens of function-completion exercises raised a 1.3B model's HumanEval
  pass@1 from 29% to 50.6% (arXiv 2306.11644 §2.1, §2.2, Fig. 2.1).
- *Instruction tuning:* OpenCoder-1.5B from base to two-stage SFT, HumanEval+ 49.4 → 67.7 and MBPP+ 58.7 → 61.9
  (arXiv 2411.04905 Tables 6 and 7; not a controlled comparison).

**Format: completion, not chat.** An example is a prompt in the benchmarks' form, then its solution and BOS, the loss on
the solution's tokens and BOS alone; the tokenizer has no chat tokens (`corpus/tokenizer.py` `SPECIAL_TOKENS`).
Candidate data: `OpenCoder-LLM/opc-sft-stage2` (`educational_instruct`, `package_instruct`) and
`OpenCoder-LLM/opc-annealing-corpus` (`synthetic_code_snippet`, `algorithmic_corpus`). Before any trains the model:
decontamination against every scored benchmark by entry point and 10-gram overlap (as arXiv 2411.04905 §4.4); FIM
examples in `pack.py`'s format in the mix; the fine-tuned model reported beside the final base checkpoint.

## Layout

- `run.py`, `conf/`: the run app (Ablations).
- `metrics.py`: the run log (`metrics.jsonl`) and every metric: bits per byte, parse outcomes, a run's compared
  measures; torch-free, imported by training, evaluation and the ablations.
- `paths.py`: the checkout's layout (`artifacts/`, `results/`). `native.py`: builds and loads the CUDA kernels and the
  Rust crates.
- `model/`: `gpt.py` (the model, scaling parameters and FLOPs per token), `config.py` (vocabulary, head width, RoPE
  base), `ops/` and `kernels/` (the ops above and the decoder).
- `train/`: `loop.py` (setup, resume, the steps, checkpoints and records), `optim.py` (optimizers, weight
  decay, gradient clip), `schedule.py` (plan, each step's stage, lr multiplier), `stages.py` (`max_ratio`, each stage's
  micro-batches in the run's order), `checkpoint.py`.
- `eval/`: the checkpoint's evaluation (`suite.py`), the likelihood sums (`bpb.py`), the code benchmarks
  (`code_eval.py`) and the post-generation checks (`verify.py`).
- `inference/`: the native decoder (`decoder.py`), the Python 3.12 syntax constraint
  (`syntax_constraint.py`, its Rust crate and grammar in `syntax/`).
- `corpus/`: the corpus pipeline, one module per step (`__init__.py` lists them; `mixture.py` also holds the
  micro-batch and repository-stage share, `publish.py` the dataset, `train_tokenizer.py` the tokenizer's sample and
  trainer); `crate/`, its Rust crate (`native.py`): the SA-BPE trainer (and frequency BPE), repository samples and
  the packer.
- `ablations/`: `records.py` (an experiment's records in `results/`), `contrast.py` (the pre-registered tests every
  ablation decides by), `peak_lr.py`, `recipe.py` (the logit softcap, AdamW group lr and Muon weight-decay timescale
  analyses), `tokenizer.py` (the tokenizer ablation, `docs/tokenizer-ablation.md`).
- `docs/`: `pretraining.md` (this document), `corpus.md` (the corpus's specification), `tokenizer-ablation.md`.
- `scripts/`: `setup.sh`, `container.sh`, `release.sh`. `docker/Dockerfile`: the image. `results/`: the records.

## References

Papers this document cites, by arXiv identifier (`arxiv.org/abs/<id>`); the corpus and tokenizer documents cite
theirs inline by identifier and section.

| arXiv | Title |
|---|---|
| 1904.09751 | The Curious Case of Neural Text Degeneration |
| 1910.04209 | On the adequacy of untuned warmup for adaptive optimization |
| 2101.00027 | The Pile: An 800GB Dataset of Diverse Text for Language Modeling |
| 2107.03374 | Evaluating Large Language Models Trained on Code |
| 2203.07814 | Competition-Level Code Generation with AlphaCode |
| 2205.10287 | On the SDEs and Scaling Rules for Adaptive Gradient Algorithms |
| 2207.10551 | Scaling Laws vs Model Architectures: How does Inductive Bias Influence Scaling? |
| 2207.14255 | Efficient Training of Language Models to Fill in the Middle |
| 2208.08227 | MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation |
| 2305.01210 | Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation |
| 2306.11644 | Textbooks Are All You Need |
| 2308.12950 | Code Llama: Open Foundation Models for Code |
| 2309.14322 | Small-scale proxies for large-scale Transformer training instabilities |
| 2401.02954 | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism |
| 2401.14196 | DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence |
| 2402.14905 | MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases |
| 2402.17988 | Constrained Decoding for Fill-in-the-Middle Code Language Models via Efficient Left and Right Quotienting of Context-Sensitive Grammars |
| 2402.19173 | StarCoder 2 and The Stack v2: The Next Generation |
| 2404.19737 | Better & Faster Large Language Models via Multi-token Prediction |
| 2405.13698 | How to set AdamW's weight decay as you scale model and dataset size |
| 2405.14591 | Base of RoPE Bounds Context Length |
| 2407.21783 | The Llama 3 Herd of Models |
| 2409.12186 | Qwen2.5-Coder Technical Report |
| 2409.19913 | Scaling Optimal LR Across Token Horizons |
| 2410.01131 | nGPT: Normalized Transformer with Representation Learning on the Hypersphere |
| 2410.06205 | Round and Round We Go! What makes Rotary Positional Encodings useful? |
| 2411.04905 | OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models |
| 2412.04403 | Establishing Task Scaling Laws via Compute-Efficient Model Ladders |
| 2502.15938 | Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs |
| 2502.16982 | Muon is Scalable for LLM Training |
| 2503.04715 | Predictable Scale: Part I, Step Law -- Optimal Hyperparameter Scaling Law in Large Language Model Pretraining |
| 2503.13423 | SuperBPE: Space Travel for Language Models |
| 2503.19786 | Gemma 3 Technical Report |
| 2504.09246 | Type-Constrained Code Generation with Language Models |
| 2504.11393 | DataDecide: How to Predict Best Pretraining Data with Small Experiments |
| 2505.13738 | Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training |
| 2505.18789 | From Output to Evaluation: Does Raw Instruction-Tuned Code LLMs Output Suffice for Fill-in-the-Middle Code Generation? |
| 2508.13144 | Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation |
| 2512.22382 | Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration |
| 2601.22887 | MoVE: Mixture of Value Embeddings -- A New Axis for Scaling Parametric Memory in Autoregressive Models |
| 2603.15031 | Attention Residuals |
| 2604.07769 | An Empirical Study on Influence-Based Pretraining Data Selection for Code Large Language Models |
| 2606.21619 | The Alignment Problem in Constrained Code Generation |
| 2608.18062 | TokEval: A Tokenizer Evaluation Suite |

Repositories, at the commit, file or record cited: `karpathy/nanochat` (92d63d4, e85db6b, `dev/LOG.md`);
`KellerJordan/modded-nanogpt` (records #5, #8, #14, #49); `Dao-AILab/flash-attention` (FlashAttention-4,
`flash_attn.cute`), `Dao-AILab/quack`; `evalplus/evalplus`, `nuprl/MultiPL-E`, `openai/human-eval-infilling`;
`huggingface/transformers` (`modeling_qwen3.py`, `TopPLogitsWarper`), `python/cpython` 3.12.3 (`Parser/tokenizer.c`);
`guidance-ai/llguidance` (issue #107); `NVIDIA/Megatron-LM` (PR 5395), `deepspeedai/DeepSpeed` (issue 8439);
`OpenCoder-LLM/opc-sft-stage2`, `OpenCoder-LLM/opc-annealing-corpus` (Hugging Face datasets). Token healing: "The Art
of Prompt Design: Prompt Boundaries and Token Healing" (Guidance, 2023).
