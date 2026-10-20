# Ablations

Everything measured to choose this repository's architecture, training regime and H100 implementation, with the code
that measures it. Two kinds:

- **Architecture and regime** (§1): pre-registered experiments, one file per decision that `docs/h100-pretraining.md`
  records without a study of its own and whose result would change production: the peak lr and value embeddings, on
  one shared mechanism (`harness.py`, `model.py`) whose runs train and evaluate through the production code. Not yet
  run.
- **H100 implementation** (§2): kernel and throughput measurements that chose each op (torch.profiler over training steps), and the
  parity records of the code paths (§3).

H100 80GB HBM3 SXM5 (700 W cap), image `nanochat` (NGC 26.09), bf16, compiled as `base_train`, unless stated.

## 1. Architecture and regime (pre-registered)

### Method

- **Arms.** An experiment is production and one arm per alternative at each of its settings, each on the same seeds;
  an arm differs from production only in the fields of `harness.Run`. Every run goes through production's code:
  `scripts/base_train.train` (the file-level then the repository stage, the production data order by seed, τ-law
  weight decay) and `train/metrics.loss_evaluation` (the same held-out rows for every run); its model is
  `model.Candidate`, which is production's GPT unless `value_embeds` differs. Its plan (batch, steps, peak lr) is the
  production model's at the setting (`harness.plan_of`), so arms match in tokens and hyperparameters, and every run
  completes its own schedule (warmup, then linear decay to zero over its length): a longer run cut short would be
  compared before its decay, where loss curves cross and rankings flip (Wen et al., arXiv 2509.02046; Hägele et al.,
  arXiv 2405.18392). An lr arm's weight decay follows its lr by the τ law, λ = 1/(η τ T), as production's would: the
  sweep varies η at the fixed EMA timescale τ that sets the decay's effect (Wang & Aitchison, arXiv 2405.13698; Power
  Lines, arXiv 2505.13738). A run that diverges (a non-finite loss or gradient norm; its failure record) is an outcome
  of a sweep's high end (Wortsman et al., arXiv 2309.14322): it is recorded with its step and excluded from the fits
  and comparisons, and the experiment continues; any other failure stops it.
- **Settings.** No affordable run reaches production (d24 at 490.8 tokens per scaling parameter). `lr.py` measures the
  optimum on production's model at a short horizon, d24 at 5 (3,307 steps of 294,912 tokens, the longest d24 run a
  sweep affords), and the optimum's horizon trend at d8 between 5 (746 steps of 98,304 tokens; B_opt is 0.46
  micro-batches there, so the plan's floor of one) and 200 (14,937 steps of 196,608): 200 is the shortest horizon at
  which d8's law batch has grown to 2 micro-batches (B_opt 1.52, against 2.04 at 490.8; both round to 2), so the batch
  correction is the same as at 490.8 at less than half the cost (Analysis); one seed per setting, the cheap settings
  first: at d8 at 20 the arms × 1 and × 2 differed by 0.0179 and 0.0187 bits per byte on
  seeds 0 and 1 (Records). `value_embeds.py` measures its decision's dependence on training length at d8 and on width at a
  short length and combines them as separable (log-linear in each): d16 (67.1M scaling parameters, 1/2.9 of d24) at
  Chinchilla's 20, 4 seeds, the decision's setting; d8 at 20, 4 seeds, and at production's ratio, one seed, for the
  gain's trends in width and length: it shrinks with scale (−0.016 at d12, −0.010 at d20, MoVE), and d8's tables are
  2.3× its scaling parameters against d24's 1.55×. Architecture rankings can change with scale (Tay et al., arXiv
  2207.10551); for data recipes a single proxy 7× below the target picks the target's better of two in about 80% of
  pairs, best with likelihood metrics such as bits per byte (DataDecide, arXiv 2504.11393).
- **Order.** `lr.py` first; its multiplier is production's peak lr (`scripts/production.py` `LR_SCALE`, hashed into
  production's tag). `value_embeds.py`'s arms train at the law's lr, which `lr.py` puts 2.0 to 3.5 times below each
  measured optimum: its baseline is under-tuned unless its settings take their own optima (under-tuned baselines
  inflate gains: Kaddour et al., arXiv 2307.06440; Wen et al.). A run's tag hashes its peak lr, so runs at a superseded
  lr are never reused.
- **Measures** (`harness.evaluation`, bits per byte unless stated). *code*: 8K validation of the code group at
  production's corpus weights, the decision measure. *solutions*: the HumanEval+, MBPP+ and HumanEval-X C++ reference
  solutions given their prompts, pooled (the benchmarks' pass@1 is near zero at this size). *repo_32k*: 32K
  repository rows of the code group; *repo_beyond_8k*: the same rows' positions past 8,192, the context only the
  repository stage trains. *text*, *math*; tokens per second (median over training steps). Efficiency (from the run's
  run and end records): parameters, non-embedding FLOPs per token, peak memory.
- **Analysis** (from `results/<experiment>.jsonl` and `.json`), pre-registered. *lr* (computed by `lr.py` into
  `results/lr.json`'s `analysis`, every number from the runs' records and plans): per setting, while its lowest-code
  run is at an end of the grid (a diverged run counting as worse than any evaluated one) the grid gains the next factor
  of 2 in that direction and the experiment runs again (`lr.py` repeats this until every grid brackets its optimum);
  then the vertex u* of the parabola in u = ln(multiplier) through the lowest-code run and its two neighbours (positive
  curvature by construction), or that run's own u when the neighbour above it diverged. Amended 2026-10-20, after d8 at
  5's grid (× 1, 2, 4: code 0.89491, 0.78969, 0.76960) put its lowest run at the grid's top with one quadratic over
  all runs placing the vertex inside (× 3.33): that quadratic's vertex is pulled by the far arms of an asymmetric curve
  (planted below: an optimum × 3.0 four times steeper below than above, read × 3.95 by one quadratic over × 1–8, × 3.72
  by the bracketing parabola). The measured d8 ratio r = exp(u*_d8@200 −
  u*_d8@5) is the optimum's change over a token ratio m_D,d8 = 40.0, assumed width-independent (Bjorck et al.'s
  size-independent horizon exponent, arXiv 2409.19913), and the test of the transfer rules, which disagree on its
  direction: Complete^(d)P's AdamW SDE rule, η ∝ √(m_B / m_D) (Mlodozeniec et al., arXiv 2512.22382, Table 1), over the
  law's own fall predicts √(2 / 40.0) / 0.631 = 0.354 (d8's batch goes from 1 to 2 micro-batches, the law's lr falls to
  0.631 of itself); Step-Law (Li et al., arXiv 2503.04715, η = 1.79 N^−0.713 D^0.307) predicts 40.0^0.307 / 0.631 =
  4.92; both are reported against the measurement, with the horizon exponent ln r / ln m_D,d8. Production's multiplier
  of the law's lr is exp(u*_d24@5) · r^(ln m_D,prod / ln m_D,d8) · √(m_B,prod / m_B,d8): the exponent carried 1.24×
  further in log-horizon, to production's m_D,prod = 98.2 (every rule in play is a power law in D: the SDE rule,
  Bjorck et al., Step-Law, the law), and production's batch growth over its range, 4× (294,912 to 1,179,648), against
  d8's 2×, carried by the square-root batch rule (Malladi et al., arXiv 2205.10287; Complete^(d)P's √m_B), so √(4 / 2).
  m_B and m_D come from the runs' recorded plans and production's (`harness.plan_of` at `max_ratio` of d24). Weight
  decay follows each run's lr by the τ law (Power Lines'
  fit for single-epoch pretraining), not Complete^(d)P's λ ∝ √(m_B / m_D). *Value embeddings*: per
  setting, the alternative's mean code minus production's, the two-sided pooled t-test's p and the difference's 95%
  confidence interval (scipy `ttest_ind`; pooled with equal arm sizes is robust to unequal variances); their gain g is
  code without them minus code with them. Production's gain is predicted as
  g(d16, 20) · g(d8, 490.8) / g(d8, 20) · (g(d16, 20) / g(d8, 20))^0.70 (0.70 = ln(195.1 / 67.1) / ln(67.1 / 14.7),
  log-linear in scaling parameters), or g(d16, 20) alone if g(d8, 20)'s interval includes zero. They are removed iff
  that prediction is below the compute their removal frees, 0.05 · L · ln r (L production's code, r the alternative's
  tokens per second over production's at d16; Kaplan et al., arXiv 2001.08361 §1.2: loss ∝ C^−0.050); otherwise they
  stay, their gain and its interval reported. Unpaired although arms share seeds: the six-run comparison's per-pair
  differences (sd 0.0065) show no gain from pairing over independent runs (√2 σ = 0.0048).
- **Equal tokens.** The production run's length is set by its data (no document read twice), not by time.

### Reference code models

Where production departs from the code models it is measured against, and what each departure is worth deciding:

| Choice | Production | DeepSeek-Coder / Qwen2.5-Coder / StarCoder2 / Code Llama |
|---|---|---|
| Attention | SSSL: 3 of 4 layers a 512-token window | full attention (DeepSeek-Coder, Qwen2.5-Coder, Code Llama); StarCoder2 a 4,096-token window in a 16,384 context |
| MLP | ReLU², hidden 4C | SwiGLU (DeepSeek-Coder, Qwen2.5-Coder, Code Llama); StarCoder2 GELU |
| Value embeddings | alternating layers, gated (nanochat) | none |
| Optimizer | Muon on matrices, AdamW on tables and vectors | AdamW, β = (0.9, 0.95) (DeepSeek-Coder §3.4, Code Llama) |
| Peak lr | DeepSeek LLM's law × 2.5467 (`lr.py`), as the peak of linear decay to zero | the law's lr held for 80% of tokens, then 31.6% and 10% of it (DeepSeek LLM §2.3, the schedule DeepSeek-Coder §3.4 follows) |
| Weight decay | τ law (d8: 0.55, d24: 0.1246 at its peak lr) | 0.1 (DeepSeek LLM §2.3) |
| Gradient clipping | 1.0 | 1.0 (DeepSeek LLM §2.3) |
| Context stages | 8K file level, then the last 5.45% of steps at 32K on repositories | Qwen2.5-Coder: the same 8K → 32K repository stage; DeepSeek-Coder: 16K repository-ordered files; Code Llama: a 16K long-context stage |
| Data | code:text:math 87:10:3 | DeepSeek-Coder 87% code, 10% code-related English, 3% Chinese; Qwen2.5-Coder code:text:math 70:20:10 |
| FIM | 50% PSM in the packed data | 50% (DeepSeek-Coder, StarCoder2); Code Llama 90% |

What those models ablate: Code Llama the RoPE base of its long-context stage (Appendix G.4, Table 18: without the
raised base, retrieval fails at 8K despite fine-tuning at 16K) and FIM's cost to left-to-right completion;
DeepSeek-Coder the FIM rate (Fig. 3); StarCoder2 RoPE against learned positions; OpenCoder its data (deduplication
level, annealing data, GitHub stars). None ablates architecture details such as gains, scalars or attention windows.

### Experiments

One per decision `docs/h100-pretraining.md` records without a study of its own whose result would change production
(the rest: Not tested):

| File | Settings | Production (the recorded decision) | Alternative | Production cost if adopted |
|---|---|---|---|---|
| `lr.py` | d8 at 5 and 200, d24 at 5 (1 seed each): the d24 optimum times d8's horizon trend carried to production | DeepSeek LLM's law (§3.1) × 2.5467, adopted 2026-10-20 (§3, lr.py) | × 2 and × 4 (× 8 at d8): the law was fit for AdamW under a multi-step schedule, 7.5× below production's compute, and the transfer rules disagree on the optimum's horizon trend (d8 ratio: SDE rule × 0.354, Step-Law × 4.92); × 2 beat × 1 by 0.018 at d8 at 20 | none |
| `value_embeds.py` | d16 at 20 (4 seeds), d8 at 20 (4) and 490.8 (1) | value embeddings (MoVE: −0.016 at d12), absent from every reference model; 302M parameters, 1.55× the scaling parameters | none | −4.7% of the d24 step, 12 vocabulary tables fewer |

**Power.** The paired comparison's six runs (two configurations × three seeds, d8 at 20 tokens per parameter,
`docs/h100-pretraining.md`) give a within-configuration σ of 0.0034 bits per byte on code (0.0044 on solutions, 0.0036
on 32K repository); both experiments' 4-seed settings measure it where they run. Value embeddings at d16, 4 runs
against 4 (6 df): standard error 0.71σ = 0.0024, 80% power at an effect of 0.008, below the published 0.010 to 0.016.
lr: each vertex rests on the arms' differences within one seed, which varied by 0.0008 across 2 seeds at d8 at 20
against a 0.018 effect; at the curvature Bjorck et al.'s Llama-1 estimate implies (0.027 nats for a 2.5× lr; about
0.026 bits per byte per (ln lr)² at 3.5 bytes per token [assumed]) and per-run noise σ, a three-point fit places a
vertex within about ±14% (one standard error, 0.13 in u). The d8 ratio r combines two vertices' errors, about ±20%
(0.185 in u); raised to 1.24 for production's horizon range it contributes 0.23 in u, and with the d24 vertex
production's multiplier is within about ±30% (0.26 in u). The two rules' predictions for r differ by 14× (ln 14 = 2.6
in u), far beyond its error. `lr.py` measures the curvatures themselves.

**Cost** (measured: whole runs, child process, compilation, both stages, loss evaluation; §2). d8 at 5: about 3
minutes [estimate]; d8 at 20: 234 s; d8 at 200: about 24 minutes [estimate]; d8 at 490.8: about 54 minutes (52
minutes training, 91 s evaluation); d24 at 5: about 65 minutes (52 minutes training at 311K tokens per second, 711 s
evaluation); d16 at 20: about 35 minutes (30 minutes training at 746K tokens per second, 299 s evaluation). `lr.py`:
3 × 3 min + 4 × 24 min + 3 × 65 min ≈ 5.0 h; `value_embeds.py`: 18 runs ≈ 7.0 h; ≈ 12.0 h ($51 at $4.29/h).

**Records.** A run's tag is production's `config.run_tag` over its fields (its peak lr and the dataset revision
hashed with them), and a rerun experiment resumes: a run with an evaluation or a divergence is not trained again, a
stopped one trains again from its start (its one checkpoint is its final model); each run is launched from its fields
(`python -m ablations.harness '<fields>'`). Per run, in production's schemas:
`artifacts/metrics/<tag>.jsonl` (`nanochat/train/metrics.py`: every step and attempt, the learning curve and gradient
norms included; the run record holds the model config, the run's fields as `variant`, parameters, FLOPs per token,
the clip threshold, the plan, lr and weight decay; the end record the peak memory; a failure record the step a run
diverged at), `artifacts/base_eval/<tag>-step<steps>.json` (`train/metrics.save_evaluation`, loss evaluation only) and
`artifacts/logs/<tag>.log` (the run's output, a failure's traceback included). Each experiment gathers every tracked
metric of its runs into `results/`, so tables and figures need no rerun and no `artifacts/`:
`<experiment>.jsonl`, a record per run (arm, tag, setting, fields; `train/metrics.record`: its metrics records but the
per-step ones: run, stage, end, failure; its evaluation verbatim: per corpus and context bucket bits, bytes and rows at
8K and 32K, per benchmark solution bits and bytes; the compared measures; efficiency; the step it diverged at,
measures and efficiency null
then); `<experiment>-curves.jsonl`, per run its per-step train records as columns (step, tokens, loss, lr, gradient
norm before clipping, tokens per second, attempt, time; 2,987 steps of d8 at 20 take 360 KB, so all runs about 48 MB);
and `<experiment>.json`, per setting depth, tokens per parameter, seeds, production's evaluated runs, and per
alternative its fields, evaluated runs and per measure difference, p and confidence interval, null for one run per
arm; `lr.json` adds the Analysis (`analysis`: per setting runs, curvature, vertex and grid extension; m_B, m_D, the
law's fall, both predictions, the measured d8 ratio, production's multiplier and lr, null while a grid needs
extending).

### Not tested, and why

Decisions `docs/h100-pretraining.md` records with their own evidence, which the reference code models share, or which
no result here could change:

- **Weight decay.** The τ law (Power Lines, arXiv 2505.13738) is the regime's documented choice, tested with the
  regime as a whole by the paired comparison.
- **Warmup.** 2/(1 − β₂) = 40 steps (Ma & Yarats, arXiv 1910.04209): 0.11% of d8's run at 490.8, 0.05% of
  production. Warmup's benefit is a larger stable peak lr (Kalra & Barkeshli, arXiv 2406.09405) by bounding early
  update size, which normalized matrix updates such as Muon's already bound (Kosson et al., arXiv 2410.23922);
  `lr.py`'s × 2 arm probes that margin, its per-step loss and gradient norms showing any early instability.
- **Gradient clipping.** Adopted at 1.0 without an experiment: its one measurement, a paired d8 A/B (§3), had it lower
  on all 11 corpora by 0.003 bits per byte, which an affordable experiment (resolving about 0.008, Power) could not
  separate from zero; Adam's and Newton–Schulz's updates are invariant to a slowly varying gradient scale, so it
  damps only norm jumps, at no cost on one device (`docs/h100-pretraining.md`, Regime).
- **RoPE base staging.** Production's single base, 10^6, meets Men et al.'s lower bound for a 32K context (arXiv
  2405.14591) from the first step, as Llama 3 trains at 5 × 10^5 from the start (arXiv 2407.21783 §3.2); Code Llama's
  ablation (Appendix G.4) shows the base must be raised for long context, which production's already is. Staging
  (10^4 at 8K, as Code Llama and Qwen2.5-Coder) would add a base change for the repository stage to adapt to, with no
  published gain at 8K.
- **SwiGLU.** nanochat measured it worse than ReLU² at d12 and d24 under Muon (LOG 2026-02-05); at matched parameters
  it cost 5.1% of d8's throughput and would need a gated decode MLP (`csrc/decode.cu`).
- **The repository stage's schedule.** The linear decay continues through it, as DeepSeek-Coder (§3.6) and
  Qwen2.5-Coder continue theirs; re-warming (Gupta et al., arXiv 2308.04014) is measured for continued pretraining
  after a finished schedule.
- **Learned QK-norm gains and the residual and x0 scalars.** 2·depth scalars and two head-width vectors per layer,
  measured by nanochat (LOG 2026-01-11) and Qwen3's QK norm; no reference code model ablates such details.
- **SSSL windows.** Gemma 3 (§5.2) measures local:global up to 7:1 and windows down to 512 nearly lossless in
  perplexity, and StarCoder2 trains code under a 4,096-token window; full attention costs 8.8% of d8's throughput
  (§2) and ≈ 4.4% of the d24 step.
- **Block Attention Residuals and the partial key offset, not adopted.** +12% d24 step time for a published 1.1% loss
  gain at 194M (arXiv 2603.15031); a 124M speedrun result (modded-nanogpt record #49); either needs a native decode
  path.
- **NorMuon.** Its reported gain (arXiv 2510.05491: 6% efficiency at 124M) is about 0.002 bits per byte, below the
  experiments' resolution.
- **The repository stage itself.** A model never trained at 32K is worse at 32K by construction (Qwen2.5-Coder §3.2).
- **Shape.** One head narrower and deeper at d16's scaling parameters (31 × 384 against 16 × 512, MobileLLM §2.2.2)
  takes 26.7% more FLOPs per token (685M against 541M) and trains 27% slower (540K against 741K tokens per second,
  measured), so an equal-token comparison gives it more compute and the d24 analogue (35 × 640) would lengthen
  production by about as much [inference].
- **head_dim 64.** The decoder's attention is built for head_dim 128 (`config.HEAD_DIM`, compiled into its kernels),
  so adoption rewrites it; nanochat measured 128 better (LOG 2026-01-17).
- **Mixture.** The code metric falls with the code share by construction, and the coding gain Qwen2.5-Coder reports
  for 70:20:10 (Table 3, 7B on 5.5T tokens; non-monotone: 100:0:0 best, 85:10:5 worst) is a capability no small run
  shows; production keeps DeepSeek-Coder's 87:10:3.
- **FIM rate and format.** DeepSeek-Coder's ablation (Fig. 3): 50% PSM balances infilling and left-to-right
  completion; Bavarian et al. (arXiv 2207.14255) find FIM nearly free for left-to-right. A test repacks every corpus.
- **Vocabulary (2^15).** A test retrains the tokenizer and repacks every corpus.
- **Grouped-query attention.** Its benefit is KV-cache memory; d24's cache for the decoder's 256 slots (`csrc/decode.cu`
  MAX_BATCH) of the longest benchmark request (1,158 tokens) is 21.8 GB, which an H100 holds.
- **Untied head, the batch law:** the published evidence and pre-launch measurements in `docs/h100-pretraining.md`.
- **Production's width and length together.** No run: the experiments combine d8's length trend with the short
  settings' width trend, assuming separability (Bjorck et al.'s size-independent β for the lr; a log-linear trend for
  value embeddings), which only a d24 run at 490.8 could check.
- **Library alternatives without a measured gain at this scale** (a 2026 survey of torchtitan, OLMo-core, Megatron,
  nanotron, Marin, modded-nanogpt, nanochat): FP8 training, Liger and Cut Cross-Entropy losses, fused RMSNorm/RoPE
  libraries, DCP checkpointing, config frameworks.

### Results

`results/lr.jsonl`, `.json` and `results/value_embeds.jsonl`, `.json`; none yet.

## 2. H100 implementation

### The experiments' runs

d8 file-level throughput (steps 10–29 of a run on real rows, 2026-10-21): production 2.287M tokens per second, no
value embeddings 2.383M. Production's d24 step (12 micro-batches of 12 rows, file-level stage, torch.profiler, two runs): 3,700 / 3,689 ms, GEMM
1,966 ms, FA4 935 ms, peak 69.6 GiB.

Where a run's time goes (2026-10-22). Steps (file-level stage, micro-batches of 12 rows, torch.profiler): d8 (one
micro-batch) 43 ms per 98,304 tokens, GPU 93% busy (GEMM 49%, FA4 18%, cross entropy 12%, Inductor 11%); d16 (two)
264 ms per 196,608 tokens, 98% busy (GEMM 49%, FA4 25%); d24 (two) 633 ms per 196,608 tokens (311K tokens per
second), 98% busy. The loss
evaluation reads 640M tokens whatever the depth (390M at 8K over 11 corpora, 250M at 32K over 6): 8K, 32K and
solutions take 52.5, 35.7 and 2.4 s at d8; 173.7, 115.5 and 9.4 s at d16; 409.8, 282.9 and 18.6 s at d24. A d8 run at
20 (`d8-r20-s0-d0-2449fca0cf29`): 9 s to its run record, 135 s training, 5.5 s at the repository stage's switch
(snapshot and recompilation at 32K), 90 s evaluation. So the runs are bound by library kernels (QuACK GEMMs,
FlashAttention-4) at ≥ 93% GPU occupancy; d8's 7% idle is the most CUDA graphs could recover (≈ 11 minutes of
`lr.py`), against capture constraints on the varlen and custom-kernel paths production does not need at d24 (98%
busy), so no kernel or graph work is spent on the experiments. Their cost lever is the number and size of runs (§1).

### q/k prologue

Composite d24 (Block with QK gains, VE gate 2 sigmoid, torch AdamW + Muon), real data, one step of 1,179,648 tokens
(torch.profiler, the micro-batches and rows of each row below), 2026-10-19. GPU 99% busy in each: GEMM 54%, FA4 24%, Inductor 12%.

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

Real data, 12 × 12 rows (d24, 12 micro-batches of 12 rows, file-level stage, torch.profiler):

| Configuration | Step | Throughput | Notes |
|---|---|---|---|
| Per-role QuACK configs measured at M = 98,304, cuBLAS weight gradients | 3,785 ms | 311.7K tok/s | q/k prologue backward 168.4 ms; VE-gate fp32 GEMMs on CUDA cores (sm80_xmma f32 + simt sgemm) 80.6 ms |
| VE gate as fp32 broadcast-sum, persistent atomic-free q/k prologue backward | 3,724 ms | 316.8K tok/s | q/k prologue forward 75.2 ms, backward 92.7 ms |
| Production regime (d2z, τ-law weight decay, no clipping), per-role configs | 3,736 ms (step 5) | 315.7K tok/s | steps 2–7: 3,461 / 3,719 / 3,775 / 3,790 / 3,795 / 3,781 ms; peak 69.6 GiB; GPU 100% busy: GEMM 52.9%, FA4 24.9%, Inductor 14.3%, other 6.4%, cross entropy 1.5%. 1 × 12 rows: step 4 327 ms, 301.0K tok/s, peak 67.6 GiB |
| QuACK's own autotuner (`tuned=True`) for every GEMM, 2026-10-20 | 4,051 ms (step 5) | 291.2K tok/s | GEMM 2,408.1 ms (59.4%), FA4 875.0, Inductor 491.1, q/k prologue 168.1, cross entropy 57.5; peak 76.7 GiB (the autotuner's operand copies on a cold cache) |
| QuACK's own autotuner, repository stage (3 × 32,768 rows) | 4,325 ms (step 5) | 272.7K tok/s | GEMM 2,403.2 ms, FA4 997.2 ms; peak 69.4 GiB; 96% busy |
| **Final:** QuACK's autotuner for forward and dX, cuBLAS (`torch.mm`) for every fp32 weight gradient, 2026-10-20 | 3,705 / 3,713 ms (two runs) | 318.4K / 317.7K tok/s | GEMM 1,976.7 / 1,974.2 ms (53.3%), FA4 937, Inductor 496, q/k prologue 171, cross entropy 57; peak 69.6 GiB |
| **Final**, repository stage (3 × 32,768 rows) | 4,016 ms (step 5) | 293.8K tok/s | GEMM 1,974.3 ms, FA4 1,064.3 ms; peak 69.6 GiB; 96% busy |

Why the autotuner alone was slower (per GEMM on d24's real operands after two optimizer steps, CUDA-graph median of
5): the weight gradient dY^T X has a small output (768 × 768 or 768 × 3,072) and K = 98,304 tokens. QuACK's
`gemm` defaults to split-K 1, so the output's 24–96 tiles leave most of the 132 SMs idle, and its autotuner never
offers cuBLAS: c_q/c_k/c_v/c_o dW 0.544 ms (rel. error vs fp64 2.3e-4) vs cuBLAS 0.191 ms (1.8e-5), 1,152 calls per
step (+406 ms); MLP dW 0.700 / 0.751 vs 0.658 / 0.653 ms. With split-K, QuACK still trails cuBLAS (768 × 768: 0.214 vs
0.163 ms). Its forward and dX picks are within noise of the old per-role table (proj_fwd 0.174 vs 0.180 ms, mlp_dx
0.675 vs 0.706, proj_dx 0.198 vs 0.180, mlp_down 0.693 vs 0.649). The autotuner times each candidate once and caches
its pick on disk (`artifacts/cache/quack`), so a re-tune can move a role by ±20 ms per step.

Value-embedding gate alone, forward + backward of gate and mix per layer at M = 98,304 (`torch.utils.benchmark`
median), relative error vs fp64 of gate / dW:

| Implementation | Time | Error |
|---|---|---|
| fp32 `F.linear` | 1.195 ms | 1.3e-7 / 5.6e-7 |
| bf16 `F.linear` | 0.575 ms | 8.1e-4 / 3.0e-3 |
| TF32 | 1.008 ms | 8.0e-5 / 2.2e-4 |
| fp32 broadcast-sum (kept) | 0.881 ms | 5.3e-8 / 4.1e-7 |

### Inductor coordinate-descent tuning

`TORCHINDUCTOR_COORDINATE_DESCENT_TUNING`, d24 12 × 12 file, 2026-10-19, GPU alone, one profiled step each: off 3,741
ms (315.3K tok/s), Inductor kernels 533.3 ms; on 3,711 ms (317.9K tok/s), Inductor kernels 500.2 ms (GEMM and FA4
differ by run noise, ±10 ms). On, it picks kernel configurations by timing, so two processes of the same code compile
different kernels: d8 compiled loss differs by 6.7e-6 and per-token losses by up to 3.5e-2 between two baseline runs;
off, both are bitwise equal across processes (2026-10-20) → off: 0.8% of the step for reproducible losses.

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

Concurrent d8 training processes, 2026-10-20 (file-level stage, each timing a 16-micro-batch step of
1,572,864 tokens, started together; peak 13.3 GiB each): 1 process 2,598,140 tok/s; 2 processes 1,155,960 + 1,179,686 =
2,335,646 aggregate (−10%); 4 processes 589,606 + 614,366 + 672,177 + 692,019 = 2,568,168 (−1%) → one d8 run saturates
the GPU (its 280–440 W power draw understates SM occupancy): runs train one at a time.

### Decoding and checking

- **Native decoder** (`inference/decoder.py`), greedy, no stop rule, 125,952 tokens per run, under `gpu.lock`,
  2026-10-20: d24 `max_len` 32,768 (the context), 32 slots: 2,379 tok/s; `max_len` 1,158 (longest HumanEval+ prompt
  391 + 768 − 1), 64 slots: 3,008 tok/s (+26%); outputs identical on all 164 HumanEval+ prompts (d8, every `max_len`
  tried) → the KV cache sized to the workload.
- **Decoder kernels** (`csrc/decode.cu`), random-init d8 and d24 (zero lm_head: greedy never stops), 200-token prompts
  + 256 tokens, the GPU alone, 2026-10-20. Nsight Compute on the 64-slot GEMV (a warp per 16-row tile, all slots,
  full K): achieved occupancy 1.6–3.1%, DRAM throughput 0.7–3.6%; the 768-row matrices (o, proj) ran 48 warps on the
  whole GPU (65% of the d24 step), lm_head's argmax 2M `atomicMax` on 64 addresses (0.6 ms a step). After (a one-warp
  block per 16-row tile and 16-slot group, register weight ring of 8 blocks, `cp.async` inputs, the norm once per step
  in `k_norm`, a warp's argmax reduced before its atomic, 256 slots), ms per step:

  | | d8, 1 | d8, 64 | d8, 256 | d24, 1 | d24, 64 | d24, 256 |
  |---|---|---|---|---|---|---|
  | before | 0.98 | 3.42 | — | 6.78 | 18.05 | — |
  | after | 0.36 | 0.57 | 1.02 | 2.78 | 3.84 | 9.94 |

  d8 checkpoint: 64 greedy HumanEval+ tasks 2.7 → 0.5 s; 144 healed, syntax-constrained, sampled requests 2.1 → 0.8 s;
  tokens identical to the 64-slot build on both and under reversed submission order.
- **Prompt store** (`inference/decoder.py` `prompt_store`), d8 checkpoint, 48 healed, syntax-constrained HumanEval+
  prompts × 3 rounds (128 tokens each), 2026-10-20: rounds after the first 0.39–0.40 s without the store, 0.26 s with
  it (−35%); tokens identical with and without it, and to the requests decoded alone.
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
- **Peak lr at d8 at 20** (`harness.main`, production vs × 2, seeds 0 and 1, 2026-10-20; GPU shared, so throughput
  is not comparable): code bits per byte 0.6323 and 0.6324 at the law's lr, 0.6144 and 0.6137 at × 2 (paired
  differences −0.0179, −0.0187); solutions −0.014, 32K repository −0.018, text −0.030, math −0.033. The law's lr is
  below the optimum at this horizon; `lr.py`'s grids start at × 1 where they look upward.
- **lr.py** (2026-10-20, dataset revision `1e711aa`, seed 0; `results/lr.json`, `lr.jsonl`, `lr-curves.jsonl`): code
  bits per byte at × 1 / 2 / 4 / 8 of the law's lr: d8 at 5 0.89491 / 0.78969 / 0.76960 / 0.81791 (× 8 the amended
  rule's extension), vertex × 3.4669; d8 at 200 0.52774 / 0.51750 / 0.51592 / 0.52431, × 3.1565; d24 at 5 0.44230 /
  0.43787 / 0.44201, × 2.0235. The measured d8 ratio r = 0.9105 (horizon exponent −0.0254) is neither rule's (SDE
  × 0.354, Step-Law × 4.92): relative to the law the optimum barely moves, the law's own C^−0.125 carrying most of the
  horizon trend. Production's multiplier × 2.5467 = 2.0235 · 0.9105^(ln 98.17 / ln 40.05) · √2, peak lr 2.3887e-3
  (weight decay 0.1246), adopted as `scripts/production.py` `LR_SCALE`. No run diverged (up to × 8 at d8, 6.66e-3 at
  d24); each setting's best arm beats the law by 0.1253 (d8 at 5), 0.0118 (d8 at 200) and 0.0044 (d24 at 5). Checks:
  every derived number recomputed from the raw run records by an independent script (closed-form parabola) agrees to
  6e-15; each run's logged lr equals plan × schedule at every step and its weight decay the τ law (lr × wd constant
  within a setting); seeds and evaluation rows are identical within a setting; the last-5% train loss ranks each
  setting's arms as code does; `torch.optim.Muon` decays by the unscaled lr and steps by lr · 0.2√max(A, B)
  (Moonlight's form). Against references: nanochat's hand-tuned Muon lr 0.02 under its √max(1, rows/cols) scaling
  (karpathy/nanochat 92d63d4 `optim.py:393`) is the AdamW-matched 0.1/√C, 3.61e-3 at width 768 and 6.25e-3 at 256,
  against the measured optima 3.37e-3 (d24 at 5) and 6.47e-3 (d8 at 200); d8's optimal lr falls to 0.574 over 40×
  the tokens with its batch doubled (exponent 0.150, 0.244 net of √2 for the batch), against DeepSeek LLM's 0.125 under
  the same batch co-scaling and Bjorck et al.'s 0.32 at fixed batch and weight decay, where the τ law moves the weight
  decay instead (Wang & Aitchison, arXiv 2405.13698); the SDE rule's exponent 1/2 is too large, as arXiv 2604.27077
  also finds. At d24 at 5's curvature (0.0089 per (ln lr)²) a 30% lr error costs 0.0006. Wall time 13:09–18:39 UTC,
  37 minutes of it the GPU idle between the two `lr.py` invocations (a waiting script matched its own tmux session).
- **Compiler caches**, d8 (`base_train`, seed 0, real data), 2026-10-20: from two Inductor/Triton caches the same code
  agrees on the first steps and then differs by ~1e-6 relative (coordinate-descent tuning times kernels); over 2,987
  steps the mean loss of the last 5% of steps differed by 2e-4 (1.70273 vs 1.70255).
- **Gradient norms** (the clipping argument, §1): d8 at 20 tokens per parameter (`d8-r20-s0-d0-2449fca0cf29`), median
  over steps 0–298 0.464 (9% of steps above 1), 298–1,493 1.093 (57%), 1,493–2,987 1.522 (100%); d24 cut run
  (`d24-r490.805-s0-d0-c7c5fbbe4b31`), steps 0–42 1.866 (86%), 42–211 1.145 (83%), 211–423 0.969 (47%).
- **The experiments' components**, 2026-10-20:
  - *Arms* share production's plan at their setting: d8 at 50, 7,468 steps of 98,304 tokens (`repo_start` 7,061, peak
    lr 2.4378e-3); d8 at 490.8, 36,656 steps of 196,608 tokens (34,657, 1.8324e-3); d24 at 5, 3,307 steps of 294,912
    tokens (3,127, 1.6641e-3); the τ law sets each arm's weight decay from its lr, keeping the decay's timescale as the
    lr moves. Production's d24 plan unchanged: 81,171 steps, `repo_start` 76,743.
  - *Candidate* (`model.py`, real rows, 2026-10-21): with production's fields `Candidate` equals `GPT` bitwise.
  - *Every arm* (2026-10-22): `lr.py`'s at d8 over 50 and over 490.80456005542607 (`max_ratio` of d24, 20.6 s; × 0.25
    included) and at d24 over 5, `value_embeds.py`'s at d8 over 20 and 490.8 and at d16 over 20, compiled and trained
    4 steps through both stages on real rows with finite losses; weight decay at d8 over 490.8: 0.3597 (1.4387 at
    × 0.25, 0.7194 at × 0.5, 0.1798 at × 2), over 50: 0.3982, d24 over 5: 0.3914; parameters: d8 56,627,216, d24
    522,252,336, d16 218,124,320 (83,890,208 without value embeddings).
  - *lr.py's plans* (2026-10-20, `harness.plan_of`, τ-law weight decay at each arm's lr): d8 at 5, 746 steps of 98,304
    tokens (`repo_start` 705), law lr 3.2509e-3, weight decay 0.8878 / 0.4439 / 0.2219 at × 1 / 2 / 4; d8 at 200,
    14,937 steps of 196,608 (14,122), 2.0500e-3, 0.4916 / 0.2458 / 0.1229 / 0.0614 at × 1 / 2 / 4 / 8; d24 at 5,
    3,307 steps of 294,912 (3,127), 1.6641e-3, 0.3914 / 0.1957 / 0.0979. From the plans: m_B 4 (production) and 2
    (d8), m_D 98.18 (production) and 40.05 (d8), the law's fall 0.6306; predictions for the d8 ratio × 0.3544 (SDE
    rule) and × 4.9231 (Step-Law).
  - *Analysis* (planted code, `lr.py` through `harness.main` over planted records, optima × 2.4 and × 3.0 at d8 over 5
    and 200, × 1.8 at d24 over 5; before the 2026-10-20 amendment): a quadratic in u is recovered exactly (vertices
    × 2.4000, 3.0000, 1.8000; r × 1.2500, horizon exponent 0.060472, production's multiplier × 3.359307 = 1.8 ·
    1.25^(ln 98.18 / ln 40.05) · √2, lr 3.1509e-3), also with d8 over 200's × 8 diverged (its 3 other runs); an optimum
    outside its grid (× 8 at d8 over 5) is reported as extend "up" with the derived values null; a curve steeper above
    its optimum (0.02 (e^d − 1 − d), d = u − u*) biases each vertex low, × 2.240, 2.554, 1.679 (r × 1.140, multiplier
    × 2.795 against 3.359). The amended fit (2026-10-20, synthetic records): an optimum × 2.4 on a symmetric curve read
    × 2.4000 from × 1–8; × 3.0 on a curve four times steeper below read × 3.72 (one quadratic: × 3.95); a lowest run at
    either end of the grid extends it; a diverged run above the lowest gives the lowest run's multiplier.
  - *Production's arm, run 0 at 20 tokens per parameter* (whole run through `harness.evaluation` and its efficiency,
    2026-10-22, 248 s, unclipped): code 0.6354, solutions 0.6964, repo_32k 0.6201, repo_beyond_8k 0.4048, text 1.3619,
    math 1.3498; 2.28M tokens per second; peak memory 14.0 GB. The same run clipped at 1.0, launched from its fields
    (`d8-r20-s0-d0-73e955e0631c`, 237 s): code 0.6320, solutions 0.6958, repo_32k 0.6158, repo_beyond_8k 0.3938, text
    1.3614, math 1.3473, 2.27M tokens per second; 75.4% of its steps clipped (one seed each, under σ's resolution).
  - *Divergence* (planted failure records, `harness.main`): diverged runs are skipped on rerun, written with their step
    and null measures, and dropped from the comparisons (an arm left without runs is reported with 0 evaluated runs).
  - *Export* (`d8-r20-s0-d0-73e955e0631c` through `harness.main`): its record holds the run, stage and end records, the
    evaluation verbatim (11 corpora at 8K, 6 at 32K, 3 benchmarks) and the measures; its curve 8 columns of 2,987 steps.
  - *Harness end to end* (2026-10-20, a scratch experiment through `harness.main`: d8 at 20, seed 3, production and
    × 2, before and after the harness's consolidation): both runs trained, evaluated and exported; the current harness
    over the earlier run's artifacts writes the same `.json` bytes, the same curves and the same records less the
    dropped `probes` field. Retrained, the runs agree on their first 5 steps bitwise and then drift by the backward's
    atomics (Reproducibility, `docs/h100-pretraining.md`): last-5% train loss 1.69657 vs 1.69695 (× 1), 1.65248 vs
    1.65172 (× 2); code 0.63008 vs 0.63015, 0.61311 vs 0.61251.
  - *Real data* (screen run 0, `d8-r20-s0-d0-2449fca0cf29`): the data on disk is the pinned Hub revision
    (`6744e406…`, the download's recorded commit); the run record's mixture is the 11 production corpora at their
    weights; the first training micro-batch, rebuilt as `train` reads it (data seed 0), decodes to real C++ files and
    a FIM example; the loss evaluation reads the held-out val splits (stack_edu_python's first document: a word2vec
    module) and the benchmarks' reference solutions (HumanEval+ 164, MBPP+ 378, HumanEval-X C++ 164). The trained
    model's loss on that batch is 1.869 nats against 11.705 on uniformly random tokens (ln V = 10.397).
