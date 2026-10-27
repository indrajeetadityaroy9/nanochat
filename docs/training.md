# Training

How the base model (d24) is trained on one GH200: stages, plan, optimizers, schedule, the peak learning-rate
calibration, the run app and its records, hardware and throughput, reproducibility and the commands.
The code is the specification. Markers: [arithmetic] derived from code numbers; [inference] reasoning not measured;
[estimate]; [unverified] a figure without a record. Measurements name their commit (GH200 at 700 W, exclusive A/B).

## Code

|File|Section|
|---|---|
|`run.py` (`RunConfig`, `run_name`, `plan_of`, `checkpoint_steps`, `analysis_steps`, `training`, `evaluations`, `job`)|§2, §5, §6|
|`conf/run.yaml`, `conf/experiment/pretrain.yaml`|§5, §6, §9|
|`train/schedule.py` (`plan`, `RunPlan`, `WARMUP_STEPS`, `lr_lambda`)|§2, §4|
|`train/stages.py` (`drawn`, `stage_batches`, `max_tokens_per_param`)|§1, §5, §8|
|`train/optim.py` (`optimizers`, `Muon`, `tau_weight_decay`, `update`, `MAX_GRAD_NORM`)|§3|
|`train/loop.py` (`train`, `SEED`), `train/diagnostics.py` (`Diagnostics`, `sidecar`), `train/checkpoint.py`|§6, §8|
|`ablations/peak_lr.py` (`lr_multiplier`); `conf/experiment/peak_lr_d24.yaml`; `results/peak_lr_d24.jsonl` and its `-curves.jsonl`|§5|
|`ablations/records.py` (`Records`, `record`)|§6|
|`scripts/setup.sh`, `scripts/container.sh`, `scripts/release.sh`, `paths.py`|§7, §9|
|`docker/Dockerfile`, `pyproject.toml`, `uv.lock`|§8|

## 1. Data and stages

The corpus side (mixture, packs, FIM, loader) is `docs/corpus.md` §7–§9. Training reads two stages
(`corpus/mixture.py` `FILE_LEVEL_STAGE`, `REPO_LEVEL_STAGE`; `RunPlan.stage_at`): 8,192-token files, then
32,768-token repositories over the last `REPO_LEVEL_SHARE` (5.45%) of the steps, in micro-batches of
`MICRO_BATCH_TOKENS` = 98,304 tokens (12 rows at 8K, 3 at 32K; the size: §7; the stages' basis: `docs/corpus.md` §7).
The loss is next-token cross entropy, the packs' FIM pieces read as FIM examples (`docs/corpus.md` §8). At the
repository stage's first step the loop sets the 32K context and recompiles the model.

`train/stages.py` `stage_batches`: a stage's micro-batches in the run's order, each corpus's packed rows permuted once
by the seed and drawn by the stage's weights at the rows the run draws (`mixture.weights`). `max_tokens_per_param`:
the tokens per non-embedding parameter of the most whole micro-batches bisection finds at which no stage overdraws,
not necessarily the largest (not monotone, since the batch steps up with the count). A run's `tokens_per_param`
defaults to it, so no stage reads a document twice.

| | The base model | Reference code models |
|---|---|---|
| Context stages | 8K files, then the last 5.45% of steps at 32K on repositories | Qwen2.5-Coder the same 8K → 32K repository stage (arXiv 2409.12186 §3.2); DeepSeek-Coder repository-ordered files (arXiv 2401.14196 §2.2), then 1,000 steps at 16K (§3.6); Code Llama a 16,384-token long-context stage (arXiv 2308.12950 §2.4) |

## 2. Plan

`train/schedule.py` `plan(model, tokens_per_param)`, with N the non-embedding parameters (`num_non_embedding_params`:
decoder layers and `lm_head`, what the token budget counts) and M DeepSeek LLM's non-embedding FLOPs per token at the
file-level context (`non_embedding_flops_per_token`, Eq. 2, arXiv 2401.02954 §3.2; both `docs/architecture.md` §1):

- tokens D = `round(tokens_per_param · N)`, compute C = M·D;
- batch B = 0.2920·C^0.3271 tokens in whole micro-batches, steps = D // B;
- `repo_level_start` = steps − `round(REPO_LEVEL_SHARE · steps)`;
- the law's peak lr η = 0.3118·C^−0.1250 (§3.1 Eq. 1), times `learning_rate_scale` (`run.py` `plan_of`; §5).

Both experiments take the d24 plan at their `tokens_per_param` (`run.py` `plan_of`): the base run at
`max_tokens_per_param`, the `peak_lr` arms at 5. Recomputed on the CPU from master, the corpus's packs and
`results/peak_lr_d24.jsonl` (peak lr at the base multiplier 2.7192600, §5; weight decay by §3's law); the d16 and d8
rows, the settings of the ablations (`docs/experiments.md` §1), by `67192ab`'s code, where `RunConfig` still set the
depth:

| Run | Tokens/param | Micro-batches | Steps | Repo start | Tokens | Law lr | Peak lr | Weight decay |
|---|---|---|---|---|---|---|---|---|
| d24 base (`max_tokens_per_param` 526.134) | 526.127 | 15 (1,474,560 tokens) | 69,611 | 65,814 | 102.65B (97.05B at 8K, 5.60B at 32K) | 8.6290e-4 | 2.3465e-3 | 0.15342 |
| d24 at `mixture.MAX_STEPS` | 525.893 | 15 | 69,580 | 65,785 | 102.60B | 8.6295e-4 | 2.3466e-3 | 0.15345 |
| d16 at 20 | 19.998 | 3 | 4,552 | 4,304 | 1.342B | 1.6686e-3 | 4.5373e-3 | 0.2166 |
| d8 at 20 | 19.997 | 1 | 2,987 | 2,824 | 0.294B | 2.4438e-3 | 6.6455e-3 | 0.2253 |
| d8 at 200 | 199.993 | 2 | 14,937 | 14,122 | 2.937B | 1.8326e-3 | 4.9834e-3 | 0.2022 |
| d24 at 5 (`peak_lr`) | 4.999 | 3 | 3,307 | 3,127 | 0.975B | 1.5443e-3 | | |

N and M: `docs/architecture.md` §1. At `67192ab`, `max_tokens_per_param` of the corpus's tokenizer was 6,990 at d8
and 1,529 at d16. The stages' token budgets are sized by `mixture.MAX_STEPS` (`docs/corpus.md` §7), the table's
second d24 row (`b065ad7`); the plan on the packs is 69,611 steps (+0.04%).

**FLOP count (`ba01bb5`, `b065ad7`).** The plan uses Eq. 2's M (full attention at the 8K context in every layer),
the definition §3.1 Eq. 1's constants were fit with (§3.2: "M includes the computational overhead of attention
operation"); nanochat 92d63d4's per-window count is 1.82× lower at d24/8K (12.06 instead of 14.67 micro-batches, lr
×1/0.928). The batch is not ablated: Muon's loss is flat across a 2× batch at 8× Chinchilla (arXiv 2509.02046 App.
C.9 Table 85: 130M, batch 128 and 256 both 3.240, 512 3.257).

## 3. Optimizers

`train/optim.py` `optimizers`, `update`:

| Group | Parameters | Optimizer | Weight decay |
|---|---|---|---|
| tables | `embed_tokens`, `lm_head`, the 12 value-embedding tables | AdamW (fused), β = (0.9, 0.95), PyTorch's eps | λ |
| vectors | QK-norm gains, `resid_lambdas`, `x0_lambdas` (every 1-D parameter) | AdamW, same | 0 |
| matrices | q/k/v/o, up/down, value-embedding gates (every other 2-D parameter) | `Muon` (NorMuon) | λ |

- **Muon** (`Muon.step`): `torch.optim.Muon`'s update (momentum 0.95, Nesterov, 5 bf16 Newton–Schulz steps with
  PyTorch's coefficients, decoupled weight decay), the Newton–Schulz batched over same-shape matrices (one `baddbmm`
  chain per shape instead of `torch.optim.Muon`'s per-matrix `mm`/`addmm`, as it has no foreach path; `a81410c`);
  NorMuon (arXiv 2510.05491 Alg. 1): each row of the orthogonalized update O divided by
  √v + 1e-10, v an EMA (β₂ 0.95) of the row's mean square, then rescaled to O's Frobenius norm; step size
  `match_rms_adamw`, lr·0.2·√max(fan_out, fan_in) (Moonlight, arXiv 2502.16982 §2.2).
- **One lr and one weight decay** for every group, as Moonlight shares "learning rate η, weight decay λ" among matrix
  and non-matrix parameters (§2.2). Per-group candidates were not adopted (`docs/experiments.md` §4).
- **Weight decay** by the τ law (`tau_weight_decay`): the EMA timescale 1/(η·λ) is τ* = 1.084·TPP^−0.527 of the run's
  steps (Power Lines, arXiv 2505.13738 Finding 3, App. E.3.1; arXiv 2405.13698), so λ = 1/(η·τ*·steps), 0.15342 for
  the base run. The law is fit on AdamW over TPP 20–1280 (the base run's 526 inside). Muon takes the same law: under
  μP a fixed independent weight decay stays optimal across TPP for both Adam and Muon (arXiv 2512.05620 App. H.5), and
  Muon's only scaling-sensitive hyperparameter is its lr (arXiv 2509.02046 §3.3 Table 4). The Muon-specific timescale
  runs: `docs/experiments.md` §4.
- **Clipping**: the accumulated gradient's global norm clipped to `MAX_GRAD_NORM` = 1.0 before both optimizers
  (DeepSeek LLM §2.3), the pre-clip norm logged every step. Adam is invariant to the gradient's scale (arXiv 1412.6980
  §2.1) and Newton–Schulz divides its input by its norm (`torch/optim/_muon.py` `_zeropower_via_newtonschulz`, after
  KellerJordan/Muon), so under a slowly varying gradient scale the clip damps only spikes [inference]; the largest
  recorded norm is 10.76 (`results/peak_lr-curves.jsonl` at `81fb386`, d8 at 200, ×1, step 14,202), far below the
  ~5e7–2e11 at which Megatron-LM PR 5395 reports pre-Muon clipping failing [unverified: PR text].

| | The base model | Reference code models |
|---|---|---|
| Optimizer | NorMuon on matrices, AdamW on tables and vectors | AdamW, β = (0.9, 0.95) (DeepSeek LLM §2.3, followed by DeepSeek-Coder §3.4; Code Llama) |
| Peak lr | the law × the measured multiplier (§5), peak of linear decay to zero | the law's lr held for 80% of tokens, then 31.6% and 10% of it (DeepSeek LLM §2.3; DeepSeek-Coder §3.4) |
| Weight decay | τ law, 0.153 | 0.1 (DeepSeek LLM §2.3) |
| Gradient clip | 1.0 | 1.0 (DeepSeek LLM §2.3) |

**NorMuon's adoption (`646a986`).** A screen at d16 at 20 tokens per parameter, one seed per arm paired with base seed
0, at ×1 (the law's lr, below the base multiplier): NorMuon's training loss over the last 50% of steps −0.0029 below
the base's (negative in all 23 hundred-step windows), every bits-per-byte measure better by 0.0003–0.0013. One seed
cannot resolve that: base seeds 1 and 0 differ by 0.0014–0.0027 bpb and the treatment effect's seed spread is
unestimated. NorMuon is adopted on the literature: 11.31% over Muon at 1.1B (arXiv 2510.05491 abstract);
modded-nanogpt PR #144 (on top of Polar Express, −15 steps over 20 runs); karpathy/nanochat `dev/LOG.md` 2026-01-10
("a very small improvement", kept). A d24 check (200 steps, the earlier recipe's ×2.797, seed 0, 8K): last-20 loss
2.2946 against 2.2967, largest gradient norm after step 40 1.153 against 1.387, no spike, −0.08% tok/s, equal peak
(77.5 GiB). The screen's other arms: `docs/experiments.md` §4.

## 4. Schedule

`train/schedule.py` `lr_lambda`, every group's `LambdaLR` multiplier:

- **Warmup**: linear over `WARMUP_STEPS` = 2/(1 − β₂) = 40 steps (arXiv 1910.04209 §4.3, Eq. 14: "linear warmup
  over τ = 2·(1−β2)^−1 iterations"; karpathy/nanochat 92d63d4 uses 40 at d24); step s trains at (s + 1)/40. In the
  earlier recipe's 11 `peak_lr` runs (×1 to ×8, §5) the largest gradient norm of the first 40 steps is 2.46–3.00 at d8
  and 3.21–4.07 at d24, and none diverged (`results/peak_lr-curves.jsonl` at `81fb386`). No paper fixes a warmup for
  Muon.
- **Decay**: linear to zero, (steps − s)/(steps − 40), through the repository stage: linear decay to zero is "around
  0.84% better than the optimal WSD", its gap to a 10× decay grows with tokens per parameter, and its gain over
  cosine decay to zero is small but consistent across peak lrs (arXiv 2502.15938 §4.2, Figs. 6 and 8).
  DeepSeek-Coder instead holds its final lr for long context (arXiv 2401.14196 §3.6);
  Qwen2.5-Coder states no schedule for its repository stage.
- **Repository stage**: its first step at 3,797/69,571 = 5.46% of peak [arithmetic].
- **Last step**: 1/(steps − 40) of peak, not 0: 1.437e-5 at 69,611 steps [arithmetic], ~70× below arXiv 2502.15938's
  smallest nonzero minimum, 0.1% of peak (App. C.7).

## 5. Peak learning-rate calibration

**Varied.** The multiplier of DeepSeek LLM's law η = 0.3118·C^−0.125 (arXiv 2401.02954 §3.1 Eq. 1), fit for AdamW
under a multi-step schedule (§2.3) at compute below the base model's, applied here to Muon (RMS-matched) as the peak
of linear decay to zero; at a fixed model it falls with tokens as D^−0.125 [arithmetic]. The transfer rules disagree
on how the optimum moves with the horizon: Complete^(d)P's AdamW SDE rule η ∝ √(m_B/m_D) (arXiv 2512.22382 §2.1
Table 1) lowers it, as arXiv 2409.19913 §3.2 Eq. 1 and the law do; Step-Law η = 1.79·N^−0.713·D^0.307, fit over AdamW
models (arXiv 2503.04715 §1 Eq. 1), raises it. The multiplier is measured at d24 itself (the base model's shape,
standard parameterization, not μP: no width or depth transfer) over 5 tokens per parameter, and carried to
the base plan by the law, whose η(C) and batch B(C) were fit jointly across compute (arXiv 2401.02954 §3.1): the
base multiplier is d24 at 5's optimum. That the multiplier carries is an assumption [inference]: the law was fit at
compute-optimal allocations, while here the model is fixed and its tokens grow 105×. No smaller model stands in for
d24's horizon trend; the one measurement of the assumption is the earlier recipe's d8 (below), whose optimal
multiplier of the law changed ×0.9105 over 40× its tokens (the optimum itself ×0.574, the law's lr ×0.6306).

**Arms.** × 1, 2, 4 at d24 at 5; seed 0; 3 runs, each the current recipe at its setting (corpus tokenizer and mixture,
cap, `generate` "parse"): `conf/experiment/peak_lr_d24.yaml`, whose records are `results/peak_lr_d24.jsonl` and its
`-curves.jsonl` (§6).

**Rule** (`ablations/peak_lr.py` `lr_multiplier`): the base multiplier is the vertex of the parabola through the three
runs' code bits per byte against u = ln(multiplier) (`metrics.parabola_vertex`, `docs/evaluation.md` §5; arXiv
2409.19913 §3.1 fits quadratics in log lr); the base lr is it times the law's lr on the base plan. The recorded
minimum is interior (× 2) and no run diverged, so the vertex lies inside the grid. `67192ab`'s `fit` also covered the
outcomes the records did not take (extend the grid × 2 when the lowest-code run is at an end; that run itself when
its upper neighbour diverged) and re-checked each record against the current recipe (`check`); the code keeps the
rule the records took.

**Result** (`results/peak_lr_d24.jsonl`; 3,307 steps of 294,912 tokens; bits per byte; no run diverged):

| Measure | × 1 | × 2 | × 4 | Vertex | Curvature | Base multiplier | Base peak lr |
|---|---|---|---|---|---|---|---|
| code (8K, the fitted measure) | 0.41322 | 0.40544 | 0.40591 | × 2.7193 | 0.0086 | 2.7192600 | 2.3465e-3 |
| code at 32K (`repo_32k`) | 0.40859 | 0.40131 | 0.40227 | | | | |
| solutions | 0.49200 | 0.48710 | 0.48498 | | | | |

The best arm (× 2) beats the law (× 1) by 0.0078 code bpb; × 4 is 0.0005 behind it. The vertex, 2.7192600 times the
law's lr, is the base multiplier, 2.8% below the earlier recipe's 2.7970369 [arithmetic]: peak lr 2.3465e-3 and
weight decay 0.15342 on the base plan (§2).

**Precision and robustness** (each evaluation's `errors`, `docs/evaluation.md` §2, §5, with `row_clusters`, added to
the arms' records at `dba27e6`: val rows clustered by document, corpora as strata; the × 2 arm): 8K code 0.0022
(49,765 rows, 44,829 clusters), 32K code 0.0157 (2,667 rows, 2,255 clusters; 93% of its variance one 58.8M-token
repository), 8K all 0.0020. The arms score identical rows,
so their differences are paired (arXiv 2411.00640 §4.2; `metrics.mixture_bpb_summary` of the per-row bit
differences [arithmetic]): 8K code × 1 − × 2 = +0.00778 (standard error 0.00003), × 4 − × 2 = +0.00047 (0.00002).
The val sample therefore resolves every arm; what it does not measure is training noise, one seed per arm: two seeds
of one d16 arm differ by 0.0019 code bpb (`results/muon_wd_timescale_20tpp.jsonl`, τ × ½), four times the × 2–× 4
gap. The measure brackets a flat optimum over × 2–× 4, and × 2.7193 is the parabola's point estimate without an
interval. The same fit on the other likelihood measures of the three evaluations (`metrics.measures`):

| Measure | × 1 | × 2 | × 4 | Lowest | Vertex |
|---|---|---|---|---|---|
| code, 8K (the rule) | 0.41322 | 0.40544 | 0.40591 | × 2 | × 2.719 |
| code, 32K | 0.40859 | 0.40131 | 0.40227 | × 2 | × 2.610 |
| code, 8K and 32K by the base run's stage tokens (97.05B, 5.60B) | | | | | × 2.713 |
| repositories beyond 8K (`repo_beyond_8k`) | 0.36366 | 0.35736 | 0.35858 | × 2 | × 2.527 |
| benchmark solutions (`solutions`; 706 rows, standard error 0.0090) | 0.49200 | 0.48710 | 0.48498 | × 4 | × 4.782 |
| all, 8K / 32K | 0.49386 / 0.49250 | 0.48499 / 0.48403 | 0.48533 / 0.48479 | × 2 | × 2.756 / × 2.672 |
| math, 8K / 32K | 0.53770 / 0.53733 | 0.52497 / 0.52459 | 0.52556 / 0.52518 | × 2 | × 2.743 / × 2.744 |
| text, 8K / 32K | 1.07390 / 1.07466 | 1.05797 / 1.05881 | 1.05734 / 1.05821 | × 4 | × 2.910 / × 2.906 |

The mixture measures' vertices span × 2.53–2.91: −7% to +7% of the base lr, ≤ 0.00005 bpb at the fitted curvature
[arithmetic]. The solutions measure alone falls through × 4 and puts its vertex beyond the grid, but its × 2–× 4 gap
(0.0021) is 1.5 paired standard errors (0.0014; marginal 0.0090 over 706 rows); at the code fit's curvature its × 4.78
would cost 0.0027 code bpb [arithmetic]. The parse measure's generations (single seed, greedy): HumanEval+ base pass@1
0.067 / 0.104 / 0.110 (standard error 0.020–0.025), MBPP+ 0.114 / 0.132 / 0.127 (0.016–0.017), × 2's paired
differences from × 1 and × 4 at most 2.1 standard errors (HumanEval+ against × 1, exact sign test p = 0.07); MBPP+
parse rate 0.825 / 0.894 / 0.775, where × 2 parses 70 problems × 4 does not and × 4 25 that × 2 does not (exact sign
test p = 4e-6), one trained model each, while two seeds of one D7 arm differ more (pooled unparsed 0.293 and 0.133,
`docs/experiments.md` §1). The rule reads none of them.

**Dynamics** (the arms' `metrics.jsonl`, `train` records; × 1 / × 2 / × 4):
- training loss (trailing 50-step means): × 4 leads until step 468, then trails × 2 (+0.011 at step 1,000, +0.002 at
  the end); × 2 − × 1 stays within −0.020 to −0.024 from step 1,500; final 1.230 / 1.208 / 1.210;
- clipping: 41 / 35 / 25 steps above `MAX_GRAD_NORM`, the last at steps 45 / 40 / 26; median gradient norm after
  step 40 0.181 / 0.131 / 0.110; every loss and norm finite;
- 31 loss spikes in the file-level stage (a step above the previous 50-step mean by 0.15) at the same steps in all
  three arms and of the same size (+0.188 on average): batches of the fixed data order, not the lr; Stack-Edu Python's
  own loss exceeds its mean by > 0.3 nats in 16 of them;
- largest logsumexp 16.6 / 17.9 / 19.1 (steps 2,531 / 1,129 / 1,344) and largest QK-norm gain 1.69 / 1.93 / 2.25,
  the gains at that level from step ~3,000;
  parameter norm of the Muon group at the end 377 / 719 / 1,426, roughly ∝ lr, as the τ law's fixed η·λ implies
  [inference];
- the repository stage (step 3,127, lr at 5.5% of peak): the 20-step mean loss falls by 0.012 in each arm;
- 361–362K / 288K tok/s at 8K / 32K, MFU 0.481–0.483, peak 71.10 GiB, 2,739–2,774 s of training per arm, independent
  of the lr. The sweep ran at `5e06eb8`: about 10,320 s for the three arms with their evaluations (668–697 s each in
  parse mode, from the `end` record to the evaluation file [unverified: the records were rewritten at `dba27e6`, so
  their file times no longer show it]).

**Adopted** for the base run: `learning_rate_scale` = `lr_multiplier` = 2.7192599564485445 (records at `b7f5be9`) on
the §2 plan (69,611 steps of 1,474,560 tokens, peak lr 2.3465e-3, weight decay 0.15342), run
`generate=benchmarks,learning_rate_scale=2.7192599564485445,num_checkpoints=10` (§6, §9). Its absolute peak lr lies
between the × 1 and × 2 arms' (1.5443e-3, 3.0886e-3): the law lowers the lr with the base plan's compute.

**Earlier recipe** (`results/peak_lr.jsonl` at `81fb386`, removed since; code bits per byte; no run diverged). 11 arms
(× 1–8 at d8 at 5 and at 200, × 1–4 at d24 at 5) trained under an earlier recipe: the reference tokenizer's data
(dataset revision `1e711aa7`), no output-logit cap, nanochat's per-window FLOP count for the law's lr (Eq. 2's M, §2,
lowers it by 1/1.1186 at d8 and 1/1.0776 at d24, hence the re-based vertices), and an earlier mixture (each run's
`run` and `stage` records): at 8K eleven corpora (Stack-Edu, UltraData code and Stack-v3 L2 Python and C++, Kaggle
notebooks, Codeforces CoTs, ClimbMix 0.10, two math corpora), at 32K the six Stack-Edu and UltraData Python and C++
corpora (file-level code, not repositories) over the last 41 of 746, 815 of 14,937 and 180 of 3,307 steps.

| Setting | Plan | × 1 | × 2 | × 4 | × 8 | Vertex (recorded lr) | Curvature | Re-based |
|---|---|---|---|---|---|---|---|---|
| d8 at 5 | 746 steps of 98,304 tokens | 0.89491 | 0.78969 | 0.76960 | 0.81791 | × 3.4669 | 0.0712 | × 3.8781 |
| d8 at 200 | 14,937 steps of 196,608 | 0.52774 | 0.51750 | 0.51592 | 0.52431 | × 3.1565 | 0.0104 | × 3.5309 |
| d24 at 5 | 3,307 steps of 294,912 | 0.44230 | 0.43787 | 0.44201 | | × 2.0235 | 0.0089 | × 2.1804 |

The best arm beat the law by 0.1253, 0.0118 and 0.0044. r = 0.9105 (predicted: SDE rule 0.3544, Step-Law 4.9231; law
fall 0.6306; horizon exponent −0.0254, −0.1193 net of the batch; m_B,d8 = 2, m_D,d8 = 40.05): the optimum barely
moved with the horizon, against both rules. The removed d8-transfer estimator, vertex(d24 at 5)·m_D^h·√m_B (h the
net horizon exponent −0.1193, m_D = 105.25 and m_B = 5 the base plan's token and batch ratios over d24 at 5), gave
the base multiplier 2.7970369: peak lr 2.4136e-3 on the base plan of `81fb386`'s packs.

**`lr_multiplier`.** `conf/run.yaml` sets every run's `learning_rate_scale` to `${lr_multiplier:}`, a cached resolver
(`ablations/peak_lr.py` `lr_multiplier`, registered by `run.py`): it fits the experiment's records at launch and
returns the vertex; no copy is stored. `RunConfig`'s own default 1.0 is the law's lr (the `peak_lr_d24` experiment
sets it per arm, so it never resolves the multiplier), and the base run's name carries the resolved multiplier (§6).
A launch other than `peak_lr_d24` fails until its records exist.

## 6. Runs and records

Both experiments' runs are `run.py` `RunConfig`s (Hydra, hydra-core 1.3.8, omegaconf 2.3.1), trained by
`train/loop.py` `train` and evaluated by `eval/suite.py` `evaluate` (`docs/evaluation.md` §1). The model is d24
(`docs/architecture.md` §1), initialized and its data ordered by `train/loop.py` `SEED` = 0. The fields' defaults are
the base model's recipe:

| Field | Default | Meaning |
|---|---|---|
| `tokens_per_param` | None | the data's `max_tokens_per_param` (§1) |
| `learning_rate_scale` | 1.0 | times the law's lr; `conf/run.yaml` sets `${lr_multiplier:}` (§5) |
| `num_checkpoints` | 1 | evenly spaced evaluated checkpoints, the last the final model; > 1 adds `repo_level_start` and the analysis checkpoints |
| `generate` | "parse" | `evaluate`'s mode: "parse" (`peak_lr`) or "benchmarks" (base) |

`conf/run.yaml` composes the `run_schema` ConfigStore node, an experiment (`conf/experiment/<name>.yaml`, which sets
or sweeps fields) and the command line's overrides. Every invocation is a Hydra multirun (`hydra.mode: MULTIRUN`, one
job without sweep parameters), so the run's directory is `RUNS/<run_name>` (`hydra.sweep.dir` `${runs:}`, `subdir`
`${run_name:}`) and the experiment's `Records` callback runs when the sweep ends: `python run.py experiment=pretrain`
(`num_checkpoints` 10, `generate` "benchmarks") and `python run.py experiment=peak_lr_d24` (§5). The output-logit cap
is the model's, not a run field (`docs/architecture.md` §1). The run logs of `67192ab` and earlier carry the
`RunConfig` fields since removed (`num_hidden_layers` 24, `seed` 0, `tokenizer` "corpus" in each `run` record of
`results/peak_lr_d24.jsonl`'s `records`), all at their defaults, which no run name includes; each line's `config` is
the current `RunConfig`, as `ablations/records.py` regenerates the file from the run directories.

**Names.** `run_name`: the fields that differ from `RunConfig()`, sorted, `k=v` joined by `,` (`defaults` if none).
The base model's run (`experiment=pretrain`) is
`generate=benchmarks,learning_rate_scale=2.7192599564485445,num_checkpoints=10` (the multiplier resolved by
`lr_multiplier`, §5). Its directory `artifacts/runs/<run_name>/` holds `.hydra/` (the composed config),
`metrics.jsonl` (the run log; each line carries `run_name`, `attempt`, `kind`, `time`), the diagnostics sidecar
(below), `checkpoint-<step>.pt` and
`weights-<step>.pt`, `eval-<step>.json` with its sampled-completions sidecar `eval-<step>.samples.json.gz`
(`docs/evaluation.md` §1) and `run.log`. The run log's records (`train/loop.py`; field definitions in
`docs/evaluation.md` §10):

| `kind` | When | Content |
|---|---|---|
| `run` | each process start | first step, `plan`, `config` (the `RunConfig`), the tokenizer's `data_key`, the model config, `num_parameters`, M (`flops_per_token`), PaLM's nominal `model_flops_per_token` (`docs/evaluation.md` §5), each group's `learning_rate` and `weight_decay`, the stage and its mixture |
| `train` | each step | `loss`, `num_input_tokens_seen`, `learning_rate`, `grad_norm` (before clipping), `step_seconds`, `tokens_per_second`, executed `model_flops`, `mfu`, the diagnostics |
| `stage` | the repository stage's first step | the stage and mixture, its nominal FLOPs per token, the peak allocated and reserved memory since the process start (the file-level stage, or its part after a resume) |
| `end` | after the last step | the last stage's peak allocated and reserved memory |
| `failure` | a non-finite loss or gradient norm | the step |

A step's `loss` is the mean next-token loss in nats over its valid labels (`corpus/loader.py` `IGNORE_INDEX` marks
padding and a BOS that is no target): each micro-batch's mean, weighted by its share of the step's valid labels
(`model/ops/cross_entropy.py` `step_shares`), so Σ loss / Σ labels pooled over the micro-batches, computed in fp32 by
the same `lm_loss` call, on the capped logits, that gives the gradient. A step's values are copied to pinned host
memory without blocking and logged after the next step is launched, so the device never drains; its time runs
between CUDA events at the previous step's end and its own, so a process's first step and the stage switch's carry
the graph's compilation, and a step after a save the save (below); `tokens_per_second` is the step's row positions
(padding included) over that time.

**Diagnostics** (`train/diagnostics.py`). In the `train` record: per-corpus training loss (Σ loss / Σ valid labels
over the corpus's rows, rows assigned by `mixture.mixture_rows`); the capped logits' logsumexp maximum and mean
square over valid positions (PaLM's z-loss quantity, arXiv 2204.02311 §5); the largest QK-norm gain
(attention-logit growth, the quantity Kimi K2's QK-Clip bounds, arXiv 2507.20534 §2.1); every
`UPDATE_STATS_EVERY` = 10th step the GPU's telemetry, read on the host from NVML (the image's pynvml, initialized
once with a handle to the model's GPU) when the record is written, while the next step runs (a record written during
a checkpoint save, the first compilation or after the last step reads an idle GPU), with no device synchronization:
`gpu_power_watts` (`nvmlDeviceGetPowerUsage`, a 1 s running average on Hopper, NVML API
reference), `gpu_sm_clock_mhz` (SM clock), `gpu_temperature_c` (die) and `gpu_memory_used_bytes` (the device's,
every process included), which relate `step_seconds` to the 700 W cap and the clock lock (§7). These are the run's
only GPU telemetry. In the sidecar, per tensor and per layer:

| File | Row | Columns (`diagnostics.json` names them) |
|---|---|---|
| `diagnostics-step.npy` | step s, every step (69,611 × 316 for the base run) | `grad_norm/<tensor>`: each of the 220 parameter tensors' gradient L2 norm before clipping, in `named_parameters` order; `gain_max/<tensor>`: each layer's q and k QK-norm gain ‖g‖∞ after the step (48); `resid_lambdas/<i>`, `x0_lambdas/<i>` after the step (24 each) |
| `diagnostics-update.npy` | step 10·r, every `UPDATE_STATS_EVERY` = 10th step (6,962 × 440) | `param_norm/<tensor>`: L2 norm after the step; `update_rms/<tensor>`: RMS of lr × update without weight decay (AdamW's lr·m̂/(√v̂ + eps) from its state, NorMuon's adjusted-lr update) |

The arrays are `.npy` float32 (`numpy.load(..., mmap_mode="r")`), preallocated at the run's first start with NaN in
every row not yet logged; `diagnostics.json` also lists each tensor's name, shape, optimizer group (`adamw[0]`
tables, `adamw[1]` vectors, `muon[0]` matrices) and decoder layer. A row is written when its step is logged, so a
resumed attempt overwrites the rows of the steps it repeats, as its records supersede in `metrics.jsonl`; a diverged
step's row is written before its `failure` record. The rows reach disk with each full-state checkpoint. The sidecar
is the one record of per-tensor and per-group values: a group's parameter norm is √(Σ norm²) and its update RMS
√(Σ n·rms² / Σ n) over its tensors (n their element counts). `metrics.jsonl`'s `grad_norm` is the L2 norm of the
step's row of `grad_norm/` columns: `optim.update` takes the per-tensor norms with `torch._foreach_norm`, as
`torch.nn.utils.clip_grad_norm_` does, and clips with their total (`clip_grads_with_norm_`). Disk for the base run:
88.0 MB and 12.3 MB [arithmetic]. All values reach the host through the step's pinned buffer, read one step late;
the per-step columns add one `torch.stack` of norms the clip already computes and two 24-element copies, and the
every-10th columns reuse the `_foreach_norm` and AdamW-state kernels the per-group values used [inference]. Taking
the per-group values every step cost 0.50% of d24 throughput (351.4K / 351.8K against 349.9K / 349.8K tok/s, GH200,
alternated 25-step windows after 8 warm-up steps; every 10th step within the windows' spread, 341,290–341,471 tok/s
with or without, `7e77833`). The per-micro-batch statistics are taken without autograd: a graph through the
pinned-buffer copies would chain every step's onto the last, a leak of ~12.7 MiB per step at d24 (`1f23203`).

**Job** (`run.job`): the run is skipped if every checkpoint's evaluation exists or it diverged; else training
(`run.training`, which resumes, below), then the unevaluated checkpoints' evaluations (`run.evaluations`,
overlapped, §11), each phase in a fresh spawned process (`torch.compile`'s recompile limit and the caching allocator
are per process). A relaunch thus finishes a run's training and evaluates the checkpoints still missing an
evaluation. Runs are not re-checked against the current recipe: `67192ab`'s stale guard `run.recipe_changed`
(a recorded plan, tokenizer data key, config or stage mixture differing from the current code's) is removed.

**Checkpoints and resume.** Each is written to a temporary file and renamed, so a kill mid-save leaves the previous
one (`train/checkpoint.py`). Full state (`checkpoint-<step>.pt`: `model`, `optimizers`, model `config`,
`stage_rows`) is saved at `run.checkpoint_steps`, `round(steps·k/n)` for k = 1…n plus `repo_level_start` when
n > 1, the only checkpoints evaluated. A run with n > 1 also keeps `run.analysis_steps`' checkpoints, never
evaluated; the schedule is experimental configuration, held only in `run.py`'s constants:

| Set | Steps (base run) | Content | Count × size | Basis |
|---|---|---|---|---|
| evaluated | each tenth of 69,611, and 65,814 | full state | 11 × 5.588 GB | the evaluations of the run (§9) |
| grid | every `FULL_STATE_EVERY` = 1,000 below the last step | full state | 69 × 5.588 GB | Pythia's checkpoint every 1,000 iterations (arXiv 2304.01373 §2.4); OLMo's intermediate checkpoints at intervals of 1,000 steps (arXiv 2402.00838 §5) |
| early | 1, 2, 4, …, `EARLY_WEIGHTS_LAST` = 512 | fp32 weights and model `config` (`weights-<step>.pt`) | 10 × 2.089 GB | Pythia's log-spaced early checkpoints (arXiv 2304.01373 §2.4) |
| after the stage switch | `repo_level_start` + `AFTER_STAGE_WEIGHTS` (100, 200, 400, 800, 1,600): 65,914 … 67,414 | fp32 weights and `config` | 5 × 2.089 GB | the early set's log spacing at the context switch |

The base run's sets are disjoint: 80 full states and 15 weights-only files, 478.4 GB [arithmetic] (a full state
is the fp32 weights, AdamW's two moments of the 352,327,728 table and vector elements, Muon's momentum of
169,924,608 and NorMuon's row moments). Weights-only steps exclude every full-state step. The grid's optimizer
state gives a resume point every 1,000 steps and any 1,000-step point as a branch for an intervention, as Pythia's
intermediate optimizer state did (arXiv 2304.01373 §3.1). The early and post-switch weights are fp32: a step-1
update relative to the initial weight RMS, ≈ 0.2·lr/40 ÷ 768^−½ = 3.3e-4 [arithmetic], is below bf16's 2^−8.
A save's time falls into the next step's `step_seconds`: in the d24 rehearsal (§7), over the 8K median step, a
weights-only save added 0.84–1.02 s to it, a full state 2.18–2.25 s (4.01 s over the 32K median at the stage switch,
the 32K graph's compilation included; the resumed process's first step, compilation without a save, 3.31 s),
≈ 190 s over the base run's 95 saves, 0.07% of its training [arithmetic]. A relaunched run resumes from its newest
full-state checkpoint (weights, optimizer state, the scheduler at its step, the data position), never from a
weights-only one; a run with one checkpoint restarts from step 0. A resumed attempt's records supersede the earlier
ones for the steps it repeats (`metrics.train_steps`).

**Divergence.** A non-finite loss or gradient norm writes a `failure` record and ends training; the job returns, the
sweep continues and a rerun skips the run (`metrics.diverged`); its experiment record has `measures` null and
`diverged_at` set. Both instabilities of arXiv 2309.14322 §3.1 "appear at high learning rates in small models"; no
`peak_lr` arm diverged (§5). Any other failure fails its job: the sweep's other jobs still run (Hydra's basic
launcher runs every job of the sweep), then the sweep raises and writes no experiment records.

**Experiment records** (`ablations/records.py` `Records`, an experiment's Hydra callback, when the sweep ends):

| File | Content |
|---|---|
| `results/<experiment>.jsonl` | per run: `experiment`, `eval/suite.py` `record` (name, run-log records, evaluations), `config`, `measures` and `errors` (`docs/evaluation.md` §5), `offset_bpb` (32K code bits per byte by byte offset), `efficiency` (parameters, FLOPs per token, peak memory), `diverged_at` |
| `results/<experiment>-curves.jsonl` | per run: its train records as columns (`metrics.curves`) |

The ablation records and their status: `docs/experiments.md` §1. `results/muon_wd_timescale_20tpp.jsonl` and
`results/scalar_learning_rates.json[l]` (whose `.json` holds an analysis, `docs/experiments.md` §4) were written by
earlier code: without the output-logit cap (`model.final_logit_softcapping` null) and with `RunConfig` fields since
removed.

## 7. Hardware and throughput

**Machine.** One GH200: a Hopper GPU (sm_90, 96 GB HBM3, 97,871 MiB reported), 64 Neoverse-V2 cores, 525 GB of memory
(`nvidia-smi`, `lscpu`, `free`).

**Power and clock.** The GPU's enforced limit is 700 W, set outside the VM (`nvidia-smi -pl 900` is accepted but not
enforced; the software power cap is active in every sample of the d24 8K step). The module (Grace + Hopper + memory)
shares a 1,000 W budget and draws 888–910 W with the CPU busy [unverified: samples not kept], so no heavy CPU job runs
beside a measured run. `scripts/setup.sh` locks the SM clock at `clocks.max.sm` (1,980 MHz): the d24 8K step 3.0%
faster, d16 1.7% (`4ade225`). Under training the cap still lowers the clock to averages of 1,492–1,647 MHz across the
d24 profiles: sustained GEMMs run ~14% below their isolated speed. The lock was found off before the d24 rehearsal
(`nvidia-smi`'s locked clocks do not persist across a driver reload or reboot; how it was lost is unknown). [inference]
The unlocked rate at `104488d`, 366,864 tok/s, equals `067e60b`'s 365,466 with the softcap's −0.58% in between, so
that row and the later A/B arms were likely unlocked too (their relative gains are unaffected: both arms alike).
Re-applied mid-rehearsal, the 8K rate went from a median of 360,410 tok/s (steps 0–599) to 374,203 (steps
600–3,126; not a controlled A/B; `4ade225`'s +3.0% is); `scripts/setup.sh` re-applies the lock and is re-run after a
reboot (§9).

**Step profile** (d24, 8K, nsys; the baseline of `373b549`'s A/B): 4,355 ms of GPU time per step, 99.4% busy: GEMMs
60.5% (~680 TFLOP/s), FlashAttention-4 18.9% (34.7% at 32K; the 512-window layers at 283–336 TFLOP/s against 511–600
for full attention), memory-bound kernels 19.8%, the optimizer 0.6%. The step is GPU-bound and its GEMMs run at the
700 W cap's ceiling, 703 of 989 dense bf16 TFLOP/s (`a81410c`). Each kernel landed since, with its gain:
`docs/architecture.md` §3.

**Throughput** (tokens per second; MFU: executed model FLOPs over 989.4 TFLOP/s, `docs/evaluation.md` §5):

| Measurement | d8 8K / 32K | d16 8K / 32K | d24 8K | d24 32K |
|---|---|---|---|---|
| Baseline (the profile above) | 2.81M / 2.37M | 803,750 (MFU 0.391) / 638,405 | 338,039 (0.450) | 273,536 (0.465) |
| Master as a whole, `eeb0c53` | | | 356,259 (0.473) | 282,256 (0.483) |
| Latest A/B arm, `067e60b` (unlocked) | | | 365,466 (0.4857) | 287,476 |
| d24 rehearsal, `a4abe78`: in-loop medians of the whole run, locked from step 600 | | | 373,168 (0.498) | 298,565 (0.508) |

Later landings were timed as A/B arms only, spanning 362,972–365,466 tok/s at 8K and 286,555–287,482 at 32K
(`fb86efc`, `db1e90e`, `067e60b`). The output-logit cap (`e24333d`) adds 1.56 ms per micro-batch, 0.58% of the 8K
step (its 32K cost unmeasured; `docs/architecture.md` §3). The rehearsal (`dev-ignore`, not a recipe): the base recipe
at 5 tokens per parameter (3,307 steps of 294,912 tokens, the repository stage from 3,127, the peak-lr sweep's d24
plan) at the calibrated × 2.7193, `num_checkpoints=3`, killed after step 1,050 and resumed from step 1,000.

**Memory.** The d24 10-step peak is 71.11 GiB allocated at 8K, warm and cold (`e24333d`), of the GPU's 95.58 GiB
(97,871 MiB). A micro-batch holds whole rows of both contexts, so its size is a multiple of 32,768 tokens;
`MICRO_BATCH_TOKENS` is 3 of them, the most that fits: the step's persistent state (fp32 weights and gradients of
the 522,252,336 parameters, AdamW's two moments of the 352,327,728 outside Muon, Muon's momentum and bf16 GEMM copies
of its 169,924,608) is 7.47 GiB [arithmetic]. The allocator's peak reserve is 73.14 GiB (`results/peak_lr_d24.jsonl`
`peak_reserved_bytes`, 2.04 GiB above the allocated peak); 4 rows of 32K would scale its 65.67 GiB above the state
by 4/3, to ≈ 95.0 GiB in all [estimate: activation memory proportional to tokens], 0.5 GiB below the device before
the CUDA context, which the allocator does not count. The micro-batch also quantizes every run's batch (§2).

**Time estimate** [estimate]: 97.05B tokens at 365,466 tok/s and 5.60B at 287,476 are 73.8 + 5.4 h; with the cap's
+0.58% at 8K, 79.6 h; across the arms' span ≈ 79–80 h of training, plus the 95 checkpoint saves (§6; their time
is the next steps' `step_seconds` excess, unmeasured) and the 11 evaluated checkpoints' benchmark evaluations.

Concurrent runs and MPS sharing are not used (`docs/experiments.md` §5).

## 8. Reproducibility

**Environment.** `docker/Dockerfile` pins by digest uv 0.13.6 and NGC PyTorch 26.09 (PyTorch 2.14.0a0 with Triton,
CUDA and nvcc; torch, Triton, numpy and nvidia-ml-py come from the image and are excluded from the lock,
`pyproject.toml` `exclude-dependencies`), Rust 1.99.1 for the crates, and openai/human-eval-infilling (`88062ff`) and
nuprl/MultiPL-E (`3025a53`) by commit; the image's own FlashAttention is uninstalled so the locked `flash_attn.cute`
is imported. `pyproject.toml` pins every direct dependency at its newest release (e.g. `hydra-core==1.3.8`,
`quack-kernels==0.6.5`, `flash-attn-4==4.0.0b36`, `duckdb==2.0.0`; evalplus `26d6d00` by git revision),
`uv.lock` the rest. `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` keeps fp32 matmuls fp32
(the NGC image sets 1); `OMP_NUM_THREADS=1` (parallelism is by process); compiler and kernel caches live in
`artifacts/cache`.

**Data order.** Each corpus's packed rows are permuted once by `train/loop.py` `SEED` = 0 (NumPy `RandomState`, whose
stream is frozen across NumPy versions; `stage_batches`), which also seeds the initialization; rows go to corpora by the
loader's Weyl sequence (`mixture.mixture_rows`; `docs/corpus.md` §9). The order does not depend on the CPU
[inference].

**Determinism.** Gradients are not bitwise across processes (FA4's and the embedding backwards' atomics;
`docs/architecture.md` §4). Kernel choices do not depend on timing: QuACK runs fixed tile configurations
(`docs/architecture.md` §3), and Inductor's coordinate-descent tuning, which picks configurations by timing, stays at
PyTorch's default, off (`torch._inductor.config.coordinate_descent_tuning`, on only with
`TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1`, which neither the image nor `scripts/container.sh` sets). Python's
garbage collector, whose scans are slow, is frozen and disabled after the first step and collects only after a
checkpoint save (`train/loop.py`; karpathy/nanochat 07c4dd4).

## 9. Reproducing the run

On a fresh GH200 host with Docker and the NVIDIA driver (Lambda `gpu_1x_gh200`):

```bash
git clone https://github.com/indrajeetadityaroy9/nanochat nanochat && cd nanochat
HF_TOKEN=<read token> scripts/setup.sh
```

`scripts/setup.sh` writes the CDI spec, locks the SM clock (§7), joins the docker group, builds the `nanochat` image
(`docker build -f docker/Dockerfile -t nanochat .`) and pulls the curated corpus (`python -m corpus.publish pull`);
it is re-run after a reboot (the clock lock). After logging in again, every command runs in the image through
`scripts/container.sh` (the checkout mounted at `/workspace/nanochat`, NGC's documented run flags, an unlimited stack
hard limit for evalplus's `reliability_guard`), whose entrypoint creates `artifacts/` (§11, `paths.py`).

1. **Corpus**: the corpus's tokenizer, tokenization and packing, in the order of `docs/corpus.md` §10.
2. **Peak lr**: `scripts/container.sh python run.py experiment=peak_lr_d24` (§5; the 3 runs and their evaluations),
   writing the records `lr_multiplier` reads.
3. **Train**: `tmux new -d -s run 'scripts/container.sh python run.py experiment=pretrain'` trains both stages with
   11 evaluated checkpoints (each tenth of the 69,611 steps and step 65,814, the repository stage's start) and the
   84 analysis checkpoints (§6), evaluates the 11 with every benchmark and writes `results/pretrain.jsonl` and
   `results/pretrain-curves.jsonl`. Expected time: §7.
4. **Release**: `HF_TOKEN=<write token> scripts/release.sh <run_name> <repo>` uploads the run's full-state and
   weights-only checkpoints, `metrics.jsonl` and the diagnostics sidecar, evaluations, `.hydra/`, the corpus tokenizer
   (`artifacts/corpus/tokenizer/`, which `load_model`'s tokenizer reads) and `results/` to a
   Hugging Face model repository; `scripts/container.sh hf download <repo> --local-dir /workspace/nanochat` restores
   them where `load_model` and a resumed run read them.

Everything a run reads or writes lives in the checkout (`paths.py`): `artifacts/` (gitignored) holds `corpus/`,
`runs/<run_name>/` and `cache/`; `results/` (tracked) holds the records the paper's tables are drawn from.

## 10. Post-training

Not implemented: the repository pre-trains and evaluates a base model. Supervised fine-tuning on function-completion
exercises has the largest published effect at this scale: 180M tokens of exercises took a 1.3B model's HumanEval
pass@1 from 29% to 50.6% (arXiv 2306.11644 §2.1, §2.2, Fig. 2.1); OpenCoder-1.5B went from HumanEval+ 49.4 to 67.7
and MBPP+ 58.7 to 61.9 base to instruct (arXiv 2411.04905 Tables 6, 7; not controlled). The format would be
completion, not chat (no chat tokens, `corpus/tokenizer.py` `SPECIAL_TOKENS`), the loss on the solution and BOS alone;
candidate data `OpenCoder-LLM/opc-sft-stage2` and `opc-annealing-corpus`; preconditions: decontamination by entry
point and 10-gram overlap (arXiv 2411.04905 §4.4) and FIM examples in `corpus/pack.py`'s format in the mix.

## 11. Implementation reference

**`run.py`.** The app process plans on the meta device (`model/modeling.py` `meta_model`) and never initializes CUDA.
Its top level imports no torch, model, training or evaluation code because the workers of `eval/code_eval.py`'s
sanitize pool (a forkserver) import it as the phases' main module. The resolvers and the schema are registered only
in the app's process (`if __name__ == "__main__"`): `ablations/records.py` imports the file again as `run`, whose
registrations would fail. `run_name` omits default fields, so a field added with a default renames no run, and a
changed default would keep a run's name under a new meaning. `job`: a training phase that fails without a `failure`
record raises `ChildProcessError`; one that diverged ends the job.
`evaluations` runs the checkpoints' evaluations in fresh processes, checkpoint k + 1's started once k's has released
the GPU (`eval/suite.py` `evaluate` sets its `released` Event, `docs/evaluation.md` §1) or ended; once any started one
has failed no further evaluation starts, and `ChildProcessError` is raised once every started one has ended.
`evaluation` silences `ast.parse`'s `SyntaxWarning`s on generated code (`eval/verify.py`, `inference/`).
`conf/run.yaml` disables Hydra's job logging, whose default file `${hydra.job.name}.log` would be
`RUNS/<run_name>/run.log`, the phases' log.

**`ablations/peak_lr.py`.** `lr_multiplier` reads `results/peak_lr_d24.jsonl` in its recorded order (× 1, 2, 4) and
returns e^vertex; the fitted parabola's curvature (0.0086, §5) is positive. It imports no `run` code: the resolver
needs only the records.

**`train/loop.py`.** A step composes `optim.py` (optimizers, weight decay, clip), `schedule.py` (stage, lr),
`stages.py` (micro-batches in data order) and `model/ops/cross_entropy.py` `step_shares` (micro-batch loss weights).
`pending` holds the last step's `log` arguments (step, lr, model FLOPs, host values, layout, start and end events).

**`train/diagnostics.py`.** Groups are named `adamw[i]`, `muon[i]` in step order. The logsumexp statistics come from
`lm_loss`'s per-position outputs (`model/ops/cross_entropy.py`), which require grad. AdamW's update RMS is computed
from its state as lr·m̂/(√v̂ + eps) (`torch/optim/adamw.py`), Muon's from `Muon.update_norms`; `end_step` runs after
the optimizers stepped but before the schedulers, so each group's lr is still the step's. Host values (two pinned
fp64 buffers, alternating by step): 4 scalars (loss, `grad_norm`, logsumexp max and square sum), the step row, the
update row (measured steps only), then each row's loss sum and valid count; `record` copies the rows into the
sidecar's memory maps as float32 and reads `qk_norm_gain_max` as the step row's largest `gain_max/` value.

**`train/optim.py`.** `Muon` follows `torch/optim/_muon.py` `_single_tensor_muon` (momentum lerp, Nesterov, which is
`torch.optim.Muon`'s default and the only mode `Muon.step` implements, bf16 Newton–Schulz, decoupled weight decay,
`_adjust_lr`, imported privately); the batched `bmm`/`baddbmm` chain may round differently from per-matrix
`mm`/`addmm`. State `second_momentum_buffer` is one fp32 per row (output neuron in
`nn.Linear`'s (out, in) layout). The rescale to O's Frobenius norm (zichongli5/NorMuon `normuon.py` `normuon_update`)
lets `match_rms_adamw` scale the update as Muon's, the target of Alg. 1 line 10's rescale to RMS 0.2·lr.
`update_norms` (per shape: the parameters and their adjusted-lr update norms) excludes weight decay. `update`'s
in-place gradients are those the ops accumulate into (`model/modeling.py` `accumulated_weights`); it never
synchronizes the host.

**`train/stages.py`.** `corpus/pack.py` packs each stage's documents apart, so no document is read twice;
`PretrainingBatches` stops at an exhausted corpus, which `max_tokens_per_param` rules out for the base run; the
`peak_lr` arms at 5 tokens per parameter trained all 3,307 steps (`results/peak_lr_d24.jsonl`). `67192ab`'s
`stage_mixture` also raised on an overdraw.

**`ablations/records.py`.** The peak memory is the maximum over the `stage` record (the file-level stage's peak) and
the `end` record (the last stage's). Each run's log is read once.

**`paths.py`.** `artifacts/` is gitignored. The image runs `paths.py` before every command (a hook in NGC's
`entrypoint.d`, which its entrypoint sources before executing the command; a failure ends the container): it creates
`CORPUS`, `RUNS`, `NATIVE`, `PCH` and the cache directories the image's environment names (`CACHE_VARIABLES`), so
a fresh checkout or an emptied cache needs no setup step and code creates no directory (Hydra creates
`RUNS/<run_name>` before its job starts). `CORPUS` holds the corpus pipeline's stages (`corpus/__init__.py`), filled
by `python -m corpus.publish pull`; `results/` is tracked.

**Scripts.** `setup.sh` (Lambda's Ubuntu image): the CDI spec serves `container.sh`'s `--device nvidia.com/gpu=all`;
the docker group applies at the next login, so `sg docker` runs the script's docker commands; idempotent through the
build cache; the clock lock does not survive a reboot, so the script is re-run after one. `container.sh`:
`--ipc=host --ulimit memlock=-1 --ulimit stack=67108864` are NGC's documented flags, the stack hard limit unlimited
(`:-1`) for `evalplus/eval/utils.py`'s `setrlimit`; `-e HF_TOKEN` passes the caller's Hub token through, never stored
(the dataset is private, so `corpus.publish pull` needs a read token; `release.sh` a write token). `<run_name>` is the
directory name under `artifacts/runs`; `release.sh` uploads an explicit list of the run's files.

**`docker/Dockerfile`.**
- NGC supplies torch, Triton, the CUDA libraries, nvcc (`native.py`), numpy (the image's numba pins it below 2.5) and
  nvidia-ml-py (`pynvml`, `train/diagnostics.py`).
  The venv (`--system-site-packages`) gives every locked package the lock's version over the image's copy. Digests
  are each tag as resolved when pinned.
- The image's regular `flash_attn` package (FA2 with an FA4 beta inside) is uninstalled: it would hide
  `flash_attn.cute`. `UV_COMPILE_BYTECODE`: the container's user cannot write bytecode into `/opt/venv`;
  `UV_LINK_MODE=copy`: the uv cache is a separate build mount, which hard links cannot cross. The dependency layer
  rebuilds only when `pyproject.toml` or `uv.lock` change.
- Rust: rustup-init 1.29.2 for the build's `TARGETARCH` (arm64 or amd64), checked against its published SHA-256,
  installs toolchain 1.99.1 in the minimal profile; crates fetched on first build go to `CARGO_HOME` in
  `artifacts/cache`, since the container's user cannot write `/usr/local`.
- openai/human-eval-infilling: `read_problems` reads `<package>/../data`, outside the package, so the checkout goes on
  `sys.path` through a `.pth` file (its `setup.py` declares a console script uv rejects). Its README's "enable
  execution" step uncomments `execution.py:84` `exec(check_program, exec_globals)`; `data.py:13` names
  `HumanEval-RandomSpan.jsonl.gz`, the repository ships `HumanEval-RandomSpanInfilling.jsonl.gz` (symlinked).
  Its requirements: fire and tqdm pinned in `pyproject.toml`, numpy the image's.
- nuprl/MultiPL-E: `evaluation/src/main.py` (`eval_cpp.py`: `g++ -std=c++17`, the image's g++) and `pass_k.py`;
  dependencies (tqdm, numpy) as above.
- evalplus's `untrusted_check` (`evalplus/eval/__init__.py`) is patched in the image: its shared `progress`, `stat`
  and `details` take no lock and the parent joins the child after terminating or killing it. Unpatched, a sample
  killed at its time-out while holding the shared array's lock left the parent blocked on its read (line 274) forever:
  the d24 rehearsal's checkpoint-1102 evaluation hung on a sampled HumanEval/148 completion, and through the shared
  timed-children budget (`docs/evaluation.md` §1) the filtered search with it. The child is the arrays' only writer, so
  the parent's reads after the join see its final values; a sample stuck in one C call is killed and scored time-out
  in 5.1 s. The patch asserts each replaced source string occurs once, so another evalplus revision fails the build.
- The cache variables (`HF_HOME`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `QUACK_CACHE_DIR`,
  `FLASH_ATTENTION_CUTE_DSL_CACHE_DIR`, `CARGO_HOME`) are set under `XDG_CACHE_HOME`, the checkout's
  `artifacts/cache` (`paths.CACHE`); `90-nanochat-paths.sh` in NGC's `entrypoint.d` runs `paths.py`.
- `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` (NGC's) would run every fp32 cuBLAS matmul in TF32 (10-bit mantissa).
  `OMP_NUM_THREADS=1`: pipeline and evaluation parallelize by process (pools of `os.cpu_count()` workers, rayon).
  `HF_XET_HIGH_PERFORMANCE` saturates network and CPUs on Hub pulls. Inductor, Triton, quack, FlashAttention-4 CuTe
  and cargo caches and `XDG_CACHE_HOME` (evalplus's data releases and ground truth via appdirs) live in
  `artifacts/cache`, reused by every later container.

**`pyproject.toml`.** `lark[interegular]`: `Lark(strict=True)` checks terminal collisions
(`inference/syntax_constraint.py`); `omegaconf==2.3.1` is the newest hydra-core 1.3.8 accepts (`<2.4`). Excluded:
the image's torch, Triton, numpy and nvidia-ml-py, and evalplus's model-provider clients and transformers, which only
its code generation imports.

## References

| arXiv | Title |
|---|---|
| 1412.6980 | Adam: A Method for Stochastic Optimization |
| 1910.04209 | On the adequacy of untuned warmup for adaptive optimization |
| 2204.02311 | PaLM: Scaling Language Modeling with Pathways |
| 2304.01373 | Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling |
| 2306.11644 | Textbooks Are All You Need |
| 2308.12950 | Code Llama: Open Foundation Models for Code |
| 2309.14322 | Small-scale proxies for large-scale Transformer training instabilities |
| 2401.02954 | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism |
| 2401.14196 | DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence |
| 2402.00838 | OLMo: Accelerating the Science of Language Models |
| 2405.13698 | How to set AdamW's weight decay as you scale model and dataset size |
| 2409.12186 | Qwen2.5-Coder Technical Report |
| 2409.19913 | Scaling Optimal LR Across Token Horizons |
| 2411.04905 | OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models |
| 2502.15938 | Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs |
| 2502.16982 | Muon is Scalable for LLM Training |
| 2503.04715 | Predictable Scale: Part I, Step Law – Optimal Hyperparameter Scaling Law in Large Language Model Pre-training |
| 2505.13738 | Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training |
| 2507.20534 | Kimi K2: Open Agentic Intelligence |
| 2509.02046 | Fantastic Pretraining Optimizers and Where to Find Them |
| 2510.05491 | NorMuon: Making Muon more efficient and scalable |
| 2512.05620 | Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales |
| 2512.22382 | Completed Hyperparameter Transfer across Modules, Width, Depth, Batch & Duration |

Repositories: `karpathy/nanochat` (92d63d4, 07c4dd4, `dev/LOG.md` 2026-01-10); `KellerJordan/modded-nanogpt` (PR #144);
`KellerJordan/Muon`;
`NVIDIA/Megatron-LM` (PR 5395); `openai/human-eval-infilling`, `nuprl/MultiPL-E`, `evalplus/evalplus` (§8's commits);
`OpenCoder-LLM/opc-sft-stage2`, `OpenCoder-LLM/opc-annealing-corpus` (datasets).
