# Training

How the base model (d24) is trained on one GH200: stages, plan, optimizers, schedule, the peak learning-rate
calibration, the run app and its records, hardware and throughput, reproducibility and the commands.
The code is the specification. Markers: [arithmetic] derived from code numbers; [inference] reasoning not measured;
[estimate]; [unverified] a figure without a record. Measurements give their setting (GH200 at 700 W, exclusive A/B).

## Code

|File|Section|
|---|---|
|`run.py` (`RunConfig`, `run_name`, `plan_of`, `checkpoint_steps`, `analysis_steps`, `training`, `evaluations`, `job`)|§2, §5, §6|
|`conf/run.yaml`, `conf/experiment/pretrain.yaml`|§5, §6, §9|
|`train/schedule.py` (`plan`, `RunPlan`, `WARMUP_STEPS`, `lr_lambda`)|§2, §4|
|`train/stages.py` (`stage_batches`, `max_tokens_per_param`)|§1, §8|
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
`MICRO_BATCH_TOKENS` = 98,304 tokens (12 rows at 8K, 3 at 32K; the size: §7). The loss is next-token cross entropy;
the packs' FIM pieces are read as FIM examples (`docs/corpus.md` §8). At the repository stage's first step the loop
switches to the 32K context and recompiles the model.

`train/stages.py` `stage_batches` permutes each corpus's packed rows once by the seed and draws them by the stage's
weights (`mixture.weights`). `max_tokens_per_param` is the tokens per non-embedding parameter of the most whole
micro-batches bisection finds at which no stage overdraws a corpus (not necessarily the largest such count: overdraw
is not monotone, since the batch steps up with the count). A run's `tokens_per_param` defaults to it, so no document
is read twice; the `peak_lr` arms at 5 tokens per parameter trained all 3,307 steps (`results/peak_lr_d24.jsonl`).

| | The base model | Reference code models |
|---|---|---|
| Context stages | 8K files, then the last 5.45% of steps at 32K on repositories | Qwen2.5-Coder the same 8K → 32K repository stage (arXiv 2409.12186 §3.2); DeepSeek-Coder repository-ordered files (arXiv 2401.14196 §2.2), then 1,000 steps at 16K (§3.6); Code Llama a 16,384-token long-context stage (arXiv 2308.12950 §2.4) |

## 2. Plan

`train/schedule.py` `plan(model, tokens_per_param)`, with N the non-embedding parameters (`num_non_embedding_params`:
decoder layers and `lm_head`, what the token budget counts) and M DeepSeek LLM's non-embedding FLOPs per token at the
file-level context (`non_embedding_flops_per_token`, Eq. 2, arXiv 2401.02954 §3.2; both `docs/architecture.md` §1):

- tokens D = `round(tokens_per_param · N)`, compute C = M·D;
- batch B = 0.2920·C^0.3271 tokens rounded to the nearest whole micro-batch, steps = D // B (the remainder untrained,
  so the plan's `tokens_per_param` = steps·B/N falls below the requested one);
- `repo_level_start` = steps − `round(REPO_LEVEL_SHARE · steps)`;
- the law's peak lr η = 0.3118·C^−0.1250 (§3.1 Eq. 1), times `learning_rate_scale` (`run.py` `plan_of`; §5).

The base run takes the d24 plan at `max_tokens_per_param`, the `peak_lr` arms at 5. Recomputed on the CPU from master,
the corpus's packs and `results/peak_lr_d24.jsonl` (peak lr at the base multiplier 2.7192600, §5; weight decay by §3's
law); the d16 and d8 rows, the ablations' settings (`docs/experiments.md` §1), by the earlier multi-depth code:

| Run | Tokens/param | Micro-batches | Steps | Repo start | Tokens | Law lr | Peak lr | Weight decay |
|---|---|---|---|---|---|---|---|---|
| d24 base (`max_tokens_per_param` 528.122) | 528.115 | 15 (1,474,560 tokens) | 69,874 | 66,063 | 103.03B (97.41B at 8K, 5.62B at 32K) | 8.6250e-4 | 2.3454e-3 | 0.15322 |
| d24 at `mixture.MAX_STEPS` | 525.893 | 15 | 69,580 | 65,785 | 102.60B | 8.6295e-4 | 2.3466e-3 | 0.15345 |
| d16 at 20 | 19.998 | 3 | 4,552 | 4,304 | 1.342B | 1.6686e-3 | 4.5373e-3 | 0.2166 |
| d8 at 20 | 19.997 | 1 | 2,987 | 2,824 | 0.294B | 2.4438e-3 | 6.6455e-3 | 0.2253 |
| d8 at 200 | 199.993 | 2 | 14,937 | 14,122 | 2.937B | 1.8326e-3 | 4.9834e-3 | 0.2022 |
| d24 at 5 (`peak_lr`) | 4.999 | 3 | 3,307 | 3,127 | 0.975B | 1.5443e-3 | | |

The base row: `max_tokens_per_param` = 528.122, the tokens of 1,048,124 whole micro-batches per N, is the plan's
input; the run trains steps·B = 103,033,405,440 tokens, 528.115 per N, the `RunPlan.tokens_per_param` of the base
run's `run` record (with 69,874 steps, `repo_level_start` 66,063 and lr 2.3453539656816073e-3) [arithmetic]. The
stages' token budgets are sized by `mixture.MAX_STEPS` (`docs/corpus.md` §7), the second d24 row; the base plan on the
decontaminated packs (`docs/corpus.md` §3) is 0.42% longer.

**FLOP count.** The plan uses Eq. 2's M (full attention at the 8K context in every layer), the definition Eq. 1's
constants were fit with (§3.2: "M includes the computational overhead of attention operation"). The earlier
per-window count (6 per decoder-layer matrix weight + 12·d·window per layer, arXiv 2204.02311 App. B with the attended
window as context; the earlier `peak_lr` records' `flops_per_token` 1,557,467,136) is 1.82× lower at d24/8K (on the
second d24 row's plan: 12.06 instead of 14.67 micro-batches, lr ×1/0.928). The batch is not ablated: Muon's loss is
flat across a 2× batch at 8× Chinchilla (arXiv 2509.02046 App. C.9 Table 85: 130M, batch 128 and 256 both 3.240, 512
3.257).

## 3. Optimizers

`train/optim.py` `optimizers`, `update`:

| Group | Parameters | Optimizer | Weight decay |
|---|---|---|---|
| tables | `embed_tokens`, `lm_head`, the 12 value-embedding tables | AdamW (fused), β = (0.9, 0.95), PyTorch's eps | λ |
| vectors | QK-norm gains, `resid_lambdas`, `x0_lambdas` (every 1-D parameter) | AdamW, same | 0 |
| matrices | q/k/v/o, up/down, value-embedding gates (every other 2-D parameter) | `Muon` (NorMuon) | λ |

- **Muon** (`Muon.step`): `torch.optim.Muon`'s update (momentum 0.95, Nesterov, 5 bf16 Newton–Schulz steps with
  PyTorch's coefficients, decoupled weight decay), the Newton–Schulz batched over same-shape matrices (one `baddbmm`
  chain per shape instead of per-matrix `mm`/`addmm`, which may round differently); NorMuon (arXiv 2510.05491 Alg. 1
  lines 7–9): each row of the orthogonalized update O divided by √v + 1e-10, v an EMA (β₂ 0.95) of the row's mean
  square, then rescaled to O's Frobenius norm (zichongli5/NorMuon `normuon_update`), so that the step size
  `match_rms_adamw`, lr·0.2·√max(fan_out, fan_in) (Moonlight, arXiv 2502.16982 §2.2), scales it as Muon's.
- **One lr and one weight decay** for every group, as Moonlight shares "learning rate η, weight decay λ" among matrix
  and non-matrix parameters (§2.2). Per-group candidates were not adopted (`docs/experiments.md` §4).
- **Weight decay** by the τ law (`tau_weight_decay`): the EMA timescale 1/(η·λ) is τ* = 1.084·TPP^−0.527 of the run's
  steps (Power Lines, arXiv 2505.13738 Finding 3, App. E.3.1), so λ = 1/(η·τ*·steps), 0.15322 for the base run. The
  law is fit on AdamW over TPP 20–1280 (the base run's 528 inside). Muon takes the same law: under μP a fixed
  independent weight decay stays optimal across TPP for both Adam and Muon (arXiv 2512.05620 App. H.5, IsoFLOP sweeps
  whose optima lie at 6.6–13.3 TPP), and Muon's only scaling-sensitive hyperparameter is its lr (arXiv 2509.02046 §3.3
  Table 4). The Muon-specific timescale runs: `docs/experiments.md` §4.
- **Clipping**: the accumulated gradient's global norm clipped to `MAX_GRAD_NORM` = 1.0 before both optimizers
  (DeepSeek LLM §2.3). Adam is invariant to the gradient's scale (arXiv 1412.6980 §2.1) and Newton–Schulz divides its
  input by its norm (`torch/optim/_muon.py` `_zeropower_via_newtonschulz`), so under a slowly varying gradient scale
  the clip damps only spikes [inference]. The largest norm in the earlier recipe's `peak_lr` runs is 10.76 (d8 at 200,
  ×1, step 14,202; that curves file is not on master), far below the ~5e7–2e11 at which Megatron-LM PR 5395 reports a
  1.0 clip stalling training (an SFT of Qwen3.5-35B-A3B; Muon and AdamW alike: the clipped gradients fall below the
  optimizers' eps floors).

| | The base model | Reference code models |
|---|---|---|
| Optimizer | NorMuon on matrices, AdamW on tables and vectors | AdamW, β = (0.9, 0.95) (DeepSeek LLM §2.3, followed by DeepSeek-Coder §3.4; Code Llama §2.6) |
| Peak lr | the law × the measured multiplier (§5), peak of linear decay to zero | the law's lr held for 80% of tokens, then 31.6% and 10% of it (DeepSeek LLM §2.3; DeepSeek-Coder §3.4) |
| Weight decay | τ law, 0.153 | 0.1 (DeepSeek LLM §2.3) |
| Gradient clip | 1.0 | 1.0 (DeepSeek LLM §2.3) |

**NorMuon's adoption.** A screen at d16 at 20 tokens per parameter, one seed per arm paired with base seed 0, at ×1
(the law's lr, below the base multiplier) [unverified: no record; the screen's code is not on master]: NorMuon's
training loss over the last 50% of steps −0.0029 below the base's (negative in every hundred-step window), every
bits-per-byte measure better by 0.0003–0.0013. One seed cannot resolve that: base seeds 1 and 0 differ by
0.0014–0.0027 bpb and the treatment effect's seed spread is unestimated. NorMuon is adopted on the literature: 11.31%
over Muon at 1.1B (arXiv 2510.05491 abstract); Adam's final loss in 21.74% / 13.91% fewer steps at 1.1B / 5.4B, Muon's
10.43% / 6.08% (arXiv 2510.05491 §4.1.2 Table 1). A d24 check (200 steps, the earlier recipe's ×2.797, seed 0, 8K;
alone on the GPU, not an exclusive A/B): last-20 loss 2.2946 against 2.2967, largest gradient norm after step 40 1.153
against 1.387, no spike, −0.08% tok/s, equal peak (77.5 GiB). The screen's other arms: `docs/experiments.md` §4.

## 4. Schedule

`train/schedule.py` `lr_lambda`, every group's `LambdaLR` multiplier:

- **Warmup**: linear over `WARMUP_STEPS` = 2/(1 − β₂) = 40 steps (arXiv 1910.04209 §4.3, Eq. 14: "linear warmup
  over τ = 2·(1−β2)^−1 iterations"); step s trains at (s + 1)/40. In the earlier recipe's 11 `peak_lr` runs (×1 to ×8,
  §5) the largest gradient norm of the first 40 steps is 2.46–3.00 at d8 and 3.21–4.07 at d24, and none diverged. No
  paper fixes a warmup for Muon.
- **Decay**: linear to zero, (steps − s)/(steps − 40), through the repository stage: linear decay to zero is "around
  0.84% better than the optimal WSD" (610M, 80 TPP), its gap to a 10× decay grows with tokens per parameter, and its
  gain over cosine decay to zero is small but consistent across peak lrs (arXiv 2502.15938 §4.2, Figs. 6 and 8, Table
  1's caption). DeepSeek-Coder instead holds its final lr for long context (arXiv 2401.14196 §3.6); Qwen2.5-Coder
  states no schedule for its repository stage.
- **Repository stage**: its first step at 3,811/69,834 = 5.46% of peak [arithmetic].
- **Last step**: 1/(steps − 40) of peak, not 0: 1.432e-5 at 69,874 steps [arithmetic], ~70× below arXiv 2502.15938's
  smallest nonzero minimum, 0.1% of peak (App. C.7).

## 5. Peak learning-rate calibration

**Varied.** The multiplier of DeepSeek LLM's law η = 0.3118·C^−0.125 (arXiv 2401.02954 §3.1 Eq. 1), fit for AdamW
under a multi-step schedule (§2.3) at compute below the base model's, applied here to Muon (RMS-matched) as the peak
of linear decay to zero; at a fixed model it falls with tokens as D^−0.125 [arithmetic]. The transfer rules disagree
on how the optimum moves with the horizon: Complete^(d)P's AdamW SDE rule η ∝ √(m_B/m_D) (arXiv 2512.22382 §2.1
Table 1) lowers it, as arXiv 2409.19913 §3.2 Eq. 1 and the law do; Step-Law η = 1.79·N^−0.713·D^0.307, fit over AdamW
models (arXiv 2503.04715 §1 Eq. 1), raises it. The multiplier is measured at d24 itself (standard parameterization,
no width or depth transfer) over 5 tokens per parameter and carried to the base plan by the law, whose η(C) and B(C)
were fit jointly (arXiv 2401.02954 §3.1). That it carries is an assumption [inference]: the law was fit at
compute-optimal allocations, while here the model is fixed and its tokens grow 105×; its one measurement, the earlier
recipe's d8 (below): the optimal multiplier changed ×0.9105 over 40× the tokens (the optimum ×0.574, the law ×0.6306).

**Arms.** × 1, 2, 4 at d24 at 5; seed 0; 3 runs under the current recipe (corpus tokenizer and mixture, cap,
`generate` "parse"): `conf/experiment/peak_lr_d24.yaml`, records `results/peak_lr_d24.jsonl` and its `-curves.jsonl`
(§6). At 5 tokens per parameter a stage draws too few rows to read its supply corpora whole, so their shares are the
base run's (`corpus/mixture.py` `weights`; each arm's `run` and `stage` records hold the drawn mixture): Kaggle
notebooks 0.0156 and CodeXGLUE Python 0.0006 at 8K, Codeforces CoTs 0.0554 at 32K.

**Rule** (`ablations/peak_lr.py` `lr_multiplier`): the base multiplier is the vertex of the parabola through the three
runs' code bits per byte against u = ln(multiplier) (`metrics.parabola_vertex`, `docs/evaluation.md` §5; arXiv
2409.19913 §3.1 fits quadratics in log lr); the base lr is it times the law's lr on the base plan. The recorded
minimum is interior (× 2), the curvature positive and no run diverged, so the vertex lies inside the grid.

**Result** (`results/peak_lr_d24.jsonl`; 3,307 steps of 294,912 tokens; bits per byte; no run diverged). On 8K code
the vertex is × 2.7193 (curvature 0.0086): the base multiplier 2.7192600, 2.8% below the earlier recipe's 2.7970369
[arithmetic], peak lr 2.3454e-3 and weight decay 0.15322 on the base plan (§2). The best arm (× 2) beats the law (× 1)
by 0.0078; × 4 is 0.0005 behind. The same fit on the other likelihood measures (`metrics.measures`; the evaluations'
`8k`/`32k` `bpb` groups, `docs/evaluation.md` §2):

| Measure | × 1 | × 2 | × 4 | Lowest | Vertex |
|---|---|---|---|---|---|
| code, 8K (the rule) | 0.41322 | 0.40544 | 0.40591 | × 2 | × 2.719 |
| code, 32K | 0.40859 | 0.40131 | 0.40227 | × 2 | × 2.610 |
| code, 8K and 32K by the base run's stage tokens (97.41B, 5.62B) | | | | | × 2.713 |
| repositories beyond 8K (`repo_beyond_8k`) | 0.36366 | 0.35736 | 0.35858 | × 2 | × 2.527 |
| benchmark solutions (`solutions`; 706 rows, standard error 0.0090) | 0.49200 | 0.48710 | 0.48498 | × 4 | × 4.782 |
| all, 8K / 32K | 0.49386 / 0.49250 | 0.48499 / 0.48403 | 0.48533 / 0.48479 | × 2 | × 2.756 / × 2.672 |
| math, 8K / 32K | 0.53770 / 0.53733 | 0.52497 / 0.52459 | 0.52556 / 0.52518 | × 2 | × 2.743 / × 2.744 |
| text, 8K / 32K | 1.07390 / 1.07466 | 1.05797 / 1.05881 | 1.05734 / 1.05821 | × 4 | × 2.910 / × 2.906 |

**Precision and robustness** (each evaluation's `errors` with `row_clusters`, `docs/evaluation.md` §2, §5: val rows
clustered by document, corpora as strata; the × 2 arm): 8K code 0.0022 (49,765 rows, 44,829 clusters), 32K code
0.0157 (2,667 rows, 2,255 clusters; 93% of its variance one 58.8M-token repository), 8K all 0.0020. The arms score
identical rows, so their differences are paired (arXiv 2411.00640 §4.2; `metrics.mixture_bpb_summary` of the per-row
bit differences [arithmetic]): 8K code × 1 − × 2 = +0.00778 (standard error 0.00003), × 4 − × 2 = +0.00047 (0.00002).
The val sample resolves every arm; it does not measure training noise (one seed per arm): two seeds of one d16 arm
differ by 0.0019 code bpb (`results/muon_wd_timescale_20tpp.jsonl`, τ × ½), four times the × 2–× 4 gap. The measure
brackets a flat optimum over × 2–× 4; × 2.7193 is a point estimate without an interval. The mixture measures'
vertices span × 2.53–2.91: −7% to +7% of the base lr, ≤ 0.00005 bpb at the fitted curvature [arithmetic]. Only the
solutions measure falls through × 4, but its × 2–× 4 gap (0.0021) is 1.5 paired standard errors (0.0014); at the code
fit's curvature its × 4.78 would cost 0.0027 code bpb [arithmetic]. The parse measure's generations (single seed,
greedy): HumanEval+ base pass@1 0.067 / 0.104 / 0.110 (standard error 0.020–0.025), MBPP+ 0.114 / 0.132 / 0.127
(0.016–0.017), × 2's paired differences at most 2.1 standard errors (HumanEval+ against × 1, exact sign test p = 0.07);
MBPP+ parse rate 0.825 / 0.894 / 0.775 (× 2 parses 70 problems × 4 does not, × 4 25 that × 2 does not; p = 4e-6),
while two seeds of one D7 arm differ more (pooled unparsed 0.293 and 0.133, `docs/experiments.md` §1). The rule reads
none of them.

**Dynamics** (`results/peak_lr_d24-curves.jsonl`, the arms' `train` records; × 1 / × 2 / × 4):
- training loss (trailing 50-step means): × 4 leads until step 468, then trails × 2 (+0.011 at step 1,000, +0.002 at
  the end); × 2 − × 1 stays within −0.020 to −0.024 from step 1,500; final 1.230 / 1.208 / 1.210;
- clipping: 41 / 35 / 25 steps above `MAX_GRAD_NORM`, the last at steps 45 / 40 / 26; median gradient norm after
  step 40 0.181 / 0.131 / 0.110; every loss and norm finite;
- 31 loss spikes in the file-level stage (a step above the previous 50-step mean by 0.15) at the same steps in all
  three arms (× 2 and × 4 have 3 and 5 more) and of the same size (+0.187–0.189 on average): batches of the fixed data
  order, not the lr; Stack-Edu Python's own loss exceeds its previous 50-step mean by > 0.3 nats in 16 of them;
- largest logsumexp 16.6 / 17.9 / 19.1 (steps 2,531 / 1,129 / 1,344) and largest QK-norm gain 1.69 / 1.93 / 2.25,
  the gains within 2% of it from steps 1,860 / 1,839 / 2,081; parameter norm of the Muon group at the end (step
  3,300) 377 / 719 / 1,426, roughly ∝ lr, as the τ law's fixed η·λ implies [inference];
- the repository stage (step 3,127, lr at 5.5% of peak): the mean loss of its first 20 steps is 0.011–0.012 below that
  of the last 20 file-level steps in each arm;
- 361–362K / 288K tok/s at 8K / 32K, MFU 0.481–0.483, peak 71.10 GiB, 2,739–2,774 s of training per arm, independent
  of the lr; about 10,320 s for the three arms with their evaluations (668–697 s per parse-mode evaluation
  [unverified: no record]).

**Adopted** for the base run (§6, §9): `learning_rate_scale` = `lr_multiplier` = 2.7192599564485445. Its absolute
peak lr lies between the × 1 and × 2 arms' (1.5443e-3, 3.0886e-3): the law lowers the lr with the base plan's compute.
`conf/run.yaml` sets every run's `learning_rate_scale` to the resolver `${lr_multiplier:}`, which fits the records at
launch (no stored copy); the `peak_lr_d24` experiment sets the scale per arm.

**Earlier recipe** (`results/peak_lr.jsonl` of the earlier recipe, not on master; code bits per byte; no run
diverged). 11 arms (× 1–8 at d8 at 5 and at 200, × 1–4 at d24 at 5) under the reference tokenizer's dataset, no
output-logit cap, nanochat's per-window FLOP count for the law's lr (Eq. 2's M, §2, lowers it by 1/1.1186 at d8 and
1/1.0776 at d24, hence the re-based vertices) and an earlier mixture (eleven corpora at 8K; at 32K file-level Python
and C++ code, not repositories, over the last 41 of 746, 815 of 14,937 and 180 of 3,307 steps).

| Setting | Plan | × 1 | × 2 | × 4 | × 8 | Vertex (recorded lr) | Curvature | Re-based |
|---|---|---|---|---|---|---|---|---|
| d8 at 5 | 746 steps of 98,304 tokens | 0.89491 | 0.78969 | 0.76960 | 0.81791 | × 3.4669 | 0.0712 | × 3.8781 |
| d8 at 200 | 14,937 steps of 196,608 | 0.52774 | 0.51750 | 0.51592 | 0.52431 | × 3.1565 | 0.0104 | × 3.5309 |
| d24 at 5 | 3,307 steps of 294,912 | 0.44230 | 0.43787 | 0.44201 | | × 2.0235 | 0.0089 | × 2.1804 |

The best arm beat the law by 0.1253, 0.0118 and 0.0044. r = 0.9105 (predicted: SDE rule 0.3544, Step-Law 4.9231; law
fall 0.6306; horizon exponent −0.0254, −0.1193 net of the batch; m_B,d8 = 2, m_D,d8 = 40.05): the optimum barely
moved with the horizon, against both rules. The d8-transfer estimator, vertex(d24 at 5)·m_D^h·√m_B (h the net horizon
exponent −0.1193, m_D = 105.25 and m_B = 5 the base plan's token and batch ratios over d24 at 5), gave the multiplier
2.7970369 (peak lr 2.4136e-3 on that recipe's base plan), the multiplier of the ablations (`docs/experiments.md` §1).

## 6. Runs and records

Every run is a `run.py` `RunConfig` (Hydra), trained by `train/loop.py` `train` and evaluated by `eval/suite.py`
`evaluate` (`docs/evaluation.md` §1). The model is d24 (`docs/architecture.md` §1), its initialization and data order
seeded by `train/loop.py` `SEED` = 0. The fields' defaults are the base recipe:

| Field | Default | Meaning |
|---|---|---|
| `tokens_per_param` | None | the data's `max_tokens_per_param` (§1) |
| `learning_rate_scale` | 1.0 | times the law's lr; `conf/run.yaml` sets `${lr_multiplier:}` (§5) |
| `num_checkpoints` | 1 | evenly spaced evaluated checkpoints, the last the final model; > 1 adds `repo_level_start` and the analysis checkpoints |
| `generate` | "parse" | `evaluate`'s mode: "parse" (`peak_lr`) or "benchmarks" (base) |

The experiments: `python run.py experiment=pretrain` (`num_checkpoints` 10, `generate` "benchmarks") and
`python run.py experiment=peak_lr_d24` (§5). **Names** (`run_name`): the fields that differ from `RunConfig()`,
sorted, `k=v` joined by `,`; the base run's directory is `artifacts/runs/<run_name>/` (run log, diagnostics,
checkpoints, evaluations, composed config) with `<run_name>` =
`generate=benchmarks,learning_rate_scale=2.7192599564485445,num_checkpoints=10`.

**Run log** (`train/loop.py` `train`; every record carries `run_name`, `attempt` (the 0-based count of earlier `run`
records), `kind`, `time`, `step`):

| `kind` | When | Fields |
|---|---|---|
| `run` | each process start | `step` (the first step it trains), `plan` (`schedule.RunPlan`), `config` (the `RunConfig`), `data_key` (the tokenizer's), `model` (`ModelConfig` at the stage of the starting weights: the file-level stage, or the resumed checkpoint's), `num_parameters`, `flops_per_token` (M), `model_flops_per_token` (PaLM's nominal count at that stage, `docs/evaluation.md` §5), `learning_rate` and `weight_decay` (`{"adamw": [...], "muon": [...]}`, one value per param group: the groups' `initial_lr` and `weight_decay`), `stage` (the first step's stage with its `mixture`) |
| `train` | each step (0-based) | `num_input_tokens_seen` = (step + 1) · batch size (row positions, padding included), `loss`, `learning_rate` (the step's, `LambdaLR`'s), `grad_norm` (before clipping), `step_seconds`, `tokens_per_second`, executed `model_flops`, `mfu`; the diagnostics (below): `max_logsumexp`, `mean_square_logsumexp`, `qk_norm_gain_max`, `corpus_loss` (a corpus with no rows in the step is omitted), and every 10th step the NVML fields |
| `stage` | the repository stage's first step | `stage` with its `mixture`, `model_flops_per_token` (32K), `peak_memory_bytes` and `peak_reserved_bytes` (`torch.cuda.max_memory_allocated`/`max_memory_reserved` since the process start: the file-level stage, or its part after a resume); both peaks are reset after this record |
| `end` | after the last step | `peak_memory_bytes`, `peak_reserved_bytes` since the reset (the repository stage) or the process start |
| `failure` | a non-finite loss or gradient norm | only `step`; training then stops |

A step's `loss` is the mean next-token loss in nats over its valid labels (Σ loss / Σ labels pooled over the
micro-batches; `corpus/loader.py` `IGNORE_INDEX` marks padding and a BOS that is no target), in fp32 on the capped
logits, from the same `lm_loss` call that gives the gradient. A step's time runs between CUDA events at the previous
step's end and its own, so a process's first step and the stage switch's include compilation, and a step after a save
the save; `tokens_per_second` is the step's row positions (padding included) over that time.

**Diagnostics** (`train/diagnostics.py`). The `train` record's per-corpus loss; the capped logits' logsumexp maximum
and mean square (PaLM's z-loss quantity, arXiv 2204.02311 §5; at most c + ln V = 25.4 under the cap [arithmetic]);
the largest QK-norm gain (q and k RMS-normed per head, the scores scaled by 1/√d: a head's logits are at most
√d·‖g_q‖∞·‖g_k‖∞ [arithmetic]; under Muon, Kimi K2's attention logits grew past 1000, arXiv 2507.20534 §2.1 Fig. 2);
every `UPDATE_STATS_EVERY` = 10th step NVML telemetry, the run's only GPU telemetry, relating `step_seconds`
to the 700 W cap and the clock lock (§7): `gpu_power_watts` (a 1 s running average on Hopper, NVML API reference),
`gpu_sm_clock_mhz`, `gpu_temperature_c` (the die), `gpu_memory_used_bytes` (the device's, every process included).
Per tensor and per layer, a float32 sidecar (`diagnostics.json` names the columns and each tensor's group and layer):

| File | Row | Columns |
|---|---|---|
| `diagnostics-step.npy` | step s, every step (69,874 × 316 for the base run) | `grad_norm/<tensor>`: each of the 220 parameter tensors' gradient L2 norm before clipping, in `named_parameters` order; `gain_max/<tensor>`: each layer's q and k QK-norm gain ‖g‖∞ after the step (48); `resid_lambdas/<i>`, `x0_lambdas/<i>` after the step (24 each) |
| `diagnostics-update.npy` | step 10·r, every `UPDATE_STATS_EVERY` = 10th step (6,988 × 440) | `param_norm/<tensor>`: L2 norm after the step; `update_rms/<tensor>`: RMS of lr × update without weight decay (AdamW's lr·m̂/(√v̂ + eps) from its state, NorMuon's adjusted-lr update) |

Disk for the base run: 88.3 MB and 12.3 MB [arithmetic]. Taking the per-group values every step cost 0.50% of d24
throughput (351.4K / 351.8K against 349.9K / 349.8K tok/s, GH200, alternated 25-step windows after 8 warm-up steps);
every 10th step is within the windows' spread (30-step runs 341,290–341,471 tok/s with or without them).

**Run checks** (the base run's `train` records and sidecars through step 20,147, read on the CPU; means over the 500
steps ending at the step unless stated). Steps 0–20,146 are contiguous in one attempt; `learning_rate` equals §4's
schedule at every step (largest relative difference 0); `num_input_tokens_seen` = (step + 1) · 1,474,560; no loss or
gradient norm is non-finite. Every train document is read once (`docs/corpus.md` §8), so a step's loss is on text no
earlier update saw, an online held-out loss [inference].

| Step | Loss | sd over steps | lr | Gradient norm | QK-norm gain max | logsumexp max |
|---|---|---|---|---|---|---|
| 1,000 | 1.3823 | 0.055 | 2.32e-3 | 0.125 | 1.62 | 17.40 |
| 2,000 | 1.2233 | 0.025 | 2.29e-3 | 0.136 | 1.86 | 18.33 |
| 5,000 | 1.1266 | 0.021 | 2.19e-3 | 0.192 | 2.46 | 18.75 |
| 10,000 | 1.0783 | 0.022 | 2.02e-3 | 0.279 | 3.21 | 18.75 |
| 15,000 | 1.0561 | 0.022 | 1.85e-3 | 0.306 | 3.62 | 18.73 |
| 20,147 | 1.0402 | 0.021 | 1.68e-3 | 0.324 | 3.84 | 18.69 |

- **Loss.** Each 1,000-step block's mean is below the previous one's; the drop per block shrank from 0.066 (steps
  2K–3K) to 0.002–0.004 (15K–20K). Every corpus's loss fell between steps 10K and 20,147, from −2.2% (ClimbMix) to
  −6.9% (UltraData-Math L3 QA); the synthetic code and mathematics sources fastest, Stack-Edu and web text slowest.
- **Loss curve.** The scaling law with LR annealing, L = L₀ + A·S₁^−α − C·S₂, S₁ the summed lr and S₂ the summed
  annealing momentum mᵢ = λ·mᵢ₋₁ + (ηᵢ₋₁ − ηᵢ) (arXiv 2408.11029 §1 Eq. 1, §3.2 Eq. 5), the warmup counted at the
  peak lr as the paper counts it (§3.4), fitted to the window means (η in units of the peak; α on a grid, L₀, A and C
  by least squares, where the paper minimizes a Huber loss on the log loss, §3.4 Eq. 6). Fitted before step 15,000,
  it predicts steps 15,250–19,750 with mean error (predicted − observed) −0.0020, every window low (largest 0.0031),
  a power law in steps +0.0052, every window high (0.0081); fitted through step 20,147, it predicted the windows of
  steps 20,147–23,147 with mean error −0.0008, mean absolute 0.0011 (largest 0.0017). It puts the file-level stage's
  last step (66,062) at 0.940–0.941 for λ = 0.99–0.999 [estimate],
  the power law in steps at 1.011. Fits ending earlier projected lower (0.871–0.876 ending at step 8K, 0.924 at 15K)
  and under-predicted their holdouts, so the law's extrapolation has run low [inference]; S₂ also overstates the drop
  as η nears 0 (§6.3; η at step 66,062 is 5.5% of the peak). The paper fits validation loss, this the online held-out
  training loss.
- **Stability.** No step after 100 was clipped (gradient norm at most 0.75 against `MAX_GRAD_NORM` = 1.0); 2 steps
  exceeded twice the trailing 500 steps' median gradient norm (largest 2.47×, step 16,305) and 3 losses the trailing
  200 steps' mean by 4 standard deviations, none by 6 (largest 5.7, step 12,257). The logsumexp maximum has stayed at
  18.6–18.8 since step 3K (logits capped at c = 15), its mean square near 167. The largest QK-norm gain (layer 0's k,
  3.85; median over the 48 gains 2.22) grows ever slower: +0.117, +0.113 and +0.037 per 1,000 steps over steps
  2K–3K, 9K–10K and 19,147–20,147. `resid_lambdas` span 0.149–1.884 and `x0_lambdas` −0.404–1.257, the largest change
  since step 10K 0.50.
- **Throughput.** 381K tok/s and MFU 0.508 in every window since step 2K; step time median 3.866 s and 99th
  percentile 3.906 s, the 21 steps above 1.2× the median the checkpoint saves (largest 7.65 s, step 13,975); 671 W,
  1,398–1,399 MHz, 66–67 °C. `gpu_memory_used_bytes` rose from 75.0 to 80.0 GiB: `nvidia-smi` reported 80,848 MiB
  used of which the process 75,704, the difference about the 5,135 MiB of page cache in the HBM's NUMA node (node 1's
  `FilePages`) [inference].

**Optimizer state.** Decoupled weight decay (§3) drives a normalized layer's gradient-to-weight ratio toward
√(2λ/η_t), so a decaying η raises the gradient norm and lowers the weight norm (arXiv 2506.02285 §3 Eq. 1, §3.1
Eq. 2); with Muon's update RMS proportional to η the same steady state puts a matrix's norm ∝ √η [inference].
Parameter norms (L2 over each group's tensors, `diagnostics-update.npy`):

| Group | Step 2,000 | 10,000 | 20,140 | 10K → 20,140 |
|---|---|---|---|---|
| `q_proj` | 244.3 | 267.9 | 248.7 | −7.2% |
| `k_proj` | 241.3 | 260.9 | 241.4 | −7.5% |
| `v_proj` | 286.1 | 244.1 | 217.6 | −10.9% |
| `o_proj` | 291.3 | 323.6 | 302.0 | −6.7% |
| `up_proj` | 521.2 | 622.2 | 596.3 | −4.2% |
| `down_proj` | 531.4 | 580.2 | 531.4 | −8.4% |
| `embed_tokens` | 370.7 | 389.1 | 356.1 | −8.5% |
| `lm_head` | 415.8 | 434.8 | 423.6 | −2.6% |
| value embeddings | 1,027.1 | 1,418.3 | 1,349.4 | −4.9% |
| `ve_gate` | 15.7 | 26.3 | 36.5 | +38.9% |
| q / k QK-norm gains (no weight decay) | 77.2 / 76.6 | 86.7 / 85.5 | 91.9 / 90.0 | +6.0% / +5.3% |

From step 10K to 20,140 η fell to 0.830 of itself, whose square root is −8.9%; the tables and the attention and MLP
matrices shrank 2.6–10.9%. Over the 144 attention and MLP matrices (medians of 500-step means), update RMS (without
weight decay) over η stays within 0.1752–0.1760 from step 5K to 20,147: the applied step tracks the schedule's lr.
Per matrix the gradient-to-weight ratio rose by a median 1.26× from step 5K to 20,147 (gradient norm 1.17×, weight
norm 0.91×), against 1.14× from η's change alone: the gradient norm rises where the steady state holds it level
[inference], and the paper's runs show it steepening as η nears 0. The rise is uneven across layers: since step 10K
the MLP down projections of layers 12 and 14 grew 12× and 6×, to 5.9e-2 and 1.1e-2 (the 24 down projections' median
3.8e-3), and the down projections' share of the squared global norm went from 3.8% to 10.1%; the v projections hold
70–72% of it, largest in value-embedding layers (15, 19, 13, 17). The value-embedding gates (zero-initialized, so 1
at the start) saturated open early. On 12 validation documents (the first 4 of ClimbMix, Stack-Edu Python and
UltraData-Math L3 textbook exercises, at most 1,024 tokens each, `inference/modeling_nanochat.py` in fp32 on the CPU),
at checkpoint 1,000 the gates of layers 7–23 exceed 1.99 of their 2 at 99.8–100% of positions (layers 1–5 at
95–99%), the sigmoid's input past ±10 at 98–100%, and their gradient norms fell to 1e-7–1e-13 in layers 11–19 at
steps 1,000 and 2,000 (the v projections' 1e-2). By checkpoint 20,000 layers 5, 9, 13, 15 and 17 had left
saturation (input past ±10 at 10–34%), layers 7, 21 and 23 not (at least 96%), and the gated embedding's norm
reached 0.76–1.41× the value's in layers 13–19 (0.07–0.10× at checkpoint 1,000).

**Job** (`run.job`): the run is skipped if every checkpoint's evaluation exists or it diverged; else training
(resuming, below), then the unevaluated checkpoints' evaluations, overlapped (`run.evaluations`,
`docs/evaluation.md` §1), each phase in a fresh process. Runs are not re-checked against the current recipe.

**Checkpoints.** Full state (`checkpoint-<step>.pt`: model, optimizers, model config, stage rows) is saved at
`run.checkpoint_steps`, `round(steps·k/n)` for k = 1…n plus `repo_level_start` when n > 1, the only checkpoints
evaluated. A run with n > 1 also keeps `run.analysis_steps`' checkpoints, never evaluated:

| Set | Steps (base run) | Content | Count × size | Basis |
|---|---|---|---|---|
| evaluated | each tenth of 69,874, and 66,063 | full state | 11 × 5.588 GB | the evaluations of the run (§9) |
| grid | every `FULL_STATE_EVERY` = 1,000 below the last step | full state | 69 × 5.588 GB | Pythia's checkpoint every 1,000 iterations (arXiv 2304.01373 §2.4) |
| early | 1, 2, 4, …, `EARLY_WEIGHTS_LAST` = 512 | fp32 weights and model `config` (`weights-<step>.pt`) | 10 × 2.089 GB | Pythia's log-spaced early checkpoints (arXiv 2304.01373 §2.4) |
| after the stage switch | `repo_level_start` + `AFTER_STAGE_WEIGHTS` (100, 200, 400, 800, 1,600): 66,163 … 67,663 | fp32 weights and `config` | 5 × 2.089 GB | the early set's log spacing at the context switch |

The base run's sets are disjoint: 80 full states and 15 weights-only files, 478.4 GB [arithmetic] (a full state is
the fp32 weights, AdamW's two moments of the 352,327,728 table and vector elements, Muon's momentum of 169,924,608 and
NorMuon's row moments). The grid's optimizer state gives any 1,000-step point as a resume point or a branch for an
intervention, as Pythia's intermediate optimizer state did (arXiv 2304.01373 §3.1). The weights-only files are fp32:
a step-1 update relative to the initial weight RMS, ≈ 0.2·lr/40 ÷ 768^−½ = 3.3e-4 [arithmetic], is below bf16's 2^−8.
In the d24 rehearsal (§7), over the 8K median step, a weights-only save added 0.84–1.02 s to the next step, a full
state 2.18–2.25 s (4.01 s over the 32K median at the stage switch, compilation included; a resumed process's first
step, compilation without a save, 3.31 s): ≈ 190 s over the base run's 95 saves, 0.07% of its training [arithmetic].

**Resume and divergence.** A relaunched run resumes from its newest full-state checkpoint (weights, optimizer state,
scheduler, data position); a run with one checkpoint restarts from step 0. A resumed attempt's records supersede the
earlier ones for the steps it repeats (`metrics.train_steps`). A non-finite loss or gradient norm writes a `failure`
record and ends training; the sweep continues, a rerun skips the run (`metrics.diverged`), and its experiment record
has `diverged_at` set and `measures`, `errors`, `offset_bpb` and `efficiency` null.

**Experiment records** (`ablations/records.py` `Records`, an experiment's Hydra callback, when the sweep ends):

| File | Content |
|---|---|
| `results/<experiment>.jsonl` | per run: `experiment`, `eval/suite.py` `record` (`run_name`, the run log's non-train records, the checkpoints' evaluations), `config`, `measures` and `errors` (`docs/evaluation.md` §5), `offset_bpb` (32K code bits per byte by byte offset), `efficiency` (the first `run` record's `num_parameters`, `flops_per_token` and `model_flops_per_token`; the peak allocated and reserved bytes over the `stage` and `end` records), `diverged_at` |
| `results/<experiment>-curves.jsonl` | per run: `experiment`, `run_name` and its train records as columns (`metrics.curves`) |

The ablation records and their status: `docs/experiments.md` §1. `results/peak_lr_d24.jsonl`'s `run` records carry
the earlier multi-depth code's `RunConfig` fields (`num_hidden_layers` 24, `seed` 0, `tokenizer` "corpus"), all at
their defaults. `results/muon_wd_timescale_20tpp.jsonl` and `results/scalar_learning_rates.json[l]` were written by
earlier code, without the output-logit cap (`model.final_logit_softcapping` null).

## 7. Hardware and throughput

**Machine.** One GH200: a Hopper GPU (sm_90, 96 GB HBM3, 97,871 MiB reported), 64 Neoverse-V2 cores, 525 GiB of
memory (`nvidia-smi`, `lscpu`, `free -g`).

**Power and clock.** The GPU's enforced limit is 700 W, set outside the VM (the software power cap is active in every
sample of the d24 8K step); the module (Grace + Hopper + memory) shares a 1,000 W budget and draws 888–910 W with the
CPU busy [unverified: samples not kept], so no heavy CPU job runs beside a measured run. `scripts/setup.sh` locks the
SM clock at `clocks.max.sm` (1,980 MHz; lost on a driver reload or reboot): the d24 8K step 3.0% faster, d16 1.7%
(exclusive A/B). Under training the cap still lowers the clock: 1,492–1,647 MHz on average across the d24 profiles,
sustained GEMMs ~14% below their isolated speed [unverified: profiles not kept]; the d24 rehearsal's NVML medians are
1,395 MHz and 669 W. The latest A/B arms were likely unlocked [inference]: a whole-master rate measured before the
rehearsal, 366,864 tok/s, is 1.0% above the latest arm's 365,466 less the softcap's 0.58% (363,359), not +3.0% (their
relative gains are unaffected: both arms alike). Locked from step 600, the rehearsal's 8K median went from 360,410
tok/s (steps 0–599) to 374,203 (steps 600–3,126; not a controlled A/B).

**Step profile** (d24, 8K, nsys; the baseline of the folded post-attention norm's A/B): 4,355 ms of GPU time per
step, 99.4% busy: GEMMs 60.5% (~680 TFLOP/s), FlashAttention-4 18.9% (34.7% at 32K; the 512-window layers at 283–336
TFLOP/s against 511–600 for full attention), memory-bound kernels 19.8%, the optimizer 0.6% [unverified: profiles not
kept]. The step is GPU-bound and its GEMMs run at the 700 W cap's ceiling, 703 of 989 dense bf16 TFLOP/s (d24 8K nsys
profile). Each kernel landed since, with its gain: `docs/architecture.md` §3.

**Throughput** (tokens per second; MFU: executed model FLOPs over 989.4 TFLOP/s, `docs/evaluation.md` §5):

| Measurement | d8 8K / 32K | d16 8K / 32K | d24 8K | d24 32K |
|---|---|---|---|---|
| Baseline (before the folded post-attention norm; d24 8K: its A/B's base run; the other cells [unverified: no record]) | 2.81M / 2.37M | 803,750 (MFU 0.391) / 638,405 | 338,039 (0.450) | 273,536 (0.465) |
| Master as a whole (loop rate before the A/B-timed landings) | | | 356,259 (0.473) | 282,256 (0.483) |
| Latest A/B arm, QuACK's scaled-exp linear cross entropy (likely unlocked, above) | | | 365,466 (0.4857) | 287,476 |
| d24 rehearsal: in-loop medians of the whole run, locked from step 600 | | | 373,168 (0.498) | 298,565 (0.508) |

Later landings were timed as A/B arms only, spanning 362,972–365,466 tok/s at 8K and 286,555–287,482 at 32K (fixed
QuACK tile configurations, FA4's dpsum from o_proj's epilogue, the scaled-exp linear cross entropy). The output-logit
cap (fused-softcap A/B) adds 1.56 ms per micro-batch, 0.58% of the 8K step (its 32K cost unmeasured;
`docs/architecture.md` §3). The rehearsal (not a recipe): the base recipe at 5 tokens per parameter (3,307 steps of
294,912 tokens, the repository stage from 3,127, the peak-lr sweep's d24 plan) at the calibrated × 2.7193,
`num_checkpoints=3`, killed after step 1,050 and resumed from step 1,000.

**Memory.** The d24 10-step peak is 71.11 GiB allocated at 8K, warm and cold (fused-softcap A/B), of the GPU's 95.58
GiB (97,871 MiB). A micro-batch holds whole rows of both contexts, so its size is a multiple of 32,768 tokens;
`MICRO_BATCH_TOKENS` is 3 of them, the most that fits: the step's persistent state (fp32 weights and gradients of the
522,252,336 parameters, AdamW's two moments of the 352,327,728 outside Muon, Muon's momentum and bf16 GEMM copies of
its 169,924,608) is 7.47 GiB [arithmetic]. The allocator's peak reserve is 73.14 GiB (`results/peak_lr_d24.jsonl`
`peak_reserved_bytes`, 2.04 GiB above the allocated peak); 4 rows of 32K would scale its 65.67 GiB above the state by
4/3, to ≈ 95.0 GiB in all [estimate: activation memory proportional to tokens], 0.5 GiB below the device before the
CUDA context, which the allocator does not count. The micro-batch also quantizes every run's batch (§2).

**Time estimate** [estimate]: 97.41B tokens at 365,466 tok/s and 5.62B at 287,476 are 74.0 + 5.4 h; with the cap's
0.58% at 8K, 79.9 h; across the arms' span ≈ 80 h of training, plus ≈ 190 s of checkpoint saves (§6) and the 11
evaluated checkpoints' benchmark evaluations.

Concurrent runs and MPS sharing are not used (`docs/experiments.md` §5).

## 8. Reproducibility

**Environment.** `docker/Dockerfile` pins by digest uv 0.13.6 and NGC PyTorch 26.09 (PyTorch 2.14.0a0 with Triton,
CUDA and nvcc; torch, Triton, numpy and nvidia-ml-py come from the image and are excluded from the lock,
`pyproject.toml` `exclude-dependencies`), Rust 1.99.1 for the crates, and openai/human-eval-infilling and
nuprl/MultiPL-E at the versions pinned in `docker/Dockerfile`; the image's own FlashAttention is uninstalled so the
locked `flash_attn.cute` is imported. `pyproject.toml` pins every direct dependency to one version (e.g.
`hydra-core==1.3.8`, `quack-kernels==0.6.5`, `flash-attn-4==4.0.0b36`, `duckdb==2.0.0`; evalplus by git revision),
`uv.lock` the rest. The image patches evalplus's `untrusted_check` (`evalplus/eval/__init__.py`): its shared values
take no lock and the parent joins the child after terminating it, so a sample killed at its time-out while holding a
lock cannot block the parent; verdicts are unchanged (the child is the values' only writer) and the build fails on
another evalplus revision. `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` keeps fp32 matmuls fp32 (the NGC image sets 1);
`OMP_NUM_THREADS=1` (parallelism is by process); compiler and kernel caches live in `artifacts/cache`.

**Data order.** Each corpus's packed rows are permuted once by `train/loop.py` `SEED` = 0 (NumPy `RandomState`, whose
stream is frozen across NumPy versions; `stage_batches`), which also seeds the initialization; rows go to corpora by
the loader's Weyl sequence (`mixture.mixture_rows`; `docs/corpus.md` §9). The order does not depend on the CPU
[inference].

**Determinism.** Gradients are not bitwise across processes (FA4's and the embedding backwards' atomics;
`docs/architecture.md` §4). Kernel choices do not depend on timing: QuACK runs fixed tile configurations
(`docs/architecture.md` §3), and Inductor's coordinate-descent tuning, which picks configurations by timing, stays at
PyTorch's default, off (`torch._inductor.config.coordinate_descent_tuning`; neither the image nor
`scripts/container.sh` sets `TORCHINDUCTOR_COORDINATE_DESCENT_TUNING`). Python's garbage collector is frozen and
disabled after the first step and collects only after each checkpoint or weights save (`train/loop.py` `train`).

## 9. Reproducing the run

On a fresh GH200 host with Docker, the NVIDIA driver and the NVIDIA Container Toolkit (`nvidia-ctk`), and sudo
(Lambda `gpu_1x_gh200`):

```bash
git clone https://github.com/indrajeetadityaroy9/nanochat nanochat && cd nanochat
HF_TOKEN=<read token> scripts/setup.sh
```

`scripts/setup.sh` writes the CDI spec, locks the SM clock (§7; re-run it after a reboot), joins the docker group,
builds the `nanochat` image (`docker build -f docker/Dockerfile -t nanochat .`) and pulls the dataset
(`python -m corpus.publish pull`: the curated shards, their decontamination record and the shipped tokenizer). After
logging in again, every command runs in the image through `scripts/container.sh` (the checkout mounted at
`/workspace/nanochat`, NGC's documented run flags, an unlimited stack hard limit for evalplus's `reliability_guard`),
whose entrypoint creates `artifacts/` (`paths.py`).

1. **Corpus**: `scripts/container.sh python -m corpus.tokenize`, then `scripts/container.sh python -m corpus.pack`
   (`docs/corpus.md` §10 steps 6–7; the tokenizer is the pulled one, not retrained).
2. **Peak lr**: `scripts/container.sh python run.py experiment=peak_lr_d24` (§5; the 3 runs and their evaluations),
   writing the records `lr_multiplier` reads.
3. **Train**: `tmux new -d -s run 'scripts/container.sh python run.py experiment=pretrain'` trains both stages with
   11 evaluated checkpoints (each tenth of the 69,874 steps and step 66,063, the repository stage's start) and the
   84 analysis checkpoints (§6), evaluates the 11 with every benchmark and writes `results/pretrain.jsonl` and
   `results/pretrain-curves.jsonl`. Expected time: §7.
4. **Release**: `HF_TOKEN=<write token> scripts/release.sh <run_name> <repo>` uploads the run's checkpoints,
   `metrics.jsonl`, the diagnostics sidecar, evaluations, `.hydra/`, the corpus tokenizer
   (`artifacts/corpus/tokenizer/`) and `results/` to a Hugging Face model repository; the first command below restores
   them where `load_model` and a resumed run read them. A checkpoint as a transformers model (`docs/evaluation.md` §8)
   goes to its own model repository (the export and upload below).

```bash
scripts/container.sh hf download <repo> --local-dir /workspace/nanochat
scripts/container.sh python -m inference.export_hf \
  artifacts/runs/<run_name>/checkpoint-<step>.pt artifacts/export/<run_name>-<step>
HF_TOKEN=<write token> scripts/container.sh hf upload <model repo> \
  /workspace/nanochat/artifacts/export/<run_name>-<step> . --repo-type model
```

Everything a run reads or writes lives in the checkout (`paths.py`): `artifacts/` (gitignored) holds `corpus/`,
`runs/<run_name>/` and `cache/`; `results/` (tracked) holds the records the paper's tables are drawn from.

## 10. Post-training

Not implemented: the repository pre-trains and evaluates a base model. The design, its evidence and the fetched data
(supervised fine-tuning, distillation, GRPO with unit-test rewards): `docs/post-training.md`.

## References

| arXiv | Title |
|---|---|
| 1412.6980 | Adam: A Method for Stochastic Optimization |
| 1910.04209 | On the adequacy of untuned warmup for adaptive optimization |
| 2204.02311 | PaLM: Scaling Language Modeling with Pathways |
| 2304.01373 | Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling |
| 2308.12950 | Code Llama: Open Foundation Models for Code |
| 2401.02954 | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism |
| 2401.14196 | DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence |
| 2408.11029 | Scaling Law with Learning Rate Annealing |
| 2409.12186 | Qwen2.5-Coder Technical Report |
| 2409.19913 | Scaling Optimal LR Across Token Horizons |
| 2411.00640 | Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations |
| 2502.15938 | Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs |
| 2502.16982 | Muon is Scalable for LLM Training |
| 2503.04715 | Predictable Scale: Part I, Step Law – Optimal Hyperparameter Scaling Law in Large Language Model Pre-training |
| 2505.13738 | Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training |
| 2506.02285 | Why Gradients Rapidly Increase Near the End of Training |
| 2507.20534 | Kimi K2: Open Agentic Intelligence |
| 2509.02046 | Fantastic Pretraining Optimizers and Where to Find Them |
| 2510.05491 | NorMuon: Making Muon more efficient and scalable |
| 2512.05620 | Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales |
| 2512.22382 | Completed Hyperparameter Transfer across Modules, Width, Depth, Batch & Duration |

Repositories: `zichongli5/NorMuon` (`normuon.py` `normuon_update`); `NVIDIA/Megatron-LM` (PR 5395);
`openai/human-eval-infilling`, `nuprl/MultiPL-E`, `evalplus/evalplus` (§8's pinned versions).
