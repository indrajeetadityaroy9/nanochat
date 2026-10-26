# Training

How the base model and every ablation run are trained on one GH200: stages, plan, optimizers, schedule, the peak
learning-rate calibration, the run app and its records, hardware and throughput, reproducibility and the commands.
The code is the specification. Markers: [arithmetic] derived from code numbers; [inference] reasoning not measured;
[estimate]; [unverified] a figure without a record. Measurements name their commit (GH200 at 700 W, exclusive A/B).

## Code

|File|Section|
|---|---|
|`run.py` (`RunConfig`, `run_name`, `same_config`, `plan_of`, `base_plan`, `checkpoint_steps`, `phase`, `job`, `lr_multiplier`)|§2, §5, §6|
|`conf/run.yaml`, `conf/experiment/pretrain.yaml`|§5, §6, §9|
|`train/schedule.py` (`plan`, `RunPlan`, `WARMUP_STEPS`, `lr_lambda`)|§2, §4|
|`train/stages.py` (`max_tokens_per_param`, `stage_batches`)|§1, §8|
|`train/optim.py` (`optimizers`, `Muon`, `tau_weight_decay`, `update`, `MAX_GRAD_NORM`)|§3|
|`train/loop.py` (`train`), `train/diagnostics.py` (`Diagnostics`), `train/checkpoint.py`|§6, §8|
|`ablations/peak_lr.py` (`fit`, `analysis`); `results/peak_lr.jsonl`, `peak_lr-curves.jsonl`, `peak_lr.json`|§5|
|`ablations/records.py` (`Records`, `record`)|§6|
|`scripts/setup.sh`, `scripts/container.sh`, `scripts/release.sh`, `paths.py`|§7, §9|
|`docker/Dockerfile`, `pyproject.toml`, `uv.lock`|§8|

## 1. Data and stages

The corpus side (mixture, packs, FIM, loader) is `docs/corpus.md` §7–§9. Training reads two stages
(`corpus/mixture.py` `FILE_LEVEL_STAGE`, `REPO_LEVEL_STAGE`; `RunPlan.stage_at`): 8,192-token files, then
32,768-token repositories over the last `REPO_LEVEL_SHARE` (5.45%) of the steps, in micro-batches of
`MICRO_BATCH_TOKENS` = 98,304 tokens (12 rows at 8K, 3 at 32K); their basis: `docs/corpus.md` §7.
The loss is next-token cross entropy, the packs' FIM pieces read as FIM examples (`docs/corpus.md` §8). At the
repository stage's first step the loop sets the 32K context and recompiles the model.

`train/stages.py` `stage_batches`: a stage's micro-batches in the run's order, each corpus's packed rows permuted once
by the seed and drawn by the stage's weights at the rows the run draws (`mixture.weights`); a run that would draw more
rows of a corpus than its packs hold raises. `max_tokens_per_param`: by bisection, the most whole micro-batches of
tokens per non-embedding parameter at which no stage overdraws (not monotone, since the batch steps up with the count:
a larger fitting count may exist). A run's `tokens_per_param` defaults to it, so no stage reads a document twice.

| | The base model | Reference code models |
|---|---|---|
| Context stages | 8K files, then the last 5.45% of steps at 32K on repositories | Qwen2.5-Coder the same 8K → 32K repository stage (arXiv 2409.12186 §3.2); DeepSeek-Coder repository-ordered files (arXiv 2401.14196 §2.2), then 1,000 steps at 16K (§3.6); Code Llama a 16,384-token long-context stage (arXiv 2308.12950 §2.4) |

## 2. Plan

`train/schedule.py` `plan(model, tokens_per_param)`, with N the non-embedding parameters (`num_non_embedding_params`:
decoder layers and `lm_head`, what the token budget counts) and M DeepSeek LLM's non-embedding FLOPs per token at the
file-level context (`non_embedding_flops_per_token`, Eq. 2, arXiv 2401.02954 §3.2; both `docs/architecture.md` §1):

- tokens D = `round(tokens_per_param · N)`, compute C = M·D;
- batch B = 0.2920·C^0.3271 tokens in whole micro-batches (at least one), steps = D // B;
- `repo_level_start` = steps − `round(REPO_LEVEL_SHARE · steps)`;
- the law's peak lr η = 0.3118·C^−0.1250 (§3.1 Eq. 1), times `learning_rate_scale` (`run.py` `plan_of`; §5).

Every run, an ablation arm included, takes the base model's plan at its `num_hidden_layers` and `tokens_per_param`
(`run.base_plan`), so runs differing in other fields match in batch, steps and lr. Recomputed on the CPU from master,
the corpus's packs and `results/peak_lr.jsonl` (peak lr at the base multiplier 2.7970369; weight decay by §3's law):

| Run | Tokens/param | Micro-batches | Steps | Repo start | Tokens | Law lr | Peak lr | Weight decay |
|---|---|---|---|---|---|---|---|---|
| d24 base (`max_tokens_per_param` 526.134) | 526.127 | 15 (1,474,560 tokens) | 69,611 | 65,814 | 102.65B (97.05B at 8K, 5.60B at 32K) | 8.6290e-4 | 2.4136e-3 | 0.14916 |
| d24 at `mixture.MAX_STEPS` | 525.893 | 15 | 69,580 | 65,785 | 102.60B | 8.6295e-4 | 2.4137e-3 | 0.14918 |
| d16 at 20 | 19.998 | 3 | 4,552 | 4,304 | 1.342B | 1.6686e-3 | 4.6671e-3 | 0.2105 |
| d8 at 20 | 19.997 | 1 | 2,987 | 2,824 | 0.294B | 2.4438e-3 | 6.8355e-3 | 0.2191 |
| d8 at 200 | 199.993 | 2 | 14,937 | 14,122 | 2.937B | 1.8326e-3 | 5.1259e-3 | 0.1966 |
| d8 at 5 (`peak_lr`) | 4.994 | 1 | 746 | 705 | 0.073B | 2.9062e-3 | | |
| d24 at 5 (`peak_lr`) | 4.999 | 3 | 3,307 | 3,127 | 0.975B | 1.5443e-3 | | |

N and M per depth: `docs/architecture.md` §1. `max_tokens_per_param` of the corpus's tokenizer is 6,990 at d8 and
1,529 at d16. The stages' token budgets are sized by `mixture.MAX_STEPS` (`docs/corpus.md` §7), the table's second
d24 row (`b065ad7`); the plan on the packs is 69,611 steps (+0.04%), whose draws `stage_batches` checks.

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
  steps (Power Lines, arXiv 2505.13738 Finding 3, App. E.3.1; arXiv 2405.13698), so λ = 1/(η·τ*·steps), 0.14916 for
  the base run. The law is fit on AdamW over TPP 20–1280 (the base run's 526 inside). Muon takes the same law: under
  μP a fixed independent weight decay stays optimal across TPP for both Adam and Muon (arXiv 2512.05620 App. H.5), and
  Muon's only scaling-sensitive hyperparameter is its lr (arXiv 2509.02046 §3.3 Table 4). The Muon-specific timescale
  runs: `docs/experiments.md` §4.
- **Clipping**: the accumulated gradient's global norm clipped to `MAX_GRAD_NORM` = 1.0 before both optimizers
  (DeepSeek LLM §2.3), the pre-clip norm logged every step. Adam is invariant to the gradient's scale (arXiv 1412.6980
  §2.1) and Newton–Schulz divides its input by its norm (`torch/optim/_muon.py` `_zeropower_via_newtonschulz`, after
  KellerJordan/Muon), so under a slowly varying gradient scale the clip damps only spikes [inference]; the largest
  recorded norm is 10.76 (`results/peak_lr-curves.jsonl`, d8 at 200, ×1, step 14,202), far below the ~5e7–2e11 at
  which Megatron-LM PR 5395 reports pre-Muon clipping failing [unverified: PR text].

| | The base model | Reference code models |
|---|---|---|
| Optimizer | NorMuon on matrices, AdamW on tables and vectors | AdamW, β = (0.9, 0.95) (DeepSeek LLM §2.3, followed by DeepSeek-Coder §3.4; Code Llama) |
| Peak lr | the law × the measured multiplier (§5), peak of linear decay to zero | the law's lr held for 80% of tokens, then 31.6% and 10% of it (DeepSeek LLM §2.3; DeepSeek-Coder §3.4) |
| Weight decay | τ law, 0.149 | 0.1 (DeepSeek LLM §2.3) |
| Gradient clip | 1.0 | 1.0 (DeepSeek LLM §2.3) |

**NorMuon's adoption (`646a986`).** A screen at d16 at 20 tokens per parameter, one seed per arm paired with base seed
0, at ×1 (the law's lr, below the base multiplier): NorMuon's training loss over the last 50% of steps −0.0029 below
the base's (negative in all 23 hundred-step windows), every bits-per-byte measure better by 0.0003–0.0013. One seed
cannot resolve that: base seeds 1 and 0 differ by 0.0014–0.0027 bpb and the treatment effect's seed spread is
unestimated. NorMuon is adopted on the literature: 11.31% over Muon at 1.1B (arXiv 2510.05491 abstract);
modded-nanogpt PR #144 (on top of Polar Express, −15 steps over 20 runs); karpathy/nanochat `dev/LOG.md` 2026-01-10
("a very small improvement", kept). A d24 check (200 steps, ×2.797, seed 0, 8K): last-20 loss 2.2946 against 2.2967,
largest gradient norm after step 40 1.153 against 1.387, no spike, −0.08% tok/s, equal peak (77.5 GiB). The screen's
other arms: `docs/experiments.md` §4.

## 4. Schedule

`train/schedule.py` `lr_lambda`, every group's `LambdaLR` multiplier:

- **Warmup**: linear over `WARMUP_STEPS` = 2/(1 − β₂) = 40 steps (arXiv 1910.04209 §4.3, Eq. 14: "linear warmup
  over τ = 2·(1−β2)^−1 iterations"; karpathy/nanochat 92d63d4 uses 40 at d24); step s trains at (s + 1)/40. In the
  11 `peak_lr` runs (×1 to ×8) the largest gradient norm of the first 40 steps is 2.46–3.00 at d8 and 3.21–4.07 at
  d24, and none diverged (`results/peak_lr-curves.jsonl`). No paper fixes a warmup for Muon.
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
models (arXiv 2503.04715 §1 Eq. 1), raises it. So it is measured at d24 (the base model's shape, standard
parameterization, not μP: no width or depth transfer) over `SHORT` = 5 tokens per parameter, times the optimum's
horizon trend measured at d8 between 5 and `LONG` = 200 (where d8's law batch is 2 micro-batches, so the trend
includes batch growth, as the base run's does). The ×8 arm at d8 at 200 is where Step-Law points.

**Arms.** × 1, 2, 4, 8 at d8 at 5 and at 200; × 1, 2, 4 at d24 at 5; seed 0; 11 runs.

**Rule** (`ablations/peak_lr.py` `analysis`, pre-registered):
- per setting (`fit`), code bits per byte against u = ln(multiplier), a diverged run worse than any evaluated one:
  extend the grid ×2 while the lowest-code run is at an end; else the vertex of the parabola through the lowest-code
  run and its two neighbours (`metrics.parabola_vertex`, `docs/evaluation.md` §5; arXiv 2409.19913 §3.1 fits
  quadratics in log lr), or that run itself when its upper neighbour diverged;
- the vertices, multipliers of the law's lr the runs recorded, re-based onto the current law's lr (`base_plan`), whose
  batch and steps must equal the recorded runs' (else it raises);
- r = vertex(d8 at 200)/vertex(d8 at 5), against the SDE rule's √(m_B/m_D) and Step-Law's m_D^0.307, each divided by
  the law's own fall between the two plans;
- the net horizon exponent h = ln(r/√m_B,d8)/ln m_D,d8 (d8's trend net of its batch growth by the square-root rule,
  arXiv 2205.10287 Def. 5.2), and the base multiplier vertex(d24 at 5)·m_D,base^h·√m_B,base, m_D,base and m_B,base
  the base plan's tokens and batch over d24 at 5's.

The trend is assumed width-independent: arXiv 2409.19913 §4 fits its horizon exponent on models of 760M parameters
and more and observes a larger one at 50M and 125M (d8 has 14.7M non-embedding parameters). The net exponent is
carried to the base model's horizon as every rule in play is a power law in D; the square-root batch rule is also
Complete^(d)P's (arXiv 2512.22382 Table 1).

**Result** (`results/peak_lr.jsonl`; code bits per byte; no run diverged; recomputed on the CPU):

| Setting | Plan | × 1 | × 2 | × 4 | × 8 | Vertex (recorded lr) | Curvature | Re-based |
|---|---|---|---|---|---|---|---|---|
| d8 at 5 | 746 steps of 98,304 tokens | 0.89491 | 0.78969 | 0.76960 | 0.81791 | × 3.4669 | 0.0712 | × 3.8781 |
| d8 at 200 | 14,937 steps of 196,608 | 0.52774 | 0.51750 | 0.51592 | 0.52431 | × 3.1565 | 0.0104 | × 3.5309 |
| d24 at 5 | 3,307 steps of 294,912 | 0.44230 | 0.43787 | 0.44201 | | × 2.0235 | 0.0089 | × 2.1804 |

The best arm beats the law by 0.1253, 0.0118 and 0.0044. The runs recorded the law's lr under nanochat's per-window
FLOP count; Eq. 2's M (§2) lowers it at the same batch and steps by 1/1.1186 at d8 and 1/1.0776 at d24, hence the
re-basing. r = 0.9105 (predicted: SDE rule 0.3544, Step-Law 4.9231; law fall 0.6306; horizon exponent −0.0254, −0.1193
net of the batch; m_B,d8 = 2, m_D,d8 = 40.05): the optimum barely moves with the horizon, against both rules. The
base multiplier is 2.7970369 (m_B,base = 5, m_D,base = 105.25): peak lr 2.4136e-3 on the base plan.

**`lr_multiplier`.** `conf/run.yaml` sets every run's `learning_rate_scale` to `${lr_multiplier:}`, a cached resolver:
`run.py` `lr_multiplier` evaluates `analysis` over `results/peak_lr.jsonl` at launch and returns `base_multiplier`.
No copy is stored, since it depends on the base plan, hence on the corpus tokenizer's `max_tokens_per_param`
(`plan_of(RunConfig())`), so every launch reads the corpus's packs. `RunConfig`'s own default 1.0 is the law's lr,
so every run name carries the resolved multiplier (§6).

**Records.** The runs predate the current record format (experiment `lr`; the ×1 arm `arm` = `production`; fields
under `fields`: `n_layer`, `ratio`, `lr_scale`, `value_embeds`, `seed`; the setting under `setting`; plan keys
`total_batch_size`, `num_iterations`, `lr`). They trained on the reference tokenizer's data (dataset revision
`1e711aa7`, no longer published) without the output-logit cap; that the optimum carries to the current recipe is
assumed. `results/peak_lr.json` holds that format's analysis, per setting (`settings[]`: `depth`,
`tokens_per_param`, `seeds`, `baseline_runs`, `alternatives`) each arm's `difference` from ×1 per measure, with `p`,
`ci_low`, `ci_high` null at one seed; not the vertices `fit` computes.

## 6. Runs and records

Every run, an ablation's or the base model's, is a `run.py` `RunConfig` (Hydra, hydra-core 1.3.7), trained by
`train/loop.py` `train` and evaluated by `eval/suite.py` `evaluate` (`docs/evaluation.md` §1). Its defaults are the
base model's recipe:

| Field | Default | Meaning |
|---|---|---|
| `num_hidden_layers` | 24 | depth; the width follows (`docs/architecture.md` §1) |
| `tokens_per_param` | None | the data's `max_tokens_per_param` (§1) |
| `seed` | 0 | initialization and data order |
| `tokenizer` | "corpus" | `CORPUS/tokenizer`, else an arm under `CORPUS/tokenizer-ablation/` |
| `learning_rate_scale` | 1.0 | times the law's lr; `conf/run.yaml` sets `${lr_multiplier:}` (§5) |
| `final_logit_softcapping` | 15.0 | c·tanh(z/c) on the output logits (`docs/architecture.md` §1) |
| `num_checkpoints` | 1 | evenly spaced checkpoints, the last the final model; > 1 adds `repo_level_start` |
| `generate` | "parse" | `evaluate`'s mode: "parse" (ablations) or "benchmarks" (base) |

`conf/run.yaml` composes the `run_schema` ConfigStore node, an experiment (`conf/experiment/<name>.yaml`, which sets
or sweeps fields) and the command line's overrides. Every invocation is a multirun sweep run one job at a time
(`max_batch_size` 1): `python run.py experiment=<name>`, or one run, e.g. `python run.py num_hidden_layers=8
tokens_per_param=0.5 tokenizer=sa-bpe`. The one experiment in `conf/` is `pretrain` (`num_checkpoints` 10,
`generate` "benchmarks", the `Records` callback without an analysis).

**Names.** `run_name`: the fields that differ from `RunConfig()`, sorted, `k=v` joined by `,` (`defaults` if none), so
a run two experiments share is trained once. The base model's run is `learning_rate_scale=2.7970368997577726`, a d16
seed-1 run `learning_rate_scale=2.7970368997577726,num_hidden_layers=16,seed=1`. Its directory
`artifacts/runs/<run_name>/` holds `.hydra/` (the composed config), `metrics.jsonl` (the run log; each line carries
`run_name`, `attempt`, `kind`, `time`), `checkpoint-<step>.pt`, `eval-<step>.json` and `run.log`. The run log's
records (`train/loop.py`; field definitions in the `metrics.py` docstring):

| `kind` | When | Content |
|---|---|---|
| `run` | each process start | first step, `plan`, `seed`, `config` (the `RunConfig`), the tokenizer's `data_key`, the model config, `num_parameters`, M (`flops_per_token`), PaLM's nominal `model_flops_per_token` (`docs/evaluation.md` §5), `max_grad_norm`, each group's `learning_rate` and `weight_decay`, the stage and its mixture |
| `train` | each step | `loss`, `num_input_tokens_seen`, `learning_rate`, `grad_norm` (before clipping), `step_seconds`, `tokens_per_second`, executed `model_flops`, `mfu`, the diagnostics |
| `stage` | the repository stage's first step | the stage and mixture, its nominal FLOPs per token, the file-level stage's peak allocated and reserved memory |
| `end` | after the last step | the last stage's peak allocated and reserved memory |
| `failure` | a non-finite loss or gradient norm | the step |

A step's values are copied to pinned host memory without blocking and logged after the next step is launched, so
the device never drains; its time runs between CUDA events at the previous step's end and its own.

**Diagnostics** (`train/diagnostics.py`, `7e77833`): per-corpus training loss (rows assigned by
`mixture.mixture_rows`); the output logits' logsumexp maximum and mean square over valid positions (PaLM's z-loss
quantity, arXiv 2204.02311 §5); the largest QK-norm gain (attention-logit growth, the quantity Kimi K2's QK-Clip
bounds, arXiv 2507.20534 §2.1); every `UPDATE_STATS_EVERY` = 10th step each optimizer
group's update RMS (without weight decay) and parameter norm, which cost 0.50% of d24 throughput when taken every step
(351.4K / 351.8K against 349.9K / 349.8K tok/s, GH200, alternated 25-step windows after 8 warm-up steps; every 10th
step within the windows' spread, 341,290–341,471 tok/s with or without, `7e77833`). The per-micro-batch statistics
are taken without autograd: a graph through the pinned-buffer copies would chain every step's onto the last, a leak
of ~12.7 MiB per step at d24 (`1f23203`).

**Job** (`run.job`): the stale guard (a recorded `run` record whose `plan`, `data_key` or `config` differs raises
"recipe changed", stopping the sweep until the run's directory is deleted; `same_config` matches a field the record
lacks iff the current value is its `RunConfig()` default); the run is skipped if its final checkpoint's evaluation
exists or it diverged; else training, then each checkpoint's evaluation, each in a fresh spawned process
(`run.phase`: `torch.compile`'s recompile limit and the caching allocator are per process).

**Checkpoints and resume.** `run.checkpoint_steps`: `round(steps·k/n)` for k = 1…n, plus `repo_level_start` when
n > 1. A checkpoint (`train/checkpoint.py`) is the run's full state (`model`, `optimizers`, model `config`,
`stage_rows`), written to a temporary file and renamed, so a kill mid-save leaves the previous one. A relaunched run
resumes from its newest checkpoint (weights, optimizer state, the scheduler at its step, the data position); a run
with one checkpoint restarts from step 0. A resumed attempt's records supersede the earlier ones for the steps it
repeats (`metrics.train_steps`).

**Divergence.** A non-finite loss or gradient norm writes a `failure` record and ends training; the job returns, the
sweep continues and a rerun skips the run (`metrics.diverged`). Both instabilities of arXiv 2309.14322 §3.1 "appear
at high learning rates in small models"; `peak_lr`'s `fit` counts such a run worse than any evaluated one. Any other
failure stops the sweep.

**Experiment records** (`ablations/records.py` `Records`, an experiment's Hydra callback, when the sweep ends):

| File | Content |
|---|---|
| `results/<experiment>.jsonl` | per run: `experiment`, `eval/suite.py` `record` (name, run-log records, evaluations), `config`, `measures` and `errors` (`docs/evaluation.md` §5), `offset_bpb` (32K code bits per byte by byte offset), `efficiency` (parameters, FLOPs per token, peak memory), `diverged_at` |
| `results/<experiment>-curves.jsonl` | per run: its train records as columns (`metrics.curves`) |
| `results/<experiment>.json` | `{"analysis": ...}`, the pre-registered decision, when the yaml names one |

The ablation records and their status: `docs/experiments.md` §1. `muon_wd_timescale_20tpp.jsonl` and
`scalar_learning_rates.json[l]` were trained without the output-logit cap (`config.final_logit_softcapping` null); a
run of the same name today carries cap 15, so the stale guard raises "recipe changed" on their directories.

## 7. Hardware and throughput

**Machine.** One GH200: a Hopper GPU (sm_90, 96 GB HBM3, 97,871 MiB reported), 64 Neoverse-V2 cores, 525 GB of memory
(`nvidia-smi`, `lscpu`, `free`).

**Power and clock.** The GPU's enforced limit is 700 W, set outside the VM (`nvidia-smi -pl 900` is accepted but not
enforced; the software power cap is active in every sample of the d24 8K step). The module (Grace + Hopper + memory)
shares a 1,000 W budget and draws 888–910 W with the CPU busy [unverified: samples not kept], so no heavy CPU job runs
beside a measured run. `scripts/setup.sh` locks the SM clock at `clocks.max.sm` (1,980 MHz): the d24 8K step 3.0%
faster, d16 1.7% (`4ade225`). Under training the cap still lowers the clock to averages of 1,492–1,647 MHz across the
d24 profiles: sustained GEMMs run ~14% below their isolated speed.

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
| Latest A/B arm, `067e60b` | | | 365,466 (0.4857) | 287,476 |

Later landings were timed as A/B arms only, spanning 362,972–365,466 tok/s at 8K and 286,555–287,482 at 32K
(`fb86efc`, `db1e90e`, `067e60b`). The output-logit cap (`e24333d`) adds 1.56 ms per micro-batch, 0.58% of the 8K
step (its 32K cost unmeasured; `docs/architecture.md` §3).

**Memory.** The d24 10-step peak is 71.11 GiB allocated at 8K, warm and cold (`e24333d`), of 96 GB; a larger
micro-batch than `MICRO_BATCH_TOKENS` would change every run's batch rounding (§2).

**Time estimate** [estimate]: 97.05B tokens at 365,466 tok/s and 5.60B at 287,476 are 73.8 + 5.4 h; with the cap's
+0.58% at 8K, 79.6 h; across the arms' span ≈ 79–80 h of training, plus the 11 checkpoints' benchmark evaluations.

Concurrent runs and MPS sharing are not used (`docs/experiments.md` §5).

## 8. Reproducibility

**Environment.** `docker/Dockerfile` pins by digest uv 0.13.4 and NGC PyTorch 26.09 (PyTorch 2.14.0a0 with Triton,
CUDA and nvcc; torch and numpy are excluded from the lock, `pyproject.toml` `exclude-dependencies`), Rust 1.99.1 for
the crates, and openai/human-eval-infilling (`88062ff`) and nuprl/MultiPL-E (`3025a53`) by commit; the image's own
FlashAttention is uninstalled so the locked `flash_attn.cute` is imported. `pyproject.toml` pins every direct
dependency (e.g. `hydra-core==1.3.7`, `quack-kernels==0.6.5`, `flash-attn-4==4.0.0b35`, `tokenizers==0.23.3`;
evalplus `26d6d00` and TokEval `7d03a61` by git revision; `tree-sitter-language-pack==1.22.1` by override), `uv.lock`
the rest. `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` keeps fp32 matmuls fp32 (the NGC image sets 1); `OMP_NUM_THREADS=1`
(parallelism is by process); compiler and kernel caches live in `artifacts/cache`.

**Data order.** Each corpus's packed rows are permuted once by the run's seed (NumPy `RandomState`, whose stream is
frozen across NumPy versions; `stage_batches`), which also seeds the initialization; rows go to corpora by the
loader's Weyl sequence (`mixture.mixture_rows`; `docs/corpus.md` §9). The order does not depend on the CPU
[inference].

**Determinism.** Gradients are not bitwise across processes (FA4's and the embedding backwards' atomics;
`docs/architecture.md` §4). Kernel choices do not depend on timing: QuACK runs fixed tile configurations
(`docs/architecture.md` §3), and Inductor's coordinate-descent tuning, which picks configurations by timing, stays at
its default, off (`scripts/container.sh`). Python's garbage collector, whose scans are slow, is frozen and disabled
after the first step and collects only after a checkpoint save (`train/loop.py`; karpathy/nanochat 07c4dd4).

## 9. Reproducing the run

On a fresh GH200 host with Docker and the NVIDIA driver (Lambda `gpu_1x_gh200`):

```bash
git clone https://github.com/indrajeetadityaroy9/nanochat nanochat && cd nanochat
HF_TOKEN=<read token> scripts/setup.sh
```

`scripts/setup.sh` writes the CDI spec, locks the SM clock (§7), joins the docker group, builds the `nanochat` image
(`docker build -f docker/Dockerfile -t nanochat .`), logs in to the Hub, creates `artifacts/` (`python -m paths`)
and pulls the curated corpus (`python -m corpus.publish pull`). After logging in again, every command runs in the
image through `scripts/container.sh` (the checkout mounted at `/workspace/nanochat`, NGC's documented run flags, an
unlimited stack hard limit for evalplus's `reliability_guard`).

1. **Corpus**: the tokenizer ablation's vocabularies, intrinsic measures and data, the corpus's tokenizer, tokenize,
   pack and the smoke check, in the order of `docs/corpus.md` §10.
2. **Train**: `tmux new -d -s run 'scripts/container.sh python run.py experiment=pretrain'` trains both stages with
   11 checkpoints (each tenth of the 69,611 steps and step 65,814, the repository stage's start), evaluates each with
   every benchmark and writes `results/pretrain.jsonl` and `results/pretrain-curves.jsonl`. Expected time: §7.
3. **Release**: `scripts/release.sh <run_name> <repo>` uploads the run's checkpoints, `metrics.jsonl`, evaluations,
   `.hydra/` and `results/` to a Hugging Face model repository; `scripts/container.sh hf download <repo> --local-dir
   /workspace/nanochat` restores them where `load_model` and a resumed run read them.

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

## References

| arXiv | Title |
|---|---|
| 1412.6980 | Adam: A Method for Stochastic Optimization |
| 1910.04209 | On the adequacy of untuned warmup for adaptive optimization |
| 2204.02311 | PaLM: Scaling Language Modeling with Pathways |
| 2205.10287 | On the SDEs and Scaling Rules for Adaptive Gradient Algorithms |
| 2306.11644 | Textbooks Are All You Need |
| 2308.12950 | Code Llama: Open Foundation Models for Code |
| 2309.14322 | Small-scale proxies for large-scale Transformer training instabilities |
| 2401.02954 | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism |
| 2401.14196 | DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence |
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
