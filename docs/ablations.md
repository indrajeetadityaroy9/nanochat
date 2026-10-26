# Ablations

One pre-registered experiment per recipe decision the literature does not settle. Code: `ablations/<experiment>.py`
(each module's docstring holds its rule), `ablations/records.py` (the records); arms in
`conf/experiment/<experiment>.yaml`. Records in `results/`. The corpus and its dataset: `docs/corpus.md`. The evidence
narrative behind each decision to run, stop or not run an experiment: `docs/findings.md`, "Design decisions" (D5–D7)
and §5.22–§5.24; this document holds each experiment's design, rule, records and the literature that settled it.

| Experiment | Decides | Setting | Runs | State |
|---|---|---|---|---|
| `peak_lr` | the base model's peak lr multiplier | d8 at 5 and 200, d24 at 5 | 11 | done; its estimate is every run's default `learning_rate_scale` |
| `muon_wd_timescale_20tpp` | Muon's weight-decay timescale (at 20) | d16 at 20, sa-bpe data | 3 × 2 seeds | stopped after 3 runs; the τ law kept on the literature; code removed |
| `muon_wd_timescale_200tpp` | Muon's timescale law (with the 20 result) | d8 at 200, corpus tokenizer's data | 3 × 2 seeds | not run; code removed |
| `logit_softcap` | the output-logit softcap | d16 at 20, sa-bpe data | 2 × 4 seeds | not run; cap 15 adopted on the literature; code removed |
| `tokenizer_split_merge` | split pattern and merge score | d16 at each arm's `max_tokens_per_param` | 4 × 2 seeds | `vocabularies`, `intrinsic`, `data`, `repack` done; runs not run (sa-bpe kept); runs, decision and `adopt` removed |
| `adamw_group_lr` | the AdamW groups' lr | d16 at sa-bpe's `max_tokens_per_param` | 3 × 2 seeds (× 1 shared) | ×¼/×4 not run; code removed |
| `scalar_learning_rates` | per-group lrs of the residual scalars | d16 at 20, sa-bpe data | 1 (against D7's τ×1 seed 0) | run; not adopted; code removed |
| `pretrain` | (the base model's pre-training) | d24 at `max_tokens_per_param` | 1 | not run: awaits its own approval |

Run order: only `pretrain` remains (`conf/experiment/pretrain.yaml`: the run app's defaults, 10 checkpoints and the
repository stage's start, each evaluated with every benchmark, no analysis), awaiting its own approval. Every other
experiment was run, stopped or not run; its yaml, module and knobs (`RunConfig.muon_tau_scale`,
`adamw_learning_rate_scale`, `scalar_learning_rates`) and `ablations/contrast.py` are removed, and its design below
stays as the record. What remains of `ablations/`: `peak_lr.py` (its estimator, evaluated at every launch),
`tokenizer_split_merge.py` (four steps, no runs), `records.py`.

## Method

- **Runs.** Every run is a `run.py` `RunConfig` (Hydra; its defaults are the base model's recipe, an experiment's yaml
  sweeps fields against them), trained by `train/loop.py` and evaluated by `eval/suite.py evaluate`, the base model's
  code paths: `python run.py experiment=<name>`. A sweep runs one job at a time (`hydra.sweeper.max_batch_size` 1,
  `conf/run.yaml`). Run names, artifacts, resumption and the stale guard (`run.same_config`, "recipe changed"):
  `docs/pretraining.md`, Runs.
- **Plans.** An arm differs from its baseline only in its yaml's fields. Its plan (batch, steps, peak lr) is the base
  model's at its `num_hidden_layers` and `tokens_per_param` (`run.base_plan`, `train/schedule.py` `plan`); `plan_of`
  multiplies the law's lr by `learning_rate_scale`, by default the base model's multiplier (`${lr_multiplier:}` in
  `conf/run.yaml`, resolved once per process); `tokens_per_param: null` means the data's `max_tokens_per_param`
  (`train/stages.py`). Every run completes its own schedule (warmup, linear decay to zero). Weight decay follows each
  run's lr, steps and tokens per parameter by the τ law (`train/optim.py` `tau_weight_decay`: EMA timescale
  1/(η·λ), arXiv 2405.13698 §1, set to a fraction of the run by Power Lines, arXiv 2505.13738 Finding 3), so an lr
  sweep varies η at a fixed timescale.
- **Divergence.** A non-finite loss or gradient norm ends the job with a failure record (`metrics.diverged`, the
  record's `diverged_at`); the sweep continues and a rerun skips the run. Both instabilities of arXiv 2309.14322 §3.1
  "appear at high learning rates in small models". Any other failure stops the sweep. `peak_lr`'s `fit` counts a
  diverged run worse than any evaluated one.
- **Settings.** No affordable run reaches the base model (d24 at 526.13 tokens per non-embedding parameter). Ranking
  at a single small size (150M parameters) picks the better of two data recipes at the 1B target in ~80% of
  comparisons (DataDecide, arXiv 2504.11393 Abstract); architecture rankings can change with scale (arXiv 2207.10551
  §1).
- **Measures** (`metrics.measures`; bits per byte unless stated):

| Measure | Definition |
|---|---|
| `code` | 8K validation of the code group at its corpus weights; the decision measure |
| `solutions` | HumanEval+, MBPP+ and HumanEval-X C++ reference solutions given their prompts, pooled |
| `repo_32k`, `repo_beyond_8k` | 32K repository rows of the code group; their targets beyond 8,192 |
| `text`, `math` | 8K validation of each group |
| `unparsed`, `indentation`, `syntax` | shares of HumanEval+ and MBPP+ problems, pooled (542), whose healed greedy completion does not compile; by IndentationError; by another SyntaxError |
| `pass@1` | greedy base pass@1 of the same problems |
| `tokens_per_second`, `mfu` | medians over the file-level stage's steps (executed MFU) |

- **Error bars** (`metrics.measure_errors`; Miller, arXiv 2411.00640): bits per byte by the val rows' weighted ratio
  (statsmodels WLS, HC1); shares and pass@1 by their CLT standard error over problems (Eq. 1).

### Shared tests (were `ablations/contrast.py`; removed with the experiments that used them)

An arm against its baseline over the seeds where both have an evaluation, `ALPHA` = 0.05:

- **Parsing** (`PARSED` = MBPP+): per problem, the seeds whose healed greedy completion compiles under the arm minus
  under the baseline; the exact sign test (`scipy.stats.binomtest`) over the non-zero problems (McNemar's exact test
  at one seed, TokEval's companion post, App. D). Better or worse at p < 0.05, else equal. MBPP+ alone: its prompts
  leave the function header to the model, and "an evaluation that uses HumanEval alone does not expose this type of
  defect" (the post, *HumanEval*). The test treats problems as independent within one trained model; between-seed
  variance is not in it (D7's two τ×½ seeds: pooled unparsed 0.293 and 0.133).
- **Code bits per byte:** the difference of the arms' seed means, better or worse beyond z·σ·√(2/n), z = 1.96, n the
  seeds per arm, else equal. σ = `CODE_SIGMA` = 0.0034, the pooled seed standard deviation of six d8 runs at 20 tokens
  per parameter (two configurations × data seeds 0–2) on the reference tokenizer's data, assumed to carry to other
  depths and to the corpus; those runs' records are not in `results/` [unverified]. Margin: 0.0094 at n = 1, 0.0067 at
  n = 2, 0.0047 at n = 4.
- **`adopted`:** code better and parsing not worse.

### Records

`ablations/records.py` `Records` (each yaml's Hydra callback), when the sweep ends:

| File | Content |
|---|---|
| `results/<experiment>.jsonl` | per run: `experiment`, `run_name`, its run log's non-train records and its evaluations (`eval/suite.py` `record`), `config`, `measures`, `errors`, `offset_bpb` (32K code bits per byte by byte offset), `efficiency` (parameters, FLOPs per token, peak memory), `diverged_at` |
| `results/<experiment>-curves.jsonl` | per run: its per-step train records as columns (`metrics.curves`) |
| `results/<experiment>.json` | `{"analysis": ...}`: the experiment's pre-registered decision over the `.jsonl`, when its yaml names one |

A run's name (`run.run_name`) is its fields that differ from `RunConfig()`'s, sorted, `k=v` joined by `,`, so a run
two experiments share is trained once. `RunConfig`'s `learning_rate_scale` is 1.0 (the law's lr), so every name
carries the resolved multiplier (`learning_rate_scale=2.7970368997577726`).

In `results/`: `peak_lr.jsonl`, `-curves.jsonl`, `.json` (an earlier format, below); `muon_wd_timescale_20tpp.jsonl`
(its last two runs trained at `acb1e5e`) and `scalar_learning_rates.jsonl`, `.json` (`61e316a`), both written by
`Records` and committed in `60b972d`, without their `-curves.jsonl`;
`tokenizer_split_merge-{vocabularies,intrinsic,data}.json`. The D7 and scalar-lr runs trained without the
output-logit cap (`config.final_logit_softcapping` null): a run of the same name today carries cap 15, so the stale
guard (`run.same_config`) raises "recipe changed" on their directories.

## peak_lr

**Varies.** The multiplier of DeepSeek LLM's law η = 0.3118·C^-0.125 (arXiv 2401.02954 §3.1 Eq. 1), fit for AdamW
(§2.3) at compute budgets of 1e17 to 2e19 FLOPs (§3.1), below the base model's. The transfer rules disagree on how
the optimum moves with the horizon: Complete^(d)P's SDE rule η ∝ √(m_B/m_D) (arXiv 2512.22382 §2.1 Table 1; the
horizon rule §2.3) lowers it, Step-Law η = 1.79·N^-0.713·D^0.307 (arXiv 2503.04715 §1 Eq. 1, fit over AdamW models)
raises it. So the multiplier is measured: at d24 over `SHORT` = 5 tokens per parameter (the base model's width, no
transfer), times the optimum's horizon trend measured at d8 between 5 and `LONG` = 200.

**Arms.** × 1, 2, 4, 8 at d8 at 5 and at 200; × 1, 2, 4 at d24 at 5; seed 0.

**Rule** (`peak_lr.analysis`, evaluated by `run.py` `lr_multiplier` when a run launches, since it depends on the
adopted tokenizer's `max_tokens_per_param`; no copy is stored):
- per setting (`fit`): the grid is extended × 2 while the lowest-code run is at an end; else the vertex of the
  least-squares parabola in ln(multiplier) through the lowest-code run and its two neighbours
  (`metrics.parabola_vertex`; arXiv 2409.19913 §3.1 fits a second-degree polynomial in log lr), or the lowest-code run
  itself when its upper neighbour diverged. The vertices, multipliers of the law's lr the runs recorded, are re-based
  onto the current law's lr (`base_plan`), whose batch and steps must equal the recorded runs';
- the d8 ratio r = vertex(d8 at 200)/vertex(d8 at 5), against the SDE rule's and Step-Law's predictions, each over the
  law's own fall between the two plans;
- the base multiplier: vertex(d24 at 5) · (r/√m_B,d8)^(ln m_D,base / ln m_D,d8) · √m_B,base — d8's trend net of its
  batch growth by the square-root rule (arXiv 2205.10287 Def. 5.2), carried to the base model's horizon, times the
  base model's batch growth by the same rule. The trend is assumed width-independent: arXiv 2409.19913 §4 fits its
  horizon exponent on models of 760M parameters and more and observes a larger one for 50M and 125M (d8 has 14.7M
  non-embedding parameters).

**Result** (`results/peak_lr.jsonl`; code bits per byte; seed 0; no run diverged):

| Setting | Plan | × 1 | × 2 | × 4 | × 8 | Vertex (recorded lr) | Re-based |
|---|---|---|---|---|---|---|---|
| d8 at 5 | 746 steps of 98,304 tokens | 0.89491 | 0.78969 | 0.76960 | 0.81791 | × 3.4669 | × 3.8781 |
| d8 at 200 | 14,937 steps of 196,608 | 0.52774 | 0.51750 | 0.51592 | 0.52431 | × 3.1565 | × 3.5309 |
| d24 at 5 | 3,307 steps of 294,912 | 0.44230 | 0.43787 | 0.44201 | | × 2.0235 | × 2.1804 |

The best arm beats the law by 0.1253, 0.0118 and 0.0044. The runs recorded the law's lr under the earlier per-window
FLOP count; DeepSeek LLM's Eq. 2 M lowers the law's lr at the same batch and steps by 1/1.1186 at d8 and 1/1.0776 at
d24, hence the re-basing. The d8 ratio is × 0.9105 (predicted: SDE rule × 0.3544, Step-Law × 4.9231; horizon exponent
−0.0254, −0.1193 net of the batch, m_B,d8 = 2, m_D,d8 = 40.05): the optimum barely moves with the horizon, against
both rules. The base model's multiplier is × 2.7970 (m_B,base = 5, m_D,base = 105.25): peak lr 2.414e-3 at the base
plan (69,611 steps of 1,474,560 tokens, 102.6B), every run's default `learning_rate_scale` (`conf/run.yaml`).

The records were written in an earlier format (a run's fields under `fields`: `n_layer`, `ratio`, `lr_scale`,
`value_embeds`, `seed`; its setting under `setting`; plan keys `total_batch_size`, `num_iterations`, `lr`), on the
reference tokenizer's data (dataset revision `1e711aa7`, no longer published: `docs/corpus.md`) and a recipe without
the output-logit cap; that the optimum carries to the current recipe is assumed. `results/peak_lr.json` holds that
format's paired differences per setting, not the vertices, which `fit` computes from the `.jsonl`.

## muon_wd_timescale (D7; stopped after 3 runs; removed)

**Varied.** Muon's weight-decay EMA timescale × ½ and × 2 (its weight decay × 2 and × ½) under the τ law
(`train/optim.py` `tau_weight_decay`), AdamW's untouched. Question: Power Lines (arXiv 2505.13738) fits the τ law
with AdamW (§2.2); its App. B (Limitations) states that the EMA view "applies directly" to other decoupled-decay
optimizers such as MuonClip, without a Muon fit.

**Arms.** {½, 1, 2} × seeds 0, 1, in two experiments: d16 at 20 tokens per parameter on the sa-bpe arm's data (4,552
steps of 294,912 tokens, peak lr 4.667e-3); d8 at 200 on the corpus tokenizer's data. Rule: per setting the scale's
3-point vertex in ln(scale), adopted iff its end beat × 1 by the shared tests; Muon's own law through both points
unless both were 1.

**Outcome** (`results/muon_wd_timescale_20tpp.jsonl`; no decision, a sanity check):

| Run | Code | 32K code | Pooled unparsed | MBPP+ unparsed | pass@1 | tokens/s | Peak allocated |
|---|---|---|---|---|---|---|---|
| τ×½ seed 0 | 0.42535 | 0.40585 | 0.293 | 0.333 | 0.096 | 447,572 | 50.9 GB |
| τ×½ seed 1 | 0.42721 | 0.40779 | 0.133 | 0.143 | 0.066 | 868,428 | 39.3 GB |
| τ×1 seed 0 | 0.42724 | 0.40966 | 0.054 | 0.037 | 0.118 | 867,613 | 39.3 GB |

τ×½ seed 0 against τ×1 seed 0: code −0.0019 (margin 0.0094 at one seed: equal); MBPP+ parsing worse (7 problems parse
more under it, 119 under the baseline). τ×½ seed 0 trained with a memory leak in `train/diagnostics.py` (autograd
kept on the step's statistics; fixed in `acb1e5e`, from seed 1 on): its throughput and peak memory are not
comparable; that the leak left the update unchanged is [inference] from the fix (statistics only).

**Why stopped** (the τ law stays for Muon): arXiv 2512.05620 App. H.5: as tokens per parameter vary at fixed FLOPs,
fixing the independent weight decay yields optimal hyperparameters, Muon included, "consistent with Bergsma et al."
(Power Lines); App. H.3: halving or doubling the lr or weight decay gives higher loss or differences below the 0.1%
noise floor. arXiv 2509.02046 Table 4: Muon's only scaling-sensitive hyperparameter is its lr (§3.2: 130M–520M at 1×
Chinchilla, 130M at 2×–8×); doubling Muon's weight decay (0.1 → 0.2) costs 0.009–0.020 loss there (App. C.9 Tables
82–87). Power Lines fits tokens per parameter 20–1280 (§1), the base model's 526 inside. Muon's weight decay is the τ
law's.

## logit_softcap (D5; not run; removed)

**Varied.** The output-logit softcap c·tanh(z/c) (`RunConfig.final_logit_softcapping`, `model/modeling.py`): c = 15
against none, in the training loss, the evaluation losses and decoding.

**Design** (the record): d16 at 20 tokens per parameter on the sa-bpe arm's data, 4 paired seeds; adopt iff code bits
per byte better by > 0.0047 (n = 4) and MBPP+ parsing not worse; ~9 GPU-h.

**Outcome.** Not run; c = 15 adopted on the literature, `RunConfig.final_logit_softcapping` = 15.0 by default (as
karpathy/nanochat 92d63d4 `gpt.py:511-515`). Basis: QK norm does not bound these logits: it "does not solve the
problem of logit shift in the final softmax" (Chameleon, arXiv 2405.09818 §2.3). Output-logit divergence "occurs in
models with no weight decay regardless of scale, and z-loss resolves this instability. Weight decay also mitigates
this instability for the larger models we test" (arXiv 2309.14322 §3.1.2): `lm_head` is weight-decayed
(`train/optim.py` `optimizers`) and no `peak_lr` run diverged up to × 8, so no z-loss. The cap's measured effect is
small: modded-nanogpt record #9 added the cap at 30 as its last step, val loss 3.2844 → 3.2791 (−0.0053 nats, one run
each: the record's `+momentum warmup` and `+tanh logit capping` logs), and record #18 lowered it to 15, cutting the
steps to the 3.28 target from 1,490 to 1,390 (80 runs: mean 3.2791, p = 0.0001); at d24, karpathy/nanochat's
`dev/LOG.md` 2026-03-02 found caps from 5 to 30 about equal except 5 (much worse), 20 best by ~1e-3 val loss (0.716 →
0.715), while its code at 92d63d4 (2026-07-03) keeps 15. The fused cap's cost, +0.58% of the 8K step:
`docs/findings.md` §5.24.

## tokenizer_split_merge

Was to decide the split pattern and merge score of the corpus's tokenizer; its runs were not run, so the corpus's
tokenizer stays sa-bpe (`CORPUS/tokenizer`, the sa-bpe arm's copy: `docs/corpus.md` §5). Code:
`ablations/tokenizer_split_merge.py` (`python -m ablations.tokenizer_split_merge vocabularies|intrinsic|data|repack`),
over `corpus/train_tokenizer.py` (sample, trainer, fidelity). Each step's record holds its provenance (commit, host,
packages) and timings.

| Step | Writes | State |
|---|---|---|
| `vocabularies` | `CORPUS/tokenizer-ablation/<arm>/`; `results/tokenizer_split_merge-vocabularies.json` | done (`5ba5025`) |
| `intrinsic` | the arms' tokenized val splits; `results/tokenizer_split_merge-intrinsic.json` | done (`ebd8058`) |
| `data` | each arm's tokenized and packed training set; `results/tokenizer_split_merge-data.json` | done (`ba01bb5`) |
| `repack` | each arm's packs again from its tokenized splits (after a packer change); the data record's arms | done with FIM's line-start rule, 176 s |
| runs, decision, `adopt` | | not run; removed |

### Fixed across arms

- **Vocabulary:** 32,768 ids: 256 bytes, 32,508 merges, BOS and three FIM sentinels (`docs/corpus.md` §5).
- **Trainer:** the corpus crate's byte-level BPE (`corpus/crate/train.rs`) within the pattern's chunks; every merge an
  argmax with fixed tie-breaks, so the four arms train concurrently and independently (624–690 s each, 737 s for the
  step). Encoder: tiktoken.
- **Sample** (`train_tokenizer.sample`, one draw for every arm): each of the file stage's 17 sources contributes its
  share of the stage's bytes (`byte_shares`) of `SAMPLE_BYTES` = 13.38 GB, systematically in shard order (row i taken
  when ⌊(i+1)·f⌋ > ⌊i·f⌋, f the fraction of rows). Tokenizer training data should be "representative of model training
  data" (arXiv 2405.05417 §5); across tokenizer training mixes a subset's compression "only varies by 5 to 6 percentage
  points", higher as the mix holds more of it (arXiv 2402.01035 §2.3); proportional weights gave lower FLORES+ bits
  per byte than equal ones (arXiv 2610.12144 App. D Table 6, §5.2). Codeforces-CoTs (32K stage only) is not sampled.
- **Authorship units** (SA-BPE's R): the lowercased repository for repository files and functions (SA-BPE's unit);
  every other document its own unit (suggested for natural language, arXiv 2604.14053 §6, untested there).
- **Evaluation text:** the val splits of the corpus's 18 sources, 6.21 GB.

The draw (`-vocabularies.json`): 3,988,600 documents, 13.44 GB, 3,429,396 units; code 85.6% of its bytes; every
source within 0.99–1.05 of its target.

| Source | Target GB | Fraction of rows | Documents | GB | Units |
|---|---|---|---|---|---|
| stack_edu_python | 1.185 | 0.0178 | 435,558 | 1.187 | 384,949 |
| ultradata_stack_v3_l2_python | 1.462 | 0.0178 | 311,593 | 1.473 | 291,342 |
| ultradata_code_l2_python | 2.631 | 0.0178 | 747,214 | 2.662 | 643,565 |
| stack_edu_cpp | 0.770 | 0.0178 | 285,543 | 0.781 | 232,119 |
| ultradata_stack_v3_l2_cpp | 0.842 | 0.0178 | 219,730 | 0.845 | 145,740 |
| ultradata_code_l2_cpp | 0.645 | 0.0178 | 233,492 | 0.645 | 139,446 |
| ultradata_code_l3_python | 0.352 | 0.0345 | 135,008 | 0.352 | 135,008 |
| ultradata_code_l3_cpp | 0.363 | 0.0345 | 147,939 | 0.363 | 147,939 |
| nemotron_rewriting_python | 2.227 | 0.0345 | 624,869 | 2.227 | 624,869 |
| nemotron_transpilation_cpp | 0.756 | 0.0345 | 176,998 | 0.755 | 176,998 |
| kaggle_notebooks | 0.197 | 0.0383 | 22,001 | 0.207 | 22,001 |
| codexglue_python | 0.008 | 0.0383 | 9,554 | 0.008 | 3,738 |
| climbmix | 1.588 | 0.0341 | 532,506 | 1.586 | 532,506 |
| ultradata_math_l3_qa | 0.081 | 0.0309 | 33,157 | 0.082 | 33,157 |
| ultradata_math_l3_conversation | 0.077 | 0.0309 | 20,645 | 0.078 | 20,645 |
| ultradata_math_l3_multi_style | 0.106 | 0.0309 | 20,995 | 0.105 | 20,995 |
| ultradata_math_l3_textbook_exercise | 0.089 | 0.0309 | 31,798 | 0.089 | 31,798 |

Repository sources exceed their targets by the path comment prepended to each file (`sources.texts`), which the
Parquet-footer targets do not count.

### Arms

| Arm | Split pattern | Merge score | Role |
|---|---|---|---|
| sa-bpe | `SPLIT_PATTERN` | F·ln R | baseline |
| sa-bpe-fused | `FUSED_PATTERN` | F·ln R | positive control: line breaks fused to punctuation |
| hybrid | `HYBRID_PATTERN` | F·ln R | line breaks and indentation as chunks of their own |
| bpe | `SPLIT_PATTERN` | F | SA-BPE's score against frequency |

F is the pair's frequency, R the authorship units holding it in the current segmentation (one unit scores 0), ln
natural as in pchizhov/sa-bpe `scoring.py` (arXiv 2604.14053 §4.1).

```text
SPLIT_PATTERN   '(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++|\s*[\r\n]|\s+(?!\S)|\s+
FUSED_PATTERN   '(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
HYBRID_PATTERN  (?:[^\s\p{L}\p{N}]| )?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++|\n+|\r+|\t+| +|[^\S\r\n\t ]+
```

- **sa-bpe:** cl100k's pattern with digit runs of one or two and no line break after a punctuation run. Of SA-BPE's
  Table 3 scores, F·log R has the highest compression (3.643) and far fewer verified under-trained tokens than F (72
  against 640). Not Table 4's best F·log R·L: for natural language "the language aspect must be avoided" (§6).
- **sa-bpe-fused:** karpathy/nanochat's pattern (92d63d4): `SPLIT_PATTERN` plus `[\r\n]*` after the punctuation run, so
  `"):\n"` is one chunk. TokEval's companion post ("Tokenizer Code Ablations", *Differing one regex clause*) deletes
  that clause from GPT-4o's pattern, the tokenizer corpus fixed: MBPP pass@1 0.224, 0.246, 0.220 (gpt4onl) against
  0.167, 0.072, 0.018 (gpt4o), three seeds; across tokenizer corpora of 0–50% code the non-fusing pattern scores
  +0.016 to +0.156 above (*Varying regex clause and tokenizer training corpus independently*: one run per cell, exact
  McNemar per App. D, the 0% cell not significant, p = 0.41). Prompts are healed (arXiv 2608.18062 §4.3, App. C.1).
  This sample is 85.6% code, beyond the post's highest rung. If the parse test missed this effect at d16, it could not
  decide the others.
- **hybrid** (three clauses at once, untested as a whole):
  1. `\n+`, `\r+` are chunks of their own: a line break joins neither the punctuation nor the whitespace before it nor
     the indentation after it (the post's MinGram `cb_nl`, MBPP 0.216 against `cb`'s 0.016). No prompt ends inside a
     token spanning its last line break (the token-boundary bias of arXiv 2412.19437 §4.1).
  2. ` +`, `\t+` are chunks of their own and a letter run takes no tab before it, so an indentation of ≥ 2 spaces or
     ≥ 1 tab ends at a chunk end: a statement has the same tokens at every depth. `SPLIT_PATTERN`'s `\s+(?!\S)` gives
     the indentation's last space to the statement (`" if"`), its `[^\r\n\p{L}\p{N}]?+` a tab (`"\tif"`).
  3. No contraction clause (Punct, arXiv 2402.01035 App. E, Fig. 3), so `'self` is not cut into `'s` `elf`.
  Kept: one leading punctuation character or space on a letter run (GPT-4's pattern gives "an additional 5%
  compression over Punct at no significant cost to performance", App. E), digit pairs, punctuation runs.
- **bpe:** SA-BPE's Table 3 F row; close to F·log R in compression, so a run tie was likely, which val coverage breaks.

Not run: tiktoken encodes by merge rank alone, which cannot express PickyBPE's tokenization, replaying merges and
removals in event order (arXiv 2409.04599 Alg. 2; the removals are trained by Alg. 1), or BoundlessBPE's supermerges
(trained §3.1, applied by its inference procedure, §3.2, arXiv 2504.00178); SuperBPE moved HumanEval and MBPP pass@10
in opposite directions, neither significantly (arXiv 2503.13423 Table 1).

### Intrinsic measures (`intrinsic`)

| Measure | Definition | Use |
|---|---|---|
| bytes per token | per source: val bytes over val tokens; file stage: Σ w_s / Σ (w_s / b_s) over its byte shares | an arm's tokens at equal bytes; tokenize and pack budgets |
| extent margin | per group read in part (synthetic, text, math): train tokens under the arm over the group's draw | below 1 the corpus's extent is too small for the arm (`mixture.SPREAD`) |
| validity (asserted) | every arm holds the same bytes per document; on the first row group of each source's val split the chunks cover the text, tiktoken equals the merges replayed (`train_tokenizer.fidelity`), every text decodes | a failure stops the step |
| line-break entries | entries holding a line break; fusing punctuation with one (`[^\s\p{L}\p{N}][\r\n]`); ending in a line break and indentation | the post's most stable MBPP measurement (Spearman ρ = −0.719 over 58 tokenizers), the largest MBPP correlate of its App. C table |
| depth invariance | of each program source's first 50,000 indented val lines, the share whose tokens are the indentation's then the statement's at depth 0 | the hybrid's clause 2 |
| merged reads | merged tokens the val documents read never, < 10, < 100 times | the tie-break (SA-BPE's coverage, arXiv 2604.14053 Table 3, graded) |
| TokEval | `metrics.tokeval` (arXiv 2608.18062; cimeister/tokenizer-intrinsic-evals): compression, fertility, Rényi efficiency, bigram/trigram entropy, UTF-8 integrity, AST boundary alignment, identifier fragmentation, indentation consistency, vocabulary utilization, reconstruction, on each source's leading val documents up to `TOKEVAL_BYTES` = 2,000,000 | recorded, read by no decision: of its metrics only AST alignment predicts MBPP, attributed "to the vocabulary's line-break handling" (§5), measured above directly; Rényi efficiency predicts neither (Table 2) |

### Intrinsic results (`-intrinsic.json`)

Every arm passed every validity check. The record (`ebd8058`) predates the current step: it holds no TokEval, its
fidelity check covered each source's first 200 val documents (not the first row group), and its `seconds` keys are
`compile`, `measures`, `all`; rerun `intrinsic` to record the current step.

| Measure | sa-bpe | sa-bpe-fused | hybrid | bpe |
|---|---|---|---|---|
| bytes per token, file stage | 3.624 | 3.868 | 3.616 | 3.623 |
| repository | 3.442 | 3.698 | 3.432 | 3.443 |
| synthetic | 3.861 | 4.161 | 3.857 | 3.861 |
| notebooks | 3.403 | 3.617 | 3.413 | 3.403 |
| functions | 4.018 | 4.314 | 4.020 | 4.019 |
| text | 4.243 | 4.295 | 4.237 | 4.230 |
| math | 3.186 | 3.306 | 3.178 | 3.185 |
| val tokens | 1,739,851,130 | 1,626,775,376 | 1,744,375,884 | 1,739,939,319 |
| extent margin, synthetic | 1.096 | 1.017 | 1.097 | 1.096 |
| extent margin, text | 1.071 | 1.058 | 1.072 | 1.074 |
| extent margin, math | 1.169 | 1.126 | 1.171 | 1.169 |
| entries holding a line break | 72 | 748 | 9 | 67 |
| punctuation fused with a line break | 0 | 693 | 0 | 0 |
| ending in a line break and indentation | 5 | 0 | 0 | 5 |
| merged tokens read never | 5 | 5 | 5 | 15 |
| read < 10 times | 11 | 11 | 13 | 26 |
| read < 100 times | 54 | 54 | 56 | 84 |

Bytes per token, per source:

| Source | Val MB | sa-bpe | sa-bpe-fused | hybrid | bpe |
|---|---|---|---|---|---|
| stack_edu_python | 679 | 3.442 | 3.645 | 3.432 | 3.443 |
| stack_edu_cpp | 440 | 3.074 | 3.357 | 3.060 | 3.075 |
| ultradata_stack_v3_l2_python | 924 | 3.706 | 3.958 | 3.691 | 3.706 |
| ultradata_stack_v3_l2_cpp | 459 | 3.022 | 3.279 | 3.009 | 3.023 |
| ultradata_code_l2_python | 1,630 | 3.734 | 4.002 | 3.728 | 3.735 |
| ultradata_code_l2_cpp | 345 | 2.978 | 3.235 | 2.970 | 2.981 |
| ultradata_code_l3_python | 104 | 3.892 | 4.084 | 3.881 | 3.892 |
| ultradata_code_l3_cpp | 106 | 3.615 | 3.823 | 3.607 | 3.614 |
| nemotron_rewriting_python | 651 | 3.960 | 4.282 | 3.957 | 3.959 |
| nemotron_transpilation_cpp | 221 | 3.697 | 4.033 | 3.695 | 3.696 |
| codexglue_python | 1 | 4.018 | 4.314 | 4.020 | 4.019 |
| kaggle_notebooks | 51 | 3.403 | 3.617 | 3.413 | 3.403 |
| codeforces_cots | 11 | 3.469 | 3.603 | 3.465 | 3.467 |
| climbmix | 469 | 4.243 | 4.295 | 4.237 | 4.230 |
| ultradata_math_l3_qa | 27 | 2.932 | 3.058 | 2.929 | 2.933 |
| ultradata_math_l3_conversation | 26 | 3.492 | 3.579 | 3.486 | 3.490 |
| ultradata_math_l3_multi_style | 35 | 3.306 | 3.437 | 3.298 | 3.303 |
| ultradata_math_l3_textbook_exercise | 29 | 3.061 | 3.186 | 3.048 | 3.061 |

Depth invariance (% of indented val lines; codexglue_python has 34,317):

| Source | sa-bpe | sa-bpe-fused | hybrid | bpe |
|---|---|---|---|---|
| stack_edu_python | 0.9 | 0.9 | 99.9 | 0.9 |
| stack_edu_cpp | 13.8 | 13.8 | 98.9 | 13.6 |
| ultradata_stack_v3_l2_python | 0.3 | 0.3 | 99.9 | 0.3 |
| ultradata_stack_v3_l2_cpp | 2.4 | 2.4 | 99.1 | 2.4 |
| ultradata_code_l2_python | 0.0 | 0.0 | 100.0 | 0.0 |
| ultradata_code_l2_cpp | 3.7 | 3.7 | 99.5 | 3.7 |
| ultradata_code_l3_python | 0.0 | 0.0 | 100.0 | 0.0 |
| ultradata_code_l3_cpp | 0.0 | 0.0 | 99.1 | 0.0 |
| nemotron_rewriting_python | 0.0 | 0.0 | 100.0 | 0.0 |
| nemotron_transpilation_cpp | 0.0 | 0.0 | 97.9 | 0.0 |
| codexglue_python | 0.0 | 0.0 | 100.0 | 0.0 |

The hybrid's non-invariant lines are indented outside clause 2's condition (one space, or a tab then a space).
sa-bpe-fused compresses best and is expected to parse worst; the post finds code bits per byte "shows no association
with MBPP at any level" (*Code bits per byte*), so compression does not rank the arms. bpe and sa-bpe compress
equally; bpe leaves more rarely-read merged tokens (84 against 54 under 100).

### Data (`data`, `-data.json`)

The same document set tokenized and packed under every arm (`corpus/tokenize.py`, `pack.py`). Size: a d16 run at
`TOKENS_PER_PARAM` = 20 under sa-bpe (arXiv 2203.15556 Table 3; 4,552 steps of 294,912 tokens, repository stage from
4,304), each source at its share of the base model's stage: each code source the fewest leading train shards holding
its file-stage draw, so code binds every arm; notebooks, functions and traces their draw, read once; text, math and
the repository corpora `MARGIN` = 1.5 times their draws, so they never bind. The set: one leading train shard of each
repository source, two of every other file-stage source, one more of climbmix and each math source for the repository
stage, two of codeforces_cots, and the repository corpora's partition 99. Whole shards hold more than the draw:

| Arm | `max_tokens_per_param` | Steps of 294,912 (3 micro-batches) | Repository stage from |
|---|---|---|---|
| sa-bpe | 31.42 | 7,151 | 6,761 |
| sa-bpe-fused | 29.48 | 6,710 | 6,344 |
| hybrid | 31.49 | 7,167 | 6,776 |
| bpe | 31.42 | 7,151 | 6,761 |

The record's `run` and `seconds` keep the earlier key names (`depth`, `ratio`, `num_iterations`, `total_batch_size`,
`repo_start`; `compile`, `pack`); its `arms`, rewritten by `repack`, the current ones.

### Runs and decision (not run; removed)

The design, kept as the record: 4 arms × seeds 0, 1 at d16, each at its `max_tokens_per_param` (equal bytes), decided
by the shared tests against sa-bpe with sa-bpe-fused as the positive control, then `adopt` of the decided vocabulary.
The experiment's yaml and the module's `analysis` and `adopt` are removed; `CORPUS/tokenizer` is the sa-bpe arm's
copy, already tokenized and packed.

**Why not run** (sa-bpe kept): bpe ties sa-bpe in compression (3.623 against 3.624 bytes per token, as designed);
the fused control's harm is documented (the post: MBPP 0.22–0.25 against 0.02–0.17) and sa-bpe holds no entry fusing
punctuation with a line break; the hybrid compresses 0.2% worse and is untested as a whole, an open idea rather than a
worse one. Within a vocabulary-construction method corpus token count varies little while pre-tokenization and
segmentation change accuracy (arXiv 2402.18376 §5.3); GPT-4's and Punct's patterns show no significant difference on
code at 7B (arXiv 2402.01035 App. E, Table 5; at 1.5B and 32K ids Punct scored higher, Table 11).

**Limits.** One draw, one vocabulary size; two seeds at d16; the post's evidence is one ~1.27B model with 128K–131K
ids, mostly on MBPP; σ from d8 runs on other data; the control's effect documented up to 50% code; document units of
non-repository text untested; the hybrid changes three clauses at once.

## adamw_group_lr (D6; not run; removed)

**Varied.** The AdamW groups' lr (`train/optim.py` `optimizers`: the tables `embed_tokens`, `lm_head` and the value
embeddings, weight-decayed; the 1-D parameters, not) × ¼ and × 4 with weight decay ÷ the same factor (the tables'
decay timescale held), Muon's untouched. Pre-training uses one lr and weight decay for every group, as Moonlight
shares them "among matrix and non-matrix parameters" (arXiv 2502.16982 §2.2; its matched update RMS covers Muon's
update size, not the AdamW groups' lr); karpathy/nanochat 92d63d4 uses per-group lrs.

**Design** (the record): {¼, 1, 4} × seeds 0, 1, d16 on the sa-bpe arm's data at its `max_tokens_per_param`; rule:
the scale's 3-point vertex, adopted iff its end beat × 1 by the shared tests (margin 0.0067); ~4 GPU-h. Not run; the
AdamW groups train at the peak lr.

### scalar_learning_rates (one candidate run; not adopted; removed)

**Varied.** The residual scalars `x0_lambdas` and `resid_lambdas` (1-D, AdamW) at karpathy/nanochat 92d63d4's lrs,
0.5 and 0.005 × √(B/2^19) (0.375 and 0.00375 at d16's 294,912 tokens per step), every other group unchanged, against
the shared AdamW lr (`docs/findings.md` §5.23). One run at d16 at 20 on the sa-bpe arm's data (`61e316a`: `acb1e5e`
plus the field), against D7's τ×1 seed 0 (the same tree and recipe). Rule: adopt iff code better by > 0.0094 (one
seed) and MBPP+ parsing not worse.

**Outcome** (`results/scalar_learning_rates.json[l]`): not adopted. Code 0.42635 against 0.42724 (−0.0009: equal,
inconclusive at one seed); 32K code 0.40815 against 0.40966; train loss over the last half −0.0022; MBPP+ parsing
worse (9 problems parse more under it, 35 under the baseline, p = 1.1e-4; MBPP+ unparsed 0.106 against 0.037, pooled
0.120 against 0.054). Given the seed spread under Shared tests, "worse" is weak evidence; the rule keeps one lr.

## References

| arXiv | Paper |
|---|---|
| 2203.15556 | Training Compute-Optimal Large Language Models (Chinchilla) |
| 2205.10287 | On the SDEs and Scaling Rules for Adaptive Gradient Algorithms |
| 2207.10551 | Scaling Laws vs Model Architectures: How does Inductive Bias Influence Scaling? |
| 2309.14322 | Small-scale proxies for large-scale Transformer training instabilities |
| 2401.02954 | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism |
| 2402.01035 | Getting the most out of your tokenizer for pre-training and domain adaptation |
| 2402.18376 | Tokenization Is More Than Compression |
| 2405.05417 | Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models |
| 2405.09818 | Chameleon: Mixed-Modal Early-Fusion Foundation Models |
| 2405.13698 | How to set AdamW's weight decay as you scale model and dataset size |
| 2409.04599 | BPE Gets Picky: Efficient Vocabulary Refinement During Tokenizer Training |
| 2409.19913 | Scaling Optimal LR Across Token Horizons |
| 2411.00640 | Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations |
| 2412.19437 | DeepSeek-V3 Technical Report |
| 2502.16982 | Muon is Scalable for LLM Training (Moonlight) |
| 2503.04715 | Predictable Scale: Part I, Step Law -- Optimal Hyperparameter Scaling Law in Large Language Model Pretraining |
| 2503.13423 | SuperBPE: Space Travel for Language Models |
| 2504.00178 | Boundless Byte Pair Encoding: Breaking the Pre-tokenization Barrier |
| 2504.11393 | DataDecide: How to Predict Best Pretraining Data with Small Experiments |
| 2505.13738 | Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training |
| 2509.02046 | Fantastic Pretraining Optimizers and Where to Find Them |
| 2512.05620 | Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales |
| 2512.22382 | Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration |
| 2604.14053 | From Where Words Come: Efficient Regularization of Code Tokenizers Through Source Attribution (SA-BPE) |
| 2608.18062 | TokEval: A Tokenizer Evaluation Suite; companion post cimeister.github.io/blog/code_tokenizer_ablations |
| 2610.12144 | Language-Specific Effects of Tokenizer Choice in Multilingual Language Models |
