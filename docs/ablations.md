# Ablations

One pre-registered experiment per recipe decision the literature does not settle. Code: `ablations/<experiment>.py`
(each module's docstring holds its rule), `ablations/contrast.py` (the shared tests), `ablations/records.py` (the
records); arms in `conf/experiment/<experiment>.yaml`. Records in `results/`. No experiment runs without the user's
approval. The corpus and its dataset: `docs/corpus.md`.

| Experiment | Decides | Setting | Runs | State |
|---|---|---|---|---|
| `peak_lr` | the base model's peak lr multiplier | d8 at 5 and 200, d24 at 5 | 11 | done |
| `muon_wd_timescale_20tpp` | Muon's weight-decay timescale (at 20) | d16 at 20, sa-bpe data | 3 × 2 seeds | waits |
| `logit_softcap` | output-logit softcap | d16 at 20, sa-bpe data | 2 × 4 seeds | waits |
| `muon_wd_timescale_200tpp` | Muon's timescale law (with the 20 result) | d8 at 200, corpus tokenizer's data | 3 × 2 seeds | waits |
| `tokenizer_split_merge` | split pattern and merge score | d16 at each arm's `max_tokens_per_param` | 4 × 2 seeds | intrinsic and data done; runs wait |
| `adamw_group_lr` | the AdamW groups' lr | d16 at sa-bpe's `max_tokens_per_param` | 3 × 2 seeds (× 1 shared) | waits |

Run order (`ACTIONS.md` §0 Open): the user's approval → `muon_wd_timescale_20tpp` → `logit_softcap` →
`muon_wd_timescale_200tpp` → `tokenizer_split_merge` runs (with `adamw_group_lr`, whose × 1 arm they train) → `adopt`,
tokenize, pack → `pretrain` (its own approval). `muon_wd_timescale_200tpp` reads its 20 tpp half's result;
`adamw_group_lr`'s baseline is the tokenizer ablation's sa-bpe runs; `pretrain` waits for every decision.

## Method

- **Runs.** Every run is a `run.py` `RunConfig` (Hydra; its defaults are the base model's recipe, an experiment's yaml
  sweeps fields against them), trained by `train/loop.py` and evaluated by `eval/suite.py evaluate`, the base model's
  code paths: `python run.py experiment=<name>`.
- **Plans.** An arm differs from its baseline only in its yaml's fields. Its plan (batch, steps, peak lr) is the base
  model's at its `num_hidden_layers` and `tokens_per_param` (`run.base_plan`; `plan_of` applies
  `learning_rate_scale`); `tokens_per_param: null` means the data's `max_tokens_per_param` (`train/stages.py`). Every
  run completes its own schedule (warmup, linear decay to zero). Weight decay follows each run's lr by the τ law, so
  an lr sweep varies η at a fixed EMA timescale (arXiv 2405.13698; Power Lines, arXiv 2505.13738).
- **Divergence.** A non-finite loss or gradient norm is recorded with its step and drops that seed from the
  contrasts on both sides (arXiv 2309.14322 §3.1); any other failure stops the sweep.
- **Settings.** No affordable run reaches the base model (d24 at 525.9 tokens per non-embedding parameter). A single
  proxy 7× below the target picks the target's better of two data recipes in about 80% of pairs (DataDecide, arXiv
  2504.11393 Abstract, §3.1); architecture rankings can change with scale (arXiv 2207.10551).
- **Measures** (`metrics.measures`; bits per byte unless stated):

| Measure | Definition |
|---|---|
| `code` | 8K validation of the code group at its corpus weights; the decision measure |
| `solutions` | HumanEval+, MBPP+ and HumanEval-X C++ reference solutions given their prompts, pooled |
| `repo_32k`, `repo_beyond_8k` | 32K repository rows of the code group; their targets beyond 8,192 |
| `text`, `math` | 8K validation of each group |
| `unparsed`, `indentation`, `syntax` | shares of HumanEval+ and MBPP+ problems whose healed greedy completion does not compile; by IndentationError; by another SyntaxError |
| `pass@1` | greedy base pass@1 of the same problems |
| `tokens_per_second`, `mfu` | medians over the file-level stage's steps (executed MFU) |

- **Error bars** (`metrics.measure_errors`; Miller, arXiv 2411.00640): bits per byte by the val rows' weighted ratio
  (statsmodels WLS, HC1); shares and pass@1 by their CLT standard error over problems (Eq. 1); paired differences
  (Eq. 7, `metrics.paired_difference`); rates by Wilson's interval (`metrics.proportion_interval`).

### Shared tests (`ablations/contrast.py`)

An arm against its baseline over the seeds where both have an evaluation, `ALPHA` = 0.05:

- **Parsing** (`PARSED` = MBPP+): per problem, the seeds whose healed greedy completion compiles under the arm minus
  under the baseline; the exact sign test (`scipy.stats.binomtest`) over the non-zero problems (McNemar's exact test
  at one seed, TokEval's companion post, App. D). Better or worse at p < 0.05, else equal. MBPP+ alone: its prompts
  leave the function header to the model, and "an evaluation that uses HumanEval alone does not expose this type of
  defect" (the post).
- **Code bits per byte:** the difference of the arms' seed means, better or worse beyond z·σ·√(2/n), z = 1.96, n the
  seeds per arm, else equal. σ = `CODE_SIGMA` = 0.0034, the pooled seed standard deviation of six d8 runs at 20 tokens
  per parameter (two configurations × data seeds 0–2) on the reference tokenizer's data, assumed to carry to other
  depths and to the corpus; those runs' records are not in `results/`. Margin: 0.0067 at n = 2, 0.0047 at n = 4.
- **`adopted`:** code better and parsing not worse.
- **`scaled`** (a scale swept over {low, 1, high}): the vertex of the exact 3-point quadratic in ln(scale) through the
  arms' mean code, clamped to [low, high] (a concave or flat fit: the lower-code end); adopted iff the lower-code end
  is `adopted` against × 1, else 1.

### Records

`ablations/records.py` `Records` (each yaml's Hydra callback), when the sweep ends:

| File | Content |
|---|---|
| `results/<experiment>.jsonl` | per run: its evaluation record, `config`, `measures`, `errors`, `offset_bpb` (32K code bits per byte by byte offset), `efficiency`, `diverged_at` |
| `results/<experiment>-curves.jsonl` | per run: its per-step train records as columns (`metrics.curves`) |
| `results/<experiment>.json` | `{"analysis": ...}`: the experiment's pre-registered decision over the `.jsonl` |

A run's name (`run.run_name`) is its fields that differ from `RunConfig()`'s, sorted, `k=v` joined by `,`, so a run
two experiments share is trained once. Per-run artifacts and resumption: `docs/pretraining.md`, Runs.

## peak_lr

**Varies.** The multiplier of DeepSeek LLM's law η = 0.3118·C^-0.125 (arXiv 2401.02954 §3.1 Eq. 1), fit for AdamW at
compute below the base model's. The transfer rules disagree on how the optimum moves with the horizon: Complete^(d)P's
SDE rule η ∝ √(m_B/m_D) (arXiv 2512.22382 Table 1, §2.3) lowers it, Step-Law η = 1.79·N^-0.713·D^0.307 (arXiv
2503.04715 §1 Eq. 1) raises it. So the multiplier is measured: at d24 over `SHORT` = 5 tokens per parameter (the base
model's width, no transfer), times the optimum's horizon trend measured at d8 between 5 and `LONG` = 200.

**Arms.** × 1, 2, 4, 8 at d8 at 5 and at 200; × 1, 2, 4 at d24 at 5; seed 0.

**Rule** (`peak_lr.analysis`, evaluated by `run.py` `lr_multiplier` when `pretrain` launches, since it depends on
the adopted tokenizer's `max_tokens_per_param`; no copy is stored):
- per setting (`fit`): the grid is extended × 2 while the lowest-code run is at an end; else the vertex of the
  parabola in ln(multiplier) through the lowest-code run and its two neighbours (arXiv 2409.19913 §3.1 fits quadratics
  in log lr). The vertices, multipliers of the law's lr the runs recorded, are re-based onto the current law's lr
  (`base_plan`), whose batch and steps must equal the recorded runs';
- the d8 ratio r = vertex(d8 at 200)/vertex(d8 at 5), against the SDE rule's and Step-Law's predictions;
- the base multiplier: vertex(d24 at 5) · (r/√m_B,d8)^(ln m_D,base / ln m_D,d8) · √m_B,base — d8's trend net of its
  batch growth by the square-root rule (arXiv 2205.10287 Def. 5.2), carried to the base model's horizon, times the
  base model's batch growth by the same rule. The trend is assumed width-independent (arXiv 2409.19913 §4 finds it
  size-independent only from 760M parameters).

**Result** (`results/peak_lr.jsonl`; code bits per byte):

| Setting | Plan | × 1 | × 2 | × 4 | × 8 | Vertex (recorded lr) |
|---|---|---|---|---|---|---|
| d8 at 5 | 746 steps of 98,304 tokens | 0.89491 | 0.78969 | 0.76960 | 0.81791 | × 3.4669 |
| d8 at 200 | 14,937 steps of 196,608 | 0.52774 | 0.51750 | 0.51592 | 0.52431 | × 3.1565 |
| d24 at 5 | 3,307 steps of 294,912 | 0.44230 | 0.43787 | 0.44201 | | × 2.0235 |

No run diverged. The best arm beats the law by 0.1253, 0.0118 and 0.0044. The d8 ratio is × 0.9105 (horizon exponent
−0.0254, −0.1193 net of the batch): the optimum barely moves with the horizon, against both rules. The base model's
multiplier is × 2.7972 at 102.6B tokens (`conf/experiment/logit_softcap.yaml` fixes it). The records were written in
an earlier format (a run's fields under `fields`: `n_layer`, `ratio`, `lr_scale`, `seed`; its setting under `setting`),
on data no longer published (`docs/corpus.md`); `results/peak_lr.json` holds that format's paired differences, not
the vertices, which `fit` computes from the `.jsonl`.

## muon_wd_timescale

**Varies.** Muon's weight-decay EMA timescale × ½ and × 2 (its weight decay × 2 and × ½) under the τ law
(`train/optim.py` `tau_weight_decay`; Power Lines, arXiv 2505.13738, fit with AdamW; its App. B extends the EMA view to
decoupled-decay optimizers), AdamW's untouched (`muon_tau_scale`).

**Arms.** {½, 1, 2} × seeds 0, 1, in two experiments:
- `muon_wd_timescale_20tpp`: d16 at 20 tokens per parameter on the sa-bpe arm's data;
- `muon_wd_timescale_200tpp`: d8 at 200 on the corpus tokenizer's data (d8 at 200 reads 2.94B tokens, beyond the
  sa-bpe set's 2.11B; d16 at 200 would read 13.4B, six runs ~29 GPU-h against ~2 h; `ACTIONS.md` §0).

**Rule.** Per setting, m = `scaled` over {½, 1, 2} (`analysis`). `law` (the 200 half): if m_20 = m_200 = 1 the τ law
stays for Muon; else Muon's own law through both points, τ* = c·TPP^e with c = `TAU_C`·m_20·20^-D, e = `TAU_E` + D,
D = ln(m_200/m_20)/ln 10. The two points' depths differ, so the law takes m as independent of depth between d8 and d16
(Power Lines' τ is a function of TPP alone, arXiv 2505.13738 §2.3).

**State.** Waits.

## logit_softcap

**Varies.** The output logits capped at c = 15 (`final_logit_softcapping`; karpathy/nanochat 92d63d4 `gpt.py:511-515`;
modded-nanogpt record #18 "lowers that to 15") against none.

**Arms.** {none, 15} × seeds 0–3, d16 at 20 tokens per parameter (Chinchilla, arXiv 2203.15556 Table 3) on the sa-bpe
arm's data, `learning_rate_scale` 2.7972 (the base model's multiplier, fixed because `${lr_multiplier:}` needs the
corpus tokenizer; the cap's published effect is a high-lr one).

**Rule.** Adopt c = 15 iff `adopted`; else no cap (equal keeps none: the cap costs the loss, eval and decoder paths).

**State.** Waits.

## tokenizer_split_merge

Decides the split pattern and merge score of the corpus's tokenizer. Code: `ablations/tokenizer_split_merge.py`
(`python -m ablations.tokenizer_split_merge vocabularies|intrinsic|data|repack|adopt`, then the runs), over
`corpus/train_tokenizer.py` (sample, trainer). Each step's record holds its provenance (commit, host, packages) and
timings.

| Step | Writes | State |
|---|---|---|
| `vocabularies` | `CORPUS/tokenizer-ablation/<arm>/`; `results/tokenizer_split_merge-vocabularies.json` | done (`5ba5025`) |
| `intrinsic` | the arms' tokenized val splits; `results/tokenizer_split_merge-intrinsic.json` | done (`ebd8058`) |
| `data` | each arm's tokenized and packed training set; `results/tokenizer_split_merge-data.json` | done (`ba01bb5`) |
| `repack` | each arm's packs again from its tokenized splits (after a packer change); the data record's arms | done with FIM's line-start rule |
| runs | `results/tokenizer_split_merge.{jsonl,json}` | waits |
| `adopt` | `CORPUS/tokenizer` | after the runs |

### Fixed across arms

- **Vocabulary:** 32,768 ids: 256 bytes, 32,508 merges, BOS and three FIM sentinels (`docs/corpus.md` §5).
- **Trainer:** the corpus crate's byte-level BPE (`corpus/crate/train.rs`) within the pattern's chunks; every merge an
  argmax with fixed tie-breaks, so the four arms train concurrently and independently. Encoder: tiktoken.
- **Sample** (`train_tokenizer.sample`, one draw for every arm): each of the file stage's 17 sources contributes its
  share of the stage's bytes (`byte_shares`) of `SAMPLE_BYTES` = 13.38 GB, systematically in shard order (row i taken
  when ⌊(i+1)·f⌋ > ⌊i·f⌋, f the fraction of rows). Tokenizer training data should be "representative of model training
  data" (arXiv 2405.05417 §5); a subset's compression rises with its share of the mix (arXiv 2402.01035 §2.3);
  proportional weights gave lower bits per byte than equal ones (arXiv 2610.12144 App. D, Table 6). Codeforces-CoTs
  (32K stage only) is not sampled.
- **Authorship units** (SA-BPE's R): the lowercased repository for repository files and functions (SA-BPE's unit);
  every other document its own unit (suggested for natural language, arXiv 2604.14053 §6, untested there).
- **Evaluation text:** the val splits of the corpus's 18 sources, 6.21 GB.

The draw (`-vocabularies.json`): 3,988,600 documents, 13.44 GB, 3,429,396 units; code 85.6% of its bytes.

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
  Table 3 scores, F·log R has the highest compression and far fewer verified under-trained tokens than F. Not
  Table 4's best F·log R·L: for natural language "the language aspect must be avoided" (§6).
- **sa-bpe-fused:** karpathy/nanochat's pattern (92d63d4): `SPLIT_PATTERN` plus `[\r\n]*` after the punctuation run, so
  `"):\n"` is one chunk. TokEval's companion post ("Tokenizer Code Ablations", *Differing one regex clause*) deletes
  that clause from GPT-4o's pattern, the tokenizer corpus fixed: MBPP pass@1 0.224, 0.246, 0.220 (gpt4onl) against
  0.167, 0.072, 0.018 (gpt4o), three seeds; across tokenizer corpora of 0–50% code the non-fusing pattern scores
  +0.016 to +0.156 above (exact McNemar, App. D). Prompts are healed (arXiv 2608.18062 §4.3, App. C.1). This sample is
  85.6% code, beyond the post's highest rung. If the parse test misses this effect at d16, it cannot decide the others.
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
- **bpe:** SA-BPE's Table 3 F row; close to F·log R in compression, so a run tie is likely, which val coverage breaks.

Not run: PickyBPE's removals (arXiv 2409.04599 Alg. 2) and BoundlessBPE's supermerges (arXiv 2504.00178 §3.2), which
tiktoken's rank-only encoding cannot express; SuperBPE moved HumanEval and MBPP pass@10 in opposite directions,
neither significantly (arXiv 2503.13423 Table 1).

### Intrinsic measures (`intrinsic`)

| Measure | Definition | Use |
|---|---|---|
| bytes per token | per source: val bytes over val tokens; file stage: Σ w_s / Σ (w_s / b_s) over its byte shares | an arm's tokens at equal bytes; tokenize and pack budgets |
| extent margin | per group read in part (synthetic, text, math): train tokens under the arm over the group's draw | below 1 the corpus's extent is too small for the arm (`mixture.SPREAD`) |
| validity (asserted) | every arm holds the same bytes per document; on each source's first 200 val texts the chunks cover the text, tiktoken equals the merges replayed (`train_tokenizer.fidelity`), every text decodes | a failure stops the step |
| line-break entries | entries holding a line break; fusing punctuation with one (`[^\s\p{L}\p{N}][\r\n]`); ending in a line break and indentation | the post's strongest MBPP correlate (Spearman ρ = −0.719 over 58 tokenizers) |
| depth invariance | of each program source's first 50,000 indented val lines, the share whose tokens are the indentation's then the statement's at depth 0 | the hybrid's clause 2 |
| merged reads | merged tokens the val documents read never, < 10, < 100 times | the tie-break (SA-BPE's coverage, arXiv 2604.14053 Table 3, graded) |
| TokEval | `metrics.tokeval` (arXiv 2608.18062; cimeister/tokenizer-intrinsic-evals): compression, fertility, Rényi efficiency, bigram/trigram entropy, UTF-8 integrity, AST boundary alignment, identifier fragmentation, indentation consistency, vocabulary utilization, reconstruction, on each source's leading val documents up to `TOKEVAL_BYTES` | recorded, read by no decision: of its metrics only AST alignment predicts MBPP, attributed "to the vocabulary's line-break handling" (§5), measured above directly |

### Intrinsic results (`-intrinsic.json`)

Every arm passed every validity check. The record predates TokEval's addition: rerun `intrinsic` to record it.

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
with MBPP at any level", so compression does not rank the arms. bpe and sa-bpe compress equally; bpe leaves more
rarely-read merged tokens (84 against 54 under 100).

### Data (`data`, `-data.json`)

The same document set tokenized and packed under every arm (`corpus/tokenize.py`, `pack.py`). Size: a d16 run at
`TOKENS_PER_PARAM` = 20 under sa-bpe (arXiv 2203.15556 Table 3), each source at its share of the base model's stage:
each code source the fewest leading train shards holding its file-stage draw, so code binds every arm; notebooks,
functions and traces their draw, read once; text, math and the repository corpora `MARGIN` = 1.5 times their draws,
so they never bind. Whole shards hold more than the draw:

| Arm | `max_tokens_per_param` | Steps of 294,912 | Repository stage from |
|---|---|---|---|
| sa-bpe | 31.42 | 7,151 | 6,761 |
| sa-bpe-fused | 29.48 | 6,710 | 6,344 |
| hybrid | 31.49 | 7,167 | 6,776 |
| bpe | 31.42 | 7,151 | 6,761 |

### Runs and decision

**Runs:** 4 arms × seeds 0, 1 at d16, each at its `max_tokens_per_param` (equal bytes: each reads the set's code
documents once; text and math by rows), so a better compressor trains on fewer tokens and steps. Contexts 8,192 and
32,768; `offset_bpb` reports bits per byte by byte offset, the context a better compressor sees (arXiv 2503.13423
§3.1, App. B.1).

**Decision** (`analysis`; each challenger against sa-bpe by the shared tests; `-json` `"analysis" "decision"`):
- the control gates: unless sa-bpe-fused parses worse, keep sa-bpe's pattern and score;
- else a challenger qualifies iff it is worse on neither test and better on one (sa-bpe-fused, worse, never does);
- pattern: the hybrid's iff it qualifies, else `SPLIT_PATTERN`;
- score: frequency iff bpe qualifies, or ties on both and leaves fewer merged tokens read < 100 times than sa-bpe;
  else F·ln R. With 84 against 54, frequency needs bpe to qualify.

**`adopt`** writes the decided vocabulary to `CORPUS/tokenizer` over the provisional one (the sa-bpe arm's copy): the
matching arm's files, or, for the frequency score on the hybrid's pattern (no arm), that pair trained on the sample.
Tokenized and packed data are keyed by the vocabulary's data key, so adopting sa-bpe reuses the provisional corpus.

**Limits.** One draw, one vocabulary size; two seeds at d16; the post's evidence is one ~1.27B model with 128K–131K
ids, mostly on MBPP; σ from d8 runs on other data; the control's effect documented up to 50% code; document units of
non-repository text untested; the hybrid changes three clauses at once.

## adamw_group_lr

**Varies.** The AdamW groups' (tables, `lm_head`, vectors) lr × ¼ and × 4 at their decay timescale, Muon's untouched
(`adamw_learning_rate_scale`). Pre-training uses one lr for every group; karpathy/nanochat 92d63d4 uses per-group lrs.

**Arms.** {¼, 1, 4} × seeds 0, 1, d16 on the sa-bpe arm's data at its `max_tokens_per_param`; × 1 is the tokenizer
ablation's sa-bpe runs (same run names), so 4 runs train.

**Rule.** `scaled` over {¼, 1, 4}: `adamw_learning_rate_scale` the adopted scale, else one lr for all groups.

**State.** Waits for the tokenizer runs.

## References

| arXiv | Paper |
|---|---|
| 2203.15556 | Training Compute-Optimal Large Language Models (Chinchilla) |
| 2205.10287 | On the SDEs and Scaling Rules for Adaptive Gradient Algorithms |
| 2207.10551 | Scaling Laws vs Model Architectures: How does Inductive Bias Influence Scaling? |
| 2309.14322 | Small-scale proxies for large-scale Transformer training instabilities |
| 2401.02954 | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism |
| 2402.01035 | Getting the most out of your tokenizer for pre-training and domain adaptation |
| 2405.05417 | Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models |
| 2405.13698 | How to set AdamW's weight decay as you scale model and dataset size |
| 2409.04599 | BPE Gets Picky: Efficient Vocabulary Refinement During Tokenizer Training |
| 2409.19913 | Scaling Optimal LR Across Token Horizons |
| 2411.00640 | Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations |
| 2412.19437 | DeepSeek-V3 Technical Report |
| 2503.04715 | Predictable Scale: Part I, Step Law -- Optimal Hyperparameter Scaling Law in Large Language Model Pretraining |
| 2503.13423 | SuperBPE: Space Travel for Language Models |
| 2504.00178 | Boundless Byte Pair Encoding: Breaking the Pre-tokenization Barrier |
| 2504.11393 | DataDecide: How to Predict Best Pretraining Data with Small Experiments |
| 2505.13738 | Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training |
| 2512.22382 | Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration |
| 2604.14053 | From Where Words Come: Efficient Regularization of Code Tokenizers Through Source Attribution (SA-BPE) |
| 2608.18062 | TokEval: A Tokenizer Evaluation Suite; companion post cimeister.github.io/blog/code_tokenizer_ablations |
| 2610.12144 | Language-Specific Effects of Tokenizer Choice in Multilingual Language Models |
