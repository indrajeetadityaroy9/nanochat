# Experiments

Every recipe and implementation idea that was measured and not adopted, adopted on the literature instead of a run, or
considered and not run: what was tried, why, the measurement (record, commit or branch), the verdict by its rule, and
what the code does instead. The adopted choices and their basis live in the owning documents, cited per entry. Markers:
[arithmetic] derived from code or record numbers; [inference] reasoning not measured; [estimate]; [unverified] a figure
without a record, or a paper value its extracted text lacks.

## Code

| Code or record | Section |
|---|---|
| `ablations/tokenizer_split_merge.py` at `67192ab`, removed (`ARMS`, `FUSED_PATTERN`, `HYBRID_PATTERN`; steps `vocabularies`, `intrinsic`, `data`, `repack`) | §2, §7 |
| `results/tokenizer_split_merge-intrinsic.json`, `results/tokenizer_split_merge-data.json` | §2 |
| `results/muon_wd_timescale_20tpp.jsonl`, `results/scalar_learning_rates.json[l]` | §1, §4 |
| Branches `qkv-epi` `0503800`, `mlp-out-mix` `c165408`, `mlp-out-mix-land` `76e32e8`, `mlp-out-mix-v2` `ce30108`, `window-attn` `2bf0e12` | §5 |
| Branch `scalar-learning-rates` `e6e3837` (on `origin`; the run trained at its child `61e316a`) | §4 |
| Commit `f05d17c` (removes the D6/D7 experiments, the tokenizer runs, `RunConfig.muon_tau_scale`, `adamw_learning_rate_scale`, `scalar_learning_rates`, `ablations/contrast.py`) | §1, §2, §4 |
| Commit `7794e2f` (removes the native vLLM model class) | §6 |
| `eval/rescore.py`, `inference/export_hf.py`, `inference/modeling_nanochat.py`, `scripts/standard_eval.sh` at `67192ab`, removed | §6 |

## 1. Protocol and status

**Status.** Every experiment ran as `run.py` `RunConfig` arms trained and evaluated by the base model's code paths of
its commit; depth, seed and tokenizer were `RunConfig` fields until `67192ab`, the code now runs only the base run and
the `peak_lr` calibration (`docs/training.md` §5, §6). Names, records and divergence handling are `docs/training.md`
§6, the measures and their error bars `docs/evaluation.md` §5. The `peak_lr` experiment, whose estimate is every
run's default `learning_rate_scale`, is `docs/training.md` §5.

| Experiment | Decides | Setting | Designed runs | Status |
|---|---|---|---|---|
| `muon_wd_timescale_20tpp` (D7) | Muon's weight-decay timescale at 20 TPP | d16 at 20, sa-bpe data | 3 × 2 seeds | 3 run, stopped; τ law kept (§4) |
| `muon_wd_timescale_200tpp` (D7) | Muon's timescale law (with the 20 result) | d8 at 200 | 3 × 2 seeds | not run (§4) |
| `logit_softcap` (D5) | the output-logit softcap | d16 at 20, sa-bpe data | 2 × 4 seeds | not run; c = 15 adopted on the literature (§3) |
| `tokenizer_split_merge` | split pattern and merge score | d16 at each arm's `max_tokens_per_param` | 4 × 2 seeds | `vocabularies`, `intrinsic`, `data`, `repack` done (module at `67192ab`); runs not run; sa-bpe kept (§2) |
| `adamw_group_lr` (D6) | the AdamW groups' lr | d16 at sa-bpe's `max_tokens_per_param` | 3 × 2 seeds | not run (§4) |
| `scalar_learning_rates` (D6) | per-group lrs of the residual scalars | d16 at 20, sa-bpe data | 1 | run; not adopted (§4) |
| Optimizer screen | NorMuon, Polar Express, cautious weight decay | d16 at 20 | 1 seed per arm | NorMuon adopted (`docs/training.md` §3); the rest §4 |
| GQA screen | 2:1 query:KV heads | d16 at 20 | 1 | rejected (§3) |

The removed experiments' yamls, modules and knobs went in `f05d17c`, the remaining ablation fields and steps after
`67192ab`; the designs below are their record.

**Proxy settings.** No affordable run reaches the base model (d24 at 526.13 tokens per non-embedding parameter,
`docs/training.md` §2). Ranking at a single small size (150M parameters) picks the better of two data recipes at the
1B target in ~80% of comparisons (DataDecide, arXiv 2504.11393 Abstract); architecture rankings can change with scale
(arXiv 2207.10551 §1). The d16 proxy at 20 tokens per parameter on the sa-bpe arm's data trains 4,552 steps of 294,912
tokens at peak lr 4.667e-3 (law × 2.797, the earlier recipe's multiplier, `docs/training.md` §5); karpathy/nanochat's
~320-run optimizer sweep (`dev/LOG.md` 2026-01-19..22) found gains shrinking from ~0.002 bpb at d12 to 0.0007 at d20,
with d12-tuned values hurting at d20 [unverified: external log].

**Shared tests** (were `ablations/contrast.py`, removed in `f05d17c`). An arm against its baseline over the seeds where
both have an evaluation, α = 0.05:
- *Parsing* (MBPP+): per problem, the seeds whose healed greedy completion compiles under the arm minus under the
  baseline; the exact sign test (`scipy.stats.binomtest`) over the non-zero problems (McNemar's exact test at one
  seed, as TokEval's companion post App. D). Better or worse at p < 0.05, else equal. MBPP+ alone: its prompts leave
  the function header to the model, and "an evaluation that uses HumanEval alone does not expose this type of defect"
  (the post, *HumanEval*). The test treats problems as independent within one trained model; between-seed variance is
  not in it: two seeds of one D7 arm give pooled unparsed shares 0.293 and 0.133 (§4).
- *Code bits per byte*: the arms' seed means differ beyond z·σ·√(2/n), z = 1.96, n seeds per arm, else equal; σ =
  0.0034, the pooled seed standard deviation of six d8 runs at 20 tokens per parameter on the reference tokenizer's
  data [unverified: those runs' records are not in `results/`], assumed to carry to other depths and data. Margins
  0.0094 (n = 1), 0.0067 (n = 2), 0.0047 (n = 4) [arithmetic].
- *Adopted*: code better and parsing not worse.

**Kernel landing rule** (§5). A kernel change lands at ≥ +0.3% tok/s at d24 8K in an exclusive A/B (GH200 at 700 W,
arms alternated, arm means), with an op check against fp64 (or bitwise to master) and a 10-step loss equivalence at
the base lr (the earlier recipe's × 2.797) within master's run-to-run spread. The softcap's adoption bar was its loss
cost ≤ 0.3% of the 8K step with a cold peak ≤ 93 GiB. The measured effects of what landed: `docs/architecture.md` §3.

**Cost.** At `c8e97a2`: d8 2.81M / 2.37M tok/s (8K / 32K), d16 0.80M / 0.64M, d24 0.34M / 0.27M; compile 20 s + 16 s;
parse evaluation d8 ~1.7 min, d16 ~4.5 min; d16 benchmark evaluation 1,064 s (862 s after `4385748`) [unverified: no
record]. Per run: d16 at 20 tokens per parameter ~33 min, d16 at 31.4 ~50 min, d8 at 200 ~20 min; all designed
ablations ≈ 20 GPU-h; D7's 200 half on d16 would be 13.4B tokens, ~29 GPU-h [estimate]. Not running the tokenizer runs
saved ~6.7 GPU-h, D7's remaining runs ~3.7 GPU-h, the softcap design ~9 GPU-h [estimate].

## 2. Tokenizer

**`tokenizer_split_merge`.** Would decide the split pattern and merge score of the corpus tokenizer; its runs were not
run, so the corpus tokenizer is the sa-bpe arm (`docs/corpus.md` §5, which holds the vocabulary, trainer, sample,
authorship units and sa-bpe's own numbers; its build step trains sa-bpe alone, `docs/corpus.md` §10). The ablation's
code is removed; it is cited at `67192ab` (`python -m ablations.tokenizer_split_merge <step>`):

| Step | Writes | Record |
|---|---|---|
| `vocabularies` | each arm's vocabulary; the sample's composition | `results/tokenizer_split_merge-vocabularies.json` (`5ba5025`; 737 s for the four arms) |
| `intrinsic` | each arm's tokenized val splits; the measures | `results/tokenizer_split_merge-intrinsic.json` (`ebd8058`) |
| `data` | each arm's tokenized and packed training set | `results/tokenizer_split_merge-data.json` (`ba01bb5`) |
| `repack` | each arm's packs again after a packer change; the data record's `arms` | same record, `seconds.repacked` 176 s |

**Arms** (`ARMS`; the trainer and its F·ln R score: `docs/corpus.md` §5).

| Arm | Split pattern | Merge score | Role |
|---|---|---|---|
| sa-bpe | `corpus/tokenizer.py` `SPLIT_PATTERN` | F·ln R | baseline (the corpus tokenizer) |
| sa-bpe-fused | `FUSED_PATTERN` | F·ln R | positive control: line breaks fused to punctuation |
| hybrid | `HYBRID_PATTERN` | F·ln R | line breaks and indentation as chunks of their own |
| bpe | `SPLIT_PATTERN` | F | SA-BPE's score against frequency |

- **sa-bpe-fused:** karpathy/nanochat's pattern (92d63d4): `SPLIT_PATTERN` plus `[\r\n]*` after the punctuation run,
  so `"):\n"` is one chunk. TokEval's companion post ("Tokenizer Code Ablations", *Differing one regex clause*) deletes
  that clause from GPT-4o's pattern, the tokenizer corpus fixed: MBPP pass@1 0.224, 0.246, 0.220 (gpt4onl) against
  0.167, 0.072, 0.018 (gpt4o), three seeds; across tokenizer corpora of 0–50% code the non-fusing pattern scores
  +0.016 to +0.156 above (one run per cell, exact McNemar per App. D, the 0% cell not significant, p = 0.41). Prompts
  are healed (arXiv 2608.18062 App. C.1). The sample is 85.6% code, beyond the post's highest rung. If the parse test
  missed this effect at d16, it could not decide the other arms.
- **hybrid** (three clauses at once, untested as a whole): (1) `\n+`, `\r+` are chunks of their own, so a line break
  joins neither the punctuation nor the whitespace before it nor the indentation after it (the post's MinGram `cb_nl`,
  MBPP 0.216 against `cb`'s 0.016), and no prompt ends inside a token spanning its last line break (the token-boundary
  bias, arXiv 2412.19437 §4.1); (2) ` +`, `\t+` are chunks of their own and a letter run takes no tab before it, so an
  indentation of ≥ 2 spaces or ≥ 1 tab ends at a chunk end and a statement has the same tokens at every depth
  (`SPLIT_PATTERN`'s `\s+(?!\S)` gives the indentation's last space to the statement, `" if"`, and its
  `[^\r\n\p{L}\p{N}]?+` a tab, `"\tif"`); (3) no contraction clause (Punct, arXiv 2402.01035 App. E, Fig. 3), so
  `'self` is not cut into `'s` `elf`. Kept: one leading punctuation character or space on a letter run (GPT-4's pattern
  gives "an additional 5% compression over Punct at no significant cost to performance", App. E), digit pairs,
  punctuation runs. `HYBRID_PATTERN` has no lookahead, so a long whitespace run costs fancy-regex no backtracking,
  where `SPLIT_PATTERN`'s `\s+(?!\S)` beside `\s*[\r\n]` can stop tiktoken past its backtrack limit.
- **bpe:** SA-BPE's Table 3 F row; close to F·log R in compression, so a run tie was likely, which val coverage breaks.

Not built: tiktoken encodes by merge rank alone, which cannot express PickyBPE's token removals (arXiv 2409.04599
Alg. 1–2) or BoundlessBPE's supermerges (arXiv 2504.00178 §3.1–3.2); SuperBPE moved HumanEval and MBPP pass@10 in
opposite directions, neither significantly (arXiv 2503.13423 Table 1).

**Intrinsic measures** (`intrinsic` at `67192ab`, on each arm's tokenized val splits):

| Measure | Definition | Use |
|---|---|---|
| bytes per token | per source: val bytes over val tokens; file stage: Σ w_s / Σ (w_s / b_s) over its byte shares | an arm's tokens at equal bytes; tokenize and pack budgets |
| extent margin | per group read in part (synthetic, text, math): train tokens under the arm over the group's draw, tokens estimated from bytes | below 1 the corpus's extent is too small for the arm (`mixture.SPREAD`; the tokenized recount: `docs/corpus.md` §1) |
| validity (asserted) | every arm holds the same bytes per document; on the first row group of each source's val split the chunks cover the text, tiktoken equals the merges replayed (`corpus/train_tokenizer.py` `fidelity` at `67192ab`), every text decodes | a failure stops the step |
| line-break entries | entries holding a line break; fusing punctuation with one (`[^\s\p{L}\p{N}][\r\n]`); ending in a line break and indentation | the post's most stable MBPP measurement (Spearman ρ = −0.719 over 58 tokenizers) |
| depth invariance | of each program source's first 50,000 indented val lines, the share whose tokens are the indentation's then the statement's at depth 0 | the hybrid's clause (2) |
| merged reads | merged tokens the val documents read never, < 10, < 100 times | the tie-break (SA-BPE's coverage, arXiv 2604.14053 Table 3, graded) |

**Intrinsic results** (`results/tokenizer_split_merge-intrinsic.json`, `ebd8058`). Every arm passed every validity
check. The record predates `intrinsic` at `67192ab`: it holds no TokEval measures (arXiv 2608.18062; `67192ab`
computes none either), its fidelity check covered each source's first 200 val documents (`67192ab`: the first row
group), and its `seconds` keys are `compile`, `measures`, `all`.

| Measure | sa-bpe | sa-bpe-fused | hybrid | bpe |
|---|---|---|---|---|
| bytes per token, file stage | 3.624 | 3.868 | 3.616 | 3.623 |
| repository | 3.442 | 3.698 | 3.432 | 3.443 |
| synthetic | 3.861 | 4.161 | 3.857 | 3.861 |
| notebooks | 3.403 | 3.617 | 3.413 | 3.403 |
| functions | 4.018 | 4.314 | 4.020 | 4.019 |
| text | 4.243 | 4.295 | 4.237 | 4.230 |
| math | 3.186 | 3.306 | 3.178 | 3.185 |
| val tokens (6.21 GB) | 1,739,851,130 | 1,626,775,376 | 1,744,375,884 | 1,739,939,319 |
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

Depth invariance (% of each program source's first 50,000 indented val lines; codexglue_python has 34,317):

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
with MBPP at any level" (*Code bits per byte*), so compression does not rank the arms. bpe leaves more rarely-read
merged tokens than sa-bpe (84 against 54 under 100).

**Data** (`data`, `results/tokenizer_split_merge-data.json`). The same document set tokenized and packed under every
arm. Size: the §1 d16 proxy run under sa-bpe (`TOKENS_PER_PARAM` = 20, arXiv 2203.15556 Table 3; repository stage
from 4,304), each source at its share of the base model's stage: each code source the fewest
leading train shards holding its file-stage draw, so code binds every arm; notebooks, functions and traces their draw,
read once; text, math and the repository corpora `MARGIN` = 1.5 times their draws, so they never bind. Whole shards
hold more than the draw:

| Arm | `max_tokens_per_param` | Steps of 294,912 (3 micro-batches) | Repository stage from |
|---|---|---|---|
| sa-bpe | 31.42 | 7,151 | 6,761 |
| sa-bpe-fused | 29.48 | 6,710 | 6,344 |
| hybrid | 31.49 | 7,167 | 6,776 |
| bpe | 31.42 | 7,151 | 6,761 |

The record's `run` and `seconds` keep earlier key names (`depth`, `ratio`, `num_iterations`, `total_batch_size`,
`repo_start`; `compile`, `pack`); its `arms`, rewritten by `repack`, the current ones.

**Runs (designed, not run).** 4 arms × seeds 0, 1 at d16, each at its `max_tokens_per_param` (equal bytes), decided by
the shared tests (§1) against sa-bpe with sa-bpe-fused as the positive control, then adoption of the decided
vocabulary. **Verdict: not run; sa-bpe kept.** bpe ties sa-bpe in compression (3.623 against 3.624 bytes per token, as
designed); the fused control's harm is documented (the post: MBPP 0.22–0.25 against 0.02–0.17) and sa-bpe holds no
entry fusing punctuation with a line break; the hybrid compresses 0.2% worse and is untested as a whole, an open idea
rather than a worse one. Within a vocabulary-construction method corpus token count varies little while
pre-tokenization and segmentation change accuracy (arXiv 2402.18376 §5.3); GPT-4's and Punct's patterns show no
significant difference on code at 7B (arXiv 2402.01035 App. E, Table 5; at 1.5B and 32K ids Punct scored higher,
Table 11). **Limits:** one draw, one vocabulary size; two seeds at d16; the post's evidence is one ~1.27B model with
128K–131K ids, mostly on MBPP; σ from d8 runs on other data; the control's effect documented up to 50% code; document
units of non-repository text untested.

## 3. Architecture

**Output-logit softcap as an experiment (`logit_softcap`, D5).** c·tanh(z/c) with c = 15 against none, in the
training loss, the evaluation losses and decoding; z-loss the other option. Design: d16 at 20 tokens per parameter on
the sa-bpe arm's data, 4 paired seeds, adopt iff code bits per byte better by > 0.0047 and MBPP+ parsing not worse;
~9 GPU-h. The only cap-against-none measurement, modded-nanogpt record #9 (−0.0053 nats), is ≈ 0.0021 bpb at
3.6–3.7 bytes per token, which this design detects ~14% of the time [arithmetic: one-sided power at the n = 4
margin, σ = 0.0034]. Verdict: not run; c = 15 adopted on the literature at the fused cap's cost, +0.58% of the 8K
step, above the 0.3% bar (basis, and why no z-loss: `docs/architecture.md` §2; implementation §3).

**Grouped-query attention.** 2:1 query:KV heads at d16 (4 query, 2 KV heads; seed 0) against the optimizer screen's
MHA baselines (§4): last-200 train loss 1.3358 against 1.3244 (s0) / 1.3168 (s1); code bpb 8K / 32K 0.4435 / 0.4254
against 0.4394 / 0.4212 (s1 0.4380 / 0.4196); all bpb 0.5263 / 0.5132 against 0.5219 / 0.5087; HumanEval+ 0.067 /
0.061 against 0.067 / 0.055, MBPP+ 0.090 / 0.071 against 0.074 / 0.061 (within noise); peak 28.7 against 32.0 GiB
[unverified: no record]. Verdict: rejected, loss and bpb worse beyond the seed spread; not committed. All arms ran at
the law's lr (× 1), not the earlier recipe's base × 2.797 [inference: the screen's baselines; §4]. The code keeps MHA
with the KV-cache arithmetic of `docs/architecture.md` §2. Other attention levers ranked with it (more local layers,
attention sinks, gated attention) were not run; full-efficiency short windows in the literature align the window to
tiles, an architecture change (§5, window attention).

**Block Attention Residuals** (arXiv 2603.15031 §3.2): Block AttnRes matches a baseline trained with 1.25× more compute
(Abstract); a 1.1% loss gain at 194M parameters [unverified: not stated in the extracted text, only its scaling table's
194M row]. Not adopted: it would need a native decode path (`docs/evaluation.md` §6) [inference]. Not run.

**Partial key offset** (modded-nanogpt record #49). Not adopted for the same reason [inference]. Not run.

**Multi-token prediction.** Reported to hurt models of this size on code benchmarks (arXiv 2404.19737 Fig. 3). Not
adopted.

## 4. Optimization

**Muon's weight-decay timescale (`muon_wd_timescale`, D7).** Varied Muon's EMA timescale × ½ and × 2 (its weight
decay × 2 and × ½) under the τ law, AdamW's untouched: Power Lines fits the law with AdamW (arXiv 2505.13738 §2.2) and
its App. B states that the EMA view "applies directly" to other decoupled-decay optimizers, without a Muon fit. Design:
{½, 1, 2} × seeds 0, 1 on the §1 d16 proxy and at d8 at 200; per setting the scale's 3-point vertex in ln(scale),
adopted iff its end beat × 1 by the shared tests. Three runs
(`results/muon_wd_timescale_20tpp.jsonl`; no softcap, `final_logit_softcapping` null):

| Run | Code | 32K code | Pooled unparsed | pass@1 | tokens/s | Peak allocated |
|---|---|---|---|---|---|---|
| τ×½ seed 0 | 0.42535 | 0.40585 | 0.293 | 0.096 | 447,572 | 50.9 GB |
| τ×½ seed 1 | 0.42721 | 0.40779 | 0.133 | 0.066 | 868,428 | 39.3 GB |
| τ×1 seed 0 | 0.42724 | 0.40966 | 0.054 | 0.118 | 867,613 | 39.3 GB |

τ×½ seed 0 against τ×1 seed 0: code −0.0019 (one-seed margin 0.0094: equal); MBPP+ unparsed 0.333 against 0.037,
parsing worse (7 problems parse more under it, 119 under the baseline; recomputed from the records' `parse`
`mbpp_plus` `outcomes`). τ×½ seed 0 trained with autograd kept on the step's statistics in `train/diagnostics.py`
(fixed in `acb1e5e`, from seed 1 on): its throughput and peak memory are not comparable; that the update was
unchanged is [inference] (statistics only). Verdict: stopped after 3 of 12 runs; the τ law stays for Muon (basis
`docs/training.md` §3). Specific to this experiment: halving or doubling lr or weight decay does equal or worse
(arXiv 2512.05620 App. H.3); doubling Muon's weight decay (0.1 → 0.2) costs 0.009–0.020 loss (arXiv 2509.02046 App.
C.9 Tables 82–87), ≈ 0.0035–0.008 bpb [arithmetic at 3.62 bytes per token], at or below the margin; a d8 point at 200
is the small-proxy transfer the nanochat sweep warns against (§1).

**AdamW group learning rate (`adamw_group_lr`, D6).** The AdamW groups (the tables `embed_tokens`, `lm_head`, value
embeddings, weight-decayed; the 1-D parameters, not) × ¼ and × 4 with weight decay ÷ the same factor (the tables'
decay timescale held), Muon untouched; karpathy/nanochat 92d63d4 uses per-group lrs. Design: {¼, 1, 4} × seeds 0, 1,
d16 on the sa-bpe arm's data at its `max_tokens_per_param`; the 3-point vertex, adopted iff its end beat × 1 (margin
0.0067); ~4 GPU-h. Verdict: not run; one lr for every group, as Moonlight shares lr and weight decay "among matrix and
non-matrix parameters" (arXiv 2502.16982 §2.2; `docs/training.md` §3).

**Scalar learning rates (`scalar_learning_rates`, D6).** `x0_lambdas` and `resid_lambdas` (1-D, AdamW) at
karpathy/nanochat 92d63d4's lrs, 0.5 and 0.005 × √(B/2^19) (0.375 and 0.00375 at 294,912 tokens per step), every other
group unchanged: a gap from the shared lr too large to bound from the literature. One run (branch
`scalar-learning-rates` `e6e3837`; trained at `61e316a`, `e6e3837` plus `acb1e5e`'s diagnostics fix; recorded at
`60b972d`), d16 at 20, against D7's τ×1 seed 0 (same recipe, trees differing by the field). Rule: adopt iff code
better by > 0.0094 and MBPP+ parsing not worse. Record
(`results/scalar_learning_rates.json` `analysis`, `.jsonl`): code 0.42635 against 0.42724 (−0.0009, `code` equal);
32K code 0.40815 against 0.40966; train loss over the last half −0.0022; MBPP+ parsing worse (9 problems parse more
under it, 35 under the baseline, p = 1.1e-4); pooled unparsed 0.120 against 0.054. Verdict: not adopted; given the
between-seed spread (§1), "worse" is weak evidence, and the rule keeps one lr (`docs/training.md` §3).

**Polar Express** (arXiv 2505.16932). In the d16 optimizer screen at × 1 (setup, baselines and the one-seed caveat:
`docs/training.md` §3): code bpb 8K / 32K 0.4385 / 0.4199 against base s0 0.4394 / 0.4212; last-200 train loss
1.3217 against 1.3244; paired Δ over the last half −0.0019; 825.9K against 828.8K tok/s (−0.35%) [unverified: no
record]. At × 1 Polar Express's larger update (RMS × 1.02–1.25 that of torch's Newton–Schulz on CPU) accounts for
~0.002–0.004 bpb by the lr curvature, its whole gain [inference]. The NorMuon + Polar Express stack at seed 1 lost to
base s1: paired Δ +0.0008 over the last half (22/22 windows positive), code 0.4381 / 0.4196 against 0.4380 / 0.4196,
against an additive prediction of −0.0048. Verdict: dropped; NorMuon alone adopted (`docs/training.md` §3).

**Cautious weight decay** (arXiv 2510.12402). Same screen: lower loss mid-run (−0.022 near 50%), worse at the end
(last-200 +0.0108, code bpb +0.0023 / +0.0043), −1.25% tok/s [unverified: no record]. Verdict: rejected (the paper
finds the best λ unchanged under it).

## 5. Kernels and systems

All measurements at d24 on the GH200 at 700 W, exclusive A/B, arm means, unless stated; the rule is §1's landing
rule; what landed and its effect is `docs/architecture.md` §3, the step profile `docs/training.md` §7. The d8 and d16
proxies ran with their own GEMM tiles (`CONFIGS` at C = 256, 512), value-embedding gate instances (H = 2, 4) and
cross-entropy strip tile (`strip_tile_n`), all at `67192ab`; the code is d24's.

**FP8 GEMMs.** (a) FP8 MLP GEMMs, e4m3 rowwise, d16 on real rows, 300 steps: mean loss over steps 250–299 +0.0048
over bf16 against a bf16 two-run spread of 0.0014, growing. (b) torchao float8: d16 rowwise −28.8%, tensorwise −9.5%;
d24 tensorwise −2.8% (unfused casts). (c) Transformer Engine at d24 8K (40 steps) against the bf16 QuACK baseline
340,283 tok/s, 69.3 GB: MLP delayed scaling 327,018 (−3.9%, 77.1 GB), MLP current 306,165 (−10.0%), MLP + q/k/v/o
delayed 323,765 (−4.9%, 79.3 GB), block scaling 1×128/128×128 ~−14% (84.9 GB); loss within noise. Causes: not the
power cap (TE held higher SM clocks, 1,681–1,750 against 1,521 MHz at ~660 W); DRAM utilization 57% → 64–69% from
separate cast passes and the transposed FP8 copies Hopper's FP8 weight-gradient GEMMs need, which at K = 768 outweigh
the faster GEMMs; +8–10 GB from TE saving the bf16 fc1 output and FP8 transposes; likely lost Inductor fusion
[inference]. (d) cuTile 1.7.0 on sm_90: a naive FP8 GEMM 361–666 TFLOP/s against cuBLAS FP8 856–1,334 on the d24
shapes. DeepGEMM's README requires FP8 casts fused into the producing kernels, which these were not. Verdict: FP8 does
not pay at this size; training is bf16 (`docs/architecture.md` §4). [unverified: no record for (a)–(d)]

**Two runs on one GPU.** d16, 150 steps: one run 808.7K tok/s; two time-sliced processes 743.9K together (−8.0%); two
MPS clients 825.0K (+2.0%). Under MPS an illegal memory access in one client kills every client (a device-side assert
does not); time-sliced processes stay isolated. Verdict: one run saturates the GPU; throughput is measured with the
GPU alone. [unverified: no record]

**Step-level settings** (profile under nsys, d16 and d24, 8K and 32K): `torch.optim.Muon` unbatched −4.3% (the
batched Newton–Schulz stays), max-autotune +0.05%, deterministic mode −14%, cuDNN attention slower than FA4 everywhere,
lm_head on QuACK −10 ms/step with a cold-autotune peak of 89.8 GB, a `ve_dw` rewrite −2.6 ms/step; CUDA graphs cannot
help a GPU-bound step. Verdict: none adopted. [unverified: no record]

**Attention kernels.** Every FA4 forward/backward tile and scheduling variant tried is slower than the defaults;
static persistent schedules lose to the hardware scheduler on varlen tiles. Hand CUDA kernels against FA4 on real d24
segments, ms per call at 8K (32K): window forward 0.369 (0.364) against FA4 0.413 and persistent FA4 0.359 (0.355);
forward with QK-norm + RoPE fused 0.985 (1.029) against prologue + forward 0.546 (0.556), each K block transformed
~5 times; window backward 1.193 (1.294) against FA4 0.771 (0.828); a prologue backward reading FA4's fp32 dq
accumulator 0.823 against 0.405 for the dq post-process plus the plain prologue backward, so the combined op is
−0.90% at 8K. PDL in the qk prologue +0.02%. FA3 not pursued. FlashMLA fuses Q-norm + RoPE into attention (Blackwell),
the evidence that motivated the fused forward. FA4's software exp2 polynomial in place of the SFU in the capped loss:
+2.9 to +3.3 ms. Verdict: rejected; FA4 with the persistent window forward and the separate qk prologue
(`docs/architecture.md` §3). [unverified: no record]

**Window attention (`window-attn` `2bf0e12`).** Real 8K micro-batches: the window forward runs at 332 TFLOP/s true
(446 computed), the backward main kernel at 363 (473), full layers 524–551. A sweep fit puts ~32% of the window forward
and ~27% of its backward in per-tile fixed cost (per CTA 3,160 cycles of load and 6,600 of epilogue, exit and
relaunch) and ~15% in masked waste. Tried: a warpgroup whose 64 keys are fully masked skips its S, dP, dV and dK GEMMs
(bitwise out/dk/dv; +0.27% / +0.18% at 8K / 32K: below the bar); persistent backwards (+44%; +18% / +13% with refill
overlap; +18% / +12% without: one CTA per SM keeps epilogue and next K/V load serial); 64-key edge trimming ≤ 0.2%,
inside the kernel's run-to-run variation (up to 5%). Verdict: rejected; FA4 as is (`docs/architecture.md` §3).

**QK-norm + RoPE in the fused-QKV GEMM epilogue (`qkv-epi` `0503800`).** Would remove the qk-prologue pass (302 MB
per layer and micro-batch, ≤ 66 ms/step by byte arithmetic). A QuACK epilogue on QuACK's SM90 mainloop is correct
(q|k|v bitwise master's; normed q/k vs fp64 as master's, rel 1.661e-3) but takes 2,065 µs per call at 8K (≥ 2,257 at
32K) against master's GEMM + `k_forward` 801 (816); the bar, +0.3% of a 4,058 ms step over 360 calls, is ≤ 767 µs.
Bisect: plain store 544, + aux 660, + v-store predication 649, + per-head rstd 771, + RoPE partner from registers
1,110 µs: the norm alone spends the budget. Two TMA-staged tables in one QuACK epilogue deadlock the GEMM. Verdict:
rejected; the qk prologue stays (`docs/architecture.md` §3). Separately, the qk prologue runs at 3.36 / 3.3 TB/s
forward / backward (~7% headroom), so TileKernels' prologue design was not ported.

**Residual mix + norm in the down-projection epilogue (`mlp-out-mix` `c165408`, `mlp-out-mix-land` `76e32e8`,
`mlp-out-mix-v2` `ce30108`).** Ceiling ≤ 58 ms/step (floor 15) by byte arithmetic. Route (a), the row's norm finished
across CTAs by a last-arriving CTA: deterministic but 969 µs per call against master's `gemm_add` 556 + Inductor's mix
and norm 170 = 726 (the L2 re-read is latency-bound). Route (b), x = fma(λr, bf16(h + acc), x0·λx) in the epilogue and
the norm separate: 587 + 85 = 671 µs; fp32 x0 bitwise master's compiled mix (0 of 75.5M elements differ at layers 1,
12, 22), peak 71.1 → 68.0 GiB. The 10-step check fails for every bf16 x0 path (an RMSNorm op with a bf16 backward, a
bf16 sum of the 23 gradient terms, bf16 x0 in the forward: step-8 losses 9.40437–9.40449 against master's 9.40705,
systematic, since with `o_proj` and `down_proj` zero-initialised the first steps run through x0 → `lm_head`) and
passes with fp32 x0 (≤ 1.6e-5); against fp64 all three trees are equally accurate (median 1.20e-2). A/Bs: fp32 x0
+0.25% / +0.49%, bf16 x0 +0.41% / +0.47%: the passing tree misses the 8K bar and the faster one fails the 10-step.
Master's Inductor backward is bandwidth-bound (198 ms/step at 3.5–3.6 TB/s). The pre-attention norm fold was not
built: n is also dW_qkv's and the gate's saved operand. Verdict: rejected; the Inductor mix and norm stay.

**Norm statistics from the producing epilogue (B2).** Dropped by analysis: the remaining Inductor norm/mix (231
ms/step) runs at ~3.6 TB/s, and deferring the norm into the down projection moves a scaling pass elsewhere (net ≈ 0)
[inference].

**q/k/v weight gradients as one 3C × C GEMM** into a contiguous fp32 buffer instead of three C × C `addmm`: −0.26% /
−0.16% at 8K / 32K [unverified: no record]. Rejected; three `addmm` (`docs/architecture.md` §3).

**GEMM libraries and dW paths.** DeepGEMM on the d24 shapes (TFLOP/s): forward q/k/v/o QuACK 828, torch 705, DeepGEMM
710; up projection QuACK 652, torch 543, DeepGEMM 779 (attributed to drift by an interleaved rerun); dW C×C torch.mm
fp32 662, DeepGEMM fp32 accumulation 295–315 (no split-K in its persistent scheduler: a 768×768 output in 128-row tiles
gives ≤ 36 tiles for 132 SMs; its split-K kernel needs K-major operands). Not adopted as a library. Split-K dW (ms per
call, cuBLAS `addmm` β = 1 → best split-K): C×C 0.194 → 0.184, 4C×C 0.734 → 0.751, C×4C 0.738 → 0.744: −0.04 ms per
layer and micro-batch, inside the run spread: rejected. cuBLASLt with a 16 MiB workspace (DeepGEMM's; torch's is
1 MiB) matches DeepGEMM on the 4C dW shapes, but as a process-global setting lost −0.53% / −0.40% at 8K / 32K, and
scoped to the two 4C dW sites −0.09% / −0.03%: rejected. Hand sm_90a relu² GEMMs (DeepGEMM's producer/consumer layout)
are bitwise QuACK's but slower: backward 2.78 against 2.45 ms, up projection 3.02 against 1.64 ms, 2-CTA multicast 6–7
ms: rejected. Verdict: QuACK and cuBLAS `addmm` (`docs/architecture.md` §3). [unverified: no record]

**Inductor's PDL setting.** −0.17% / −0.05% at 8K / 32K [unverified: no record]: rejected.

**Linear cross entropy's strip GEMMs at d24.** The dX (8,192 × 768, K = 2^15) and dW (2^15 × 768, K = 8,192) GEMMs
of `model/ops/cross_entropy.py` swept over QuACK 0.6.5's SM90 space (`quack/gemm_config.py` `_get_sm90_configs`: tile_N
128–256, cooperative or pingpong, clusters (2, 1) and (1, 2); tile_M 128 and 256) on one real capped d24 chunk;
every output bitwise the production config's. dX: production (128, 192, cluster (2, 1)) within noise of the best,
(256, 192, (2, 1)) −2.3% isolated (0.18 ms per micro-batch); dW: (128, 256, (2, 1)) against production (128, 192,
(1, 2)) −5.5% isolated on random operands, +0.8% on the real chunk, and −0.54 ms (−2.1%) for the whole 98,304-row op
in 5 alternated rounds; tile_M 256 dW +15–45%. At most ~0.5 ms of a ~269 ms d24 micro-batch (≈ 0.2% [estimate]),
below the §1 landing rule: the production config, QuACK's rule (`TILE_N2`, `docs/architecture.md` §3), stays.

**Smaller items not pursued.** A two-rows-per-warp VE-gate forward (199 → 168 µs per call, identical output, ~0.13%
[estimate]); a fused clip-grad-norm and optimizer pass (~13 ms/step [estimate]); FA4's backward pre/post-processing,
q/k norm + RoPE in the projection GEMM, RMSNorm in the next GEMM and dW accumulation in the GEMM, each estimated
≤ 1% at `a81410c` [estimate] — later superseded: RMSNorm in the GEMM landed as `373b549`, dW accumulation as
`f499be7`, the preprocess half of the FA4 item as `db1e90e` (their gains: `docs/architecture.md` §3), while
q/k norm + RoPE in the GEMM lost (`qkv-epi`, above); Liger and Cut Cross-Entropy losses (Cut Cross-Entropy, arXiv
2411.09009, is at best as fast as torch.compile, its gain memory), fused RMSNorm/RoPE libraries, DCP checkpointing.
DeepEP does not apply (one GPU, dense model).

## 6. Evaluation and inference

**Native vLLM model class** (removed in `7794e2f`). d16 served by vLLM 0.32.0 as an out-of-tree model (fused QKV,
vLLM's linear layers): logits against the training forward KL 3.0–4.8e-4, top-1 98.0–99.1%; on the d16 request sets
NativeDecoder 175 s against vLLM's 312 s (vLLM host-bound at 19% GPU); fixed-length decode (2,048 × 512 tokens) d16
NativeDecoder 66.5K against vLLM 31.0K tok/s, d24 21.8K against 23.1K (commits `3c03023`–`ec1c841`). The
Transformers backend ran at the same rate and, in the same execution mode, gave the same logits, so the class and its
plugin image were removed as a second implementation of the model. The export and vLLM path that replaced it:
`docs/evaluation.md` §8.

**Export, vLLM path and scorer replay** (removed at `67192ab`). `inference/export_hf.py` and
`inference/modeling_nanochat.py` ran checkpoints under transformers and vLLM, `scripts/standard_eval.sh` the
harnesses' own CLIs on them, `eval/rescore.py` replayed a record's scorers on its stored completions; each validated
the evaluation (their measurements: `docs/evaluation.md` §1, §8) and none produces a record of the base run or the
`peak_lr` calibration, so the code keeps `NativeDecoder` and the scorers `evaluate` runs (`docs/evaluation.md` §1,
§6). Without the export nothing imports transformers or safetensors.

**Type-constrained decoding** (arXiv 2504.09246). Not used; Python syntax is constrained and types are checked after
generation (basis: `docs/evaluation.md` §7).

**Filtered-mode checks removed** (both d16 checkpoints at 5 tokens per parameter, step 1,707: every HumanEval+ and
MBPP+ task's full 10 candidates as `verify.search` builds them, each scored by EvalPlus and by each check; `2d49bb1`;
a candidate's tokens do not depend on earlier acceptances, so the pools give each rule set's filtered pass@1 exactly).
As the only rejecting check, the kept rules rejected 280 of 3,223 incorrect HumanEval+ candidates and the examples 923
(MBPP+: 3 and 2,379 of 7,517); no subset of the rules changed any task's outcome. Removed:
- `reportReturnType`: rejects 3 HumanEval+ reference solutions (/10, /17 fall through after a loop, /20
  `tuple | None`) and alone rejected no incorrect HumanEval+ candidate (6 of 7,517 on MBPP+).
- the syntax check of the unsanitized prompt + completion: rejected 8 of the 100 candidates that pass EvalPlus plus,
  all cut at the token budget, which `sanitize` recovers and EvalPlus scores correct; removing it gained 1 task in 2 of
  4 benchmark × checkpoint cells, lost none, and decoded 0.7–5.4% fewer tokens.
- the prompts' examples the reference program fails (HumanEval/47's median 15.0, all three of /116's): every correct
  program fails them. The kept checks and their basis: `docs/evaluation.md` §7.

**bf16 KV cache in `NativeDecoder`** (a measurement variant of `decode.cu`: K and V cached as bf16, scales 1, the
scores by bf16 MMAs; the d16 checkpoint above, `81fb386`; greedy continuations of the 542 HumanEval+/MBPP+ prompts,
512 new tokens, against the compiled training forward's). Identical continuations 363 against 216 with the FP8 cache
(for scale: the uncompiled training forward 340, the export under vLLM 379, `docs/evaluation.md` §8); divergence
before the EvalPlus stop 164 against 299; EvalPlus-scored completions differing from the training forward's 25 and
138 (HumanEval+, MBPP+) against 52 and 249; pass@1 HumanEval+ 0.0488 for both, MBPP+ base/plus 0.0238/0.0212 (equal
to the training forward's) against 0.0265/0.0238 with FP8. The cache's bytes per position grow 1.94× (73,728 against
38,016 at d24 [arithmetic]); its decode speed and the full evaluation's pass@1 are not measured. The decoder keeps
the FP8 cache (`docs/evaluation.md` §6).

**Speculative decoding.** Lossless by construction; its gains target large-model serving [inference]. Not used.

**Self-correction prompting.** Not used: the model is a base model without instruction tuning.

## 7. Implementation reference

**`ablations/tokenizer_split_merge.py` at `67192ab`** (the tokenizer ablation, §2; removed). Each arm's vocabulary
was trained by the corpus trainer (`corpus/train_tokenizer.py`: one sample, one trainer, `VOCAB_SIZE` ids, the crate's
score `"sa-bpe"` or `"frequency"`); the corpus tokenizer is sa-bpe (`corpus/tokenizer.py` `TOKENIZER`). SA-BPE's
F·ln R: F the pair's frequency, R the authorship units holding it in the current segmentation, a pair of one unit
scoring 0; ln as in pchizhov/sa-bpe 9fff32d `sabpe/scoring.py`.

- **`FUSED_PATTERN`**: `SPLIT_PATTERN` with cl100k's punctuation clause; the positive control (arXiv 2608.18062
  companion post, App. C.1).
- **`HYBRID_PATTERN`** derivation. BPE merges never cross a chunk end (tiktoken encodes chunk by chunk), so every chunk
  end is a token end; no clause looks behind, so the chunks after a chunk end j are the chunks of the text from j
  alone. Clause (1) `\n+`, `\r+`: a line break shares a chunk only with line breaks of its kind. Clause (2) ` +`,
  `\t+`, `[^\S\r\n\t ]+`: runs of spaces, tabs and other whitespace are chunks, and a letter run takes no tab or line
  break before it, so an indentation of d ≥ 2 spaces or d ≥ 1 tabs ends at a chunk end; a single space before a
  letter run stays its prefix (`" foo"`). Clause (3): the letter clause takes a whole letter run. `[^\S\r\n\t ]+` puts
  every character in some chunk (tiktoken drops unmatched text); without lookahead fancy-regex does not backtrack on
  long whitespace runs.
- **Constants.** `NUM_HIDDEN_LAYERS` = 16: the d16 run the set is sized for. `MARGIN` = 1.5 sits above whole-shard
  rounding of the code shards and the arms' spread in compression. `PROGRAMS` feeds depth invariance. `INDENTED`: an
  indented line is its leading spaces and tabs, then its statement; `INDENTED_LINES` are the first such lines of each
  program source's val split.
- **`vocabularies`**: all arms trained at once in threads on one draw of the sample (each merge an argmax with fixed
  tie-breaks, so arms are independent); writes `CORPUS/tokenizer-ablation/<arm>/{tokenizer.tiktoken,
  tokenizer.pattern, merges.json}`.
  Record: `sample` (documents, bytes, units, seconds, per source: target bytes, fraction of rows taken, documents,
  bytes and units taken), `arms` (pattern, score, merges, seconds, data_key), `seconds`, `provenance`.
- **`provenance`**: commit, UTC time, host, machine, cpus, memory_gb, Python and the versions of tiktoken, tokenizers,
  numpy, scipy, pyarrow and regex.
- **`val_split`/`describe`**: per-token counts and per-document bytes of an arm's tokenized val split; `describe`
  writes `sources` (bytes, tokens, bytes_per_token), `file_stage_bytes_per_token` (all and per group: the weighted
  harmonic mean over `byte_shares`), `train_tokens_over_draw` per group read in part (`mixture.PART`, `IN_PART`;
  `prepare.extent`, `mixture.SPREAD`), `line_break_entries` (line_break, punctuation_line_break,
  line_break_indentation: a line break then indentation ending the entry, the companion post's MinGram `cb`) and
  `merged_reads`. Validity (`fidelity`) runs on the first row group of each source's val split.
- **`intrinsic`**: one process pool runs depth invariance per program source and fidelity per arm and source (neither
  reads tokenized data) beside the tokenize jobs; every val split is tokenized at once (a val split is a few row
  groups, so one at a time would leave the pool idle) and counted as it finishes. Record: `arms` (describe's keys,
  `depth_invariance`, `fidelity`), `seconds` (`tokenized`: wall time to the last tokenize job's end, the checks
  overlapping; `all`), `provenance`.
- **`document_set`**: shard tokens estimated from text bytes at sa-bpe's val bytes per token; each corpus's draw is its
  share of the base model's stage (`Stage.shares` of the sources' train tokens; the repositories corpora, unestimated,
  alike within their group) at the run's tokens in the stage. Code sources take the fewest leading train shards
  holding the file-stage draw; supply sources (`REPO_LEVEL_STAGE.supply`) their draw, read whole once; the others
  `MARGIN` × draw. Returns `file` (leading shards), `repository` (shards after those), `draws` per seq_len; the
  repositories corpora's partitions come from `data`.
- **`pack_corpus`**: the repositories corpora pack all their train shards; other corpora the set's shards for the
  stage; the val split whole.
- **`data`**: each corpus of each arm tokenized then packed as its own job in one process pool;
  `tokenize_repo_level_stage` (under sa-bpe, the trailing partitions holding `MARGIN` × the 32K draws) chooses the
  partitions and tokenizes sa-bpe's repositories corpora while the sources the set check does not gate tokenize; then
  the set check (the file stage's repository shards must not reach the partitions), the repository sources' jobs, and
  the other arms' repositories corpora one arm after another (each fills the cores). Record: `run`, `set` (file,
  repository, partitions, draws), `arms` (`arm_record`: data_key, tokens_per_param, max_steps, batch_size,
  repo_level_start, learning_rate, per stage each corpus's rows, weight, read share), `seconds` (`tokenized`, packs
  overlapping; `all`), `provenance`.
- **`repack`**: after a packer change (`corpus/pack.py`, the crate's `pack.rs`), each corpus of each arm a thread job
  (the crate releases the GIL); rewrites `arms` and `seconds.repacked`.

## References

| arXiv | Paper |
|---|---|
| 2203.15556 | Training Compute-Optimal Large Language Models (Chinchilla) [unverified: paper not in the local collection] |
| 2207.10551 | Scaling Laws vs Model Architectures: How does Inductive Bias Influence Scaling? |
| 2402.01035 | Getting the most out of your tokenizer for pre-training and domain adaptation |
| 2402.18376 | Tokenization Is More Than Compression |
| 2404.19737 | Better & Faster Large Language Models via Multi-token Prediction |
| 2409.04599 | BPE Gets Picky: Efficient Vocabulary Refinement During Tokenizer Training |
| 2411.09009 | Cut Your Losses in Large-Vocabulary Language Models (Cut Cross-Entropy) |
| 2412.19437 | DeepSeek-V3 Technical Report |
| 2502.16982 | Muon is Scalable for LLM Training (Moonlight) |
| 2503.13423 | SuperBPE: Space Travel for Language Models |
| 2504.00178 | Boundless Byte Pair Encoding: Breaking the Pre-tokenization Barrier |
| 2504.09246 | Type-Constrained Code Generation with Language Models [unverified: paper not in the local collection] |
| 2504.11393 | DataDecide: How to Predict Best Pretraining Data with Small Experiments |
| 2505.13738 | Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training |
| 2505.16932 | The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm |
| 2509.02046 | Fantastic Pretraining Optimizers and Where to Find Them |
| 2510.12402 | Cautious Weight Decay [unverified: paper not in the local collection] |
| 2512.05620 | Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales |
| 2603.15031 | Attention Residuals (Block AttnRes) |
| 2604.14053 | From Where Words Come: Efficient Regularization of Code Tokenizers Through Source Attribution (SA-BPE) |
| 2608.18062 | TokEval: A Tokenizer Evaluation Suite; companion post cimeister.github.io/blog/code_tokenizer_ablations |
