# Results

Measured results of the d24 base model and, as they are produced, of its post-trained successors. The base model is
the pretraining run's final checkpoint (step 69,874: all 69,874 updates, 103.03B tokens [arithmetic]; 195,096,576
non-embedding and 522,252,336 total parameters, docs/architecture.md §1). Benchmarks are run by their official tools
(protocol: docs/evaluation.md); this document holds the numbers, their comparison with published models and their
analysis. Markers: [measured] from the records below; [arithmetic] derived from them; [inference] reasoning, not
measured; [unverified] a figure without a record. Paper values are the papers' own tables, cited per row.

## Records

|Record|Holds|
|---|---|
|`artifacts/runs/<base run>/official-69874/summary.json`|per-problem verdicts and pass@1 with standard errors and 95% intervals per benchmark|
|`…/official-69874/{evalplus,multipl-e,lm-eval,infilling}/`|each tool's own samples and verdict files (`lm-eval/`: MBPP 3-shot with normalized line endings, task `mbpp_lf`)|
|`…/official-69874/lm-eval-crlf/`|MBPP 3-shot on lm-evaluation-harness's unmodified CRLF prompts (task `mbpp`)|
|`…/official-69874/compare-eval-69874.json`|per-problem agreement of the official tools with the in-repository harness|
|`artifacts/runs/<base run>/likelihood-<step>.json`|bits per byte per domain at 8K and 32K context, per saved step from 62,887|
|`results/pretrain/trajectory.json`|those steps' bits per byte with standard errors, and each step's paired change from step 66,063 (8K, solutions) or 66,163 (32K) and from the step before (`eval/trajectory.py`)|
|`artifacts/runs/<SFT run>/metrics.jsonl`|the SFT run's train records and its `val` records (held-out response loss)|
|`artifacts/runs/<SFT run>/termination-<step>.json`|the natural-termination probe of an SFT checkpoint (docs/evaluation.md §10): per-set summaries and every response's ending; `termination-1000-v1.json` the first step-1000 probe, before the per-response fields; `termination-2000-chunked.json` the step-2000 probe in 2,048-token rounds with loop stopping|
|`artifacts/runs/<SFT run>/termination-budget-<step>.json`|budget forcing and sampling sensitivity of an SFT checkpoint (docs/evaluation.md §10)|
|`artifacts/runs/<SFT run>/serving-check-<step>.json`|the exported checkpoint in vLLM against the training model (docs/evaluation.md §10)|
|`artifacts/runs/<SFT run>/loop-onsets-<step>-chunked.json`, `loop-onsets-budget-<step>.json`|each never-closed response's loop onset (`eval/official_termination.py` `--onsets`), from the probe records that store the reasoning|
|`results/posttrain/sft-data-repetition.json`|the SFT documents' reasoning repetition per threshold and reasoning-length bucket (`--data-repetition`)|

`<base run>` is `generate=benchmarks,learning_rate_scale=2.7192599564485445,num_checkpoints=10`; `<SFT run>` is
`base=<base run>,generate=none,learning_rate_scale=0.06449599703255417,sft_epochs=2.0`.

## 1. Base model, step 69,874

Greedy pass@1 unless noted; vLLM 0.32.0 on the exported checkpoint (bf16 weights, fp32 output head); the interval is
the 95% normal interval of the problem mean [measured].

|Benchmark (tool)|n|pass@1|SE|95% interval|
|---|---|---|---|---|
|HumanEval (EvalPlus)|164|0.268|0.035|[0.200, 0.336]|
|HumanEval+ (EvalPlus)|164|0.220|0.032|[0.156, 0.283]|
|MBPP (EvalPlus)|378|0.444|0.026|[0.394, 0.495]|
|MBPP+ (EvalPlus)|378|0.365|0.025|[0.316, 0.414]|
|EvalPlus average (the four above, per problem)|542|0.324|0.020|—|
|HumanEval-C++ (MultiPL-E), greedy / sampled|161|0.217 / 0.220|0.033 / 0.029|[0.153, 0.281] / [0.164, 0.276]|
|MBPP-C++ (MultiPL-E), greedy / sampled|397|0.368 / 0.351|0.024 / 0.021|[0.320, 0.415] / [0.309, 0.393]|
|MBPP 3-shot (lm-evaluation-harness, task `mbpp_lf`), see below|500|0.322|0.021|[0.281, 0.363]|
|HumanEval infilling, single-line (PSM)|1,033|0.553|—|—|
|HumanEval infilling, multi-line (PSM)|5,815|0.261|—|—|
|HumanEval infilling, random-span (PSM)|1,640|0.113|—|—|

Sampled MultiPL-E is n = 20 at temperature 0.2, top-p 0.95. Infilling is pass@1 by human-eval-infilling's
evaluator over greedy middles that end at the model's end-of-text token (BOS) or the benchmark's longest reference
middle in tokens.

lm-evaluation-harness's unmodified MBPP prompt keeps the original dataset's CRLF line endings in its three few-shot
solutions: 11 carriage returns per prompt, in all 500 prompts [measured]. Corpus standardization maps every "\r\n" and
"\r" to "\n" (`corpus/pretrain/prepare.py`), so the model has never trained on a carriage return. The row above is the
task `mbpp_lf`, the same prompt with its fields' line endings normalized as the corpus normalizes them
(`eval/official_lm_eval/`, docs/evaluation.md §4); scoring is unaffected, as MBPP judges a response by its tests, not
by the reference code. Same model, settings and 500 problems [measured]:

|MBPP 3-shot prompt|pass@1|Responses not parsing|Text glued after a block header's colon|Responses at the 256-token limit|
|---|---|---|---|---|
|CRLF (task `mbpp`)|0.184|132|133|25|
|LF (task `mbpp_lf`)|0.322|10|6|15|

A glued response holds a `def`, `class`, `if`, `elif`, `else`, `for`, `while`, `try`, `except`, `finally` or `with`
line whose colon is followed by a character that is neither whitespace nor `#` (`def binary_to_decimal(n):n`). Per
problem, CRLF and LF both pass 83, CRLF only 9, LF only 78, neither 330 (exact McNemar p = 7.5e-15); against the
in-repository harness, which prompted with the CRLF shots (docs/evaluation.md §8), LF passes 70 problems the harness
fails and fails 12 it passes (p = 4.1e-11) [measured]. On EvalPlus, whose prompts hold no carriage return, no failure
is a syntax error (§3).

## 2. Comparison with published base models

Greedy pass@1 (%), each paper's own evaluation; training tokens from the same papers.

|Model|Parameters|Training tokens|HumanEval|HumanEval+|MBPP|MBPP+|MBPP 3-shot|Source|
|---|---|---|---|---|---|---|---|---|
|**d24, step 69,874**|0.20B non-embedding (0.52B)|**0.10T**|**26.8**|**22.0**|**44.4**|**36.5**|32.2 (§1)|this document|
|Qwen2.5-Coder-0.5B|0.5B|5.5T (continued pretraining)|28.0|23.8|52.9|47.1|40.4|arXiv 2409.12186 §6.1|
|Qwen2.5-0.5B|0.49B|—|28.1|—|—|—|29.6|arXiv 2509.24945 Table 6|
|MobileLLM-R1-360M base|0.36B|4.2T|32.9|—|—|—|20.8|arXiv 2509.24945 Table 6|
|SmolLM2-360M|0.36B|—|0.0|—|—|—|19.4|arXiv 2509.24945 Table 6|
|StarCoderBase-3B|3B|—|21.3|17.1|42.6|35.8|—|arXiv 2402.19173 Table 9|
|DeepSeek-Coder-1.3B|1.3B|2T|34.8|26.8|55.6|46.9|46.2|arXiv 2409.12186 §6.1|
|DeepSeek-Coder-1.3B|1.3B|2T|28.7|23.8|55.4|46.9|—|arXiv 2402.19173 Table 9|
|StarCoder2-3B|3B|3.1T|31.7|27.4|57.4|47.4|—|arXiv 2402.19173 Table 9|

DeepSeek-Coder's token count is arXiv 2401.14196's abstract. The same model's HumanEval differs by 6.1 points between
the two papers that report it, so a single-benchmark comparison across papers carries several points of protocol
variance on top of this model's own ±6.8-point interval on HumanEval (§1).

- **Token efficiency.** HumanEval and HumanEval+ equal the 0.5B models' within the interval (26.8 / 22.0 against
  Qwen2.5-Coder-0.5B's 28.0 / 23.8) at 1/53 of Qwen2.5-Coder's continued-pretraining tokens and 1/19 of
  DeepSeek-Coder's [arithmetic]; all four EvalPlus scores exceed StarCoderBase-3B's.
- **MBPP lags.** MBPP and MBPP+ are 8.5 and 10.6 points below Qwen2.5-Coder-0.5B, where HumanEval is level. MBPP
  states each task in prose; the corpus is code-heavy, and the model's held-out text is its weakest domain (0.84 bits
  per byte at 8K against 0.29 for code, §4) [measured; the cause: inference].
- **C++ follows Python.** HumanEval-C++ is 0.81 of HumanEval and MBPP-C++ 0.83 of MBPP (greedy) [arithmetic].

## 3. Failure analysis

Each EvalPlus problem in one class: passed; passed the base tests only; else a budget-exhausted completion (≥ 760 of
EvalPlus's 768 new tokens, the model never stopped), unparseable code, the entry point undefined, a timeout, or, by
re-running its first failing base test, a runtime exception or a wrong output. MultiPL-E's classes are its judge's
statuses [measured].

|Class|HumanEval+ (164)|MBPP+ (378)|
|---|---|---|
|Passed|22.0%|36.5%|
|Passed base tests, failed the plus tests|4.9%|7.9%|
|Wrong output|51.8%|30.7%|
|Budget exhausted (never stopped)|13.4%|17.7%|
|Runtime exception|7.9%|5.6%|
|Entry point undefined|0%|1.3%|
|Timeout|0%|0.3%|
|Unparseable code|0%|0%|

|Judge status (MultiPL-E, greedy)|HumanEval-C++ (161)|MBPP-C++ (397)|
|---|---|---|
|OK|21.7%|36.8%|
|Exception (failed assertion or crash)|70.8%|47.9%|
|Compile error|6.8%|14.9%|
|Timeout|0.6%|0.5%|

- **Wrong logic dominates.** Half of HumanEval+ fails on a wrong output; the exceptions are mostly `TypeError` and
  `NameError`.
- **One completion in seven never ends.** 13–18% of the EvalPlus completions run to the token budget.
- **Python syntax is not a failure mode**: no EvalPlus failure is unparseable. C++ compile errors are 7–15%.
- **Infilling's failures are stopping failures.** In single-line infilling 38.3% of the greedy middles span more than
  one line, i.e. the model did not emit its end-of-text token when the hole was filled [measured], and the 5.6% of
  verdicts that are syntax errors are over-generation's [inference]. Bavarian et al. describe the failure: a model that
  "fails to generate an <EOT> token" connects badly to the suffix, "a common failure in FIM sampling" (arXiv
  2207.14255 §3, §6). Middles that ran to the budget, by `summary.json` `at_budget` (tokens re-encoded from the text):
  136 of 1,033 single-line (13.2%), 835 of 5,815 multi-line (14.4%) and 1,029 of 1,640 random-span (62.7%); the
  others ended at BOS or another special token [measured].

## 4. Likelihood at the end of pretraining

Held-out bits per byte (lower is better) of each saved model from step 62,887, per domain; 8K is the file-level stage's
context, 32K the repository stage's, whose first saved model is step 66,163 (`results/pretrain/trajectory.json`
`bpb`) [measured].

|Step|8K all|8K code|8K text|8K math|32K all|32K code|32K text|32K math|
|---|---|---|---|---|---|---|---|---|
|62,887|0.3637|0.2964|0.8614|0.3430|—|—|—|—|
|64,000|0.3619|0.2946|0.8584|0.3402|—|—|—|—|
|66,063|0.3584|0.2916|0.8522|0.3349|—|—|—|—|
|66,163|0.3600|0.2934|0.8524|0.3365|0.3612|0.2924|0.8531|0.3363|
|66,463|0.3606|0.2943|0.8513|0.3354|0.3588|0.2897|0.8519|0.3351|
|66,863|0.3607|0.2947|0.8501|0.3339|0.3572|0.2880|0.8507|0.3336|
|67,000|0.3608|0.2948|0.8496|0.3337|0.3568|0.2876|0.8502|0.3334|
|68,000|0.3602|0.2946|0.8466|0.3305|0.3543|0.2852|0.8472|0.3302|
|69,000|0.3592|0.2938|0.8439|0.3280|0.3525|0.2836|0.8445|0.3277|
|69,874|0.3584|0.2932|0.8424|0.3266|0.3515|0.2827|0.8430|0.3263|

Paired change at step 69,874 from step 66,063, the last file-level model, over identical rows (`change`; ± one
standard error) [measured]:

|8K all|8K code|8K repositories|8K synthetic|8K notebooks|8K functions|8K text|8K math|
|---|---|---|---|---|---|---|---|
|+0.00003 ± 0.00003|+0.00162 ± 0.00004|−0.00413 ± 0.00004|+0.01340 ± 0.00004|+0.00248 ± 0.00012|+0.00293 ± 0.00108|−0.00978 ± 0.00003|−0.00829 ± 0.00004|

- **The file-level stage's decay improves every domain** to step 66,063.
- **The repository stage costs 8K code, and recovers it only in part.** 8K code rises 0.0032 (+1.1%) by step 67,000
  while 8K text and math keep falling and 32K falls fast; it then falls again but ends at 0.2932 against step
  66,063's 0.2916 (paired change +0.0016, +0.6%). The 8K average is back to step 66,063's 0.3584 (paired change
  +0.00003 ± 0.00003) because text (−0.0098) and math (−0.0083) gained what code lost. Within 8K code the repository
  group improved (−0.0041) and the groups only the file-level stage trains worsened: synthetic code +0.0134,
  notebooks +0.0025, functions +0.0029 [measured]. The repository stage's mixture holds none of those three groups
  (`docs/corpus.md` §7), the likely cause [inference].
- **The benchmark solutions improve through the repository stage**: per-byte bits of the reference solutions fall
  from step 66,063 to 69,874 by 0.0104 ± 0.0014 (HumanEval+) and 0.0250 ± 0.0016 (MBPP+); HumanEval-X C++ is
  unchanged (−0.0007 ± 0.0007) [measured].
- **The model was not saturated**: at 32K every domain was still falling at the last steps (all 0.3525 → 0.3515 from
  step 69,000 to 69,874, at learning rates below 1.3% of the peak [arithmetic]); more tokens would likely keep helping
  [inference].

## 5. Expectations for post-training

- **Fine-tuning is the documented lever at this scale.** phi-1's base model rose from 29% to 50.6% HumanEval pass@1
  after fine-tuning on under 200M tokens of exercises, and its 350M variant reached 45% (arXiv 2306.11644 abstract,
  §1, §2). The SFT trains explicit endings (`</think>`, then the answer; docs/post-training.md), which addresses the
  13–18% of completions that never stop (§3) [inference].
- **LiveCodeBench v6 will be low single digits** [inference]: after reasoning post-training, MobileLLM-R1-360M scores
  5.1, Qwen2.5-0.5B-Instruct 3.6, and MobileLLM-R1-950M and DeepSeek-R1-Distill-Qwen-1.5B 19.9 (pass@1, n = 16;
  arXiv 2509.24945 Table 7). EvalPlus under the reasoning protocol is the more sensitive measure of the SFT's effect.

## 6. Post-training

The SFT run (docs/post-training.md §4) from the base model of §1, and the stages after it (docs/post-training.md §8).
Each table has one row per evaluated checkpoint; later checkpoints and the post-SFT model append rows. The probe's
method is docs/evaluation.md §10: 64 HumanEval+ prompts in EvalPlus's chat rendering and 64 held-out SFT prompts, one
seeded sample each at T 0.6, top-p 0.95, up to the 32,768-token context.

### 6.1 Held-out loss

Mean next-token loss (nats) on the SFT val split's responses, 53,474,648 labels [measured: `metrics.jsonl` `val`].

| Model | Step | Held-out loss |
|---|---|---|
| base (§1) | 0 | 1.1574 |
| SFT | 1,000 | 0.8533 |
| SFT | 2,000 | 0.8139 |

### 6.2 Natural termination

Exclusive outcomes: closed `</think>` and ended with BOS / closed, the answer at the limit / never closed, at the
limit / never closed, ended [measured: `termination-<step>.json`, one-shot generation].

| Step | Set | Closed, ended | Closed, at limit | Unclosed, at limit | Unclosed, ended | P(closed ≤ 1K / 2K / 4K / 8K / 16K / 32K) |
|---|---|---|---|---|---|---|
| 1,000 | HumanEval+ (64) | 51 | 0 | 13 | 0 | 0.484 / 0.766 / 0.797 / 0.797 / 0.797 / 0.797 |
| 1,000 | held-out SFT (64) | 37 | 1 | 26 | 0 | 0.156 / 0.344 / 0.500 / 0.594 / 0.594 / 0.594 |
| 1,000 | all (128) | 88 | 1 | 39 | 0 | 0.320 / 0.555 / 0.648 / 0.695 / 0.695 / 0.695 |
| 2,000 | HumanEval+ (64) | 44 | 0 | 20 | 0 | 0.375 / 0.625 / 0.672 / 0.672 / 0.688 / 0.688 |
| 2,000 | held-out SFT (64) | 29 | 0 | 35 | 0 | 0.156 / 0.266 / 0.391 / 0.453 / 0.453 / 0.453 |
| 2,000 | all (128) | 73 | 0 | 55 | 0 | 0.266 / 0.445 / 0.531 / 0.563 / 0.570 / 0.570 |

- **Closure fell from step 1,000 to 2,000**: 89 → 73 of 128 closed; per prompt (same prompts and seeds), 26 closed
  only at 1,000 and 10 only at 2,000, exact McNemar p = 0.011 [measured; arithmetic], while the held-out loss fell
  (§6.1).
- **Closure is early or not at all**: the latest closure is at 6,361 reasoning tokens at step 1,000 and 11,355 at
  step 2,000; every closed answer holds a fenced code block, and all but one (step 1,000) end with BOS.
- **The never-closed responses loop**: at step 1,000, 36 of 39 have ≥ 99.85% of their last 4,096 tokens inside a
  32-token window that recurs there; the other 3 are enumerations with an incrementing number ("The combination
  (1,2,3,4,571) … (1,2,3,4,577)") that never repeat a window exactly. At step 2,000, 53 of 55 repeat ≥ 99% of their
  last 4,096 tokens [measured]. Loop onset (the first token of the repeating tail, `loop_onset`) has median 1,342
  tokens on HumanEval+ (18 loops) and 2,525 on held-out prompts (36) in the protocol sampling of
  `termination-budget-2000.json`, and 1,332 on HumanEval+ (19) in the chunked probe [measured:
  `loop-onsets-budget-2000.json`, `loop-onsets-2000-chunked.json`; the stored reasoning re-encoded].
- **Loops are near-deterministic**: in greedy generations from held-out prompts the chosen tokens' mean
  log-probability is −0.0016 nats at 2–4K tokens and −0.0009 at 4–8K, identical in both paths (§6.3) [measured:
  `serving-check-2000.json` decode buckets]. The SFT data hold almost no loops: 8 of the 797,950 train documents
  have reasoning repetition ≥ 0.8 (all with reasoning ≥ 8K tokens; 0 of the 8,013 val documents), 1.0e-5
  [measured: `results/posttrain/sft-data-repetition.json`; arithmetic].
- **Chunked probe** (`termination-2000-chunked.json`: 2,048-token rounds, a never-closed response stopped once its
  last 4,096 tokens repeat ≥ 99%): 68 of 128 closed (42 HumanEval+, 26 held-out), 54 loop-stopped, 6 at the limit;
  against the one-shot probe's 73 and the 16K-capped protocol replicate's 70 (§6.4). Per prompt it agrees with the
  one-shot probe on 119 of 128 closures; its generation took 387 s against 1,102 s [measured]. Every round after the
  first reseeds the sampler (docs/evaluation.md §10), so per-prompt differences are sampling noise [inference].
- **Held-out closure falls with the teacher's reasoning length** (rank correlation −0.43 at step 1,000, −0.53 at
  2,000 [arithmetic]); closed of prompts:

  | Step | < 2K (17) | 2–4K (8) | 4–8K (14) | 8–16K (19) | ≥ 16K (6) |
  |---|---|---|---|---|---|
  | 1,000 | 17 | 5 | 6 | 8 | 2 |
  | 2,000 | 13 | 4 | 8 | 4 | 0 |

- **By source** (closed of prompts; the table below).

| Step | Exercism (5) | KodCode online-judge (2) | instruct (7) | complete (11) | Cascade Stage-1 (18) | Stage-2 (15) | CP-v2 C++ (2) | Python (4) |
|---|---|---|---|---|---|---|---|---|
| 1,000 | 5 | 2 | 5 | 5 | 11 | 8 | 1 | 1 |
| 2,000 | 5 | 1 | 4 | 6 | 8 | 3 | 1 | 1 |

### 6.3 Serving faithfulness

The `--chat` export in vLLM against the training model (docs/evaluation.md §10) [measured: `serving-check-2000.json`].

| Step | Prefill: largest bucket difference in mean NLL (0–512 … 16–32K; 98,470 tokens) | Decode argmax agreement 0–512 / 512–2K / 2–4K / 4–8K | Mean chosen-token log-probability, vLLM / training |
|---|---|---|---|
| 2,000 | 0.0009 nats (0–512); ≤ 0.0006 elsewhere | 98.7% / 99.85% / 100% / 100% | equal within 0.001 nats in every bucket |

Disagreements are near ties: their margins average 0.07 nats at 0–512 and 0.05 at 512–2K, at most 0.42, and do not
grow with position [measured]. The loops of §6.2 are therefore the model's, not the serving path's [inference].

### 6.4 Budget forcing and sampling sensitivity

HumanEval+ (the probe's 64 problems), one sample each, reasoning stopped at `</think>` or 16,384 tokens; forced
closure at B appends `</think>` to the first B reasoning tokens of a response not closed by B (docs/evaluation.md
§10); pass counts on the base / plus tests [measured: `termination-budget-<step>.json`].

| Model | Sampling | Closed by 16K (HumanEval+ / held-out) | Loops (HumanEval+ / held-out) | Natural closures: base / plus | Forced at B = 4K / 8K / 16K: all 64, base / plus |
|---|---|---|---|---|---|
| SFT 2,000 | T 0.6, top-p 0.95 (protocol) | 42 / 28 | 18 / 36 | 26 / 26 of 42 | 34 / 33, 34 / 33, 33 / 31 |
| SFT 2,000 | T 0.8, top-p 0.95 | 46 / 32 | 16 / 18 | 29 / 29 of 46 | 16K: 32 / 32 |
| SFT 2,000 | T 1.0, top-p 0.95 | 59 / 44 | 3 / 4 | 34 / 32 of 59 | 16K: 35 / 32 |
| SFT 2,000 | T 0.6, top-p 0.95, top-k 20, presence penalty 1.0 (Qwen3's) | 55 / 31 | 9 / 26 | 32 / 29 of 55 | 16K: 35 / 31 |
| base (§1) | greedy, completion protocol | — | — | — | 37 / 31 |

- **Forcing recovers few answers**: at the protocol's sampling, 8 / 7 of the 22 unclosed responses pass when closed
  at 4K or 8K, 7 / 5 at 16K; at T 1.0 only 5 are forced and 1 / 0 pass.
- **No setting differs from the base beyond noise on these 64**: the SFT checkpoint's best plus count is 33 against
  the base's 31, its best base-test count 35 against 37. One sample per problem at T > 0 varies by about ±2–3 passes
  per 64 [estimate: most problems pass always or never, so only the rest contribute binomial variance], and vLLM's
  seeded sampling is not batch-invariant, so a rerun with the same seeds need not reproduce a row.
- **Temperature removes loops** (HumanEval+ loops 18 → 3 from T 0.6 to T 1.0) at no measured cost in passes. The
  evaluation-time settings these rows select are disclosed secondary settings (docs/post-training.md §5).

### 6.5 Scoring correction

EvalPlus's tests scored inside a process forked from vLLM's fail tests that allocate much: its address-space guard
(RLIMIT_AS) counts the vLLM process's mappings, so HumanEval/15's plus input failed with MemoryError. The plus counts
stored in `termination-1000.json` and `termination-2000.json` are therefore low by at least one [measured: the stored
answers re-scored in a clean process]; §6.4's counts and every later record are scored in a clean CPU process
(docs/evaluation.md §10). The same guard, inflated by per-core thread pools, affected the base's scoring
(docs/evaluation.md §4).

## References

| arXiv | Title |
|---|---|
| 2207.14255 | Efficient Training of Language Models to Fill in the Middle |
| 2306.11644 | Textbooks Are All You Need |
| 2401.14196 | DeepSeek-Coder: When the Large Language Model Meets Programming – The Rise of Code Intelligence |
| 2402.19173 | StarCoder 2 and The Stack v2: The Next Generation |
| 2409.12186 | Qwen2.5-Coder Technical Report |
| 2509.24945 | MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes |
