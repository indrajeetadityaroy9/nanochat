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
|`…/official-69874/{evalplus,multipl-e,lm-eval,infilling}/`|each tool's own samples and verdict files|
|`…/official-69874/compare-eval-69874.json`|per-problem agreement of the official tools with the in-repository harness|
|`artifacts/runs/<base run>/likelihood-<step>.json`|bits per byte per domain at 8K and 32K context, per saved step from 62,887|

`<base run>` is `generate=benchmarks,learning_rate_scale=2.7192599564485445,num_checkpoints=10`.

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
|MBPP 3-shot (lm-evaluation-harness), see below|500|0.184|0.017|[0.150, 0.218]|
|HumanEval infilling, single-line (PSM)|1,033|0.553|—|—|
|HumanEval infilling, multi-line (PSM)|5,815|0.261|—|—|
|HumanEval infilling, random-span (PSM)|1,640|0.113|—|—|

Sampled MultiPL-E is n = 20 at temperature 0.2, top-p 0.95. Infilling is pass@1 by human-eval-infilling's
evaluator over greedy middles that end at the model's end-of-text token (BOS) or the benchmark's longest reference
middle in tokens.

The MBPP 3-shot row is measured with lm-evaluation-harness's unmodified prompt, whose three few-shot solutions keep the
original dataset's CRLF line endings: 11 carriage returns per prompt, in all 500 prompts [measured]. Corpus
standardization maps every "\r\n" and "\r" to "\n" (`corpus/prepare.py`), so the model has never trained on a carriage
return. With them in the prompt, 64 of 500 responses glue text after a block's colon (`def binary_to_decimal(n):n`)
and 131 (26.2%) do not parse [measured]; on EvalPlus, whose prompts hold none, no failure is a syntax error (§3). The
row is superseded by the measurement with the few-shot code's line endings normalized as the corpus normalizes them;
scoring is unaffected, as MBPP judges a response by its tests, not by the reference code.

## 2. Comparison with published base models

Greedy pass@1 (%), each paper's own evaluation; training tokens from the same papers.

|Model|Parameters|Training tokens|HumanEval|HumanEval+|MBPP|MBPP+|MBPP 3-shot|Source|
|---|---|---|---|---|---|---|---|---|
|**d24, step 69,874**|0.20B non-embedding (0.52B)|**0.10T**|**26.8**|**22.0**|**44.4**|**36.5**|18.4 (§1)|this document|
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
  states each task in prose; the corpus is code-heavy, and the model's held-out text is its weakest domain (0.85 bits
  per byte at 8K against 0.29 for code, §4) [inference].
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
  2207.14255 §3, §6).

## 4. Likelihood at the end of pretraining

Held-out bits per byte (lower is better) of each saved model from step 62,887, per domain; 8K is the file-level stage's
context, 32K the repository stage's, whose first saved model is step 66,163 [measured].

|Step|8K all|8K code|8K text|8K math|32K all|32K code|
|---|---|---|---|---|---|---|
|62,887|0.3637|0.2964|0.8614|0.3430|—|—|
|64,000|0.3619|0.2946|0.8584|0.3402|—|—|
|66,063|0.3584|0.2916|0.8522|0.3349|—|—|
|66,163|0.3600|0.2934|0.8524|0.3365|0.3612|0.2924|
|66,463|0.3606|0.2943|0.8513|0.3354|0.3588|0.2897|
|66,863|0.3607|0.2947|0.8501|0.3339|0.3572|0.2880|

- **The file-level stage's decay improves every domain** to step 66,063.
- **The repository stage costs file-level code at 8K for a while**: +1.1% bits per byte on 8K code by step 66,863 while
  8K text and math keep falling and 32K falls fast. The repository stage's data mix is the likely cause [inference].
- **The model was not saturated**: every domain at the stage's own context was still falling at the last steps
  [measured to 66,863]; more tokens would likely keep helping [inference]. The steps to 69,874 complete this table.

## 5. Expectations for post-training

- **Fine-tuning is the documented lever at this scale.** phi-1's base model rose from 29% to 50.6% HumanEval pass@1
  after fine-tuning on under 200M tokens of exercises, and its 350M variant reached 45% (arXiv 2306.11644 abstract,
  §1, §2). The SFT trains explicit endings (`</think>`, then the answer; docs/post-training.md), which addresses the
  13–18% of completions that never stop (§3) [inference].
- **LiveCodeBench v6 will be low single digits** [inference]: after reasoning post-training, MobileLLM-R1-360M scores
  5.1, Qwen2.5-0.5B-Instruct 3.6, and MobileLLM-R1-950M and DeepSeek-R1-Distill-Qwen-1.5B 19.9 (pass@1, n = 16;
  arXiv 2509.24945 Table 7). EvalPlus under the reasoning protocol is the more sensitive measure of the SFT's effect.

## References

| arXiv | Title |
|---|---|
| 2207.14255 | Efficient Training of Language Models to Fill in the Middle |
| 2306.11644 | Textbooks Are All You Need |
| 2401.14196 | DeepSeek-Coder: When the Large Language Model Meets Programming – The Rise of Code Intelligence |
| 2402.19173 | StarCoder 2 and The Stack v2: The Next Generation |
| 2409.12186 | Qwen2.5-Coder Technical Report |
| 2509.24945 | MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes |
