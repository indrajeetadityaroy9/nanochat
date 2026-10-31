# Evaluation and inference

How a checkpoint is scored: the likelihoods, the parse measure and the 1@10 mode by the repository's own evaluation,
the code benchmarks by their official tools on the exported checkpoint; every metric and error bar the repository
computes (`metrics.py`), the decoder, the syntax constraint and checks of the 1@10 mode, and the Hugging Face export
with its agreement and the cross-check of the two scorings.

Markers: [arithmetic] derived from code numbers; [inference] reasoning, not measured; [estimate]; [unverified] a
figure without a record, or a paper value its extracted text lacks. Measurements give their setting
(hardware, checkpoint, protocol); "d16 checkpoint" is a 20-tokens-per-parameter d16 run's final checkpoint.

## Code

| File | Section |
|---|---|
| `eval/suite.py` (`evaluate`, `record`, `evaluation_path`, `pin`, `resolved`, `source_digest`, `write_atomically`) | §1 |
| `eval/bpb.py` (`likelihood`, `val_bits`, `solution_bits`, `sums`, `sample_rows`, `row_clusters`) | §2 |
| `eval/trajectory.py` (`main`, `score`, `trajectory`, `rows_by_measure`, `groups`, `summaries`, `likelihoods`, `saved_steps`, `likelihood_path`) | §2 |
| `eval/code_eval.py` (`tasks`, `parse_generation`, `parse_record`, `truncated`, `filtered_generation`, `filtered_completions`, `codegen_runs`, `vetted`, evaluator wrappers: `evalplus`, `multipl_e`) | §1, §3, §7 |
| `scripts/official_eval.sh`, `docker/Dockerfile.eval`, `eval/official.py`, `eval/official_infilling.py`, `eval/official_site/sitecustomize.py` | §4, §8 |
| `scripts/official_reasoning.sh`, `scripts/official_termination.sh`, `scripts/official_termination_watch.sh`, `eval/official_reasoning_parser.py`, `eval/official_lcb.py`, `eval/official_tokenize.py`, `eval/official_evalplus.py`, `eval/official_reasoning.py`, `eval/official_termination.py` | the post-trained model's protocol, `docs/post-training.md` §5; its requests, §8 |
| `eval/verify.py` (`search`, `Search`, `python_diagnostics`, `compile_error`, `solution_diagnostics`, `cpp_diagnostics`, `compiler`, `examples_pass`, `vetted_examples`), `eval/cache.py` (`VerdictCache`) | §7 |
| `metrics.py` (every function); `results/pretrain/mbpp_3shot_contamination.json` (MBPP 3-shot's overlap with the train corpus) | §5; §4 |
| `inference/decoder.py` (`NativeDecoder`, `Request`, `heal_tokens`, `Reemit`), `model/kernels/decode.cu` | §6 |
| `inference/syntax_constraint.py`; `inference/syntax/` (`lib.rs`, `lexer.rs`, `parser.rs`, `vocab.rs`, `python.lark`, `Cargo.toml`, `Cargo.lock`) | §7 |
| `inference/export_hf.py` (`export`), `inference/modeling_nanochat.py` (`NanochatForCausalLM`, `NanochatModel`, `NanochatConfig`) | §8 |
| `eval/__init__.py`, `inference/__init__.py` (empty package markers) | — |

## 1. Protocol

`eval/suite.py` `evaluate(tokenizer, run_name=, step=, generate=, previous=, released=)` loads the checkpoint onto the
GPU and writes `RUNS/<run>/eval-<step>.json`: the likelihood metrics (`"8k"`, `"32k"`, `"solution"`, §2), the parse
measure (`"parse"`, §3) and, with `generate="benchmarks"`, the 1@10 mode (`"codegen"` `<benchmark>."1@10"` and
`"evalplus_average"` `"1@10"`, §4, §7), null with `generate="parse"`; `"provenance"` holds a SHA-256 over the code and
configuration files (`source_digest`), the Python version and installed distributions (the records in `results/`
predate it). Base-model checkpoints are evaluated with `"benchmarks"`, ablation runs with `"parse"`, SFT runs not at
all (`generate="none"`, `docs/post-training.md` §5); `record` gathers a run's non-train log records and
evaluations into `results/<experiment>.jsonl` (both: `docs/training.md` §6). The code benchmarks' standard protocols
run outside `evaluate`, in their official tools on the exported checkpoint (`scripts/official_eval.sh`, §4). Earlier
records also hold the harness's removed generation modes (greedy, sampled, MBPP 3-shot, infilling, SantaCoder FIM;
`docs/experiments.md` §6) and a `"samples"` sidecar of the sampled completions.

**Pipelined scoring.** `evaluate` runs the parse decode, the likelihood pass, then the filtered searches (§7); each
benchmark's official evaluator (CPU, a fresh process) runs on a scorer thread as soon as its completions exist. The
last `DECODE_CORES` = 16 cores of the process's affinity hold the evaluation's process tree; every timed check
(basedpyright, g++, example runs, the official evaluators) runs on the other cores, at most one timed child per core
(`verify.timed_slots`), as EvalPlus asks of a test bed (its `docs/execution.md` "Parallelism"). Basis of 16
[unverified: no record]: a d24 benchmarks evaluation (GH200, 64 cores) used 13.0–15.5 cores (30-s means) in the
filtered searches. Checkpoints overlap (`run.py` `evaluations`): once checkpoint k's GPU work ends, its process frees
the GPU, moves to the timed-check cores (`pin`) and sets `released`; k + 1's official scoring and filtered searches
wait for k's process to end (`previous`). Results do not depend on this order (a request's tokens depend on the
model and the request alone, §6) except timed checks, which heavy CPU load can fail.

## 2. Likelihood metrics

`eval/bpb.py` `likelihood` runs the compiled `CausalLM.losses`. All values are bits per byte (§5 `bpb`: summed bits
over summed UTF-8 bytes of the targets; tokenizer-independent, arXiv 2101.00027 §3.1). Targets of zero bytes (BOS,
padding, FIM sentinels) are excluded from both sums. Sums are fp64 on the device, reduced in a fixed order (no
atomics).

- **Held-out corpora** (`val_bits`): each corpus's packed val split (`docs/corpus.md` §8) in the stage's rows: at 8K
  (`FILE_LEVEL_STAGE`) always, at 32K (`REPO_LEVEL_STAGE`) only for a model at the 32K context, else null. Val splits
  hold no FIM examples: the AR loss of arXiv 2207.14255 §2. Per corpus: bits and bytes per context bucket
  (`CONTEXTS = (8192,)`, a target's context being its position in its segment + 1: ≤ 8K, which both stages train, and
  > 8K, which only the repository stage trains; Code Llama's perplexity by context length, arXiv 2308.12950 §3.3,
  Fig. 4a), per byte-offset bucket (`OFFSETS`: doublings from 1 KiB to 128 KiB of the target's segment before it, so
  tokenizers compare at equal bytes of context, arXiv 2503.13423 §3.1, App. B.1; the last bucket holds 0.15% of a 32K
  evaluation's bytes) and per sampled row, with the row's cluster.
- **Evaluation mixture.** Groups (code, text, math) and `all` are pooled with `metrics.mixture_bpb`, `offset_bpb` and
  `mixture_bpb_summary` (§5) by one fixed mixture, `Stage.shares` (`docs/corpus.md` §7) at the stage's total packed
  train rows: a "supply" corpus weighs its packed rows over all packed rows, not its share of a run's draw, so supply
  corpora weigh 0.62× their base-run share at 8K (notebooks 0.0095 against 0.0155) and 0.85× at 32K (traces 0.0465
  against 0.0544) [arithmetic: the packs' train rows; the base run's 66,063 and 3,811 steps]. Every record, the
  peak-lr calibration's included, uses these weights computed from the packs at its evaluation; the records in
  `results/` and `artifacts/runs/` differ from the current packs' weights by at most 3.2e-4 at 8K and 4.9e-4 at 32K.
  Under the base run's mixture the × 2 arm's 8K code bpb (`results/pretrain/peak_lr_d24.jsonl`) is 0.00062 higher
  (0.40607 against 0.40544, 0.29 of its standard error), `all` 0.00054, text and math unchanged, 32K code +0.00161 (0.10
  standard errors) [arithmetic, from the record's per-corpus sums].
- **Sample** (`sample_rows`, `VAL_TOKENS = 80 × 2^19`): per corpus, n = min(rows, ⌈VAL_TOKENS / (micro-batch ×
  seq_len)⌉ × micro-batch) rows, row ⌊j · rows / n⌋ for j < n: systematic, because the packer orders each shard's rows
  longest first, so a prefix would over-represent long documents. Models of one tokenizer are scored on identical rows,
  models of different tokenizers on identical documents when a split lies within the sample, so the subsample's
  sampling error cancels in differences.
- **Clusters** (`row_clusters`): the packer cuts a document longer than a row into pieces on several rows, so sampled
  rows are not independent: rows holding pieces of one document, directly or through other rows, share a cluster (the
  components of the row–document graph), the sample's independent unit. At 8K the 49,765 code rows form 44,829
  clusters (up to 135 rows); at 32K the 2,667 code rows form 2,255, and one `repositories_python` document of 58.8M
  tokens spans 93 of that corpus's 1,281 sampled rows: 7.7% of its bytes at 0.10 bits per byte against the corpus's
  0.39 and 93% of the 32K code variance (the × 2 arm of `results/pretrain/peak_lr_d24.jsonl`) [arithmetic, from the
  packs]. The benchmarks-mode d24 records `eval-3307.json`, `eval-1102.json`, `eval-2205.json` predate `row_clusters`;
  their `"errors"` treat rows as independent (the × 2 checkpoint's 8K code se 0.00080 there against 0.00217 in its
  parse-mode record).
- **Reference solutions** (`solution_bits`, `SOLUTION_BENCHMARKS`): the bits of HumanEval+'s and MBPP+'s canonical
  solutions (EvalPlus) and HumanEval-X C++'s (arXiv 2303.17568; MultiPL-E has none) given their prompts, one example
  per 8K row, BOS first: the task loss of arXiv 2412.04403 §2.2. A continuous correct-answer likelihood makes
  HumanEval and MBPP decisions predictable where accuracy is trivial (arXiv 2504.11393 §3.4, Fig. 6); BPB raises the
  signal-to-noise ratio of most benchmarks (1B models; arXiv 2508.13144 §5.3, Fig. 2); the loss on a benchmark's
  problems and reference solutions correlates with pass@1 more strongly than held-out loss does (Spearman over 20
  checkpoints of one 1B code model; arXiv 2604.07769 §3.3.1, Table 2). A solution's leading blank lines count as
  prompt: EvalPlus's solutions begin with them and a tokenizer joins them to the prompt's last line break, so the
  prompt would otherwise end inside a token and tokenizers would score different bytes. Scoring starts at the token
  that starts at the prompt's end (`decode_with_offsets`).
- **Trajectory** (`eval/trajectory.py`, `python -m eval.trajectory <run_name> <reference step> <first step>
  <out.json>`): `likelihood` of every saved model from a step on. Evaluated checkpoints contribute their evaluations;
  the analysis checkpoints (`docs/training.md` §6; `train/checkpoint.py` `load_model` reads a weights-only file where
  a step has no full state) are scored into `RUNS/<run>/likelihood-<step>.json` (`"8k"`, `"32k"`, `"solution"` and
  `"provenance"`, as in an evaluation). The models of a run score identical rows and solutions, so a change between
  two is a paired difference: `metrics.mixture_bpb_summary` of the per-row bit differences (`metrics.paired_rows`),
  whose standard error drops the covariance of the two models' row scores (arXiv 2411.00640 §4.2 Eq. 8, within each
  stratum). Per step and measure (`8k` and `32k` by the stage's groups, `code` and `all`; `solution` per benchmark
  and pooled over `metrics.solution_strata`): the bits per byte with its standard error and the paired change from
  the reference step (for `32k`, from the first step at 32K where the reference is at 8K); per interval between
  consecutive steps, the paired change and the summed learning rate of the updates between them (checkpoint s holds
  the model after steps 0 … s − 1).

## 3. Parse measure

`code_eval.parse_generation`: one greedy, unconstrained completion of each HumanEval+ (164) and MBPP+ (378) prompt,
BOS first, the prompt healed at its boundary (`decoder.heal_tokens`, §6), as arXiv 2608.18062 generates (§4.3;
App. C.1), with EvalPlus's prompt, 768 + 1 new tokens (`EVALPLUS_MAX_NEW_TOKENS` and the healed token) and stop
strings; the completion is cut at its first stop, special tokens skipped. A completion that used its whole budget with
no stop sequence (`code_eval.truncated`) is "truncated", not compiled: cut off mid-program, it fails by length, not
syntax; in the d24 peak-lr records (`docs/training.md` §5) 96–99% of the non-parsing completions were cut so [measured:
completions re-encoded, ≥ 753 tokens], and records written before this outcome count them as syntax or indentation
failures. Every other `prompt + completion` is classified by `metrics.parse_outcome` (§5) into parses,
IndentationError (TabError included) or another SyntaxError, the classification of TokEval's companion post; the
record holds each outcome's share and the completions' EvalPlus base and plus pass@1. The measure shows the failures a
vocabulary's line-break tokens cause (SyntaxError when punctuation fuses with a line break, IndentationError when a
line break fuses with the indentation after it, in the companion post's models) where pass@1 is near zero. It departs
from TokEval's protocol in its prompts (EvalPlus's 0-shot base prompts; TokEval: lm-evaluation-harness, MBPP 3-shot,
App. C.1), its budget and stops (the post: 256 tokens) and in compiling the unsanitized completion. It is a diagnostic
of tokenization and training, outside the metric set of §4.

## 4. Code benchmarks

The model is pretrained on Python and C++ code (`docs/corpus.md` §1), so every benchmark is in Python or C++. Each
runs in its official tool, unmodified, at a pinned commit (`docker/Dockerfile.eval`, the `nanochat-eval` image on
`vllm/vllm-openai:v0.32.0`), on the checkpoint exported by `inference/export_hf.py` and generated by vLLM 0.32.0's
Transformers backend with the head in fp32 (§8):

```bash
VLLM_GPU_MEMORY_UTILIZATION=0.75 scripts/official_eval.sh <run name> <step> [<eval record to compare against>]
```

It writes `RUNS/<run>/official-<step>/`: the export (`model/`), each tool's samples and verdicts (`evalplus/`,
`multipl-e/`, `lm-eval/`, `infilling/`), `tools.json` (the pins and the inference stack), `timings.tsv`, and
`eval/official.py`'s `summary.json` in `evaluate`'s layout (`codegen`, `fim`, `evalplus_average`; each cell a
`metrics.pass_cell`, infilling `metrics.summary` clustered by HumanEval problem, §5) and, given a record,
`compare-<record>.json` (`metrics.compare` and the per-problem agreement). A rerun keeps the export and skips the
samples and verdicts already there. CPU work is bound to NUMA node `CPU_NODE` (default 0; on GH200 node 1 is the GPU's
memory). `VLLM_GPU_MEMORY_UTILIZATION` sets every `vllm.LLM`'s `gpu_memory_utilization`
(`eval/official_site/sitecustomize.py`; MultiPL-E's `automodel_vllm.py` fixes 0.95, EvalPlus's provider leaves vLLM's
0.92), so the tools fit beside another process on the GPU; it sizes the KV cache only.

| Benchmark | Tool, pin | Protocol | Departure from the tool's default, and why |
|---|---|---|---|
| HumanEval+ (164, v0.1.10; arXiv 2305.01210), MBPP+ (378, v0.2.0, not comparable with the 399-task v0.1.0) | EvalPlus `26d6d00` | `evalplus.codegen --backend vllm --greedy`: the base-model prompt, 768 new tokens (`provider/base.py:13`), its EOS list and `sanitize`; `evalplus.evaluate`: base and plus statuses, columns `humaneval`/`mbpp` (base tests) and `humaneval_plus`/`mbpp_plus` (both); the EvalPlus average of the four (Qwen3, arXiv 2505.09388 §3.3) | `evaluate` runs with `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` = 1: its `reliability_guard` caps a test process's address space (RLIMIT_AS), which counts the mappings the evaluator's process holds, inflated by per-core thread pools on 64 cores, so HumanEval/15's plus input `[1000000]` failed with MemoryError (69,874: HumanEval+ 35 → 36 problems) [measured] |
| HumanEval, MBPP in C++ (161, 397; arXiv 2208.08227) | MultiPL-E `3025a53` | `automodel_vllm.py` (`max_model_len` 2,048, 1,024 new tokens, the dataset's stop tokens): greedy (T 0, one completion) and sampled (T 0.2, top-p 0.95, 20); `evaluation/src/main.py`; `pass_k.py` | greedy added to MultiPL-E's sampled protocol, as DeepSeek-Coder reports C++ (arXiv 2401.14196 §4.1, Table 3); one `main.py --file` process a problem file (Evaluator processes, below) |
| MBPP 3-shot (500, task ids 11–510; arXiv 2108.07732) | lm-evaluation-harness `d6de816` | `lm_eval --model vllm`, task `mbpp` (v1.0): tasks 2–4 as shots, greedy, until `"[DONE]"` and the EOS string lm-eval appends (`models/utils.py` `handle_stop_sequences`), 256 new tokens (`max_gen_toks`); HF `evaluate`'s `code_eval` on completion + asserts | `add_bos_token=True`: the tokenizer's post-processor puts BOS first, as on every training row; the task `mbpp_lf` (`eval/official_lm_eval/`): `mbpp` with the prompt's line endings normalized (below) |
| HumanEval infilling single-line, multi-line, random-span (1,033, 5,815, 1,640; arXiv 2207.14255) | human-eval-infilling `88062ff` | `evaluate_functional_correctness` (3 s a sample) on the samples of `eval/official_infilling.py` (the package has no generator): `corpus.tokenizer.fim_prompt`'s PSM ids (BOS, the sentinels, prefix and suffix encoded apart: the packer's layout, `docs/corpus.md` §8), greedy, the middle until BOS (the paper's EOT) or another special token | the paper samples 100 (single- and multi-line) or 200 (random-span) middles a task at T 0.2 in SPM (Fig. 6, Tables 1, 4) and names no budget: here one greedy middle in PSM, as every benchmark here is greedy [inference: one order measures the trained format; the packer trains both], at most the benchmark's longest reference middle (44, 263, 169 tokens, `official_infilling.budget`) |

The post-trained model's protocol (EvalPlus's chat prompt and LiveCodeBench, `scripts/official_reasoning.sh`):
`docs/post-training.md` §5. Benchmarks and modes no longer run: `docs/experiments.md` §6.

Results of the base run's final checkpoint (step 69,874), their comparison with published models and their failure
analysis: `docs/results.md` §1–§3.

- **Line endings.** The corpus normalizes `"\r\n"`, then `"\r"`, to `"\n"` (`corpus/pretrain/prepare.py`), so the model
  never trained on `"\r"`. lm-evaluation-harness's three MBPP shots carry MBPP's CRLF code (`tasks/mbpp/utils.py`
  `list_fewshot_samples`): every one of the 500 prompts holds 11 carriage returns (their effect on the score:
  `docs/results.md` §1). The task `mbpp_lf` is `mbpp.yaml` with the prompt's fields (statement, code, asserts)
  normalized (`utils.py` `process_docs`, `list_fewshot_samples`); scoring is unchanged, since MBPP is judged by its
  asserts, not by the reference code. The
  EvalPlus, MultiPL-E and infilling prompts hold no `"\r"` [measured on 69,874's prompts], nor do EvalPlus's chat
  messages; 177 of LiveCodeBench v6's 454 window problems do, and `eval/official_lcb.py` normalizes lcb_runner's
  messages alike (`messages`) [measured: `eval/official_tokenize.py`]. `sitecustomize.py` makes every
  `vllm.LLM.generate` refuse a prompt holding one (token-id prompts decoded), and `scripts/official_reasoning.sh`
  checks every message it will serve before generation (`eval/official_tokenize.py`).
- **Comparability.** MultiPL-E's sampled mode draws 20 samples a problem where Codex (arXiv 2107.03374 §2.1) and
  MultiPL-E drew 200 and the BigCode leaderboard 50 (arXiv 2402.19173 §7.1.2): the same estimate with more variance
  per problem [inference] (MultiPL-E notes its pass@1 rates stabilize around 20 samples, arXiv 2208.08227 §IV-B,
  footnote 6). evalplus is pinned by commit (`26d6d00`): PyPI 0.3.1 has a 1 s minimum time limit, 4 s from `af49138`,
  and fails every HumanEval/32 solution, fixed in `6eb1e19` [unverified: read from the repository's history, not
  run]. `char` is unsigned on ARM64 and signed on x86, so a MultiPL-E C++ program that depends on it can score
  differently [inference].
- **The 1@10 mode** (`evaluate` with `"benchmarks"`, `filtered_generation`; `"codegen"` `<benchmark>."1@10"`, EvalPlus
  and MultiPL-E): the greedy candidate, then samples at T = 0.8 (`verify.TEMPERATURE`, Codex's pass@100 optimum for a
  679M model, §3.3) from the nucleus `verify.TOP_P` 0.95, attempt a of task i seeded `ATTEMPTS` · i + a, until one has
  no post-generation diagnostic and passes those of its prompt's examples that the reference solution passes
  (`vetted`, §7), at most `verify.ATTEMPTS` = 10; prompts healed, Python under the syntax constraint (§7). AlphaCode's
  n@k (arXiv 2203.07814 §2.2) at n = 1, k ≤ 10 with example-test filtering (Codex §3.5's filtered pass@k): the
  evaluator's pass rate of the accepted candidate, else the greedy candidate. A system measurement, kept out of
  comparison tables; the prompts' examples overlap the scored base tests (MBPP+: 373 of the 378 prompt asserts call
  the entry function on a base input; HumanEval+: 121 of the 176 doctest examples of its 75 tasks with examples do
  [arithmetic: EvalPlus data, `verify.examples`]), so its primary EvalPlus score is the plus column.
- **Contamination.** The train splits are decontaminated against the EvalPlus, MultiPL-E, HumanEval-X and infilling
  texts (`corpus/decontaminate.py`; `docs/corpus.md` §3): a row sharing a 10-gram of words with a task's prompt or
  reference program is dropped (Qwen2.5-Coder's rule, arXiv 2409.12186 §5), 491,203 rows, among them text of every
  HumanEval+ task and 372 of 378 MBPP+ tasks; before it, `ultradata_stack_v3_l2_python` alone held 6,447 train
  documents with both a HumanEval+ task's signature and its docstring's first line, 161 of the 161 tasks checkable
  that way [unverified: a probe without a record]. Paraphrases that share no 10-gram remain, as under the cited rule.
  The original MBPP texts are outside the decontamination (224 of MBPP 3-shot's 500 test problems are MBPP+ tasks,
  matched through EvalPlus's revised prompt and solution); measured with the same rule over every decontaminated train
  shard (1,794 shards of 18 sources; `results/pretrain/mbpp_3shot_contamination.json`, per task, with the rule, the
  dataset revision and a key of the shards' paths and sizes), the statement (`text`) of 298 of the 500 shares a 10-gram
  with 4,315 train rows, and statement, code, asserts and setup code together match 429 tasks in 275,479 rows
  [measured]. LiveCodeBench v6's 454 problems are not in the SFT prompts: no prompt holds ≥ 30% of a problem's
  distinctive 8-word sequences, and the 10-word rule's hits are boilerplate [measured]; the overlap with the
  pre-training data is not measured.
- **Evaluator processes.** EvalPlus's `sanitize` parses every line span of a program, hence a sanitize worker per
  CPU in `evaluate` (`code_eval.sanitizers`). MultiPL-E's `main.py --dir` judges a directory's problems serially and
  its default worker count oversubscribes g++ (a compile past its limit scores "SyntaxError", §9), hence one
  `main.py --file` process a problem file (`--max-workers` at most 4) in `official_eval.sh` and `--max-workers 1` per
  problem in `evaluate`.

## 5. Metrics and statistics

`metrics.py` is the repository's one source of metrics, torch-free so training, evaluation and the experiment records
import it alike.

| Function | Computes | Source definition; library |
|---|---|---|
| `append`, `records`, `train_steps`, `diverged`, `curves` | the run log `RUNS/<run>/metrics.jsonl` (record kinds and fields: `docs/training.md` §6): UTF-8 JSON lines, each with `run_name`, `attempt` (0-based process start), `time`, `kind` (NaN and infinities raise); a resumed run's latest attempt's record of a step supersedes earlier ones; the last failure's step; train records as columns in step order | — |
| `bpb(*pooled)` | Σ bits / Σ bytes over context buckets and arguments | The Pile, arXiv 2101.00027 §3.1 |
| `mixture_bpb(corpora, weights, buckets=)` | Σ_c w_c · bits_c / rows_c over Σ_c w_c · bytes_c / rows_c (rows_c the corpus's scored rows): the bits per byte of val rows drawn with row weights w | The Pile's whole-set figure aggregates its components' sums, weighted by size (arXiv 2101.00027 App. E.2); here by w |
| `offset_bpb` | `mixture_bpb` per byte-offset bucket (None where no target is); a bucket pools the documents long enough to reach it, so the curve mixes context use with document composition | §2 |
| `mixture_bpb_summary(corpora, weights)` | `mixture_bpb` of the rows, R = Σ_c a_c · bits_c / Σ_c a_c · bytes_c (a_c = w_c / rows_c), with its standard error as a stratified cluster sample: the corpora are strata at fixed weights, each corpus's clusters of rows (§2) its independent units; by the delta method se² = Σ_c a_c² · m_c · s_c² / (Σ_c a_c · bytes_c)², s_c² the sample variance (ddof 1) of the m_c cluster sums of bits − R · bytes; 95% normal interval. For one corpus of one-row clusters it is the HC1 error of the WLS fit of row bpb on a constant weighted by bytes (statsmodels; equal to 1e-15 on real records) | Miller, arXiv 2411.00640 Eq. 1, 4 within each stratum, Eq. 3 for the interval (the strata and the ratio's delta method are not Miller's); numpy |
| `paired_rows(a, b)` | b's rows with each row's bits minus a's, for two scorings of identical rows (equal `row_bytes` and `row_clusters`, else ValueError); `mixture_bpb_summary` of them is the paired change b − a with its clustered standard error (§2, trajectory) | Miller §4.2 Eq. 8 within each stratum |
| `solution_strata(solution)` | an evaluation's reference-solution rows as two strata: HumanEval (HumanEval+ and HumanEval-X C++ solutions of problem i one cluster) and MBPP+ (a cluster per problem); `mixture_bpb_summary` at weights {stratum: rows} is their pooled bpb | Miller Eq. 4 |
| `parse_outcome(source)` | one `compile()` → "parses", "indentation" (IndentationError, TabError; checked first) or "syntax" (other SyntaxError; also the parser's MemoryError and RecursionError, and the ValueError CPython 3.12.3 raises for an f-string debug expression in a format spec, `f"{x:{y=}}"`, python/cpython#121130) | TokEval's companion post, section "How the failing generations fail" (linked from arXiv 2608.18062 App. F.3, fn. 15) |
| `parse_shares` | each outcome's share of a benchmark's completions (`PARSE_OUTCOMES`: parses, indentation, syntax, truncated) | — |
| `measures(evaluation, steps, repo_level_start=)` | a run's compared measures (table below) | — |
| `measure_errors(evaluation)` | their error bars: `code`, `text`, `math`, `repo_32k` from the evaluation's `errors` (`mixture_bpb_summary`); `solutions` by `mixture_bpb_summary` over `solution_strata` at their solution counts; `unparsed`, `indentation`, `syntax`, `truncated` and `pass@1` by `summary` over problems; none for `repo_beyond_8k`, `tokens_per_second`, `mfu` | Miller Eq. 1, 4 |
| `summary(scores, clusters=)` | mean of question scores (a problem's pass@k estimate is one question score, as Miller §3.1 averages a question's K answers [inference]); OLS on a constant: unclustered, se = s/√n (Miller Eq. 1); clustered, Liang–Zeger CR0 without small-sample correction (Miller Eq. 4 except for its (n − 1) in the diagonal term); 95% normal interval (Eq. 3), not clipped to [0, 1]; a constant score has se 0 (`pass_cell` adds the Wilson interval for single-completion cells) | statsmodels `OLS` |
| `pass_cell(passed)` | a pass@1 table cell: `summary` of each problem's mean pass over its samples; when every problem has one sample, also `wilson`, the Wilson 95% score interval (inside [0, 1], not degenerate at 0 or at all passes) | Miller, arXiv 2411.00640 §3.1 (average a question's samples first), Eq. 1, 3; statsmodels `proportion_confint(method="wilson")` |
| `evalplus_average(humaneval, mbpp)` | the EvalPlus score of one mode's `humaneval_plus` and `mbpp_plus` records: the mean of the HumanEval, HumanEval+, MBPP and MBPP+ pass@1, computed as ½(m_HE + m_MBPP) of per-task (base + plus)/2 scores; a task's two columns form one cluster and the benchmarks are independent, so se² = ¼(se²_HE + se²_MBPP), each from `summary`; 95% normal interval; n counts tasks | Qwen3, arXiv 2505.09388 §3.3 (the four-way average); Miller Eq. 1, 3 |
| `compare(a, b)` | paired comparison of two evaluation records (an `evaluate` record or `eval/official.py`'s summary) over the `codegen` (every mode and column) and `fim` sections both hold, on identical problems (else ValueError); one row per `<section>/<benchmark>/<mode>/<column>`: n; means a and b of the per-problem pass rates; the difference b − a with its paired se and 95% interval (clustered by HumanEval problem for `fim` rows); `corr`, the correlation of the per-problem scores (None when either is constant); p from exact McNemar for single-completion modes (greedy, 1@10, MBPP 3-shot, infilling; it treats an infilling problem's lines as independent), else a two-sided normal test of the difference over its se; `p_holm`, Holm-adjusted over all rows; `mde`, the difference detectable at α 0.05 with 80% power, (z₀.₀₂₅ + z₀.₂) · se. n · se², the variance of the per-problem difference, is Eq. 10's ω² + σ²_A/K_A + σ²_B/K_B less twice the covariance of A's and B's sample means: sample k of problem i has the same seed in every record, so the two records' samples are positively correlated [inference] and this paired se and mde are those of that design (common random numbers) | Miller Eq. 7, 8 (paired se), Eq. 3, Table 5 (correlation), Eq. 10 and App. C (detectable difference); statsmodels `mcnemar(exact=True)`, `multipletests(method="holm")`; scipy `norm` |
| `model_flops_per_token(matmul_params, hidden_size, windows, context)` | nominal training FLOPs per token: 6N + 12 · C · Σ_layers min(window_l + 1, context), each layer's most keys a query attends, itself included (`layer_windows`: 512, or the context for full layers); records written before the query's own key was counted (`results/pretrain/muon_wd_timescale_20tpp.jsonl`, `results/pretrain/scalar_learning_rates.jsonl`, d16) hold 6N + 12 · C · Σ min(window_l, context), lower by 12 · C per windowed layer (d16: 73,728; 641,826,816 at 8K, 1,245,806,592 at 32K; d24: 165,888 lower) | PaLM, arXiv 2204.02311 App. B's 6N + 12LHQT; torchtitan's count without the causal halving |
| `attended_keys(lengths, window)` | Σ over segments of Σ_p min(p, window + 1): the keys each query attends causally within its segment, itself included | — |
| `executed_model_flops(lengths, matmul_params, hidden_size, windows)` | 6N per position + 12 · C per attended key per layer over a micro-batch's segments (QKᵀ and PV, 4C per key forward, twice that backward; FA4's score recomputation not counted) | PaLM App. B; causal, per-document counting as Megatron-LM `num_floating_point_operations` (which excludes padding) |
| `mfu(flops, seconds)` | flops / seconds / `PEAK_FLOPS` (989.4e12, the H100 SXM5 / GH200 dense BF16 peak; half NVIDIA's 1,979 TFLOP/s "with sparsity" is 989.5 [arithmetic]) | PaLM §4.1 |
| `parabola_vertex(x, y)` | (a, −b/2a) of the least-squares parabola (numpy `polyfit`, degree 2); used by `docs/training.md` §5 | — |
| `latency(starts, firsts, ends, new_tokens, duration=)` | `ttft` (first − start, start when the request takes a decoder slot), `tpot` ((end − first)/(n − 1), requests of > 1 token), `e2el`, each mean, median, p99; `output_throughput` | vLLM v0.32.0 `vllm/benchmarks/serve.py` `calculate_metrics` |

**Measures** (`measures`, the quantities experiments compare; bits per byte unless stated):

| Measure | Definition |
|---|---|
| `code` | 8K validation of the code group at its corpus weights |
| `solutions` | HumanEval+, MBPP+ and HumanEval-X C++ reference solutions given their prompts, pooled (`bpb`) |
| `repo_32k`, `repo_beyond_8k` | 32K validation of the code group; the same over targets beyond 8,192 positions (the last context bucket), at the 32K evaluation's code-corpus weights restricted to `REPO_LEVEL_STAGE.code` |
| `text`, `math` | 8K validation of each group |
| `unparsed`, `indentation`, `syntax` | shares of the 542 HumanEval+ and MBPP+ problems [arithmetic: 164 + 378], pooled, whose healed greedy completion stopped within its budget and does not compile; by IndentationError; by another SyntaxError (§3). Records written before `truncated` have `unparsed` = 1 − parses, truncated completions included |
| `truncated` | share of the 542 whose completion used its whole budget without a stop sequence (§3) |
| `pass@1` | greedy base pass@1 of the parse measure's completions, pooled |
| `tokens_per_second`, `mfu` | medians over the file-level stage's steps (before `repo_level_start`) of the train records' tokens per second and executed MFU |

**Model FLOPs and MFU.** The train records' `mfu` is executed model FLOPs per step (`executed_model_flops` of each
micro-batch's segment lengths, read from the loader without a device synchronization, which
`torch.utils.flop_counter` would need) over the step's device time and `PEAK_FLOPS`; the run record's
`model_flops_per_token` is the nominal count. DeepSeek LLM's M (arXiv 2401.02954 §3.2 Eq. 2) counts full attention in
every layer and stays the scaling plan's compute, not an MFU numerator; the d24 values of M and the nominal count:
`docs/architecture.md` §1. Executed FLOPs per token of a row holding one document are 1,479,503,808 at 8K and
2,160,975,600 at 32K [arithmetic, recomputed on CPU], below nominal (a query attends only its own segment's earlier
keys) and lower still with packed documents; M is 3.8× the executed count at 32K [arithmetic], so an M-based
utilization would exceed 100%. On the d24 rehearsal's train records (`docs/training.md` §7) a step's executed FLOPs
are 0.744–0.799 (median 0.771) of nominal × its tokens at 8K and 0.442–0.652 (median 0.548) at 32K [recomputed from
`model_flops`, itself recomputed exactly from the loader's segment lengths on sampled steps]: a nominal-count MFU
would read 0.644 at 8K and 0.926 at 32K (medians), the latter above the 0.71 the step's GEMMs reach at the 700 W cap
(`docs/training.md` §7). FLOPs count 2 per multiply-add and exclude rematerialization (PaLM §4.1, App. B).
`tokens_per_second` is the observed throughput of PaLM's MFU (§4.1); `peak_memory_bytes`
(`torch.cuda.max_memory_allocated`) is the most memory the run's tensors held, `peak_reserved_bytes` the allocator's
(`max_memory_reserved`). Measured MFU: `docs/training.md` §7.

## 6. Decoder

`inference/decoder.py` `NativeDecoder` with `model/kernels/decode.cu` generates every completion of `evaluate` (the
parse measure, §3, and the 1@10 mode, §4); the official tools generate with vLLM (§8).

- **Batching.** Up to `MAX_BATCH` = 1,024 slots (fewer if device memory is short), continuous: a finished slot takes
  the next request. A greedy step is one captured CUDA graph and a sampled or constrained step two (body, then head),
  each graph's kernels programmatic dependents of each other (PDL); a second set of graphs over the first
  `SMALL_BATCH` = 256 slots runs while no slot at index ≥ `SMALL_BATCH` is active. A slot's KV cache holds
  2 · L · max_length · C bf16 values, 73,728 bytes per position at d24 [arithmetic]. A many-slot step is bound by its
  KV reads and its GEMMs; a few-slot step by weight reads and kernel launches [inference]. Each pass sizes its own
  decoder (d24, the corpus tokenizer [arithmetic]): parse measure `max_length` 1,172 with no store (each healed prompt
  is requested once); the 1@10 search 1,172 with a prompt store of 96,770 positions (the 1,100 EvalPlus and
  MultiPL-E prompts).
- **Step.** The model's forward at one position (`docs/architecture.md` §1), GEMMs on wgmma tensor cores fed by TMA,
  attention streaming each slot's window through shared memory with the scores on tensor cores (bf16 q and k, fp32
  accumulators), softmax and the value sum in fp32. Numerics follow the compiled forward without autograd (the
  evaluation's graph) op by op: bf16 copies of the matrices; K and V cached in bf16, the operands the forward's
  attention reads; fp32 rotary table, q/k norm gains and value-embedding gate (its dot product in `ve_gate.cu`'s lane
  order); the residual mix as Inductor fuses it (one fma, one bf16 rounding); x0 the norm of the fp32 embedding row,
  since that graph drops `CausalLM.hidden`'s bf16 cast (`decode.cu` `k_norm`); the GEMM epilogues as QuACK's: o_proj
  and down_proj add the residual to the fp32 accumulator and round once, the MLP's rstd comes from o_proj's unrounded
  sum (`k_rstd`), and the up projection reads the un-normed stream, ReLU² of accumulator × rstd (`gemm_norm_act`).
  Attention differs from FA4's in summation order and in the probabilities' precision: FA4's P·V GEMM reads them as
  bf16 (`flash_fwd_sm90.py`), this sum as fp32. `k_attn` stages two runs of 64 keys' K and V (64 KB of shared memory,
  the carveout at its maximum): 3 blocks and 96 KB of K and V in flight per SM [arithmetic].
- **Prefill and prompt store.** A new request's prompt but its last token runs as rows of one batched pass through the
  step's body (up to 1,024 rows of any prompts per launch, each row computed as a step would), so its slot starts at
  the prompt's last token. Its K and V are copied to the prompt store, and a later request with the same prompt copies
  them back (K and V depend on the prompt alone); a prompt is stored once per decoder and never evicted. The store is
  host memory bound to the calling CPU's NUMA node (the GH200's LPDDR5X; `mbind`, which a container needs
  `CAP_SYS_NICE` for, `scripts/container.sh`) and registered for the copy engines over NVLink-C2C, so the slots keep
  the device memory: the positions requested, at most half of what that node can give.
- **Request** (`Request`): `input_ids`, `max_new_tokens`, `stopping_criteria`, `temperature` (0: argmax, ties to the
  lowest id; > 0: Gumbel-max sampling of softmax(z / T), the noise a Philox4x32-10 function of (seed, generated
  position, token) alone, arXiv 1411.0030 §2), `top_p` (< 1: the fewest most likely allowed tokens holding top_p of
  the probability, a tie at the cut kept whole, arXiv 1904.09751 §3.1, `k_nucleus`; transformers' `TopPLogitsWarper`,
  the sampler of MultiPL-E's `automodel.py` and bigcode-evaluation-harness), `seed`, `constraint`. Sampled logits are
  capped, z = c·tanh(s/c) with the model's `final_logit_softcapping`; a greedy choice needs no cap (tanh is
  increasing). A request needs `max_new_tokens` ≥ 1 and len(input_ids) + max_new_tokens − 1 ≤ `max_length`, and
  `max_length` at most the rotary table's; each violation raises ValueError. Generation ends at BOS (excluded), the
  budget, or the stopping criteria (evaluated while the next step runs).
- **Constraints.** A step with a sampled or constrained slot launches the body, writes the constraints' masks while it
  runs (the syntax constraints' in one parallel call straight into pinned mask rows, §7), then the head, which chooses
  within each slot's mask and also computes its best allowed token but BOS; a chosen BOS that the syntax constraint
  rejects gives way to that alternative (the same choice as masking BOS beforehand). `heal_tokens` drops the prompt's
  last token and returns its `Reemit` constraint: each token must be a proper prefix of the owed bytes or start with
  them, until they are re-emitted, possibly over several tokens (token healing, Guidance 2023; generation constrained
  until the dropped content is re-emitted, as arXiv 2608.18062 App. C.1).
- **Determinism.** Every sum's order depends on the shapes and the slot's position alone, so a request's tokens do not
  depend on the batch, the store, `max_length` or timing.

## 7. Syntax-constrained decoding and post-generation checks

**Syntax, not types.** Type-constrained decoding (arXiv 2504.09246) covers TypeScript only. An incomplete constrainer
(one rejecting valid programs) lowers functional correctness below unconstrained decoding (arXiv 2606.21619,
abstract), while a complete one helps, smaller models the most (§5.3, a TOML syntax constrainer; its rejection as a
design: `docs/experiments.md` §6). Python 3 syntax can be constrained completely, and every Python file of the code
categories parses as Python 3 (`curate.repaired`, `docs/corpus.md` §3; UltraData-Code L3's exercises and the Kaggle
notebooks are not checked), so the model's distribution of program files lies inside the constrainer's language
[inference]. Types are checked after generation instead.

**`SyntaxConstraint`** (`inference/syntax_constraint.py`, Rust crate `inference/syntax/`, built from the caller's
tokenizer's vocabulary by `python_syntax`). Token t is allowed iff `prefix + generated + t` can still become a valid
Python 3.12 module, the text ending anywhere; FIM sentinels never; BOS iff the grammar ends a module after the text and
the FIM suffix and `ast.parse` accepts both (`ends`). While a healed prompt owes bytes, each mask is intersected with
the `Reemit`'s. `masks` writes a batch of masks in parallel (rayon) without the GIL. Layers:

- lexer (`lexer.rs`): CPython 3.12.3's tokenizer (`Parser/tokenizer.c`) as a character-level state machine, so a
  prefix may end mid-token, mid-indent or mid-character; `\r\n` and `\r` read as `\n`;
- indentation (`parser.rs`): INDENT/DEDENT (tab size 8), the tab-consistency check, the bracket level;
- parser (`parser.rs`): an LALR(1) grammar (`python.lark`) accepting every token sequence of CPython 3.12's PEG
  grammar, tables built by Lark at load (strict mode proves them conflict-free), read as a viable-prefix recognizer;
  the soft keywords `match`, `case`, `type` read both ways until one parse dies;
- masks (`vocab.rs`): per lexer state, every token's events and end state computed once over the vocabulary's byte
  trie; a step runs the parser once per distinct event sequence.

It never rejects what CPython accepts; it over-approximates (accepts some prefixes CPython rejects): assignment,
deletion and `with` targets are expressions, parameter and argument orders are free, escapes and nesting limits are
not checked, among others (`inference/syntax_constraint.py`). BOS is exact.
Libraries not used: SynCode (arXiv 2403.01632; its Python grammar, of Lark's `python.lark` lineage, lexes an f-string
as one regex token, so PEP 701 f-strings such as `f"{x["k"]}"` and PEP 695 type parameters are rejected, and it pins
transformers and a Hugging Face tokenizer), llguidance (no INDENT/DEDENT, open guidance-ai/llguidance issue #107; one
byte may emit several DEDENTs), arXiv 2402.17988 (no released library). C++ is not constrained.

Checks (Python files replayed token by token through the constraint, CPU): 0 false rejections over 11.4M token steps
of 3,000 Python files (the standard library, site-packages, this repository; BPE) and 9.2M randomly segmented; no
prefix that `ast.parse` accepts without the grammar's end; a d16 checkpoint's unconstrained completions (EvalPlus and
SPM infilling, greedy and sampled) replayed through their constraints: every masked token invalid, outputs
token-identical wherever nothing was masked; all 8,488 infilling reference middles (BPE and two random segmentations)
accepted; `viable_end` agrees with `allowed` on 8,002,126 prefixes. Tables built from another vocabulary masked the
true token in 1,475 of 3,199 steps, hence the caller's tokenizer. Mask time per step 0.7–2.6 ms.

**Post-generation checks** (`eval/verify.py`, the 1@10 mode's acceptance test):

- Python (`python_diagnostics`, on the program the scorer runs, EvalPlus's sanitized solution): the program compiles
  (`compile_error`: one `compile()`, which also rejects SyntaxErrors that `ast.parse` and the syntax constraint
  accept, such as `'return' outside function` or a duplicate argument: on a d16 checkpoint 5, 60 and 6 accepted
  single-line, multi-line and random-span middles of the removed infilling 1@10 mode did not compile), then
  basedpyright 1.40.2 with every rule off but `reportUndefinedVariable`, `reportAttributeAccessIssue` and
  `reportCallIssue`, in concurrent processes over groups of at least `PYRIGHT_GROUP` = 100 programs, a group halved if
  it exceeds
  `PYRIGHT_TIMEOUT` = 60 s; a program basedpyright cannot check (an internal error, a timeout alone, an exit without a
  report) fails. EvalPlus solutions (`solution_diagnostics`): the solution's syntax error, an entry function missing
  or without a `return <value>` in its own body.
- Examples (`examples_pass`): the prompt's doctest `>>>` examples and Python `assert` lines that the reference program
  passes (`vetted_examples` via `code_eval.vetted`: EvalPlus prompt + canonical solution), each run by value (equal to
  the want, or None/True for a bare expression) in a fresh interpreter under EvalPlus's `reliability_guard`, 3 s
  (HumanEval's timeout); true without examples. Of the 554 EvalPlus examples,
  the references fail 4 (HumanEval/47's median 15.0, all three of /116's), which every correct program fails. By value
  rather than doctest's output text because HumanEval writes some wants in an equal but differently printed form
  (HumanEval/65, /113) and some examples as bare comparisons without output (/108, /145); MBPP+ asserts compare tuples
  as lists.
- C++ (`cpp_diagnostics`): `g++ -std=c++17 -fsyntax-only`, the MultiPL-E judge's compiler and standard, 15 s; C++
  prompts carry no examples here. This check and MultiPL-E's judge (§4) force `<bits/stdc++.h>` first as a
  precompiled header (`verify.compiler`; GCC manual, "Precompiled Headers"). Every MultiPL-E C++ prompt (558 of 558)
  begins `#include<assert.h>` then `#include<bits/stdc++.h>`, so a judged program sees no declaration it would not see
  anyway; a compile is 4.0× and `-fsyntax-only` 5.3× faster [unverified: 30 programs on a 20-core aarch64 host, no
  record].
- Verdict cache (`eval/cache.py` `VerdictCache`, SQLite at `paths.VERDICTS`): a check's result keyed by its kind, its
  identity (tool configuration and sources, the interpreter and every installed package) and the exact program text,
  so an evaluation skips programs an earlier one checked. Results ended by a timeout or a signal are not kept.
  EvalPlus's verdicts are not kept: a test past its time limit fails like a wrong answer, so a timing failure cannot be
  told from the program's own.

`search` runs all four filtered searches from one `generate` call in attempt order (every task's attempt 0, then
attempt 1, …), skipping tasks already accepted, while each search's verdicts run round by round on a thread of their
own; every candidate is a serial search's (a candidate's tokens depend on its request and attempt alone, §6) and
acceptance is the first accepted attempt. Checks stop at the first failure (compile, basedpyright, then examples).

Basis (both d16 checkpoints' full 10-candidate HumanEval+ and MBPP+ pools): the three rules and the vetted examples
rejected none of the 100 candidates that pass EvalPlus plus (Clopper–Pearson 95% interval's upper bound 3.6%
[arithmetic]); without the examples check filtered plus pass@1 fell 1.1–1.8 points. `reportCallIssue` rejects one
reference, Mbpp/130 (`max(d, key=d.get)` on a `defaultdict`, an overload miss); Mbpp/630's reference fails as EvalPlus
scores it (sanitize drops its helper). The
checks not kept: `docs/experiments.md` §6.

## 8. Export, vLLM and the standard-protocol cross-check

`inference/export_hf.py` `export` writes a checkpoint (`checkpoint-<step>.pt` or `weights-<step>.pt`) as a Hugging
Face model directory that `AutoModelForCausalLM.from_pretrained(<dir>, trust_remote_code=True)` and
`AutoTokenizer.from_pretrained(<dir>)` load (transformers 5.21.0, tokenizers 0.23.3, safetensors 0.8.0):

- `model.safetensors`: bf16 the token and value embeddings and every matrix (the precision the training forward reads
  them in); fp32 the q/k gains, `resid_lambdas`, `x0_lambdas` and the value-embedding gates, which a bf16 load keeps
  fp32 (`_keep_in_fp32_modules_strict`). The compiled forward-only graph's x0 is the norm of the fp32 embedding row
  (§6; the training graph rounds the row to bf16 first), the export's the bf16 row's. d24: 522,190,848 bf16 and 61,488
  fp32 values, a 1,044,651,640-byte file.
- `config.json`: `model_type` "nanochat" with `auto_map` to `modeling_nanochat.py`; MHA with `head_dim` 128; the
  per-layer windows as Gemma 3's `layer_types` + `sliding_window` = 513 (transformers' window counts the query);
  `value_embedding_layers`, `final_logit_softcapping`, `rms_norm_eps` = FLT_EPSILON; BOS (32,764) as BOS and EOS.
- `tokenizer.json`: transformers' `TikTokenConverter` of the corpus tokenizer with `SPECIAL_TOKENS` after the ranks
  (`docs/corpus.md` §5), BOS prepended to every encoding.

**Model file** (`inference/modeling_nanochat.py`, torch and transformers only): `CausalLM` (`docs/architecture.md`
§1) in Qwen3's layout (`NanochatForCausalLM`, `NanochatModel`, `NanochatDecoderLayer`), with the training kernels'
precision as PyTorch expressions: q/k RMS norm, fp32 gain and rotation by −θ with fp64-computed fp32 tables
(`qk_row.cuh` `qk_rotate`; transformers' rotary embeddings rotate by +θ), the value-embedding mix v + 2σ(W·n)·ve in
fp32, the residual mix λr·x + λ0·x0 in fp32 rounded once (`docs/architecture.md` §1, §4), and the head's fp32 logits
(`F.linear` of the operands cast to fp32, equal to the bf16 GEMM's fp32 output up to summation order: a product of
two bf16 values is exact in fp32) capped in fp32; the MLP and the residual additions run in the load's dtype.
Attention is transformers' attention interface (`sdpa` by default) under transformers' causal and sliding-window
masks, positions from 0 per sequence; `generate` keeps a `DynamicCache` built from `layer_types`. The training forward
attends within each packed piece (`cu_seqlens`); the model file attends over its whole input, so a packed row is
scored segment by segment.

**Agreement with the training forward** (CPU, fp32 load, `sdpa`): the d24 rehearsal's `checkpoint-1102.pt` (8K;
`docs/training.md` §7) against a re-evaluation of that checkpoint by `eval/suite.py`, whose per-row bits come from the
compiled bf16 `CausalLM.losses` on the GH200. (The run directory's `eval-1102.json`, written by earlier code, has the
same rows and bytes and per-corpus bpb 0.07% lower to 0.34% higher than the re-evaluation's [unverified: cause not
established].) Val rows j = 0 and ⌊n/2⌋ of `"8k"` `"corpora"` (n the record's `"rows"`), each piece its own sequence
from position 0:

| Corpus | j | Pieces | Bytes (record = export) | Record bits | Export bits | Δbpb |
|---|---|---|---|---|---|---|
| `stack_edu_python` | 0 | 1 | 30,002 | 17,775.800 | 17,776.240 | +1.47e-5 |
| `stack_edu_python` | 2,562 | 3 | 28,866 | 15,539.657 | 15,541.906 | +7.79e-5 |
| `stack_edu_cpp` | 0 | 1 | 31,781 | 10,972.315 | 10,968.337 | −1.25e-4 |
| `stack_edu_cpp` | 2,562 | 2 | 19,245 | 8,904.176 | 8,902.762 | −7.35e-5 |
| `nemotron_rewriting_python` | 0 | 3 | 24,182 | 16,724.599 | 16,724.021 | −2.39e-5 |
| `nemotron_rewriting_python` | 2,562 | 8 | 32,999 | 12,859.194 | 12,858.569 | −1.90e-5 |
| `climbmix` | 0 | 1 | 35,508 | 41,729.856 | 41,728.255 | −4.51e-5 |
| `climbmix` | 2,562 | 11 | 34,957 | 44,428.046 | 44,427.145 | −2.58e-5 |
| `ultradata_math_l3_qa` | 0 | 1 | 23,677 | 17,729.465 | 17,724.509 | −2.09e-4 |
| `ultradata_math_l3_qa` | 558 | 8 | 23,068 | 12,879.215 | 12,880.222 | +4.37e-5 |
| Pooled | | | 284,285 | 199,542.32 | 199,531.96 | −3.64e-5 |

Every row's bytes equal the record's; the pooled bits differ by −5.19e-5 relative. Loaded in bf16 (CPU), the same rows
differ by at most 1.15e-4 bpb and by +6.7e-6 relative pooled. The 164 HumanEval+ `"solution"` rows (27,104 bytes,
median 148 per row) differ in fp32 by −3.99e-4 relative pooled (Δbpb −1.97e-4), per row |Δbpb| mean 8.8e-4 and max
5.4e-3 (per-row bit differences' standard deviation 0.134 bits), in bf16 by −8.4e-5 pooled, per row mean 1.0e-3 and
max 4.7e-3 (0.157 bits): per-row differences of one size in either precision, on rows this short rounding noise rather
than a difference of the model [inference]. `tokenizer.json` encodes 171 texts (the HumanEval+ prompts with
solutions, the decoded pieces of the rows j = 0) as BOS + tiktoken's `encode_ordinary`, and maps the four special
tokens to 32,764–32,767. The 32K exports of `checkpoint-3307.pt` and `weights-3227.pt` give finite logits on a
9,001-token input. Greedy `generate` of 64 tokens from HumanEval/0's prompt equals the uncached forward's argmax at
every step, and a bf16 load generates the same tokens.

**vLLM** (`vllm/vllm-openai:v0.32.0`, its Transformers backend over this model file): the model declares
`_supports_attention_backend`; `config.json`'s `head_dtype` "float32" runs vLLM's head in fp32
(`LogitsProcessor._apply_head`; transformers ignores the field); the value-embedding gate is a `Gate` module outside
`nn.Linear`, since the backend's linear layers carry a weight loader that skips `_keep_in_fp32_modules_strict`. With
`--chat` the tokenizer carries the SFT chat template (`CHAT_TEMPLATE`, `docs/post-training.md` §5).

- Logit agreement (a d16 checkpoint at 5 tokens per parameter, step 1,707; 16 val rows of 8K from 8 corpora, 130,911
  scored tokens; greedy continuations of the 542 HumanEval+/MBPP+ prompts; GH200): the bf16 head was the main
  difference. With the fp32 head the export's mean KL(train ‖ export) under transformers is 1.97·10⁻⁴ (bf16 head
  3.09·10⁻⁴), top-1 agreement 0.9940 (0.9903), the bpb bias 5·10⁻⁶ (6.3·10⁻⁵, its interval excluding 0); under vLLM
  top-1 0.9941 (0.9904) and 379 of 542 greedy continuations identical to the training forward's (306). For scale: the
  uncompiled training forward against the compiled one, KL 3.06·10⁻⁴, top-1 0.9926, 340 identical. ReLU² and x0 in
  fp32 changed KL by −1% and −0.6%, inside their intervals; fp32 residual additions by −14%, with no gain in greedy
  agreement or pass@1 under vLLM and 7–10% slower transformers prefill and decode. EvalPlus pass@1 of the export's
  greedy continuations stays within one standard error (HumanEval+ 0.0488/0.0488 for every variant; MBPP+
  0.0265/0.0238 → 0.0238/0.0212).

**Cross-check of the official tools against the harness** (step 69,874: the official summary against the harness's
record of the same checkpoint, written by the removed harness modes at `be9027f` with `NativeDecoder`, §6, by
`metrics.compare`; MBPP 3-shot from the official run on lm-eval's own CRLF shots, `official-69874/lm-eval-crlf/`, as
the harness prompted). Per problem (exact McNemar; Holm over the rows):

| Benchmark | Harness | Official | Both / harness only / official only / neither | p (Holm) |
|---|---|---|---|---|
| HumanEval | 0.2683 | 0.2683 | 43 / 1 / 1 / 119 | 1.0 (1.0) |
| HumanEval+ | 0.2256 | 0.2195 | 36 / 1 / 0 / 127 | 1.0 (1.0) |
| MBPP (EvalPlus base) | 0.4233 | 0.4444 | 159 / 1 / 9 / 209 | 0.021 (0.26) |
| MBPP+ | 0.3519 | 0.3651 | 132 / 1 / 6 / 239 | 0.125 (1.0) |
| HumanEval-C++ greedy | 0.2236 | 0.2174 | 35 / 1 / 0 / 125 | 1.0 (1.0) |
| MBPP-C++ greedy | 0.3602 | 0.3678 | 141 / 2 / 5 / 249 | 0.45 (1.0) |
| HumanEval-C++ sampled (20) | 0.2255 | 0.2199 | — | 0.29, normal (1.0) |
| MBPP-C++ sampled (20) | 0.3503 | 0.3510 | — | 0.83, normal (1.0) |
| MBPP 3-shot | 0.206 | 0.184 | 86 / 17 / 6 / 391 | 0.035 (0.38) |
| Infilling single-line PSM | 0.5537 | 0.5528 | 565 / 7 / 6 / 455 | 1.0 (1.0) |
| Infilling multi-line PSM | 0.2600 | 0.2612 | 1,479 / 33 / 40 / 4,263 | 0.48 (1.0) |
| Infilling random-span PSM | 0.1165 | 0.1134 | 170 / 21 / 16 / 1,433 | 0.51 (1.0) |

No row differs after Holm's adjustment. The prompts are token-identical (EvalPlus, MultiPL-E, MBPP 3-shot 500 of 500,
infilling through the same `fim_prompt`), and so are the budgets and stops but two: MultiPL-E's 1,024 positions with
the prompt in the harness against 1,024 new tokens in `automodel_vllm.py`, and MBPP's stop, `"[DONE]"` in both, plus
the EOS string lm-eval adds and vLLM's EOS id (BOS), where `NativeDecoder` ends at BOS. All 44 greedy
problems whose verdicts differ (HumanEval+ 2, MBPP+ 11, HumanEval-C++ 1, MBPP-C++ 7, MBPP 3-shot 23) are token
divergences: the two completions differ at a token before either ends, so no stop or budget decides one. At the first
differing token the export's fp32 forward on CPU ranks the official token first 25 times and the harness's 19, by a
median |log-probability margin| of 0.038 nats (43 of the 44 below 0.15; HumanEval/96 1.5): near-ties that the two
bf16 decoders break apart [inference]. MBPP 3-shot's 23: the harness passes 17 and the official tool 6; among the 17,
fp32 prefers the official token 11 times, among the 6 the harness's 3 times; first divergence at generated token
0–70, median |margin| 0.048 nats. Infilling middles are identical for 971 of 1,033, 5,011 of 5,815 and 1,169 of 1,640
problems, and about as many run to the budget (harness 138, 843, 1,028; official 136, 835, 1,029). The harness's own
modes were removed on this evidence (`docs/experiments.md` §6); `evaluate` keeps the likelihoods, the parse measure
and the 1@10 mode.

**The reasoning protocol's requests** (`scripts/official_reasoning.sh`, `docs/post-training.md` §5) as `vllm serve`
received them [measured: `--enable-log-requests` on a step-1,000 SFT export, two HumanEval+ problems per EvalPlus
mode and lcb_runner's first requests, each tool run as the script runs it]:

| Tool | Requests | n | T | top-p | max_tokens | seed | stop, stop ids | End |
|---|---|---|---|---|---|---|---|---|
| EvalPlus greedy | one per problem | 1 | 0 | 1.0 (sent 0.95; vLLM sets 1 for T 0) | 32,615 (32,768 asked, capped at the context less the 153-token prompt) | none | none | eos BOS |
| EvalPlus sampled | one per problem | 20 | 0.6 | 0.95 | 32,615 | none | none | eos BOS |
| lcb_runner | one per problem | 16 | 0.6 | 0.95 | 32,337 for its first problem's 431-token prompt (32,768 capped) | none | none | eos BOS |

Every sampling value comes from the request; the export's `generation_config.json` defaults (T 0.6, top-p 0.95) would
apply only to a field a request omits, and none does. The end of text is not in any request: it is the export's
`eos_token_id`, BOS (`config.json`, `generation_config.json`), with `ignore_eos` false. No tool seeds its requests.
Every prompt's ids start with one BOS (32764) and the message's encoding (`<|bos|>Please provide a …` for EvalPlus,
`<|bos|>You are an expert …` for lcb_runner); the full ids' equality with the SFT context is
`eval/official_tokenize.py`'s check (§4).
One fix came from this check: EvalPlus's OpenAI backend drops `--max_new_tokens` (`evalplus.provider.make_model`
builds `OpenAIChatDecoder` without it), so every request asked 768 new tokens, `DecoderBase`'s default, and every
smoke sample ended at exactly 768 [measured]; `eval/official_evalplus.py` sets the decoder's budget from the flag.

## 9. Validation of the harness

- **MultiPL-E judging** (stored completions rescored, GH200's 64 cores): 64 concurrent `main.py` processes × 63
  threads oversubscribed g++, and a compile past its wall-clock limit is scored "SyntaxError"; with `--max-workers 1`
  per file, a d16 checkpoint's stored sampled completions gave HumanEval C++ "OK" 70 → 175 of 3,220 and pass@1
  0.022 → 0.054, MBPP C++ "OK" 307 → 712 of 7,940 and 0.039 → 0.090, greedy and filtered statuses unchanged.
- **Record replay** (`eval/rescore.py`, not on master; an empty verdict cache): every scorer and checker (EvalPlus
  base and plus, MultiPL-E's judge, the infilling checks, parse outcomes, the filtered mode's verdicts) rerun on a d24
  `eval-3307.json`'s stored completions, with 22,000 sampled completions synthesized from the stored ones, and compared
  field by field: 0 fields differed and the sidecars' sha256 matched.
- **Scorer replay** (an empty verdict cache): the official evaluators on every stored completion of the d24 benchmarks
  record `generate=benchmarks,learning_rate_scale=2.0,tokens_per_param=5.0` `eval-3307.json` (every mode) gave the
  stored verdict for 63,899 of 63,902 samples; the other 3, MBPP C++ sampled completions stored "Timeout", ran
  "Exception" (a stack overflow after 13.4–14.6 s against the 15-s limit), so no score changed. The harnesses' own
  CLIs on its greedy completions (human-eval-infilling on single-line and random-span SPM, MultiPL-E's
  `main.py --dir` and `pass_k.py` on both C++ benchmarks) gave its pass@1 exactly (0.0823, 0.0195, 0.0745, 0.1184).
  Every stored score of the seven d24 evaluation records (the peak-lr arms' `eval-3307.json`, the × 1 and × 2
  benchmarks records, the rehearsal's `eval-1102.json` and `eval-2205.json`) recomputes exactly from the per-sample
  fields; EvalPlus's and `pass_k.py`'s aggregate formulas agree with `"scores"` within 4 ulp.
- **Reference model** [unverified: no record]: deepseek-coder-1.3b-base through DeepSeek's harness gave HumanEval
  0.305 (paper 0.348, arXiv 2401.14196 Table 3; transformers 5 drops its BOS) and MBPP 0.466 (paper 0.462); this
  suite's EvalPlus scorer agreed on 163 of 164 problems.
- **d16 reference** [unverified: no record; which d16 checkpoint and harness revision produced it is not recorded]: at
  20 tokens per parameter, code bpb 0.439, HumanEval+ 0.061, MBPP+ 0.071/0.050. The suite's greedy MBPP+ on the
  step-4,552 checkpoint was 0.045/0.037, so the two are not the same evaluation [inference].

## References

| arXiv | Title |
|---|---|
| 1411.0030 | A* Sampling |
| 1904.09751 | The Curious Case of Neural Text Degeneration |
| 2101.00027 | The Pile: An 800GB Dataset of Diverse Text for Language Modeling |
| 2107.03374 | Evaluating Large Language Models Trained on Code |
| 2108.07732 | Program Synthesis with Large Language Models |
| 2203.07814 | Competition-Level Code Generation with AlphaCode |
| 2204.02311 | PaLM: Scaling Language Modeling with Pathways |
| 2207.14255 | Efficient Training of Language Models to Fill in the Middle |
| 2208.08227 | MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation |
| 2303.17568 | CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Benchmarking on HumanEval-X |
| 2305.01210 | Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation |
| 2308.12950 | Code Llama: Open Foundation Models for Code |
| 2401.02954 | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism |
| 2401.14196 | DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence |
| 2402.17988 | Constrained Decoding for Fill-in-the-Middle Code Language Models via Efficient Left and Right Quotienting of Context-Sensitive Grammars |
| 2402.19173 | StarCoder 2 and The Stack v2: The Next Generation |
| 2403.01632 | SynCode: LLM Generation with Grammar Augmentation |
| 2409.12186 | Qwen2.5-Coder Technical Report |
| 2411.00640 | Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations |
| 2412.04403 | Establishing Task Scaling Laws via Compute-Efficient Model Ladders |
| 2503.13423 | SuperBPE: Space Travel for Language Models |
| 2504.09246 | Type-Constrained Code Generation with Language Models |
| 2504.11393 | DataDecide: How to Predict Best Pretraining Data with Small Experiments |
| 2505.09388 | Qwen3 Technical Report |
| 2508.13144 | Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation |
| 2604.07769 | An Empirical Study on Influence-Based Pretraining Data Selection for Code Large Language Models |
| 2606.21619 | The Alignment Problem in Constrained Code Generation |
| 2608.18062 | TokEval: A Tokenizer Evaluation Suite; companion post cimeister.github.io/blog/code_tokenizer_ablations |

Repositories and other sources, at the pins of `docker/Dockerfile.eval` (the official tools) and `docker/Dockerfile`
(the harness's judges): `evalplus/evalplus` (`26d6d00`; the harness's in `pyproject.toml`), `nuprl/MultiPL-E`
(`3025a53`; dataset in `eval/code_eval.py`), `EleutherAI/lm-evaluation-harness` (`d6de816`: `tasks/mbpp/`,
`models/vllm_causallms.py`, `models/utils.py`), `openai/human-eval-infilling` (`88062ff`),
`LiveCodeBench/LiveCodeBench` (`28fef95`), `zai-org/humaneval-x`, `google-research-datasets/mbpp`; at the files
cited: `huggingface/evaluate` (`metrics/code_eval/`), `pytorch/torchtitan`, `NVIDIA/Megatron-LM`, `python/cpython`
3.12.3 (`Parser/tokenizer.c`, `Grammar/python.gram`; issue #121130), `guidance-ai/llguidance` (issue #107),
`vllm-project/vllm` (v0.32.0: `vllm/benchmarks/serve.py`, the Transformers backend, `LogitsProcessor`),
`huggingface/transformers` (`modeling_qwen3.py`, `TikTokenConverter`), NVIDIA's H100 SXM specifications, Philox
("Parallel Random Numbers: As Easy as 1, 2, 3", SC '11).
Token healing: "The Art of Prompt Design: Prompt Boundaries and Token Healing" (Guidance, 2023).
