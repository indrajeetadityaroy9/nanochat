# Evaluation and inference

How a checkpoint is scored (likelihoods, the parse measure, the code benchmarks), every metric and error bar the
repository computes (`metrics.py`), the decoder that generates every completion, the syntax constraint and checks of
the filtered mode, and the export through which transformers, vLLM and the benchmarks' own harnesses run the model.
The code is the specification; each statement names the file and symbol that define it.

Markers: [arithmetic] derived from code numbers; [inference] reasoning, not measured; [estimate]; [unverified] a
figure without a record or commit, or a paper value its extracted text lacks. Measurements name their commit and
setting; "d16 checkpoint" is a 20-tokens-per-parameter d16 run's final checkpoint.

## Code

| File | Section |
|---|---|
| `eval/suite.py` (`evaluate`, `record`, `evaluation_path`) | §1 |
| `eval/bpb.py` (`likelihood`, `val_bits`, `solution_bits`, `sums`, `sample_rows`) | §2 |
| `eval/code_eval.py` (`tasks`, `parse_generation`, `generation`, `fim_generation`, `filtered_generation`, `fim_filtered_generation`, evaluator wrappers) | §1, §3, §4 |
| `eval/verify.py` (`search`, `python_diagnostics`, `solution_diagnostics`, `cpp_diagnostics`, `examples_pass`) | §7 |
| `metrics.py` (every function) | §5 |
| `inference/decoder.py` (`NativeDecoder`, `Request`, `fim_prompt`, `heal_tokens`, `Reemit`), `model/kernels/decode.cu` | §6 |
| `inference/syntax_constraint.py`; `inference/syntax/` (`lib.rs`, `lexer.rs`, `parser.rs`, `vocab.rs`, `python.lark`, `Cargo.toml`, `Cargo.lock`) | §7 |
| `inference/export_hf.py`, `inference/modeling_nanochat.py`, `scripts/standard_eval.sh` | §8 |
| `eval/__init__.py`, `inference/__init__.py` (empty package markers) | — |

## 1. Protocol

`eval/suite.py` `evaluate(model, tokenizer, run_name=, step=, generate=)` writes `RUNS/<run>/eval-<step>.json`: the
likelihood metrics (`"8k"`, `"32k"`, `"solution"`, §2), the parse measure (`"parse"`, §3) and, with
`generate="benchmarks"`, the generation benchmarks (`"codegen"`, `"fim"`, §4); with `generate="parse"` both are null.
The base model's checkpoints are evaluated with `"benchmarks"`, ablation runs with `"parse"` (which checkpoints and
when: `docs/training.md` §6). `record(run_name, log)` gathers a run's non-train log records and its evaluations in
step order, the unit of `results/<experiment>.jsonl` (`docs/training.md` §6).

Order of work (`evaluate`): the unconstrained decodes (`parse_generation`, `generation`, `fim_generation`), the
likelihood pass, then the filtered searches, last because their checks are timed and the CPU-light likelihood pass has
drained most of the scoring queue. Each benchmark's scoring (its official evaluator, CPU, in a fresh process) goes to
one scorer thread as soon as its completions exist; EvalPlus's sanitize workers (`code_eval.sanitizers`,
`os.cpu_count()` forkserver processes) live for the call. Results do not depend on this order (a request's tokens
depend on the model and the request alone, §6) except timed checks, which heavy CPU load can fail.

Measured on a d16 checkpoint (GH200): before the pipeline, the codegen half's 969 s were 36% CPU scoring with the GPU
idle, 28% likelihood, 15% generation (85–96% SM busy), 22% filtered search [unverified: no record or commit];
pipelining (`a41415e`) took the full benchmarks evaluation from 1,472 to 1,064 s (−28%), GPU idle 57.4% → 34.2%,
scoring with no GPU work 483 → 8.4 s, 0 per-sample changes where the protocol was unchanged; with the decoder of
`4385748` (§6), 862 s.

## 2. Likelihood metrics

`eval/bpb.py` `likelihood` runs the compiled `CausalLM.losses` (one graph per row shape). All values are bits per
byte (§5 `bpb`: summed bits over summed UTF-8 bytes of the targets, The Pile, arXiv 2101.00027 §3.1;
tokenizer-independent). Targets of zero bytes (BOS, padding, FIM sentinels) are excluded from both sums, as
karpathy/nanochat's `evaluate_bpb`. Sums are fp64 on the device, reduced in a fixed order.

- **Held-out corpora** (`val_bits`): each corpus's packed val split (`docs/corpus.md` §8) in the stage's rows: at 8K
  (`FILE_LEVEL_STAGE`) always, at 32K (`REPO_LEVEL_STAGE`) only for a model at the 32K context
  (`max_position_embeddings == 32768`), else null. This is the AR loss of arXiv 2207.14255 §2. Per corpus: bits and
  bytes per context bucket (`CONTEXTS = (8192,)`: positions ≤ 8K, which both stages train, and > 8K, which only the
  repository stage trains; Code Llama's loss by context length, arXiv 2308.12950 §3.3, Fig. 4a), per byte-offset
  bucket (`OFFSETS`: doublings from 1 KiB to 128 KiB of the target's segment before it, so tokenizers compare at equal
  bytes of context, arXiv 2503.13423 §3.1, App. B.1) and per sampled row. Pooled per group (code, text, math) and
  over all corpora by the stage's train row shares (`Stage.shares`, `docs/corpus.md` §7) with `metrics.mixture_bpb`,
  `offset_bpb` and `mixture_bpb_summary` (§5).
- **Sample** (`sample_rows`, `VAL_TOKENS = 80 × 2^19`, nanochat's `--eval-tokens`): per corpus, n = min(rows,
  ⌈VAL_TOKENS / (micro-batch × seq_len)⌉ × micro-batch) rows, row ⌊j · rows / n⌋ for j < n: systematic, because the
  packer orders each shard's rows longest first, so a prefix would over-represent long documents. Models of one
  tokenizer are scored on identical rows, models of different tokenizers on identical documents when a split lies
  within the sample (scored whole), so the subsample's sampling error cancels in differences. `cu_seqlens` is padded
  to the corpora's most segments (one compiled graph).
- **Reference solutions** (`solution_bits`, `SOLUTION_BENCHMARKS`): the bits of HumanEval+'s and MBPP+'s canonical
  solutions (EvalPlus) and HumanEval-X C++'s (arXiv 2303.17568; MultiPL-E has none) given their prompts, one example
  per 8K row, BOS first: the task loss of arXiv 2412.04403 §2.2. A continuous correct-answer likelihood makes
  HumanEval and MBPP decisions predictable where accuracy is trivial (arXiv 2504.11393 §3.4, Fig. 6); BPB raises the
  signal-to-noise ratio of most benchmarks (arXiv 2508.13144 §5.3); the loss on a benchmark's problems and reference
  solutions tracks pass@1 where held-out loss does not (arXiv 2604.07769 §3.3.1). A solution's leading blank lines
  count as prompt: EvalPlus's solutions begin with them and a tokenizer joins them to the prompt's last line break, so
  the prompt would otherwise end inside a token and tokenizers would score different bytes. Scoring starts at the
  token that starts at the prompt's end (`decode_with_offsets`; none raises).

## 3. Parse measure

`code_eval.parse_generation`: one greedy, unconstrained completion of each HumanEval+ (164) and MBPP+ (378) prompt,
BOS first, the prompt healed at its boundary (`decoder.heal_tokens`, §6: the last prompt token dropped and generation
constrained until its bytes are re-emitted), as arXiv 2608.18062 generates (§4.3; App. C.1), with EvalPlus's prompt,
768 + 1 new tokens (`EVALPLUS_MAX_NEW_TOKENS` and the healed token) and stop strings; the completion is cut at its first
stop, special tokens skipped. Each `prompt + completion` is classified by `metrics.parse_outcome` (§5) into parses,
IndentationError (TabError included) or another SyntaxError, the classification of TokEval's companion post; the
record holds each outcome's share and the completions' EvalPlus base and plus pass@1. The measure shows the failures a
vocabulary's line-break tokens cause (SyntaxError when punctuation fuses with a line break, IndentationError when a
line break fuses with the indentation after it, in arXiv 2608.18062's models) where pass@1 is near zero. It departs
from TokEval's protocol in its prompts (EvalPlus's 0-shot base prompts; TokEval: lm-evaluation-harness, MBPP 3-shot,
App. C.1), its budget and stops (the post: 256 tokens) and in compiling the unsanitized completion.

## 4. Code benchmarks

`eval/code_eval.py` `tasks()` loads the benchmarks; each is scored by its official evaluator in a fresh process.

| Benchmark | Problems | Prompt; budget; stops | Evaluator | Data pin |
|---|---|---|---|---|
| HumanEval+, MBPP+ (EvalPlus, arXiv 2305.01210) | 164, 378 | `prompt.strip() + "\n"`; 768 new tokens; EvalPlus's EOS + `extra_eos_for_direct_completion`, special tokens skipped, tabs → 4 spaces, `sanitize(prompt + completion, entry point)` (EvalPlus's base-model codegen) | `evalplus.evaluate` (all CPUs, `EVALPLUS_EVALUATE`): base and plus pass@1 | evalplus `26d6d00` (`pyproject.toml`); HumanEvalPlus v0.1.10, MbppPlus v0.2.0 |
| HumanEval, MBPP in C++ (MultiPL-E, arXiv 2208.08227) | 161, 397 | `prompt.strip()`; 1,024 positions of prompt and completion, BOS included (a prompt of ≥ 1,024 tokens gets the empty completion, as MultiPL-E's `automodel.py` returns); the row's stop tokens, the first special token ends it | MultiPL-E `evaluation/src/main.py --file … --max-workers 1` per problem, `os.cpu_count()` at once (g++ -std=c++17, 15 s compile, 15 s run); `pass_k.py` | dataset `nuprl/MultiPL-E` at `3a9e8d2` (the revision MultiPL-E's `completions.py` pins); evaluator `3025a53` (`docker/Dockerfile`) |
| HumanEval-X C++ (arXiv 2303.17568) | 164 | reference solutions only (§2) | — | `zai-org/humaneval-x` at `62c7862` |
| HumanEval infilling single-line, multi-line, random-span (arXiv 2207.14255) | 1,033, 5,815, 1,640 [arithmetic: `read_problems`] | PSM and SPM prompts (`decoder.fim_prompt`, the packer's layout, `docs/corpus.md` §8); middle until BOS, at most the benchmark's longest reference middle in tokens; not healed (a FIM prompt ends with a sentinel or with the prefix encoded alone, the training format) | the package's `check_correctness` per middle (3 s) in a fork pool of `os.cpu_count()` workers (`INFILLING_EVALUATE`) | `openai/human-eval-infilling` `88062ff` (`docker/Dockerfile`) |

Generation modes (`generation`, `fim_generation`, `filtered_generation`, `fim_filtered_generation`):

| Mode | Decoding | Score |
|---|---|---|
| `greedy` | left-to-right argmax, one completion per problem; FIM in both orders | pass@1 as DeepSeek-Coder (arXiv 2401.14196 §4.1), StarCoder2 (arXiv 2402.19173 §7.1.1) and EvalPlus's leaderboard report it |
| `sampled` (EvalPlus, MultiPL-E) | `N_SAMPLE` 20 per problem at `TEMPERATURE` 0.2 from the nucleus `TOP_P` 0.95 (arXiv 1904.09751); sample k of problem i seeded N_SAMPLE · i + k | pass@1 (and pass@10) by Codex's unbiased estimator (arXiv 2107.03374 §2.1; T and p from §3.2–3.3), as MultiPL-E and StarCoder2 (§7.1.2) sample |
| `filtered` | the greedy candidate, then samples at T = 0.8 (`verify.TEMPERATURE`, seed = attempt) until one has no post-generation diagnostic and passes its prompt's own examples, at most `verify.ATTEMPTS` = 10; left-to-right prompts healed; Python and FIM middles under the syntax constraint (§7); FIM in SPM only (`FILTERED_MODE`) | filtered pass@1 of the accepted candidate, else the greedy one (Codex §3.5; AlphaCode, arXiv 2203.07814 §4.5): a system measurement, not comparable to pass@1 tables |

Each record also holds `"scores"` (`metrics.summary` of the per-problem scores, §5), the completions and the
decoding's `"latency"` (`metrics.latency`); a filtered record holds the attempts per task and whether one was accepted.

- **Infilling.** The middle is scored as generated, up to BOS or the budget (the package's protocol, no
  post-processing). `greedy_reference_lines` also scores the greedy single- and multi-line middles cut to the
  reference middle's line count (`fim_lines`; arXiv 2505.18789 Table 1, Fig. 4): a separate metric, since it reads the
  reference. The references' middles are whole lines while the model, trained on character-level and AST spans
  (`docs/corpus.md` §8), rarely ends at a line end: on a d16 checkpoint single-line SPM greedy pass@1 was 0.074 as
  generated against 0.437 cut (`a41415e`). Infilling records add CrossCodeEval's similarity scores against the
  reference middle (§5), all clustered by HumanEval problem.
- **Comparability.** The sampled mode draws 20 samples per problem where Codex and MultiPL-E drew 200 and the
  BigCode leaderboard 50: the same estimate with more variance per problem [inference] (MultiPL-E reports its pass@1
  stable from about 20 samples, arXiv 2208.08227 [unverified: section not checked]). The filtered mode's T = 0.8 is
  Codex's pass@100 optimum (higher temperatures are optimal for larger k, §3.3). HumanEval infilling is greedy,
  not arXiv 2207.14255's sampling. The prompts' examples overlap the scored base tests (MBPP+: nearly every prompt
  assert calls the entry function on a base input; HumanEval+: many tasks share a doctest input), so the filtered mode
  favours base over plus pass@1 (`verify.py`). Prompts, stops, budgets and scorers were checked against EvalPlus,
  MultiPL-E and human-eval-infilling at their pinned commits (the FIM line cut, the MultiPL-E data pin and the
  example checks' sandbox were the deviations, fixed in `a41415e`). `char` is unsigned on ARM64 and signed on x86, so
  a MultiPL-E C++ program that depends on it can score differently [inference].
- **Contamination.** `corpus/` has no decontamination step against the scored benchmarks (the only decontaminated
  data are the publisher's `*_decontaminated` splits of `open-r1/codeforces-cots`, `corpus/sources.py`): a benchmark
  problem present in the training data inflates its score.
- **Evaluator processes.** EvalPlus's `sanitize` parses every line span of a program, hence a sanitize worker per
  CPU. MultiPL-E's `main.py --dir` judges a directory's problems serially, hence a `--file` process per problem.
  human-eval-infilling's CLI forks a Manager and a Process per sample from one interpreter's thread pool, so its rate
  is bound by that interpreter's GIL (4 or 64 threads about the same), hence `INFILLING_EVALUATE`'s fork pool.
  MultiPL-E's `automodel.py` returns the empty completion for a prompt of ≥ 1,024 tokens except a 1,024-token prompt
  whose one generated token outspans the truncated prompt; this suite gives it the empty completion.

## 5. Metrics and statistics

`metrics.py` is the repository's one source of metrics, torch-free so training, evaluation and analyses import it
alike. Every public function:

| Function | Computes | Source definition; library |
|---|---|---|
| `append`, `records`, `train_steps`, `diverged`, `curves` | the run log `RUNS/<run>/metrics.jsonl` (record kinds and fields: §10, `metrics.py`): append one JSON record with `run_name`, `attempt`, `time` (NaN and infinities raise, `allow_nan=False`); read it; the latest attempt's train record per step; the last failure record's step or None; train records as columns in step order (None where a record lacks the field: `update_rms`, `param_norm` exist every 10th step) | — |
| `bpb(*pooled)` | Σ bits / Σ bytes over context buckets and arguments | The Pile, arXiv 2101.00027 §3.1 |
| `mixture_bpb(corpora, weights, buckets=)` | Σ_c w_c · bits_c / rows_c over Σ_c w_c · bytes_c / rows_c: the bits per byte of val rows drawn with row weights w | as The Pile pools its components |
| `offset_bpb` | `mixture_bpb` per byte-offset bucket (None where no target is); a bucket pools the documents long enough to reach it, so the curve mixes context use with document composition | arXiv 2308.12950 Fig. 4a; 2503.13423 App. B.1 |
| `bpb_summary(bits, nbytes, row_weights=)` | the ratio Σw·bits / Σw·bytes over rows (0-byte rows dropped) as the WLS fit of row bpb on a constant with weights w × bytes; standard error HC1 (the delta-method ratio standard error × √(n/(n−1))); 95% normal interval | Miller, arXiv 2411.00640 Eq. 1, 5–6; statsmodels `WLS` |
| `mixture_bpb_summary` | `bpb_summary` of every sampled row of a group, a row of corpus c weighted w_c / rows_c (the `mixture_bpb` weights) | as above |
| `parse_outcome(source)` | `compile()` → "parses", "indentation" (IndentationError, TabError; checked first) or "syntax" (other SyntaxError; MemoryError and RecursionError of the parser, and its ValueError of §10, count here) | TokEval's companion post ("How the failing generations fail"; arXiv 2608.18062 App. F.3, fn. 15) |
| `parse_shares` | each outcome's share of a benchmark's completions | — |
| `measures(evaluation, steps, repo_level_start=)` | a run's compared measures (table below) | — |
| `measure_errors(evaluation)` | their error bars: `code`, `text`, `math`, `repo_32k` by `mixture_bpb_summary`; `solutions` by `bpb_summary`; the parse shares and `pass@1` by `summary` over problems; none for `repo_beyond_8k`, `tokens_per_second`, `mfu` | Miller Eq. 1 |
| `line_exact_match` | `prediction.strip() == reference.strip()` | SantaCoder (arXiv 2301.03988 §6.1.2; bigcode-evaluation-harness `santacoder_fim.py`), DeepSeek-Coder §4.2 |
| `code_lines`, `exact_match` | comment-stripped (`#`, `//` to the line end), stripped non-empty lines; equal lines | CrossCodeEval (arXiv 2310.11248 §4; amazon-science/cceval `40c68d2`) |
| `edit_similarity` | 100 when equal, 0 when either is empty, else round(100 × difflib `SequenceMatcher` ratio) of the comment-stripped strings | cceval's fuzzywuzzy 0.18 without python-Levenshtein (difflib) |
| `identifiers`, `identifier_match` | identifiers after removing comments and string literals, Python keywords excluded (True and False kept); exact match of the lists, F1 over the sets (0 when both are empty) | cceval `extract_identifiers` |
| `completion_scores` | per sample: `line_em`, `em`, `es` (0–100), `id_em`, `id_f1` | cceval's per-sample results |
| `pass_at_k(passed, k)` | each problem's pass@k estimate from its samples | Codex's unbiased estimator (arXiv 2107.03374 §2.1); `human_eval_infilling` `estimate_pass_at_k` |
| `summary(scores, clusters=)` | mean of question scores (each score a mean of question-level scores, a problem's pass@k estimate one of them: Miller §3.1 [unverified: paper text not in the extracted set]); OLS on a constant: unclustered, se = s/√n (sample variance over n − 1, Miller Eq. 1); clustered, Liang–Zeger CR0 without small-sample correction (Miller Eq. 4 except for Eq. 4's (n − 1) in its diagonal term); 95% normal interval (Eq. 5–6); a constant score has se 0 | statsmodels `OLS` |
| `model_flops_per_token(matmul_params, hidden_size, windows, context)` | nominal training FLOPs per token: 6N + 12 · C · Σ_layers min(window_l + 1, context), each layer's most keys a query attends: itself and up to window_l before it (`layer_windows`: 512, or the context for full layers), the convention of `attended_keys`; records written before the query's own key was counted (`results/muon_wd_timescale_20tpp.jsonl`, `results/scalar_learning_rates.jsonl`, d16) hold 6N + 12 · C · Σ min(window_l, context), 73,728 lower: 641,826,816 at 8K, 1,245,806,592 at 32K | PaLM, arXiv 2204.02311 App. B; torchtitan's count without the causal halving |
| `attended_keys(lengths, window)` | Σ over segments of Σ_p min(p, window + 1): the keys each query attends causally within its segment, itself included | — |
| `executed_model_flops(lengths, matmul_params, hidden_size, windows)` | 6N per position + 12 · C per attended key per layer over a micro-batch's segments (QKᵀ and PV, 4C per key forward, twice that backward; FA4's score recomputation not counted) | PaLM App. B; Megatron-LM's causal, per-document counting |
| `mfu(flops, seconds)` | flops / seconds / `PEAK_FLOPS` (989.4e12, the H100 SXM5 / GH200 dense BF16 peak) | PaLM App. B |
| `parabola_vertex(x, y)` | (a, −b/2a) of the least-squares parabola (numpy `polyfit`, degree 2); used by `docs/training.md` §5 | — |
| `latency(starts, firsts, ends, new_tokens, duration=)` | `ttft` (first − start), `tpot` ((end − first)/(n − 1), requests of > 1 token; None if none), `e2el`, each mean, median, p99; `output_throughput` | vLLM `benchmarks/serve.py` `calculate_metrics` |
| `token_offsets`, `tokeval_wrapper`, `tokeval` | TokEval's full analysis of repository tokenizers (below) | arXiv 2608.18062; cimeister/tokenizer-intrinsic-evals `7d03a61` |

**Measures** (`measures`, the quantities experiments compare; bits per byte unless stated):

| Measure | Definition |
|---|---|
| `code` | 8K validation of the code group at its corpus weights |
| `solutions` | HumanEval+, MBPP+ and HumanEval-X C++ reference solutions given their prompts, pooled (`bpb`) |
| `repo_32k`, `repo_beyond_8k` | 32K validation of the code group; the same over targets beyond 8,192 positions (the last context bucket), at the 32K evaluation's code-corpus weights restricted to `REPO_LEVEL_STAGE.code` |
| `text`, `math` | 8K validation of each group |
| `unparsed`, `indentation`, `syntax` | shares of the 542 HumanEval+ and MBPP+ problems [arithmetic: 164 + 378], pooled, whose healed greedy completion does not compile; by IndentationError; by another SyntaxError (§3) |
| `pass@1` | greedy base pass@1 of the parse measure's completions, pooled |
| `tokens_per_second`, `mfu` | medians over the file-level stage's steps (before `repo_level_start`) of the train records' tokens per second and executed MFU |

**Model FLOPs and MFU.** The train records' `mfu` is executed model FLOPs per step (`executed_model_flops` of each
micro-batch's segment lengths, read from the loader without a device synchronization) over the step's device time and
`PEAK_FLOPS`; the run record's `model_flops_per_token` is the nominal count. DeepSeek LLM's M (arXiv 2401.02954 §3.2 Eq.
2) counts full attention in every layer and stays the scaling plan's compute, not an MFU numerator; the d24 values of M
and the nominal count: `docs/architecture.md` §1. Executed FLOPs per token of a row holding one document are
1,479,503,808 at 8K and 2,160,975,600 at 32K [arithmetic, recomputed on CPU], below nominal (a query attends only its
own segment's earlier keys) and lower still with packed documents; M is 3.8× the executed count at 32K [arithmetic], so
an M-based utilization would exceed 100%. `torch.utils.flop_counter` would count the same once each custom op registered
its formula, but per step it would read `cu_seqlens` on the host (a device sync) and dispatch every op through Python.
FLOPs count 2 per multiply-add; rematerialization is not counted (PaLM App. B's MFU definition). `tokens_per_second` is
the observed throughput of PaLM's MFU; `peak_memory_bytes` is the footprint FlashAttention reports (arXiv 2205.14135),
`peak_reserved_bytes` the allocator's. Measured MFU: `docs/training.md` §7.

**TokEval** (`tokeval`): `UnifiedTokenizerAnalyzer.run_analysis` in its published form (`normalize_results`), MorphScore
off (it needs its git submodule and data) and plots off, CER unbudgeted (`cer_time_budget_s=0`). Inputs: prose by
language, code by language (via TokEval's `CodeDataLoader`: CRLF → LF, trailing whitespace stripped; read by the AST,
identifier and indentation metrics) and math. `tokeval_wrapper`: `encode` is `encode_ordinary`; vocabulary keys are
GPT-2's byte-to-unicode rendering of each token's bytes (a bijection, so partial characters survive); `pretokenize` is
the split pattern; offsets from each token's byte length (`token_offsets`). Tree-sitter grammars:
`tree-sitter-language-pack==1.22.1` (`pyproject.toml` override). Results per tokenizer: `docs/experiments.md` §2.

## 6. Decoder

`inference/decoder.py` `NativeDecoder` with `model/kernels/decode.cu` generates every completion of §3–§4.

- **Batching.** Up to `MAX_BATCH` = 1,024 slots (fewer if device memory is short), continuous: a finished slot takes the
  next request. A greedy step is one captured CUDA graph and a sampled or constrained step two (body, then head), each
  graph's kernels programmatic dependents of each other (PDL); a second set of graphs over the first `SMALL_BATCH` = 256
  slots runs while no slot at index ≥ `SMALL_BATCH` is active. A slot's KV cache holds 2 · L · max_length · (C + 4H)
  bytes, 38,016 bytes per position at d24 [arithmetic]. A many-slot step is bound by its KV reads (each active slot
  reads its window's K and V in every layer) and its GEMMs; a few-slot step by weight reads and kernel launches
  [inference].
- **Step.** The model's forward at one position (`docs/architecture.md` §1), GEMMs on wgmma tensor cores fed by TMA,
  attention streaming each slot's window through shared memory with the scores on tensor cores (f16, exact for e4m3
  keys and the bf16 q). Numerics follow the compiled training forward op by op (bf16 copies of the matrices; fp32
  rotary table, q/k norm gains and value-embedding gate; the residual mix as Inductor fuses it, `ca18db5`), except the
  KV cache: e4m3 with one fp32 scale per position and head (amax / 448), half the bytes a step reads; scores and
  softmax stay fp32.
- **Prefill and prompt store.** A new request's prompt but its last token runs as rows of one batched pass through
  the step's body (up to 1,024 rows of any prompts per launch, each row computed as a step would), so its slot starts
  at the prompt's last token. Its K and V are kept in a paged store (`PAGE` = 64 positions; vLLM's PagedAttention,
  arXiv 2309.06180, pages likewise; prefix reuse as SGLang's RadixAttention, arXiv 2312.07104), and a later request
  with the same prompt copies them; only prompts unused in the current and previous `generate` call can be evicted,
  least recently used first, and if that frees too few pages the new prompt is not stored (so a search's rounds keep
  the prompts they reuse, and a too-small store keeps a fixed part rather than cycling) [inference].
- **Request** (`Request`): `input_ids`, `max_new_tokens`, `stopping_criteria`, `temperature` (0: argmax, ties to the
  lowest id; > 0: Gumbel-max sampling of softmax(z / T), the noise a Philox function of (seed, generated position,
  token) alone, arXiv 1411.0030 §2), `top_p` (< 1: the fewest most likely allowed tokens holding top_p of the
  probability, arXiv 1904.09751 §3.1, `k_nucleus`; transformers' `TopPLogitsWarper`, the sampler of MultiPL-E's
  `automodel.py` and bigcode-evaluation-harness), `seed`, `constraint`. Sampled logits are capped, z = c·tanh(s/c)
  with the model's `final_logit_softcapping`; a greedy choice needs no cap (tanh is increasing). Generation ends at
  BOS (excluded), the budget, or the stopping criteria (evaluated while the next step runs).
- **Constraints.** A step with a sampled or constrained slot launches the body, writes the constraints' masks while it
  runs (the syntax constraints' in one parallel call straight into pinned mask rows, §7), then the head, which chooses
  within each slot's mask and also computes its best allowed token but BOS; a chosen BOS that the syntax constraint
  rejects gives way to that alternative (the same choice as masking BOS beforehand; a nucleus-sampled slot's BOS is
  decided before the step). `heal_tokens` drops the prompt's last token and returns its `Reemit` constraint: each
  token must be a proper prefix of the owed bytes or start with them, until they are re-emitted (token healing,
  Guidance 2023, extended over several tokens as arXiv 2608.18062 App. C.1).
- **Determinism.** Every sum's order depends on the shapes and the slot's position alone, so a request's tokens do not
  depend on the batch, the store or timing (64 of 64 greedy and 64 of 64 sampled requests identical alone and
  batched, `4385748`). The FP8 cache makes them differ from the bf16 forward's where two logits nearly tie: agreement
  with the training forward's greedy choice 0.9916 at `4385748` (137 of 16,261 tokens; every mismatch a top-2 gap
  < 0.57, median 0.0625) and 0.9920 at `ca18db5`.
- **Memory.** `memory_fraction` caps the decoder's device memory (slots, prefill rows, graphs, store) at a fraction of
  the device's total (`NANOCHAT_DECODER_MEMORY_FRACTION`; unset: all free memory), so a job can share the GPU.
- **Speed** (`4385748`, GH200, fixed-length benchmark: 2,048 random-init requests, 256-token prompts, 512 new tokens):
  94,778 tok/s at d24 and 189,307 at d16, against 21,809 and 66,559 for the per-slot-group GEMV decoder it replaced
  (4.35×, 2.84×); d24 step 8.27 ms at 1,024 slots, one slot 1.29 ms (GEMV decoder 3.16 ms). A d16 checkpoint's
  per-problem results changed little and in both directions (MBPP+ greedy +4/−5 base, +2/−7 plus; HumanEval+ greedy
  unchanged; FIM line exact match within 0.002).
- **Tuned constants** (d24, GH200, 256-token prompts; 1,024 slots and 512 steps unless stated). GEMM tiles and
  `WG_STAGES` (`4385748`): 64-row × 64-slot tiles 8.27 ms per step against 8.33 ms for 128 × 128 where they cover
  every SM; 4 stages against 8.82, 8.44, 8.33, 9.22 ms at 2, 3, 6, 8. `ATTN_RUN` (`4385748`, under the earlier tile
  dispatch): one MMA tile per warp 8.31 ms against 8.80 (two) and 15.45 (four). `SMALL_BATCH` = 256 (`55e64de`):
  fastest or within 0.04 ms at 1, 64, 128, 256 active slots against 64, 128, 512 and no narrow graphs (1 slot
  1.29 ms, none 1.46; 256 slots 2.81, none 2.97).

## 7. Syntax-constrained decoding and post-generation checks

**Syntax, not types.** Type-constrained decoding (arXiv 2504.09246) covers TypeScript only. An incomplete constrainer
(one rejecting valid programs) lowers functional correctness below unconstrained decoding, while a complete one helps
(arXiv 2606.21619, abstract; §5.3 and a larger gain for small models [unverified: paper text not in the extracted
set]; its rejection as a design: `docs/experiments.md` §6). Python 3 syntax can be constrained completely, and every
Python file of the code categories parses as Python 3 (`curate.repaired`, `docs/corpus.md` §3; UltraData-Code L3's
exercises and the Kaggle notebooks are not checked), so the model's distribution of program files lies inside the
constrainer's language [inference]. Types are checked after generation instead.

**`SyntaxConstraint`** (`inference/syntax_constraint.py`, Rust crate `inference/syntax/`):
`SyntaxConstraint(tokenizer, prefix, *, suffix, healing=None)` (bytes; the vocabulary tables are the caller's
tokenizer's, built once per tokenizer by `python_syntax`; the crate compiled on first use by `native.crate_module`).
Token t is allowed iff `prefix + generated + t` can still
become a valid Python 3.12 module, the text ending anywhere; FIM sentinels never; BOS iff `ast.parse` accepts the text
followed by the FIM suffix (`ends`: the grammar ends a module after text and suffix, then `ast.parse`). While a healed
prompt owes bytes, each mask is intersected with the `Reemit`'s. A prefix that cannot become valid Python 3.12 raises
`ValueError`. `masks(constraints, out, rows)` writes a batch of masks in parallel (rayon) without the GIL. Layers:

- lexer (`lexer.rs`): CPython 3.12.3's tokenizer (`Parser/tokenizer.c`) as a character-level state machine, so a
  prefix may end mid-token, mid-indent or mid-character; `\r\n` and `\r` read as `\n`;
- indentation (`parser.rs`): INDENT/DEDENT (tab size 8), the tab-consistency check, the bracket level;
- parser (`parser.rs`): an LALR(1) grammar (`python.lark`) accepting every token sequence of CPython 3.12's PEG
  grammar, tables built by Lark at load (strict mode proves them conflict-free), read as a viable-prefix recognizer;
  the soft keywords `match`, `case`, `type` read both ways until one parse dies;
- masks (`vocab.rs`): per lexer state, every token's events and end state computed once over the vocabulary's byte
  trie; a step runs the parser once per distinct event sequence.

Where it is looser than CPython it accepts (over-approximations listed in §10, `inference/syntax_constraint.py`:
assignment targets as expressions, free parameter order, unchecked escapes and nesting limits, among others); BOS is
exact.
Libraries not used: SynCode (arXiv 2403.01632; its Python grammar, of Lark's `python.lark` lineage, lexes an f-string
as one regex token, so PEP 701 f-strings such as `f'{x["k"]}'` and PEP 695 type parameters are rejected, and it pins
transformers and a Hugging Face tokenizer), llguidance (no INDENT/DEDENT: one byte may emit several DEDENTs,
guidance-ai/llguidance issue #107), arXiv 2402.17988 (no released library). C++ is not constrained.

Checks (`55e64de`, `ba01bb5`, `d0a2ec0`): 0 false rejections over 11.4M token steps of 3,000 Python files (the
standard library, site-packages, this repository; BPE) and 9.2M randomly segmented; no prefix that `ast.parse` accepts
without the grammar's end; a d16 checkpoint's unconstrained completions (EvalPlus and SPM infilling, greedy and
sampled) replayed through their constraints: every masked token invalid, outputs token-identical wherever nothing was
masked; all 8,488 infilling reference middles (BPE and two random segmentations) accepted; `viable_end` agrees with
`allowed` on 8,002,126 prefixes. Tables built from another vocabulary masked the true token in 1,475 of 3,199 steps,
hence the caller's tokenizer. Mask time per step 0.7–2.6 ms (3.4–15.8 ms before batching into pinned rows).

**Post-generation checks** (`eval/verify.py`, the filtered mode's acceptance test):

- Python (`python_diagnostics`): the program parses (`ast.parse`; parser overflow and the parser's ValueError of §10
  failures), then basedpyright 1.40.2 (one process per batch) with every rule off but `reportUndefinedVariable`,
  `reportAttributeAccessIssue`, `reportCallIssue`, `reportReturnType` (definite runtime failures or annotation
  contradictions; style, unknown-type, possibly-unbound and optional-member rules are off because a correct program
  may trigger them, argument types because pyright also judges them against its own inferred types and rejects
  HumanEval/130's canonical solution). A program basedpyright cannot check (an exception inside its checker, e.g. a
  d16 completion's chained assignments overflowing its stack) fails with that error.
  EvalPlus solutions (`solution_diagnostics`): the generated text's syntax error, an entry function missing or without
  a `return <value>` in its own body (sanitize keeps a parseable span and would hide both).
- Examples (`examples_pass`): the prompt's doctest `>>>` examples and Python `assert` lines, run by value (equal to
  the want, or None/True for a bare expression) in a fresh interpreter under EvalPlus's `reliability_guard`, 3 s
  (HumanEval's timeout; Codex §3.5's 3 s is its APPS timeout); true without examples. By value rather than doctest's
  output text because HumanEval writes some wants in an equal but differently printed form (HumanEval/65, /113) and
  some examples as bare comparisons without output (/108, /145); MBPP+ asserts compare tuples as lists.
- C++ (`cpp_diagnostics`): `g++ -std=c++17 -fsyntax-only`, the MultiPL-E judge's compiler and standard, 15 s; C++
  prompts carry no examples here.

`search` decodes every task without an accepted candidate yet, one round per attempt, at most 10; `chosen` returns
the accepted candidate, else the greedy one. FIM candidates are checked as prefix + middle + suffix against the
prefix's examples.

On the reference solutions (CPU, `8be5e90`) the acceptance test rejects 5 of 164 HumanEval+ and 2 of 378 MBPP+:
`reportReturnType` on HumanEval/10 and /17 (loops that return inside, then an implicit fall-through) and /20
(`tuple | None` against the annotation), `reportCallIssue` on Mbpp/130 (`max(d, key=d.get)` on a `defaultdict`); the
prompts' own wrong examples on HumanEval/47 (median 15.0) and /116; Mbpp/630, whose helper EvalPlus `sanitize` drops
(EvalPlus's scoring fails it too). A correct candidate rejected there is resampled and may end as the greedy one, which
biases these problems' filtered pass@1 down.

## 8. Export, vLLM and the standard-protocol cross-check

**Export** (`python -m inference.export_hf <checkpoint.pt> <dir> [<tokenizer dir>]`, nanochat image) writes a Hugging
Face directory:

- `model.safetensors`: matrices and tables bf16; `_keep_in_fp32_modules_strict` (the per-dim and per-layer vectors,
  the value-embedding gates) fp32;
- `config.json`: `model_type` "nanochat", `auto_map` to `modeling_nanochat.py` (copied beside it); the per-layer
  windows as Gemma 3's `layer_types` + `sliding_window` = 513 (FA4's `window_left` + 1: 512 keys before the query
  and its own); `value_embedding_layers`; `final_logit_softcapping`; `rms_norm_eps` = FLT_EPSILON;
- `generation_config.json`: BOS ends a generation (a document's and a FIM middle's end in training);
- `tokenizer.json`: transformers' `TikTokenConverter` of the tiktoken ranks and split pattern, special tokens in
  `SPECIAL_TOKENS` order (the same ids, `docs/corpus.md` §5), BOS prepended.

**Model file** (`inference/modeling_nanochat.py`, torch and transformers only): the model for transformers
(`AutoModelForCausalLM.from_pretrained(<dir>, trust_remote_code=True)`) and for vLLM's Transformers backend (`vllm
serve <dir> --trust-remote-code`, `LLM(<dir>, trust_remote_code=True)`: paged attention with per-layer sliding windows,
CUDA graphs, torch.compile, `_keep_in_fp32_modules_strict` honored). Layout as transformers' Qwen3; no architectural
change was needed. The model's own pieces are PyTorch expressions at the training kernels' precision: q/k RMS norm,
fp32 gain and rotation by −θ with fp64-computed fp32 tables (transformers' and vLLM's rotary embeddings rotate by +θ,
so neither is used), the value embeddings (they read `input_ids` in every layer), the residual mix in fp32
(`docs/architecture.md` §1, §4), and the head as the evaluation computes it (fp32 output, the cap in fp32; vLLM via
`config.json`'s `head_dtype`); the MLP and the residual additions stay bf16. vLLM runs in its own image
(`vllm/vllm-openai:v0.32.0` at a pinned digest, `scripts/standard_eval.sh`), which the NGC-based nanochat image cannot
hold.

**Agreement with the training forward** (a d16 checkpoint at 5 tokens per parameter, step 1,707; 16 val rows of 8K
from 8 corpora, 130,911 scored tokens; greedy continuations of the 542 HumanEval+/MBPP+ prompts; GH200, `2d49bb1`
and the head variant): the bf16 head was the main difference. With the fp32 head the export's mean KL(train ‖
export) under transformers is 1.97·10⁻⁴ (bf16 head 3.09·10⁻⁴), top-1 agreement 0.9940 (0.9903), the bpb bias
5·10⁻⁶ (6.3·10⁻⁵, its interval excluding 0); under vLLM top-1 0.9941 (0.9904) and 379 of 542 greedy continuations
identical to the training forward's (306). For scale: the uncompiled training forward against the compiled one, KL
3.06·10⁻⁴, top-1 0.9926, 340 identical; `NativeDecoder` (§6), 216 identical. ReLU² and x0 in fp32 changed KL by −1%
and −0.6%, inside their intervals; fp32 residual additions by −14%, with no gain in greedy agreement or pass@1 under
vLLM and 7–10% slower transformers prefill and decode. EvalPlus pass@1 of the export's greedy continuations stays
within one standard error (HumanEval+ 0.0488/0.0488 for every variant; MBPP+ 0.0265/0.0238 → 0.0238/0.0212).

**Agreement of the removed class** (a d16 checkpoint served by the earlier native vLLM class, commits
`3c03023`–`ec1c841`, the bf16 head; its logit agreement and speed: `docs/experiments.md` §6): `tokenizer.json`
encodes 108,026 held-out texts exactly as tiktoken; greedy HumanEval+ 0.067/0.067, MBPP+ 0.056/0.045 (base/plus),
FIM single-line line exact match PSM 0.022, SPM 0.368, against `NativeDecoder`'s (before `4385748`) 0.061/0.061,
0.045/0.037, 0.022, 0.367 (standard error of HumanEval+ ≈ 0.019). The Transformers backend that replaced the class
(`7794e2f`) ran at its rate and, in the same execution mode, gave the same logits (eager: mean KL 4.25·10⁻⁴ on
HumanEval+ for both) [unverified: no record or commit].

**Standard-protocol cross-check** (`scripts/standard_eval.sh <exported dir> <out dir> [generate|score]`): the
harnesses' own CLIs, unmodified, on the export served by vLLM: `evalplus.codegen --backend vllm --greedy` on
HumanEval+ and MBPP+ (direct completion) scored by `evalplus.evaluate`, and MultiPL-E's `automodel_vllm.py` on
humaneval-cpp and mbpp-cpp (T = 0.2, top-p 0.95, 20 completions) judged by `evaluation/src/main.py` and `pass_k.py`,
both at the §4 pins. `generate` runs on the GPU in vLLM's image into an empty output directory (both harnesses skip
cached samples); `score` is CPU work in the nanochat image. The protocols as published differ from §4 in prompt
tokenization (text through `tokenizer.json`), MultiPL-E's budget (`max_model_len` 2,048 and 1,024 new tokens,
against 1,024 positions in all here) and their samplers. On the d16 checkpoint above (commits as above), EvalPlus's
CLI gave HumanEval+ 0.067/0.067 and MBPP+ 0.053/0.042 against this suite's greedy 0.061/0.061 and 0.045/0.037.

## 9. Validation of the harness

- **Metrics against their sources** (`c8e97a2`, `5d733a6`, `ee7b645`): edit similarity by difflib as cceval runs it
  (rapidfuzz, thefuzz and torchmetrics disagree with it on 46–71% of real strings); bpb bucket sums kept as code
  (torchmetrics `SumMetric` differs); `measures` identical to the previous code on four real d8 runs; MultiPL-E
  scoring of six real tasks identical after the CSV parse moved to `csv.DictReader`. A d16 checkpoint's FIM
  single-line SPM greedy line exact match 0.367 ± 0.020, pass@1 0.436.
- **TokEval dependency**: `tree-sitter-language-pack` 1.22.1, as TokEval's reference numbers were computed; 0.13.0
  changes 35 of 135 values [unverified: no record or commit].
- **MultiPL-E judging** (`9d61a1b`): 64 concurrent `main.py` processes × 63 threads oversubscribed g++, and a compile
  past its wall-clock limit is scored "SyntaxError"; with `--max-workers 1` per file, HumanEval C++ sampled "OK"
  completions 70 → 175 of 3,220 and pass@1 0.022 → 0.054 (the other C++ benchmark 0.039 → 0.090 [inference:
  MBPP C++]). FIM scoring through `check_correctness` in a fork pool: 211 → 33 s, 0 changed verdicts.
- **EvalPlus scoring** (`b8057ff`): the evaluator's 20 s exit stall removed (`os._exit` after writing) and all cores
  used: HumanEval+ 22.5 → 9.6 s, MBPP+ 28.0 → 12.6 s; identical verdicts per task against the previous code on a d16
  checkpoint's completions.
- **Reference model** [unverified: no record or commit]: deepseek-coder-1.3b-base through DeepSeek's harness gave
  HumanEval 0.305 (paper 0.348, arXiv 2401.14196 Table 3; transformers 5 drops its BOS) and MBPP 0.466 (paper 0.462);
  this suite's EvalPlus scorer agreed on 163 of 164 problems.
- **d16 reference** [unverified: no record or commit]: at 20 tokens per parameter, code bpb 0.439, HumanEval+ 0.061,
  MBPP+ 0.071/0.050.

## 10. Implementation reference

Per module and symbol; what §1–§9 state is not repeated.

### `metrics.py`

- **Run log** `RUNS/<run_name>/metrics.jsonl` (`append`, `records`): UTF-8 JSON lines, each an object `run_name`,
  `attempt` (the 0-based process start; a resumed run repeats the steps after its checkpoint, the latest attempt's
  record of a step supersedes earlier ones, records are never rewritten), `kind`, `time` (Unix seconds), then the
  kind's fields; NaN and infinities are never written (`allow_nan=False`). Kinds and fields:

| Kind | Fields |
|---|---|
| `run` (each process start) | `step` (first step it trains), `plan` (`schedule.RunPlan`), `seed` (initialization and data order), `config` (`run.py` `RunConfig`), `data_key` (`Tokenizer.data_key`; with `plan`, what the stale guard compares), `model` (`ModelConfig` at the stage of the starting weights: file-level, or the checkpoint's), `num_parameters`, `flops_per_token` (`non_embedding_flops_per_token`, M of arXiv 2401.02954 §3.2 Eq. 2), `model_flops_per_token` (at that stage), `max_grad_norm`, `learning_rate` and `weight_decay` (`{"adamw": [per group], "muon": [per group]}` as the optimizers hold them; a resumed process's from its checkpoint), `stage` (`corpus/mixture.py` `Stage` as run, mixture as {corpus: row weight}) |
| `stage` (switch to the repo-level stage) | `step`, `stage`, `model_flops_per_token` (repo-level), `peak_memory_bytes`, `peak_reserved_bytes` (`torch.cuda.max_memory_allocated`/`max_memory_reserved` over the file-level stage; both reset at the switch) |
| `train` (each optimizer step, 0-based) | `step`; `num_input_tokens_seen` (row positions after it, padding included); `loss` (mean next-token loss in nats over valid labels, fp64); `learning_rate` (LambdaLR's); `grad_norm` (global L2 before clipping, arXiv 1211.5063 §3.2); `step_seconds` (device time between CUDA events at the previous step's end and its own; `train/loop.py` reads a step's values one step late); `tokens_per_second` (row positions / `step_seconds`); `model_flops` (`executed_model_flops` of the step's micro-batches); `mfu`; `train/diagnostics.py`'s `max_logsumexp`, `mean_square_logsumexp` (logits' logsumexp over valid positions: maximum and mean square, PaLM's z-loss quantity, arXiv 2204.02311 §5), `qk_norm_gain_max`, `corpus_loss` ({corpus: mean loss of its rows}); every 10th step `update_rms` ({optimizer[group]: update RMS without decoupled weight decay}) and `param_norm` ({optimizer[group]: L2 norm after the step}) |
| `end` (last step trained) | `step`, `peak_memory_bytes`, `peak_reserved_bytes` over the last stage (since the switch or the process start) |
| `failure` (non-finite loss or gradient norm; the run stops) | `step`, `loss`, `grad_norm` (null) |

- `train_steps`: attempts only grow down the file, so the last record per step wins. `curves`: columns in first-seen
  field order; none for a run diverged at step 0.
- `mixture_bpb_summary`: corpora carry `"row_bits"`, `"row_bytes"` (`eval/bpb.py` `val_bits`' per sampled row) and
  `"rows"`. `offset_bpb`: corpora carry `"offset_bits"`, `"offset_bytes"`, `"rows"`; bucket i holds the targets after
  [OFFSETS[i−1], OFFSETS[i]) bytes of their segment.
- `bpb_summary`: statsmodels `WLS(...).fit(cov_type="HC1", use_t=False)`, whose estimate is the ratio of sums; HC1's
  √(n/(n−1)) plays Eq. 1's n − 1. `summary`: `OLS` with `use_t=False`; clustered fit `cov_type="cluster"`,
  `use_correction=False`; returns `n` and the number of clusters.
- `parse_outcome`: warnings suppressed during `compile`; MemoryError and RecursionError count as "syntax", as
  `syntax_constraint.syntax_error` counts them, and so does the ValueError CPython 3.12.3's compiler raises in place of
  parsing an f-string debug expression in a format spec, `f"{x:{y=}}"` (python/cpython#121130, fixed in 3.12.5): the
  image's interpreter cannot run such a program. The classification source is the post's section "How the failing
  generations fail".
- CrossCodeEval patterns: `COMMENT` (cceval `eval_utils.py` `remove_comments`: `#` and `//` to the line end, strings
  included), `STRING` (`string_pattern`, removed before identifiers are read), `IDENTIFIER` (`IDENTIFIER_REGEX`, matched
  at the start of NLTK `RegexpTokenizer(r"\w+")`'s words); `code_lines` is cceval `process_examples`, `identifiers`
  `extract_identifiers` (repeats kept), the scores `scripts/eval_metric.py` `process_examples` and `compute_metric_stmt`
  and `eval_utils.py` `cal_edit_sim`. `line_exact_match` is bigcode-evaluation-harness `santacoder_fim.py`
  `process_results` (`g.strip() == reference.strip()`; DeepSeek-Coder §4.2, Table 6); CrossCodeEval's scores:
  DeepSeek-Coder §4.3, Table 7. `edit_similarity`: cceval's requirements install fuzzywuzzy without python-Levenshtein,
  so `fuzz.ratio` is difflib's `SequenceMatcher` (autojunk on); 100 when equal, 0 when either is empty, rounding by
  `utils.intr`.
- `model_flops_per_token`: embedding tables (lookups) and norm gains (elementwise) are not matmul weights; windows are
  `model/modeling.py` `layer_windows`. Nominal T without the causal halving (pytorch/torchtitan
  `quadratic_attention_flops_per_token`); executed counting per segment (`cu_seqlens`: a document, a FIM example, a
  row's padding) as NVIDIA/Megatron-LM `num_floating_point_operations`. `executed_model_flops` groups equal windows
  (`Counter`). `PEAK_FLOPS`: NVIDIA H100 datasheet, dense BF16 without sparsity.
- `latency`: tpot = (end − first)/(new tokens − 1); p99 is serve.py's default `--metric-percentiles 99`; `requests`
  counts the requests; ttft is measured from the request taking a decoder slot.
- `parabola_vertex`: the optimum of a sweep (`ablations/peak_lr.py`).
- TokEval (`token_offsets`, `tokeval_wrapper`, `tokeval`): tokens' bytes are the text's UTF-8 bytes in order; a byte's
  character is the count of lead and ASCII bytes up to it, less one. The wrapper's `encode` is tiktoken
  `encode_ordinary` (special tokens' text encodes as ordinary text, as `corpus/tokenize.py` does), `decode` joins token
  bytes (special ids skipped by default), `pretokenize` is the split pattern, `get_unk_token_id` None (byte-level),
  `from_config({"path"})` loads `corpus/tokenizer.py` `get_tokenizer`. Vocabulary keys: transformers' `bytes_to_unicode`
  of each mergeable token's bytes (HuggingFace's ByteLevel form of tiktoken vocabularies), a bijection on byte strings,
  so keys are distinct and keep partial characters; special tokens (ids after the mergeable ones, in order) keep their
  text. TokEval relies on this form: `utf8_token_integrity` (`metrics/utf8_integrity.py` `_detect_gpt2_encoding`)
  recognizes the marker keys and maps them back to bytes; `indentation_consistency` (`metrics/code_ast.py`
  `_is_whitespace_token`) and `BaseMetrics._build_char_decode_table` read space, LF, tab, CR as Ġ, Ċ, ĉ, č. `tokeval`
  inputs: `texts` names must differ from the code languages and "math" (the domain keys of `metrics/basic.py`); `code`
  goes through `code_ast_config` and `math` through `math_data_path` (the digit metrics and operator isolation's math
  domain), written to a temporary directory; tree-sitter parsers run in subprocesses; non-finite floats become null
  (`cli/run_analysis.py` `_convert_for_json_public`).

### `eval/suite.py`

- `evaluate`: record `{"run_name", "step", "8k", "32k", "solution", "parse", "codegen", "fim"}`; NaN and infinities
  raise (`allow_nan=False`). Timed checks: EvalPlus's time limits, `verify.py`'s and the evaluators' timeouts.
  `resolved` replaces each scoring future by its result in key order.
- `record`: `{"run_name", "records" (non-train kinds: run, stage, end, failure), "evaluations"}`, one per run in
  `RESULTS/<experiment>.jsonl` (`ablations/records.py`).

### `eval/bpb.py`

- `sums(losses, batches, token_bytes, num_segments=)` → `{"bits", "bytes"}` per context bucket (bucket i:
  contexts in (CONTEXTS[i−1], CONTEXTS[i]]; a target's context is its input's position in its segment + 1),
  `{"offset_bits", "offset_bytes"}` per offset bucket and `{"row_bits", "row_bytes"}` per row, over every batch;
  batches are `corpus/loader.py` `PretrainingBatches` tuples (a loader ends with its sample's last micro-batch).
  Padding `cu_seqlens` by its last offset leaves empty segments the attention never reads; `searchsorted(right=True)`
  maps a position to its segment. OFFSETS reach 128 KiB, beyond a 32K row's span in bytes.
- `val_bits` → `{"corpora": {corpus: {"bits", "bytes", "offset_bits", "offset_bytes", "row_bits", "row_bytes",
  "rows"}}, "bpb": {group | "all"}, "errors": {group}, "offset_bpb": {group}, "weights": {corpus}}`; weights are
  `Stage.shares` at the total train packed rows; the last micro-batch is filled with empty rows (index −1), dropped
  from the row sums.
- `solution_bits` → `{benchmark: sums | per-solution "row_bits", "row_bytes" in benchmark order}`, in
  `FILE_LEVEL_STAGE` micro-batches, one example and one attention segment per row. The prompt ends after the
  solution's leading blank lines (through the last leading line break); the first scored token is found by
  `decode_with_offsets`. `likelihood`'s benchmarks: `code_eval.tasks()`.

### `eval/code_eval.py`

- Upstream lines followed: EvalPlus base-model codegen `prompt.strip() + "\n"` (`evalplus/codegen.py:71`),
  `max_new_tokens` 768 (`provider/base.py:13`), stops EOS + `extra_eos_for_direct_completion`
  (`provider/vllm.py:41-42`), `skip_special_tokens` (`provider/hf.py:96`), cut at the first stop, tabs → four spaces
  (`provider/hf.py:104-105`, `vllm.py:76`), `sanitize(prompt + completion, entry point)` (`codegen.py:79-82`). MultiPL-E
  (`3025a53`): dataset revision `multipl_e/completions.py` `DATASET_REVISION`; `automodel.py` prompt `.strip()`
  (`:119`), 1,024-position `max_length` (`completions.py --max-tokens`; `:67`, prompt truncated at `:56`, output past
  the full prompt at `:114`), first special token ends it (`:99-101`), `completions.py:216` `stop_at_stop_token`; judged
  program `prompt + completion + "\n" + tests` (`main.py:34`), g++ -std=c++17 then the binary, 15 s each
  (`eval_cpp.py:9-36`, `safe_subprocess/__init__.py:33`); `pass_k.py` counts status "OK" with exit code 0. FIM orders:
  the two the packer trains (arXiv 2207.14255 App. D). MultiPL-E's T = 0.2 and `completions.py --top-p 0.95`.
- `Task`: `canonical_solution` the reference, `stopwords` the stop sequences, `entry_point` EvalPlus's entry point or
  MultiPL-E's tests. `CodegenRun`: `special` the completion's treatment of special tokens, `check` filtered_completions'
  check, `score` the evaluator. `FimRun.prompts`: per FIM order.
- `KeywordsStoppingCriteria`: incremental UTF-8 decoding; search restarts where a stop sequence could still begin; the
  healed bytes (`decoder.Reemit`, possibly over several tokens) are skipped. `completion`: `decode_bytes` then UTF-8
  with replacement, as `enc.decode`. `fim_lines`: a non-empty middle written up to BOS or the budget gets a final "\n";
  an empty middle stays empty.
- `decoder`: `max_length` = len(input_ids) + max_new_tokens − 1; created after `torch.cuda.empty_cache()`, since slots
  and prompt store are sized by free memory (for the filtered passes, which run after the likelihood pass, this returns
  that pass's blocks); `NANOCHAT_DECODER_MEMORY_FRACTION` unset = all free memory, so a job sharing the GPU keeps the
  rest. Prompt stores: codegen, each prompt shared by its greedy request and samples; filtered, the healed prompts a
  search's rounds repeat (the healed prompt a token shorter, its budget a token longer); FIM greedy none (no repeats);
  FIM filtered the prompts.
- `generate_groups`: `done(g, ids in request order, latency)`, latency's duration from the group's first start to its
  last end with other groups interleaved; groups finish as soon as their last request ends.
- `filtered`: record `{"attempts", "accepted", "latency"}` per task; unaccepted tasks score the greedy candidate.
  `filtered_completions`: healed candidates get one more token of budget, the dropped token's bytes re-emitted by
  `Reemit`, or for Python by the `SyntaxConstraint` built with `healing=`; a fresh constraint per candidate.
- `sanitizers`: workers import the main module (`run.py`, whose top level must stay light); in a spawned process,
  multiprocessing's exit handler joins the workers before concurrent.futures' would stop them, hence the caller's
  `with`. `sanitizer_init` makes EvalPlus's `syntax_check` (which catches SyntaxError and MemoryError) count that
  ValueError as a syntax error, so `code_extract` skips such spans: uncaught, one d16 completion ended a benchmark
  evaluation (`8be5e90`).
- `EVALPLUS_EVALUATE`: `evaluate(..., parallel=os.cpu_count())` (default half), `os._exit(0)` after the output file:
  `stucking_checker`, a non-daemon thread sleeping 20 s per check. `evalplus` → `{"base pass@1", "plus pass@1"
  (evaluator's `estimate_pass_at_k`), "scores": {"base", "plus"}, "ids", "base_status", "plus_status"}`; plus passes
  when base_status == plus_status == "pass".
- `multipl_e` → `{"pass@1", "scores", "ids", "status"}`: a completions file per problem as `completions.py` writes it,
  `os.cpu_count()` judges at once with `--max-workers 1` (main.py's default `cpu_count() − 1`; a compile past its
  wall-clock limit scores "SyntaxError", `eval_cpp.py`). `pass_k.py`'s CSV
  (`Dataset,Pass@k,Estimate,NumProblems,MinCompletions,MaxCompletions`) yields no row when a directory mixes
  temperatures. `multipl_e_check`'s program ends at the function's closing "}" (the stop token "\n}" cut).
- `INFILLING_EVALUATE`: `evaluation.py` `evaluate_functional_correctness`'s `check_correctness` at its default 3 s per
  sample; `reliability_guard` disables what a run-time import would call, hence the CLI modules imported first.
  `infilling` → `{"pass@1", "scores", "ids", "completions", "result"}` (per problem "passed", "failed: <exception>" or
  "timed out", `execution.py`); clusters from task id `<Benchmark>/HumanEval/<n>/<span>`.
- Records: `generation` `{benchmark: {"greedy": scores | "completions", "latency"; "sampled": scores | "n_sample",
  "latency"}}`; `parse_generation` `{benchmark: {"parses", "indentation", "syntax", "outcomes", "completions",
  "latency"} | evalplus}`; `fim_generation` `{"<benchmark>/<mode>": {"greedy", "greedy_reference_lines"}}`;
  `FimRun.lines`: single-line 1, multi-line each reference's line count. Every evaluator is a fresh process (`exec`),
  never a fork of the torch/CUDA process.

### `eval/verify.py`

- `PYRIGHT_RULES` from basedpyright's configuration reference (`config-files`); everything else off, argument types
  (`reportArgumentType`) included. `python_diagnostics`: "line:col: SyntaxError: message" for unparsed programs
  ("MemoryError: …", "RecursionError: …" or "ValueError: …", as `syntax_error`); each parsed program written as
  `program.py` in its own directory; one process (`--pythonpath sys.executable`, so the installed packages resolve)
  because its start dominates a check; exit status 1 on errors, the JSON report on stdout either way. An exception in
  one file's check aborts the whole run (pyright.js `sourceFile.check` rethrows after its internal-error diagnostic,
  `analyzeProgram` exits 2 without a report): the file named on stderr gets "1:1: basedpyright: internal error: …"
  and the rest are checked again. Measured case: one program of a d16 checkpoint's 346-program filtered MBPP+ round
  (`RangeError: Maximum call stack size exceeded`); the other 345 programs' results equal those of a run without it.
- `solution_diagnostics`: sanitize reduces source (prompt + completion) to the entry function and its dependencies and
  keeps a parseable span; the entry function is its last top-level definition, nested functions, lambdas and classes
  excluded from the `return` search; a solution that does not parse returns its syntax error.
- `cpp_diagnostics`: error lines ("program.cpp:line:col: error: …", "… fatal error: …"), else the last stderr line.
- `examples`: HumanEval doctest examples; EvalPlus MBPP+ prompts' asserts (one per docstring; HumanEval/149's
  `assert f(x) => y` prose is not Python). Literals are those `tokenize` yields before it stops (TokenError at an open
  bracket or string, IndentationError at a partial dedent); a literal doctest rejects (`doctest.py` `_check_prefix`)
  contributes no `>>>` examples, as `python -m doctest` would run none.
- `RUN_EXAMPLES`: reads stdin JSON (program, source/want pairs), applies `reliability_guard` with EvalPlus's memory
  limit (destructive os, shutil, subprocess functions disabled), runs the program as module "program", each example in
  a copy of its namespace plus modules it names but the program does not bind (MBPP+ asserts call `math.isclose`,
  imported by MBPP's `test_imports`); a failing example exits 1.
- `verdicts`: Python diagnostics from one `python_diagnostics` call; C++ diagnostics and examples from a thread pool,
  each a child process. `search`: per round, attempt a uses temperature 0 for a = 0, else TEMPERATURE with seed a; each
  candidate's check is built on a thread as it completes while the generator runs the round's others; latency's
  duration sums the rounds' decoding.

### `scripts/standard_eval.sh`

- vLLM image `vllm/vllm-openai:v0.32.0` at a pinned digest with its own torch; the export's `modeling_nanochat.py`
  loads with `trust_remote_code`. EvalPlus as `pyproject.toml` pins it, fetched as a GitHub archive (the image has no
  git; `SETUPTOOLS_SCM_PRETEND_VERSION`), into a venv over the image's vLLM; MultiPL-E at `docker/Dockerfile`'s commit.
  Direct completion (the tokenizer has no chat template). Outputs: `evalplus/<dataset>/…_eval_results.json`,
  `multipl-e/<dataset>/`, `pass_k.csv`; `*.raw.jsonl` (codegen's unsanitized copy) skipped in scoring. Skipped cached
  samples: evalplus codegen's "already cached", MultiPL-E's `completion_limit`; the model is always `/model`. Caches
  (Hugging Face datasets, vLLM's compiled graphs) in `artifacts/cache`; the model and output mount at `/model`, `/out`.

### `inference/syntax_constraint.py`

- **Layers** (crate `inference/syntax/`, compiled on first use by `native.crate_module`):
  - lexer (`lexer.rs`): CPython 3.12.3's `Parser/tokenizer.c` (`tok_get_normal_mode`, `tok_get_fstring_mode`,
    `verify_end_of_number`) as a character-level machine whose states are values, so a prefix may end mid-token,
    mid-indent or mid-UTF-8 character. Input is the text as `compile()` sees it: `\r\n` and `\r` read as `\n`
    (`translate_newlines`), a final `\n` implied at the end.
  - indentation (`parser.rs`): `tokenizer.c`'s INDENT/DEDENT (tab size 8; the tab-consistency check at tab size 1; a
    backslash line's first column) and its bracket level (newlines inside brackets are not NEWLINE tokens).
  - parser (`parser.rs`): `python.lark`, LALR(1), accepting every token sequence of CPython's `Grammar/python.gram`
    (3.12); `match`/`case`/`type` read as a name and as a keyword on two parser stacks until one dies.
- **Over-approximations** (a prefix allowed that CPython rejects; never the reverse): assignment, deletion and `with`
  targets are expressions; parameter and argument orders are free; `(a as b)` parses anywhere (the `with` statement's
  parenthesized items); patterns accept f-strings and any dotted name as a mapping key; escapes, the f-string conversion
  character, a replacement field's unparenthesized lambda and the nesting limits are not checked; `<>` is `!=` (PEP 401
  `barry_as_FLUFL`, `_PyToken_TwoChars` of CPython's `Parser/token.c`, called by `tokenizer.c`); a partial token at the
  end is judged by the lexemes it can begin with.
- **BOS**: the crate proposes BOS when the lexer and parser end a module (always, before a suffix); a chosen BOS stands
  only if the grammar ends a module after the suffix too and `ast.parse` accepts text + suffix (`ends`); otherwise the
  decoder takes the step's best other token (`NativeDecoder.generate`). The grammar's end implies valid UTF-8 (the lexer
  rejects any other), so `ends` decodes safely. `masks` proposes no BOS while a healing `Reemit` owes bytes.
- **Masks** (`vocab.rs`): per lexer state, every token's events (tokens emitted, newlines, line-start whitespace) and
  end state computed once over the vocabulary's byte trie; the subtrie after each first byte is shared by the lexer
  states that byte takes to one state; a step runs the parser once per distinct event sequence and maps verdicts back
  to tokens. `masks` writes into the decoder's pinned C-contiguous uint32 mask rows, one row per constraint, in
  parallel without the GIL, sets BOS's bit for a candidate end, then intersects each row with its `Reemit`'s mask
  (BOS excluded) while that owes bytes.
- `python_syntax`: Lark's LALR tables dense (terminals and nonterminals by sorted name; shift = state, reduce = `~rule`,
  `NO_ACTION` = int32 min as in `parser.rs`), lexemes' terminals (keywords, soft keywords as NAME + keyword,
  operators), Python's non-ASCII identifier classes (`str.isidentifier` of c, of `"a" + c`, as inclusive ranges), the
  vocabulary's byte trie (tokens below BOS), and per-lexer-state token groupings filled on first use.
- `syntax_error`: CPython's "Parser stack overflowed" `MemoryError`, `RecursionError` and the ValueError of
  `metrics.parse_outcome` (python/cpython#121130) are failures to parse.

### `inference/decoder.py`

- Flow: a request's prompt but its last token is prefilled (its positions run as rows of one batched pass, with any
  other new requests' prompts); captured CUDA graphs then advance every active slot a token per step (argmax or a
  seeded sample, optionally from its nucleus, within its constraint's allowed tokens) until BOS, the budget or the
  stopping criteria; a freed slot takes the next request. A stage change (context, windows, rotary) needs a new decoder;
  the model's weights must not change while one lives (it reads them in place).
- `fim_prompt`: the layout the packer trains (`corpus/pack.py`, `corpus/loader.py`); `NativeDecoder.generate` stops at
  the BOS ending the middle.
- `Request`: `stopping_criteria` is None or a predicate of the generated ids, true once done; sampled logits are capped
  by `ModelConfig.final_logit_softcapping` if set; the draw at generated position i is a function of (seed, i) only
  (`decode.cu` `gumbel`); nucleus in `decode.cu` `k_nucleus`; `constraint`: the tokens allowed at each step.
- `Reemit`: a healed Python prompt's `SyntaxConstraint` holds it; `syntax_constraint.masks` intersects with its mask.
  Proper prefixes found by bisect over `prefix_index`; the returned mask is the decoder's layout.
- `heal_tokens`: Guidance, "The Art of Prompt Design: Prompt Boundaries and Token Healing" (2023), constrained until the
  dropped token's bytes are re-emitted, possibly over several tokens (TokEval).
- `step_masks`: the syntax constraints' masks by one `syntax_constraint.masks` call, in parallel without the GIL,
  straight into the decoder's pinned rows; a non-eager BOS candidate is decided by the caller once chosen (`confirm`).
  A rejected eager BOS clears its bit; an emptied row counts as allowing none.
- `PAGE` = 64: a stored prompt takes whole pages anywhere in the store; at most 63 positions of a prompt unused.
- `Stored.used`: the run (`NativeDecoder.runs`, one per `generate` call) that last used it.
- `NativeDecoder`:
  - Determinism: decode.cu computes each slot and each prefill row alone in a fixed order; a sample's noise depends on
    the seed and generated position only. The e4m3 KV cache (a scale per position and head) lets tokens differ from
    the bf16 forward's where two logits nearly tie. Tokens do not depend on `max_length`.
  - `batch`: `dec_create`'s slot count, at most `MAX_BATCH`, bounded by free device memory.
  - `max_length`: the longest `len(input_ids) + max_new_tokens - 1` the caller runs. A slot's cache is
    2 · num_hidden_layers · max_length · (hidden_size + 4 · num_attention_heads) bytes.
  - `prompt_store`: at most the memory left after the slots, 2 · num_hidden_layers · (hidden_size + 4 ·
    num_attention_heads) bytes a position. A prompt's K and V at all but its last position are kept once prefilled; a
    later request with the same prompt copies them (K and V depend on the prompt alone, so tokens are unchanged).
  - `memory_fraction`: covers slots, prefill rows, graphs and the prompt store, capped by free memory (as vLLM's
    `gpu_memory_utilization`, counting this decoder alone).
  - Weights: the matrices are the model's bf16 GEMM copies (`CausalLM.refresh_gemm_weights`: q/k/v concatenated per
    layer, o, up, down), read in place, with bf16 copies of `lm_head` and the value-embedding tables; the embedding
    table stays fp32 because the compiled forward's x0 is the fp32 row's norm (`decode.cu` `k_norm`); value-embedding
    table and gate only on VE layers.
  - `dec_create` config: L, C, F (`up_proj` is (F, C)), vocabulary, max_length, prompt store, BOS (the lazy row: a
    syntax constraint's end, confirmed once chosen in `generate`), memory budget in MiB (0: none).
- `NativeDecoder.prefill`: `dec_prefill` takes every position as a row, `prefill_rows` rows a launch, a prompt's
  positions in order; `save` reads the K and V the launched prefill wrote.
- `NativeDecoder.confirm`: `chosen` are the syntax-constrained slots, their tokens in `self.tokens`; eager (nucleus)
  slots' BOS was decided before the step; the alternative equals the choice with BOS masked.
- `NativeDecoder.generate`:
  - yields at BOS (excluded), after `max_new_tokens`, or once `stopping_criteria` holds (that token included);
    prompts that took a slot since the last step are prefilled before the next.
  - a step's stopping criteria run while the next step runs; a stopped slot's extra token is dropped.
  - a step with a sampled or constrained slot launches the body, writes masks meanwhile (`step_masks`; a mask follows
    from the previous token), then the head (choice by mask and temperature); other steps launch the whole greedy step.
  - a constraint allowing no token (not even BOS) raises ValueError.
  - `timings`: read on the host once each step's tokens arrive (no further device synchronization): when the request
    took its slot, when its first generated token arrived (end for a request without one), and the step finishing it.

### `inference/export_hf.py`

- `model.safetensors`: the training state dict's names under `model.` (`lm_head.weight` as is); bf16 as the training
  ops round them; the fp32 set is `modeling_nanochat`'s `_keep_in_fp32_modules_strict`, as it reads them.
- `config.json`: `architectures` NanochatForCausalLM; `model.modeling.layer_windows` as Gemma 3's `layer_types` +
  `sliding_window` (keys attended including the query's own); `model.modeling.has_ve`'s layers; MLP width the
  `up_proj` rows; eps FLT_EPSILON (`model.ops.linear` `EPS`, the norms' epsilon).
- tokenizer: ByteLevel BPE, the split pattern as an Oniguruma Split (its possessive quantifiers are Oniguruma syntax);
  special tokens after the mergeable ones in `corpus.tokenizer.SPECIAL_TOKENS` order (the converter's order, so the
  same ids); BOS prepended by `TemplateProcessing`; `tokenizer_config.json` beside `tokenizer.json`.

### `inference/modeling_nanochat.py`

- Copied into the export directory and named in `config.json`'s `auto_map` (Hugging Face's custom-code convention);
  torch and transformers only. vLLM's Transformers backend (vLLM docs, "Transformers backend") honors
  `_keep_in_fp32_modules_strict`, the per-layer sliding windows (`layer_types`), `final_logit_softcapping` and
  `head_dtype` (`LogitsProcessor._apply_head`: the head's fp32 output, the cap in fp32) with its own attention, KV
  cache, torch.compile and CUDA graphs.
- Attention through `ALL_ATTENTION_FUNCTIONS` with each layer's window (`layer_types` "sliding_attention" /
  "full_attention" and `sliding_window`, as Qwen3 and Gemma 3), so the same code runs in transformers (eager, sdpa,
  `generate` with a cache) and in vLLM.
- Precision, as the training kernels keep fp32:
  - q/k (`model/kernels/qk_row.cuh`): per-head RMS norm (eps FLT_EPSILON), fp32 gain, rotation by −θ over the full
    head, (m1, m2) → (m1 cos + m2 sin, m2 cos − m1 sin), angles t·θ^(−2i/D) in fp64 rounded once to fp32
    (`CausalLM.init_buffers`), one bf16 rounding. transformers' and vLLM's rotary embeddings rotate by +θ.
  - value embeddings (`model/kernels/ve_gate.cu`): v + 2·sigmoid(n Wᵀ) per head × table[input_ids], gate logits fp32,
    the mix rounded once.
  - per layer h = resid_lambda·h + x0_lambda·x0 in fp32, rounded once; the per-dim and per-layer vectors and the gate
    stay fp32 (`_keep_in_fp32_modules_strict`).
  - head: `torch.mm(…, out_dtype=float32)` of the bf16 weight, the cap in fp32, as `model/ops/cross_entropy.py`
    `lm_losses`.
- `NanochatConfig` fields: vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads
  (= num_key_value_heads), head_dim, max_position_embeddings, rope_parameters {"rope_theta"}, sliding_window (keys
  attended including the query's own), layer_types, value_embedding_layers, final_logit_softcapping, rms_norm_eps.
- `Gate`: vLLM's Transformers backend swaps every `nn.Linear` for its own layer, whose weight
  `_keep_in_fp32_modules_strict` does not upcast (its keep_in_fp32 skips tensors with a `weight_loader`); an
  `nn.Linear` gate would be read in bf16 there instead of training's fp32.

### `model/kernels/decode.cu`

- **State and prefill.** Positions, input tokens, KV caches, the active-slot mask and each slot's sampling parameters
  and allowed tokens live in device memory. A slot starts at its prompt's last token (`dec_set_prompt`), the positions
  before it cached: run as rows of a prefill (`dec_prefill`: up to `MAX_BATCH` (slot, position, token) rows of any
  prompts through the step's body at once, each row computed as a step would, the rows' K and V cached before any
  row's attention), or copied from the prompt store (`dec_save_prompt`, `dec_load_prompt`; K and V depend on the
  prompt's tokens alone). Inactive slots are skipped by every kernel.
- **Graphs.** The step is captured whole (`dec_launch`: the greedy argmax for every slot) and split as its body (up to x
  before the final norm; the head starts with the final norm) and a head choosing by each slot's allowed tokens,
  temperature and top_p (`dec_launch_body`, `dec_launch_head`); the host writes the allowed tokens, which follow from
  each slot's previous token, while the body runs. The head also chooses each slot's best allowed token but one lazy row
  (BOS); the host confirms a chosen lazy row only then and otherwise takes that alternative (`dec_take_alt`), the same
  choice as excluding the row beforehand (greedy, or Gumbel-max whose noise is fixed per row). The whole step's GEMMs
  compile without the choice (the GEMM kernels' `CHOICE` = false).
- **The step** (modeling.py's forward at one position; C width, H heads of HEAD_DIM, x0 the normed embedding):

| Kernel | Computes |
|---|---|
| `k_norm` | x0 = norm(wte[token]) (fp32, unrounded) |
| per layer i: `k_norm` | xm = resid_lambdas[i]·x + x0_lambdas[i]·x0 (x = x0 before layer 0), h = norm(xm) |
| `k_wgmma` | [q k v] = h [Wq Wk Wv]ᵀ |
| `k_prologue` | VE layers: v += 2·sigmoid(W_g[head]·h)·value_embeds[i][token] (per head, fp32); q, k = rope(norm_D(q)·q_gain), rope(norm_D(k)·k_gain) (`qk_row.cuh`, as `qk_prologue.cu`); k, v cached e4m3 with a scale per position and head |
| `k_attn` | y = softmax(q Kᵀ / √HEAD_DIM) V over the layer's window (FlashAttention's `window_size` left) |
| `k_wgmma` | xo = xm + y Woᵀ |
| `k_norm` | hn = norm(xo) |
| `k_wgmma` | u = relu(hn Wfcᵀ)² |
| `k_wgmma` | x = xo + u Wprojᵀ |
| `k_norm` | hn = norm(x) |
| `k_wgmma` | the choice from s = hn W_lmᵀ over the slot's allowed tokens (all if none): argmax s at temperature 0, else argmax s/T + Gumbel noise, by atomicMax on a packed key; a nucleus-sampled slot's s stored instead (head only) |
| `k_nucleus` | a nucleus-sampled slot's choice: that argmax over the fewest most likely allowed tokens holding top_p of softmax(s/T) (head only) |
| `k_commit` | the choice (and its alternative) becomes each active slot's next input |

- **Numerics** (model/modeling.py's compiled forward, torch.compile as training and bpb run it, op by op): bf16 copies
  of the fp32 matrices and tables; fp32 rotary table, q/k norm weights and value-embedding gate (no contraction), the
  gate's mix rounded once to bf16; the residual mix as Inductor fuses it (`k_norm` `RESID_NORM`: fp32 lambdas and x0,
  one fma, one bf16 rounding, the norm of the unrounded mix); ReLU² from the fp32 accumulators as QuACK's `gemm_act`
  `relu_sq` (`quack/activation.py`), one bf16 rounding; F.rms_norm's fp32 epsilon; the choice over fp32 logits. KV
  cache: e4m3 (1 byte, a scale per position and head: amax / 448), half the bytes a step reads; scores and softmax
  fp32. A slot's arithmetic does not depend on the other slots (its own GEMM column and attention block, its own cache
  scales), so its tokens do not depend on the batch; a sample's noise depends only on the seed and generated position.
- **Launch shapes.** A GEMM runs a (slot tile, row tile) block of a TMA producer and a wgmma warpgroup (`k_wgmma`);
  attention a (head, slot) block of HEAD_DIM threads streaming the slot's window (`k_attn`).
- Constants: `MAX_BATCH` = 1,024 is both slots per decoder and rows per prefill launch. `SMALL_BATCH` = 256: the
  fastest step, or within 0.04 ms of it, at d24 (256-token prompts) at 1, 64, 128, 256 active slots. mma m16n8k16:
  lanes form MMA_N groups (groupID = lane / GROUP) of GROUP threads (threadID_in_group = lane % GROUP), each holding
  ACC fp32 accumulators. `NORM_WARPS`: slots (a warp each) per block, the fewest letting an SM fill its threads within
  its resident-block limit (CUDA C++ Programming Guide, "Technical Specifications"). `NUCLEUS_THREADS` = 1,024: the
  most a block holds.
- `CUDA_CHECK` covers an allocation lost to another process between `cudaMemGetInfo` and `cudaMalloc`, and an
  asynchronous kernel or copy error surfacing at the step's synchronization.
- `gumbel`: Philox4x32-10 ("Parallel Random Numbers: As Easy as 1, 2, 3", SC '11; cuRAND `curand_Philox4x32_10`), so
  the draw depends on (seed, i, row) alone; u = (2r + 1)·2^−DBL_MANT_DIG of DBL_MANT_DIG − 1 random bits r is exact in
  double and never 0 or 1; Gumbel-max: "A* Sampling" (arXiv 1411.0030 §2).
- `Choice`: `softcap` is `ModelConfig.final_logit_softcapping`; a greedy choice needs no cap because tanh is
  increasing, so the argmax is the uncapped logits' (but for fp32 ties of saturated caps). `alt` is kept for a slot not
  nucleus-sampled and read by the host through `dec_tokens`. `lazy` (BOS) is an end the host checks only once chosen.
  `allow`: bit row % WORD_BITS of word row / WORD_BITS.
- `copy16`: all of a thread's copies are in flight at once. PDL (`launch`, `pdl_start`): the step's kernels launch
  while their predecessor runs; `griddepcontrol.wait` before anything else (early exits included) means a completed
  kernel implies every earlier one completed; `launch_dependents` lets the successor's blocks be scheduled once every
  block of this grid has started.
- `epilogue`: STORE/RESID/RELU2 write the output; ARGMAX: largest score, lowest row among exact ties (`choice_key`), the
  fp32 logit, or with `CHOICE` over a masked slot's allowed rows (at every temperature) and at T > 0 the score z/T +
  gumbel, z the capped logit (c·tanh(s/c) when a cap is set); nucleus-sampled slots' logits go to `k_nucleus`.
  `argmax_fold`: accumulator columns b0 + 2t + p.
- `k_wgmma`: PTX ISA "Asynchronous Warpgroup Level Matrix Multiply-Accumulate"; shared tiles aligned to 1,024 bytes
  (a 128-byte swizzle atom's 8 rows: PTX ISA "Shared Memory Matrix Layout"); TMA is `cp.async.bulk.tensor`, one
  full and one empty mbarrier per stage; m64n64k16 from the swizzled tiles' descriptors, fp32 accumulators; one slice's
  MMAs in flight while the next issue, a stage released once its MMAs complete. Accumulator layout: mma m16n8's per
  warp (16 rows) and n8 chunk (PTX ISA "Register Fragments: wgmma .m64nNk16"), as `epilogue` and `argmax_fold`
  address it. Slot tiles without an active slot exit. `CHOICE`: ARGMAX honours each slot's allowed tokens and
  temperature (the head graph); without it, the logits' argmax (the whole-step graph). Tile 64 × 64 and `WG_STAGES` = 4:
  the fastest step at d24 (1,024 slots, 256-token prompts) (§6, tuned constants).
- `k_nucleus`: arXiv 1904.09751 ("The Curious Case of Neural Text Degeneration") §3.1. Hugging Face transformers'
  `TopPLogitsWarper` sorts probabilities ascending and drops every token whose cumulative probability is at most
  1 − top_p; here e = exp(s/T − max s/T) (0 outside the allowed tokens), and since C is a step function rising at each
  e, the least bits found by bisection are an e's. The choice's key goes to argmax for `k_commit`; the sums' fixed
  order makes the choice depend on the slot alone. A block per slot; inactive, greedy and top_p 1 slots exit.
- `k_norm`: sum of squares over K in lane-strided order, then `warp_sum`. NORM: v the slot's row of `in`,
  out = bf16(v·r). EMBED: v the row `tokens[b]` of the fp32 table `wte`; x0 = v·r stays fp32 and unrounded because
  Inductor drops modeling.py's bf16 cast of the embedding and recomputes x0 in fp32 for every layer's mix. RESID_NORM:
  v = fma(l1, h, x0·l2) with the fp32 resid and x0 lambdas l1, l2 (h the slot's row of `in`; x0 at layer 0: `in`
  NULL), as the compiled forward's fused mix-and-norm kernel (Triton contracts l1·h + l2·x0 to this fma); out =
  bf16(v·r) of the unrounded v.
- Attention (`k_prologue`, `k_attn`): each `k_attn` block streams its row's window begin..p−1 through shared memory in
  `ATTN_RUN`-key runs (cp.async, double-buffered). Each warp scores `MMA_M` keys of a run by f16 MMAs (fp32
  accumulators), updates its online softmax (m, l) once per run and adds the scaled values on the CUDA cores, a lane's
  `LANE_DIMS` dims; the warps merge in warp order, add the current key and value, and write acc / l rounded to bf16
  (FlashAttention's output). One block per window: at d24 a 1,024-slot step's 6,144 (head, slot) blocks fill the SMs.
  `ATTN_RUN` (one MMA_M tile per warp, the least a warp's MMA scores) gave the fastest step at d24 (1,024 slots,
  256-token prompts). `Attn.cache_slot`: a prefill row's slot, whose cache row b reads and writes (NULL: slot b).
- `Decoder`: `capacity` is the decoder's slots (B but in a narrow graph's view); `max_len` the caller's longest
  sequence; the narrow graphs run while no slot at index ≥ `SMALL_BATCH` is active (`dec_set_active`). The prompt store
  keeps prompts' K and V for reuse (`dec_save_prompt`, `dec_load_prompt`).
- `dec_create` ABI (`decoder.py` `NativeDecoder.__init__`): `cfg` = L, C, MLP width F (fc [F][C], proj [C][F]), vocab
  (lm_head's rows), max_len (a slot's cache positions, at most the context), store (the prompt store's positions
  requested: at most the device memory left after the slots and graphs), lazy (the row whose choice the host confirms
  after a step, `Choice::lazy`), budget (MiB the decoder takes at most: slots, prefill rows, graphs and prompt store;
  0 all free memory). `ptrs`: `GLOBAL_PTRS` = 4 (wte fp32, cos, sin: the fp32 rotary table [context][HALF] of every
  layer, lm_head), then `LAYER_PTRS` = 8 per layer (qkv, o, fc, proj, value-embedding table, gate (0 if none), q norm
  weight, k norm weight). `scal`: per layer resid lambda, x0 lambda (fp32), then the output-logit cap
  (`ModelConfig.final_logit_softcapping`, 0 none). `window`: per layer, the keys before each query that it attends
  (FlashAttention's `window_size` left).
- Graph pairs (`dec_create`): the narrow set exists because the blocks of inactive slots exit at once, but a mostly
  idle batch of MAX_BATCH slots still pays their launch. The prompt store's 2 MiB step is the device allocation
  granularity (`cuMemGetAllocationGranularity` on Hopper).
- `dec_set_prompt`: from that step on the slot chooses at `temperature` (0: argmax) with noise from `seed` (`gumbel`),
  from the nucleus holding top_p when top_p < 1 (`k_nucleus`). `dec_prefill`: rows[0, n) tokens, rows[n, 2n)
  positions, rows[2n, 3n) slots; every row computed alone in the step's order; rows' K and V cached before any row's
  attention; prefilling a prompt's positions before its last token lets its slot start at that token.
- `prompt_copy`: per cache one 2D copy of L rows (the store's rows `store` positions apart, the slot's `max_len`), per
  scale array one of L·H rows. `dec_load_prompt` is the reverse of `dec_save_prompt`.
- `dec_masks`/`dec_set_active`: the caller writes a masked slot's row before `dec_set_active`, after the previous step
  completed; `dec_set_active` sets `narrow` from the highest active slot.
- Host API (all `extern "C"`; stream-ordered unless stated): `dec_batch` the slots (`dec_create`); `dec_prefill_rows`
  the rows a `dec_prefill` call takes at most; `dec_store` the store's positions (the requested, at most what fit after
  the slots); `dec_set_prompt` sets the first choice at generated position 0; `dec_load_prompt` fills a slot's cache
  from the store; `dec_launch` one greedy step (every slot takes its logits' argmax), `dec_launch_body` everything
  before the choice, `dec_launch_head` the choice (asynchronous; `dec_tokens` waits and returns the tokens).
- Helpers: `block_max` is an exact maximum, so any order; `copies_wait<N>` waits until at most N of the thread's
  committed copy groups are pending; `mbar_wait` waits for the barrier's phase of the given parity; `launch` is
  `kernel<<<grid, block, smem, stream>>>` as a programmatic dependent of the stream's previous kernel; `to_fp8` is
  x / scale as saturating e4m3; `capture` instantiates the graph of the work `enqueue` puts on the stream; the failed
  prompt-store allocation's error is cleared (`cudaGetLastError`) before retrying.

### `inference/syntax/` (Rust crate `nanochat_syntax`)

- `lib.rs` `Syntax::new` arguments: `keywords`, `soft`, `operators`: (word, its terminals); `basic`: the terminals of
  NAME, NUMBER, STRING, FSTRING_START, FSTRING_END; `special`: of $END, NEWLINE, INDENT, DEDENT; `actions`, `goto`: the
  dense parse table (rows of `terminals`, `nonterminals` entries); `rules`: (length, left-hand side); `start`, `end`:
  the LR start and end states; `xid_start`, `xid_continue`: the non-ASCII code point ranges of `str.isidentifier()` on
  c and on "a" + c; `token_bytes`: the mergeable tokens' bytes (ids below BOS's, `corpus/tokenizer.py`, each distinct).
- `Syntax.constraint`: prefix may end mid-character; a non-empty suffix makes every step a candidate end.
- `Constraint`: its lexer and parser states are those after the text's first settled bytes; bytes pushed after them
  are settled (`settle`, in parallel without the GIL) by the next `masks` or `grammar_ends`, which raise ValueError if
  the bytes end the text's viability. `text`: prefix and generated tokens, UTF-8 possibly ending mid-character.
- `grammar_ends`: necessary for `ast.parse` because the grammar accepts every module CPython's parser does.
- `masks`: each row's other bits cleared; BOS's bit set iff a candidate end (a suffix follows or the grammar ends a
  module here); the caller decides a chosen candidate (`grammar_ends`, then `ast.parse`). The caller keeps the buffer
  alive, each row within it, and the rows distinct; each rayon job writes its own row only.
- `parser.rs`: `TAB_SIZE` = 8 (tokenizer.c `TABSIZE`; `ALTTABSIZE` 1: a tab is one alt column, as a space); a
  backslash line's first column (`Parse.cont`) is its indentation; the bracket level makes newlines inside brackets
  not NEWLINE tokens. `Tables`: `goto[state * nonterminals + nonterminal]`, each rule's (length, lhs); `NO_ACTION`: an
  action-table entry with no action.
  `Parse`: also whether the line has tokens, the line's column and alt column so far, the column of the line's first
  backslash (0: none). `shift`: the stack is `stack[..base] + tail` while reducing, so only the result is allocated.
  The indentation check runs at a line's first token outside brackets (a bracket's lexeme sets `line`).
- `vocab.rs`: events are lexemes emitted, newlines and line-start whitespace; the subtrie after each first byte is
  shared by the lexer states that byte takes to one state (`Syntax::grouping`, `Syntax::subtree`): from most states
  the first byte ends the partial token and the lexers converge. `Key`: a token after which no character completes
  the pending UTF-8 sequence has no key. `Syntax` shared by every constraint and thread. `complete(state, ends_line)`:
  `ends_line` says the text ends with a line end.
- `lexer.rs`: `tok_get_normal_mode`, `tok_get_fstring_mode`, `verify_end_of_number` restated; a prefix may end
  mid-token, mid-indent or mid-UTF-8 character; `translate_newlines` reads `\r\n`, `\r` as `\n`. A soft keyword's two
  terminals are read both ways by the parser. `Num`: an underscore state ends in U, a digit state in D. `Pend`: "{"
  after "{", "}" after "}", the character after a backslash, "{" after "\N", the end of "\N{...}". `Lexicon`: spelled
  names (keywords, soft keywords, string prefixes) and operator prefixes as tries over ASCII; Python's identifier
  classes. `Lexicon::new`: Lark's terminal names resolved to indices by the caller. `operator_start`: tokenizer.c's
  f-string punctuation check precedes the two-character operators. `byte`: a character once its bytes are complete.
  `representative`: contexts take non-ASCII characters by these classes alone (identifier start, else identifier
  character, else any character). `partial`: none = no partial token. `Sub::Sp`: between tokens; `Sub::S`: a string's
  body; in `quote_char` the third quote (after two read) opens a triple-quoted string.
- `python.lark`: Lark builds its tables (strict mode proves them conflict-free) and the crate reads them as a
  viable-prefix recognizer. Terminals beside the literals are CPython's tokenizer token types; FSTRING_MIDDLE is dropped
  because an f-string's literal text never decides a parse (`fstring_middle*` and `fstring_format_spec*` repeat
  freely); the soft keywords are read as keywords (python.gram `match_stmt`, `case_block`, `type_alias`).

## References

| arXiv | Title |
|---|---|
| 1211.5063 | On the difficulty of training Recurrent Neural Networks |
| 1411.0030 | A* Sampling |
| 1904.09751 | The Curious Case of Neural Text Degeneration |
| 2101.00027 | The Pile: An 800GB Dataset of Diverse Text for Language Modeling |
| 2107.03374 | Evaluating Large Language Models Trained on Code |
| 2203.07814 | Competition-Level Code Generation with AlphaCode |
| 2204.02311 | PaLM: Scaling Language Modeling with Pathways |
| 2205.14135 | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness |
| 2207.14255 | Efficient Training of Language Models to Fill in the Middle |
| 2208.08227 | MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation |
| 2301.03988 | SantaCoder: don't reach for the stars! |
| 2303.17568 | CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Benchmarking on HumanEval-X |
| 2305.01210 | Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation |
| 2308.12950 | Code Llama: Open Foundation Models for Code |
| 2309.06180 | Efficient Memory Management for Large Language Model Serving with PagedAttention |
| 2310.11248 | CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion |
| 2312.07104 | SGLang: Efficient Execution of Structured Language Model Programs |
| 2401.02954 | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism |
| 2401.14196 | DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence |
| 2402.17988 | Constrained Decoding for Fill-in-the-Middle Code Language Models via Efficient Left and Right Quotienting of Context-Sensitive Grammars |
| 2402.19173 | StarCoder 2 and The Stack v2: The Next Generation |
| 2403.01632 | SynCode: LLM Generation with Grammar Augmentation |
| 2411.00640 | Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations |
| 2412.04403 | Establishing Task Scaling Laws via Compute-Efficient Model Ladders |
| 2503.13423 | SuperBPE: Space Travel for Language Models |
| 2504.09246 | Type-Constrained Code Generation with Language Models |
| 2504.11393 | DataDecide: How to Predict Best Pretraining Data with Small Experiments |
| 2505.18789 | From Output to Evaluation: Does Raw Instruction-Tuned Code LLMs Output Suffice for Fill-in-the-Middle Code Generation? |
| 2508.13144 | Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation |
| 2604.07769 | An Empirical Study on Influence-Based Pretraining Data Selection for Code Large Language Models |
| 2606.21619 | The Alignment Problem in Constrained Code Generation |
| 2608.18062 | TokEval: A Tokenizer Evaluation Suite; companion post cimeister.github.io/blog/code_tokenizer_ablations |

Repositories and other sources, at the commit or file cited: `evalplus/evalplus` (`26d6d00`), `nuprl/MultiPL-E`
(`3025a53`; dataset `3a9e8d2`), `openai/human-eval-infilling` (`88062ff`), `zai-org/humaneval-x` (`62c7862`),
`amazon-science/cceval` (`40c68d2`), `bigcode-project/bigcode-evaluation-harness` (`santacoder_fim.py`),
`cimeister/tokenizer-intrinsic-evals` (`7d03a61`), `karpathy/nanochat` (`evaluate_bpb`, `--eval-tokens`),
`python/cpython` 3.12.3 (`Parser/tokenizer.c`, `Grammar/python.gram`), `guidance-ai/llguidance` (issue #107),
`vllm-project/vllm` (`benchmarks/serve.py`, Transformers backend), `huggingface/transformers` (`modeling_qwen3.py`,
`TikTokenConverter`). Token healing: "The Art of Prompt Design: Prompt Boundaries and Token Healing" (Guidance, 2023).
