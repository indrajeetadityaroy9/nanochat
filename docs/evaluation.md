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
  tokenizer are scored on identical rows. `cu_seqlens` is padded to the corpora's most segments (one compiled graph).
- **Reference solutions** (`solution_bits`, `SOLUTION_BENCHMARKS`): the bits of HumanEval+'s and MBPP+'s canonical
  solutions (EvalPlus) and HumanEval-X C++'s (arXiv 2303.17568; MultiPL-E has none) given their prompts, one example
  per 8K row, BOS first: the task loss of arXiv 2412.04403 §2.2. A continuous correct-answer likelihood makes
  HumanEval and MBPP decisions predictable where accuracy is trivial (arXiv 2504.11393 §3.4, Fig. 6); BPB raises the
  signal-to-noise ratio of most benchmarks (arXiv 2508.13144 §5.3); the loss on a benchmark's problems and reference
  solutions tracks pass@1 where held-out loss does not (arXiv 2604.07769 §3.3.1). A solution's leading blank lines
  count as prompt: EvalPlus's solutions begin with them and a tokenizer joins them to the prompt's last line break, so
  the prompt would otherwise end inside a token. Scoring starts at the token that starts at the prompt's end
  (`decode_with_offsets`; none raises).

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
  BigCode leaderboard 50: the same estimate with more variance per problem [inference]. HumanEval infilling is greedy,
  not arXiv 2207.14255's sampling. The prompts' examples overlap the scored base tests (MBPP+: nearly every prompt
  assert calls the entry function on a base input; HumanEval+: many tasks share a doctest input), so the filtered mode
  favours base over plus pass@1 (`verify.py`). Prompts, stops, budgets and scorers were checked against EvalPlus,
  MultiPL-E and human-eval-infilling at their pinned commits (the FIM line cut, the MultiPL-E data pin and the
  example checks' sandbox were the deviations, fixed in `a41415e`). `char` is unsigned on ARM64 and signed on x86, so
  a MultiPL-E C++ program that depends on it can score differently [inference].
- **Contamination.** `corpus/` has no decontamination step against the scored benchmarks (the only decontaminated
  data are the publisher's `*_decontaminated` splits of `open-r1/codeforces-cots`, `corpus/sources.py`): a benchmark
  problem present in the training data inflates its score.

## 5. Metrics and statistics

`metrics.py` is the repository's one source of metrics, torch-free so training, evaluation and analyses import it
alike. Every public function:

| Function | Computes | Source definition; library |
|---|---|---|
| `append`, `records`, `train_steps`, `diverged`, `curves` | the run log `RUNS/<run>/metrics.jsonl` (record kinds and fields: `docs/training.md` §6): append one JSON record with `run_name`, `attempt`, `time` (NaN and infinities raise, `allow_nan=False`); read it; the latest attempt's train record per step; the last failure record's step or None; train records as columns in step order (None where a record lacks the field: `update_rms`, `param_norm` exist every 10th step) | — |
| `bpb(*pooled)` | Σ bits / Σ bytes over context buckets and arguments | The Pile, arXiv 2101.00027 §3.1 |
| `mixture_bpb(corpora, weights, buckets=)` | Σ_c w_c · bits_c / rows_c over Σ_c w_c · bytes_c / rows_c: the bits per byte of val rows drawn with row weights w | as The Pile pools its components |
| `offset_bpb` | `mixture_bpb` per byte-offset bucket (None where no target is); a bucket pools the documents long enough to reach it, so the curve mixes context use with document composition | arXiv 2308.12950 Fig. 4a; 2503.13423 App. B.1 |
| `bpb_summary(bits, nbytes, row_weights=)` | the ratio Σw·bits / Σw·bytes over rows (0-byte rows dropped) as the WLS fit of row bpb on a constant with weights w × bytes; standard error HC1 (the delta-method ratio standard error × √(n/(n−1))); 95% normal interval | Miller, arXiv 2411.00640 Eq. 1, 5–6; statsmodels `WLS` |
| `mixture_bpb_summary` | `bpb_summary` of every sampled row of a group, a row of corpus c weighted w_c / rows_c (the `mixture_bpb` weights) | as above |
| `parse_outcome(source)` | `compile()` → "parses", "indentation" (IndentationError, TabError; checked first) or "syntax" (other SyntaxError; MemoryError and RecursionError of the parser count here) | TokEval's companion post ("How the failing generations fail"; arXiv 2608.18062 App. F.3, fn. 15) |
| `parse_shares`, `parsed` | each outcome's share; per completion whether it parses | — |
| `measures(evaluation, steps, repo_level_start=)` | a run's compared measures (table below) | — |
| `measure_errors(evaluation)` | their error bars: `code`, `text`, `math`, `repo_32k` by `mixture_bpb_summary`; `solutions` by `bpb_summary`; the parse shares and `pass@1` by `summary` over problems; none for `repo_beyond_8k`, `tokens_per_second`, `mfu` | Miller Eq. 1 |
| `line_exact_match` | `prediction.strip() == reference.strip()` | SantaCoder (arXiv 2301.03988 §6.1.2; bigcode-evaluation-harness `santacoder_fim.py`), DeepSeek-Coder §4.2 |
| `code_lines`, `exact_match` | comment-stripped (`#`, `//` to the line end), stripped non-empty lines; equal lines | CrossCodeEval (arXiv 2310.11248 §4; amazon-science/cceval `40c68d2`) |
| `edit_similarity` | 100 when equal, 0 when either is empty, else round(100 × difflib `SequenceMatcher` ratio) of the comment-stripped strings | cceval's fuzzywuzzy 0.18 without python-Levenshtein (difflib) |
| `identifiers`, `identifier_match` | identifiers after removing comments and string literals, Python keywords excluded (True and False kept); exact match of the lists, F1 over the sets (0 when both are empty) | cceval `extract_identifiers` |
| `completion_scores` | per sample: `line_em`, `em`, `es` (0–100), `id_em`, `id_f1` | cceval's per-sample results |
| `pass_at_k(passed, k)` | each problem's pass@k estimate from its samples | Codex's unbiased estimator (arXiv 2107.03374 §2.1); `human_eval_infilling` `estimate_pass_at_k` |
| `summary(scores, clusters=)` | mean of question scores; OLS on a constant: unclustered, se = s/√n (sample variance over n − 1, Miller Eq. 1); clustered, Liang–Zeger CR0 without small-sample correction (Miller Eq. 4 except for Eq. 4's (n − 1) in its diagonal term); 95% normal interval (Eq. 5–6); a constant score has se 0 | statsmodels `OLS` |
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
`PEAK_FLOPS`; the run record's `model_flops_per_token` is the nominal count. DeepSeek LLM's M (arXiv 2401.02954 §3.2
Eq. 2) counts full attention in every layer and stays the scaling plan's compute, not an MFU numerator; the d24
values of M and the nominal count: `docs/architecture.md` §1. Executed FLOPs per token of a row holding one document
are 1,479,503,808 at 8K and 2,160,975,600 at 32K [arithmetic, recomputed on CPU], below nominal (a query attends only
its own segment's earlier keys) and lower still with packed documents; M is 3.8× the executed count at 32K
[arithmetic], so an M-based utilization would exceed 100%. Measured MFU: `docs/training.md` §7.

**TokEval** (`tokeval`): `UnifiedTokenizerAnalyzer.run_analysis` in its published form (`normalize_results`),
MorphScore and plots off, CER unbudgeted (`cer_time_budget_s=0`). Inputs: prose by language, code by language (via
TokEval's `CodeDataLoader`: CRLF → LF, trailing whitespace stripped; read by the AST, identifier and indentation
metrics) and math. `tokeval_wrapper`: `encode` is `encode_ordinary`; vocabulary keys are GPT-2's byte-to-unicode
rendering of each token's bytes (a bijection, so partial characters survive); `pretokenize` is the split pattern;
offsets from each token's byte length (`token_offsets`). Tree-sitter grammars: `tree-sitter-language-pack==1.22.1`
(`pyproject.toml` override). Results per tokenizer: `docs/experiments.md` §2.

## 6. Decoder

`inference/decoder.py` `NativeDecoder` with `model/kernels/decode.cu` generates every completion of §3–§4.

- **Batching.** Up to `MAX_BATCH` = 1,024 slots (fewer if device memory is short), continuous: a finished slot takes
  the next request. Each step is one captured CUDA graph, its kernels programmatic dependents of each other (PDL); a
  second set of graphs over the first `SMALL_BATCH` = 256 slots runs while no higher slot is active. A slot's KV
  cache holds 2 · L · max_length · (C + 4H) bytes, 38,016 bytes per position at d24 [arithmetic].
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
  with the same prompt copies them; prompts unused in the current and previous `generate` call are evicted first.
- **Request** (`Request`): `input_ids`, `max_new_tokens`, `stopping_criteria`, `temperature` (0: argmax, ties to the
  lowest id; > 0: Gumbel-max sampling of softmax(z / T), the noise a Philox function of (seed, generated position,
  token) alone, arXiv 1411.0030 §2), `top_p` (< 1: the fewest most likely allowed tokens holding top_p of the
  probability, arXiv 1904.09751 §3.1, `k_nucleus`), `seed`, `constraint`. Sampled logits are capped, z = c·tanh(s/c)
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
  (4.35×, 2.84×); d24 step 8.27 ms at 1,024 slots, one slot 1.29 ms. A d16 checkpoint's per-problem results changed
  little and in both directions (MBPP+ greedy +4/−5 base, +2/−7 plus; HumanEval+ greedy unchanged; FIM line exact
  match within 0.002).

## 7. Syntax-constrained decoding and post-generation checks

**Syntax, not types.** Type-constrained decoding (arXiv 2504.09246) covers TypeScript only, and an incomplete
constrainer lowers functional correctness while a complete syntax constrainer gained (arXiv 2606.21619 §5.3; its
rejection as a design: `docs/experiments.md` §6). Python 3 syntax can be constrained completely, and every Python file
of the code categories parses as Python 3 (`curate.repaired`, `docs/corpus.md` §3; UltraData-Code L3's exercises and
the Kaggle notebooks are not checked), so the model's distribution of program files lies inside the constrainer's
language [inference]. Types are checked after generation instead.

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

Where it is looser than CPython it accepts (over-approximations listed in the module docstring: assignment targets as
expressions, free parameter order, unchecked escapes and nesting limits, among others); BOS is exact.
Libraries not used: SynCode (arXiv 2403.01632; its one-regex f-strings reject valid 3.12 code), llguidance (no
INDENT/DEDENT, guidance-ai/llguidance issue #107), arXiv 2402.17988 (no released library). C++ is not constrained.

Checks (`55e64de`, `ba01bb5`, `d0a2ec0`): 0 false rejections over 11.4M token steps of 3,000 Python files (the
standard library, site-packages, this repository; BPE) and 9.2M randomly segmented; no prefix that `ast.parse` accepts
without the grammar's end; a d16 checkpoint's unconstrained completions (EvalPlus and SPM infilling, greedy and
sampled) replayed through their constraints: every masked token invalid, outputs token-identical wherever nothing was
masked; all 8,488 infilling reference middles (BPE and two random segmentations) accepted; `viable_end` agrees with
`allowed` on 8,002,126 prefixes. Tables built from another vocabulary masked the true token in 1,475 of 3,199 steps,
hence the caller's tokenizer. Mask time per step 0.7–2.6 ms (3.4–15.8 ms before batching into pinned rows).

**Post-generation checks** (`eval/verify.py`, the filtered mode's acceptance test):

- Python (`python_diagnostics`): the program parses (`ast.parse`, parser overflow a failure), then basedpyright 1.40.2
  (one process per batch) with every rule off but `reportUndefinedVariable`, `reportAttributeAccessIssue`,
  `reportCallIssue`, `reportReturnType` (definite runtime failures or annotation contradictions; argument types are
  off because pyright rejects HumanEval/130's canonical solution). EvalPlus solutions (`solution_diagnostics`): the
  generated text's syntax error, an entry function missing or without a `return <value>` in its own body (sanitize
  keeps a parseable span and would hide both).
- Examples (`examples_pass`): the prompt's doctest `>>>` examples and Python `assert` lines, run by value (equal to
  the want, or None/True for a bare expression) in a fresh interpreter under EvalPlus's `reliability_guard`, 3 s
  (HumanEval's timeout); true without examples.
- C++ (`cpp_diagnostics`): `g++ -std=c++17 -fsyntax-only`, the MultiPL-E judge's compiler and standard, 15 s; C++
  prompts carry no examples here.

`search` decodes every task without an accepted candidate yet, one round per attempt, at most 10; `chosen` returns
the accepted candidate, else the greedy one. FIM candidates are checked as prefix + middle + suffix against the
prefix's examples.

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
(`docs/architecture.md` §1, §4). vLLM runs in its own image (`vllm/vllm-openai:v0.32.0` at a pinned digest,
`scripts/standard_eval.sh`), which the NGC-based nanochat image cannot hold.

**Agreement** (a d16 checkpoint served by the earlier native vLLM class, commits `3c03023`–`ec1c841`; its logit
agreement and speed: `docs/experiments.md` §6): `tokenizer.json` encodes 108,026 held-out texts exactly as tiktoken;
greedy HumanEval+ 0.067/0.067, MBPP+ 0.056/0.045 (base/plus), FIM single-line line exact match PSM 0.022, SPM 0.368,
against `NativeDecoder`'s (before `4385748`) 0.061/0.061, 0.045/0.037, 0.022, 0.367 (standard error of HumanEval+
≈ 0.019). The Transformers backend that replaced the class (`7794e2f`) ran at its rate and, in the same execution
mode, gave the same logits (eager: mean KL 4.25·10⁻⁴ on HumanEval+ for both) [unverified: no record or commit].

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

## References

| arXiv | Title |
|---|---|
| 1411.0030 | A* Sampling |
| 1904.09751 | The Curious Case of Neural Text Degeneration |
| 2101.00027 | The Pile: An 800GB Dataset of Diverse Text for Language Modeling |
| 2107.03374 | Evaluating Large Language Models Trained on Code |
| 2203.07814 | Competition-Level Code Generation with AlphaCode |
| 2204.02311 | PaLM: Scaling Language Modeling with Pathways |
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
