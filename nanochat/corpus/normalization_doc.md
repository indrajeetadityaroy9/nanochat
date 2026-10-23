# Pretraining corpus: schema, normalization and mixture

Objective: pretraining from scratch for code generation in Python and C++, with the 19 sources of
`nanochat/corpus/sources.py`. The reference run is production's d24: 195.1M scaling parameters, 102.6B tokens, an 8K
file-level stage followed by a 32K repository stage, and a linear decay of the learning rate to zero.

## 1. Schema

| Table     | Purpose                                        | Fields                                                 |
| --------- | ---------------------------------------------- | ------------------------------------------------------ |
| `samples` | Every pretraining document                     | `id`, `source_id`, `text`, `group_key`, `relative_path` |
| `sources` | Dataset identification and sampling categories | `source_id`, `name`, `domain`                          |

There are eight named columns across two tables. Each sample has five, two of them nullable.

```sql
CREATE TABLE sources (
    source_id SMALLINT PRIMARY KEY,
    name VARCHAR NOT NULL,
    domain VARCHAR NOT NULL
);

CREATE TABLE samples (
    id VARCHAR NOT NULL,      -- hex SHA-1 of text's UTF-8 bytes
    source_id SMALLINT NOT NULL,
    text VARCHAR NOT NULL,
    group_key VARCHAR,        -- the unit a train/validation split must not divide
    relative_path VARCHAR     -- set only for repository source files
);
```

- **`id` is the content hash, not a sequence.** A `nextval()` id depends on the order of ingestion, so it changes
  between rebuilds. A SHA-1 of the text is the same in every rebuild and gives exact deduplication across sources for
  free. `nanochat/corpus` already computes it.
- **`group_key` replaces `repo_name`.** Splits must keep together more than repositories:
  - repository files and GitHub issues: the repository, `owner/name`;
  - Codeforces-CoTs: the problem. A problem has up to three rows across the two subsets (one Python generation, the
    shortest and the longest C++ one), so splitting by sample would put it on both sides. The key is the smallest of
    the problem's `id` and its `aliases`, because Div. 1 and Div. 2 rounds republish the same problem under two ids;
  - everything else: NULL, meaning each sample is its own group.

  Repository-level packing (the 32K stage) groups rows that have a `relative_path` by `group_key`.
- **No index or foreign key on `samples`.** At about 966M rows, constraints only slow bulk loading, and training reads
  the table sequentially.
- **Storage.** The text is about 3.6 TB uncompressed [estimate]. One Parquet directory per source, read by DuckDB as
  `samples`, keeps rebuilds per source and is what the tokenizer and packer already read. A single DuckDB file of that
  size gains nothing.

### Why reasoning is not a separate domain

Reasoning is not exclusive of the other domains. Codeforces-CoTs is code with reasoning; UltraData-Math L3 is
mathematics with reasoning. The model should read the whole sequence, steps and answer together, so reasoning stays
inside `text`. The source identifies which samples are synthetic or reasoning traces, and the mixture (§7) weights
those sources separately.

## 2. Domains

| Domain             | Contents                                                     | DeepSeek-Coder's category (arXiv 2401.14196 §2)         |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------- |
| `code`             | repository source files, synthetic exercises, solution traces | source code                                            |
| `engineering_text` | GitHub issues, Kaggle notebooks                              | English code-related natural language (Markdown, StackExchange) |
| `mathematics`      | mathematical web text and synthetic mathematics              | none                                                    |
| `language`         | general English web text                                     | code-unrelated natural language                          |

## 3. Normalization for every source

| Source                                       | `text`                                                                  | `group_key`               | `relative_path`               |
| -------------------------------------------- | ----------------------------------------------------------------------- | ------------------------- | ----------------------------- |
| Stack-Edu Python / C++                       | the blob fetched from Software Heritage, decoded with its `src_encoding` | `repo_name`               | `path` without its leading "/" |
| UltraData-Stack-v3-L2 Python / C++           | `content`                                                               | `repo_name`               | `relative_path`               |
| UltraData-Code L2 Python / C++               | `content`                                                               | `repo_name`               | `relative_path`               |
| UltraData-Code L3 Python / C++               | `content` (the task and its solution)                                   | NULL                      | NULL                          |
| UltraData-Math L1                            | `content`                                                               | NULL                      | NULL                          |
| UltraData-Math L3 (four formats)             | `content`                                                               | NULL                      | NULL                          |
| Codeforces-CoTs Python / C++                 | `prompt` followed by `generation`                                       | the canonical problem id  | NULL                          |
| GitHub issues                                | `text`                                                                  | `repo_name`               | NULL                          |
| Kaggle notebooks                             | `text`                                                                  | NULL                      | NULL                          |
| ClimbMix                                     | `text`                                                                  | NULL                      | NULL                          |
| OPC FineWeb Math                             | `text`                                                                  | NULL                      | NULL                          |

Rules for every source:

- A row without text is dropped. That includes a Stack-Edu blob the bucket does not hold or that does not decode.
- Repository source files are kept as their language reads them: CR LF and a lone CR become LF, and a leading U+FEFF
  is dropped (Python Language Reference §2.1.2 and §2.1.4; C++23 [lex.phases] phases 1 and 2). The catalogued
  Stack-Edu C++ sample, for example, carries CR LF.
- Every other text is kept verbatim.

### UltraData-Code L3: the publisher's `content`

Use `content`, the publisher's serialization of the task and its solution: the task as a code comment, as unfenced
text or as Markdown, in 50 / 25 / 25 proportions (UltraData-Code report, App. C.3).

- **This is the format the evidence supports.** In the report's 10B-token runs, L2 plus task-and-solution L3 scored
  39.84 EvalPlus / 28.45 MultiPL-E. L2 plus all four fields (task, analysis, solution, tests) scored 37.49 / 25.28,
  because task and solution "concentrates more tokens on the task-solution relationship" (UltraData-Code report).
- **Task, analysis and solution without tests is untested.** It is the natural ablation arm. The DuckDB expression
  for it:

```sql
concat_ws(E'\n\n',
          '### Task' || E'\n' || task,
          CASE WHEN analysis IS NOT NULL AND trim(analysis) <> '' THEN '### Analysis' || E'\n' || analysis END,
          '### Solution' || E'\n' || solution) AS text
```

- **Tests stay out of both formats.** They are LLM-written candidates, and the four-field format that includes them
  scored lower.
- **Not stored:** `raw_content` (the seed file the exercise was written from), `full_content` and the format fields.

### Codeforces-CoTs

- **Text.** `prompt || generation`: the prompt already ends with a line break, and this reproduces `nanochat/data`'s
  documents row for row. Keep `<think>` and the final code; never split reasoning and code into columns.
- **Keep only `finish_reason = 'stop'`.** Truncated traces stop before their code: 4.6–7.3% of rows, measured on
  `nanochat/data`'s copy.
- **The traces are not verified.** "We did not filter the traces for correctness, but found that around 84% of the
  Python ones pass the public tests" (card). They show code-associated reasoning; they are not correct solutions.
- **Length.** 73% of these documents exceed 8,192 tokens (measured on `nanochat/data`'s copy). At 8K, most traces are
  cut before their code, which argues for the 32K stage or the low-learning-rate tail (§7).

## 4. Source registry

```sql
INSERT INTO sources VALUES
(1,  'stack_edu_python',                    'code'),
(2,  'stack_edu_cpp',                       'code'),
(3,  'ultradata_stack_v3_l2_python',        'code'),
(4,  'ultradata_stack_v3_l2_cpp',           'code'),
(5,  'ultradata_code_l2_python',            'code'),
(6,  'ultradata_code_l2_cpp',               'code'),
(7,  'ultradata_code_l3_python',            'code'),
(8,  'ultradata_code_l3_cpp',               'code'),
(9,  'ultradata_math_l1',                   'mathematics'),
(10, 'ultradata_math_l3_qa',                'mathematics'),
(11, 'ultradata_math_l3_conversation',      'mathematics'),
(12, 'ultradata_math_l3_multi_style',       'mathematics'),
(13, 'ultradata_math_l3_textbook_exercise', 'mathematics'),
(14, 'codeforces_cots_python',              'code'),
(15, 'codeforces_cots_cpp',                 'code'),
(16, 'github_issues',                       'engineering_text'),
(17, 'kaggle_notebooks',                    'engineering_text'),
(18, 'climbmix',                            'language'),
(19, 'opc_fineweb_math',                    'mathematics');
```

`codeforces_cots_python` is the `solutions_py_decontaminated` subset and `codeforces_cots_cpp` is
`solutions_short_and_long_decontaminated`. `nanochat/corpus/sources.py` currently reads both subsets as one source.

Every mixture weight is per `source_id` (§7). The weights are configuration in the experiment's manifest, not a
column.

## 5. Deduplication and decontamination

- **Exact duplicates:** one row per `id`, across all sources.
  - UltraData-Code L2 Python: 34.8% of its rows are exact copies of another row.
  - UltraData-Math L2-preview: 59.7% (both measured on `nanochat/data`'s copies).
- **Near duplicates:** MinHash at the file level across every code source. OpenCoder found file-level deduplication
  "better than … repository level deduplication a lot" (arXiv 2411.04905 §6.1).
  - Code is where most of the volume goes: `nanochat/data`'s curation kept 25.7M of UltraData-Code L2 Python's 70.1M
    normalized rows.
  - The mixture's code supply (§7.4) is counted after this step.
- **Decontamination:** DeepSeek-Coder's rule (arXiv 2401.14196 §2). Drop a file that shares any 10-gram with the test
  data. For test strings shorter than 10-grams but at least 3-grams, use exact match. The benchmarks are HumanEval,
  MBPP, GSM8K and MATH, plus the evaluation suite's MultiPL-E, HumanEval+ and MBPP+.
  - Codeforces-CoTs' own decontamination covers AIME, GPQA, MATH-500 and LiveCodeBench only (card).
  - `nanochat/data`'s acquisition analysis found HumanEval task text in UltraData-Stack-v3-L2's Python files.

## 6. What is not stored

| Fields                                        | Decision                                                                 |
| --------------------------------------------- | ------------------------------------------------------------------------ |
| Publisher ids, UUIDs, blob ids                | Discard after retrieval (the blob id fetches Stack-Edu's text)           |
| Codeforces problem ids                        | Keep only the canonical id, as `group_key`                               |
| Licensing metadata                            | Exclude from the training tables; record in the manifest                 |
| Timestamps, file sizes, line counts           | Discard                                                                  |
| Quality and relevance scores                  | Discard after any source-level filtering                                 |
| API token counts, `finish_reason`             | Discard after the `finish_reason = 'stop'` filter                        |
| Serialized duplicates (`full_content`, `raw_content`, `messages`) | Discard                                                 |
| Codeforces statements' fields and tests       | Discard after decontamination                                            |
| UltraData-Code L3 tests and analysis          | Not in `text` (`content` omits them)                                     |
| Programming language per sample               | Do not store; the source distinguishes Python from C++                   |
| Per-sample token counts                       | Do not store; tokenization is downstream                                 |
| Content hash                                  | Stored as `id` and used for exact deduplication                          |

The experiment's manifest, outside the tables, records:

- each source's Hub revision. `nanochat/corpus/prepare.py` reads every dataset at its latest revision and so far
  only prints it;
- the preprocessing rules, the deduplication and decontamination parameters, the mixture weights and the tokenizer;
- the license records. UltraData-Code, for example, names obligations owed to the source repositories.

## 7. Mixture

### 7.1 DeepSeek-Coder (arXiv 2401.14196)

DeepSeek-Coder (1.3B to 33B, 2T tokens from scratch) trains on "87% source code, 10% English code-related natural
language corpus, and 3% code-unrelated Chinese natural language corpus".

- **The 10%** is "GitHub's Markdown and StackExchange", used "to enhance the model's understanding of code-related
  concepts and improve its ability to handle tasks like library usage and bug fixing".
- **The code** is 798 GB in 603M files over 87 languages. It is filtered with StarCoder's rules, ordered by
  file dependencies within a repository, deduplicated at the repository level, screened by a compiler and a quality
  model, and decontaminated (§5).
- **FIM.** It is applied at the document level before packing, at a rate of 0.5 in PSM mode. In the FIM ablation, a
  100% FIM rate gave the best infilling and the weakest completion; 50% PSM beat MSP (§3.1).
- **No mixture ablation.** The paper does not vary these shares.
- **Its v1.5 is the one comparison.** DeepSeek-Coder-v1.5 continues DeepSeek-LLM-7B on 70% code, 10% Markdown and
  StackExchange, 7% code-related language, 7% mathematical language and 6% bilingual language (Table 9). It shows "a
  slight decrease in coding performance" for gains in mathematics and language (§5).

**Mapping onto this corpus.** DeepSeek-Coder's 10% is code-related English, which is this corpus's `engineering_text`.
Its 3% is general language, and it has no mathematics. Production's `config.MIXTURE` reads 87:10:3 as code : ClimbMix :
math instead, which matches neither the 10% nor the 3%.

### 7.2 Evidence from comparable models

| Model (arXiv) | Size | Tokens | Code | Code-related language | Math | General language | Mixture evidence |
|---|---|---|---|---|---|---|---|
| DeepSeek-Coder (2401.14196) | 1.3–33B | 2T, scratch | 87% | 10% | 0 | 3% | none (design choice) |
| StarCoder2-3B (2402.19173) | 3B | 3.1T, 4.98 epochs | ≈86% | ≈12%: PRs, issues, notebooks, StackOverflow, docs | ≈0 (exercise sets ≈2%) | 0 | none; OpenWebMath, Wikipedia and arXiv excluded at 3B (§4) [inference: shares from Table 4] |
| Code Llama (2308.12950) | 7–70B | +500B on Llama 2 | 85% | 8% | 0 | 7% | language "improves the performance of our models on MBPP" (§2.2, no numbers) |
| OpenCoder (2411.04905) | 1.5B, 8B | 2T (4 epochs) + 100B anneal | ≈90% | ≈10% code-related web | 0 | 0 | the anneal without its algorithmic and synthetic data "drops a lot" (§6.2) |
| Arctic-SnowCoder (2409.02326) | 1.3B | 500B + 50B + 5B | 100% | 0 | 0 | 0 | HumanEval+ 14.0 → 21.3 → 28.0 over its raw, high-quality and synthetic phases (Table 2) |
| CodeGemma (2406.11409) | 2B; 7B | +0.5–1T on Gemma | 100%; 80% | in the 20% | in the 20% | in the 20% | none |
| Granite Code (2405.04324) | 3–34B | 4T code, then 500B | 100%, then 80% | issues within code | in the 20% | in the 20% | none; math and StackExchange added in the second phase |
| Qwen2.5-Coder (2409.12186) | 0.5–32B | 5.5T on Qwen2.5 | 70% | within code | 10% | 20% | 7B, coding Common / BigCodeBench: 100:0:0 49.8 / 40.3; 70:20:10 48.3 / 38.3; 85:15:5 43.3 / 36.2 (Table 3) |
| DeepSeek-Coder-V2 (2406.11931) | 16B, 236B MoE | +6T on DeepSeek-V2 | 60%, with Markdown and issues | within code | 10% | 30% | none |
| To Code, or Not To Code? (2408.10914) | 470M | 200B | 0–100% | 15% replacing code | — | rest | below |
| UltraData-Code report (2610.23960) | 1.3B | +10B, +100B | L2 and L3, 1:1 | — | — | — | below |
| UltraData-Math (2602.09003) | 1.2B | 120B, scratch | 17% | — | 8% | 75% | tiered (L1 → L3) beats mixed |

### 7.3 What the evidence settles

1. **Code at about 87%.** Code-first models trained from scratch or specialized for code use 85–90% code; the
   smallest completion models use 100%.
   - In a controlled sweep at 470M, code benchmarks rise linearly with the code share up to 100% code, 2.6 times the
     25% model (arXiv 2408.10914 §3.3).
   - At 7B, all code scored highest on coding (Qwen2.5-Coder Table 3).
   - The 60–70% shares of Qwen2.5-Coder, DeepSeek-Coder-V2 and DeepSeek-Coder-v1.5 buy mathematics and general
     language at some cost to code (v1.5's "slight decrease in coding performance").
2. **About 10% code-related language, from issues and notebooks.** This is DeepSeek-Coder's 10%, StarCoder2-3B's ≈12%
   and Code Llama's 8%.
   - The one controlled measurement is negative for Python. Replacing 15% of a 470M code-only model's data with
     issues, StackExchange and notebooks (5% each) lowered its code score by 9.4% (arXiv 2408.10914 §3.4).
   - Ling-Coder-Lite reports that notebooks "significantly enhance" a 1B model's code ability (arXiv 2503.17793
     §2.3.3, no numbers).
   - So notebooks, which are code-dense, are read whole and issues fill the rest of the 10%.
3. **About 3% general language.** This is DeepSeek-Coder's 3% and Code Llama's 7% ("improves … MBPP"). At 100% code,
   language reasoning drops sharply, by 18.3% relative (arXiv 2408.10914 §3.3). HumanEval and MBPP prompts are prose.
4. **Mathematics only in the low-learning-rate tail, and little of it.**
   - No code-first model trained from scratch above includes web mathematics. Code Llama, which has none, loses GSM8K
     to Llama 2.
   - Where mathematics was measured for code, the comparison is between kinds of mathematics, not against none. As
     the 30% verification slice of a decay, UltraData-Math L3 gave MBPP 49.27 / HumanEval 32.93, against L1's 44.71 /
     29.88 (arXiv 2602.09003 Table 6).
   - Granite adds mathematics only in its second, annealing-like phase. Hence UltraData-Math L3, in the tail only.
5. **Synthetic exercises in both phases, concentrated in the tail.**
   - In a 470M code model, 10% synthetic code (3.2B tokens) raised code benchmarks by 44.9% (arXiv 2408.10914 §3.4).
   - L2 plus L3 at 1:1 gave EvalPlus 57.06 against L2's 46.95 after 100B tokens (UltraData-Code report Table 6).
   - Late placement wins:
     - OpenCoder's anneal on 84% of its pretraining distribution plus algorithmic and synthetic data;
     - Arctic-SnowCoder's final 5B synthetic tokens;
     - at 1.2B from scratch, quality tiers ordered L1 → L3 beat mixing them: MBPP 16.34 against 12.06
       (arXiv 2602.09003 Table 7).
   - Against staging at small scale: SmolLM2's 135M and 360M models did better with "a single-stage training approach
     with consistently high-quality data" (arXiv 2502.02737 §6). So L3 enters from the start at a moderate share and
     rises in the tail.
6. **Reasoning traces only in the tail.** No paper measures R1 traces in pretraining. Seed-Coder uses
   `open-r1/codeforces-cots` in its post-training long-chain-of-thought warmup (arXiv 2506.03524 §3.2.2). The source
   is 0.34B tokens, so it is read once, in the tail.
7. **Repetition up to about 4 epochs costs little.**
   - "Training with up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data"
     (arXiv 2305.16264).
   - StarCoder2-3B reads its data 4.98 times; OpenCoder-1.5B 4 times.
   - Arctic-SnowCoder's top 12.5B code tokens read 4 times beat 50B read once on HumanEval(+) and EvoEval, not on
     MBPP (Table 6).
   - Production's rule that no document is read twice is stricter than needed. The mixture below needs no second
     pass except possibly for issues.
8. **FIM stays at DeepSeek-Coder's 0.5**, half PSM and half SPM as production, on repository source files only
   (`relative_path IS NOT NULL`). Code Llama's 0.9 cost its 7B model 0.6 points on HumanEval and MBPP (Table 5).
   Seed-Coder lowers FIM to 0.1 in continued pretraining.

### 7.4 The mixture

The phases follow production's plan:

- **Main:** the file-level stage's first 90% of tokens.
- **Tail:** the file-level stage's last 10%, where the linear decay puts the learning rate below about 15% of its
  peak.
- **Repository:** the 32K stage, unchanged. It reads repository source files only, at `REPO_SHARE` (5.45% of tokens).

Within a group, sources are weighted by their deduplicated tokens (production's `mixture_weights`).

| Group (sources) | Main | Tail | Repository | Tokens of 102.6B | Share |
|---|---|---|---|---|---|
| Repository code (1–6) | 79% | 50% | 100% | 79.4B | 77.4% |
| Synthetic exercises (7, 8) | 8% | 30% | — | 9.9B | 9.6% |
| Reasoning traces (14, 15) | — | 3% | — | 0.3B | 0.3% |
| Code-related language (16, 17) | 10% | 7% | — | 9.4B | 9.2% |
| General language (18) | 3% | 3% | — | 2.9B | 2.8% |
| Mathematics (10–13) | — | 7% | — | 0.7B | 0.7% |
| **Tokens** | 87.3B | 9.7B | 5.6B | 102.6B | |

Over the run that is 87.3% code, 9.2% code-related language and 3.5% other: DeepSeek-Coder's 87 : 10 : 3, read
with its own categories.

The supply, in production's tokenizer:

| Group | Supply | Read |
|---|---|---|
| Repository code | 88.8B after `nanochat/data`'s curation of these 6 corpora | 89%, once |
| Synthetic exercises | 34.2B of task-and-solution text: Python 16.24B, C++ 18.00B (UltraData-Code report Table 15, its tokenizer) | 29% |
| Reasoning traces | 0.34B | one pass of the `stop` rows |
| Code-related language | ≈10.1B: issues ≈8.7B [estimate], notebooks 1.42B | ≈93%; issues may need a partial second pass after deduplication |
| General language | ≈382B [estimate] | 0.8%: about 50 of ClimbMix's 6,543 files |
| Mathematics (L3) | ≈64B [estimate] | 1% |

Not read: UltraData-Math L1 (367 GB on the Hub), OPC FineWeb Math, and all but about 50 ClimbMix files. Fetching only
what the mixture reads cuts the download by about 1 TB.

### 7.5 What the literature leaves open at 195M

- **Placement of the synthetic exercises.** The evidence splits between staging at 1.2B (arXiv 2602.09003) and a
  single stage at 135M and 360M (arXiv 2502.02737). The ablation is L3 at 8% throughout against the staged 8% / 30%
  above, at equal tokens.
- **Code-related language against code.** 10% of issues and notebooks against the same tokens of repository code,
  measured on Python and C++ pass@1. The one controlled result (arXiv 2408.10914 §3.4) evaluated Python only.

## 8. Training representation

| Sample type                          | Representation                                                        |
| ------------------------------------ | --------------------------------------------------------------------- |
| Repository Python / C++ source files | causal and FIM (rate 0.5); repository-level packing in the 32K stage  |
| UltraData-Code L3                    | causal: task, then solution                                           |
| Codeforces-CoTs                      | causal: prompt, reasoning, code                                       |
| UltraData-Math L3                    | causal mathematical text                                              |
| Kaggle notebooks                     | causal Markdown and code                                              |
| GitHub issues                        | causal engineering discussion                                         |
| ClimbMix                             | causal natural language                                               |

- **FIM eligibility is `relative_path IS NOT NULL`**, never `domain = 'code'`. The domain also holds exercises and
  traces, whose order (task before solution) FIM would break.
- **Downstream of DuckDB:** document boundaries (the tokenizer's document token between samples), FIM, tokenization
  and packing.
- **Explanations are not verified.** Pretraining on explanations next to code lets the model learn how prose relates
  to implementations. It is neither an alignment objective nor a guarantee that the explanations are correct.

Decision: two tables, eight columns (with `group_key` and a content-hash `id`), four domains, no reasoning schema and
no publisher metadata. Two sources need their text constructed: Codeforces-CoTs (`prompt || generation`, finished
traces only) and UltraData-Code L3 (the publisher's `content`). The mixture is DeepSeek-Coder's 87 : 10 : 3 with its
own categories, plus UltraData-Code L3 from the start and synthetic exercises, traces and mathematics concentrated in
the low-learning-rate tail.
