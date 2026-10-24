# Pretraining corpus for Python and C++ code generation: specification

This document specifies `nanochat/corpus`, which replaces `nanochat/data`: pretraining from scratch for code
generation in Python and C++, from the 18 sources of `nanochat/corpus/sources.py`. Every rule is a
decision. Every number is quoted from a paper, read from the code, or measured on the GB10; `[inference]` marks a step
no source states, `[arithmetic]` a figure derived from others.

The reference run is production's d24: 195.1M scaling parameters, 102.6B tokens in 86,975 steps
(`docs/h100-pretraining.md`, `train/schedule.plan`); an 8K file stage, then a 32K repository stage over the last 5.45%
of steps (82,231–86,975, `config.REPO_SHARE`); a linear decay of the learning rate to zero after 40 warmup steps
(`train/schedule.py`); document masking.

Papers are read with `hf papers read <arXiv id>`; where that markdown drops a table's body, the source is named. The
UltraData-Code report (arXiv 2610.23960) is read from `tech_report/UltraData_Code_Tech_Report.pdf` in
`openbmb/UltraData-Code`. Measurements name their data: this corpus's files, or (R1–R5) `nanochat/data`'s copies of
the sources and production's packs.

| Area | Decision | § |
|---|---|---|
| Sources | 18 read, each at its latest Hub revision, recorded in the manifest; only the columns standardization reads are fetched | 1 |
| Schema | one Parquet schema per document category; `id` is the SHA-1 of the document's text | 2 |
| Text | LF line ends and no leading byte-order mark; otherwise as published; Codeforces-CoTs finished traces only | 3 |
| Filters | every Python file parses as Python 3, after repair; no generated, vendored or C++ syntax filter | 4.1, 4.2 |
| Deduplication | exact copies only, across every source read, first occurrence kept | 4.3 |
| Split | 1% of groups (repository, problem or document) held out; repository names case-insensitive | 4.4 |
| Mixture | 87 : 10 : 3 code, text, mathematics in both stages; synthetic code 30% of the 8K stage's code; Kaggle notebooks and CodeXGLUE's functions read whole; 32K code: whole repositories and Codeforces-CoTs | 5 |
| FIM | rate 0.5, joint PSM/SPM, context level, AST-FIM's middles with 10% random spans, repository files only | 6 |
| Sequences | Best-Fit-Decreasing packing, document masking, cuts between statements; 32K documents per (repository, language) | 7 |
| Dynamics | 102.6B tokens; one switch, to 32K; linear decay to zero; no document read twice | 8 |
| Tokenizer | retrained on this corpus at the 8K stage's shares of bytes; production's vocabulary size; pattern and score decided on the GH200 | 9 |
| Publication | the curated text in the private `iaroy/nanochat-data` at `a4ec1a08`, one folder per source; tokenizer, compile, packing and FIM run after the pull | 10 |

## 1. Sources

| Source (`sources.py`) | Hub dataset @ revision | Category (§2) | Text | Files read | Standardized rows |
|---|---|---|---|---|---|
| `stack_edu_python`, `stack_edu_cpp` | HuggingFaceTB/stack-edu @ `eeec5caa` | REPOSITORY_FILE | the Software Heritage blob | 5 of 5, 4 of 4 | 25,285,896, 16,246,723 |
| `ultradata_stack_v3_l2_python`, `_cpp` | openbmb/UltraData-Stack-v3-L2 @ `62077439` | REPOSITORY_FILE | `content` | 45 of 45, 30 of 30 | 24,671,724, 17,990,102 |
| `ultradata_code_l2_python`, `_cpp` | openbmb/UltraData-Code @ `b20cabcf` | REPOSITORY_FILE | `content` | 119 of 119, 31 of 31 | 72,844,227, 23,290,266 |
| `ultradata_code_l3_python`, `_cpp` | openbmb/UltraData-Code @ `b20cabcf` | EXERCISE | `content` | 28 of 147, 32 of 167 | 3,952,421, 4,330,659 |
| `nemotron_rewriting_python`, `nemotron_transpilation_cpp` (`synthetic-rewriting`, `synthetic-transpilation`) | nvidia/Nemotron-Pretraining-Code-v2 @ `7b1a453d` | SYNTHETIC_FILE | `content` | 13 of 64, 13 of 64 | 20,292,444, 5,972,251 |
| `codexglue_python` (code-to-text, `python/train-*`) | google/code_x_glue_ct_code_to_text @ `2678080e` | FUNCTION | `code` | 2 of 2 | 251,820 |
| `ultradata_math_l3_qa`, `_conversation`, `_multi_style`, `_textbook_exercise` | openbmb/UltraData-Math @ `fe10db8e` | TEXT | `content` | 2, 2, 3 and 2 of 50 | 1,084,548, 675,242, 686,726, 1,040,216 |
| `codeforces_cots` (`solutions_py_decontaminated`, `solutions_short_and_long_decontaminated`) | open-r1/codeforces-cots @ `39ac85c1` | CONVERSATION | `prompt`, then `generation` | 9 of 9 | 22,914 |
| `kaggle_notebooks` | HuggingFaceTB/issues-kaggle-notebooks @ `ef882ad1` | NOTEBOOK | `text` | 11 of 11 | 580,195 |
| `climbmix` | karpathy/climbmix-400b-shuffle @ `915333b4` | TEXT | `text` | 187 of 6,543 | 15,814,656 |

Not read: web mathematics (UltraData-Math L1, OPC FineWeb Math) and StarCoder2's GitHub issues, §5.2;
Nemotron-Pretraining-Code-v2's GitHub metadata and dialogue subsets, §3.3; CodeXGLUE's Python validation and test
splits, the benchmark's, §3.4.

- **Revision:** each source is read at its latest revision, resolved once by `prepare` (`dataset_info(repo).sha`) and
  recorded with its files and rows in `CORPUS/manifest.json`.
- **Rows:** every raw file holds exactly the rows of its Hub file's Parquet footer (all 18 sources). Standardization
  drops two sets:
  - Stack-Edu's 146 rows (of 25,286,019 and 16,246,746) whose blob Software Heritage's bucket does not hold:
    `404 NoSuchKey` over unsigned HTTP, against `200 OK` for blobs with text from the same files;
  - Codeforces-CoTs' 1,485 traces cut at the generation limit (of 24,399; §3.2).
- **Fetch** (`prepare.fetch`): ranged reads of the columns the standardizer reads; for UltraData-Code L3 that is
  `content`, 50.1 of the release's 340.7 GB (Parquet footers). Stack-Edu's metadata is fetched whole and its text by
  blob id from Software Heritage's public S3 bucket. A raw file is written under a temporary name and renamed when
  whole: a Parquet writer closed by an error still writes a valid footer, and one Stack-Edu file so held 1,342,000 of
  its 5,057,204 rows (observed). 4 files are fetched at once (`prepare.FETCHES`): reading one column of 8 L3 files, the
  link gave 8.5 MB/s with 1 file in flight, 11.5 with 2, 12.4 with 4 and 10.5 with 8 (measured).
- **A source read in part fetches its extent** (`prepare.extent`): the fewest Hub files, in the sha1 order of their
  paths (random, and nested as the extent grows), whose text bytes hold its group's fraction of both stages' draws at
  production's bytes per token, over the train split's 99 of 100 residues (§4.4), widened by 6% (`mixture.SPREAD`) for
  a retrained tokenizer's compression. Across tokenizer training mixes "the NSL on any given subset only varies by 5 to
  6 percentage points" (arXiv 2402.01035 §2.3); synthetic code and UltraData-Math L3 enter a sample that held none of
  them, so the bound is borrowed [inference]. `tokenizer.main` reports each group's train tokens over its draw under
  the retrained tokenizer, and `pack.fit` names a source that runs short.

  | Group | Files | Standardized text | Over the draw | Curated train text | Over the draw |
  |---|---|---|---|---|---|
  | Synthetic code | UltraData-Code L3: 28 of 147 (Python), 32 of 167 (C++); Nemotron: 13 of 64 (rewriting), 13 of 64 (transpilation) | 109.67 GB | 1.136 | 107.17 GB | 1.110 |
  | ClimbMix | 187 of 6,543 | 47.13 GB | 1.075 | 46.61 GB | 1.063 |
  | UltraData-Math L3 | 2, 2, 3 and 2 of 50 per format | 11.57 GB | 1.185 | 11.45 GB | 1.173 |

  One Nemotron file is 1.56% of its subset, so the synthetic group holds 13.6% more than its draw where the widening
  and the split ask 7.1% [arithmetic]. Nemotron's copies (§4.3) are short files: 9.61% of rewriting's rows, while its
  curated train text is 98.5% of its standardized text (64.54 of 65.53 GB, the 1% val split included).

## 2. Schema

One standardized Parquet file per Hub file, `CORPUS/standardized/<source>/<k>.parquet`, in its category's schema
(`schema.py`):

| Category | Fields | Split group (§4.4) |
|---|---|---|
| REPOSITORY_FILE | `id`, `repo_name`, `relative_path`, `content`, `language` | `repo_name` |
| EXERCISE | `id`, `content`, `language` | `id` |
| SYNTHETIC_FILE | `id`, `content`, `language` | `id` |
| FUNCTION | `id`, `repo_name`, `content`, `language` | `repo_name` |
| CONVERSATION | `id`, `problem_id`, `content` | `problem_id` |
| NOTEBOOK | `id`, `content`, `language` | `id` |
| TEXT | `id`, `content` | `id` |

- **`id` is the hex SHA-1 of `content`'s UTF-8 bytes,** the final text: the same in every rebuild, and exact
  deduplication is a `GROUP BY id`.
- **`problem_id` is Codeforces-CoTs' `id`.** Both subsets hold the same 8,133 problems (one Python and two C++ rows
  each), and no problem's `id` appears among any row's `aliases` (measured), so `id` alone keeps a problem on one side
  of the split.
- **`language` is the source's,** a Linguist name for code.
- **A category** (`schema.Category`) also fixes the split's group and whether its content is one whole program text
  (`code`: repository files, synthetic files and functions), which in Python must parse (§4.2).
- **Not stored:** publisher and blob ids; quality scores; timestamps, sizes and counts; serialized duplicates
  (`full_content`, `raw_content`, `messages`); L3's `task`, `analysis`, `solution` and `test`; Nemotron's `prompts`
  and seed fields; Codeforces-CoTs' statement fields, tests and `finish_reason`.
- **Records:** `CORPUS/manifest.json` holds each source's repo, revision, Hub files and rows (`prepare.py`), and at
  publication the code's commit and each split's shards, rows and text bytes (`publish.push`); `CORPUS/packed.json`
  holds each stage's rows and weights (`pack.py`).

## 3. Standardization

- **Rows without text are dropped:** a null at standardization (§1), an empty text at deduplication (§4.3).
- **Every text has LF line ends and no leading U+FEFF** (`prepare.lines`). CR LF and a lone CR become LF and every
  leading mark is dropped, as Python and C++ read a source file (Python Language Reference §2.1.2 and §2.1.4; C++
  [lex.phases] phases 1 and 2); both drop one mark and reject a second, and 19 Stack-Edu Python and 44 C++ files begin
  with two or three (measured). In other text the CR and the mark come from its platform and encoding. Measured on the
  raw files: 152,139 CR LF files and 2,157 leading marks in Stack-Edu Python's first file, and 866 of Kaggle's 580,195
  notebooks with a CR.
- Stack-Edu's `relative_path` is its `path` without the leading "/".
- **Otherwise every text is as published.** Kaggle notebooks are the publisher's Markdown: "we start with the notebook
  title, dataset description when available and put the notebook (converted to a Python script) in a code block"
  (card).

### 3.1 UltraData-Code L3: the publisher's `content`

**`content` is the task followed by its solution,** the task as a code comment, unfenced text or Markdown. The release
holds 20,750,202 Python and 22,600,686 C++ exercises, 54.1 and 55.4 GB of `content`, median 2,023 and 1,932 bytes
(measured on all 147 and 167 files).
- The formats are 50 / 25 / 25, as the report trains: "examples are randomly assigned to code comment, no fence, and
  Markdown formats with proportions of 50%, 25%, and 25%, respectively" (App. C.3); measured on the first file of each
  language: 0.50 / 0.25 / 0.25 (Python), 0.499 / 0.248 / 0.252 (C++).
- It is the variant the report measured best: in 10B-token runs, "UltraData-Code-L3 reaches 39.84 on EvalPlus and
  28.45 on MultiPL-E", against 37.49 and 25.28 with analysis and tests added (§3.2.2).
- **Its analysis is left out:** prose is already dense in `content`. Measured share of bytes in comments and docstrings
  (Python's `tokenize`; a C++ lexer for `//` and `/* */`), on 5,000 random files per repository corpus and on 800
  Python and 480 C++ L3 exercises:

| Text | Python | C++ |
|---|---|---|
| Repository files (Stack-Edu, Stack-v3-L2, UltraData-Code L2) | 21.7–25.3% | 15.9–20.8% |
| L3 solutions alone | 41.4% | 26.7% |
| L3 `content` (task and solution) | 61.2% | 54.1% |
| L3 with its analysis added | 75.2% | 72.8% |

- **No execution filter.** Each of the 800 Python and 480 C++ exercises was run against its own `test` (Python 3.12;
  g++ 13 at C++20, solution and test in one translation unit; no network, CPU and memory limits), failing tests rerun
  one assert at a time:

| Outcome | Python | C++ |
|---|---|---|
| Every assert passes | 378 (47.3%) | 231 (48.1%) |
| Fails, passing at least half of its asserts | 343 (42.9%) | 151 (31.5%) |
| Passes fewer than half | 57 (7.1%) | 9 (1.9%) |
| The test does not parse / the program does not compile | 10 (1.3%) | 80 (16.7%), at least 10 for a missing Eigen or `solution.h` |
| Does not run, timeout, crash, no asserts | 12 (1.5%) | 9 (1.9%) |

  A failing exercise misses few asserts (median 80% and 82% of its 10 passed), and the test can be the wrong side:
  `count_prefix_suffix_pairs(["ab", "ba", "aba", "bab"]) == 1` asserts what the stated task makes 0. Filtering on the
  tests would drop about half of L3, correct solutions included, and the report's gains are on the unfiltered release.
- **L3 in pretraining is this design's choice.** The report says "L3 provides task-oriented data for mid-training or
  SFT" (§2), and its evidence is continued pretraining from "the same 1B checkpoint, obtained after approximately 1.3T
  tokens of pre-training" (§3.1). The from-scratch evidence is in §5.2.

### 3.2 Codeforces-CoTs

- **Text:** `prompt` followed by `generation`; the prompt ends with a line break, and `<think>`, the reasoning and the
  final code stay one sequence.
- **Finished traces only: `finish_reason = 'stop'`.** Measured on the raw subsets:

| | Rows | Reach `</think>` | Code block after it |
|---|---|---|---|
| Python, `stop` | 7,771 | 100% | 100% |
| Python, `length` | 362 (4.45%) | 32.6% | 28.7% |
| C++, `stop` | 15,143 | 99.99% | 99.94% |
| C++, `length` | 1,123 (6.90%) | 34.4% | 30.0% |

  Finished traces are 1.079 GB, 0.305B tokens at 3.536 bytes per token.
- **Correctness is the card's:** "We did not filter the traces for correctness, but found that around 84% of the
  Python ones pass the public tests"; that covers the Python rows only.
- **Read at 32K** (§5.1). On production's compiled train split, 72.95% of documents, 90.4% of the tokens, exceed 8K
  tokens, and at 8K 48.0% of the tokens fall in pieces without the problem statement; at 32K, 0.02% of documents
  (0.04% of tokens) exceed the context, and the median document is 14,303 tokens.

### 3.3 Nemotron-Pretraining-Code-v2: rewritten and transpiled source files

**`content` is a whole source file an LLM wrote from a GitHub file:** "Using Qwen3 32B, we rephrased all of our raw
Python source code using a combination of the Style-Guided Code Rewriting (SGCR) and Self-Contained Optimization
Rewriting (SCOR) prompts", and "We applied this Python to C++ transpilation procedure to all Python source files in our
source-code corpus" (arXiv 2512.20848 §2.2.2). Measured on a row group of each (20,000-row samples):

| Subset | Hub rows (64 files) | Text (Parquet footers) | Starts with a code fence | Parses (`ast.parse`, 5,000 rows) | Production's bytes per token |
|---|---|---|---|---|---|
| Rewriting (Python; prompts: NVIDIA's 68.8%, SGCR 15.6%, NVIDIA's and SCOR 15.6%) | 99,900,825 | 322.7 GB | 0.0% | 99.7% | 3.909 |
| Transpilation (C++, from Python) | 29,401,922 | 114.3 GB | 0.0% | — | 3.651 |

- The files are code without prose around them, so they are SYNTHETIC_FILE rows: the Python 3 check applies to the
  rewritten Python (§4.2), and no path comment precedes them (they record none).
- **Not read:** the GitHub metadata (`repo`, a 7-character `commit_id`, `rel_path`, `language` for 376,752,751 files)
  holds no blob id, so its text is one GitHub request per file; the code-review (85.5M rows, 322 GB), student-teacher
  (44.9M, 118 GB) and question-answering (390.7M, 997 GB) dialogues have no measured pretraining effect.
- **License:** the NVIDIA Data Access Agreement for Model Training; a model trained on it "may be subject to
  redistribution and use requirements in the Qwen License Agreement" (card).

### 3.4 CodeXGLUE code-to-text (Python): functions with their docstrings

**`content` is the function's `code`, which holds its docstring** after the `def` line, the layout of HumanEval's
prompts. The pairs are CodeSearchNet's GitHub functions, documentation "truncated to the first full paragraph",
"(near) duplicate functions" removed (arXiv 1909.09436 §2); CodeXGLUE drops examples whose code does not parse, whose
documentation is shorter than 3 or longer than 256 tokens, holds special tokens such as URLs, or is not English
(arXiv 2102.04664 §3.6, §3.9). Measured on the Hub's Python splits:

| | Train (read) | Validation, test (not read) |
|---|---|---|
| Pairs; repositories | 251,820; 11,116 | 13,914, 14,918; 622, 623 |
| Code | 223.0 MB, median 623 bytes | 12.9, 14.0 MB |
| Production's tokenizer (5,000-row samples) | 4.028 bytes per token: 55.4M tokens | 3.3M, 3.4M tokens |
| Parses as Python 3 (5,000 rows) | 99.0% | 99.0%, 98.9% |

- `code_tokens` and `docstring_tokens` are not text (tokens without whitespace, and the docstring's first sentence).
- **Largely new text:** of 20,000 random train pairs, 25.0% come from a (repository, path) that a repository source
  holds, and 15.6% appear verbatim inside that file (measured).
- **License:** C-UDA (card); used for research.

## 4. Curation

`curate` runs the Python 3 check with repair, exact deduplication, the split and the shards
(`CORPUS/curated/<source>/<split>/<k>.parquet`); every output, the shards' row groups included, is a function of the
standardized files. A repository shard's row group holds whole samples, at least 65,536 files (`curate.SAMPLE_ROWS`;
the published shards', the same rows in the same order, followed the shuffle's files instead). The curated corpus is
verified on every row before publication: 178,887,291 rows; no empty text, CR or leading U+FEFF; every `id` its
text's sha1; no text twice across all sources and splits; a parse sample of 101,528 checked Python files, all
parsing; one schema per source; each source's rows its input less its duplicates.

### 4.1 No generated, vendored or C++ syntax filter

- **Generated files: none are left.** StarCoder2 removes files go-enry's `is_generated` marks and files with
  "auto-generated" and similar markers in their first 5 lines (arXiv 2402.19173 §2.1). Measured on 20,000 random files
  per repository corpus: `is_generated` marks 0.01% of Stack-Edu Python and 0% of the other five; the markers add none.
- **Vendored files: kept.** go-enry's `is_vendor` marks 0.12–2.67% of files (200,000 per corpus): committed virtual
  environments and bundled libraries (`third_party/`, `deps/`), but also a project's own `.github/`, `cache/` or
  `external/`, as it reads the path only. 96% of UltraData-Code L2's vendored files, and 78–91% of
  UltraData-Stack-v3-L2's, have another exact copy (measured), which deduplication removes. No paper filters them.
- **C++: no syntax filter.** tree-sitter-cpp 0.23.4, the packer's grammar, fails on 5.5–12.5% of C++ files (20,000 per
  corpus); in 400 failing files per corpus the first error is on a preprocessor line in 14–18% and on a macro-like call
  in 12–24%, and most of the rest is valid code after unexpanded macros, Qt's `public slots:` or C++/CLI's `gcnew`. A
  grammar without the preprocessor would drop valid code.
- **No line-length or character-class rules.** DeepSeek-Coder's ("average line length exceeding 100 characters or a
  maximum line length surpassing 1000 characters", "fewer than 25% alphabetic characters", §2.1) would drop at most
  0.02% of each repository corpus and 1.86% of Kaggle's notebooks, 90% of those for a prose paragraph on one Markdown
  line (measured, 100,000 rows per corpus).

### 4.2 Python 3 check, with repair

Every Python file, a repository's, a synthetic one (§3.3) or a function (§3.4), must parse with `ast.parse` (Python
3.12). A file that fails is repaired by the smallest set of these steps that makes it parse (`curate.repaired`: of the
steps that change it, every subset by size, then in step order), and the repaired text is kept:
1. Stack v3's placeholders `<EMAIL>`, `<KEY>`, `<NAME>` and `<PASSWORD>` become the bare word.
2. Leading-indentation tabs expand to 8 columns, Python 2's rule (Python 2 Language Reference §2.1.8).
3. Every 2to3 fixer of `fissix.fixes`, the maintained fork of lib2to3, which Python 3.12 lacks.

Applied cumulatively instead, the steps rewrite more than a file needs: of 539 and 424 files 2to3 repaired in 15,000
rows each of UltraData-Stack-v3-L2 and UltraData-Code L2 Python, 514 and 397 parse after 2to3 alone, while 103 and 93
also had their tabs expanded (inside triple-quoted strings too) and 94 and 25 their placeholders rewritten (measured).

| Source | Standardized rows | Kept, as is or repaired | Dropped |
|---|---|---|---|
| Stack-Edu Python | 25,285,896 | 24,792,202 (98.05%) | 493,694 (1.95%) |
| UltraData-Stack-v3-L2 Python | 24,671,724 | 24,298,490 (98.49%) | 373,234 (1.51%) |
| UltraData-Code L2 Python | 72,844,227 | 72,194,279 (99.11%) | 649,948 (0.89%) |
| Nemotron rewriting | 20,292,444 | 20,237,191 (99.73%) | 55,253 (0.27%) |
| CodeXGLUE Python | 251,820 | 251,728 (99.96%) | 92 (0.04%) |

- On Stack-Edu Python's first file, 88.81% parse as is and 9.00% after 2to3: its Python 2 files are the largest share.
- **Repair leaves artefacts:** of 128 sampled 2to3 repairs, 3 gained `print((a, b))` and 3 `list(d.keys())`;
  placeholders inside strings and comments change too.
- **Precedents drop files with syntax errors:** DeepSeek-Coder (§2.4), Seed-Coder (arXiv 2506.03524 §2.2.1),
  StarCoder2's notebooks (§2.4.2). None ablates the check or repairs Python 2. HumanEval and MBPP are Python 3.

### 4.3 Exact deduplication only

**One row per `id` across every source read and both splits,** the first occurrence kept (`sources.py` order, then
file and row). Measured on this corpus, after the Python check:

| Source | Rows | Copies of an earlier row | Kept |
|---|---|---|---|
| Stack-Edu Python | 24,792,202 | 22,274 (0.09%) | 24,769,928 |
| Stack-Edu C++ | 16,246,723 | 9,447 (0.06%) | 16,237,276 |
| UltraData-Stack-v3-L2 Python | 24,298,490 | 6,567,995 (27.03%) | 17,730,495 |
| UltraData-Stack-v3-L2 C++ | 17,990,102 | 5,494,464 (30.54%) | 12,495,638 |
| UltraData-Code L2 Python | 72,194,279 | 29,690,879 (41.13%) | 42,503,400 |
| UltraData-Code L2 C++ | 23,290,266 | 10,020,093 (43.02%) | 13,270,173 |
| UltraData-Code L3 Python | 3,952,421 | 35 | 3,952,386 |
| UltraData-Code L3 C++ | 4,330,659 | 14 | 4,330,645 |
| Nemotron rewriting | 20,237,191 | 1,945,778 (9.61%) | 18,291,413 |
| Nemotron transpilation | 5,972,251 | 790,924 (13.24%) | 5,181,327 |
| CodeXGLUE Python | 251,728 | 497 (0.20%) | 251,231 |
| UltraData-Math L3, four formats | 3,486,732 | 0 | 3,486,732 |
| Codeforces-CoTs | 22,914 | 0 | 22,914 |
| Kaggle notebooks | 580,195 | 0 | 580,195 |
| ClimbMix | 15,814,656 | 31,118 (0.20%) | 15,783,538 |
| All | 233,460,809 | 54,573,518 (23.38%) | 178,887,291 |

- The six repository corpora repeat heavily: 28.97% of their 178,812,062 rows are copies. The most-copied texts are
  vendored: pip's `_structures.py` (350,881 copies), `intranges.py` (182,110), a `utils.cpp` (158,584) (R5).
- Repetition is what this prevents: "performance of an 800M parameter model can be degraded to that of a 2x smaller
  model (400M params) by repeating 0.1% of the data 100 times" (arXiv 2205.10487).
- **Which copy is kept: the first.** OpenCoder keeps the copy with "the highest star count as well as the latest
  commit time" (arXiv 2411.04905 §2.1.1), but only UltraData-Stack-v3-L2 carries stars. Over the six corpora's
  20,412,334 (repository, language) groups (R5, vendored copies counted):

| Rule | Rows removed as another group's copy | Groups intact | Groups losing some files (rows lost) | Groups losing all |
|---|---|---|---|---|
| keep-first | 17.41% | 85.17% | 7.90% (24.87M) | 6.93% |
| the copy in the group with the most files | 17.22% | 84.94% | 7.21% (21.53M) | 7.85% |

- **No near deduplication.** Near duplicates differ in their code and stay as training text. Beyond exact copies it
  would remove 0.54% / 0.53% (Stack-Edu), 2.09% / 3.67% (UltraData-Stack-v3-L2) and 9.49% / 12.76% (UltraData-Code L2)
  of rows within each corpus, Python / C++ (R1). Set aside: The Stack's near deduplication "improves HumanEval
  performance from 27.21% to 37.00% pass@100" (arXiv 2211.15533 §4.2, 350M); SantaCoder's further one "improves
  HumanEval performance by 1-3% and MBPP by 1-4%" (arXiv 2301.03988 §6.1).
- Near duplicates can sit on both sides of the split; Nemotron's files record no repository, so a validation
  repository's file can have a rewritten copy in training.

### 4.4 Split

A group goes to validation when `md5_number(group) % 100 = 0` (`schema.SPLIT`, production's split), the group being
§2's. A repository's group, and a function's, is its lowercased name: GitHub's names are case-insensitive, and of
4,146,969 names shared by UltraData-Stack-v3-L2 Python and C++ and UltraData-Code L2 Python, 63 differ only in case, 2
of them on both sides of production's case-sensitive split (measured). Deduplication runs over both splits first.

Measured on the curated shards: 0.92–1.04% of each source's rows are validation rows (CodeXGLUE 0.71%: its validation
holds whole repositories); no repository of the seven sources that name one, no Codeforces problem and no text is on
both sides.

## 5. Mixture

### 5.1 Stages and shares

| Stage | Tokens | Context | Steps | Learning rate (share of peak) |
|---|---|---|---|---|
| File | 97.0B | 8K | 0–82,231 | 1.0 → 0.055 |
| Repository | 5.6B (5.45%: Qwen2.5-Coder's 300B of 5.2T + 300B, §3.2.1–3.2.2) | 32K | 82,231–86,975 | 0.055 → 0 |

| File-stage group | Sources | Share | Tokens | Supply (production's tokenizer) | Read |
|---|---|---|---|---|---|
| Repository code | Stack-Edu, UltraData-Stack-v3-L2, UltraData-Code L2 | the rest of code's 87%: ≈59.4% | ≈57.6B | more than 88.8B (`nanochat/data`'s curation) | once, with the 32K stage's ≈4.6B |
| Synthetic code | UltraData-Code L3 (Python, C++), Nemotron rewriting (Python) and transpilation (C++) | 30% of code: 26.1% | 25.3B | 143.41B (Parquet footers): L3 14.14B and 15.42B, rewriting 82.55B, transpilation 31.31B | 17.65% (§1); by the tokens held: L3 9.4% and 10.3%, rewriting 58.3%, transpilation 22.1% |
| Notebooks | Kaggle notebooks | its supply: ≈1.5% | ≈1.5B | 1.417B | once, whole |
| Functions | CodeXGLUE code-to-text, Python train | its supply: ≈0.057% | ≈55M | 55.4M (§3.4) | once, whole |
| General text | ClimbMix | 10% | 9.7B | ≈372B (232.0 GB in 955 of 6,543 files, scaled; 4.275 bytes per token) | 2.8% with the 32K stage's (§1) |
| Mathematics | UltraData-Math L3, four formats | 3% | 2.9B | 82.5B (260.5 GB at 2.92–3.46 bytes per token) | 3.7% with the 32K stage's (§1) |

| Repository-stage group | Sources | Share | Tokens |
|---|---|---|---|
| Code | repository documents per language (§7.2), the rest of 87%; Codeforces-CoTs' finished traces, read whole | 87% | 4.87B, 0.305B of it the traces |
| General text | ClimbMix | 10% | 0.56B |
| Mathematics | UltraData-Math L3 | 3% | 0.17B |

- **Both stages keep 87 : 10 : 3 in tokens,** 84.5 : 12.2 : 3.3 in bytes [arithmetic, at production's bytes per token:
  code 3.405, text 4.275, mathematics 3.911].
- **The 32K stage keeps text and mathematics,** where production reads code alone (`config.REPO_STAGE`): OpenCoder's
  anneal keeps "84% of the annealing data" from "the original distribution" against "catastrophic forgetting"
  (arXiv 2411.04905 §2.2), and To Code keeps "10% of code in text mixture data … to avoid a full distribution shift"
  (arXiv 2408.10914, footnote 5). On production's runs, after its switch to code alone the step-to-step spread of the
  training loss grows 1.15–2.26 times (measured, `results/lr-curves.jsonl`).
- **Within a group,** sources weigh by their packed rows (`pack.fit`), which follow their tokens. The 32K stage
  reads the trailing residue partitions of the repository sources, the file stage the rest, as import-connected
  samples (production's 8K unit).
- **A group read in part** reads its rows in `id` order, a uniform sample of its extent; a document source's train
  split is 100 shards of 1%, and the file stage packs the leading shards it needs, the 32K stage the next.
- **Every draw fits its rows** (`pack.fit`): each stage's weights are replayed through the loader's `mixture_rows`; a
  source read in part packs another shard while its draw exceeds its rows, and a source read whole weighs as many rows
  less as the replay overdraws it.
- **Token counts before this corpus's tokenizer** use production's bytes per token per source
  (`mixture.BYTES_PER_TOKEN`; measured on its compiled train splits, and on samples of the sources it did not compile):
  Stack-Edu 3.460 / 3.065, UltraData-Stack-v3-L2 3.638 / 3.015, UltraData-Code L2 3.709 / 2.935 (Python / C++);
  UltraData-Code L3 3.826 / 3.593; Nemotron rewriting 3.909, transpilation 3.651; CodeXGLUE 4.028; Kaggle 3.459;
  Codeforces-CoTs 3.536; ClimbMix 4.275; UltraData-Math L3 2.917 (QA), 3.457 (conversation), 3.332 (multi-style),
  3.032 (textbook-exercise).

### 5.2 Why these shares

**Code 87%, text 10%, mathematics 3%.**
- DeepSeek-Coder: "87% source code, 10% English code-related natural language corpus, and 3% code-unrelated Chinese
  natural language corpus" (arXiv 2401.14196 §2). Its text and Chinese are other kinds of text than ClimbMix and
  mathematics, which are production's.
- To Code, or Not To Code? (arXiv 2408.10914, 470M): "the 100% code leads to a 2.6x increase in the code benchmarks
  compared to the 25% code model", and language reasoning shows "a sharp relative drop of 18.3%" at 100% code (§3.3).
- Qwen2.5-Coder 7B (arXiv 2409.12186 Table 3, in the arXiv HTML), Coding Common / BigCodeBench: 100:0:0 scored
  49.8 / 40.3, 70:20:10 48.3 / 38.3, 85:15:5 43.3 / 36.2. "the 7:2:1 ratio outperformed the others" holds for the
  Average column only.
- Code Llama: "adding batches sampled from our natural language dataset improves the performance of our models on
  MBPP" (arXiv 2308.12950 §2.2).
- E1 compared 87 : 10 : 3 against 97 : 3 : 0 at d8 (14.7M scaling parameters, 0.73B tokens, one run per arm):

| Metric | 87 : 10 : 3 | 97 : 3 : 0 | Change |
|---|---|---|---|
| Code bits per byte, 8K | 0.5686 | 0.5618 | −1.2% |
| Code bits per byte, 32K | 0.5541 | 0.5469 | −1.3% |
| Solution bits per byte (HumanEval+, MBPP+, HumanEval-X C++) | 0.6210 | 0.6055 | −2.5% |
| Text bits per byte | 1.2745 | 1.3974 | +9.6% |
| Mathematics bits per byte | 1.2563 | 1.3996 | +11.4% |
| Greedy completions that parse (HumanEval+ / MBPP+) | 78.7% / 24.1% | 70.7% / 17.5% | −7.0 points pooled |

  Ten more points of code bought 1.2–2.5% in code bits per byte and cost 10–11% in text and mathematics and 7 points
  of parse rate; pass@1 is at the floor in both. The split stays at 87 : 10 : 3.

**Synthetic code 30% of code (26.1% of the file stage), weighted by tokens.** No paper sweeps the share of synthetic
code from scratch; the share is the rephrased-data ratio measured from scratch on web text, carried over to code
[inference], and repository code keeps 70% of code.
- Kang et al. (arXiv 2510.01631, more than 1,000 models from scratch): "pre-training on rephrased synthetic data alone
  is not faster than pre-training on natural web texts; while pre-training on 1/3 rephrased synthetic data mixed with
  2/3 natural web texts can speed up 5-10x (to reach the same validation loss) at larger data budgets", and good ratios
  are "empirically converging to ~30% for rephrased synthetic data".
- To Code, at 470M from scratch, tried 10%: "the inclusion of synthetic data leads to relative improvements of 9% on
  natural language reasoning, and 44.9% on code benchmarks compared to the baseline of web-code-only", and "a key
  future lever of improvement is increasing the proportion of such high-quality code data sources" (§3.4).
- Nemotron's rewritten Python alone, annealed for 10B tokens into a 3B model: HumanEval 48.2 and MBPP 64.0, against
  SwallowCode's 43.9 and 43.4 (arXiv 2606.10087 Table 6). Real code keeps MBPP: "Mix1 with no raw code achieves only
  38.6, while Mix3 with 20% raw code recovers to 52.9, approaching the raw-only baseline of 58.7" (RQ3).
- Continued pretraining agrees: SwallowCode "improves pass@1 by +17.0 on HumanEval" against Stack-Edu (arXiv 2505.02881
  §1); NVIDIA found rephrasing "improved downstream code-generation accuracies" and transpiled C++ "improved downstream
  C++ code-generation accuracy" (arXiv 2512.20848 §2.2.2, no numbers); UltraData-Code L2 plus L3 at 1:1 reached
  EvalPlus 57.06 against L2's 46.95 (Table 6).

**Mathematics: UltraData-Math L3.** As the 30% slice of a ~105B-token decay on a 1.3T-token base, L3 gave MBPP 49.27
and HumanEval 32.93 against L1's 44.71 and 29.88 (arXiv 2602.09003 Table 6), the one code-relevant measurement.
Qwen2.5-Coder keeps 10% mathematics (Table 3). Web mathematics is not read.

**Codeforces-CoTs at 32K.** The traces are kept for their stepwise reasoning before code; no paper measures reasoning
traces in pretraining (Seed-Coder uses them "for a distilled model", arXiv 2506.03524 §3.2.1). At 0.305B tokens they
are 5.4% of the 32K stage, read once.

**CodeXGLUE's Python train split, read once.** At 55.4M tokens, a fixed share would repeat it (1% of the stage is 17.5
readings), and "repeating 0.1% of the data 100 times" degrades a model (arXiv 2205.10487). The measured gain from such
pairs is a separate stage: PanGu-Coder (317M) went from HumanEval pass@1 10.37 to 17.07 after a second stage on
docstring–code pairs with the loss on the code only (arXiv 2207.11280 §3.4.4, Table 7). Here the pairs are pretraining
text with a loss on every token; a code-only-loss stage is a training-side option.

**Kaggle notebooks, no GitHub issues.** Production reads Kaggle notebooks, and StarCoder2 trains on them (§2.4.2).
Replacing "15% percentage of code tokens with code-adjacent datasets - including GitHub issues (5%), StackExchange (5%)
and Jupyter Notebooks (5%)" caused a "9.4% drop in code evaluation" (arXiv 2408.10914 §3.4).

**One file-stage mixture, no late phase.** SmolLM2's 135M and 360M models "benefited from a single-stage training
approach with consistently high-quality data" (arXiv 2502.02737 §6). Under a decay to zero, "placing the
highest-quality and most-recent data in the very final phase, while using D2Z … may be suboptimal" (arXiv 2502.15938
§4.2); the file stage's last 10% carries 1.9% of the learning-rate integral [arithmetic, `schedule.py`]. Against: in
UltraData-Math's one comparison (arXiv 2602.09003 Table 7, 1.2B, one run per arm) three consecutive quality tiers beat
the same tokens mixed, MBPP 16.34 against 12.06.

| Model (arXiv) | Size; tokens | Code | Code-related language | Mathematics | General language |
|---|---|---|---|---|---|
| DeepSeek-Coder (2401.14196) | 1.3–33B; 2T from scratch | 87% | 10%: Markdown, StackExchange | 0 | 3% Chinese |
| StarCoder2-3B (2402.19173) | 3B; 3.1T, 4.98 epochs | 84.5–85.8% (Table 4) | 12.1–12.3%: PRs, issues, notebooks, StackOverflow, docs | 0.93% LHQ, with GSM8K | 0 |
| Code Llama (2308.12950) | 7–34B; +500B on Llama 2 | 85% | 8% "discussions about code" | 0 | 7% |
| Qwen2.5-Coder (2409.12186) | 0.5–32B; 5.2T + ≈300B repository-level | 70% | Text-Code Grounding | 10% | 20% |
| To Code (2408.10914) | 470M; 200B | 0–100% sweep | 15% replacing code: −9.4% | — | the rest |

## 6. Fill-in-the-middle

`corpus/pack.py` calls the data crate's pack (`crate/pack.rs`) as `nanochat/data/pack.py` does, without its line
widening:
- **Rate 0.5, at the context level:** each piece of a cut document draws its own FIM decision. On production's packs
  49.96–50.02% of repository pieces and tokens are FIM examples; with repository code at 59.4% of the file stage, FIM
  examples are 29.7% of its tokens [arithmetic].
- **Order:** SPM with probability 0.5 (Bavarian App. D, "SPM variant 2"), otherwise PSM; prefix, suffix and middle
  are encoded separately, and the BOS ends the middle.
- **Middle:** a random character span with probability 0.1, otherwise an AST span from tree-sitter, Single-Node or
  Aligned-Span masking, half each; a piece whose parse has an error takes a random span. tree-sitter-cpp fails on
  5.5–12.5% of C++ files (§4.1), so C++'s random share is 15–21% [arithmetic]; Python files fail in 0–0.03%.
- **No line widening.** Production widens half of the AST middles to whole lines; no paper trains on widened AST
  middles, and a line-level policy trades random-span infilling for line infilling (Bavarian Table 2: single-line /
  multi-line / random-span 0.586 / 0.269 / 0.015 line-level, 0.557 / 0.250 / 0.321 character-level).
- **Determinism:** the draw is keyed by document and piece (ChaCha8), the same in every run and seed.
- **Repository source files only, train split, both stages.** Validation, synthetic code and functions are read left
  to right, as their measured gains were (§5.2); AST-FIM applies "the standard L2R objective without any FIM" to
  natural language (§5.1).

| Source (scale) | Finding |
|---|---|
| Bavarian et al. (2207.14255), 50M–6.9B from scratch | at 50% FIM "the left-to-right test loss is unaffected" (§4.1); "a FIM rate even up to 90% does not cause any degradation in left-to-right capabilities" (§4.2); "context-level FIM yields a consistent and significant improvement over document-level FIM" (§4.4); "always including some character-level random spans" (§8.1) |
| AST-FIM (2506.00204), 1B and 8B | "an equal mix of two masking methods: Single-Node Masking and Aligned-Span Masking" (§3.3); "90% use our AST-FIM, and 10% use Rand-FIM … 50% PSM and 50% SPM mode" (§5.1); AST-FIM keeps "L2R performance nearly identical to the baseline", Rand-FIM "experiences a significant performance decrease" on HumanEval+ and MBPP+ (§6.1) |
| SantaCoder (2301.03988), 1.1B | rate 0.5, context-level, joint SPM+PSM: pass@100 "2-4% lower on HumanEval and 1% lower on MBPP" than no FIM (§6.1) |
| Code Llama (2308.12950), 7B continued | rate 0.9: "The 7B model loses 0.6 percentage points on average across HumanEval and MBPP" (§3.2) |
| DeepSeek-Coder (2401.14196), 1.3B | 100% FIM "results in the weakest code completion capability"; "we ultimately choose the 50% PSM rate" (§3.1.2) |

## 7. Sequence construction

### 7.1 Packing and masking

- **Best-Fit-Decreasing per curated shard,** no token discarded (arXiv 2404.10830 §1). Against concatenate-and-split it
  raised a 7B code model's HumanEval pass@1 from 17.54% to 18.32% and pass@100 from 35.28% to 40.57% (Table 9).
  Production's rows are 99.71–99.98% full at 8K and 99.93–99.99% at 32K (measured).
- **Every piece is its own attention segment:** "Intra Doc improves over Uni Chunk in both the 2K (−1.3 PPL) and 8K
  (−2.0)" on GitHub (arXiv 2402.13991 §3.2, 1.3B).
- **Cuts between whole statements, else after a line end, else between characters,** so cut Python pieces stay
  parseable and their FIM middles can be AST spans. tree-sitter fails on cut pieces of 2,000 Stack-Edu documents per
  language (measured):

| Pieces | Python, statement cuts | Python, every 8,190 tokens | C++, statement cuts | C++, every 8,190 tokens |
|---|---|---|---|---|
| first | 0.0% | 58.5% | 85.8% | 97.2% |
| later | 24.4% | 83.2% | 87.6% | 97.4% |

  The rule helps Python only. Cut pieces hold 14.7% of production's file-stage tokens.
- **Rows hold one corpus; steps hold every group.** A row is packed from one curated shard, Zhao et al.'s "Uni Chunk"
  (lower perplexity than mixed chunks by 0.34 at 2K and 0.21 at 8K, §3.2). The loader assigns run row n to a corpus by
  frac((n + 1) · 0.618…) against the cumulative weights (`loader.mixture_rows`); replayed over production's file stage,
  every step holds 124–126 code, 14–15 text and 4–5 mathematics rows of its 144, where independent draws would leave a
  mathematics corpus out of 10.7–12.1% of steps.

### 7.2 Repository documents for the 32K stage

**One document per (repository, `language`) across the six repository sources,** the repository its lowercased name,
a corpus per language (`repositories_python`, `repositories_cpp`).
- A repository's files, in path order, are cut into import-connected components (`repository_samples`), each in
  dependency order (DeepSeek-Coder's topological sort, §2.2; a cycle in path order); components follow the order of
  their first file. Of a path several sources hold, the first source's file is kept.
- Each file is preceded by a comment line holding its path (`# path`, `// path`; `corpus/sources.texts`), as
  DeepSeek-Coder: "a comment indicating the file's path is added at the beginning of each file" (§2.2). A line break in
  a path is written `\n` or `\r` (495 of UltraData-Code L2 Python's 72.8M paths hold one); a file without a final line
  end gets one, and a C++ file ending in a backslash an empty line too, so the next comment starts its own line.
- The stage reads the fewest trailing partitions whose documents hold its draw (`compile.stage_parts`).

Measured (R3) against production's samples, import-connected components within one corpus:

| 32K stage | Production | Per (repository, language) |
|---|---|---|
| Targets with more than 8K tokens of context | 14.7% | 39.0% |
| Tokens in multi-file documents | 33.5% | 88.6% |

Documents longer than 32K hold 35.6% of the tokens and are cut as in §7.1; 19.7% of repositories appear in more than
one corpus. "Group by language and repo" beat "Group by repo": HumanEval(+) 17.1 (15.9) against 12.8 (10.4)
(Arctic-SnowCoder, arXiv 2409.02326 Table 3, 1.3B).

## 8. Training dynamics

- **Linear decay to zero,** "around 0.84% better than the optimal WSD" at 610M and 80 tokens per parameter
  (arXiv 2502.15938 §4.2).
- **One switch, at 94.55% of steps and 5.5% of peak learning rate;** context, rows per micro-batch and mixture change
  together. Crystal's switch to 63% code "requires about 70% of Phase 2 to restore performance levels"
  (arXiv 2411.04156 §5.1, 7B).
- **The 32K stage learns at a low rate:** 0.30% of the learning-rate integral, 2.7% of peak on average [arithmetic,
  `schedule.py`], where the comparable models' long-context stages run at 6.7–10% of their peak (DeepSeek-Coder §3.6,
  StarCoder2 Tables 7 and 8, Code Llama §2.6). No paper measures it, so the decay stays linear to zero.
- **No document read twice:** `pack.fit` makes every draw fit its rows (§5.1).

## 9. Tokenizer

Production's tokenizer (`nanochat/data/tokenizer.py`): byte-level BPE scored by SA-BPE's F·ln R (a pair's frequency
times the log of the number of authorship units holding it); 32,768 ids (256 bytes, 32,508 merges, BOS and the three
FIM sentinels); the split pattern below (`SPLIT_PATTERN`), cl100k's with numbers in one or two digits; trained on equal
bytes from each curated source.

```
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++|\s*[\r\n]|\s+(?!\S)|\s+
```

**Retrained on this corpus, at the file stage's shares of bytes** (`corpus/tokenizer.sample`: each source's share of
the 8K stage's tokens at its bytes per token, a systematic sample of its curated train split; Codeforces-CoTs, 0.3% of
the run, is not sampled).
- Production's sample holds none of the synthetic code or UltraData-Math L3, 29.1% of the file stage, and 18.27% of
  its bytes are web mathematics this corpus does not read (measured).
- "tokenizers should be trained on the data mix that they are expected to see during training/inference"
  (arXiv 2402.01035 §2.3); "Leaving a language out of tokenizer training raises its BPB in every language we study"
  (arXiv 2610.12144).
- Authorship units: `repo_name` for repository files, the document otherwise. The sample targets 13.38 GB
  (`SAMPLE_BYTES`), production's size.
- Every token figure in this document is in production's tokenizer; a retrained one rescales each by its bytes per
  token.

**Vocabulary: 32,768, as production.** SA-BPE ran at 32,768 ids (arXiv 2604.14053 Table 5); DeepSeek-Coder uses
32,000 (§3.2); StarCoder2 found "increasing the vocabulary size to 100K did not improve performance" (§6.2); Dagan et
al. measured HumanEval 20.52–20.84 from 32K to 256K (§3.2). Of production's 32,508 merges, 3 never occur in its
104.25B compiled train tokens (measured).

**Split pattern and score: decided on the GH200** by `ablations/tokenizer.py` ported to this corpus (`gh200.md`).
Measured properties of the pattern, on 20,000 documents per corpus:
- Numbers come in pairs from the left (`12345` → `12 34 5`). Digit tokens are 8.3–11.5% of UltraData-Math L3's tokens
  and 4.2–7.5% of code's; single digits would add 2.4–3.3% to UltraData-Math L3's tokens.
- A space before a number is its own token: 3.4–4.8% of UltraData-Math L3's tokens, 0.77–1.89% of repository code's.
- A LaTeX command after a space or punctuation is two chunks (` \` then `frac`): binding them saves at most 4.9% of
  mathematics tokens, 0.15% of the run's [arithmetic].
- Indentation of d spaces costs 2 tokens; whitespace-only tokens are 15.6–19.2% of repository code's tokens.

## 10. Pipeline and status

The pipeline splits at the tokenizer. Everything through curation is text and runs on the GB10; the curated shards,
the manifest and a card go to the private Hub dataset `iaroy/nanochat-data` (`publish.push` replaces its tree and
squashes its history). Everything after needs the tokenizer and runs on the GH200 from `publish.pull` (`gh200.md`):
packing cuts pieces at token offsets in rows of `seq_len` tokens, FIM encodes its parts around sentinel ids, and the
32K stage's partitions are counted in tokens (`compile.stage_parts`).

| Stage | Module | Status |
|---|---|---|
| Prepare: fetch, standardize, manifest | `prepare.py` | done: 18 sources (§1) |
| Curate: Python check, deduplication, split, shards | `curate.py` | done: 178,887,291 rows, verified (§4) |
| Publish | `publish.py` | done: `iaroy/nanochat-data` @ `a4ec1a08367ce8ca02690e2b9d79190e0afeaf6c`, one commit and one branch: `<source>/<split>/<k>.parquet` (1,812 shards), the manifest and the card (code `aa37215`); viewer off |
| Tokenizer: decision, sample, training | `tokenizer.py`, `ablations/tokenizer.py` | on the GH200, after the pull |
| Compile: 8K samples, 32K repository documents | `compile.py` | on the GH200 |
| Mixture, pack, loader | `mixture.py`, `pack.py`, `loader.py` | written; run on the GH200 |
| Training reads the corpus; the crate moves to `nanochat/corpus`; `nanochat/data` retired | outside `nanochat/corpus` | when `nanochat/data` is retired |
