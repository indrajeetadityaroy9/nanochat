# Pretraining corpus for Python and C++ code generation: specification

This document specifies `nanochat/corpus` and the parts of `nanochat/data` it replaces: pretraining from scratch for
code generation in Python and C++, from the 20 sources of `nanochat/corpus/sources.py`. Every rule is a decision;
none is optional. Every number is either quoted from a paper, read from the code, or measured on the GB10.

The reference run is production's d24:
- 195.1M scaling parameters, 102.6B tokens, 86,975 steps (`docs/h100-pretraining.md`; `train/schedule.plan`);
- an 8K file stage, then a 32K repository stage over the last 5.45% of steps (82,231–86,975; `config.REPO_SHARE`);
- a linear decay of the learning rate to zero after 40 warmup steps (`train/schedule.py`), and document masking.

Evidence:
- **Papers** are read as markdown with `hf papers read <arXiv id>`; where that markdown drops a table's body, the
  source is named. The UltraData-Code report (arXiv 2610.23960, absent from Hugging Face Papers) is read from
  `tech_report/UltraData_Code_Tech_Report.pdf` in `openbmb/UltraData-Code`.
- **Measured** figures come from runs on the GB10 over `nanochat/data`'s copies of the sources and production's packs,
  and over `nanochat/corpus`'s own fetched, standardized and Hub files (§1). Each states its data and method where it
  is used.
- `[inference]` marks a step no source states.

| Area | Decision | § |
|---|---|---|
| Sources | each read at its latest Hub revision, resolved once per build and recorded; only the columns standardization reads are fetched | 1 |
| Schema | one Parquet schema per document category; `id` is the SHA-1 of the document's text | 2 |
| Text | every text with LF line ends and no leading byte-order mark; repository files as their language reads them; UltraData-Code L3 the publisher's `content`; Nemotron's rewritten and transpiled files as published; Codeforces-CoTs finished traces only; otherwise verbatim | 3 |
| Filters | every Python file parses as Python 3, after repair; no generated, vendored or C++ syntax filter | 4.1, 4.2 |
| Deduplication | exact copies only, across every source read, first occurrence kept | 4.3 |
| Split | production's: 1% of groups (repository, problem or document) held out; repository names case-insensitive | 4.4 |
| Mixture | 87 : 10 : 3 code, text, mathematics in both stages. 8K code: repository files, synthetic code 10% (UltraData-Code L3's exercises, Nemotron's rewritten Python and Python transpiled into C++), Kaggle notebooks whole; 32K code: whole repositories and Codeforces-CoTs | 5 |
| FIM | rate 0.5, joint PSM/SPM, context level, AST-FIM's middles with 10% random spans, repository files only; production's line widening dropped | 6 |
| Sequences | Best-Fit-Decreasing packing, document masking, cuts between statements; 32K documents per (repository, language) across sources | 7 |
| Dynamics | 102.6B tokens; one switch, to 32K; linear decay to zero; no document read twice | 8 |
| Tokenizer | retrained on this corpus's sources at the file stage's shares of bytes; production's vocabulary size, split pattern and score | 9 |
| Publication | the curated text, tokenizer-independent, in the private `iaroy/nanochat-data`; tokenizer, compile, packing and FIM run after the pull, as they need the tokenizer | 10 |

## 1. Sources

| Source (`sources.py`) | Category (§2) | Text | Read |
|---|---|---|---|
| `stack_edu_python`, `stack_edu_cpp` | REPOSITORY_FILE | the Software Heritage blob | yes |
| `ultradata_stack_v3_l2_python`, `ultradata_stack_v3_l2_cpp` | REPOSITORY_FILE | `content` | yes |
| `ultradata_code_l2_python`, `ultradata_code_l2_cpp` | REPOSITORY_FILE | `content` | yes |
| `ultradata_code_l3_python`, `ultradata_code_l3_cpp` | EXERCISE | `content` | in part (§5.1) |
| `nemotron_rewriting_python`, `nemotron_transpilation_cpp`: Nemotron-Pretraining-Code-v2's `synthetic-rewriting`, `synthetic-transpilation` | SYNTHETIC_FILE | `content` | in part (§5.1) |
| `ultradata_math_l3_qa`, `_conversation`, `_multi_style`, `_textbook_exercise` | TEXT | `content` | in part |
| `codeforces_cots`: `solutions_py_decontaminated`, `solutions_short_and_long_decontaminated` | CONVERSATION | `prompt`, then `generation` | yes, at 32K |
| `kaggle_notebooks` | NOTEBOOK | `text` | yes |
| `climbmix` | TEXT | `text` | in part |
| `ultradata_math_l1`, `opc_fineweb_math` | TEXT | `content`, `text` | no (§5.2) |
| `github_issues` | ISSUE | `text` | no (§5.2) |
| Nemotron-Pretraining-Code-v2's GitHub metadata, code-review, student-teacher and question-answering subsets | — | — | no (§3.3) |

**The dataset** (`CORPUS/manifest.json`; Parquet footers for the files still being prepared):

| Source | Hub dataset @ revision | Files read | Standardized rows |
|---|---|---|---|
| `stack_edu_python`, `stack_edu_cpp` | HuggingFaceTB/stack-edu @ `eeec5caa` | 5 of 5, 4 of 4 | preparing |
| `ultradata_stack_v3_l2_python`, `_cpp` | openbmb/UltraData-Stack-v3-L2 @ `62077439` | 45 of 45, 30 of 30 | preparing |
| `ultradata_code_l2_python`, `_cpp` | openbmb/UltraData-Code @ `b20cabcf` | 119 of 119, 31 of 31 | 72,844,227; preparing |
| `ultradata_code_l3_python`, `_cpp` | openbmb/UltraData-Code @ `b20cabcf` | 11 of 147, 13 of 167 | 1,552,729, 1,759,348 |
| `nemotron_rewriting_python`, `nemotron_transpilation_cpp` | nvidia/Nemotron-Pretraining-Code-v2 @ `7b1a453d` | 5 of 64, 5 of 64 | preparing |
| `ultradata_math_l3_qa`, `_conversation`, `_multi_style`, `_textbook_exercise` | openbmb/UltraData-Math @ `fe10db8e` | 2, 2, 3 and 2 of 50 | 1,084,548, 675,242, 686,726, 1,040,216 |
| `codeforces_cots` | open-r1/codeforces-cots @ `39ac85c1` | 9 of 9 | 22,914 |
| `kaggle_notebooks` | HuggingFaceTB/issues-kaggle-notebooks @ `ef882ad1` | 11 of 11 | 580,195 |
| `climbmix` | karpathy/climbmix-400b-shuffle @ `915333b4` | 187 of 6,543 | 15,814,656 |

- **Revision.** Each source is read at its latest revision. `prepare` resolves `dataset_info(repo).sha` once per
  source, so the listing and every file come from one snapshot, and the manifest records it.
- **Fetch.** `prepare.fetch` reads a Hub Parquet file by ranged reads and keeps only the columns its standardizer
  reads. For UltraData-Code L3 that is `content`: 50.1 of the release's 340.7 GB (Parquet footers). Stack-Edu's
  metadata is fetched whole, one file at a time, and its text by blob id from Software Heritage's public S3 bucket.
  - Every raw file is written under a temporary name and renamed when whole: a Parquet writer closed by an error
    still writes a valid footer, and one such Stack-Edu file held 1,342,000 of its 5,057,204 rows (observed).
  - Each process keeps one metadata row group's blobs in flight (about 1,000). Measured per process: 490, 1,665,
    2,085 and 2,692 blobs a second with 64, 256, 1,024 and 4,096 in flight. A request retries for up to 15 minutes;
    obstore's default backoff exhausted its retries on a reset connection within 8.2 seconds and failed the source.
- **4 files at once** (`FETCHES`). Measured on this link, reading one column of 8 L3 files per setting: 8.5 MB/s with
  1 file in flight, 11.5 with 2, 12.4 with 4 and 10.5 with 8.
- **A source read in part fetches only its extent** (`prepare.extent`): the fewest Hub files, in the sha1 order of
  their paths (a random order whose prefixes nest as the extent grows), whose text bytes (Parquet footers) hold its
  group's fraction of both stages' draws at production's bytes per token, over the train split's 99 of 100 residues
  (§4.4), widened by 6% (`mixture.SPREAD`) for a retrained tokenizer's compression. Dagan et al. bound that change:
  "the NSL on any given subset only varies by 5 to 6 percentage points" across tokenizer training mixes, "about 5%" in
  its example (arXiv 2402.01035 §2.3). Their mixes change a domain's share, while the synthetic code (UltraData-Code L3,
  Nemotron's rewriting and transpilation) and UltraData-Math L3 enter a sample that held none of them, so the bound is
  borrowed [inference]. `tokenizer.main` reports each group's train tokens under the retrained tokenizer over its
  draw, and `pack.fit` names a source that runs short.
  Measured on the files fetched (text bytes at production's bytes per token, before the val split; standardized text,
  or Parquet footers where noted):

  | Group | Files | Text | Held over the draw |
  |---|---|---|---|
  | Synthetic code (footers) | UltraData-Code L3: 11 of 147 (Python), 13 of 167 (C++); Nemotron: 5 of 64 (rewriting), 5 of 64 (transpilation) | 42.50 GB | 1.150 |
  | ClimbMix | 187 of 6,543 | 47.13 GB | 1.075 |
  | UltraData-Math L3 | 2, 2, 3 and 2 of 50 per format | 11.57 GB | 1.185 |

  A source's few files make its sample random at the granularity of files: one Nemotron file is 1.56% of its subset,
  which is why the synthetic group holds 15% more than its draw.

## 2. Schema

There is one standardized Parquet file per Hub file, `CORPUS/standardized/<source>/<k>.parquet`, in its category's
schema (`schema.py`):

| Category | Fields | Split group (§4.4) |
|---|---|---|
| REPOSITORY_FILE | `id`, `repo_name`, `relative_path`, `content`, `language` | `repo_name` |
| EXERCISE | `id`, `content`, `language` | `id` |
| SYNTHETIC_FILE | `id`, `content`, `language` | `id` |
| CONVERSATION | `id`, `problem_id`, `content` | `problem_id` |
| ISSUE | `id`, `repo_name`, `content` | `repo_name` |
| NOTEBOOK | `id`, `content`, `language` | `id` |
| TEXT | `id`, `content` | `id` |

- **`id` is the hex SHA-1 of `content`'s UTF-8 bytes,** the final text. It is the same in every rebuild and makes
  exact deduplication a `GROUP BY id`.
- **`problem_id` is Codeforces-CoTs' `id`.** Measured on the raw subsets:
  - both hold the same 8,133 problems, one Python row and two C++ rows each;
  - no problem's `id` appears among any row's `aliases`, so `id` alone keeps a problem on one side of the split.
- **`language` is the source's,** a Linguist name for code.
- **Not stored:**
  - publisher and blob ids;
  - quality and relevance scores;
  - timestamps, sizes, line counts and token counts;
  - serialized duplicates (`full_content`, `raw_content`, `messages`);
  - L3's `task`, `analysis`, `solution`, `test` and format fields;
  - Nemotron's `prompts`, `seed_source`, `seed_language` and `nemotron_pretraining_id`;
  - Codeforces-CoTs' statement fields, tests and `finish_reason` (read by the filter, §3.2).
- **Records:** `CORPUS/manifest.json` holds each source's repo, revision, Hub files fetched and rows (`prepare.record`);
  `CORPUS/packed.json` each stage's rows and corpus weights (`pack.py`); the tokenizer's data key names the compiled
  directories.

## 3. Standardization

- **Rows without text are dropped:** a null at standardization, including a Stack-Edu blob that the bucket does not
  hold or that does not decode with its `src_encoding`; an empty text at deduplication (§4.3).
- **Every text has LF line ends and no leading U+FEFF** (`prepare.lines`). For a repository file that is how its
  language reads it: CR LF and a lone CR are line ends, and a leading U+FEFF is not text (Python Language Reference
  §2.1.2 and §2.1.4; C++ [lex.phases] phases 1 and 2). For any other text the CR and the mark come from its platform
  and encoding. Measured on the raw files: 152,139 CR LF files and 2,157 leading marks in Stack-Edu Python's first
  file, 866 of Kaggle's 580,195 notebooks with a CR (863 CR LF), 80 of 847,872 ClimbMix rows in 10 files, 7 of
  UltraData-Code L3's 2.76M rows and 7 of UltraData-Math L3's; Kaggle's carry Python into the code group.
- Stack-Edu's `relative_path` is its `path` without the leading "/".
- **Otherwise every text is verbatim.** Kaggle notebooks are the publisher's Markdown: "we start with the notebook
  title, dataset description when available and put the notebook (converted to a Python script) in a code block"
  (card).

### 3.1 UltraData-Code L3: the publisher's `content`

**`content` is the task followed by its solution,** with the task as a code comment, as unfenced text or as Markdown.
The release holds 20,750,202 Python and 22,600,686 C++ exercises, 54.1 and 55.4 GB of `content`, with median lengths
of 2,023 and 1,932 bytes (measured on all 147 and 167 files, standardized whole before the extent rule of §1).
- **The format proportions are 50 / 25 / 25,** as the report trains: "examples are randomly assigned to code comment,
  no fence, and Markdown formats with proportions of 50%, 25%, and 25%, respectively" (App. C.3). Measured on the
  first file of each language: Python 0.50 / 0.25 / 0.25, C++ 0.499 / 0.248 / 0.252.
- **It is the variant the report measured best.** In 10B-token runs, "UltraData-Code-L3 reaches 39.84 on EvalPlus and
  28.45 on MultiPL-E", against 37.49 and 25.28 with analysis and tests added (§3.2.2).

**Natural language is already dense in it.** Measured: the bytes of comments and docstrings, counted with Python's
`tokenize` (comments, and string-only statements) and with a C++ lexer for `//` and `/* */` outside literals. Natural
code is 5,000 random files from the first normalized file of each corpus; L3 is the 800 Python and 480 C++ exercises
of the execution sample below, with `task` and `analysis` fetched for the same rows.

| Text | Python | C++ |
|---|---|---|
| Natural repository files (Stack-Edu, Stack-v3-L2, UltraData-Code L2) | 21.7–25.3% | 15.9–20.8% |
| Files with any comment or docstring | 76.7–90.5% | 63.7–77.2% |
| L3 solutions alone | 41.4% | 26.7% |
| L3 `content` (task and solution) | 61.2% | 54.1% |
| L3 with its analysis added | 75.2% | 72.8% |

- L3 mixes code with prose as natural code does, at 2.4–3.4 times the density [arithmetic from the table]. Its
  solutions alone comment more than natural code does.
- Adding the analysis would make L3 three quarters prose. Together with the report's ablation, this settles `content`
  without the analysis.

**No execution filter.** Measured: 4 files per language were sampled, one row group each, giving 800 Python and 480 C++
exercises. Each solution was run against its own `test`:
- Python 3.12, solution then test in one file;
- g++ 13 at C++20, solution and test in one translation unit, because tests often call the function undeclared
  ("assume the function is defined above");
- no network, with CPU and memory limits.

Failing tests were rerun one assert at a time.

| Outcome | Python | C++ |
|---|---|---|
| Every assert passes | 378 (47.3%) | 231 (48.1%) |
| Fails, but passes at least half of its asserts | 343 (42.9%) | 151 (31.5%) |
| Passes fewer than half of its asserts | 57 (7.1%) | 9 (1.9%) |
| The test does not parse / the program does not compile | 10 (1.3%) | 80 (16.7%), at least 10 for a missing Eigen or `solution.h` |
| Other: the solution does not run or parse, a timeout, a crash, no asserts | 12 (1.5%) | 9 (1.9%) |

- A failing exercise misses few asserts: the median one passes 80% (Python) and 82% (C++) of its 10 asserts. Per-file
  pass rates are 41.5–51.0% (Python) and 43.3–53.3% (C++).
- The test can be the side that is wrong. `count_prefix_suffix_pairs(["ab", "ba", "aba", "bab"]) == 1` asserts what
  the stated task makes 0, and 10 Python tests do not parse.
- The release's tests therefore do not tell right solutions from wrong ones. Filtering on them would drop about half of
  L3, correct solutions included, and the report's gains were measured on the unfiltered release.

**L3 in pretraining is this design's choice, not the publisher's.** The report says "L3 provides task-oriented data for
mid-training or SFT" (§2), and all of its evidence is continued pretraining: "start from the same 1B checkpoint,
obtained after approximately 1.3T tokens of pre-training" (§3.1). The from-scratch evidence is To Code's 10% synthetic
code (§5.2).

### 3.2 Codeforces-CoTs

- **Text:** `prompt` followed by `generation`. The prompt ends with a line break, and `<think>`, the reasoning and the
  final code stay one sequence.
- **Finished traces only: `finish_reason = 'stop'`.** Measured on the raw subsets:

| | Rows | Reach `</think>` | Code block after it |
|---|---|---|---|
| Python, `stop` | 7,771 | 100% | 100% |
| Python, `length` | 362 (4.45%) | 32.6% | 28.7% |
| C++, `stop` | 15,143 | 99.99% | 99.94% |
| C++, `length` | 1,123 (6.90%) | 34.4% | 30.0% |

  The truncated rows hold 11.0% of the bytes; finished traces are 1.079 GB, 0.305B tokens at the measured 3.536
  bytes per token.
- **Correctness is the card's:** "We did not filter the traces for correctness, but found that around 84% of the
  Python ones pass the public tests". That covers only the Python rows.
- **Length** (measured on production's compiled train split): 72.95% of documents, holding 90.4% of the tokens, exceed
  8K tokens; at 8K, 48.0% of the tokens fall in pieces cut from their document's start, without the problem statement.
  At 32K, 0.02% of documents (0.04% of tokens) exceed the context; the median document is 14,303 tokens. The traces
  are read in the 32K stage (§5.1, §7.3).

### 3.3 Nemotron-Pretraining-Code-v2: rewritten and transpiled source files

**`content` is a whole source file an LLM wrote from a GitHub file:** "Using Qwen3 32B, we rephrased all of our raw
Python source code using a combination of the Style-Guided Code Rewriting (SGCR) and Self-Contained Optimization
Rewriting (SCOR) prompts", and "We applied this Python to C++ transpilation procedure to all Python source files in our
source-code corpus" (arXiv 2512.20848 §2.2.2). Measured on a row group of each (94,569 and 88,401 rows; 20,000-row
samples):

| Subset | Rows | Text (Parquet footers) | Starts with a code fence | Parses (`ast.parse`, 5,000 rows) | Production's bytes per token |
|---|---|---|---|---|---|
| Rewriting (Python; prompts: NVIDIA's 68.8%, SGCR 15.6%, NVIDIA's and SCOR 15.6%) | 99.9M | 322.7 GB | 0.0% | 99.7% | 3.909 |
| Transpilation (C++, from Python) | 29.4M | 114.3 GB | 0.0% | — | 3.651 |

- The files are code without prose around them, so they are SYNTHETIC_FILE rows: the Python 3 check applies to the
  rewritten Python (§4.2), and no path comment precedes them (they record none).
- **Not read:** the GitHub metadata (`repo`, a 7-character `commit_id`, `rel_path`, `language` for 376,752,751 files)
  holds no blob id, so its text is one GitHub request per file; the code-review (85.5M rows, 322 GB), student-teacher
  (44.9M, 118 GB) and question-answering (390.7M, 997 GB, 39% Python) dialogues have no measured pretraining effect.
- **License:** the NVIDIA Data Access Agreement for Model Training; a model trained on it "may be subject to
  redistribution and use requirements in the Qwen License Agreement" (card).

## 4. Curation

The steps run in order: the Python 3 check with repair, exact deduplication, split.

### 4.1 No generated, vendored or C++ syntax filter

**Generated files: no filter, because none are left.** StarCoder2 removes "files classified as auto-generated by the
is_generated function of go-enry" and files with "auto-generated", "autogenerated", "automatically generated",
"generated automatically" or "this file is generated" in their first 5 lines (arXiv 2402.19173 §2.1). Measured on
20,000 random files of each repository corpus: go-enry's `is_generated` marks 0.01% of Stack-Edu Python and 0% of the
other five, and the five-line markers add none.

**Vendored files: no filter.** go-enry's `is_vendor` matches Linguist's vendor paths and marks 0.12–2.67% of files
(measured, 200,000 per corpus).
- UltraData-Code L2 Python's are mostly committed virtual environments: `env/lib/python3.*/site-packages/...`, pip's
  `_vendor/chardet`, sympy and pandas tests. C++'s are bundled libraries under `third_party/`, `3rdparty/`,
  `extern/` or `deps/`, such as libc++ tests, LLVM and libigl.
- The rule reads the path only, so it also marks a project's own files under `.github/`, `cache/` or `external/`.
- 96% of UltraData-Code L2's vendored files, and 78–91% of UltraData-Stack-v3-L2's, have another exact copy (measured),
  so exact deduplication (§4.3) already keeps one copy of each.
- No paper filters vendored files; StarCoder2's filter is the generated one above.

**C++: no syntax filter.** tree-sitter-cpp 0.23.4, the packer's grammar, fails on 5.5–12.5% of C++ files (measured,
20,000 per corpus). In 400 failing files per corpus, the first error is on a preprocessor line in 14–18% and on a
macro-like call in 12–24%. Most of the rest are valid code following unexpanded macros (`rep(i,h)`), Qt's
`public slots:` or C++/CLI's `gcnew`; some are broken (merge-conflict markers, prose). A grammar without the
preprocessor would drop valid code. Seed-Coder does discard files with Tree-sitter syntax errors (arXiv 2506.03524
§2.2.1).

**No line-length or character-class rules.** DeepSeek-Coder filters "files with an average line length exceeding 100
characters or a maximum line length surpassing 1000 characters" and "files with fewer than 25% alphabetic characters"
(§2.1). Measured on 100,000 rows per corpus, these rules would drop at most 0.02% of the rows of each repository corpus.
In Kaggle notebooks they would drop 1.86% of rows, and 90% of the notebooks with a line over 1,000 characters owe it to
a prose paragraph written as one Markdown line (823 of 912 sampled).

### 4.2 Python 3 check, with repair

Every Python file, a repository's or a synthetic one (§3.3), must parse with `ast.parse` (Python 3.12). Python 2 files
are real code, so a file that fails is repaired before it is dropped, by the smallest set of the steps below that
makes it parse (`curate.repaired`: of the steps that change the file, every subset by size, then in step order), and
the kept text is the repaired one:
1. Stack v3's placeholders `<EMAIL>`, `<KEY>`, `<NAME>` and `<PASSWORD>` become the bare word (`<NAME>` → `NAME`).
2. Leading-indentation tabs are expanded to 8 columns, Python 2's rule: "tabs are replaced (from left to right) by one
   to eight spaces such that the total number of characters up to and including the replacement is a multiple of
   eight" (Python 2 Language Reference §2.1.8).
3. 2to3's fixers are applied: every fixer of `fissix.fixes`. fissix is the maintained fork of lib2to3, which Python
   3.12 lacks.

Measured (R4) over every raw UltraData-Stack-v3-L2 Python row, 24,671,724 rows and 114.0 GB, with the steps applied
cumulatively in this order:

| Parses | Rows | Share | GB |
|---|---|---|---|
| as is | 22,902,492 | 92.83% | |
| after the placeholders | 496,271 | 2.01% | 5.36 |
| after the tabs | 25,915 | 0.11% | 0.13 |
| after 2to3 | 873,799 | 3.54% | 4.06 |
| never: dropped | 373,247 | 1.51% | |

Measured on the first Hub file of the other two Python corpora, fetched through `prepare.fetch`:

| Parses | Stack-Edu Python (5,057,177 rows) | UltraData-Code L2 Python (612,143 rows) |
|---|---|---|
| as is | 88.81% | 96.19% |
| after the placeholders | 0.00% | 0.00% |
| after the tabs | 0.23% | 0.07% |
| after 2to3 | 9.00% | 2.83% |
| never: dropped | 1.95% | 0.90% |

Stack-Edu's text comes from Software Heritage without Stack v3's placeholders, and its Python 2 files are the largest
share: repair keeps 98.05% of its rows against 88.81%.

The cumulative order rewrote more than a file needed: of the files 2to3 repaired, in 15,000 rows each of
UltraData-Stack-v3-L2 and UltraData-Code L2 Python, 103 of 539 and 93 of 424 also had their leading tabs expanded,
including tabs inside triple-quoted strings, and 94 and 25 their placeholders rewritten, while 514 and 397 parse after
2to3 alone (measured). The smallest set keeps those tabs and placeholders, and repairs at least the files the
cumulative order did.

- **By failure class (R4),** repair recovers:
  - 774,851 of 782,378 print statements (99.0%);
  - 576,690 of 859,542 other syntax errors (67.1%);
  - 44,444 of 127,305 indentation or tab errors (34.9%).

  Of the dropped rows, 356,237 are rejected by 2to3's Python 2 grammar.
- **The bare word parses as R4's prefixed identifier did.** Over 3 raw files, 50,838 rows fail with a placeholder;
  33,617 parse after either substitution, with no disagreement (measured).
- **Repair leaves artefacts.** Of 128 sampled 2to3 repairs, 3 gained `print((a, b))` (2to3 keeps Python 2's tuple
  print) and 3 `list(d.keys())`. Substitution applies to the whole file, so placeholders inside strings and comments
  change too.
- **Precedents drop files with syntax errors.**
  - DeepSeek-Coder uses "a compiler and a quality model … This includes code with syntax errors" (§2.4).
  - Seed-Coder "discarded those with syntax errors" (§2.2.1).
  - StarCoder2 drops notebooks "with syntax errors" (§2.4.2).
  - The Stack found "only 0.7% of the files do not compile due to syntax errors" (arXiv 2211.15533 §3).

  None of them ablates the check, and none repairs Python 2. HumanEval and MBPP, the evaluation, are Python 3.

### 4.3 Exact deduplication only

**One row per `id` across every source read and both splits.** `nanochat/data` deduplicates within two universes,
repository files and other documents; this design uses one, a single `GROUP BY id`. The first occurrence in
`sources.py` order is kept, then file and row order within a source, as in `nanochat/data` (keep-first).

Exact copies add no text, and the heavy tail repeats some texts hundreds of thousands of times. Measured (R5) over the
six repository corpora after `nanochat/data`'s normalization: 172,953,953 rows hold 122,801,135 distinct texts, so
29.0% of rows are copies.

| Copies of a text | Rows | Bytes |
|---|---|---|
| 1 | 59.74% | 52.57% |
| 2 | 18.02% | 13.93% |
| 3–4 | 4.87% | 5.30% |
| 5–16 | 4.69% | 8.12% |
| 17–256 | 5.68% | 8.48% |
| 257–4,096 | 3.78% | 7.83% |
| more than 4,096 (216 texts) | 3.24% | 3.77% |

- **The most-copied texts are vendored:** pip's `_structures.py` (350,881 copies), `intranges.py` (182,110), and a
  `utils.cpp` (158,584). Of the texts with exactly two copies, 85% span corpora.
- **Texts with more than 16 copies** hold 26.9% of UltraData-Code L2 Python's rows (38.1% of its bytes) and 12.7% of
  its C++ rows (24.6%). In UltraData-Stack-v3-L2 they hold 0.2–0.3% of rows, and in Stack-Edu 0.02%.
- **Within each corpus,** exact copies are:
  - 0.09% (Stack-Edu Python) and 0.06% (C++);
  - 12.57% / 12.18% (UltraData-Stack-v3-L2);
  - 32.26% / 16.48% (UltraData-Code L2);
  - 0.03% of Codeforces-CoTs and 0.61% of ClimbMix (R1);
  - 306 rows of UltraData-Code L3 Python and 130 of C++ (0.001%), and 4 texts in both languages (measured).
- **Repetition is what exact deduplication prevents:** "performance of an 800M parameter model can be degraded to that
  of a 2x smaller model (400M params) by repeating 0.1% of the data 100 times" (arXiv 2205.10487).

**Which copy is kept.** The literature keeps the copy with the most stars. OpenCoder keeps "the code files with the
highest star count as well as the latest commit time" (arXiv 2411.04905 §2.1.1), and StarCoder2 does the same in
near-deduplication "to preserve repository context" (§3.1). Only UltraData-Stack-v3-L2 carries stars (in
`source_meta.github_metadata`); Stack-Edu and UltraData-Code L2, four of the six sources, carry none. Two rules were
measured over the six corpora's 20,412,334 (repository, language) groups:

| Rule | Rows removed as another group's copy | Groups intact | Groups losing some files (rows lost) | Groups losing all |
|---|---|---|---|---|
| keep-first | 17.41% | 85.17% | 7.90% (24.87M) | 6.93% |
| the copy in the group with the most files | 17.22% | 84.94% | 7.21% (21.53M) | 7.85% |

The largest-group rule spares 0.69% of groups a partial loss; keep-first needs no group sizes. The measurement predates
the removal of the vendored filter; vendored copies are counted.

**No near deduplication.** Near duplicates differ in their code, and they stay as training text.
- What near deduplication would have removed beyond exact copies (R1):
  - within each corpus: 0.54% / 0.53% (Stack-Edu), 2.09% / 3.67% (UltraData-Stack-v3-L2) and 9.49% / 12.76%
    (UltraData-Code L2) of rows, Python / C++;
  - across corpora: 2.9% / 3.7% (UltraData-Stack-v3-L2) and 12.9% / 12.3% (UltraData-Code L2).
- The evidence set aside:
  - The Stack: "the near deduplication process is on top of the exact deduplication" (§4.1), and it "improves
    HumanEval performance from 27.21% to 37.00% pass@100" (§4.2, 350M).
  - SantaCoder: a further near-deduplication of already near-deduplicated data "improves HumanEval performance by 1-3%
    and MBPP by 1-4%" (arXiv 2301.03988 §6.1).
- Near duplicates can sit on both sides of the split; validation loss counts them. Nemotron's rewritten and
  transpiled files record no repository, so a validation repository's file can have a rewritten copy in training.

### 4.4 Split

A group goes to validation when `md5_number(group) % 100 = 0` (`schema.SPLIT`, as `nanochat/data/curate.SPLIT`), the
group being the one in §2's table. A repository's group is its lowercased name: GitHub's names are case-insensitive,
and of 4,146,969 names shared by UltraData-Stack-v3-L2 Python and C++ and UltraData-Code L2 Python, 63 differ only in
case, 2 of them on both sides of production's case-sensitive split (measured). Otherwise this is production's split;
no paper sets its size. Deduplication runs over both splits first, so no validation text has an exact copy in
training.

## 5. Mixture

### 5.1 Stages and shares

The token budget is 102.6B tokens, as production:

| Stage | Tokens | Context | Steps | Learning rate (share of peak) |
|---|---|---|---|---|
| File | 97.0B | 8K | 0–82,231 | 1.0 → 0.055 |
| Repository | 5.6B (5.45%: Qwen2.5-Coder's 300B of 5.2T + 300B, §3.2.1–3.2.2) | 32K | 82,231–86,975 | 0.055 → 0 |

| File-stage group | Sources | Share | Tokens | Supply (production's tokenizer) | Read |
|---|---|---|---|---|---|
| Repository code | Stack-Edu, UltraData-Stack-v3-L2, UltraData-Code L2 | the rest of code's 87%: ≈75.5% | ≈73.2B | more than 88.8B: `nanochat/data`'s curation, near-deduplicated and without repair | once, with the 32K stage's ≈4.6B |
| Synthetic code | UltraData-Code L3 (Python, C++), Nemotron's rewriting (Python) and transpilation (C++) | 10% | 9.7B | 143.41B (Parquet footers at production's bytes per token): L3 14.14B and 15.42B (3.826, 3.593), rewriting 82.55B (3.909), transpilation 31.31B (3.651); the UltraData report's Table 17 gives L3 34.24B in an unstated tokenizer | 6.76%, from their extents (§1); weighted by tokens: L3 9.9% and 10.8%, rewriting 57.6%, transpilation 21.8% |
| Notebooks | Kaggle notebooks | its supply: ≈1.5% | ≈1.5B | 1.417B measured after near deduplication, which removed 5.33% | once, whole |
| General text | ClimbMix | 10% | 9.7B | ≈372B: 232.0 GB of text in 955 of its 6,543 files (measured), times 6,543 / 955, at its measured 4.275 bytes per token | 2.8% with the 32K stage's, from its extent |
| Mathematics | UltraData-Math L3, four formats | 3% | 2.9B | 82.5B measured: QA 22.8B, Conversation 18.3B, Multi-Style 17.4B, Textbook-Exercise 24.0B; 260.5 GB of text (Parquet footers) at 2.92–3.46 bytes per token (20,000-row samples); the paper gives 88B in an unstated tokenizer | 3.7% with the 32K stage's, from its extent |

| Repository-stage group | Sources | Share | Tokens |
|---|---|---|---|
| Code | the repository documents per language (§7.2), the rest of 87%; Codeforces-CoTs' finished traces, 0.305B, their supply read whole (§3.2) | 87% | 4.87B, 6.3% of it the traces |
| General text | ClimbMix | 10% | 0.56B |
| Mathematics | UltraData-Math L3 | 3% | 0.17B |

- **Both stages keep production's 87 : 10 : 3 in tokens,** 84.5 : 12.2 : 3.3 in bytes [arithmetic: each group's share
  times its bytes per token, measured on production's compiled train splits: code 3.405, text 4.275, mathematics
  3.911]. The file stage puts synthetic code and step-by-step mathematics in place of production's web mathematics.
- **The 32K stage keeps text and mathematics** rather than production's code alone (`config.REPO_STAGE`), because the
  papers that change a mixture keep part of the old one. OpenCoder's anneal: "A significant distribution shift can
  lead to catastrophic forgetting in the model's knowledge, and we ensure that 84% of the annealing data comes from the
  original distribution of RefineCode" (arXiv 2411.04905 §2.2). To Code keeps "10% of code in text mixture data during
  continual pre-training of code-initialized models … to avoid a full distribution shift" (arXiv 2408.10914, footnote
  5). Measured on production's runs (`results/lr-curves.jsonl`): after its switch to code alone, the step-to-step
  spread of the training loss grows 1.15–2.26 times.
- **Within a group,** sources are weighted by their packed rows (production's `mixture_weights`), which follow their
  tokens. Within repository code, the 32K stage reads a partition of repositories: curated shards are residues of
  `md5_number(repo_name)`, and the stage reads the trailing ones; the file stage reads the rest, as each source's
  import-connected samples (production's 8K unit).
- **A group read in part** reads its rows in `id` order, a uniform sample of its extent because `id` is a hash. A
  document source's train split is 100 shards of 1% (`schema.SPLIT`), as a repository source's partitions are; the
  file stage packs the leading shards it needs, the 32K stage the next.
- **Every draw fits its rows** (`pack.fit`): each stage's weights are replayed through the loader's `mixture_rows`
  over its rows; a source read in part packs its next shard while its draw exceeds its rows, and a source read whole
  ("supply": Kaggle notebooks, Codeforces-CoTs) weighs as many rows less as the replay overdraws it.

### 5.2 Why these shares

**Code 87%, text 10%, mathematics 3%.**
- DeepSeek-Coder: "87% source code, 10% English code-related natural language corpus, and 3% code-unrelated Chinese
  natural language corpus" (arXiv 2401.14196 §2). That grounds the code share. Its 10% (Markdown and StackExchange) and
  3% (Chinese) are other kinds of text than this design's ClimbMix and mathematics, which are production's and kept on
  E1 below. The shares are not ablated.
- To Code, or Not To Code? (arXiv 2408.10914), at 470M (§2.3):
  - "the 100% code leads to a 2.6x increase in the code benchmarks compared to the 25% code model" (§3.3);
  - language reasoning shows "a sharp relative drop of 18.3% when the model is trained on 100% code compared to a model
    with no code" (§3.3).
- Qwen2.5-Coder 7B (arXiv 2409.12186 Table 3; the body is in the arXiv HTML, not the hf markdown), Coding Common and
  BigCodeBench:
  - 100:0:0 scored 49.8 / 40.3;
  - 70:20:10 scored 48.3 / 38.3;
  - 85:15:5 scored 43.3 / 36.2.

  The text calls the last arm "85:10:5" and says "the 7:2:1 ratio outperformed the others", which holds for the
  Average column only.
- Code Llama: "adding batches sampled from our natural language dataset improves the performance of our models on
  MBPP" (arXiv 2308.12950 §2.2).
- E1 (measured) compared production's 87 : 10 : 3 against 97 : 3 : 0. It ran d8 (14.7M scaling parameters, 0.73B
  tokens), one run per arm at seed 0:

| Metric | 87 : 10 : 3 | 97 : 3 : 0 | Change |
|---|---|---|---|
| Code bits per byte, 8K | 0.5686 | 0.5618 | −1.2% |
| Code bits per byte, 32K | 0.5541 | 0.5469 | −1.3% |
| Solution bits per byte (HumanEval+, MBPP+, HumanEval-X C++) | 0.6210 | 0.6055 | −2.5% |
| Text bits per byte | 1.2745 | 1.3974 | +9.6% |
| Mathematics bits per byte | 1.2563 | 1.3996 | +11.4% |
| Greedy completions that parse (HumanEval+ / MBPP+) | 78.7% / 24.1% | 70.7% / 17.5% | −7.0 points pooled |

  Ten more points of code bought 1.2–2.5% in code and solution bits per byte. Removing text and mathematics cost
  10–11% in their bits per byte and 7 points of parse rate. Both arms' pass@1 is at the floor (at most 5 of 164
  HumanEval+ tasks). With one seed per arm, the parse difference is the only evidence on generation, so the split
  stays at 87 : 10 : 3.

**Synthetic code 10%: UltraData-Code L3, Nemotron's rewriting and transpilation, weighted by their tokens.**
- At 470M, "despite a small weighting of 10%, the inclusion of synthetic data leads to relative improvements of 9% on
  natural language reasoning, and 44.9% on code benchmarks compared to the baseline of web-code-only"
  (arXiv 2408.10914 §3.4), "from the perspective of training from scratch". That measurement sets the share.
- The report's L2 plus L3 at 1:1 reached EvalPlus 57.06 against L2's 46.95 (Table 6), "replacing half of L2 with L3".
  That is continued pretraining on a 1.3T-token base, so it does not set a from-scratch share.
- Rewritten code: continual pretraining of Llama-3.1-8B for 50B tokens on SwallowCode (SGCR and SCOR rewriting)
  "improves pass@1 by +17.0 on HumanEval and +16.1 on HumanEval+ compared to an equivalent budget using Stack-Edu";
  SGCR alone "results in an approximate 10-point decrease on the MBPP benchmark", which SCOR addresses
  (arXiv 2505.02881 §1, §3.3.1). Its authors, under "Language generality": "our evaluations are confined to continual
  pre-training with a 50 billion token budget". NVIDIA, for the subsets read here: "we also found that using LLMs to
  rephrase source code improved downstream code-generation accuracies", and transpiled C++ "improved downstream C++
  code-generation accuracy" (arXiv 2512.20848 §2.2.2; no numbers given).

**Mathematics: UltraData-Math L3.**
- The one code-relevant measurement favours the synthetic tier. As the 30% slice of a ~105B-token decay on a 1.3T-token
  base, UltraData-Math L3 gave MBPP 49.27 and HumanEval 32.93, against L1's 44.71 and 29.88 (arXiv 2602.09003
  Table 6).
- Precedents: Qwen2.5-Coder keeps 10% mathematics (Table 3); Granite adds "mathematical web text" in phase 2
  (arXiv 2405.04324 §2.4, §4.1).
- Web mathematics (UltraData-Math L1, OPC FineWeb Math) is not read.

**Codeforces-CoTs in the 32K stage, by decision.**
- The traces are kept for their stepwise reasoning before code. No paper measures reasoning traces in pretraining:
  Seed-Coder uses `open-r1/codeforces-cots` "for a distilled model" (arXiv 2506.03524 §3.2.1), not its base model.
- At 0.305B tokens they cost 5.4% of the 32K stage, read once and whole (§3.2).

**Kaggle notebooks as production; GitHub issues not added.**
- Production reads Kaggle notebooks in its code group, and both E1 arms include them. StarCoder2 trains on them: "We
  include Python notebooks released by the Kaggle platform" (§2.4.2).
- Production does not read GitHub issues. The only controlled measurement is negative: replacing "15% percentage of
  code tokens with code-adjacent datasets - including GitHub issues (5%), StackExchange (5%) and Jupyter Notebooks
  (5%)" caused a "9.4% drop in code evaluation", measured in Python (arXiv 2408.10914 §3.4). It mixes notebooks in, so
  it is not evidence against notebooks alone.

**One file-stage mixture, no late phase.**
- SmolLM2 ran data ablations "at the target training length" for its 135M and 360M models, the scale closest to
  this one, and found that they "benefited from a single-stage training approach with consistently high-quality data"
  (arXiv 2502.02737 §6).
- Under a linear decay to zero, the end of a run carries less weight: "both Constant and 10×, but not D2Z, overfit to
  the very final portion", so "placing the highest-quality and most-recent data in the very final phase, while using
  D2Z (e.g., as in dubey2024llama), may be suboptimal" (arXiv 2502.15938 §4.2). Here the file stage's last 10% (steps
  74,008–82,231) carries 1.9% of the learning-rate integral [arithmetic from `schedule.py`].
- Against: three consecutive 40B-token quality tiers beat the same tokens mixed in UltraData-Math's one comparison
  (arXiv 2602.09003 Table 7, 1.2B from scratch, one run per arm): MBPP 16.34 against 12.06, HumanEval 3.05 against
  2.44, code average 9.70 against 7.25, overall +1.49 points. Its schedule is not stated.

Comparable models:

| Model (arXiv) | Size; tokens | Code | Code-related language | Mathematics | General language |
|---|---|---|---|---|---|
| DeepSeek-Coder (2401.14196) | 1.3–33B; 2T from scratch | 87% | 10%: Markdown, StackExchange | 0 | 3% Chinese |
| StarCoder2-3B (2402.19173) | 3B; 3.1T, 4.98 epochs | 84.5–85.8% (Table 4) | 12.1–12.3%: PRs, issues, notebooks, StackOverflow, docs | 0.93% LHQ, with GSM8K | 0 |
| Code Llama (2308.12950) | 7–34B; +500B on Llama 2 | 85% | 8% "discussions about code" | 0 | 7% |
| Qwen2.5-Coder (2409.12186) | 0.5–32B; 5.2T + ≈300B repository-level | 70% | Text-Code Grounding | 10% | 20% |
| To Code (2408.10914) | 470M; 200B | 0–100% sweep | 15% replacing code: −9.4% | — | the rest |
| UltraData-Code report (2610.23960) | 1B; +10B and +100B on 1.3T | L2 + L3, 1:1 | — | — | — |

## 6. Fill-in-the-middle

`corpus/pack.py` calls the data crate's pack (`crate/pack.rs`) as `nanochat/data/pack.py` does, with one change (the
line widening):
- **Rate 0.5, at the context level.** Each piece of a cut document draws its own FIM decision as it is cut; a FIM
  piece's window leaves room for its three sentinels. Measured on production's packs: 49.96–50.02% of repository pieces
  and tokens are FIM examples, in both stages. With repository code at 75.5% of the file stage, FIM examples are 37.8%
  of its tokens [arithmetic].
- **Order:** SPM with probability 0.5 (Bavarian App. D, "SPM variant 2"), otherwise PSM. Prefix, suffix and middle are
  encoded separately, and the BOS ends the middle.
- **Middle:**
  - a random character span with probability 0.1;
  - otherwise an AST span from tree-sitter: Single-Node or Aligned-Span masking, half each;
  - a piece whose parse has an error takes a random span.
- **No line widening.** Production widens half of the AST middles to whole lines (`LINE_SHARE` 0.5), matching the
  middles of line-level infilling benchmarks; this design drops it.
  - No paper trains on widened AST middles.
  - Bavarian Table 2 shows the trade a line-level policy makes. Single-line / multi-line / random-span infilling:
    line-level spans 0.586 / 0.269 / 0.015, character-level 0.557 / 0.250 / 0.321.
  - AST-FIM: "A more careful test of random FIM, token FIM, and line FIM with tuned lengths is needed" (§7).
- **Determinism:** the draw is keyed by document and piece (a ChaCha8 key), so it is the same in every run and seed.
- **Eligibility: repository source files, train split, both stages.** Validation is left to right. Grounding:
  - AST-FIM: "For the natural language data, we only apply the standard L2R objective without any FIM" (§5.1);
  - StarCoder2 applies FIM to source code, not to "the repository metadata" (§5);
  - the UltraData-Code report never mentions FIM, so L3's measured gains are left to right, as are the rewriting and
    transpilation gains above (§5.2); synthetic code is read left to right.
- **C++ middles are random more often.** tree-sitter-cpp fails on 5.5–12.5% of whole C++ files (§4.1), so C++'s random
  share there is 15–21% against AST-FIM's 10% [arithmetic]; cut C++ pieces fail far more often (§7.1). Python files
  fail in 0–0.03%.

| Source (scale) | Finding |
|---|---|
| Bavarian et al. (2207.14255), 50M–6.9B from scratch | at 50% FIM "the left-to-right test loss is unaffected" (§4.1); "a FIM rate even up to 90% does not cause any degradation in left-to-right capabilities" (§4.2); "50% probability in PSM mode and with 50% probability in SPM mode" (§3); "context-level FIM yields a consistent and significant improvement over document-level FIM" (§4.4); "always including some character-level random spans" (§8.1) |
| AST-FIM (2506.00204), 1B and 8B | "an equal mix of two masking methods: Single-Node Masking and Aligned-Span Masking" (§3.3); "Within the FIM portion, 90% use our AST-FIM, and 10% use Rand-FIM. For both AST-FIM and Rand-FIM, we use 50% PSM and 50% SPM mode" (§5.1, rate 0.7); "the AST-FIM model achieves L2R performance nearly identical to the baseline L2R model", while Rand-FIM "experiences a significant performance decrease on the L2R benchmarks HumanEval+ and MBPP+" (§6.1) |
| SantaCoder (2301.03988), 1.1B | context-level, character-level, rate 0.5, joint SPM+PSM: pass@100 "2-4% lower on HumanEval and 1% lower on MBPP" than no FIM (§6.1) |
| Code Llama (2308.12950), 7B continued | rate 0.9, "to documents that are not cut across multiple model contexts only" (§2.3); "The 7B model loses 0.6 percentage points on average across HumanEval and MBPP pass@1, pass@10 and pass@100" (§3.2) |
| DeepSeek-Coder (2401.14196), 1.3B, Python | PSM only; 100% FIM "results in the weakest code completion capability"; "we ultimately choose the 50% PSM rate" (§3.1.2) |

## 7. Sequence construction

### 7.1 Packing and masking

- **Packing: Best-Fit-Decreasing per curated shard,** no token discarded: "Best-Fit-Decreasing (Eilon & Christofides,
  1971), an approximation algorithm" (arXiv 2404.10830 §1). Against concatenate-and-split, best-fit packing raised a 7B
  code model's HumanEval pass@1 from 17.54% to 18.32% and pass@100 from 35.28% to 40.57% (Table 9). Measured fill of
  production's rows (slots taken by pieces and their BOS over all slots): 99.71–99.98% at 8K, 99.93–99.99% at 32K.
- **Masking: every piece is its own attention segment.** "Intra Doc improves over Uni Chunk in both the 2K (−1.3 PPL)
  and 8K (−2.0)" on GitHub: "code pre-training is more adversely affected by the distraction of unrelated context"
  (arXiv 2402.13991 §3.2, 1.3B).
- **Cuts between whole statements, else after a line end, else between characters.** Ding et al. cut at L tokens,
  "the minimally required truncation" (§3); the statement rule is production's own. It keeps cut Python pieces
  parseable, so their FIM middles can be AST spans. Measured on 2,000 cut Stack-Edu documents per language, tree-sitter
  fails on:

| Pieces | Python, statement cuts | Python, every 8,190 tokens | C++, statement cuts | C++, every 8,190 tokens |
|---|---|---|---|---|
| first | 0.0% | 58.5% | 85.8% | 97.2% |
| later | 24.4% | 83.2% | 87.6% | 97.4% |

  Long C++ documents fail tree-sitter wherever they are cut (§4.1), so the rule helps Python only. Cut pieces hold
  14.7% of production's file-stage tokens: 16.3% of the repository corpora's (12.4–19.6% per corpus), 8.1% of
  Kaggle's and 3.9% of ClimbMix's (measured on production's packs).
- **Rows hold one corpus; steps hold every group.** A row is packed from one curated shard of one corpus, which is
  Zhao et al.'s "Uni Chunk": "This helps to avoid packing documents from different distributions (such as code and
  news text) together", lowering average perplexity against mixed chunks by 0.34 (2K) and 0.21 (8K) (§2.1, §3.2).
  The loader assigns run row n to a corpus by frac((n + 1) · 0.618…) against the mixture's cumulative weights
  (`loader.mixture_rows`). Replayed over production's file stage (11.84M rows), every step holds 124–126 code, 14–15
  text and 4–5 mathematics rows of its 144; independent draws would leave a mathematics corpus out of 10.7–12.1% of
  steps. No paper compares this interleaving with random sampling.

### 7.2 Repository documents for the 32K stage

**One document per (repository, `language`) across the six repository sources,** the repository its lowercased name
(§4.4), after deduplication, a corpus per language (`repositories_python`, `repositories_cpp`), as a pack call parses
one grammar.
- A repository's files, in path order, are cut into import-connected components (`repository_samples`), each in
  dependency order, DeepSeek-Coder's topological sort (§2.2), a cycle in path order; the components follow one another
  in the order of their first file.
- Of a path that several sources hold in one repository, the first source's file is kept (`sources.py` order,
  deduplication's); of a path one source holds twice, its first row.
- Each file is preceded by a comment line holding its path, `# path` or `// path` (`corpus/sources.texts`): "a comment
  indicating the file's path is added at the beginning of each file" (DeepSeek-Coder §2.2). A line break in a path is
  written `\n` or `\r`, so the comment stays one line: 495 of UltraData-Code L2 Python's 72.8M paths hold one, 125 of
  UltraData-Stack-v3-L2 Python's 24.7M and 370 of its C++ 18.0M (measured). A file without a final line end gets one
  (21.9% of 915,104 UltraData-Code L2 Python files), so the next file's comment starts a line; a C++
  file whose last line ends in a backslash gets an empty line too, as phase 2 would splice the next file's comment
  onto it (20 of 1,799,017 UltraData-Stack-v3-L2 C++ files, measured).
- The stage reads the fewest trailing partitions whose documents hold its draw, counted after the path rule
  (`compile.stage_parts`).

Measured (R3) against the current samples, which are import-connected components within one corpus:

| 32K stage | Current | Per (repository, language) |
|---|---|---|
| Targets with more than 8K tokens of context | 14.7% | 39.0% |
| Tokens in multi-file documents | 33.5% | 88.6% |

- Documents longer than 32K hold 35.6% of the tokens; they are cut as in §7.1.
- 19.7% of repositories appear in more than one corpus, which grouping across sources reunites.

Evidence:
- "Group by language and repo" beat "Group by repo": HumanEval(+) 17.1 (15.9) against 12.8 (10.4), and MBPP(+) 33.9
  (27.8) against 30.7 (25.9) (Arctic-SnowCoder, arXiv 2409.02326 Table 3, 1.3B).
- DeepSeek-Coder's repository-level pretraining as a whole helped cross-file completion "in the Java, TypeScript, and
  C# languages", not Python (EM 16.14 against 16.02 without it; §4.3, Table 7).
- StarCoder2 orders them randomly instead: "we ultimately arranged them in a random order within the same repository"
  (§5.1).

### 7.3 Long traces

Codeforces-CoTs is read at 32K, where 99.98% of its documents fit whole (§3.2). At 8K, 48.0% of its tokens fall in
pieces that do not hold their problem statement.

## 8. Training dynamics

- **Schedule: linear decay to zero,** "around 0.84% better than the optimal WSD" at 610M and 80 tokens per parameter
  (arXiv 2502.15938 §4.2).
- **One switch, at 94.55% of steps and 5.5% of peak learning rate.** Context, rows per micro-batch and mixture change
  together; the 32K stage adds Codeforces-CoTs and keeps text and mathematics (§5.1). Crystal's switch from "95%
  natural language and 5% code" to 63% code "requires about 70% of Phase 2 to restore performance levels" on HellaSwag
  (arXiv 2411.04156 §5.1, 7B), and it concludes that "a more gradual transition to more code data may foster smoother
  adaptation".
- **The 32K stage learns at a low rate.** It carries 0.30% of the learning-rate integral, at 2.7% of peak on average
  [arithmetic from `schedule.py`]. The long-context stages of the comparable models run at 6.7–10% of their peak:
  DeepSeek-Coder keeps its final 10% (§3.6); StarCoder2 trains at 3×10⁻⁵ against 3×10⁻⁴ (Tables 7 and 8); Code Llama
  at 2e−5 against 3e−4 (§2.6). No paper measures the learning rate of a long-context stage, so the decay stays linear
  to zero.
- **Budget: 102.6B tokens, no document read twice.** `pack.fit` makes every draw fit its rows (§5.1): a source read
  in part packs more of its shards, a source read whole weighs less, and repository code holds more than its draw.

## 9. Tokenizer

Production's tokenizer (`nanochat/data/tokenizer.py`):
- byte-level BPE scored by SA-BPE's F·ln R: a pair's frequency times the log of the number of authorship units holding
  it, so a pair seen in one unit scores 0;
- 32,768 ids: 256 bytes, 32,508 merges, and 4 special tokens (BOS and the three FIM sentinels);
- the split pattern below (`SPLIT_PATTERN`), cl100k's with numbers in one or two digits:

  ```
  '(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++|\s*[\r\n]|\s+(?!\S)|\s+
  ```

- trained on equal bytes from each curated source (`SAMPLE_SHARES = "equal"`).

**Retrain it on this corpus's sources, at the file stage's shares of bytes.**
- A source's share of the sample's bytes is its share of the file stage's tokens at its bytes per token, a
  systematic sample of its curated train split (`corpus/tokenizer.sample`). The file stage is 94.5% of the run's
  tokens; Codeforces-CoTs, read at 32K only, is 0.3% of them [arithmetic: 0.305B of 102.6B] and is not sampled.
- Measured: production's sample is 72.75 / 8.98 / 18.27% code, text and mathematics by bytes.
  - 18.27% comes from OPC FineWeb Math and UltraData-Math L2, which this design does not read.
  - None comes from the synthetic code (UltraData-Code L3, Nemotron's rewriting and transpilation) or UltraData-Math
    L3, 13% of the file stage. Production's tokenizer reads them at 3.59–3.91 and 2.92–3.46 bytes per token, against
    4.13 for OPC FineWeb Math and 3.70 for UltraData-Math L2 (measured).
- Leaving data out costs it: "Leaving a language out of tokenizer training raises its BPB in every language we
  study", and "both equal weighting and an allocation inverting the shares with respect to the language model training
  data increase their BPB" (arXiv 2610.12144, abstract; natural languages).
- "tokenizers should be trained on the data mix that they are expected to see during training/inference"
  (arXiv 2402.01035 §2.3). Magikarp advises "ensuring that it is representative of model training data"
  (arXiv 2405.05417 §5).
- Authorship units are `repo_name` for repository files and the document itself otherwise. The TEXT schema keeps no
  URL, and production used hosts as units for OPC FineWeb Math only.
- The sample targets 13.38 GB (`SAMPLE_BYTES`), production's sample size (its curated splits' Parquet footers), so
  the two vocabularies differ in their sample's composition only.
- Every token figure in this document is in production's tokenizer. A retrained tokenizer rescales each by its bytes
  per token.

**Vocabulary: 32,768, as production.**
- SA-BPE ran at 32,768 ids with 100M-parameter models (arXiv 2604.14053 Table 5); DeepSeek-Coder uses "a vocabulary
  size of 32,000" (§3.2).
- StarCoder2 found that "increasing the vocabulary size to 100K did not improve performance" (§6.2).
- Dagan et al. measured HumanEval 20.52–20.84 from 32K to 256K, "correlation coefficient of −0.13 with a p-value of
  0.87" (§3.2, Table 3).
- Measured: of production's 32,508 merges, 3 never occur in its 104.25B compiled train tokens and 14 occur fewer than
  1,000 times. Magikarp finds under-trained tokens "typically around 0.1–1% of the vocabulary".

**Split pattern and score: production's.** `ablations/tokenizer.py` tests them, with four arms at d16 and equal bytes;
it has not been run. It reads `nanochat/data`'s corpus and production's equal-bytes sample, so it decides production's
pattern and score, not this corpus's sample; its `sa-bpe-mixture` vocabulary, the sample at the mixture's shares, is
described there and not trained on. Measured properties of the pattern, on 20,000 documents per corpus:
- **Numbers come in pairs from the left** (`12345` → `12 34 5`), and all 110 one- and two-digit tokens exist. Digit
  tokens are 8.3–11.5% of UltraData-Math L3's tokens, 4.2–7.5% of code's (4.4–4.9% of UltraData-Code L3's) and 2.0% of
  text's.
  - Single digits, as Llama 2 (§2.2) and DeepSeek LLM (§2.1) split them, would add a token per two-digit token:
    2.4–3.3% more of UltraData-Math L3's tokens, 0.6–0.8% of UltraData-Code L3's, 0.91–3.14% of production's corpora.
  - Right-to-left chunks help arithmetic at inference: "up to 20% higher" for 3-digit chunks in GPT-3.5 and GPT-4
    (arXiv 2402.14903). No paper compares chunkings in training.
- **A space before a number is its own token,** because `\p{N}{1,2}` takes no leading space; GPT-2's pattern does
  (` ?\p{N}+`). Such tokens are 3.4–4.8% of UltraData-Math L3's tokens, 2.21% of Codeforces-CoTs', 2.1–2.2% of
  UltraData-Code L3's, 0.77–1.89% of repository code's and 0.84% of ClimbMix's. That bounds the saving if a number took
  its space; no paper measures the choice.
- **A LaTeX command after a space or punctuation is two chunks:** the punctuation clause takes the space and backslash
  (` \`, `$\`, `(\`), the letter clause the name (`frac`), so no merge joins them. In UltraData-Math L3 (QA,
  Textbook-Exercise; 5,000 documents each) commands occur 4.5–4.9 times per 100 tokens, 90% of them cut so, and ` \` is
  2.2–4.0% of the four formats' tokens. Binding them saves at most a token per command: 4.9% of mathematics tokens,
  0.15% of the run's [arithmetic]. No paper measures LaTeX chunking.
- **Indentation of d spaces costs 2 tokens:** d − 1 spaces, then the last space joined to the statement's first token
  (99.2–99.9% of indented lines in the repository corpora). Whitespace-only tokens are 15.6–17.6% of Python
  repository tokens, 18.1–19.2% of C++'s, and 15.2% and 12.9% of UltraData-Code L3 Python's and C++'s.

## 10. Changes to the code

**The pipeline splits at the tokenizer.** Everything through curation is text: standardization, the Python 3 check
and repair, deduplication, the split, the shards and the import-connected samples. Everything after needs the
tokenizer: packing places pieces in rows of `seq_len` tokens and cuts them at token offsets (`crate/pack.rs`), FIM
encodes prefix, suffix and middle separately around sentinel ids, and the 32K stage's partitions are counted in tokens
(`compile.stage_parts`). `publish.push` puts the curated shards, the manifest and a card in the dataset, replacing its
tree and history; `publish.pull` fetches a revision where the tokenizer is trained and decided.

| Change | Where | Status |
|---|---|---|
| Fetch only the columns standardization reads; EXERCISE holds `id`, `content`, `language` | `corpus/prepare.py`, `corpus/schema.py` | done (3f9f335) |
| Codeforces-CoTs: `finish_reason = 'stop'` rows only; each source's revision, files and rows in the manifest; extents of the sources read in part | `corpus/prepare.py`, `corpus/mixture.py` | done (a64c637) |
| Python 3 check with repair (fissix in `pyproject.toml`); exact deduplication by `id`, keep-first across sources; the split; curated shards | `corpus/curate.py` | done (4985b09) |
| Tokenizer sample from this corpus's sources at the file stage's shares; authorship units from the schemas | `corpus/tokenizer.py` | done (4985b09) |
| Curated documents in tokens; repository documents per (repository, language) across sources | `corpus/compile.py` | done (4985b09, d068c80) |
| Mixture of §5.1 in both stages; stage-disjoint packs; FIM without line widening; weights fitted to the packed rows | `corpus/mixture.py`, `corpus/pack.py`, `corpus/loader.py` | done (d068c80) |
| Document sources in 100 train shards; 32K partitions counted after the path rule; the extent over the train split; byte shares in the tokenizer sample; empty texts dropped | `corpus/curate.py`, `corpus/compile.py`, `corpus/prepare.py`, `corpus/tokenizer.py` | done |
| Every text normalized (LF, no leading BOM); minimal Python repair; case-insensitive repositories; escaped path comments; atomic, resumable raw files | `corpus/prepare.py`, `corpus/curate.py`, `corpus/sources.py`, `corpus/compile.py` | done (be961a4, af2a7fb) |
| Nemotron-Pretraining-Code-v2's rewriting and transpilation in the synthetic group; SYNTHETIC_FILE | `corpus/sources.py`, `corpus/schema.py`, `corpus/mixture.py`, `corpus/prepare.py`, `corpus/curate.py` | done (2d6bcc0) |
| Every read source prepared and curated on the GB10; the curated shards published | `prepare`, `curate`, `publish` | running |
| The tokenizer decided and trained, the corpus compiled and packed, after the pull | `tokenizer`, `compile`, `pack` | after the pull |
| Training reads the corpus (`config.py`, `train/`, `eval/`, `inference/`); the crate moves to `nanochat/corpus`; `nanochat/data` retired | outside `nanochat/corpus` | when `nanochat/data` is retired |
| Packing, masking, cuts, schedule | — | unchanged |
