# Pretraining data pipeline

## Datasets

The pretraining dataset consists of multiple sources categorized by their primary content and intended purpose.
**Natural code** is sourced from [UltraData-Code](https://huggingface.co/datasets/openbmb/UltraData-Code) (L2: ~400B
tokens), [UltraData-Stack-v3-L2](https://huggingface.co/datasets/openbmb/UltraData-Stack-v3-L2) (245.5B tokens across
11 programming languages, selected from The Stack v3) and
[Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) (~125B tokens of educational code selected from
The Stack v2) for general code pretraining. **Natural mathematics** is
sourced from [UltraData-Math](https://huggingface.co/datasets/openbmb/UltraData-Math) (L1: 170.5B tokens; L2: 33.7B
tokens) and [OpenCoder FineWeb Math](https://huggingface.co/datasets/OpenCoder-LLM/opc-fineweb-math-corpus) (5.24M
mathematics-related documents) for mathematical pretraining. **Reasoning and problem-solving** data is sourced from
[Codeforces-CoTs](https://huggingface.co/datasets/open-r1/codeforces-cots) (close to 100K DeepSeek-R1 reasoning traces
and solutions for about 10K competitive programming problems; its 11 overlapping configurations hold 253,876 records),
UltraData-Code L3 (~150B tokens of synthetic programming exercises) and UltraData-Math L3 (88B tokens of mathematical
exercises and reasoning). **Software engineering text and code explanations** are sourced from
[Issues-Kaggle-Notebooks](https://huggingface.co/datasets/HuggingFaceTB/issues-kaggle-notebooks) (~12.7B tokens: 11B of
GitHub issue discussions and 1.7B of Kaggle notebooks). Finally, **general English-language pretraining** is supported
by [ClimbMix-400B-Shuffle](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle), providing English-language
text to complement the code, mathematics and reasoning datasets.

Sizes are the dataset cards' (read 2026-10-22 at `main` and at the revisions `nanochat/data/sources.py` pins; both
state the same figures). Token counts are in each publisher's tokenizer.

### The sources

| Category | Dataset | What it is (card) | Size (card) |
|---|---|---|---|
| Natural code | UltraData-Code L2 | L0 archives about 192M GitHub repositories; L1 standardizes and near-deduplicates them; L2 keeps algorithmically relevant code; 11 languages (cpp, cs, go, java, js, php, py, r, rb, rust, sh) | ~400B tokens |
| Natural code | UltraData-Stack-v3-L2 | The Stack v3 (`HuggingFaceCode/stack-v3-train`: a GitHub crawl, MinHash-deduplicated, PII redacted with placeholders) selected by UltraData's L2 classifier; 127,133,027 files, 20.10% of the source; Python, Java, JavaScript, C++, PHP, Ruby, Shell, C#, Go, Rust, R | 245.50B tokens (table total; its text says "approximately 240B") |
| Natural code | Stack-Edu | StarCoder2Data (The Stack v2) filtered by 15 language-specific StarEncoder classifiers trained on Llama3-70B-Instruct annotations, threshold 3 of 5 (Java 2); the 15 largest languages; the card holds Software Heritage ids only, contents downloaded from Software Heritage's S3 bucket | 125B tokens |
| Natural mathematics | UltraData-Math L1 | Common Crawl math pages parsed (L0), heuristically filtered and deduplicated (L1) | 170.5B tokens, 85.6M documents |
| Natural mathematics | UltraData-Math L2 | L1 selected by a quality classifier; released as `L2-preview` | 33.7B tokens; L2-preview 14.98M documents |
| Natural mathematics | OpenCoder FineWeb Math | FineWeb pages recalled by three rounds of fastText (OpenCoder, arXiv 2411.04905) | 5,241,900 documents (no token count for math alone; the card's 55B covers code- and math-related data together) |
| Reasoning and problem-solving | Codeforces-CoTs | up to five DeepSeek-R1 reasoning traces per problem on about 10K Codeforces problems; configurations include solutions with and without editorials, Python-only, decontaminated variants, checkers/interactors and test-input generators | close to 100K samples; the 11 configurations, which overlap, total 253,876 records |
| Reasoning and problem-solving | UltraData-Code L3 | L2 implementations rewritten into structured programming exercises; 11 languages | ~150B tokens |
| Reasoning and problem-solving | UltraData-Math L3 | L2 rewritten and synthesized (Q&A, dialogue, multi-style, textbook) by Qwen2.5-72B-Instruct, Qwen3-32B and GLM-4.5 | 88B tokens, 81.4M documents |
| Software engineering text and code explanations | Issues-Kaggle-Notebooks | StarCoder2's extras (`bigcode/starcoder2data-extras`): GitHub issue threads from GH Archive (`issues`, 15,549,682 rows) and Kaggle notebooks from Meta Kaggle Code rendered as Markdown with fenced Python (`kaggle`, 580,195 rows) | 11B + 1.7B tokens |
| General English | ClimbMix-400B-Shuffle | a re-upload of NVIDIA's ClimbMix (now `nvidia/Nemotron-ClimbMix`): Nemotron-CC and smollm-corpus text clustered into topics, filtered by classifiers and mixed by CLIMB (arXiv 2504.13161) [inference: the mirror's card documents no upstream] | ClimbMix: 400B tokens |

### What the pipeline takes

`nanochat/data/sources.py` pins each dataset at a Hub revision and reads the files of its Python and C++ subsets (code),
its English text or its math; the rows below are counted from the Parquet footers (Hub), the normalized store and the
curated store (`data/manifest.json`), 2026-10-22.

| Source (`sources.py`) | Dataset @ revision | Files read | Rows: Hub → normalized → curated |
|---|---|---|---|
| `stack_edu_python` | Stack-Edu @ `eeec5ca` | `Python/*.parquet` (ids; contents from Software Heritage) | 25,286,019 → 22,427,994 → 22,241,647 |
| `stack_edu_cpp` | Stack-Edu @ `eeec5ca` | `Cpp/*.parquet` | 16,246,746 → 16,246,723 → 16,083,730 |
| `ultradata_stack_v3_l2_python` | UltraData-Stack-v3-L2 @ `d6bdb3e` | `data/UltraData-Stack-v3-L2/python/*.parquet` | 24,671,724 → 22,902,492 → 15,740,023 |
| `ultradata_stack_v3_l2_cpp` | UltraData-Stack-v3-L2 @ `d6bdb3e` | `data/UltraData-Stack-v3-L2/cpp/*.parquet` | 17,990,102 → 17,990,102 → 11,128,831 |
| `ultradata_code_l2_python` | UltraData-Code @ `525c9ac` | `data/UltraData-Code-L2/py/*.parquet` | 72,844,227 → 70,096,376 → 25,722,110 |
| `ultradata_code_l2_cpp` | UltraData-Code @ `525c9ac` | `data/UltraData-Code-L2/cpp/*.parquet` | 23,290,266 → 23,290,266 → 7,400,637 |
| `kaggle_notebooks` | Issues-Kaggle-Notebooks @ `ef882ad` | `kaggle/*.parquet` | 580,195 → 580,195 → 549,280 |
| `codeforces_cots` | Codeforces-CoTs @ `39ac85c` | `solutions_py_decontaminated` (Python, 8,133), `solutions_short_and_long_decontaminated` (C++, 16,266) | 24,399 → 24,399 → 24,392 |
| `climbmix` | ClimbMix-400B-Shuffle @ `915333b` | `shard_*.parquet`, the leading 955 files | 80,764,928 → 80,764,928 → 76,705,778 |
| `opc_fineweb_math` | OpenCoder FineWeb Math @ `858b10c` | `data/train-*.parquet` | 5,241,900 → 5,241,900 → 5,055,210 |
| `ultradata_math_l2` | UltraData-Math @ `fe10db8` | `data/UltraData-Math-L2-preview/*.parquet`, the leading 94 files | 9,400,000 → 9,400,000 → 1,695,275 |

Not read by `sources.py`: UltraData-Code L3, UltraData-Math L1 and L3, the `issues` subset of Issues-Kaggle-Notebooks,
the other Codeforces-CoTs configurations, and every language but Python and C++.

### UltraData-Code: which level

The levels (UltraData report, arXiv 2610.23960 §2): **L0** the latest default-branch commit of about 192M public
GitHub repositories (July 2025 cutoff); **L1** extension-aware cleaning, then MinHash near deduplication within each
file extension, keeping ⌈n/2⌉ files of each near-duplicate component ranked by stars, creation time, length and id
(3.98B files, 44.05% removed); **L2** files embedded by Qwen3-Embedding-0.6B and kept when a role head does not
exclude them, an algorithmic-relevance head clears its threshold (0.85 for Python) and a quality score reaches 3 of 5
(about 12.23% of L1's files); **L3** L2's algorithmic files rewritten by DeepSeek-v4-flash into a task, an analysis,
a solution and tests. Only L2 (~400B tokens) and L3 (~150B) are released; L0 and L1 are not downloadable.

| | Python | C++ |
|---|---|---|
| L1 files after deduplication (Table 14) | 575,184,540 | 80,822,744 |
| L2 tokens, snapshot before decontamination (Table 15) | 102.57B | 31.94B |
| L3 tokens, task + solution / all fields (Table 17) | 16.24B / 38.88B | 18.00B / 49.09B |
| L2 here after curation (`data/manifest.json`, production's tokenizer) | 21.08B | 4.32B |

The publisher's comparison: a 1B model continually pretrained from a 1.3T-token checkpoint, each corpus at the same
token budget, EvalPlus average greedy pass@1 (and MultiPL-E):

| Corpus | Python, 10B tokens (Table 4) | 11 languages, 10B (Table 5) | 11 languages, 100B (Table 6) |
|---|---|---|---|
| UltraData-Code L1 | 25.22 | 23.62 (15.25) | 26.97 (18.16) |
| Stack-Edu | 34.67 | 27.05 (17.33) | 37.09 (21.44) |
| UltraData-Stack-v3-L2 | 41.87 | — | — |
| UltraData-Code L2 | 43.13 | 31.42 (20.38) | 46.95 (27.15) |
| L2 and L3, 1:1 | 46.43 | 39.84 (28.45) | 57.06 (39.54) |

**L2, as `sources.py` reads it.** L0 and L1 are not released, and at equal tokens L1 scores 18 points below L2 on
Python (L0 is the uncleaned crawl L1 is made from). The evidence is continual pretraining of a 1B model, not training
from scratch at d24 [inference: the ranking of selected over unselected code carries; the size of the gain need not].
**L3 is not pretraining data here:** its records have the benchmarks' own form (a task, a solution, tests), its gain
is measured by those benchmarks, the authors place it in mid-training or SFT ("L3 provides task-oriented data for
mid-training or SFT", §2), and the release does not say which benchmarks its 13-gram decontamination (§3.1) covered;
adopting it would be a mid-training experiment with its own contamination check. L2's release repeats itself: 34.8% of
its Python rows are exact copies of another of its rows (MinHash keeps ⌈n/2⌉ of a component, so identical files
survive in components of three or more [inference]); curation keeps 25,722,110 of its 70,096,376 normalized Python
rows.

### UltraData-Math: which level

The levels (the card at `fe10db8`; UltraData-Math, arXiv 2602.09003 §2.2): **L0** Common Crawl pages parsed by a
math-aware HTML parser (magic-html with w3m), MathML, KaTeX and AsciiMath turned into LaTeX; **L1** heuristic cleaning
and document-level deduplication (method unstated); **L2** L1 selected by an embedding classifier distilled from large
models' annotations (model, threshold and share kept unpublished; its selector's README reads "Coming soon"), released
as `L2-preview`, which no source defines; **L3** synthetic rewriting in four formats (question and answer, multi-turn
conversation, multi-style rewriting, textbook exercises) by Qwen2.5-72B-Instruct, Qwen3-32B and GLM-4.5, its seed data
Nemotron-CC-Math, MegaMath and FineMath (the card's acknowledgements), not stated to be L2. Mainly English and Chinese,
the split unstated.

| Level (Hub folder at `fe10db8`) | Files, Parquet size | Columns | Size (card) |
|---|---|---|---|
| L1 (`data/UltraData-Math-L1/CC-MAIN-*`: 99 Common Crawl snapshots, 2014 to 2024) | 1,485, 367 GB | `content`, `meta` | 170.5B tokens, 85.6M documents |
| L2-preview (`data/UltraData-Math-L2-preview`) | 138, 63.3 GB | `content`, `quality_label` | 33.7B tokens, 14.98M documents |
| L3 (`data/UltraData-Math-L3/{QA,Conversation,Multi-Style,Textbook-Exercise}-Synthetic`) | 4 × 50, 122.6 GB | `uid`, `content` | 88B tokens, 81.4M documents |

The publisher's comparison: MiniCPM-1.2B continually pretrained (30% the level's data, 70% general data, equal tokens
per level), greedy accuracy:

| Level | 10B tokens (Table 5): MATH500, GSM8K | ~100B-token decay (Table 6): MATH500, GSM8K, MBPP, HumanEval |
|---|---|---|
| L1 | 14.80, 32.75 | 27.78, 54.66, 44.71, 29.88 |
| L2 | 15.60, 34.04 | 29.20, 52.92, 44.50, 32.32 |
| L3 | 20.20, 41.47 | 37.02, 61.79, 49.27, 32.93 |

At the same ~100B-token budget the card puts its result (the L3 row) above FineMath-4+ (29.84, 56.25), MegaMath-Web-Pro
(32.12, 56.71) and Nemotron-CC 4plus (33.40, 58.45) on MATH500 and GSM8K.

Measured here on the copy `nanochat/data` fetched (its first 94 L2-preview files; 2026-10-22): 59.7% of the rows are
exact copies of another row, spread across files (median 3 copies, 280 at most), so curation kept 1,695,275 of
9,400,000; in 80,000 rows of four files, 0.1% hold Chinese (over 20 CJK characters in the first 2,000), 51.6% LaTeX or
dollar-delimited math, a median of 4,873 bytes.

**L2-preview for pretraining, deduplicated.** The publisher calls L2 "the core resource for the pre-training phase" and
L1 the broad pretraining pool; L2's gain over L1 is small and not consistent (MATH500 +0.8 and +1.4, GSM8K +1.3 and
−1.7), so the choice rests on the selection, not on a measured gain. Its unique rows, about 40% of 14.98M documents
[estimate: 59.7% exact copies in the first 94 files], cover a code model's math share (production's 3% of 102.6B
tokens is 3.1B); L1, five times larger and deduplicated by the publisher, is the reserve should math's share grow
beyond it. **L3 is not pretraining data here,** as UltraData-Code's L3: synthetic question-and-answer, conversation
and exercise formats whose gains are measured on the benchmarks they resemble (it also lifted MBPP by 4.56 and
HumanEval by 3.05 points over L1), placed by its authors in mid-training, SFT and RL (§2.2.4), with no decontamination
statement; adopting it would be a mid-training experiment with its own contamination check.

## Source catalog

The 19 sources of `nanochat/corpus/sources.py`, by Hub dataset: every column of the files a source reads, its type, its
meaning and what standardization keeps (`nanochat/corpus/prepare.py`). Meanings quote the dataset's card at the pinned
revision, or the cards and papers it defers to (named); `[inference]` marks a meaning read from the type and the
values. Every standardized document also gets an `id`, the sha1 of its text, computed locally; a row whose text column
is null is dropped. Rows and nulls are counted from the Parquet footers. Each sample is the first row of the source's
first file (or the datasets-server's first row, where the Hub's main is the pinned revision); long values are cut (…),
their full length in characters in parentheses.

### HuggingFaceTB/stack-edu @ `eeec5ca`

`stack_edu_python` (`Python/*.parquet`: 5 files, 25,286,019 rows) and `stack_edu_cpp` (`Cpp/*.parquet`: 4 files,
16,246,746 rows). The Hub files are metadata: "This dataset only contains the SWHIDs to download the code files and not
the content of the files itself" (card); `prepare.fetch` adds each file's text from Software Heritage's S3 bucket. The
card describes no column one by one; SmolLM2 (arXiv 2502.02737) describes the classifier.

| Field | Type | Meaning | Kept |
|---|---|---|---|
| `blob_id` | string | the file's Software Heritage content id; the card's download script reads `content/{blob_id}` from the `softwareheritage` bucket | fetches the text |
| `language` | string | the file's language: `Python`, `C++` [inference] | `language` (the source's) |
| `repo_name` | string | the GitHub repository, `owner/name` [inference] | `repo_name` |
| `path` | string | the file's path in the repository, beginning with "/" [inference] | `relative_path`, the "/" stripped |
| `src_encoding` | string | the encoding of the file's bytes, e.g. `UTF-8` [inference] | decodes the text |
| `length_bytes` | int64 | the file's size in bytes [inference: equal to the fetched text's length in both samples] | dropped |
| `score` | double | educational value, 0 to 5, by the language's StarEncoder classifier trained on Llama3-70B-Instruct annotations (card; arXiv 2502.02737) | dropped |
| `int_score` | int64 | `score` rounded [inference]; the release keeps 3 and above ("a filtering threshold of 3 (out of 5)", card; Java 2) | dropped |
| `detected_licenses` | list<string> | licenses detected in the file [inference]; empty or null in 82.0% (Python) and 83.6% (C++) of rows | dropped |
| `license_type` | string | `permissive` or `no_license`, The Stack v2's split [inference] | dropped |
| text (fetched) | string | the file's text from Software Heritage, decoded with `src_encoding`; null where the bucket lacks the blob or it does not decode (`prepare.blob_texts`) | `content` |

| Field | `stack_edu_python` (`Python/train-00000-of-00005.parquet`, row 0) | `stack_edu_cpp` (`Cpp/train-00000-of-00004.parquet`, row 0) |
|---|---|---|
| `blob_id` | `'f76b3b840b3db0b3d8c980dc620327745383a006'` | `'1e46d26eb7e4cf83cb11d677e1f51f8e7cc61edc'` |
| `language` | `'Python'` | `'C++'` |
| `repo_name` | `'17BTEC005/virat-kohli'` | `'Sntai20/CPSC-5910-Data-Structures'` |
| `path` | `'/rohit sharma.py'` | `'/Week1/Lab/Solution/date.h'` |
| `src_encoding` | `'UTF-8'` | `'UTF-8'` |
| `length_bytes` | `135` | `436` |
| `score`, `int_score` | `3.03125`, `3` | `2.984375`, `3` |
| `detected_licenses` | `[]` | `[]` |
| `license_type` | `'no_license'` | `'no_license'` |
| text (fetched) | `'#multiply 2 no.s\na=10\nb=20\nc=a*b\nprint("multiplication of 10 and 20",a,"*",b,"=", c)\n\n\n…'` (135) | `'#ifndef DATE_H\r\n#define DATE_H\r\n\r\n#include <string>\r\n\r\nclass Date {\r\npublic:\r\n    Dat…'` (436) |

### openbmb/UltraData-Stack-v3-L2 @ `d6bdb3e`

`ultradata_stack_v3_l2_python` (`data/UltraData-Stack-v3-L2/python/*.parquet`: 45 files, 24,671,724 rows) and
`ultradata_stack_v3_l2_cpp` (`.../cpp/*.parquet`: 30 files, 17,990,102 rows). The Hub's main (`6207743`) changes only
the card (it adds the tech report's URL); the data files are the pinned ones. The card describes no column: the
meanings below quote The Stack v3's card (`HuggingFaceCode/stack-v3-train`) and the classifier's card
(`openbmb/UltraData-Code-L2-Classifier`), both at their main. Seven top-level columns; `source_meta` nests 18.

| Field | Type | Meaning | Kept |
|---|---|---|---|
| `uuid` | string | a row id [inference] | dropped |
| `repo_name` | string | "`owner/name` slug of the repository on GitHub" (Stack v3) | `repo_name` |
| `relative_path` | string | "Path of the file within the repository" (Stack v3) | `relative_path` |
| `content` | string | "The UTF-8 file content (PII-redacted)" (Stack v3) | `content`, CR LF and lone CR as LF, a leading U+FEFF dropped |
| `category` | string | the file's predicted role, "one of `ALGO`, `WEB`, `TOOL`, `DATA`, `TEST`, `CONFIG`, or `EXCLUDE`" (classifier) | dropped |
| `algo_rel_score` | float | the relevance model's "Algorithmic-relevance score", kept above a per-language threshold (classifier) | dropped |
| `quality_score` | float | the quality model's score, kept above a shared threshold (classifier) | dropped |
| `source_meta.github_metadata.branch` | string | "Default branch name" (Stack v3) | dropped |
| `source_meta.github_metadata.commit_count` | int64 | "Number of commits on the default branch" (Stack v3); null in 6 Python rows | dropped |
| `source_meta.github_metadata.repo_created_at` | string | "Repository creation timestamp" (Stack v3), ISO 8601 | dropped |
| `source_meta.github_metadata.is_fork` | bool | "Whether the repository is a fork" (Stack v3) | dropped |
| `source_meta.github_metadata.is_org_owned` | bool | "Whether the repository is owned by an organization" (Stack v3) | dropped |
| `source_meta.github_metadata.forked_from` | string | "Source repository if this is a fork, else `None`" (Stack v3); null in 99.4% (Python) and 99.7% (C++) of rows | dropped |
| `source_meta.github_metadata.stars`, `.forks`, `.issues`, `.pull_requests` | int64 | star, fork, open-issue and open-pull-request counts "at crawl time" (Stack v3) | dropped |
| `source_meta.file_meta.content_id` | string | "SHA-1 of the file content" (Stack v3) | dropped |
| `source_meta.file_meta.size_bytes` | int64 | "Length of the content in UTF-8 bytes" (Stack v3) | dropped |
| `source_meta.file_meta.file_timestamp` | int64 | "File modification time (Unix seconds). Only as accurate as the GitHub metadata." (Stack v3) | dropped |
| `source_meta.file_meta.language` | string | "Programming language, detected by `go-enry`" (Stack v3): `python`, `c++` | dropped (`language`: the source's) |
| `source_meta.file_meta.is_vendor` | bool | "Whether the file is a vendored/third-party file (per `go-enry`)" (Stack v3) | dropped |
| `source_meta.file_meta.license_type` | string | "`permissive` or `no_license`" (Stack v3) | dropped |
| `source_meta.file_meta.detected_licenses` | list<string> | "SPDX license identifiers detected by ScanCode" (Stack v3); empty or null in 95.7% (Python) and 97.5% (C++) of rows | dropped |
| `source_meta.commit_id` | string | "Commit (HEAD of the default branch) the snapshot was taken from" (Stack v3) | dropped |

| Field | `..._python` (`...-python-part-00001-of-00045.parquet`, row 0) | `..._cpp` (`...-cpp-part-00001-of-00030.parquet`, row 0) |
|---|---|---|
| `uuid` | `'67da4e05-d453-4952-b3a9-36545a3d41e1'` | `'1feaa85e-6908-4793-a0bf-b07fbce92796'` |
| `repo_name` | `'danielhellebust/sunshinelocator'` | `'ljy0221/Algorithm_Study'` |
| `relative_path` | `'src/app.py'` | `'C++17/백준/Gold/2098.\u2005외판원\u2005순회/외판원\u2005순회.cc'` |
| `content` | `'import plotly.graph_objects as go\nimport pandas as pd\nimport dash\nfrom dash import dcc\nfr…'` (5,083) | `'#include <iostream>\n#include <vector>\n#include <algorithm>\n#include <climits>\n#include <cs…'` (1,715) |
| `category`, `algo_rel_score`, `quality_score` | `'DATA'`, `0.92580`, `5.96619` | `'ALGO'`, `0.99998`, `5.22406` |
| `branch`, `commit_count`, `repo_created_at` | `'main'`, `5`, `'2023-03-18T20:59:15.000Z'` | `'main'`, `116`, `'2025-05-07T01:10:26.000Z'` |
| `is_fork`, `is_org_owned`, `forked_from` | `False`, `False`, `None` | `False`, `False`, `None` |
| `stars`, `forks`, `issues`, `pull_requests` | `0`, `0`, `0`, `0` | `0`, `0`, `0`, `0` |
| `content_id` | `'8f9e29a32ebede683741a81511853d01da88f9e6'` | `'cc3bc4d3f00527baa7fa6228474c8f4362f2c3a1'` |
| `size_bytes`, `file_timestamp` | `5083`, `1679249020` | `2108`, `1753424737` |
| `language`, `is_vendor` | `'python'`, `False` | `'c++'`, `False` |
| `license_type`, `detected_licenses` | `'no_license'`, `[]` | `'no_license'`, `[]` |
| `commit_id` | `'64bd2af9de25dd6dbaa7822d1a4a52b6a1d98368'` | `'698e498b26adbabeb32cba0db355e226a2b16c51'` |

### openbmb/UltraData-Code @ `525c9ac`

`ultradata_code_l2_python` (`data/UltraData-Code-L2/py/*.parquet`: 119 files, 72,844,227 rows), `ultradata_code_l2_cpp`
(`.../cpp/*.parquet`: 31 files, 23,290,266 rows), `ultradata_code_l3_python` (`data/UltraData-Code-L3/py/*.parquet`: 147
files, 20,750,202 rows), `ultradata_code_l3_cpp` (`.../cpp/*.parquet`: 167 files, 22,600,686 rows); no nulls. The Hub's
main (`b20cabc`) adds the tech report (arXiv 2610.23960) and edits the card; the data files are the pinned ones. The
meanings quote the card. L2 is "one selected source file" per row; L3 "uses ALGO files at L2 as seeds and generates
aligned task, analysis, solution, and test fields grounded in the source implementation" (card).

| L2 field | Type | Meaning | Kept |
|---|---|---|---|
| `uuid` | string | "unique file identifier" | dropped |
| `repo_name` | string | "source repository name", `owner/name` [inference] | `repo_name` |
| `relative_path` | string | "file path relative to the repository" | `relative_path` |
| `content` | string | "source code text" | `content`, CR LF and lone CR as LF, a leading U+FEFF dropped |
| `category` | string | "predicted file role, such as `ALGO`, `WEB`, `TOOL`, `DATA`, `TEST`, `CONFIG`" | dropped |
| `algo_rel_score` | double | "algorithmic relevance score in `[0, 1]`" | dropped |
| `quality_score` | double | "code quality score in `[0, 10]`" | dropped |

| L3 field | Type | Meaning | Kept |
|---|---|---|---|
| `uuid` | string | "unique file identifier" | dropped |
| `content` | string | "serialization containing task and solution" | `content` |
| `content_format` | string | "serialization format for `content`": `markdown` (the task as prose) or `code comment` (the task as a comment above the solution) [inference: the values] | `content_format` |
| `raw_content` | string | "original generated record before serialization" (card); in both samples, the seed file the exercise was written from [inference] | `seed` |
| `task` | string | "generated standalone problem statement" | `task` |
| `analysis` | string | "algorithm, edge cases, and complexity discussion" | `analysis` |
| `solution` | string | "generated self-contained reference implementation" | `solution` |
| `test` | string | "generated test candidates": asserts (Python), a `<cassert>` program (C++) in the samples | `test` |
| `full_content` | string | "serialization containing all generated fields" | `full_content`, the document's text |
| `full_content_format` | string | "serialization format for `full_content`": `standard order` or `solution-first order`, the order of the fields [inference: the values] | `full_content_format` |

| Field | `ultradata_code_l2_python` (`UltraData-Code-L2-py-part-00001-of-00119.parquet`, row 0) | `ultradata_code_l2_cpp` (`UltraData-Code-L2-cpp-part-00001-of-00031.parquet`, row 0) |
|---|---|---|
| `uuid` | `'092b1562-d4e9-4423-b494-6260a4844eaa'` | `'c2972d9f-a857-4dc0-b059-d51e5fe94486'` |
| `repo_name` | `'gagaoqphs/dunno'` | `'sidgujrathi/topcoder'` |
| `relative_path` | `'SaitamaRobot/modules/helper_funcs/filters.py'` | `'Day 3 - Medium (Level 2)/BusSeating/BusSeating.cpp'` |
| `content` | `'from SaitamaRobot import DEV_USERS, SUDO_USERS, SUPPORT_USERS\nfrom telegram import Message…'` (1,406) | `'// BEGIN CUT HERE\n\n// END CUT HERE\n#include <algorithm>\n#include <iostream>\n#include <sst…'` (3,363) |
| `category`, `algo_rel_score`, `quality_score` | `'TOOL'`, `0.99391`, `4.87828` | `'ALGO'`, `0.99918`, `4.77715` |

| Field | `ultradata_code_l3_python` (`UltraData-Code-L3-py-part-00001-of-00147.parquet`, row 0) | `ultradata_code_l3_cpp` (`UltraData-Code-L3-cpp-part-00001-of-00167.parquet`, row 0) |
|---|---|---|
| `uuid` | `'02ef0d3e-968f-4246-9def-862554b44b2c'` | `'140ebc29-af24-4f87-97ac-5e3be649c997'` |
| `content` | ``'Write a Python function `draw_star_square(n)` that takes a positive integer `n` and returns…'`` (1,766) | `'/*\nWrite a C++ function that accepts a square matrix represented as a vector of vectors of …'` (2,309) |
| `content_format` | `'markdown'` | `'code comment'` |
| `raw_content` | `'import sys\n\ninput = lambda: sys.stdin.readline().rstrip()\nn = int(input())\n\ns=n*4-3\narr=[…'` (380) | `'//{ Driver Code Starts\n\n#include<bits/stdc++.h>\nusing namespace std;\n\n// } Driver Code En…'` (2,378) |
| `task` | ``'Write a Python function `draw_star_square(n)` that takes a positive integer `n` and returns…'`` (746) | `'Write a C++ function that accepts a square matrix represented as a vector of vectors of int…'` (780) |
| `analysis` | ``'The approach is to directly simulate the pattern. First, compute side length `s = 4*n - 3`.…'`` (1,641) | `'The algorithm simulates the zigzag traversal by tracking a current position (row, column) a…'` (1,071) |
| `solution` | `'def draw_star_square(n):\n    """\n    Generate a square pattern of concentric star borders.…'` (1,004) | `'#include <vector>\n\n// Returns the elements of a square matrix in zigzag diagonal order.\nst…'` (1,522) |
| `test` | `'assert draw_star_square(1) == ["*"]\nassert draw_star_square(2) == [\n    "*****",\n    "*  …'` (680) | `'#include <cassert>\n#include <vector>\n\n// Declaration of the function being tested (normally…'` (1,272) |
| `full_content` | ``'Write a Python function `draw_star_square(n)` that takes a positive integer `n` and returns…'`` (4,074) | `'/*\nWrite a C++ function that accepts a square matrix represented as a vector of vectors of …'` (4,660) |
| `full_content_format` | `'standard order'` | `'solution-first order'` |

### openbmb/UltraData-Math @ `fe10db8`

`ultradata_math_l1` (`data/UltraData-Math-L1/*/*.parquet`: 1,485 files, 86,032,552 rows) and L3's four formats,
`data/UltraData-Math-L3/<format>/*.parquet`, 50 files each: `ultradata_math_l3_qa` (`QA-Synthetic`, 27,113,894 rows),
`ultradata_math_l3_conversation` (`Conversation-Synthetic`, 16,881,040), `ultradata_math_l3_multi_style`
(`Multi-Style-Synthetic`, 11,445,446), `ultradata_math_l3_textbook_exercise` (`Textbook-Exercise-Synthetic`,
26,005,670); no nulls. The card and its paper (arXiv 2602.09003) document the levels, not the columns.

| Field | Type | Meaning | Kept |
|---|---|---|---|
| L1 `content` | string | a Common Crawl math page: parsed by a math parser built on magic-html and w3m that standardizes "MathML, KaTeX, and AsciiMath into LaTeX format", then cleaned "through heuristic rules" with "document-level deduplication" (card) | `content` |
| L1 `meta` | string (a JSON object) | the page's Common Crawl record [inference]: keys `url`, `timestamp` (the crawl time, ISO 8601), `content_type`, `content_length` (the response's bytes, as a string), `record_id` (the WARC record's URN), `cc-path` (the WARC file's path) | dropped |
| L3 `uid` | string | a row id, a UUID [inference] | dropped |
| L3 `content` | string | a synthesized document by Qwen2.5-72B-Instruct, Qwen3-32B or GLM-4.5 from seed pages (Nemotron-CC-Math, MegaMath, FineMath): QA "rewrite declarative documents into "Question-Answer" pairs ... with explicit reasoning steps"; Conversation "Simulate "Teacher-Student" tutoring scenarios to generate multi-turn dialogue data"; Multi-Style "Rewrite single-source data into multiple styles (such as rigorous textbook style, competition problem-solving style, intuitive popular science style)"; Textbook-Exercise "Generate systematic textbook-like content based on specific knowledge points" (card) | `content` |

| Source (row 0 of its datasets-server config) | `uid` (L1: `meta`) | `content` |
|---|---|---|
| `ultradata_math_l1` | `{"url":"http://mathhelpforum.com/pre-calculus/185019-inverse-functions.html","timestamp":"2014-04-20T15:52:35Z","content_type":null,"content_length":"69823","record_id":"<urn:uuid:78c71b75-…>","cc-path":"CC-MAIN-2014-15/segments/1398223206118.10/warc/…"}` | `"inverse functions\nWhen f(x)=x^7+2x^3+x-1 I have found f^-1(3)=1 But I need to prove this. C…"` (1,590) |
| `ultradata_math_l3_qa` | `'00b41384-3471-4eea-a288-acebf7ce087e'` | `'A cylindrical tank with a radius of 2 meters and a height of 5 meters is filled with water t…'` (2,185) |
| `ultradata_math_l3_conversation` | `'6f55f96b-a0eb-4b93-9af2-68367ae2e744'` | `"Student: I'm having trouble understanding the problem statement for the integral \\(\\int_{0}…"` (2,285) |
| `ultradata_math_l3_multi_style` | `'2ee6ebaf-ca54-4878-b835-ca86cd1204a2'` | `'## Chapter 4: Division on the Abacus\n\n### 4.1 Definition and Principle of Division\n\n**Defin…'` (5,649) |
| `ultradata_math_l3_textbook_exercise` | `'f3a96b2e-af93-4e25-904e-f4896c680361'` | `'The y-intercept \\( b \\) in the equation \\( y = mx + b \\) is the point where the line inters…'` (3,461) |

### open-r1/codeforces-cots @ `39ac85c`

`codeforces_cots` reads two decontaminated subsets, 30 columns each: `solutions_py_decontaminated` (3 files, 8,133 rows:
Python, "just one completion per problem") and `solutions_short_and_long_decontaminated` (6 files, 16,266 rows: C++,
"the shortest and longest solution from R1"); "decontaminated using 8-gram overlap against the AIME24, AIME25, GPQA
Diamond, MATH-500, and LiveCodeBench benchmarks" (card). The card describes no column; the problem's columns quote the
problems' dataset card (`open-r1/codeforces`, main).

| Field | Type | Meaning | Kept |
|---|---|---|---|
| `id` | string | the problem, `<contest_id>/<index>` [inference: the values], shared by every generation of it | `problem_id` |
| `aliases` | list<string> | "list of other problem IDs that are copies of this one (quite common between Div 1. and Div 2. contests ...)" (problems card); mostly null | dropped |
| `contest_id`, `contest_name` | string | "the ID of the contest this problem belongs to", its name (problems card) | dropped |
| `contest_type` | string | "'ICPC', 'CF' or 'IOI'" (problems card) | dropped |
| `contest_start`, `contest_start_year` | int64 | "Contest start time in unix format", "year of the contest" (problems card) | dropped |
| `index` | string | "usually a letter, indicating the index of this problem in the contest" (problems card) | dropped |
| `time_limit`, `memory_limit` | double | per test case, in seconds and in megabytes (problems card) | dropped |
| `title`, `description` | string | "problem title", "main problem description" (problems card) | dropped |
| `input_format`, `output_format` | string | how test inputs and outputs are structured (problems card) | dropped |
| `interaction_format` | string | interactive problems only: how the program interacts with the grader (problems card) | dropped |
| `note` | string | "short explanation of how the examples are solved" (problems card) | dropped |
| `examples` | list<struct<input, output>> | "example test cases that are shown in the problem statement" (problems card) | dropped |
| `editorial` | string | "explanation of the solution from the original problem authors (when available)" (problems card) | dropped |
| `prompt` | string | the prompt given to R1: an instruction ("You will be given a competitive programming problem. ...") then the problem [inference: the values] | `content`, first part |
| `generation` | string | R1's completion: a `<think>` reasoning trace, then the answer with code [inference: the values]; "We did not filter the traces for correctness" (card) | `content`, second part |
| `finish_reason` | string | the API's stop reason, e.g. `stop` [inference] | dropped |
| `api_metadata` | struct<completion_tokens, prompt_tokens, prompt_tokens_details, total_tokens> | the API's token counts [inference]; `prompt_tokens_details` always null | dropped |
| `messages` | list<struct<content, role>> | the prompt and the generation as a chat, "so can be used directly for SFT" (card) | dropped |
| `accepted_solutions`, `failed_solutions` | list<struct<code, passedTestCount, programmingLanguage, verdict, ...>> | human Codeforces submissions that passed and that failed, with their code, language and verdict [inference] | dropped |
| `public_tests`, `private_tests`, `generated_tests` | struct<input: list<string>, output: list<string>> | the statement's tests, hidden tests and generated tests ("generated and validated additional tests", problems card) as parallel lists [inference] | dropped |
| `public_tests_ms` | list<struct<input, output>> | further test cases, often null [inference] | dropped |
| `problem_type` | string | how an output is checked: `diff` (exact), `checker` (several correct outputs) [inference: the values] | dropped |

| Field | `solutions_py_decontaminated`, row 0 | `solutions_short_and_long_decontaminated`, row 0 |
|---|---|---|
| `id`, `index`, `contest_id` | `'153/B'`, `'B'`, `'153'` | `'732/B'`, `'B'`, `'732'` |
| `contest_name`, `contest_type` | `'Surprise Language Round 5'`, `'ICPC'` | `'Codeforces Round 377 (Div. 2)'`, `'CF'` |
| `contest_start`, `contest_start_year` | `1329922800`, `2012` | `1476714900`, `2016` |
| `time_limit`, `memory_limit` | `2.0`, `256.0` | `1.0`, `256.0` |
| `title` | `'Binary notation'` | `'Cormen — The Best Friend Of a Man'` |
| `description` | `'You are given a positive integer n. Output its binary notation.'` | `"Recently a dog was bought for Polycarp. The dog's name is Cormen. Now Polycarp has a lot of…"` (1,194) |
| `examples` | `[{"input": "5", "output": "101"}, {"input": "13", "output": "1101"}]` | `[{"input": "3 5\n2 0 1", "output": "4\n2 3 2"}, ...]` (3 examples) |
| `editorial`, `note`, `aliases` | `None`, `'In the first example 5 = 1 * 22 + 0 * 21 + 1 * 20.'`, `None` | `"If we don't make enough walks during days i and i + 1, it's better to make an additional wa…"` (517), `None`, `None` |
| `prompt` | `'You will be given a competitive programming problem.\nAnalyze the maximum input constraints …'` (1,212) | the same instruction, then the problem (2,914) |
| `generation` | `"<think>\nOkay, I need to solve this problem where I have to convert a given integer n into i…"` (2,527) | `"<think>\nOkay, let's tackle this problem. So the task is to adjust the number of walks each …"` (50,971) |
| `finish_reason`, `api_metadata` | `'stop'`, `{completion_tokens: 632, prompt_tokens: 273, total_tokens: 905}` | `'stop'`, `{completion_tokens: 15295, prompt_tokens: 733, total_tokens: 16028}` |
| `accepted_solutions`, `failed_solutions` | `None`, `None` | human submissions (4,745 and 6,323 characters as JSON) |
| `public_tests` | `{"input": ["5\n", "13\n"], "output": ["101\n", "1101\n"]}` | 3 tests |
| `problem_type` | `'diff'` | `'checker'` |

### HuggingFaceTB/issues-kaggle-notebooks @ `ef882ad`

`github_issues` (`issues/*.parquet`: 62 files, 15,549,682 rows) and `kaggle_notebooks` (`kaggle/*.parquet`: 11 files,
580,195 rows); no nulls. "A modified part of the StarCoder2 model training corpus, precisely the
bigcode/StarCoder2-Extras dataset. We reformat the samples to remove StarCoder2's special tokens" (card; StarCoder2,
arXiv 2402.19173).

| Field | Type | Meaning | Kept |
|---|---|---|---|
| issues `repo_name` | string | the issue's GitHub repository, `owner/name` [inference] | `repo_name` |
| issues `issue_id` | string | the GitHub issue's number [inference] | dropped |
| issues `text` | string | the thread from GH Archive in the card's template (`Title:`, `Question:` and `username_0:`, `Answers:` and `username_N:`, optional `Status: Issue closed`); bot comments and e-mail auto-replies removed, short issues and very long comments filtered, usernames anonymized and PII redacted with StarPII (card) | `content` |
| kaggle `file_id` | string | the notebook's id in Meta Kaggle Code [inference] | dropped |
| kaggle `text` | string | a notebook from Meta Kaggle Code, "formatted in Markdown format, where we start with the notebook title, dataset description when available and put the notebook (converted to a Python script) in a code block"; syntax errors and notebooks under 100 characters removed, deduplicated, PII redacted (card) | `content` (`language`: Python) |

| Source (row 0 of its datasets-server config) | Id | `text` |
|---|---|---|
| `github_issues` | `repo_name` `'keybase/client'`, `issue_id` `'268324407'` | `"Title: Feature request: Play sound in OSX notifications\n\nQuestion:\nusername_0: First of al…"` (703) |
| `kaggle_notebooks` | `file_id` `'31673201'` | `'# COVID-19 Open Research Dataset Challenge (CORD-19)\n\n### Dataset Description\n\nIn response…'` (7,241) |

### karpathy/climbmix-400b-shuffle @ `915333b`

`climbmix` (`shard_*.parquet`: 6,543 files, 553,240,576 rows). The card is front matter only (`license: mit`); the
upstream, `nvidia/Nemotron-ClimbMix` (CLIMB, arXiv 2504.13161), groups pages into 1,000 topic clusters, removes the
low-scoring ones by an advertisement and an educational-value classifier and mixes the rest by weight; that this repo
is a detokenized, shuffled copy is [inference] from its name and contents.

| Field | Type | Meaning | Kept |
|---|---|---|---|
| `text` | string | an English web document [inference] | `content` |

| Field | Row 0 of `shard_00000.parquet` |
|---|---|
| `text` | `'Protect your business and employee\'s during flu seaon\n\nStarting in 2005, the Center for Disease Con…'` (1,051) |

### OpenCoder-LLM/opc-fineweb-math-corpus @ `858b10c`

`opc_fineweb_math` (`data/train-*.parquet`: 37 files, 5,241,900 rows); no nulls. "This math-related data from Fineweb
was specifically used in OpenCoder pre-training. We employ fastText in three iterative rounds to recall a final dataset
of 55B code and math-related data" (card; OpenCoder, arXiv 2411.04905). `url`, `text`, `file_path` and `dump` are
FineWeb's (meanings from `HuggingFaceFW/fineweb`'s card).

| Field | Type | Meaning | Kept |
|---|---|---|---|
| `url` | string | "url to the original page where `text` was present" (FineWeb) | dropped |
| `tag` | string | the recall's category, `math` [inference] | dropped |
| `text` | string | "the main text content" (FineWeb) | `content` |
| `file_path` | string | "s3 path for the individual CommonCrawl warc file containing this sample" (FineWeb) | dropped |
| `dump` | string | "the CommonCrawl dump this sample was a part of" (FineWeb) | dropped |
| `file_size_in_byte` | int64 | `text`'s size in bytes [inference] | dropped |
| `line_count` | int64 | `text`'s number of lines [inference] | dropped |

| Field | Row 0 of the datasets-server's `default` config |
|---|---|
| `url` | `'http://www.chegg.com/homework-help/questions-and-answers/imagine-mike-lives-earth-rest-sara-traveling-past-thee…'` |
| `tag`, `dump` | `'math'`, `'CC-MAIN-2016-18'` |
| `text` | `'Imagine Mike lives on the earth at rest. Sara is traveling past theearth with a velocity of…'` (354) |
| `file_path` | `'s3://commoncrawl/crawl-data/CC-MAIN-2016-18/segments/1461860124045.24/warc/CC-MAIN-20160428161524-00085-…'` |
| `file_size_in_byte`, `line_count` | `354`, `1` |
