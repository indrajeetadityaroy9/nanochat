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
