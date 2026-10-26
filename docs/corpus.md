# Pretraining corpus

`corpus/` builds the pretraining data of a Python and C++ code model from 18 Hugging Face sources (`sources.py`). The
stages up to publication are text and independent of any tokenizer; the stages after the pull read a tokenizer.

| § | Stage | Command or entry point | Output under `CORPUS` (`artifacts/corpus`) |
|---|---|---|---|
| 1–2 | Prepare: fetch, standardize | `corpus.prepare` | `raw/<source>/`, `standardized/<source>/<k>.parquet` |
| 3 | Curate: check, dedup, split, shards | `corpus.curate` | `checked/<source>/`, `curated/<source>/<split>/` |
| 4 | Publish, pull | `corpus.publish push`, `pull` | `iaroy/nanochat-data` ↔ `curated/` |
| 5 | Tokenizer | `ablations.tokenizer_split_merge vocabularies` (the arms, `tokenizer-ablation/<arm>/`); sa-bpe's files copied | `tokenizer/tokenizer.{tiktoken,pattern}` |
| 6 | Tokenize | `corpus.tokenize` | `tokenized/<corpus>-<data key>/<split>.{bin,idx,json}` |
| 7–8 | Mixture, pack, FIM | `corpus.pack` | `tokenized/<corpus>-<data key>/<split>-T<seq_len>.{pieces,rows,fim}` |
| 9 | Load | `loader.PretrainingBatches` | micro-batches |

Every command runs as `python -m <module>`.

## 1. Sources and acquisition

`sources.SOURCES`, in deduplication's keep-first order (§3.2): a Hub dataset, its files per language (globs in
`sources.py`), a category (§2) and the column that becomes `content`.

| Source | Hub dataset | Category | Text | arXiv |
|---|---|---|---|---|
| `stack_edu_python`, `stack_edu_cpp` | HuggingFaceTB/stack-edu | REPOSITORY_FILE | SWH blob | 2502.02737 §3.4 |
| `ultradata_stack_v3_l2_python`, `_cpp` | openbmb/UltraData-Stack-v3-L2 | REPOSITORY_FILE | `content` | |
| `ultradata_code_l2_python`, `_cpp` | openbmb/UltraData-Code, L2 | REPOSITORY_FILE | `content` | 2610.23960 |
| `ultradata_code_l3_python`, `_cpp` | openbmb/UltraData-Code, L3 | EXERCISE | `content` | 2610.23960 |
| `nemotron_rewriting_python` | nvidia/Nemotron-Pretraining-Code-v2 | SYNTHETIC_FILE | `content` | 2512.20848 §2.2.2 |
| `nemotron_transpilation_cpp` | nvidia/Nemotron-Pretraining-Code-v2 | SYNTHETIC_FILE | `content` | 2512.20848 §2.2.2 |
| `codexglue_python` (train split) | google/code_x_glue_ct_code_to_text | FUNCTION | `code` | 2102.04664 |
| `ultradata_math_l3_*` (4 formats) | openbmb/UltraData-Math, L3 | TEXT | `content` | 2602.09003 §2.2.4 |
| `codeforces_cots` | open-r1/codeforces-cots | CONVERSATION | `prompt` + `generation` | |
| `kaggle_notebooks` | HuggingFaceTB/issues-kaggle-notebooks | NOTEBOOK | `text` | 2402.19173 §2.4.2 |
| `climbmix` | karpathy/climbmix-400b-shuffle | TEXT | `text` | |

The four UltraData-Math formats are `qa`, `conversation`, `multi_style` and `textbook_exercise` (of L3's five; the
persona-integrated one is not read); Codeforces-CoTs reads `solutions_py_decontaminated` (Python) and
`solutions_short_and_long_decontaminated` (C++). Model-generated sources: UltraData-Code L3, Nemotron rewriting and
transpilation (Qwen3-32B, arXiv 2512.20848 §2.2.2), UltraData-Math L3 and Codeforces-CoTs (DeepSeek-R1 traces).

**Fetch** (`prepare.fetch`): only the columns the source's standardizer reads (`prepare.columns`), row group by row
group, written to `<file>.partial` and renamed when whole; an existing raw file is kept. Stack-Edu's Hub files are
metadata: each blob's text is read from Software Heritage's public S3 bucket by `blob_id` and decoded with its
`src_encoding`; a blob the bucket lacks or that does not decode has no text. `FETCHES` = 4 files at once (one for
Stack-Edu, whose row groups run in a process pool).

**Extent** (`prepare.extent`): a source of a group read in part (`mixture.IN_PART`: synthetic code, ClimbMix,
UltraData-Math L3) fetches the fewest Hub files, in the SHA-1 order of their paths, whose text bytes (Parquet footers,
`sources.text_bytes`) reach

$$\text{need} = \frac{\text{draw}(g)}{\sum_{s \in g} \text{bytes}_s / \text{BPT}_s} \cdot (1 + \text{SPREAD}) \cdot
\frac{100}{99} \cdot \text{bytes}_s$$

where draw(g) is the tokens both stages draw of the group (`mixture.draw`, §7), BPT the reference tokenizer's bytes per
token (`mixture.BYTES_PER_TOKEN`, below) and 100/99 the train split's share (§3.3). Every other source fetches all its
files. `SPREAD` = 0.06 covers a retrained tokenizer compressing a source better than the reference: across tokenizer
training mixes "the NSL on any given subset only varies by 5 to 6 percentage points" (arXiv 2402.01035 §2.3). Under the
corpus's tokenizer the train splits hold 1.096 (synthetic), 1.071 (text) and 1.167 (math) × their group's draw (train
tokens of `tokenized/`, computed on master). A source whose packed shards fall short of its draw stops `pack`
(`pack.take`); a larger run or share than the extents hold needs `prepare`, `curate` and `publish push` again.

`mixture.BYTES_PER_TOKEN` is the reference tokenizer's (SA-BPE within `SPLIT_PATTERN`, trained before this corpus), used
for every token count made before the corpus's tokenizer exists (extents, the tokenizer sample):

| Source | Python | C++ |
|---|---|---|
| Stack-Edu | 3.460 | 3.065 |
| UltraData-Stack-v3-L2 | 3.638 | 3.015 |
| UltraData-Code L2 | 3.709 | 2.935 |
| UltraData-Code L3 | 3.826 | 3.593 |
| Nemotron rewriting / transpilation | 3.909 | 3.651 |
| CodeXGLUE, Kaggle, Codeforces-CoTs | 4.028, 3.459, 3.536 | |
| ClimbMix; UltraData-Math L3 QA, conversation, multi-style, textbook-exercise | 4.275; 2.917, 3.457, 3.332, 3.032 | |

## 2. Standardization and schema

`prepare.standardize` writes one standardized file per fetched file, `standardized/<source>/<k>.parquet` (k in the
extent's order), in the category's schema (`schema.py`; every column a non-null string):

| Category | Columns | Split group (§3.3) | `code` (§3.1) |
|---|---|---|---|
| REPOSITORY_FILE | `id`, `repo_name`, `relative_path`, `content`, `language` | `lower(repo_name)` | yes |
| EXERCISE | `id`, `content`, `language` | `id` | no |
| SYNTHETIC_FILE | `id`, `content`, `language` | `id` | yes |
| FUNCTION | `id`, `repo_name`, `content`, `language` | `lower(repo_name)` | yes |
| CONVERSATION | `id`, `problem_id`, `content` | `problem_id` | no |
| NOTEBOOK | `id`, `content`, `language` | `id` | no |
| TEXT | `id`, `content` | `id` | no |

- Rows whose text column is null are dropped. Stack-Edu's `relative_path` is its `path` without the leading `/`;
  CodeXGLUE's `repo_name` is its `repo` (`prepare.standardizer`).
- `prepare.lines`: CR LF and a lone CR become LF and every leading U+FEFF is dropped, as Python and C++ read a source
  file (Python Language Reference §2.1.2, §2.1.4; C++23 [lex.phases] phases 1–2). Otherwise the text is as published.
- `id` is the hex SHA-1 of `content`'s UTF-8 bytes (`schema.sha1`); `language` is the source's, `Python` or `C++`.
- Codeforces-CoTs keeps finished traces only (`finish_reason = 'stop'`); `content` is `prompt` followed by
  `generation`, `problem_id` the dataset's `id`.

## 3. Curation

`python -m corpus.curate`; every output, rows, order and row groups, is a function of the standardized files.

### 3.1 Python 3 check, with repair

Every file of a Python source whose category is `code` (Stack-Edu, UltraData-Stack-v3-L2 and UltraData-Code L2 Python,
Nemotron rewriting, CodeXGLUE) must parse with `ast.parse` (Python 3.12). A file that does not is repaired by the
smallest subset of these steps that makes it parse (`curate.repaired`: steps 1–2 where they change the text, and step
3; every subset by size, then in step order); the repaired text is kept with its `id` recomputed, and a file no subset
repairs is dropped. Output: `checked/<source>/<k>.parquet` (an existing checked file is kept; `prepare` removes them).
1. Stack v3's placeholders `<EMAIL>`, `<KEY>`, `<NAME>`, `<PASSWORD>` become the bare word.
2. Leading-indentation tabs expand to 8 columns (Python 2 Language Reference §2.1.8).
3. Every fixer of `fissix.fixes` (2to3), under the recursion limit `RECURSION` = 6,000 (2to3 recurses over the parse
   tree; at Python's default 1,000 it fails on 800 nested blocks; `ast.parse`'s depth limits are CPython's own).

### 3.2 Exact deduplication

One row per `id` over every source and both splits, the first in `SOURCES` order, then file and row
(`curate.duplicates`); every empty text is dropped with the copies.

### 3.3 Split

A row is validation when `md5_number(group) % 100 = 0` (DuckDB; `schema.SPLIT` = 100), the group being its category's
(§2): a repository (lowercased, as GitHub's names are case-insensitive), a Codeforces problem, or the document itself.

### 3.4 Shards

`curated/<source>/<split>/<k>.parquet`, zstd:
- **Repository sources** (REPOSITORY_FILE): one shard per residue partition, `val/0.parquet` and `train/1..99.parquet`.
  A partition's files are sorted by (lowercased repository, path, `id`) and cut into import-connected samples
  (`sources.repository_samples`, the crate's `topological_sort`, §6), written in sample order. Columns
  `schema.SAMPLE_FILE`: `repo_name`, `relative_path`, `content`, `sample` (no `id`, no `language`). A row group holds
  whole samples, at least `SAMPLE_ROWS` = 65,536 files (the last excepted), so a reader of row groups never splits a
  sample.
- **Every other source**: its category's columns; validation rows in `val/0.parquet`, train rows in `id` order (a
  uniform order, `id` being a hash) over `train/0..99.parquet` (`ntile(100)`), so any leading shards are a uniform
  sample.

## 4. Published dataset

`corpus/publish.py`, `DATASET` = `iaroy/nanochat-data` (private). The tree is `curated/`'s,
`<source>/<split>/<k>.parquet` (1,812 shards: 6 × 100 repository, 12 × 101 others; 160,061,439,381 bytes), and
`README.md`, the card (`publish.card`: layout, schema, per folder its contents, columns, origin and rows; the Hub's
viewer off, as its Parquet converter would merge mixed schemas). The shards were pushed at `a4ec1a08`; `main` is
`b091cb47`, whose two later commits removed a `manifest.json` and its mention in the card.

| Folder | Columns | Train rows | Val rows | Parquet GB |
|---|---|---|---|---|
| `stack_edu_python` | `repo_name`, `relative_path`, `content`, `sample` | 24,527,465 | 242,463 | 19.11 |
| `stack_edu_cpp` | same | 16,079,704 | 157,572 | 11.24 |
| `ultradata_stack_v3_l2_python` | same | 17,546,677 | 183,818 | 21.41 |
| `ultradata_stack_v3_l2_cpp` | same | 12,373,628 | 122,010 | 10.59 |
| `ultradata_code_l2_python` | same | 42,077,649 | 425,751 | 37.14 |
| `ultradata_code_l2_cpp` | same | 13,148,572 | 121,601 | 8.33 |
| `ultradata_code_l3_python` | `id`, `content`, `language` | 3,912,566 | 39,820 | 2.88 |
| `ultradata_code_l3_cpp` | same | 4,287,289 | 43,356 | 3.16 |
| `nemotron_rewriting_python` | same | 18,108,788 | 182,625 | 18.19 |
| `nemotron_transpilation_cpp` | same | 5,129,449 | 51,878 | 5.49 |
| `codexglue_python` | `id`, `repo_name`, `content`, `language` | 249,436 | 1,795 | 0.08 |
| `ultradata_math_l3_qa` | `id`, `content` | 1,073,706 | 10,842 | 0.75 |
| `ultradata_math_l3_conversation` | same | 668,528 | 6,714 | 0.76 |
| `ultradata_math_l3_multi_style` | same | 679,887 | 6,839 | 1.11 |
| `ultradata_math_l3_textbook_exercise` | same | 1,029,710 | 10,506 | 0.79 |
| `codeforces_cots` | `id`, `problem_id`, `content` | 22,689 | 225 | 0.28 |
| `kaggle_notebooks` | `id`, `content`, `language` | 574,380 | 5,815 | 1.37 |
| `climbmix` | `id`, `content` | 15,625,644 | 157,894 | 17.39 |
| All | | 177,115,767 | 1,771,524 | 160.06 |

- **Push** (`publish push`): writes the card, deletes every file the dataset holds but `.gitattributes` in one commit,
  uploads the shards in preupload calls, each closed once it reaches `PREUPLOAD` = 4 GiB (an LFS upload URL expires
  900 s after its batch call, so 4 GiB needs at least 4.6 MiB/s; the Hub skips objects it holds, so a rerun uploads
  the rest), adds them in one commit and squashes the history into one commit.
- **Pull** (`publish pull`): `snapshot_download` of `main`'s `<source>/**` into `CORPUS/curated`. `main` holds 1,814
  files: the 1,812 shards, `README.md` and `.gitattributes`.
- **Provenance.** Training reads only the packs derived from `CORPUS/curated` (§6–§9); the local shards' check
  against `b091cb47` is in `docs/findings.md`, Training setup.

## 5. Tokenizer

`corpus/tokenizer.py` loads the corpus's tokenizer from `CORPUS/tokenizer` (`tokenizer.tiktoken`, `tokenizer.pattern`),
encoded by tiktoken. It is the sa-bpe arm of `ablations/tokenizer_split_merge.py` (`vocabularies`), its two files
copied byte-identical (SHA-256) from `tokenizer-ablation/sa-bpe/`; no code path writes `CORPUS/tokenizer`. The
ablation's runs, which were to choose among the arms, were not run, so sa-bpe is final (`docs/ablations.md`,
tokenizer_split_merge). Its data key is `054a5bee…` (`results/tokenizer_split_merge-vocabularies.json`), under which
the whole corpus is tokenized and packed (§6, §8).

- **Vocabulary: 32,768 ids** (`model/config.py` `VOCAB_SIZE` = 2^15): the 256 bytes, 32,508 merges, then `<|bos|>`
  (id 32,764) and the FIM sentinels `<｜fim▁begin｜>`, `<｜fim▁hole｜>`, `<｜fim▁end｜>` (32,765–32,767; the names of
  DeepSeek-Coder's released tokenizer, `deepseek-ai/deepseek-coder-6.7b-base` `tokenizer.json`; the paper writes
  `<|fim_start|>`, `<|fim_hole|>`, `<|fim_end|>`, arXiv 2401.14196 §3.1.2). SA-BPE ran at 32,768 (arXiv 2604.14053
  App. B, Table 5); DeepSeek-Coder uses 32,000 (arXiv 2401.14196 §3.2); StarCoder2 found that "increasing the
  vocabulary size to 100K did not improve performance" (arXiv 2402.19173 §6.2); Dagan et al. measured HumanEval
  pass@1 20.52–20.84 from 32K to 256K at 1.5B, not monotone in the size (arXiv 2402.01035 §3.2, Table 3).
- **Trainer** (`train_tokenizer.train`, `crate/train.rs`): byte-level BPE within the chunks of the split pattern,
  each merge the pair of highest F·ln R (SA-BPE's F·log R, arXiv 2604.14053 §4.1, with R the number of authorship
  units holding the pair where the paper counts repositories; a pair of one unit scores 0) or of highest F (score
  "frequency"); ties to the higher F, then the smaller pair; a merge whose bytes are already a token is skipped.
- **`SPLIT_PATTERN`** (the reference): cl100k's with digit runs of one or two and no line break after a punctuation
  run.
- **Sample** (`train_tokenizer.sample`): `SAMPLE_BYTES` = 13.38e9 bytes of the curated train splits, each file-stage
  source's share its share of the file stage's bytes (`byte_shares`: its token share, §7, at `BYTES_PER_TOKEN`), drawn
  systematically in shard order; authorship units are the lowercased repository for repository files and functions,
  else the document. sa-bpe's sample: 3,988,600 documents, 13,444,424,046 bytes, 3,429,396 units
  (`results/tokenizer_split_merge-vocabularies.json`). Rationale: `docs/ablations.md`, tokenizer_split_merge, Fixed
  across arms.
- **Data key** (`Tokenizer.data_key`): SHA-256 of the pattern, BOS's id and the tiktoken file; a vocabulary's tokenized
  and packed data live in `<its directory's parent>/tokenized/<corpus>-<data key>/`.

## 6. Tokenize

`python -m corpus.tokenize` encodes every curated split under the corpus's tokenizer (special-token text encodes as
ordinary text), per corpus and split:

| File | Contents |
|---|---|
| `<split>.bin` | every document's tokens in curated order, uint16, no BOS or padding |
| `<split>.idx` | int64: each document's first offset, then the total |
| `<split>.json` | each curated shard's (or partition's) document count; `pack` packs each one's documents together |

**Documents.** A curated row, except in a repository source, where a document is one sample (§3.4): its files
concatenated in the sample's order. Each repository file is preceded by a line comment with its path (`# path`,
`// path`; DeepSeek-Coder: "a comment indicating the file's path is added at the beginning of each file", arXiv
2401.14196 §2.2) and ends with a line end, added when missing (`sources.texts`); a line break in a path is written
`\r` or `\n`, and a C++ file ending in a backslash gets an empty line, so phase 2 splices no comment onto it (C++23
[lex.phases]).

**Import-connected samples** (`crate/repository.rs` `topological_sort`, DeepSeek-Coder §2.2): a file depends on the
files its Python imports or C++ includes name, read from its tree-sitter tree and resolved by path within the
repository, without a search or include path: an absolute import by suffix (`a/b/c.py`, `a/b/c/__init__.py`, and per
imported name its submodule); a relative import of level l only from the importing file's l-th ancestor directory; a
quoted include first at its path from the including file's directory, else by suffix; an angled include by suffix.
A sample is a weakly connected component of the dependency graph, its files in dependency order (strongly connected
components in path order, where DeepSeek-Coder's Algorithm 1 orders a cycle by least in-degree); samples follow the
order of their first file; a single-file repository is one sample, unparsed.

**Repositories corpora** (`mixture.REPOSITORIES`: `repositories_python`, `repositories_cpp`): the 32K stage's
documents, one per (lowercased repository, language) across that language's three repository sources. Of a path
several sources hold, the first source's file is kept (`mixture.REPOSITORY_SOURCES` order); a repository's files are
its samples in order. `tokenize_repo_level_stage` takes train partitions from the last (99) backwards until their
documents hold the stage's repository draw, `CODE_SHARE` × the stage's tokens less Codeforces-CoTs' tokens, and adds
partition 0 as val. Under sa-bpe: partitions 95–99, 5,450,009,703 tokens for a draw of 4,563,412,330.

**Parallelism.** Row groups (sources) and (language, partition) tasks (repositories) are the units of work, written in
task order, so the output does not depend on the worker count; repository tasks in flight are bounded by `MEMORY` /
(`PEAK` = 8.18 × the largest partition's text), each encoding on a thread per core; `topological_sort` parses a
partition's files in parallel (rayon).

## 7. Mixture and stages

`corpus/mixture.py`. The stages are sized for `MAX_STEPS` = 69,580 steps of `BATCH_SIZE` = 15 × `MICRO_BATCH_TOKENS`
= 1,474,560 tokens (a micro-batch: 12 rows of 8,192 or 3 of 32,768): 102,599,884,800 tokens. Its last
`REPO_LEVEL_SHARE` = 300B / (5.2T + 300B) = 5.45% of steps (`REPO_LEVEL_STEPS` = 3,795), Qwen2.5-Coder's
repository-level share (arXiv 2409.12186 §3.2.1–3.2.2), are the 32K stage; the contexts are Qwen2.5-Coder's 8,192 and
32,768 (§3.2.1, §3.2.2). Any run draws by the stages' shares at its own size (`weights`). The base run (`run.py`
defaults: d24, `tokens_per_param` = `train/stages.py` `max_tokens_per_param` = 526.13 on sa-bpe's packs) is 69,611
steps, the 32K stage from step 65,814 (`docs/pretraining.md`); the packs hold it by construction.

| Stage | Context | Steps | Tokens | Rows |
|---|---|---|---|---|
| `FILE_LEVEL_STAGE` | 8,192 | 0–65,785 | 97.00B | 11,841,300 |
| `REPO_LEVEL_STAGE` | 32,768 | 65,785–69,580 (3,795) | 5.60B | 170,775 |

Weights: `mixture.weights` of each stage at its rows on sa-bpe's packs (computed on master; packed train rows 19,349,512
at 8K, 201,494 at 32K); tokens: share × the stage's tokens.

| Stage | Group | Sources | Share | Weight | Tokens |
|---|---|---|---|---|---|
| File | repository | the six repository sources | rest of 87% | 0.5929 | 57.51B |
| File | synthetic | UltraData-Code L3 (Python, C++), Nemotron rewriting, transpilation | 0.30 × 87% = 26.1% | 0.2610 | 25.32B |
| File | notebooks | `kaggle_notebooks` | supply | 0.0156 | 1.51B |
| File | functions | `codexglue_python` | supply | 0.0006 | 0.06B |
| File | text | `climbmix` | 10% | 0.1000 | 9.70B |
| File | math | UltraData-Math L3, four formats | 3% | 0.0300 | 2.91B |
| 32K | repositories | `repositories_python`, `repositories_cpp` | rest of 87% | 0.8146 | 4.56B |
| 32K | traces | `codeforces_cots` | supply | 0.0554 | 0.31B |
| 32K | text | `climbmix` | 10% | 0.1000 | 0.56B |
| 32K | math | UltraData-Math L3, four formats | 3% | 0.0300 | 0.17B |

- **Code 87%, text and mathematics 13%** (`CODE_SHARE`): DeepSeek-Coder's "87% source code, 10% English code-related
  natural language corpus, and 3% code-unrelated Chinese natural language corpus" (arXiv 2401.14196 §2), its natural
  language taken to cover web text (10%) and mathematics (3%).
- **Synthetic code 30% of code** [inference for code]: good mixtures' share of rephrased synthetic data, "converging
  to ∼30% for rephrased synthetic data" (Kang et al., arXiv 2510.01631 §1), measured on web text; its Limitations say
  the findings "may not directly apply to other generation methods (e.g., synthetic code, dialogues)". Code papers
  only bound the share: 10% synthetic code helps in code-only pretraining from scratch (arXiv 2408.10914 §3.4);
  UltraData-Code L2 with its synthetic L3 at 1:1 helps in continued pretraining (arXiv 2610.23960 Table 6; with
  CodeAlchemy's code-qa set instead, EvalPlus falls below L2 alone); rewritten-only code drops MBPP to 5.6 against
  58.7 for raw code and a 50-50 mix keeps 60.3 (arXiv 2606.10087 §3, RQ1, Table 4). The share is not ablated. The
  curated synthetic sources hold up to 0.30 × 1.096 ≈ 0.329 of code (§1); the file stage's repository partitions hold
  2.06× their draw at 0.30 and 1.43× at 0.
- **The 32K stage keeps text and mathematics:** OpenCoder's anneal keeps "84% of the annealing data" from "the original
  distribution" against "catastrophic forgetting" (arXiv 2411.04905 §2.2).
- **"supply"** sources are read whole once: a fixed share of these small sources would repeat them, and repeating "0.1%
  of the data 100 times" degrades an 800M model to a 400M one (arXiv 2205.10487 Abstract). A supply group's share is
  its packed rows over the stage's rows; "rest" is `CODE_SHARE` less the stage's other code groups.
- **Within a group** sources weigh by their packed rows (`Stage.shares`). `mixture.weights` replays a stage's shares
  through `mixture_rows`; a supply corpus that draws more rows than it holds weighs that many rows less, until none
  does.
- **Row order** (`mixture.mixture_rows`): run row n goes to the corpus whose interval of the cumulative weights holds
  frac((n + 1) · φ⁻¹), φ⁻¹ = (√5 − 1)/2, a Weyl sequence that spreads every prefix of the run evenly over the corpora
  (Knuth, TAOCP Vol. 3 §6.4); any stretch of the run is computed without the rows before it.

## 8. Packing and fill-in-the-middle

`python -m corpus.pack` calls the crate's `pack` (`crate/pack.rs`) once per curated shard of a tokenized split, into
rows of seq_len + 1 tokens. Output per corpus, split and context, `<split>-T<seq_len>`:

- `.pieces`: int64 × 4 per piece in row order: its token range [start, end) in the split, then its FIM example's
  range in `.fim` (empty for a piece read left to right);
- `.rows`: int64, each row's first piece, then the number of pieces;
- `.fim`: uint16, the FIM examples' tokens.

**What each stage packs** (`pack.main`; no document is packed twice):
- File stage: the repository sources' train partitions not taken by the 32K stage and the supply sources, whole; each
  source read in part, its leading shards, one more while its draw exceeds its rows (`pack.fit`).
- 32K stage: the repositories corpora and Codeforces-CoTs, whole; ClimbMix and UltraData-Math L3, the shards after
  the file stage's, as above.
- Every val split, whole, at the seq_len of each stage that reads it.

**Pieces.** A piece read left to right that ends its document fits its row with its BOS and its target BOS; a longer
document is cut, each cut the window's last statement boundary (the gaps between the items of a tree-sitter
container: module, block, translation unit, compound statement, declaration list, …; repository code only), else
after its last line-ending token, else before its last token that starts a character. A piece drawn as a FIM example
fits its row with its BOS, the example and the target BOS: its window first leaves room for the three sentinels and
shrinks by any excess of the built example. Each piece is its own attention segment (document masking: IntraDoc
lowers perplexity over UniChunk, arXiv 2402.13991 §3.2). A shard's documents are cut in parallel (rayon), longest
first; the crate's rayon workers get the process's stack limit (64 MiB under `scripts/container.sh`; `crate/lib.rs`),
as a 2 MiB worker stack overflowed when pack calls nested stolen jobs (`docs/findings.md`, Repacking and the packer
crash).

**Rows** (`best_fit_decreasing`): pieces longest first, each into the row with the fewest free slots that holds it,
else a new row (Best-Fit-Decreasing, arXiv 2404.10830 §3.1), so no piece is truncated. Within a row the pieces whose
next BOS is a target come first and a cut piece last, which keeps no slot for a BOS after it. A row's pieces come from
one curated shard.

**FIM** (`pack.FIM_RATE`, `RANDOM_SPAN_RATE`, `SINGLE_NODE_RATE`, `LINE_START_RATE`, `SPM_RATE`), in the train splits
of repository code only (`layout.fim_packed`: the six repository sources and the repositories corpora). On sa-bpe's
packs FIM examples hold 29.7% of the 8K stage's tokens and 40.7% of the 32K stage's (pieces' tokens at the stage
weights, computed on master); StarCoder2's two 50% draws (repository, then chunk) give ≈25% of its source files
(arXiv 2402.19173 §5.1, derived).

| Parameter | Value | Evidence |
|---|---|---|
| Rate, per piece after cutting (context-level) | 0.5 | [1] |
| Middle: random character span, else AST | 0.1 | "90% use our AST-FIM, and 10% use Rand-FIM" (arXiv 2506.00204 §5.1) |
| AST middle: Single-Node, else Aligned-Span | 0.5 | "an equal mix of two masking methods" (arXiv 2506.00204 §3.3) |
| AST middle starting its line: the line's indentation in the middle | 0.5 | [3] |
| SPM, else PSM | 0.5 | [2]; AST-FIM's "50% PSM and 50% SPM", untested there (arXiv 2506.00204 §5.1) |

[1] DeepSeek-Coder chose 0.5 (document level, PSM only), as a 100% rate gave its weakest code completion (arXiv
2401.14196 §3.1.2); FIM rates up to 0.9 cost no left-to-right performance (arXiv 2207.14255 §4.2), and "context-level
FIM yields a consistent and significant improvement over document-level FIM" (§4.4); at rate 0.7 AST-FIM kept
HumanEval+ and MBPP+ "nearly identical to the baseline L2R model" (arXiv 2506.00204 §6.1). No paper covers FIM over
whole multi-file repository texts, as in the 32K stage. [2] Joint PSM/SPM training matches or beats either mode alone
in Bavarian et al.'s one controlled comparison, except a near-tie (SPM single-line 0.622 joint vs 0.625 PSM-only),
and a pure-SPM model cannot infill in PSM (arXiv 2207.14255 §3.1, §4.3 Table 1); §8.1 recommends joint training.
[3] An AST node starts after its indentation, so without the rule no PSM example has a prefix ending in a line break
and a middle starting with indentation, the layout of line-level infilling (HumanEval single-line and multi-line
infilling mask whole lines, arXiv 2207.14255 §2.2: all 1,033 single-line prefixes end in a line break). Trained so,
d16 at 20 tokens per parameter opens a new definition or file after the end sentinel (PSM line exact match 0.023
against SPM 0.353, also on prompts the attention window covers whole), the train/test mismatch AST-FIM names for span
shapes it does not train (arXiv 2506.00204 §7). With the rule 15.6% of PSM examples have that layout, and d16 at 20
retrained on the repacked data infills single lines in PSM at line exact match 0.175 (from 0.022; pass@1 0.037 to
0.161; multi-line 0.008 to 0.045), SPM unchanged (0.367, 0.356), code bpb unchanged (0.4392) (`docs/findings.md`,
PSM infilling failure and fix; `dc76110`, `df62b22`, `0de5dfb`, `03d131e`).

- A FIM piece is decoded to text; its middle is a random span with probability 0.1, else an AST-FIM span on its
  tree-sitter parse, or a random span when the piece does not parse or has no node below the root.
- Prefix, suffix and middle are encoded separately (arXiv 2207.14255 §3): PSM `<｜fim▁begin｜> P <｜fim▁hole｜> S
  <｜fim▁end｜> M`, SPM `<｜fim▁begin｜> <｜fim▁hole｜> S <｜fim▁end｜> P M` (App. D, variant 2); the BOS after the middle
  is its target.
- Every draw comes from ChaCha8 keyed by (document, piece), so packs do not depend on the order of work.

## 9. Loading

`loader.PretrainingBatches` memory-maps the packed splits of a stage's mixture `{corpus: weight}` (`mixture.weights`
on the packed rows, `train/stages.py`) and assigns run rows to corpora by `mixture_rows`; each corpus's rows are read in
a seeded permutation (`train/stages.py`), resuming after the rows a start position consumed.
- A row: a BOS before every piece, the piece's tokens (reading "fim": its FIM example, if it has one), padding.
- Labels: the BOS after a piece that ends its document or a FIM middle is a target; after a piece its document
  continues, the BOS is an input only, the piece's last token labelled `IGNORE_INDEX` = −1. Padding is ignored.
- `cu_seqlens`: one attention segment per piece and per row's padding (FA4's varlen layout), a fixed count per loader
  (micro-batch rows × (the most pieces in a row + 1)), the unused entries empty; `segments` is the count used.
- Iteration stops when a corpus's rows run out, without the partial micro-batch.

## References

| arXiv | Work | § |
|---|---|---|
| 2102.04664 | CodeXGLUE | 1 |
| 2205.10487 | Hernandez et al., repeated data | 7 |
| 2207.14255 | Bavarian et al., fill-in-the-middle | 8 |
| 2401.14196 | DeepSeek-Coder | 5, 6, 7, 8 |
| 2402.01035 | Dagan et al., tokenizers for code | 1, 5 |
| 2402.13991 | Zhao et al., sequence composition | 8 |
| 2402.19173 | StarCoder2 | 1, 5, 8 |
| 2404.10830 | Ding et al., Best-Fit packing | 8 |
| 2408.10914 | Aryabumi et al., code in pretraining | 7 |
| 2409.12186 | Qwen2.5-Coder | 7 |
| 2411.04905 | OpenCoder | 7 |
| 2502.02737 | SmolLM2 (Stack-Edu) | 1 |
| 2506.00204 | AST-FIM | 8 |
| 2510.01631 | Kang et al., rephrased synthetic data | 7 |
| 2512.20848 | Nemotron 3 Nano (Nemotron-Pretraining-Code-v2) | 1 |
| 2602.09003 | UltraData-Math | 1 |
| 2604.14053 | SA-BPE | 5 |
| 2606.10087 | CodeAlchemy | 7 |
| 2610.23960 | UltraData-Code | 1, 7 |
