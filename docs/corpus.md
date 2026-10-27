# Pretraining corpus

`corpus/` builds the pretraining data of a Python and C++ code model from 18 Hugging Face sources: acquisition,
standardization, curation and publication (text, independent of any tokenizer), then the tokenizer, tokenization,
the two-stage mixture, packing with fill-in-the-middle, and loading. The training side of the stages is
`docs/training.md` §1. Markers: [arithmetic] derived from code numbers; [inference] reasoning not measured;
[estimate]; [unverified] a figure without a record, or a claim its cited source's text lacks. "Computed on master"
numbers were recomputed on CPU from the curated shards and the tokenized and packed data under the corpus's tokenizer.

## Code

| File | § |
|---|---|
| `corpus/__init__.py` (run order; `CORPUS` = `paths.CORPUS` = `artifacts/corpus`, its layout in §11) | 10, 11 |
| `corpus/sources.py` (`SOURCES`, paths, `texts`, `curated_documents`, `repository_samples`) | 1, 3, 5, 6, 11 |
| `corpus/prepare.py` (fetch, extent, standardize) | 1, 2, 10, 11 |
| `corpus/schema.py` (categories, `SPLIT`, `SAMPLE_FILE`) | 2, 3, 11 |
| `corpus/curate.py` (check, dedup, split, shards) | 3, 10, 11 |
| `corpus/decontaminate.py`, `corpus/crate/decontaminate.rs` (`Grams`) | 3, 6, 10, 11 |
| `corpus/publish.py` (`DATASET`, card, push, pull) | 4, 10, 11 |
| `corpus/tokenizer.py`, `corpus/train_tokenizer.py` (the tokenizer step), `corpus/crate/train.rs` | 5, 10, 11 |
| `results/tokenizer_split_merge-vocabularies.json` (the tokenizer sample, `5ba5025`) | 5 |
| `corpus/tokenize.py`, `corpus/crate/repository.rs` (`topological_sort`) | 3, 6, 10, 11 |
| `corpus/parallel.py` (`MEMORY`, `ordered`, `SPAWN`) | 3, 6, 10, 11 |
| `corpus/mixture.py` | 1, 6, 7, 9, 11 |
| `corpus/pack.py`, `corpus/crate/pack.rs`, `corpus/layout.py` | 1, 5, 8, 10, 11 |
| `corpus/crate/lib.rs` (module, rayon pool), `corpus/crate/Cargo.toml`, `corpus/crate/Cargo.lock` | 6, 8, 11 |
| `corpus/loader.py` | 9, 11 |

## 1. Sources and acquisition

`sources.SOURCES`, in deduplication's keep-first order (§3): a Hub dataset, its files per language (globs in
`sources.py`), a category (§2) and the column that becomes `content`.

| Source | Hub dataset | Category | Text | arXiv |
|---|---|---|---|---|
| `stack_edu_python`, `stack_edu_cpp` | HuggingFaceTB/stack-edu | REPOSITORY_FILE | SWH blob | 2502.02737 §3.4 |
| `ultradata_stack_v3_l2_python`, `_cpp` | openbmb/UltraData-Stack-v3-L2 | REPOSITORY_FILE | `content` | 2610.23960 §2.3, App. B.7 |
| `ultradata_code_l2_python`, `_cpp` | openbmb/UltraData-Code, L2 | REPOSITORY_FILE | `content` | 2610.23960 §2.3 |
| `ultradata_code_l3_python`, `_cpp` | openbmb/UltraData-Code, L3 | EXERCISE | `content` | 2610.23960 §2.4 |
| `nemotron_rewriting_python` | nvidia/Nemotron-Pretraining-Code-v2 | SYNTHETIC_FILE | `content` | 2512.20848 §2.2.2 |
| `nemotron_transpilation_cpp` | nvidia/Nemotron-Pretraining-Code-v2 | SYNTHETIC_FILE | `content` | 2512.20848 §2.2.2 |
| `codexglue_python` (train split) | google/code_x_glue_ct_code_to_text | FUNCTION | `code` | 2102.04664 §3.9 |
| `ultradata_math_l3_*` (4 formats) | openbmb/UltraData-Math, L3 | TEXT | `content` | 2602.09003 §2.2.4 |
| `codeforces_cots` | open-r1/codeforces-cots | CONVERSATION | `prompt` + `generation` | |
| `kaggle_notebooks` | HuggingFaceTB/issues-kaggle-notebooks | NOTEBOOK | `text` | 2402.19173 §2.4.2 |
| `climbmix` | karpathy/climbmix-400b-shuffle | TEXT | `text` | |

The four UltraData-Math formats are `qa`, `conversation`, `multi_style` and `textbook_exercise` (of L3's five; the
persona-integrated one is not read); Codeforces-CoTs reads `solutions_py_decontaminated` (Python) and
`solutions_short_and_long_decontaminated` (C++). UltraData-Stack-v3-L2 holds The Stack v3's files that UltraData's L2
classifiers select; UltraData-Code L3 holds exercises written from L2's algorithmic files; Nemotron rewriting is every
Python file of NVIDIA's GitHub corpus rewritten by Qwen3-32B, and transpilation Python files transpiled into C++.
Model-generated sources: UltraData-Code L3 (DeepSeek-v4-flash, arXiv 2610.23960 §2.4), Nemotron rewriting and
transpilation (Qwen3-32B, arXiv 2512.20848 §2.2.2), UltraData-Math L3 (a multi-model ensemble, arXiv 2602.09003
§2.2.4) and Codeforces-CoTs (DeepSeek-R1 traces, per its dataset card).

**Fetch** (`prepare.fetch`): only the columns the source's standardizer reads (`prepare.columns`), row group by row
group, into `raw/<source>/<Hub path>`. Stack-Edu's Hub files are
metadata, downloaded whole; each row group's blobs are read from Software Heritage's public S3 bucket by `blob_id`,
gunzipped and decoded with their `src_encoding` (`prepare.blob_texts`); a blob the bucket lacks or that does not decode
has no text. `FETCHES` = 4 files at once (Stack-Edu one at a time, `prepare.prepare`): the throughput peak reading one
column of 8 UltraData-Code L3 files, 8.5, 11.5, 12.4 and 10.5 MB/s at 1, 2, 4 and 8 files in flight (`b0c6bd9`).
Any other failure (a request failing under obstore's or the Hub client's default retries) stops `prepare`; a rerun
fetches every file again (`fetch` keeps no partial state).

**Extent** (`prepare.extent`): a source of a group read in part (`mixture.IN_PART`: synthetic code, ClimbMix,
UltraData-Math L3) fetches the fewest Hub files, in the SHA-1 order of their paths, whose text bytes (from Parquet
footers, `sources.text_bytes`: an estimate, as pyarrow exposes no exact unencoded size and an exact count scans the
column) reach

$$\text{need} = \frac{\text{draw}(g)}{\sum_{s \in g} \text{bytes}_s / \text{BPT}_s} \cdot (1 + \text{SPREAD}) \cdot
\frac{100}{99} \cdot \text{bytes}_s$$

where draw(g) is the tokens both stages draw of the group (`mixture.draw`, §7), BPT the reference tokenizer's bytes per
token (`mixture.BYTES_PER_TOKEN`, below) and 100/99 the train split's share (§3). Every other source fetches all its
files. `SPREAD` = 0.06 widens the target relatively, for a retrained tokenizer compressing a source better than the
reference [inference]. Across tokenizer training mixes NSL "only varies by 5 to 6 percentage points" (arXiv 2402.01035
§2.3); as a relative spread of token counts that is 6.9% for code (NSL 75.54–80.78) and 6.4% for English
(84.86–90.28), Fig. 2 [arithmetic], so 0.06 sits slightly below the paper's spread. It held for every arm: the
intrinsic record's `train_tokens_over_draw` is ≥ 1.017 (sa-bpe-fused, synthetic 1.0173;
`results/tokenizer_split_merge-intrinsic.json`). Under the corpus's tokenizer the train splits hold 1.092 (synthetic,
draw 25.32B), 1.071 (text, 10.26B) and 1.166 (math, 3.08B) × their group's draw (tokens of the tokenized train splits,
computed on master; the intrinsic record's 1.169 for math estimates tokens from bytes, `docs/experiments.md` §2). A
source whose packed shards fall short of its draw stops `pack` (`pack.fit`: the shard iterator is exhausted); a larger
run or share than the extents hold needs `prepare`, `curate` and `publish push` again.

`mixture.BYTES_PER_TOKEN` is the reference tokenizer's (SA-BPE within `SPLIT_PATTERN`, trained before this corpus),
used for every token count made before the corpus's tokenizer exists (extents, the tokenizer sample):

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

| Category | Columns | Split group (§3) | `code` (§3) |
|---|---|---|---|
| REPOSITORY_FILE | `id`, `repo_name`, `relative_path`, `content`, `language` | `lower(repo_name)` | yes |
| EXERCISE | `id`, `content`, `language` | `id` | no |
| SYNTHETIC_FILE | `id`, `content`, `language` | `id` | yes |
| FUNCTION | `id`, `repo_name`, `content`, `language` | `lower(repo_name)` | yes |
| CONVERSATION | `id`, `problem_id`, `content` | `problem_id` | no |
| NOTEBOOK | `id`, `content`, `language` | `id` | no |
| TEXT | `id`, `content` | `id` | no |

- Rows whose text column is null are dropped. Stack-Edu's `relative_path` is its `path` without leading `/`s;
  CodeXGLUE's `repo_name` is its `repo` (`prepare.standardizer`).
- `prepare.lines`: CR LF and a lone CR become LF and every leading U+FEFF is dropped, as Python and C++ read a source
  file (Python Language Reference §2.1.2, §2.1.4; C++23 [lex.phases] phases 1–2). Otherwise the text is as published.
- `id` is the hex SHA-1 of `content`'s UTF-8 bytes (`schema.sha1`); `language` is the source's, `Python` or `C++`.
- Codeforces-CoTs keeps finished traces only (`finish_reason = 'stop'`; a truncated trace mostly ends before its code
  [unverified]); `content` is `prompt` followed by `generation`, `problem_id` the dataset's `id`.

## 3. Curation

`python -m corpus.curate`; every output's rows and order are a function of the standardized files (a repository
shard's row groups too; a document shard's row groups are DuckDB `COPY`'s defaults). DuckDB runs in `parallel.MEMORY`
(half the lower of the machine's memory and the cgroup's `memory.max`), spilling to `CORPUS/spill`.

**Python 3 check, with repair.** Every file of a Python source whose category is `code` (Stack-Edu,
UltraData-Stack-v3-L2 and UltraData-Code L2 Python, Nemotron rewriting, CodeXGLUE) must parse with `ast.parse`
(Python 3.12). A file that does not is repaired by the smallest subset of these steps that makes it parse
(`curate.repaired`: steps 1–2 where they change the text, and 2to3; every subset by size, then in step order); the
repaired text is kept with its `id` recomputed, and a file no subset repairs is dropped. Output:
`checked/<source>/<k>.parquet` (`prepare` removes them before rebuilding).
1. Stack v3's placeholders `<EMAIL>`, `<KEY>`, `<NAME>`, `<PASSWORD>` become the bare word.
2. Leading-indentation tabs expand to tab stops of 8 (Python 2 Language Reference §2.1.8).
3. Every fixer of `fissix.fixes` (2to3), under the recursion limit `RECURSION` = 6,000 (2to3 recurses over the parse
   tree; at Python's default 1,000 it fails on 800 nested blocks, at 6,000 not on 1,600; `ast.parse`'s depth limits
   are CPython's own, its result unchanged from limit 1,000 to 100,000); `two_to_three` adds a missing final line end
   before 2to3.

**Exact deduplication.** One row per `id` over every source (before the split), the first in `SOURCES` order, then
file and row (`curate.duplicates`); every empty text is dropped with the copies.

**Decontamination** (`python -m corpus.decontaminate`, after `curate`, on its train shards). A train row is dropped
when its `content` shares a 10-gram of whitespace-separated words (Rust `split_whitespace`, case kept) with a scored
benchmark's text: Qwen2.5-Coder's rule, "any training data with a 10-gram word-level overlap with the test data was
removed" (arXiv 2409.12186 §5), the 10-gram filter of DeepSeek-Coder (arXiv 2401.14196 §2.4), whose exact match of
3–9-word test strings is not applied (a short reference solution such as HumanEval/53's `return x + y` is common
code [inference]). The benchmark texts (`decontaminate.benchmark_texts`, 1,428): each task's prompt and reference
solution, as `eval/code_eval.py` `tasks` loads them (HumanEval+ 164, MBPP+ 378, HumanEval-X C++ 164), MultiPL-E's
HumanEval and MBPP C++ prompts (161, 397; they have no reference), and each HumanEval infilling program, prompt +
reference middle + suffix (164; the three infilling benchmarks are cut from one program per task). The crate's
`Grams` (`crate/decontaminate.rs`) hashes each 10-gram of word hashes (FxHash; 53,395 distinct) and compares a hit
word by word, so no hash collision drops a row. Shards run in parallel, a process per CPU with one matching thread
(160 GB in 2 min 44 s on the GH200 host's 64 CPUs, `abeb4eb`); a shard with a dropped row is rewritten without it, its
row groups kept (a repository shard's samples stay whole, less the dropped files). Val splits are untouched: they are
held out, and every bpb record scores the same rows. `CURATED/decontamination.json` (`decontaminate.RECORD`, published
with the shards, §4) holds `n` = 10, `benchmarks` (the 1,428 text names) and, per source, each dropped row as [shard,
its row before decontamination, the indices of the benchmark texts it matched], written before any shard is
rewritten; the step runs once (`RECORD` opened exclusively). On the published curated build it drops 491,203 train
rows (0.28% of the train rows before decontamination; per source, share of its train rows before decontamination):

| Source | Dropped | Share of train | Source | Dropped | Share of train |
|---|---|---|---|---|---|
| `ultradata_code_l2_python` | 122,781 | 0.292% | `ultradata_code_l2_cpp` | 6,271 | 0.048% |
| `ultradata_stack_v3_l2_python` | 106,622 | 0.608% | `nemotron_transpilation_cpp` | 3,603 | 0.070% |
| `nemotron_rewriting_python` | 88,082 | 0.486% | `ultradata_math_l3_*` (4) | 2,426 | 0.04–0.14% |
| `ultradata_code_l3_python` | 78,937 | 2.018% | `kaggle_notebooks` | 706 | 0.123% |
| `stack_edu_python` | 57,705 | 0.235% | `codeforces_cots` | 239 | 1.053% |
| `ultradata_stack_v3_l2_cpp` | 8,619 | 0.070% | `climbmix` | 145 | 0.001% |
| `ultradata_code_l3_cpp` | 7,920 | 0.185% | `codexglue_python` | 18 | 0.007% |
| `stack_edu_cpp` | 7,129 | 0.044% | | | |

The dropped rows match every HumanEval+ task and infilling program, 372 of 378 MBPP+ tasks, 157 of 164 HumanEval-X C++
references, and 142 of 161 and 289 of 397 of MultiPL-E's HumanEval and MBPP C++ prompts. The rule also drops common
code that shares a 10-gram with a reference solution: of `ultradata_stack_v3_l2_python`'s rows, the most matched MBPP+
and HumanEval+ texts are Mbpp/733 (13,996 rows) and HumanEval/31, `is_prime` (8,539).

**Split.** A row is validation when `md5_number(group) % 100 = 0` (DuckDB; `schema.SPLIT` = 100), the group being its
category's (§2): a repository (lowercased, as GitHub's names are case-insensitive), a Codeforces problem, or the
document itself. The split is per source: a validation repository's file can occur in train rewritten (Nemotron
rewriting, transpilation) or as an exercise written from it (UltraData-Code L3), sources split by document; no
published column links the two [inference]. `SPLIT` also sets the train shards: each document source's 100
(`ntile`) and the repository sources' 99 partitions, the granularity at which `pack` adds shards of a source read in
part (§8).

**Shards**, `curated/<source>/<split>/<k>.parquet`, zstd:
- **Repository sources** (REPOSITORY_FILE): one shard per residue partition, `val/0.parquet` and `train/1..99.parquet`.
  A partition's files are sorted by (lowercased repository, path, `id`) and cut into import-connected samples
  (`sources.repository_samples`, the crate's `topological_sort`, §6), written in sample order, a sample's files in
  dependency order. Columns
  `schema.SAMPLE_FILE`: `repo_name`, `relative_path`, `content`, `sample`. A row group holds whole samples, at least
  `SAMPLE_ROWS` = 65,536 files as curated (the last excepted; decontamination's dropped files leave fewer, §3), so a
  reader of row groups never splits a sample. A row group is a tokenize task; its peak is 1.71 GiB (a 235 MiB group of
  `ultradata_code_l2_python`), 3.14 GiB at 122,880 files, and three published shards re-cut at 122,880 files tokenize
  identically (`c1a7d7b`).
- **Every other source**: its category's columns; validation rows in `val/0.parquet` and train rows over
  `train/0..99.parquet` (`ntile(100)`), each in `id` order (a uniform order, `id` being a hash), so any leading shards
  are a uniform sample.

## 4. Published dataset and provenance

`corpus/publish.py`, `DATASET` = `iaroy/nanochat-data` (private on the Hub). `push` writes the tree
`<source>/<split>/<k>.parquet` (`curated/`'s: 1,812 shards, 6 × 100 repository, 12 × 101 others; 166,229,888,890
bytes), `decontamination.json` (§3), `tokenizer/tokenizer.{tiktoken,pattern}` (`CORPUS/tokenizer`, §5) and
`README.md`, the card (`publish.card`: layout, schema, decontamination, per folder its contents, columns, origin and
rows, the tokenizer; the Hub's viewer off, as its Parquet converter would merge mixed schemas). The curated shards
were committed at `a4ec1a08` ("nanochat corpus, curated", 2026-10-24); `dc5391b4` added `tokenizer/` (the shipped
tokenizer, `tokenizer.tiktoken` sha256 `59e8b637…`, §5) on `b091cb47`, which had removed a `manifest.json`; `main` is
`dbe57a81` ("nanochat corpus, curated and decontaminated", 2026-10-27): the 1,682 shards with a dropped row rewritten
(the document sources' by pyarrow's writer where DuckDB's `COPY` had written them, 1.02–1.16× their size; the
repository shards' 0.95–1.00×), the other 130 (the 112 untouched train shards and the 18 val shards) byte-identical to
`dc5391b4`'s, the record added, the card rewritten, the tokenizer unchanged.

| Folder | Columns | Train rows | Val rows | Parquet GB |
|---|---|---|---|---|
| `stack_edu_python` | `repo_name`, `relative_path`, `content`, `sample` | 24,469,760 | 242,463 | 19.06 |
| `stack_edu_cpp` | same | 16,072,575 | 157,572 | 11.22 |
| `ultradata_stack_v3_l2_python` | same | 17,440,055 | 183,818 | 21.32 |
| `ultradata_stack_v3_l2_cpp` | same | 12,365,009 | 122,010 | 10.54 |
| `ultradata_code_l2_python` | same | 41,954,868 | 425,751 | 37.05 |
| `ultradata_code_l2_cpp` | same | 13,142,301 | 121,601 | 8.31 |
| `ultradata_code_l3_python` | `id`, `content`, `language` | 3,833,629 | 39,820 | 3.30 |
| `ultradata_code_l3_cpp` | same | 4,279,369 | 43,356 | 3.62 |
| `nemotron_rewriting_python` | same | 18,020,706 | 182,625 | 20.66 |
| `nemotron_transpilation_cpp` | same | 5,125,846 | 51,878 | 6.23 |
| `codexglue_python` | `id`, `repo_name`, `content`, `language` | 249,418 | 1,795 | 0.08 |
| `ultradata_math_l3_qa` | `id`, `content` | 1,073,272 | 10,842 | 0.83 |
| `ultradata_math_l3_conversation` | same | 667,967 | 6,714 | 0.86 |
| `ultradata_math_l3_multi_style` | same | 678,935 | 6,839 | 1.27 |
| `ultradata_math_l3_textbook_exercise` | same | 1,029,231 | 10,506 | 0.89 |
| `codeforces_cots` | `id`, `problem_id`, `content` | 22,450 | 225 | 0.29 |
| `kaggle_notebooks` | `id`, `content`, `language` | 573,674 | 5,815 | 1.58 |
| `climbmix` | `id`, `content` | 15,625,499 | 157,894 | 19.12 |
| All | | 176,624,564 | 1,771,524 | 166.23 |

- **Push** (`publish push`): writes the card, then passes the shards, the tokenizer's two files, the record and the
  card to preupload calls (which upload the LFS files among them; the small regular files travel in the commit), each
  call closed once its files reach `PREUPLOAD` = 4 GiB (an LFS upload URL expires 900 s after its batch call,
  `X-Amz-Expires=900`, `37cebf7`, so 4 GiB needs at least 4.6 MiB/s [arithmetic]; the Hub skips objects it holds, so
  a rerun uploads the rest), then replaces the tree in one commit: the dataset's files the push no longer has (all but
  `.gitattributes`) deleted, the rest added; the history is kept.
- **Pull** (`publish pull`): `snapshot_download` of `main`'s `<source>/**` and `decontamination.json` into
  `CORPUS/curated` and of `tokenizer/*` into `CORPUS/tokenizer`. At `dbe57a81` `main` holds 1,817 files: the 1,812
  shards, the record, the tokenizer's two files, `README.md` and `.gitattributes`.
- **Provenance.** The SHA-256 of every local shard in `CORPUS/curated` equals the Hub's LFS hash at `dbe57a81`
  (`HfApi.dataset_info(files_metadata=True)`, `abeb4eb`): 1,812 shards, none missing or extra, each of the Hub's size;
  the shards whose hash differs from `dc5391b4`'s are exactly the record's shards with a dropped row, and every
  folder's rows on the card equal the shards' Parquet footers (computed on master). Training reads only the packs
  derived from these shards (§6–§9); the seeded generators (data order, FIM's ChaCha8) select and transform rows but
  add no data.

## 5. Tokenizer

`corpus/tokenizer.py` loads the corpus's tokenizer from `CORPUS/tokenizer` (`tokenizer.tiktoken`, `tokenizer.pattern`),
encoded by tiktoken (`get_tokenizer`); the dataset ships it (`publish pull`, §4). It is the sa-bpe arm of the tokenizer
ablation (`docs/experiments.md` §2: the other arms, the intrinsic comparison and why the ablation's training runs were
not run). Its data key is `054a5beebb0fbcb3e21b5f62f2545e49b79d8620ba75322c1804f5a004283e3a`
(`results/tokenizer_split_merge-vocabularies.json`, `5ba5025`), under which the whole corpus is tokenized and packed
(§6, §8).

**Method and provenance.** `python -m corpus.train_tokenizer` (`train_tokenizer.main`) is the method: the sample, the
trainer and the settings below, written to `CORPUS/tokenizer`. The shipped tokenizer (tiktoken file SHA-256
`59e8b637…`, pattern `47a359ca…`) was trained by the same method (the ablation's `vocabularies` step at `5ba5025`) on
an earlier build of the curated corpus. On the dataset at `dc5391b4` (§4, before decontamination) the sample has the
record's 3,988,600 documents, but its bytes and units differ per source (stack_edu_python 1,195,186,384 bytes and
384,926 units against the record's 1,186,614,352 and 384,949; kaggle_notebooks 195,259,329 bytes against
206,881,801; codexglue_python, stack_edu_cpp, climbmix and both Nemotron sources also differ), and the step writes an
equivalent tokenizer, not the shipped one: tiktoken file `0f9f7bc0…`, data key
`3246901b3b651b320713bfd6ee408be0d53ff8098a0a9f0c1d321179105b41a0`, `tokenizer.pattern` identical; byte-identical to
the sa-bpe arm of `ablations/tokenizer_split_merge.py` at `67192ab` (removed in `cd4edbe`) on the same data (computed
at `cd4edbe`, no record) [unverified].
The build (§10) therefore pulls the shipped tokenizer, and tokenize and pack reproduce data key `054a5bee…`.

- **Vocabulary: 32,768 ids** (`model/config.py` `VOCAB_SIZE` = 2^15): the 256 bytes, 32,508 merges, then
  `SPECIAL_TOKENS`: `<|bos|>` (id 32,764) and the FIM sentinels `<｜fim▁begin｜>`, `<｜fim▁hole｜>`, `<｜fim▁end｜>`
  (32,765–32,767; the names of DeepSeek-Coder's released tokenizer, `deepseek-ai/deepseek-coder-6.7b-base`
  `tokenizer.json`; the paper writes `<|fim_start|>`, `<|fim_hole|>`, `<|fim_end|>`, arXiv 2401.14196 §3.1.2).
  SA-BPE's models use 32,768 ids (arXiv 2604.14053 App. B, Table 5); StarCoder2 found that "increasing the vocabulary
  size to 100K did not improve performance" (arXiv 2402.19173 §6.2); Dagan et al. measured HumanEval pass@1
  20.52–20.84 from 32K to 256K on an NL-pretrained 1.5B model given each tokenizer, not monotone in the size (arXiv
  2402.01035 §3.2, Table 3).
- **Trainer** (`crate/train.rs` `train`, called by `train_tokenizer.main`): byte-level BPE within the chunks of the
  split pattern, each merge the pair of highest F·ln R (SA-BPE's F·log R, arXiv 2604.14053 §4.1, ln natural as in its
  released `scoring.py`, `math.log`; F the pair's frequency, R the number of authorship units holding it in the
  current segmentation, where the paper counts repositories; a pair of one unit scores 0); ties to the higher F, then
  the smaller pair; a merge whose bytes are already a token is skipped, as a vocabulary maps bytes to one id. Every
  merge is an argmax with fixed tie-breaks, so the result does not depend on the worker count. Of SA-BPE's Table 3
  scores, F·log R has the highest compression (3.643) and far fewer verified under-trained tokens than F (72 against
  640); F·log R·L (Table 3: 3.642 and 44; Table 4's "SA-BPE") is not used, as for natural language "the language
  aspect must be avoided" (§6).
- **`SPLIT_PATTERN`**: karpathy/nanochat's (cl100k's as tiktoken 0.6.0 writes it, with digit runs of one or two)
  without its `[\r\n]*` after a punctuation run, so no token joins `"):"` to its line break:
  `'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++|\s*[\r\n]|\s+(?!\S)|\s+`.
- **Sample** (`train_tokenizer.sample`): `SAMPLE_BYTES` = 13.38e9 bytes of the curated train splits of the file
  stage's 17 sources, each source's share its share of the stage's bytes (`byte_shares`: its token share, §7, at
  `BYTES_PER_TOKEN`), drawn systematically in shard order (row i taken when ⌊(i+1)·f⌋ > ⌊i·f⌋, f the source's target
  bytes over its train split's footer text bytes; a repository file is one document, after its path comment). The
  size is the reference tokenizer's sample's, so the vocabulary trains on as much text as the reference did
  (`train_tokenizer.SAMPLE_BYTES`, §11).
  Codeforces-CoTs (32K stage only) is not sampled. Tokenizer training data should be "representative of
  model training data" (arXiv 2405.05417 §5); across tokenizer training mixes a subset's compression "only varies by 5
  to 6 percentage points", higher as the mix holds more of it (arXiv 2402.01035 §2.3); across 30 languages, weights
  proportional to each language's data gave lower FLORES+ bits per byte than equal ones (arXiv 2610.12144 §5.2, App. D
  Table 6).
- **Authorship units** (SA-BPE's R): the lowercased repository for repository files and functions (SA-BPE's unit);
  every other document its own unit (arXiv 2604.14053 §6 suggests "documents, books, or document collections" for
  natural language, untested there).
- **The draw** (`results/tokenizer_split_merge-vocabularies.json`): 3,988,600 documents, 13,444,424,046 bytes,
  3,429,396 units; code 85.6% of its bytes; drawn bytes 0.99–1.05 of the targets, as the draw takes a fraction of rows
  against footer byte targets and each repository file carries a path comment the footers omit (`sources.texts`, §6).

| Source | Target GB | Fraction of rows | Documents | GB | Units |
|---|---|---|---|---|---|
| stack_edu_python | 1.185 | 0.0178 | 435,558 | 1.187 | 384,949 |
| ultradata_stack_v3_l2_python | 1.462 | 0.0178 | 311,593 | 1.473 | 291,342 |
| ultradata_code_l2_python | 2.631 | 0.0178 | 747,214 | 2.662 | 643,565 |
| stack_edu_cpp | 0.770 | 0.0178 | 285,543 | 0.781 | 232,119 |
| ultradata_stack_v3_l2_cpp | 0.842 | 0.0178 | 219,730 | 0.845 | 145,740 |
| ultradata_code_l2_cpp | 0.645 | 0.0178 | 233,492 | 0.645 | 139,446 |
| ultradata_code_l3_python | 0.352 | 0.0345 | 135,008 | 0.352 | 135,008 |
| ultradata_code_l3_cpp | 0.363 | 0.0345 | 147,939 | 0.363 | 147,939 |
| nemotron_rewriting_python | 2.227 | 0.0345 | 624,869 | 2.227 | 624,869 |
| nemotron_transpilation_cpp | 0.756 | 0.0345 | 176,998 | 0.755 | 176,998 |
| kaggle_notebooks | 0.197 | 0.0383 | 22,001 | 0.207 | 22,001 |
| codexglue_python | 0.008 | 0.0383 | 9,554 | 0.008 | 3,738 |
| climbmix | 1.588 | 0.0341 | 532,506 | 1.586 | 532,506 |
| ultradata_math_l3_qa | 0.081 | 0.0309 | 33,157 | 0.082 | 33,157 |
| ultradata_math_l3_conversation | 0.077 | 0.0309 | 20,645 | 0.078 | 20,645 |
| ultradata_math_l3_multi_style | 0.106 | 0.0309 | 20,995 | 0.105 | 20,995 |
| ultradata_math_l3_textbook_exercise | 0.089 | 0.0309 | 31,798 | 0.089 | 31,798 |

- **Intrinsic** (`results/tokenizer_split_merge-intrinsic.json`, `ebd8058`, evaluation text the 18 sources' val splits,
  6.21 GB): sa-bpe holds 3.624 bytes per token over the file stage and 3.442 on Stack-Edu Python, and passed the
  validity checks the record holds (on 200 val documents per source: chunks cover the text, tiktoken equals the merges
  replayed, every text decodes).
- **Data key** (`Tokenizer.data_key`): SHA-256 of the JSON list [pattern, {BOS: id}, tiktoken file] (the FIM sentinels
  after BOS do not enter it); the tokenized and packed data live in `CORPUS/tokenized/<corpus>-<data key>/`
  (`layout.TOKENIZED`, `layout.split_path`).

## 6. Tokenization and repository samples

`python -m corpus.tokenize` encodes every curated split under the corpus's tokenizer (special-token text encodes as
ordinary text), per corpus and split:

| File | Contents |
|---|---|
| `<split>.bin` | every document's tokens in curated order, uint16, no BOS or padding |
| `<split>.idx` | int64: each document's first offset, then the total |
| `<split>.json` | each curated shard's (or partition's) document count; `pack` packs each one's documents together |

**Documents** (`sources.curated_documents`). A curated row, except in a repository source, where a document is one
sample (§3): its files concatenated in the sample's order. Each repository file is preceded by a line comment with its
path (`# path`, `// path`; DeepSeek-Coder: "a comment indicating the file's path is added at the beginning of each
file", arXiv 2401.14196 §2.2) and ends with a line end, added when missing (`sources.texts`; Python Language Reference
§2.1.2, C++23 [lex.phases] phase 2); a line break in a path is written `\r` or `\n`, and a C++ file ending in a
backslash gets an empty line, so phase 2 splices no comment onto it (C++23 [lex.phases]).

**The corpus crate** (`corpus/crate/`, pyo3 module `nanochat_crate`): `train` (§5), `Grams` (decontamination, §3),
`topological_sort` (below) and `pack` (§8), built by cargo and imported through `native.corpus_crate`
(`native.crate_module`).

**Import-connected samples** (`crate/repository.rs` `topological_sort`, DeepSeek-Coder §2.2): a file depends on the
files its Python imports or C++ includes name, read from its tree-sitter tree and resolved by path within the
repository, without a search or include path: an absolute import by suffix (`a/b/c.py`, `a/b/c/__init__.py`, and per
imported name its submodule); a relative import of level l only at exact paths under the importing file's l-th
ancestor (level 1: its own directory; none above the root); a
quoted include first at its path from the including file's directory, else by suffix; an angled include by suffix.
A sample is a weakly connected component of the dependency graph, its files in dependency order (strongly connected
components in path order, where DeepSeek-Coder's Algorithm 1 orders a cycle by least in-degree); samples follow the
order of their first file; a single-file repository is one sample, unparsed.

**Repositories corpora** (`mixture.REPOSITORIES`: `repositories_python`, `repositories_cpp`): the 32K stage's
documents, one per (lowercased repository, language) across that language's three repository sources. Of a path
several sources hold, the first source's file is kept (`mixture.REPOSITORY_SOURCES` order), and a path one source
holds twice keeps its first row (`repository_samples(one_per_path=True)`); a repository's files are
its samples in order. `tokenize_repo_level_stage` takes train partitions from the last (99) backwards until their
documents hold the stage's repository draw, `CODE_SHARE` × the stage's tokens less Codeforces-CoTs' train tokens, and
adds partition 0 as val. Under sa-bpe: partitions 95–99,
5,434,219,035 train tokens for a draw of 4,567,423,536, Codeforces-CoTs' train tokens being 301,057,488 (computed on
master).

**Parallelism.** Row groups (sources) and (language, partition) tasks (repositories) are the units of work, written in
task order (`pool.map` for row groups, `parallel.ordered` for repository tasks), so the output does not depend on the
worker count; repository tasks in flight are bounded by `MEMORY` / (`PEAK` × the largest partition's text, val or
train, summed over the language's three sources; `repository_window`), each encoding on a thread per core;
`PEAK` = 8.18 is the larger measured peak per text byte (`ru_maxrss` of a spawned worker, corpus tokenizer, `c1a7d7b`,
on each language's largest train partition): 6.52 for Python (partition 41: 3.02 GiB of text, 19.7 GiB at peak), 8.18
for C++ (partition 71: 2.40 GiB, 19.6 GiB).
`topological_sort` parses a partition's files in parallel (rayon). Measured on the GH200 (before → after): `b9769a8`
(a partition per `topological_sort` call, encoding on every core) tokenizing the C++ val partition 386.7 → 16.3 s, the
Python one 712 → 61 s; `9e40df4` and `a20d12a` (a tokenizer per rayon worker, `PerWorker`) pack 189 → 97.5 → 93
CPU-s per shard; `737f2e1` (a shard's documents cut longest first) a four-shard FIM pack 14.4 → 11.5 s. The crate's
release profile (`lto`, one codegen unit, `03d131e`) against cargo's defaults: pack of shard 0 of six splits 36.4 vs
42.9 s, 8,000 training merges on 252,906 documents 20.6 vs 22.1 s, outputs identical.

## 7. Mixture and stages

`corpus/mixture.py`. The stages are sized for `MAX_STEPS` = 69,580 steps of `BATCH_SIZE` = 15 × `MICRO_BATCH_TOKENS`
= 1,474,560 tokens (the d24 plan at `b065ad7`, `docs/training.md` §2; a micro-batch: 12 rows of 8,192 or 3 of
32,768): 102,599,884,800 tokens [arithmetic]. Its last
`REPO_LEVEL_SHARE` = 300B / (5.2T + 300B) = 5.45% of steps (`REPO_LEVEL_STEPS` = 3,795), Qwen2.5-Coder's
repository-level share (arXiv 2409.12186 §3.2.1–3.2.2), are the 32K stage; the contexts are Qwen2.5-Coder's 8,192 and
32,768 (§3.2.1, §3.2.2). Any run draws by the stages' shares at its own size (`weights`). The base run (69,874 steps,
the 32K stage from step 66,063; `docs/training.md` §2) fits the packs, narrowly: at its rows the file stage leaves 0
to 1 spare rows in the four `ultradata_math_l3_*` sources and 839, 945 and 1,930 in `ultradata_code_l3_python`,
`ultradata_code_l3_cpp` and `nemotron_transpilation_cpp` (the supply corpora are read whole); the 32K stage's
tightest corpus, `ultradata_math_l3_conversation`, keeps 71 (computed on master).

| Stage | Context | Steps | Tokens | Rows | Packed train rows |
|---|---|---|---|---|---|
| `FILE_LEVEL_STAGE` | 8,192 | 0–65,785 | 97,003,929,600 | 11,841,300 | 19,320,514 |
| `REPO_LEVEL_STAGE` | 32,768 | 65,785–69,580 (3,795) | 5,595,955,200 | 170,775 | 200,881 |

Weights: `mixture.weights` of each stage at its rows on sa-bpe's packs; tokens: weight × the stage's tokens (computed
on master; no supply corpus overdraws).

| Stage | Group | Sources | Share | Weight | Tokens |
|---|---|---|---|---|---|
| File | repository | the six repository sources | rest of 87% | 0.5929 | 57.51B |
| File | synthetic | UltraData-Code L3 (Python, C++), Nemotron rewriting, transpilation | 0.30 × 87% = 26.1% | 0.2610 | 25.32B |
| File | notebooks | `kaggle_notebooks` | supply | 0.0155 | 1.51B |
| File | functions | `codexglue_python` | supply | 0.0006 | 0.05B |
| File | text | `climbmix` | 10% | 0.1000 | 9.70B |
| File | math | UltraData-Math L3, four formats | 3% | 0.0300 | 2.91B |
| 32K | repositories | `repositories_python`, `repositories_cpp` | rest of 87% | 0.8153 | 4.56B |
| 32K | traces | `codeforces_cots` | supply | 0.0547 | 0.31B |
| 32K | text | `climbmix` | 10% | 0.1000 | 0.56B |
| 32K | math | UltraData-Math L3, four formats | 3% | 0.0300 | 0.17B |

- **Code 87%, text and mathematics 13%** (`CODE_SHARE`): DeepSeek-Coder's "87% source code, 10% English code-related
  natural language corpus, and 3% code-unrelated Chinese natural language corpus" (arXiv 2401.14196 §2), its natural
  language taken to cover web text (10%) and mathematics (3%) [inference]. Qwen2.5-Coder's 7B comparison of
  code:text:math 100:0:0, 85:10:5 and 70:20:10 chose 70:20:10, best on its math and general benchmarks and above 85:10:5
  on code, while 100:0:0 scored higher on its two code benchmarks (49.8 and 40.3 against 48.3 and 38.3; arXiv
  2409.12186 §3.1.2, Table 3).
- **Synthetic code 30% of code** [inference for code]: good mixtures' share of rephrased synthetic data, "converging
  to ∼30% for rephrased synthetic data" (Kang et al., arXiv 2510.01631 §1), measured on web text; its Limitations say
  the findings "may not directly apply to other generation methods (e.g., synthetic code, dialogues)". Code papers
  only bound the share: 10% synthetic code helps in code-only pretraining from scratch (arXiv 2408.10914 §3.4);
  UltraData-Code L2 with its synthetic L3 at 1:1 helps in continued pretraining (arXiv 2610.23960 Table 6; with
  CodeAlchemy's code-qa set instead, EvalPlus falls below L2 alone); annealing a base model on 100B tokens,
  rewritten-only code drops MBPP to 5.6 against 58.7 for raw code and a 50-50 mix keeps 60.3 (arXiv 2606.10087 §3,
  RQ1, Table 4). The share is not ablated. The
  curated synthetic sources hold up to 0.30 × 1.092 ≈ 0.328 of code (§1) [arithmetic]; the file stage's repository
  partitions hold 2.05× their draw at 0.30 and 1.43× at 0 (118.11B tokens; computed on master).
- **The 32K stage keeps text and mathematics:** OpenCoder's anneal keeps "84% of the annealing data" from "the original
  distribution" against "catastrophic forgetting" (arXiv 2411.04905 §2.2).
- **"supply"** sources are read whole once: a fixed share of these small sources would repeat them, and repeating "0.1%
  of the data 100 times" degrades an 800M model to a 400M one (arXiv 2205.10487 Abstract). A supply group's share is
  its packed rows over the run's rows in the stage; "rest" is `CODE_SHARE` less the stage's other code groups
  (`Stage.shares`). A run too short to read its supply whole (a supply share that would leave "rest" negative) takes,
  in `weights`, the supply shares of the stage's sizing rows instead (packed rows over `Stage.rows`, the `MAX_STEPS`
  plan's): the base run and the d16 ablation sets (whose supply is packed at the run's draw, `docs/experiments.md` §2)
  read supply whole; a short run on the corpus's packs trains on the weights of the table above. Before this rule
  (`1dda452`) such runs got negative repository weights and drew no repository rows in the 32K stage (d24 and d8 at 5
  tokens per parameter, d8 at 200 and 2.7: Codeforces-CoTs ≈ 88% of the stage); the `peak_lr` records (`b7f5be9`)
  were run under the rule (their 32K `stage` records weigh `codeforces_cots` 0.0554 in `stage.mixture`).
- **Within a group** sources weigh by their packed rows. `mixture.weights` replays a stage's shares through
  `mixture_rows`; a supply corpus that draws more rows than it holds weighs that many rows less, until none does.
- **Row order** (`mixture.mixture_rows`): run row n goes to the corpus whose interval of the cumulative weights holds
  frac((n + 1) · φ⁻¹), φ⁻¹ = (√5 − 1)/2, a Weyl sequence that spreads every prefix of the run evenly over the corpora
  (Knuth, TAOCP Vol. 3 §6.4); any stretch of the run is computed without the rows before it.

## 8. Packing and fill-in-the-middle

`python -m corpus.pack` calls the crate's `pack` (`crate/pack.rs`) once per curated shard of a tokenized split
(`WINDOW` = one call per core, the crate releasing the GIL), into rows of seq_len + 1 tokens. Output per corpus, split
and context, `<split>-T<seq_len>` (`layout.packed_path`):

- `.pieces`: int64 × 4 per piece in row order (`layout.PIECE_COLUMNS`): its token range [start, end) in the split,
  then its FIM example's range in `.fim` (empty for a piece read left to right);
- `.rows`: int64, each row's first piece, then the split's total number of pieces (rows + 1 entries);
- `.fim`: uint16, the FIM examples' tokens.

**What each stage packs** (`pack.main`; no train document is packed twice):
- File stage: the repository sources' train partitions not taken by the 32K stage and the supply sources, whole; each
  source read in part, its leading shards, one more while its draw at the stage's sizing rows (`Stage.rows`) exceeds
  its rows (`pack.fit`).
- 32K stage: the repositories corpora and Codeforces-CoTs, whole; ClimbMix and UltraData-Math L3, the shards after
  the file stage's, as above.
- Every val split, whole, at the seq_len of each stage that reads it.

**Pieces.** A piece read left to right that ends its document fits its row with its BOS and its target BOS; a longer
document is cut, each cut the window's last statement boundary (the gaps between the items of a tree-sitter
container: module, block, translation unit, compound statement, declaration list, …; repository code only,
`layout.grammar`), else after its last line-ending token, else before its last token that starts a character. A piece
drawn as a FIM example fits its row with its BOS, the example and the target BOS: its window first leaves room for the
three sentinels and shrinks by any excess of the built example. Each piece is its own attention segment (document
masking: IntraDoc lowers perplexity over UniChunk, arXiv 2402.13991 §3.2, Table 1). A shard's documents are cut in
parallel (rayon), longest first.

**Worker stack** (`crate/lib.rs` `nanochat_crate`): rayon's global pool takes the process's `RLIMIT_STACK` soft limit
(64 MiB under `scripts/container.sh`'s `--ulimit stack=67108864:-1`), not std's 2 MiB: a worker waiting in a join runs
stolen jobs on its own stack, so nesting deepens with the pack calls in flight, and at 2 MiB `pack` overflowed a
worker's stack (core dump; fixed in `0de5dfb`, the limit taken from the process since `03d131e`).

**Rows** (`best_fit_decreasing`): pieces longest first, each into the row with the fewest free slots that holds it,
else a new row (Best-Fit-Decreasing, arXiv 2404.10830 §3.1), so no piece is truncated. Within a row the pieces whose
next BOS is a target come first and cut pieces last, which keeps no slot for a BOS after them. A row's pieces come from
one curated shard.

**FIM** (`pack.FIM_RATE`, `RANDOM_SPAN_RATE`, `SINGLE_NODE_RATE`, `LINE_START_RATE`, `SPM_RATE`), in the train splits
of repository code only (`layout.fim_packed`: the six repository sources and the repositories corpora). The scope is
an untested choice: synthetic files and functions, though whole programs in a known language (`Category.code`), get
neither FIM nor statement cuts, both set by one switch (`layout.grammar`). On sa-bpe's packs FIM examples hold
49.8–50.1% of each such corpus's piece tokens (FIM example tokens over those plus the left-to-right pieces' tokens),
29.7% of the 8K stage's tokens and 40.8% of the 32K stage's (each corpus's share weighted by the table's stage weights;
computed on the packs built by `abeb4eb`); StarCoder2's two 50% draws (repository, then chunk) give ≈25% of its source
files (arXiv 2402.19173 §5.1) [arithmetic].

| Parameter | Value | Evidence |
|---|---|---|
| Rate, per piece, drawn before its cut (context-level) | 0.5 | [1] |
| Middle: random character span, else AST | 0.1 | "90% use our AST-FIM, and 10% use Rand-FIM" (arXiv 2506.00204 §5.1) |
| AST middle: Single-Node (node weighted by its bytes), else Aligned-Span (character IoU) | 0.5 | "an equal mix of two masking methods" (arXiv 2506.00204 §3.3) |
| AST middle starting its line: the line's indentation in the middle, and its line break when the middle ends at one | 0.5 | the rule: [3]; the value: the one rate tested, against no rule (`dc76110`, d16 retrain `1671b56`: PSM single-line exact match 0.022 → 0.175, SPM 0.367 → 0.356; below); the line break: the reference middles' layout, untested (below) |
| SPM, else PSM | 0.5 | [2]; AST-FIM's "50% PSM and 50% SPM", untested there (arXiv 2506.00204 §5.1) |

[1] DeepSeek-Coder chose 0.5 (document level, PSM only), as a 100% rate gave its weakest code completion (arXiv
2401.14196 §3.1.2); FIM rates up to 0.9 cost no left-to-right performance (arXiv 2207.14255 §4.2), and "context-level
FIM yields a consistent and significant improvement over document-level FIM" (§4.4); at rate 0.7 AST-FIM kept
HumanEval+ and MBPP+ "nearly identical to the baseline L2R model" (arXiv 2506.00204 §6.1). No cited paper measures
FIM over whole multi-file repository texts, as in the 32K stage (Qwen2.5-Coder applies repo-level FIM unablated, arXiv
2409.12186 §3.2.2). [2] Joint PSM/SPM training matches or beats either mode alone
in Bavarian et al.'s one controlled comparison, except a near-tie (SPM single-line 0.622 joint vs 0.625 PSM-only),
and a pure-SPM model cannot infill in PSM (arXiv 2207.14255 §3.1, §4.3 Table 1); §8.1 recommends joint training.
[3] An AST node starts after its indentation and ends before its line break, so without the rule no AST-FIM PSM
example has a prefix ending in a line break and a middle of whole lines, the layout of line-level infilling (HumanEval
single-line and multi-line infilling mask "different spans of non-empty lines", arXiv 2207.14255 §2.2; all 1,033
single-line and 5,815 multi-line prefixes end in a line break and every reference middle ends with one
[arithmetic: `read_problems`]).

**The line-start rule's basis** (d16 at 20 tokens per parameter; infilling measures `docs/evaluation.md` §4–§5). The
training format matches the evaluation prompts (sentinel ids, sections encoded separately, the BOS ending the middle).
Without the rule (before `dc76110`) the model infilled single lines 16× worse in PSM than SPM (line exact match 0.022
against 0.367, `1671b56`'s baseline; 0.023 against 0.353 at `dc76110`, also on the 982 prompts the 512-key window
covers whole, so not the window); 95.6% of 3,000 packed PSM middles started with non-whitespace [unverified] while
every single- and multi-line HumanEval middle starts with indentation after a line break [arithmetic:
`read_problems`], and the model opened a new definition or file after the end sentinel. AST-FIM's Limitations:
"Better matching training and testing distributions still result in stronger performance" (arXiv 2506.00204 §7).
With `LINE_START_RATE` = 0.5 (`dc76110`: 15.6% of PSM examples on train shards have the line-level layout; on
the packs before `81fb386` 15.1–15.7% for Python and 8.6–12.6% for C++, first 100k FIM examples per corpus),
the d16 retrained on the repacked data (`1671b56`) infills single lines in PSM at line exact match 0.175 (pass@1
0.037 → 0.161; multi-line 0.008 → 0.045), SPM 0.367 → 0.356, code bpb unchanged (8K 0.4392; 32K 0.4210 → 0.4208).
**The line break.** An AST node also ends before its line break, so without the line break's rule a PSM middle rarely
carries the final "\n" every single- and multi-line reference middle ends with: 20,000 sampled FIM examples per
corpus, PSM ones counted, on the packs of the curated build before decontamination had 1.06% of middles ending in
"\n" in `stack_edu_python`, 0.80% in `ultradata_stack_v3_l2_python` and 0.58% in `repositories_python`. With it (the
line-start draw also adding the line break when the middle ends at one, `pack.rs` `fim_example`), a uniform sample of
20,000 FIM examples per corpus of the packs built by `abeb4eb` has 35.2%, 37.2% and 35.9% (26.4% in `stack_edu_cpp`)
of PSM middles ending in "\n", whole lines (prefix empty or ending in "\n") in 34.0–36.3%, while 19.2–22.6% still start
a line and stop before its line break (the other draws), so both layouts are trained. The rule also fires at column 0,
where the indentation step changes nothing: its share is that of nodes starting their line, not of indented ones. Its
effect on infilling is not measured.

- A FIM piece is decoded to text; its middle is a random span with probability 0.1, else an AST-FIM span on its
  tree-sitter parse, or a random span when the piece does not parse or has no node below the root.
- Prefix, suffix and middle are encoded separately (arXiv 2207.14255 §3): PSM `<｜fim▁begin｜> P <｜fim▁hole｜> S
  <｜fim▁end｜> M`, SPM `<｜fim▁begin｜> <｜fim▁hole｜> S <｜fim▁end｜> P M` (App. D, variant 2); the BOS after the middle
  is its target.
- Every draw comes from ChaCha8 keyed by (document, piece, corpus), so packs do not depend on the order of work, and
  equal (document, piece) in two FIM-packed corpora draw independently; the corpus enters as its name's hash, so every
  tokenizer arm draws a corpus's examples alike (`81fb386`). Under a (document, piece) key such pairs shared every draw
  (agreement 1.0 on 182K–19.0M pairs per corpus pair), yet shared a training step no more often than independent draws
  (22 against 19.7 ± 4.2 over the base plan's first 2,000 steps).

## 9. Loading

`loader.PretrainingBatches` memory-maps the packed splits of a stage's mixture `{corpus: weight}` (`mixture.weights`
on the packed rows, called from `train/stages.py`) and assigns run rows to corpora by `mixture_rows`; each corpus's
rows are read in the order the caller gives (a seeded permutation, `docs/training.md` §8), resuming after the rows a
start position consumed.
- A row (`assemble`): a BOS before every piece, the piece's tokens (reading "fim": its FIM example, if it has one),
  padding.
- Labels: the BOS after a piece that ends its document, or (reading "fim") after a FIM middle, is a target; after a
  piece its document continues, the BOS is an input only and an ignored label (`IGNORE_INDEX` = −1). Padding is BOS
  as input, `IGNORE_INDEX` as label.
- `cu_seqlens`: one attention segment per piece and one for a row's padding when its inputs hold any (from the
  position after its last piece; an all-padding row, order entry −1, is one segment) (FA4's varlen layout), a fixed
  count per loader (micro-batch rows × (the most pieces in a row of its packed splits + 1)) so a compiled model sees
  one shape, the unused entries empty; `segments` is the count used and `lengths` the segments' lengths (attention
  work for `metrics.executed_model_flops`, `docs/evaluation.md` §5).
- Iteration stops when a corpus's rows run out, without the partial micro-batch.

## 10. Building the corpus

Every command runs in the `nanochat` image from the repository's root (`docs/training.md` §9 builds it and runs
`scripts/setup.sh`, which pulls the dataset), through `scripts/container.sh` on the GH200 or, for the text stages
(prepare, curate, decontaminate, publish) on any host with cgroup v2 (`parallel.memory` reads
`/sys/fs/cgroup/memory.max`), as

```
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e PYTHONPATH=/workspace/nanochat -w /workspace/nanochat \
  -v $PWD:/workspace/nanochat nanochat python -m corpus.<step> ...
```

(steps that read the Hub add `-e HF_TOKEN=<token>`, as `scripts/container.sh` passes it; the image sets `HF_HOME`
under `artifacts/cache`, `docker/Dockerfile`).

Once, to build the published dataset:
1. **Prepare**: `python -m corpus.prepare` fetches and standardizes every source's extent (§1–§2).
2. **Curate**: `python -m corpus.curate` (§3).
3. **Decontaminate**: `python -m corpus.decontaminate` drops the curated train rows sharing a 10-gram with a scored
   benchmark and writes `curated/decontamination.json` (§3).
4. **Tokenizer**: `python -m corpus.train_tokenizer` trains sa-bpe on the sample into `CORPUS/tokenizer`
   (`tokenizer.TOKENIZER`, §5). The shipped tokenizer was trained by the same method on an earlier curated build;
   this step does not reproduce it (§5).
5. **Publish**: `python -m corpus.publish push` uploads the curated shards, their decontamination record and the
   tokenizer (§4).

From a pull of the dataset (`python -m corpus.publish pull`, run by `scripts/setup.sh`: the curated shards, their
decontamination record and the shipped tokenizer):
6. **Tokenize**: `python -m corpus.tokenize` (§6), ending with the 32K stage's repository partitions.
7. **Pack**: `python -m corpus.pack` (§7–§8); it fails (`pack.fit`) when a corpus holds fewer packed rows than its
   draw: a corpus read in part runs out of shards, or a corpus packed whole is overdrawn.

## 11. Implementation reference

Detail of the code not stated in §1–§10, by module and symbol.

**`corpus/__init__.py`, `corpus/sources.py`, `corpus/schema.py`.** Files under `CORPUS`:

| Path | Contents |
|---|---|
| `raw/<source>/<Hub path>` | fetched Hub columns (`prepare.fetch`) |
| `standardized/<source>/<k>.parquet` | fetched file k in its category's schema |
| `checked/<source>/<k>.parquet` | a Python source's standardized file after the check |
| `curated/<source>/<split>/<k>.parquet` | curated shard k (the published dataset; `sources.CURATED`) |
| `curated/decontamination.json` | the train rows decontamination dropped (`decontaminate.RECORD`, §3) |
| `tokenizer/tokenizer.{tiktoken,pattern}` | the corpus's tokenizer |
| `tokenized/<corpus>-<data key>/<split>.*` | tokenized split (`.bin`, `.idx`, `.json`), packs `-T<seq_len>.*` |
| `README.md` | the dataset card (`publish.push` writes it) |
| `shuffle/<source>/part=<residue>/`, `spill/` | `curate`'s transient repository partitions; DuckDB's spill |

- `Source`: `files` maps each language (a Linguist name for code) to the glob of its Hub files (`codeforces_cots` maps
  two; `Source.language` requires one); `text` is the fetched column holding the text. `Category.name` keeps
  categories with equal schemas, groups and flags distinct (EXERCISE and NOTEBOOK); `group` is a DuckDB expression;
  `code`: the content is one whole program, which must parse when it is Python. `SAMPLE_FILE` numbers
  import-connected samples per shard (`sample`).
- `texts`: a path's `\r`/`\n` are escaped so the comment stays one line. `repository_samples` returns the table as one
  chunk; its callers (`curate.curate_partition`, `tokenize`) read `content` as `large_string` (64-bit offsets hold a
  partition beyond 2 GiB).
- `text_bytes` counts page headers, dictionary and level data where present (an overestimate of a plain-encoded
  column's text; a dictionary-encoded column's repeated values count once [inference]).
- `COMPRESSION` = zstd for every Parquet file the corpus writes.

**`corpus/prepare.py`.** `extent`'s SHA-1 path order is uniform and its prefixes nest as the extent grows.
`swh_store` is one client per process (connections reused); the descriptor limit is raised to the hard limit because a
row group's blob requests are all in flight at once (`blob_texts`). `fetch` reads only `columns(spec)` (Stack-Edu's
metadata whole), row group by row group. `lines`: a file may begin with several U+FEFF (re-encoded once per mark);
both languages drop one and reject a second, so all are dropped. Standardizers: `stack_edu`, `repository_file`,
`conversation` (the R1 prompt ends with a line end: in every curated row the trace's `<think>` starts a line, computed
on master), `function`, `text_only` (exercises, synthetic files, notebooks, web text). `prepare` removes the source's
standardized and checked files before rebuilding.

**`corpus/curate.py`.**
- `parses` treats `ValueError` (CPython 3.12.3's for `f"{x:{y=}}"`, python/cpython#121130; a null byte is a
  `SyntaxError` since 3.12), `MemoryError` and `RecursionError` as failures; `two_to_three` returns None on any fissix
  exception (its grammar's parse and tokenize errors and more).
- `repaired`: steps 1–2 enter only where they change the text, 2to3 always; a file 2to3 alone repairs keeps its tabs
  and placeholders.
- `check`: every file's row groups are one stream (no file's slowest group idles the pool), as many in flight
  as `MEMORY` holds of the largest standardized group (a checked group is about its standardized size [inference]); a
  result waits until every earlier one is written.
- `duplicates`: DuckDB `row_number() OVER (PARTITION BY id ORDER BY source, file, row)`; drops `n > 1` and the id of
  the empty text (`EMPTY`). `undropped`: anti-join on (file index, row number).
- `shuffle`: repository rows as Hive partitions `shuffle/<source>/part=<residue>/` (the code creates only
  `CORPUS/shuffle`; DuckDB's `COPY … PARTITION_BY` creates the rest), removed after `curate_partition`. Workers =
  min(cores, `MEMORY` / (2 × largest)), largest the biggest partition's uncompressed Parquet bytes; the factor 2 is not
  measured [inference]: a worker holds at least three copies of its partition (the read table, its sorted copy, the
  contents as Python strings for `topological_sort`). It sets the worker count only, not the output.
- `shard_documents`: `ntile(SPLIT) OVER (ORDER BY id)` over train rows. `database()` turns off
  `preserve_insertion_order` (DuckDB 2.0 documentation, Guides, Performance, "Out-of-Memory Issues", the guide of
  `parallel.MEMORY`). `RECURSION` and `SAMPLE_ROWS` are the settings the published dataset was built with.

**`corpus/publish.py`.** `CONTENTS` and `COLUMNS` are the card's account of each folder and column.

**`corpus/tokenizer.py`.** BOS starts every piece of a packed row; as a target it ends a document or a FIM middle.
`token_bytes` holds a special token's text; `token_lengths` its document length 0. `get_tokenizer` reads line i of
`tokenizer.tiktoken` as token i (base64, rank i), once per process.

**`corpus/train_tokenizer.py`.** `SAMPLE_BYTES` comes from the reference tokenizer's curated splits' footers (that
data is not on the Hub) [unverified]. `byte_shares` counts tokens at `BYTES_PER_TOKEN`. A unit id is the lowercased
repository (one unit across the sources holding it) or the document's `id` (a hex SHA-1, never a GitHub owner/name),
numbered densely in sample order. `sample` reads row groups in parallel and keeps the sample in order. `main` writes
merge i as token 256 + i (the bytes of its pair joined); it creates `CORPUS/tokenizer` only after training, and fails
there if the directory exists.

**`corpus/crate/train.rs`** (rustbpe's design, karpathy/rustbpe). Documents are split into chunks by the pattern (one
fancy-regex per worker, one chunk table per rayon fold task and hash shard, shards merged in parallel); each distinct
chunk is a `Word` (byte ids, count, sorted distinct sources). Pair frequencies update incrementally; a lazy max-heap
requeues a popped entry whose frequency changed; entries hold the words a pair may occur in. R is exact without a
per-(pair, source) table: R of a pair is the size of the union of its words' sources, counted (a `fixedbitset` per
parallel task of at least `min_len` words, where marking costs about as much as clearing) only when its entry reaches
the top. Between counts the entry keeps an upper bound (initially the number of sources): an existing pair's F and R
only decrease (a merge only creates pairs holding the new token), R changes only when F does, and a created pair
occurs only in words of the merged pair, so its R is at most the merged pair's. An entry is taken only when F and R
are exact, so every merge is the argmax of the exact score; a pair seen in one source scores 0 and merges only after
every pair seen in several; ln instead of log changes no argmax. Ties: higher F, then the smaller pair (rustbpe's
order).

**`corpus/crate/repository.rs`.**
- Queries: Python pattern 0 `import_statement`, 1 `import_from_statement` (`from __future__` is its own node kind,
  §7.11.1); C++ pattern 0 a quoted header name (q-char-sequence), 1 an angled one; an include of a macro
  (`[cpp.include]` 4) names no path.
- `Index`: an exact path matches every file with that path (paths may repeat).
- `python`: `import a.b.c` matches by suffix `a/b/c.py`, `a/b/c/__init__.py` (not the parent packages'
  `__init__.py`, which the import also runs, §5.2.1); `from a.b import n` adds `a/b/n.py`, `a/b/n/__init__.py`;
  relative level l (dots, `importlib.util.resolve_name`): base = the file's l-th ancestor (`Path::ancestors`; above
  the root no file), target = base/m, exact paths `target.py` (with a module m), `target/__init__.py`, per name
  `target/n.py`, `target/n/__init__.py`.
- `cpp`: a quoted include is searched first in the including file's directory as GCC and Clang do; `cleaned` (Plan 9's
  `cleanname`, as `path-clean` ports it) rejects a leading `..`.
- `topological_sort`: every import node of a tree is used, also of a tree with syntax errors (error recovery keeps the
  nodes it parsed); a file's dependencies exclude itself, sorted and deduplicated.

**`corpus/crate/pack.rs`.**
- `Grammar`: containers are Python `module`, `block`; C++ `translation_unit`, `compound_statement`,
  `declaration_list` (namespace, linkage specification), `field_declaration_list`, `preproc_if`, `preproc_ifdef`,
  `preproc_else`, `preproc_elif`, `preproc_elifdef`, `case_statement`. Everything else is one item (a compound
  statement with its clauses, a decorated or template declaration, a directive). Header fields `condition`, `name`,
  `value`, `alternative` (a directive's condition or name, its `#else`/`#elif` branch, a case value; a Python match
  block's first case clause) are not items: the first child of a container in each field (`child_by_field_id`).
- `statement_gaps`: an item ends before its final line break when its span holds it (a directive), as a Python
  statement does; an ERROR item drops only the gaps on either side of it (the root's leading and trailing ranges stay);
  comment lines ending a block belong to it (tree-sitter attaches them); walked with a stack (nesting depth unbounded).
  Gaps are disjoint (a container's gaps lie inside an item of its parent). `statement_cuts`: the cut follows the first
  line end in the gap, after the item before it (in the root's leading range, the first line), so the gap's other
  comments and blank lines stay with the item after it.
- `fim_draw`: the key's first 24 bytes are (document, piece, corpus) as little-endian u64 (`pack.py`: the first 8
  bytes of the corpus name's SHA-256); the first draw decides FIM.
- `ast_middle`: named nodes are the syntax tree (keywords, punctuation anonymous). Single-Node draws a node below the
  root with probability proportional to its bytes (named nodes of a tree without errors span at least one byte).
  Aligned-Span takes a `random_span`, its lowest named node, and the run of that node's named children of largest
  character IoU with the span (arXiv 2506.00204 §3.3, "character-level Intersection over Union"; characters counted as
  the bytes that are no UTF-8 continuation byte), the node itself if no child meets it; byte IoU would choose another
  middle in 204 of 160,000 sampled FIM examples (1.3e-3 of FIM tokens; `81fb386`). For a fixed last child IoU is
  largest at the last start ≤ s or the first start ≥ s (moving the start further out grows the union only, further in
  shrinks the intersection only), likewise for the end, so four candidates suffice.
- `fim_example`: line-start extension applies only when the text between the line start and the node is spaces or
  tabs, and then takes the middle's following "\n" too when the middle ends right before one; sections are encoded
  with tiktoken-rs's `CoreBPE` (OpenAI tiktoken's core, fancy-regex) one encoder per worker (`lib.rs` `PerWorker`: a
  Regex's clones share one scratch pool that workers contend for).
- `pieces`: the statement cuts are computed once per document on first need; a cut leaves at least one token to the
  rest; a left-to-right piece ends its document when at most row_len − 2 tokens remain, else is cut within
  row_len − 1; a FIM window starts at row_len − 5 and shrinks by the built example's excess.
- `best_fit_decreasing`: a stable sort (equal lengths keep document order); rows of one kind with equal room wait
  LIFO; at equal free slots a targeted-only row is taken before a row with a cut piece.
- `pack_shard`: a piece takes the longer of its range and its example in its row; documents are cut one task each
  (`with_max_len(1)`), longest first, so no long document starts last or queues behind others on one worker.
- `pack`: the mmaps are safe as nothing writes a tokenized split's `.bin` or `.idx` while `pack` runs (it writes only
  the `-T<seq_len>` files); a token starts a character unless its first byte is a UTF-8 continuation byte `10xxxxxx`
  (RFC 3629 §3); `token_bytes` index = rank.

**`corpus/crate/lib.rs`, `Cargo.toml`.** The pool's stack is the soft `RLIMIT_STACK` (the main thread's stack and
glibc's default for threads), as `std::thread::Builder` ignores the limit. Dependencies: `bytemuck` (typed views of
mapped and returned buffers), `fancy-regex` 0.19 (tiktoken 0.14's engine), `fixedbitset` (R), `libc` (`getrlimit`),
`memmap2`, `path-clean`, `petgraph` (SCCs by `kosaraju_scc`, iterative; union-find), `rand`/`rand_chacha` (FIM draws),
`rayon`, `rustc-hash`, `tiktoken-rs`, `tree-sitter` with the Python and C++ grammars.

**`corpus/tokenize.py`.** `encode` runs longest first (LPT); tiktoken releases the GIL, so a repository task encodes on
a thread per core. `repository_documents` ranks sources so the first source's file of a (repository, path) sorts
first. `main` submits every split at once (the pool runs row groups in submission order, so no core waits on a small
split), then runs the 32K stage in a pool of `repository_window()` workers, so idle workers hold no memory beyond the
tasks in flight. `tokenize_repo_level_stage` runs val partition 0, then train partitions from the last, both languages
each, and writes the chosen ones from the tokens already computed. Output directories are new per corpus and data key
(no resume).

**`corpus/pack.py`.** `packed_shards` packs a shard only as the window reaches it. `write` offsets each shard's FIM
range into the split's FIM tokens and writes each row's first piece. `pack_stage` starts each corpus read in part with
one shard; `fit` takes more while `weights` reports an overdraw. Val splits
are packed in threads beside the stages, so a val shard's longest document does not hold up the cores.

**`corpus/loader.py`.** `orders`: per corpus, row ids in read order, −1 an empty (all-padding) row; `start` skips each
corpus's rows `mixture_rows` assigns to the consumed run rows; `reading` = `"fim"` or `"left_to_right"`. `segments` is
a host tensor because the attention reads only the first `segments + 1` entries of `cu_seqlens`
(`model/ops/flash_attn.py`), FA4 spending time on every segment given; `lengths` is read without a device
synchronization. Batches are copied from page-locked memory, asynchronously. `assemble` identifies a document end by
the piece's end in `.idx` offsets.

**`corpus/mixture.py`.** `Stage.shares` takes amounts in tokens (before packing: the tokenizer sample,
`train_tokenizer`) or packed rows (after); a Stage's `tokens` are the `MAX_STEPS` plan's, and any other run draws by
its shares at its size. `PART` maps each source of a group read in part to its group (text and math have the same
sources in both stages); `IN_PART` groups read the same fraction of each source (`prepare.extent`, sized by `draw`).
`REPOSITORIES` is one corpus per language, as a pack call parses one grammar. `SPREAD` was checked per tokenizer arm
(`train_tokens_over_draw`, §1).

**`corpus/parallel.py`.** Workers are spawned (no inherited threads or state). `memory` reads the container's cgroup v2
`memory.max` (the GH200 host mounts cgroup v2); `max` means no limit. `MEMORY` is half of it, as DuckDB's guide
("Out-of-Memory Issues") advises a `memory_limit` of 50–60% because some operations bypass its buffer manager.
`ordered` keeps at most `window` tasks outstanding and yields results in task order.

## References

| arXiv | Work | § |
|---|---|---|
| 2102.04664 | CodeXGLUE | 1 |
| 2205.10487 | Hernandez et al., Scaling Laws and Interpretability of Learning from Repeated Data | 7 |
| 2207.14255 | Bavarian et al., Efficient Training of Language Models to Fill in the Middle | 8 |
| 2401.14196 | DeepSeek-Coder | 3, 5, 6, 7, 8 |
| 2402.01035 | Dagan et al., Getting the most out of your tokenizer for pre-training and domain adaptation | 1, 5 |
| 2402.13991 | Zhao et al., Analysing the Impact of Sequence Composition on Language Model Pre-Training | 8 |
| 2402.19173 | StarCoder2 and The Stack v2 | 1, 5, 8 |
| 2404.10830 | Ding et al., Fewer Truncations Improve Language Modeling | 8 |
| 2405.05417 | Land and Bartolo, Fishing for Magikarp (under-trained tokens) | 5 |
| 2408.10914 | Aryabumi et al., To Code, or Not To Code? Exploring Impact of Code in Pre-training | 7 |
| 2409.12186 | Qwen2.5-Coder | 3, 7, 8 |
| 2411.04905 | OpenCoder | 7 |
| 2502.02737 | SmolLM2 (Stack-Edu) | 1 |
| 2506.00204 | AST-FIM | 8, 11 |
| 2510.01631 | Kang et al., Demystifying Synthetic Data in LLM Pre-training | 7 |
| 2512.20848 | Nemotron 3 Nano (Nemotron-Pretraining-Code-v2) | 1 |
| 2602.09003 | Data Science and Technology Towards AGI Part I: Tiered Data Management (UltraData-Math) | 1 |
| 2604.14053 | Chizhov et al., From Where Words Come (SA-BPE) | 5 |
| 2606.10087 | CodeAlchemy | 7 |
| 2610.12144 | Language-specific effects of tokenizer choice | 5 |
| 2610.23960 | UltraData-Code | 1, 7 |
