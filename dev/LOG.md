# Experiment Log

A running summary documenting some experiments and findings. Started ~Jan 7 2026.

---

## 2026-09-27: nanochat/data/pretrain restructured around what code needs; one fetch path, no verify, no seed

The code corpora have handling the general-domain ones do not (materialized from a source, overlap and downsampling at read time, segmentation at line ends), so it now lives in `pretrain/code/`: `__init__.py` (OVERLAPS, SAMPLING, `segment`), `swh.py` (Stack-Edu and RefineCode from Software Heritage) and `stack_v3.py`. Every stage stays shared, as in NeMo Curator (code filters inside the text stages), dolma (code taggers) and datatrove. `pretrain/` went from 966 to 735 lines. Every change was checked against a copy of the code before it, on the same inputs.

- **Layout, as evaluated.** The proposed `stack_edu.py`, `refinecode.py` and `text.py` were not written: the two Software Heritage corpora differ only in their metadata query (one branch in `swh.build_manifest`), and a general-domain corpus needs only its Hub listing and a plain download (two branches in `fetch.py`). Corpus kind is now checked in `fetch.py` and `compile.py` (plus the tokenizer's per-language statistics); the corpus-name checks in verify and the `FETCHERS`/`EXTRA_COLUMNS` maps are gone.
- **One fetch path.** Every corpus's `manifest.json` lists the files of each split; fetch writes the missing val files and the first `--max-files` train files in parallel, each renamed into place once complete, and a file counts as fetched when it exists. Removed: code units with sidecar JSONs, `completed_units`, the wave sizing (`os.cpu_count()` and a files-per-unit estimate), per-unit statistics, and the text manifests' sha256 map.
- **Stack v3.** A train file is one row group of a part, read from the Hub by byte range instead of downloading the whole part; the manifest lists 92,162 row groups of 8,191 parts from their footers (the Hub rate-limits the reads; two builds wrote the same manifest). A file is the source's own columns flattened to one row per source file, not a hand-built 19-column table: 12 of 12 row groups equal the old tables in every shared column. A single row group reads in 3.6 s; 20 in parallel in 9.2 s.
- **Schema and checks removed.** `DOCUMENT_COLUMNS`, `source_dataset`, `content_hash`, `in_refinecode` and null-filled columns existed for verify; each code file now keeps the columns every code corpus has plus its source's own. `verify.py` is removed (its checks re-tested what fetch had just computed; its reports were one-off measurements, recorded above). The Software Heritage fetch keeps only the failure that occurs, a blob absent from the bucket: its sha1 check and corrupt/undecodable branches never fired on the 42k blobs sampled across both manifests or on any fetched shard, and every fetched text still re-encodes to its blob's bytes (302,329 of 302,329 documents).
- **SEED removed; manifests rebuilt.** The orders hash the data itself: md5(repository) picks the val repositories, md5(document_id) orders the train files, md5(part path) orders Stack v3's parts. Stack-Edu rebuilt in 399 s, RefineCode in 94 min (398 GiB of The Stack v2 re-downloaded and joined), Stack v3 in minutes. Both Software Heritage manifests hold the same rows and values as the seed-42 builds (0 differences in every column, `in_stack_edu` and the 17 tied RefineCode `lang` tags included); val is one shard of whole repositories that shares none with train (Stack-Edu 16,823 repositories, 249.8 MB; RefineCode 10,850, 246.7 MB), and every train shard is under 250 MB in md5 order. Which files are val and train changed, by design.
- **Parity.** Compiled rows and statistics byte-identical, old code against new, on 10 files: ClimbMix train and val, 3 Stack v3 row groups, Stack-Edu train and val, RefineCode train (with and without `--without=stack_edu`) and val (with). The tokenizer trained on ClimbMix is byte-identical (fingerprint 7b4fd1043e2d16bb).
- **Smoke, all six corpora** (val and 1 train file each; Stack v3 12 row groups): tokenizer on the six-corpus mixture, compile (RefineCode with `--without=stack_edu` drops 25,068 of 66,885 train documents: overlap and Java/HTML downsampling), base_train depth 2 for 20 steps (planned shares 0.164-0.169 against 0.167; val bpb 3.89 → 3.10) and base_eval bpb for all 12 splits.
- **Found: compile memory on ~1B-token files.** A worker peaks at 5.0 GiB on a Smol-Data or TxT360 file (tokens held for whole-file packing, plus decontamination of their ~12k-document row groups), against 1.0-1.5 GiB for ClimbMix and code; 20 such workers could reach ~100 GiB of 121 (DATA_ROADMAP Open 9).

---

## 2026-09-27: Fetch audited for accuracy and rate; Software Heritage text kept as the original bytes

Checked whether the six corpora are fetched accurately and effectively, against their live sources and against samples drawn across the whole Software Heritage manifests (not only their first shards).

- **Sanitizer removed.** `sanitize.py` deleted code. Its copyright-header step removed the leading comment block that Pygments lexes, and Pygments lexes C/C++/Objective-C preprocessor directives and PHP's `<?php` as comments, so the `#include`/`#import` lines after a copyright comment went with it: on 20,000 uniformly sampled train files, 0.90% of Stack-Edu's and 2.10% of RefineCode's documents lost code (Objective-C 94.5%, C 13.7%, C++ 11.6% of RefineCode's), and 6 sampled files were dropped as empty (5 headers whose only code was directives, and a 4.7 KB Swift file with a class that Pygments lexed entirely as comments). Its email pattern also rewrote asset names such as `icon@2x.jpg` as `<EMAIL>`. Fetch now stores each blob's text as it is, decoded with its `src_encoding`; `detect-secrets` left the dependencies (Pygments stays, required by ipython and pytest). Stack v3 is unchanged: its publisher redacts PII. Re-fetched val and 1 train shard of each: Stack-Edu 161,457 documents, 1 dropped (absent from the bucket), RefineCode 130,712, 0 dropped; every stored text encodes back to its blob's exact bytes (4,176 of them not UTF-8), and verify passes.
- **Sources.** All seven pinned revisions (six corpora and The Stack v2) resolve and are still each repository's head. Listings match the registry: ClimbMix 6,542 train + 1 val file, Smol-Data 99 + 1, TxT360 827 + 1, all with an LFS sha256; Stack v3 8,192 parts (3.55 TB).
- **Stack-Edu.** The manifest holds all 167,063,359 metadata rows, one per blob (11,776 listed paths have two versions, both listed by the source). In a uniform sample of 19,989 train blobs every blob is present, matches its ID and decodes; all 35 source encodings are known to Python, and 30 blobs of each of the 34 non-UTF-8 ones decode.
- **RefineCode.** 19,999 sampled blobs: all present, matching and decodable. The 8.25% of the metadata that does not resolve is files The Stack v2 does not have at that path: on its Go and Rust partitions, a case-insensitive match recovers 22 of 277,853 and 3 of 89,073, and 10-15% have the same file name at another path of the repository (RefineCode listed another snapshot). 0.057% of listed files resolve to two blobs, mostly the `master` and `main` branches of one repository; both are kept (RefineCode's recorded size singles one out in only 1,419 of 3,357 Go cases). Single-blob Go and Rust listings match RefineCode's recorded size in 89.4% and 92.7% of cases.
- **Rate.** Fetching is latency-bound: one process does 317 files/s with Python's default pool (24 threads), 820 with 64 and 1,610 with 128; 20 processes do 5,025 with the default pool and 11,128 with 64 threads each (120,000 blobs each). A full Stack-Edu plus RefineCode fetch (473M blobs) takes ~26 h as is; raising concurrency needs a chosen number, so it is left as a decision (DATA_ROADMAP Open 8).

---

## 2026-09-26: nanochat/data/pretrain: tuning values removed at their source

Every literal and branch of `pretrain/` was checked against what the experiment needs; two tuning values were removed by deriving them from the data or dropping them, and every change was checked against a copy of the code before it, on the same inputs (ClimbMix 1 file, Stack v3 1 part, TxT360 1 file, and val).

- **Packing buffer (`--buffer-docs`, 16000) removed.** Best fit now runs over all documents of a raw file, which is already the packing unit, so there is no buffer size. It packs at the floor set by documents longer than a row: a ClimbMix shard crops 11.39% (floor 11.3%, buffer 11.84%), a TxT360 file 36.82% (floor 36.7%, buffer 48.44%, so 22.5% more rows from the same text). Packing a TxT360 file takes 15 s against 73 s to tokenize it on 20 threads; a worker holds one file's tokens (2.07 GiB for TxT360). Compiled rows change; the counts before packing (documents, drops, segments, pieces, tokens) are identical per file. Stack v3's crop moves only from 12.30% to 11.75%: its loss is the crop rule (Open 6), not the lookahead. On 20,000 synthetic documents every row is 2049 tokens of whole documents plus at most one prefix at its end, and no document is used twice.
- **Decontamination bit filter (`FILTER_BITS`, `EvalIndex`) removed.** A document's n-gram hashes are sorted and binary-searched in the sorted eval hashes; sorting keeps the searches local (12.6 s per 125M synthetic lookups, against 2.6 s with the filter and 78.9 s unsorted). The eval n-gram set is identical (4,429,728 hashes) and so are the masks (0 of 468,387 documents differ, 446 contaminated). A compile of the samples takes 3-4 s longer (ClimbMix 50 → 53 s, Stack v3 10 shards 52 → 56 s). `<split>.json` records the set's sha256.
- **GPU only.** The loader pins every batch; the CPU branch is gone. Batches are identical to before (300 micro-batches each: train seeded from row 1000 and stored-order val, 2 ranks).
- **Stack v3's empty-file filter removed.** Its publisher already drops empty files: 0 in 297,379 documents of 2 parts, in repositories where empty files are common. Fetched shards are byte-identical to before; verify still checks that no document is empty.
- **One JSON writer.** Compile writes `<split>.json` with `storage.write_json`, as fetch writes manifests and sidecars.
- **Kept, with their source.** `SEED` (the manifests' split and order seed; changing it means new manifests), `SHARD_BYTES` and `ROW_GROUP` (ClimbMix's shard size and row groups: unit and val size, read batch), `CHUNK` (kept at the user's request), `PHI`, the FNV multiplier, 2²⁵⁶, the 13-gram rule, `--seq-len`, RefineCode's published shares, sanitize's RFC formats. `decontam.strings` still walks tuples: the eval task classes return them (110,110 items checked). Software Heritage drops stay: blobs absent from the bucket and copyright-only files occur (1 and 28 in 298k).

---

## 2026-09-26: Code corpora verified against their sources; Software Heritage manifests rebuilt

All six corpora were fetched (1 train unit and val each) into scratch and checked against their original sources, independently of the fetch code; then both Software Heritage manifests were rebuilt from the pinned sources with the current code.

- **Accuracy.** Stack-Edu and RefineCode: one manifest row per blob, no notebooks, no repository in both val and train, shards in seeded order; 800 kept blobs re-downloaded over plain HTTPS all have sha1 equal to their blob ID and re-sanitize to exactly the stored text; every document's metadata equals its manifest row; the 29 dropped blobs are 28 copyright-header-only files and 1 absent from Software Heritage (HTTP 404). Stack v3: 297,379 documents equal to the source part files field for field; no repository in both val and train. ClimbMix, Smol-Data, TxT360: every local file has the Hub's current sha256 at the pinned revision, val is excluded from train, file counts match the registry. `read_training_documents` equals an independent re-derivation on all six, including RefineCode's sampling and `--without=stack_edu`. verify passes on all six.
- **Val was not one shard.** Val took whole repositories until it passed 250 MB, and the overflow became a tiny second val unit (7 files for Stack-Edu, 1 for RefineCode). Val now keeps only the repositories that fit, so it is exactly one shard (Stack-Edu 16,337 repositories, RefineCode 9,505; 250.0 MB each); 1 boundary repository per corpus moved to train.
- **Blob deduplication was not deterministic.** RefineCode's metadata lists some (repository, path) twice with different `lang` tags; ordering by repository and path left the pick to DuckDB, and two builds disagreed on 24 blobs. The whole row now breaks ties (checked in DuckDB 1.5.5 to pick the same row in either input order). `lang` is metadata only: training and selection never read it.
- **Rebuilt.** Stack-Edu in ~350 s, RefineCode (398 GiB of The Stack v2 metadata, joined) in ~86 min, twice. Across the builds every blob, split, shard and metadata column is identical except the val fix and `lang` of 17 tied RefineCode blobs, all now the smaller tag, as the tie-break predicts; Stack-Edu's two builds are identical, `manifest.json` byte for byte. 0 `in_stack_edu` flags disagree with the rebuilt Stack-Edu manifest. The earlier Stack-Edu `manifest.json` also carried fields from an older build (`splits`, `unique_blobs`, ...) that nothing reads. Fetching the new val shards: Stack-Edu 81,342 documents (2 dropped), RefineCode 62,849 (10 dropped); verify passes.
- **Source quirk, harmless.** Stack-Edu's `length_bytes` differs from the blob's actual size for ~18% of files (the text is still exact by sha1); it only sets how many files go into a ~250 MB shard. Mismatching files are 17 times as likely to contain text the sanitizer changes (34% against 2%), which suggests the sizes were measured after PII redaction (not confirmed).
- **Docstrings.** The comments and docstrings of `pretrain/` were rewritten shorter (297 to 233 lines); every module's syntax tree without docstrings is unchanged.

---

## 2026-09-26: nanochat/data/pretrain audited against the refactoring criteria

Every pretrain module was checked line by line for dead code, duplicated logic, defensive branches outside the data contract, unjustified literals, and handwritten mechanisms that a mature library covers. An independent review did the same pass, and every change was checked against a copy of the code before it, on the same inputs.

- **Fixed.** `verify.py` hashed files with `hashlib.file_digest`, which Python 3.11 added, while `.python-version` pins 3.10. Under `uv` or the lock image, verifying a general-domain corpus stopped with an AttributeError. It now uses `huggingface_hub`'s `sha_fileobj`, the hasher that produces the LFS sha256 the manifest records. On Python 3.10 with the locked huggingface-hub 0.34.4 it matches the manifest for both ClimbMix files. `pyyaml`, imported directly by decontam and base_eval, is now declared; before, it was installed only as a dependency of other packages.
- **Simplified.** `sources.completed_units` returns only each unit's files: no caller read the unit names it paired them with. The mapping from a mixture spec to the compiled data each corpus contributes was written out separately in base_train and base_eval; it is now `sources.compiled_mixture`, the single definition both read.
- **Rejected.** A pandas `Index` hash table in place of `EvalIndex`'s bit filter and binary search gave identical masks and ran faster: 2.28 s against 3.21 s for 125M lookups against 4.4M eval hashes. It would also have removed `FILTER_BITS`. But adding pandas to the lock pulled in pandas 2.3.3 and 3.0.6, pytz and tzdata, and moved numpy to 2.5.3 for Python ≥ 3.12, which is more environment change than 12 exact, measured lines warrant. The remaining modules have no dead code or unused imports. Every other literal is justified: the FNV multiplier, the GPT-3 13-gram rule, the ClimbMix-matched shard and row-group sizes, the published RefineCode shares, the measured packing buffer, and the RFC formats in sanitize. Every caught exception is a documented data case: absent, corrupt or undecodable Software Heritage blobs, octets over 255, and paths without a Pygments lexer.
- **Parity** (ClimbMix 1 file and Stack v3 1 part plus val, scratch vocab-32768 tokenizer, seq 2048). Compiled train and val of both corpora are byte-identical (8 of 8 files, statistics included). The tokenizer (`tokenizer.pkl`, `tokenizer_stats.json`) and verify output are identical for both corpora. base_train on the 2-corpus mixture reads the same compiled data at the same planned shares, with identical losses up to step 3 and identical step-0 val bpb. Later values differ by about 1e-6, exactly as two runs of the unchanged code differ from each other (GPU nondeterminism from step 4). base_eval bpb ran per corpus. A fetch with `--max-files=11` over a 10-file Stack v3 prefix fetched exactly one more part, and verify passes.

---

## 2026-09-26: Stack v3 at 38 parts: used as published, tiktoken 0.14.0, compile fails fast

Fetched 38 Stack v3 train parts (430 shards) and val in tmux: 10,754,153 documents, 11 GB, 175 s; verify passes. Compiling them hung, which led to three findings.

- **No second deduplication.** Stack v3's publisher near-deduplicated the whole corpus: MinHash with 256 permutations over word 5-grams, LSH, candidate pairs verified at Jaccard ≥ 0.7, connected components, keeping the member with the most stars, then forks, then a permissive license, then the earliest repository. It then applied StarCoder2's quality filters. NeMo Curator's fuzzy deduplication (26.07 container, defaults: 24-character shingles, 20 bands × 13 hashes) over the 38 parts flagged 471,210 files (4.4%), holding 11.4% of the bytes. Measured against each file's cluster representative, only 5.2% of the flagged files reach Stack v3's own criterion (word 5-gram Jaccard ≥ 0.7; median 0.39). Weighted by size, 67% ± 2% of the flagged bytes reach it: 7.7% of Stack v3's bytes, mostly large generated or bundled files. These are lower bounds, since each file is compared with its cluster's representative, not its nearest member. Within one part, which is about what a small run reads (0.47B tokens), the flagged share is 7.2% of bytes (median over 39 parts; p90 17.6%). That repetition is far below the ~4 epochs at which Muennighoff et al. 2023 ([arXiv 2305.16264](https://arxiv.org/abs/2305.16264)) measure negligible loss change, so the corpus is used as published. The other five corpora measured at most 0.15% of their documents as near duplicates. A Curator drop-list implementation was built, measured and removed.
- **tiktoken crash.** 1 of the 10,754,153 documents crashed the encoder. It is a Jupyter notebook holding a 1,108,250-letter DNA string, a single pre-token; notebooks are one-line JSON, which the line-length filter exempts. On that pre-token, tiktoken 0.11.0 panics (`RuntimeError(StackOverflow)` from its regex engine, raised as pyo3's `PanicException`). Long pre-tokens also take quadratic time: 34 s at 400k letters and more than 120 s at 800k ([tiktoken #195](https://github.com/openai/tiktoken/issues/195)). tiktoken 0.13.0 fixed both (updated fancy-regex, "branch byte pair encoding to fix performance on unusual input"). `pyproject.toml` now requires `tiktoken>=0.14.0`: the notebook encodes in under a second (623,467 tokens). The token ids are identical to 0.11.0 on all 18,169,615,609 tokens of the 431 Stack v3 files (the notebook excluded), on 9,072 documents from the other five corpora and on 10 edge-case strings. The whole corpus encodes in 258 s instead of 336 s. The tokenizer fingerprint is unchanged, so compiled data stays valid. Before the fix, Llama 3 and torchtune worked around the crash by chunking input at 400,000 characters and splitting runs longer than 25,000 characters. That is not needed now, and it would change the tokens of such runs.
- **Compile hang.** `multiprocessing.Pool` workers catch only `Exception`, and `PanicException` is a `BaseException`, so the worker died and `imap` waited forever. Compile now uses `ProcessPoolExecutor`, as fetch does, which catches `BaseException` and raises in the parent. Under tiktoken 0.11.0, compiling the notebook's shard now stops in 8 s with a `PicklingError` naming `PanicException`, instead of hanging.
- **Compile.** All 430 shards and val at seq 2048 with the six-corpus vocab-32768 tokenizer took 6.5 min on 20 CPUs: 31 GB, 7,558,125 rows from 10,732,035 documents. 17,900 documents were dropped as eval-contaminated and 985,807 were segmented. Packing crops 13.0% of the train tokens (per shard median 2.1%, p90 22.0%, max 45.3%; 25 shards over 30%) and 1.2% of val (DATA_ROADMAP Open 6).

---

## 2026-09-26: General-domain corpora fetched and validated like the code corpora

The registry (`sources.DATASETS`) is now two sections: code corpora (`CodeCorpus`: stack_edu, refinecode_stackv2_reconstructed, stack_v3) and general-domain corpora (`TextCorpus`: climbmix, smol_pdfedu_dclm_fwedu, txt360_v2_web). Fetched `climbmix` (8 train files + val, 0.79 GB, 150 s) and `txt360_v2_web` (8 + val, 17 GB, 166 s) in tmux, and validated all three general-domain corpora through the same path as the code corpora: verify, the tokenizer, compile, base_train and base_eval.

- **Integrity.** Fetch now records each file's sha256 at the pinned revision (its Hub LFS id) in the text manifest, and verify checks every local file against it, the counterpart of the Software Heritage sha1 check: 21 of 21 files (18.6 GB) match, 10-12 s per corpus. verify also reports documents, characters, empty documents, exact duplicates, val documents copied in fetched train, and exact copies across the general-domain corpora. It had a latent crash for a corpus whose manifest exists but has no complete unit yet (an empty file list to DuckDB); a corpus now counts as fetched from its first complete file.
- **Registry claims.** ClimbMix: 6,543 shards of ~252M chars (56M tokens), 1,024-document row groups, text only. Smol-Data: 100 shards of ~950M tokens, FinePDFs-Edu / DCLM / FineWeb-Edu 0.47 / 0.32 / 0.21 of characters (the 50/30/20 BT mix). TxT360 web-high-medium: 828 shards of ~0.93M documents, 1.14B tokens; its Medium-High quality bucket, not only web: English web (Common Crawl, ClueWeb, HPLT) is 93% of the characters and curated text (S2ORC, PubMed Central, MegaMath, arXiv, Wikipedia, USPTO, PG-19, FreeLaw, StackExchange, ...) the rest, in the same shares in every row group sampled across both of its source files, so a short fetch (chunk0 only) and its val file (the last of chunk1) are unbiased. The registry comment is corrected. Every val file is the last in sorted order.
- **Data.** No empty document in any. Exact duplicates: ClimbMix 0.019%, Smol-Data 0.14% (DCLM and FineWeb-Edu only), TxT360 1 in 8.08M. Across corpora: 97 exact copies between ClimbMix and Smol-Data, 2 between Smol-Data and TxT360, 0 between ClimbMix and TxT360. Dropped as eval-contaminated: 0.07% / 0.27% / 0.14% of documents.
- **Val overlap.** Val is held out by file. ClimbMix and Smol-Data repeat some val documents exactly in every train file, mostly boilerplate pages (one Smol-Data page 140 times in 2 train files): per train file 0.006% of ClimbMix's val documents and 0.084% of Smol-Data's, so at most 1.0% of ClimbMix's val at 170 train files and 1.7% of Smol-Data's at 20. TxT360: none in 8 files. Recorded as DATA_ROADMAP Open 7 (dropping train documents whose exact text is in val is a method change).
- **Pipeline.** A six-corpus tokenizer (vocab 32768, 2B chars, 1/6 each; held-out prose 4.18 bytes/token), compiled all fetched files of all six corpora at seq 2048 in 18 min, then base_train (20 steps, depth 2, planned shares 0.164-0.169 against 1/6; val bpb 3.90 → 3.14) and base_eval with bpb per corpus. Tokens cropped: ClimbMix 13.0%, TxT360 48.7%, Smol-Data 59.8%, against 2.6-14.5% for the segmented code corpora: general-domain documents are packed whole and keep only their first row (Open 1, now with these numbers).

---

## 2026-09-26: nanochat/data/pretrain reduced to its mechanism

A second pass over every pretrain module: what each element does for the experiment, and whether a library or less code does the same. Every change was checked against a copy of the code before it, on the same inputs.

- **Loader** (`stream.py`). The per-component permutation cache and index bookkeeping are one generator per component (`row_order`: rows epoch after epoch, each epoch a `RandomState([seed, epoch])` permutation or stored order), advanced by every rank through the whole micro-batch. `mixture_rows` keeps counting in 1M-position passes: without them the counts are the same (checked up to 100M positions) but a pass over the whole span takes ~16 bytes per position, once at startup (planned shares) and once on resume: 305 MiB at 20M rows (41B tokens at seq 2048) and 1.5 GiB per rank at 100M, against a 16 MiB peak chunked.
- **Compile.** Workers receive the parent's tokenizer instead of each loading it again. Statistics keep what is not derivable: documents, drops, segmented documents and pieces, tokens entering packing, rows, tokens_cropped (dropped: tokens_before/after/discarded_by_segmentation, zero by construction, and tokens_packed = rows × row length).
- **Selection** (`sources.py`). Removed guards for states the data excludes: column de-duplication and null-filling `in_stack_edu` (0 nulls in 306,346,760 manifest rows). Both fetch-stage errors stay, since fetch is a separate step: a corpus never fetched fails with "Dataset '<name>' is not available locally" and its fetch command, one with too few complete files with the command for the files it needs.
- **Decontamination.** Removed the empty-input and empty-candidate branches (the arithmetic covers both). The bit filter stays, now with its measured reason: over 125M n-grams of 93k FinePDFs/DCLM/FineWeb-Edu documents, 2.2 s with it against 78 s for binary search alone, longer than tokenizing them (55 s).
- **Fetch.** A Software Heritage request that still fails after obstore's retries now stops the fetch, which a rerun resumes; previously it counted as a dropped blob, so an outage could complete a shard with blobs missing. Blobs absent from the bucket, corrupt or not decodable are still dropped. Unit sidecars keep files, documents, dropped, text bytes and seconds (dropped: characters, compressed bytes, per-language counts, which nothing read).
- **verify** reports unit completeness instead of claiming to check a prefix it did not check; the temporary-file count is gone. `storage.list_repo_files`, a pass-through, is gone: `task.py` calls `HfApi` directly.
- **Parity.** 10 of 10 compiled splits byte-identical (3.33 GB: Stack-Edu, RefineCode with and without `--without=stack_edu`, Stack v3, FinePDFs/DCLM/FineWeb-Edu, train and val at seq 2048, vocab 8192), same settings and matching statistics, same compile times; loader batches identical (400 seeded from row 0, 2 × 400 resumed across an epoch boundary on 2 ranks, 400 mixed val, 1600 single-corpus val; val re-iterates from the start), 0.059 ms per 16-row micro-batch against 0.060; 3,000 of 3,000 re-fetched Stack-Edu blobs equal the stored texts, an absent blob is dropped, an unreachable endpoint raises. base_train (20 steps, depth 2, 3-corpus mixture, planned shares 0.398/0.402/0.200) and base_eval (per-corpus bpb) ran.

---

## 2026-09-26: nanochat/data/pretrain audit and scaffolding removal

Traced fetch → sanitize → manifest → training selection → decontamination → segmentation → packing → loader → base_train/base_eval, and checked every change against a copy of the code before it, on the same inputs.

- **Defects fixed.** (1) compile named its output through the mixture rule, so a `--without` corpus that is not the overlap partner dropped documents but wrote under the plain corpus name; it now names the output from the `--without` it applied (`sources.compiled_name`), and the overlap policy has one definition (`sources.mixture_without`), which the tokenizer used to duplicate. (2) The RefineCode sampling key was a float fraction of sha256; it is now an exact integer comparison against share × 2²⁵⁶ (the float could round a key to 1.0 and drop a document at share 1.0). (3) verify printed the flag count as the Stack-Edu/RefineCode intersection; it now computes the intersection and checks every `in_stack_edu` flag against it over both whole manifests (0 of 306,346,760 disagree, 5-7 s). (4) The decontamination n-gram size travelled beside the index built with it; `EvalIndex` now carries it.
- **Lineage.** `<split>.json` of compiled data now records the settings that produced it: corpus, `--without`, sampling shares, seq_len, tokenizer fingerprint, packing buffer, n-gram size and the eval n-gram set's digest (`EvalIndex.digest`, previously unused).
- **Removed.** The `_config` alias in compile workers, `PretrainingBatches.names`, `EvalIndex.__len__` (unused), the single-use `_word_hash`, the eval-index chunk size (10,000; hashing per eval task gives the same set with a lower peak, 950 against 1,043 MiB), function-local imports, a second `SEED = 42`, the hand-appended `--max-files` suffix at three call sites, and hard-coded corpus lists in verify (now the registry and `OVERLAPS`). Fetch checks a blob's sha1 before decoding and sanitizing it.
- **Parity.** Before vs after, on 8 Stack-Edu, 8 RefineCode (with and without `--without`), 21 Stack v3 and 1 ClimbMix train files plus val at seq 2048: identical tokenizer (fingerprint and sample composition), byte-identical compiled splits (10 of 10, 5.4 GB) and statistics, identical loader batches (400 micro-batches each: seeded train from row 0, 2 ranks across an epoch boundary, stored-order val), identical eval index (digest 808a8562145ab01b), 6,000 of 6,000 re-fetched Software Heritage texts equal to the stored ones, and an identical Stack v3 manifest rebuild. base_train and base_eval ran on the new API.
- **Open (not changed).** Packing still drops code that segmentation keeps: when nothing fits a row's gap the shortest buffered document, often a long piece, is cropped and its rest discarded. At seq 2048 (vocab 8192) that is 16.2% of Stack v3's train tokens (4 of 21 shards crop 21-39%) and 3.6-3.7% of Stack-Edu's and RefineCode's; at seq 512, 31-33%. Details: [DATA_ROADMAP.md](DATA_ROADMAP.md), Open 6.

---

## 2026-09-26: RefineCode reconstruction named for what it is; Software Heritage text sanitized; OpenCoder's Java/HTML downsampling at read time

Joining RefineCode's released metadata with its own recorded sizes showed that fetching by blob reproduces RefineCode's file selection but not its training text: 93.5% of fetched files matched the recorded size exactly, while raw `.ipynb` JSON was 48.4% of the fetched bytes against 1.8% of RefineCode's recorded bytes (it converted notebooks to StarCoder's Jupyter-structured format), and it had removed copyright headers and PII; its released pipeline (`opc_data_filtering`) holds only the quality filters.

- **Named for what it is.** `refinecode` is now `refinecode_stackv2_reconstructed`: its released file selection for the files it shares with The Stack v2 (about half its raw code, none of its code-related web data). The manifest excludes notebooks rather than imitating an unpublished conversion: 306,346,760 blobs in 4,557 train shards, exactly the previous manifest's blobs minus 2,870,630 notebooks, with identical columns and Stack-Edu flags (77,016,649 shared, union 396,393,470).
- **Sanitized text.** `pretrain/sanitize.py` normalizes Software Heritage text for both Stack-Edu and RefineCode during fetch, as a documented equivalent of RefineCode's transformation rather than a byte-for-byte one: a leading comment block mentioning a copyright is removed (Pygments lexes it by path), private keys, JSON web tokens, URL passwords and detect-secrets' provider keys become `<SECRET>`, emails `<EMAIL>`, public IPv4 addresses `<IP>`. On the re-fetched shards (717,849 and 604,877 documents) 3.1% / 4.1% of documents gained `<EMAIL>`, 0.37% / 0.52% `<IP>`, 0.11% / 0.15% `<SECRET>`; 0.7% of characters changed, ~1.2 ms per file. detect-secrets' entropy and keyword detectors were rejected: on code they flag nearly every file.
- **Downsampling at read time.** OpenCoder trained on 200 of 449 GB of Java and 64 of 474 GB of HTML. `sources.read_training_documents`, now the one reader for both the tokenizer sample and compile (it also took over the `--without` overlap filter), keeps those shares by sha256 of the blob ID read as a fraction. Blob IDs themselves are not uniform enough: over 58.9M Java blobs, 0.4391 fall below 0.4454 (98 sd off); their sha256 gives 0.44547. The local corpus keeps every selected file. Smoke run (seq 512, stack_edu:0.5 + RefineCode:0.5): compile reports 85,882 of 542,022 RefineCode train documents not selected (203,857 with `--without=stack_edu`), base_train plans 0.5/0.5 and val bpb falls.
- **Fetch rate.** One process fetches ~260 files/s with the sanitizer, as without it; a 100 MB/s Hub download on the same 1 Gb/s link slowed Stack-Edu units from ~300 s to 610-710 s.

---

## 2026-09-26: Code corpora (Stack-Edu, RefineCode, The Stack v3), an explicit fetch stage, mixtures, lossless code segmentation

Removed `smol_pdf_dclm_fwedu`, `dclm_100b`, `fineweb_edu_100b` and `finepdfs_edu_100b`. Added three code corpora behind a fetch stage that is now the only network step for pretraining data (`python -m nanochat.data.pretrain.fetch`, resumable and deterministic, run in tmux); the tokenizer, compile and training read local files only and fail with the fetch command when one is missing (the tokenizer did, on a mixture needing 750M characters from 674M of fetched Stack v3). Status, rates and runbook: [DATA_ROADMAP.md](DATA_ROADMAP.md).

- **Where the text is.** `HuggingFaceTB/stack-edu` and `OpenCoder-LLM/RefineCode-code-corpus-meta` publish metadata only. Text comes from Software Heritage's S3 bucket by blob ID, anonymously; a blob ID is the plain sha1 of the file's bytes (200/200 checked; not the git blob sha1), so every fetched file is verified; a blob that cannot be fetched is dropped (4 of 1,039,077 listed so far, all absent from the bucket). Fetching goes through obstore, whose S3 client retries throttled and failed requests: 20 processes with default thread pools fetch ~5,300 files/s (one process ~270); the first client, httpx with a hand-written retry loop at 7 processes x 64 connections, measured ~4,800, and re-fetching 8 of its units with obstore gave identical shards. RefineCode has no blob IDs: its (repository, path) rows are joined with `bigcode/the-stack-v2` on `'/' || sub_path` (in shard 0, >= 90.7% of Rust files resolve through the full table, 81.8% through `-dedup`, and only 25.0% of all files through Stack-Edu, whose blob list covers just its own files). `HuggingFaceCode/stack-v3-train` has the text inline; its `content_id` equals sha1(text) for 88.2% of files (541,131 of 613,486 fetched) because the text is PII-redacted after hashing, so each document also gets `content_hash` = sha1 of its stored text, the exact identity across sources.
- **Manifests before materialization.** Stack-Edu and RefineCode metadata is grouped by language, so a DuckDB manifest holds out whole repositories for val and orders the rest by a seeded hash before cutting ~250 MB shards: Stack-Edu is 167,063,359 unique blobs in 2,105 train shards, each ~80k files and 67-69M tokens. Stack v3's 8,192 parts are hash-partitioned; the manifest orders them by a seeded hash and each row group of a part (whole repositories) becomes a shard.
- **Overlap.** ~25% of RefineCode's files are Stack-Edu files. Each corpus is materialized whole, RefineCode documents carry `in_stack_edu`, and a mixture of both reads `refinecode-without-stack_edu`, so no blob is trained on twice. (The spec asked to store shared content once; that would make RefineCode alone depend on how much of Stack-Edu is fetched, so the ~25% is stored twice.)
- **Segmentation.** Code files longer than a row are split in compile, after tokenization (pieces depend on tokenizer and seq_len), into consecutive pieces of at most a row cut after line-ending tokens, each with its own BOS: 7.7% of Stack-Edu files and 11.0% of Stack v3 files are split, 0 tokens are discarded, and packing crops 2.6% and 7.3% of tokens instead of 30-32% with crop-only. Text corpora are unchanged, and `smol_pdfedu_dclm_fwedu` (long PDF documents) loses 60.2% of its tokens to cropping at seq 2048.
- **Mixtures.** `--dataset=stack_edu:w1,refinecode:w2,stack_v3:w3,smol_pdfedu_dclm_fwedu:w4` for the tokenizer, base_train and base_eval. Rows interleave by a low-discrepancy schedule (frac(k·φ) over the cumulative weights): share error 1e-6 over 1M rows, identical batches to the single-corpus loader for one corpus, exact 2-rank and resume equivalence. The tokenizer samples the same weights and reports held-out bytes/token per language (smoke mixture: prose 4.10, Python 3.68, Java 4.19, C/C++ 3.31, Go 3.12).
- **RefineCode, measured.** The join ran in 14 min after 66 min of downloads: 91.75% of 336,845,710 rows resolve, 309,217,390 unique blobs, 77,016,649 (24.9%) also in Stack-Edu, union 399,264,100; verify finds the flags exact and no unflagged shared blob among fetched shards, and `--without=stack_edu` drops 24.8% of RefineCode's fetched train documents. Its bpb is ~2x the others' (held-out 2.31 bytes/token against Stack-Edu's 3.68) because fetching by blob gives each listed file's original bytes, not RefineCode's text. 93.5% of fetched files match RefineCode's recorded size exactly (99.3% its line count), but RefineCode converted Jupyter notebooks to StarCoder's Jupyter-structured format (the raw `.ipynb` JSON is 12.7x larger: 48.4% of fetched bytes against 1.8% of RefineCode's recorded bytes) and removed copyright headers and PII (the 5.6% of files recorded smaller, 69% of them with a copyright notice up front). Its released pipeline, opc_data_filtering, holds the quality filters, not these transformations. Joined against RefineCode's own metadata, its composition by recorded bytes is code 92.2%, text 5.1%, data 2.8%.

---

## 2026-09-26: nanochat/data split into pretrain/, posttrain/ and eval/

`nanochat/data` mixed the three stages at one level, and `tasks/` mixed SFT data (SmolTalk), benchmarks (ARC, HumanEval), datasets used by both (GSM8K, MMLU) and SFT-only machinery (`TaskMixture`). Now each module lives in one stage: `pretrain/` (sources, decontam, compile, stream), `posttrain/` (SmolTalk, and `sft.py` with `TaskMixture` and `SFTLoader`) and `eval/` (the CORE bundle as `core.py`, ARC, MMLU, GSM8K, HumanEval). `storage.py` (downloads) and `task.py` (the Task base, `HubDataset`/`load_hub_dataset`, the multiple-choice prompt format) are shared. Benchmarks live in `eval/` because they are benchmarks with graders; SFT and RL train on their train splits (RL rewards with GSM8K's grader) and pretraining decontaminates against their test splits, so all of those import from `eval/`, and `eval/` imports from no stage. Commands move with the modules: `python -m nanochat.data.pretrain.compile`. No code changed besides imports and moving `TaskMixture`.

---

## 2026-09-26: Compiled data as one numpy row file per split; loader without DataLoader workers

`nanochat/data/shards.py` is gone, and `compile.py` and `stream.py` shrank to the experiment's mechanism: BOS-aligned best-fit packing with a 16000-document buffer, 13-gram decontamination, and a seeded, elastic, resumable row order.

- **Format.** A compiled split is one flat file of token rows (`{train,val}.bin`, `numpy.min_scalar_type` of the vocabulary), read in place with `numpy.memmap`. It replaces a hand-written Megatron `.bin/.idx` codec, 64 MiB shards, `index.json`, per-shard digests and the format/tokenizer/seq-len checks. Megatron interoperability was only exercised by a test that skips without megatron-core, whose import pulls in the whole framework and needs a GPU. The rows are byte-identical to the previous shards concatenated in index order (2 ClimbMix files: 46,727 train and 23,067 val rows).
- **Compile.** Workers pack one raw file each into a temporary file, and the main process appends them in raw file order and renames the result into place when complete. Removed: resuming an interrupted compile, `--workers`, `--shard-mb`, `--splits`, `--delete-raw`, encoding in batches of 256 (with one thread, batching bought nothing) and the 0 = off mode of `--decontam-ngram`. `--max-files` is required. The packer keeps its rows exactly; `bisect`'s `key=` replaces the (length, arrival) tuples, since `insort` already keeps arrival order among equal lengths. 2 files compile in 50 s instead of 75 s.
- **Order.** Each epoch is one permutation of all rows from `numpy.random.RandomState([seed, epoch])`, whose streams are frozen across numpy versions, instead of a shard permutation with rows shuffled within blocks of 16 shards and blake2b-derived seeds. The block structure only served object-store streaming. The train order therefore differs from before; validation and `base_eval` read in stored order, which is unchanged (identical batches at 1 and 2 ranks). Each epoch is a permutation of all rows, including a micro-batch that straddles two epochs; 2 ranks x 4 rows equal 1 rank x 8 rows step by step; and a resumed run reproduces the losses exactly.
- **Loading.** The training process gathers each micro-batch from the memmap, pins it and copies it asynchronously, as the SFT loader already does. There are no DataLoader workers or `--data-workers`: on the GB10, a 16-row micro-batch takes 0.09 ms from the page cache and 0.2 ms from NVMe, and d2 steps take 70.3 ms against 70.7 ms with 2 workers. The checkpoint's data state is just the rows consumed; the check that a resumed run uses the same data and seed is gone with the digests.

---

## 2026-09-26: Remote-streaming leftovers removed

Training reads compiled shards from local disk only; the object-store streaming path had already been removed from `stream.py`, but its callers and wiring were left behind. `base_train` and `base_eval` still passed `remote=`/`cache_gb=` to `pretraining_batches`, so pretraining failed at startup with a `TypeError`. Removed: `--data-remote`/`--data-cache-gb` and the checkpoint's `data_remote`, the MinIO service, `AWS_*`, `DATA_REMOTE` and `DATA_CACHE_GB` in compose, `DATA_REMOTE` in speedrun, the `s3` extra (s3fs and 14 transitive packages leave `uv.lock`) and the direct `fsspec` dependency. The shard index entries lose the `.idx` digest and byte size, which only served transfer checks and cache sizing; each entry keeps one `digest` of its rows for the resume identity check, and the format is now `nanochat-megatron-rows-v2`, so data compiled before is rejected with a request to recompile. Multi-node runs need the same tokenizer and compiled shards on every node (a shared filesystem or a copy).

---

## 2026-09-26: Data storage and sources reduced to one path

`nanochat/data/storage.py` and `sources.py` now have one code path each, with identical data selection:

- The data root is always `<base_dir>/data` (`NANOCHAT_DATA_DIR` is gone; Docker mounts the data volume at `/runs/data`). Every download goes through one `fetch(path, download)`: a lock on `path.lock`, download if missing. HF dataset files, corpora and task data alike, live at `<data_dir>/<org>/<repo>/`.
- The lock is needed: in `hf_hub_download` (huggingface_hub 1.28), a second process fetching the same missing file deletes the first one's finished copy and downloads it again, and torchrun ranks load the same task files at once. With the lock, 4 processes fetching the same raw shard, task file and eval bundle at once download each exactly once.
- No listing cache (an uncached listing of ClimbMix's 6,545 files at the pinned commit takes ~1 s), no `text_column` (every corpus uses `text`), no local-only or `-curated` corpora (`curate.py` and the compose `curate` service are deleted), no prefetch CLI. The tokenizer and compile workers download the files they read, so `python -m nanochat.data.sources -n N` is gone from the run scripts. The removed prefetch downloaded 8 files at a time (87 MiB/s here against 43 MiB/s for one file); compile keeps parallel downloads through its workers.
- For all 7 corpora, the train and val file lists are identical to the previous implementation, and so are the documents of the first 90 ClimbMix row groups.

---

## 2026-09-25: Data pipeline: packing buffer, eval decontamination, SFT loss normalization

Measured the data pipeline on a DGX Spark (GB10) with real data (4 ClimbMix shards + val, a tokenizer trained on 1B characters, the full SFT mixture) and compared it with 2025-2026 practice. Loading is not a bottleneck anywhere: the pretraining loader delivers ~26k rows/s to the GPU (cold NVMe ~7k rows/s) while d24 at 100% of the GB10's measured 95 TFLOP/s BF16 consumes at most ~10 rows/s, and the SFT loader packs a micro-batch in ~11 ms. So the changes are about data quality and correctness.

- **Packing buffer 1000 → 16000 documents** (`--buffer-docs`). ClimbMix documents are short (median 527 tokens; the 3.1% longer than a row hold 21% of the tokens), and a 1000-document buffer rarely finds exact fits: 22% of tokens were cropped away against an 11.3% floor (a document longer than a row keeps only its first row). On one shard: buffer 1000 → 22.0% lost, 4000 → 13.5%, 16000 → 11.8%. Compiling 4 shards went from 22.3% to 12.4% cropped, +12.6% training rows from the same raw data. Buffered documents are now numpy token arrays instead of lists of Python ints (~36 → 2 bytes per token), so the larger buffer costs ~20 MB per worker. Long documents are still under-represented (documents of 2050-8191 tokens keep 42% of their tokens, 8192+ keep 15%). Chunking them first would bring the loss to ~3%, but adds rows that start mid-document; untested, left as an experiment.
- **Eval decontamination at compile time** (`nanochat/data/decontam.py`, `--decontam-ngram=13`). Nemotron-CC, the bulk of ClimbMix, is explicitly not decontaminated. In 4 shards, 0.88% of CORE items share a 13-word sequence with the data (BoolQ 3.0%, SQuAD 2.4%, mostly Wikipedia passages) and 0.04% share at least half of their 13-grams. Compile now drops every document sharing a 13-word sequence with a CORE item or a chat-eval test item: 0.06% of documents (211 of 341k train, 55 of 85k val). A one-hash bit filter (32 MiB) in front of the sorted n-gram index rejects ~98% of lookups, keeping the cost at ~10 s per shard (compile ~19 s → ~30 s per shard). Validation bpb is not comparable to data compiled before this change (the val rows differ).
- **SFT loss normalization.** `chat_sft` averaged the loss per micro-batch and divided by `grad_accum_steps`. SFT micro-batches hold 14k-26k supervised tokens (prompts, tool outputs and padding are masked), so a token's gradient weight varied up to 1.9x with the micro-batch it landed in. The loss is now summed and the accumulated gradient divided by the step's supervised-token count over all ranks (the same fix as HF's `num_items_in_batch`). On two micro-batches with 8 vs 48 supervised tokens per row, the old gradient was 103% off the single-batch token-mean gradient, the new one 2e-5 (fp32 rounding). The logged train loss is now the step's token mean rather than the last micro-batch's.

No change, with reasons: document masking at 2k context has limited effect in Llama 3, SmolLM3 and OLMo 3 (consistent with the 2026-01-13 varlen result); block shuffling is fine for this pre-shuffled corpus; exact duplicates are 0.01%; SFT padding is 0.8% (offline best-fit-decreasing would be 0.04%); tokenizing runs at 23 MB/s per core and compiling 170 shards takes minutes, so faster tokenizers or loader libraries buy nothing. The most promising data experiment left is mixing high-quality data into the final LR decay (MiniCPM, OLMo 2), which requires the decontamination above.

---

## 2026-05-05: DyT for d12 pretraining (negative)

Tried replacing normalization with [DyT](https://arxiv.org/abs/2503.10622) for d12-scale pretraining following some [hype](https://x.com/LodestoneRock/status/2050367217087512953) on X.

- DyT uses `gamma * tanh(alpha * x) + beta` with learnable scalar `alpha` and per-channel `gamma`/`beta`.
- Added separate alpha initializers for attention vs other normalization sites, following the paper's width-dependent heuristic unless overridden.
- Added optional embedding DyT plus the LLM-specific `sqrt(d_model)` embedding scale from the paper.

Every variation of the idea that was attempted, including after a bunch of parameter tuning did not outperform the baseline d12 model on master, even with steps on the x-axis. In addition, the throughput (tokens per second) was ~10% lower.

---

## 2026-03-24: Parameter-Golf Ideas Sweep (Negative)

Reviewed `openai/parameter-golf` for small/simple ideas that might transfer to nanochat pretraining without bloating the codebase. Cached notes are in `knowledge/parameter_golf.md`.

### Rationale

The parameter-golf leaderboard is a useful source of:

- tiny architecture tweaks
- short-run optimizer/schedule tricks
- Muon-related systems ideas

But much of that repo is optimized for a very different objective:

- fit in a 16MB artifact
- train in under 10 minutes on 8xH100
- evaluate on compression / bpb

So only a small subset of ideas looked worth trying in nanochat.

### Ideas Tried

**1. LeakyReLU(0.5)^2**
- Replaced `relu^2` in the MLP with `leaky_relu(x, 0.5)^2`
- **Result:** Slightly better per-step quality, but slightly slower. Net worse on wall clock.

**2. Partial RoPE**
- Applied rotary embeddings to only the first quarter of each head dimension
- **Result:** Slightly worse.

**3. LN Scale**
- Multiplied each block's normalized input by `1/sqrt(layer_idx+1)` before attention and MLP
- **Result:** Did not help.

**4. Orthogonal init**
- Switched the non-zero transformer matrices to orthogonal init while preserving zero-init output projections
- **Result:** Did not help.

**5. XSA (Exclusive Self Attention)**
- Implemented XSA on the deepest 3 non-VE layers only, so it projected against the plain `v` path rather than `v + VE`
- **Result:** Slightly better step quality but not wall clock. Not worth the extra compute in the hot attention path.

### Notes

- EMA/SWA had already been tried earlier (I skipped recording it) and did not help.
- Bigram hash embeddings had already been explored much earlier and did help somewhat, but the added parameters / VRAM / complexity were not justified at larger scale. See the Jan 27-28 entries above.

### Conclusion

This pass did not find any cheap parameter-golf transfer that clearly improves nanochat on the metric that matters: wall clock time to capability.

---

## 2026-03-04: Remove autocast, explicit dtype management, fp16 GradScaler

Replaced `torch.amp.autocast` throughout the codebase with explicit dtype management via a single `COMPUTE_DTYPE` global. Also added fp16 training support with GradScaler.

### Motivation

autocast is "magic we don't control" — it silently decides which ops run in which precision via internal allowlists. For this codebase, autocast was doing very little: the only thing it actually cast was `nn.Linear` weights from fp32 to bf16 for matmuls. `F.rms_norm`, `F.cross_entropy`, and Flash Attention all handle their own dtypes already. By making precision explicit, we gain fine-grained control (e.g. can experiment with fp32 norms) and eliminate an unnecessary layer of abstraction.

### What changed

**Core mechanism** (`nanochat/common.py`, `nanochat/gpt.py`):
- `COMPUTE_DTYPE` auto-detected from hardware: SM 80+ → bf16, pre-Ampere → fp32, CPU/MPS → fp32. Override via `NANOCHAT_DTYPE` env var.
- Custom `Linear(nn.Linear)` class that casts weights to match input dtype in forward: `F.linear(x, self.weight.to(dtype=x.dtype))`. This is the single mechanism that replaces autocast.
- Embeddings cast to `COMPUTE_DTYPE` at init (saves memory). Exception: fp16 keeps embeddings fp32 because GradScaler cannot unscale fp16 gradients.
- Embedding output explicitly cast to `COMPUTE_DTYPE` in `GPT.forward()` (no-op for bf16, active for fp16 path).
- RoPE cos/sin cache uses `COMPUTE_DTYPE` instead of hardcoded bf16.

**Autocast removal** (11 files):
- Deleted `--dtype` CLI flag, `ptdtype` variables, `autocast_ctx` definitions, and all `with autocast_ctx:` blocks from: `base_train.py`, `chat_sft.py`, `chat_rl.py`, `chat_cli.py`, `chat_eval.py`, `chat_web.py`, `base_eval.py`, `engine.py`, `bench_train_toks.py`, `test_e2e_pipeline.py`.

**fp16 + GradScaler** (`base_train.py`, `chat_sft.py`):
- `scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None`
- Backward: `scaler.scale(loss).backward()` vs plain `loss.backward()`
- After accumulation: `scaler.unscale_(optimizer)` → distributed inf-sync via `scaler._found_inf_per_device(optimizer)` all-reduced with `ReduceOp.MAX` → `scaler.step(optimizer)` → `scaler.update()`
- Zero overhead for bf16/fp32 paths (scaler is None, no branching inside kernels).

**FP8 fix** (`nanochat/fp8.py`, `base_train.py`):
- `Float8Linear.forward` explicitly casts input to `COMPUTE_DTYPE` (previously relied on autocast).
- `disable_fp8` context manager now creates our custom `Linear` (not vanilla `nn.Linear`) when swapping out Float8Linear during eval.

**Flash Attention** (`flash_attention.py`):
- FA3 Hopper kernels don't support fp16 or fp32, so `USE_FA3` (module-level constant, resolved once at import) returns False, falling back to SDPA.

---

## 2026-03-04: Dataset upgrade: FineWeb-EDU 100B → ClimbMix 400B

Switched the pretraining dataset from FineWeb-EDU 100B to ClimbMix 400B. This is by far the single biggest improvement to nanochat's GPT-2 speedrun time, bringing it down from **2 hours 46 minutes to 2 hours 1 minute** — a 27% reduction.

### What is ClimbMix?

ClimbMix 400B is a curated 400B-token pretraining mixture hosted at `karpathy/climbmix-400b-shuffle` on HuggingFace. It comes form [NVIDIA](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix). It is a blend of high-quality web text, code, math, and other sources, designed to be a better general-purpose pretraining dataset than FineWeb-EDU alone.

### What changed

- **Dataset**: `karpathy/fineweb-edu-100b-shuffle` → `karpathy/climbmix-400b-shuffle` (up to 6543 shards available vs the previous 1823 data shards, allowing for longer training in the future)
- **Data directory**: `base_data/` → `base_data_climbmix/` (clean separation from legacy data)
- **Model depth**: d26 → d24. ClimbMix trains more efficiently, so a smaller model reaches GPT-2 capability
- **Shard count**: Only approx 150 data shards (~7B tokens) are now needed for GPT-2 capability
- **Eval tokens**: doubled from 40 to 80 batches for more stable validation loss estimates
- **Legacy fallback**: added a migration warning in `list_parquet_files()` that detects the old `base_data/` directory and falls back gracefully, so existing users see clear upgrade instructions on `git pull`

### Context

This is the sixth attempt at beating FineWeb-EDU on CORE score — the previous five all failed (see entries on 2026-02-17, 2026-02-10, 2026-01-12 below). ClimbMix is the first dataset to convincingly surpass it, and the margin is large enough to also shrink the model from d26 to d24.

---

## 2026-03-02: SoftCap tuning

Quick experiment to tune logit softcap on d24 scale. Tried 5..30. 5 was terrible, the rest of them were all about equal with the exception of 20, which was the best. Minor but solid improvement: val loss improved by ~1e-3 (0.716 -> 0.715). Setting as default.

## 2026-02-19: Mixture of Experts (negative)

Implemented a DeepSeekV3-style Mixture of Experts layer as a drop-in replacement for the dense MLP. The MoE branch works and improves per-step validation loss, but is not a net improvement on wall clock time due to MoE overhead (at least for our scale of interest of approx GPT-2 capability).

### Implementation

Follows DeepSeekV3 and using torchtitan as reference:

- **8 routed experts, top-2 routing** with sigmoid gating (not softmax)
- **1 shared expert** (dense MLP processing all tokens, following DeepSeekV3)
- **Auxiliary-loss-free load balancing** (DeepSeekV3's expert bias nudging)
- **Iso-FLOP sizing**: `expert_hidden_dim = round(4 * dim / (top_k + num_shared) / 128) * 128`, so active FLOPs per token match the dense MLP
- **`torch._grouped_mm`** for dispatching tokens to experts in a single kernel (instead of a Python for-loop)
- **3D expert weight tensors** `(num_experts, hidden, dim)` — Muon's Polar Express operates on the last two dims, so each expert is independently orthogonalized
- **Active parameter counting** for scaling laws (only `top_k + shared` experts, not all 8)

### What was easy

- The core MoE forward pass: router, sort tokens by expert, grouped matmul, scatter back. Conceptually clean.
- Shared expert: just an `nn.Linear` MLP that runs on all tokens alongside the routed path.
- 3D expert params + Muon: only required fixing `second_momentum_buffer` shape to preserve leading dims.
- Load balancing: DeepSeekV3's bias nudging is simple and effective (~10 lines).

### What was hard / ugly

- **`torch._grouped_mm` quirks**: requires bf16 (not fp32), column-major right operand, int32 cumulative offsets. The API is undocumented and only discoverable by trial and error.
- **Token count padding**: torchtitan pads each expert's token count to alignment multiples (8 for bf16) for better grouped_mm throughput. We implemented this with both a pure PyTorch approach and a copy of torchtitan's Triton kernel. Both compiled cleanly (0 graph breaks), but with ~65K tokens across 8 experts, each expert already gets ~8K tokens which is well-aligned. The padding overhead (gather/scatter) actually regressed MFU from 35% to 33%. Reverted.
- **FP8 + MoE**: `torch._grouped_mm` does NOT support FP8. There's a separate `torch._scaled_grouped_mm` API that requires per-row scaling (not per-tensor like our `Float8Linear`). The backward pass for weight gradients needs per-group column-wise scaling, which torchao implements with custom Triton kernels. We investigated thoroughly (see `dev/moe_fp8.md`) but did not implement — would require either depending on `torchao.prototype` (unstable) or writing ~200 lines of custom autograd + quantization code. Partial FP8 support exists: the shared expert's `nn.Linear` layers do get converted, but the routed experts (3D `nn.Parameter`) stay in bf16.

### Results

- d18: MFU dropped from ~46% to ~35% (the grouped_mm dispatch + token sorting overhead is significant)
- Per-step improvement in validation loss does not compensate for the throughput hit
- Net negative on wall clock time

### What remains (if revisited)

- **FP8 for routed experts**: Use `torch._scaled_grouped_mm` with a custom `_Float8GroupedMatmul` autograd function, with bf16 fallback for weight gradient (avoiding the per-group column-wise Triton kernels).

What's really needed is a fused "FlashMoE" kernel that handles routing + expert dispatch + matmul in one shot (like FlashAttention did for attention), with all the needed features. This doesn't exist yet. Rawdogging MoE with current PyTorch primitives is painful — lots of sorting, gathering, scattering, and layout wrangling around the actual compute.

### Verdict

MoE is not worth the trouble for nanochat right now. The code bloat is substantial (moe.py, router, shared expert, load balancing, optimizer fixes, FP8 gaps, active param counting) and the performance is worse wall-clock at our scale of interest. The fundamental issue is that the grouped_mm dispatch overhead eats the FLOP savings from sparsity, at least at our model scales and sequence lengths.

---

## 2026-02-17: Pretraining Data: FineWeb (negative)

Tried vanilla fineweb instead of fineweb-edu dataset. Significantly, shockingly worse results:

- d26 (GPT-2): CORE 0.2602 → 0.2241

This is the fifth failed attempt to beat pure FineWeb-EDU on CORE score.

---

## 2026-02-17: Pretraining Data Mixture Experiment (negative)

Tried [hynky/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT](https://huggingface.co/datasets/hynky/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT), a mixture of FinePDFs, DCLM, and FineWeb-EDU. Slightly worse on both model sizes tested:

- d26 (GPT-2): CORE 0.2602 → 0.2549
- d18: CORE 0.199 → 0.192

This is the fourth failed attempt to beat pure FineWeb-EDU on CORE score.

---

## 2026-02-16: SFT Script Upgrades

Brought `chat_sft.py` up to parity with `base_train.py` and tuned settings based on SFT sweeps.

Tuning:

- **Optimizer warm-start** (`--load-optimizer=1`, default on): loads pretrained momentum buffers via new `load_optimizer_state()` in `checkpoint_manager.py`. LRs are reset to fresh SFT values after load. Loading the optimizer works slightly better but not by too much.
- **LR schedule**: replaced "constant 80%, linear to 0" with warmup/constant/warmdown matching `base_train.py` (`--warmup-ratio`, `--warmdown-ratio`, `--init-lr-frac`, `--final-lr-frac`). Similar to pretraining, warmdown ratio of 0.5 worked the best. `--init-lr-frac` changed from 1.0 slightly lower to 0.8.
- **LR tuning**: attempted to tune all the individual LRs (e.g. does SFT prefer lower LR for embeddings? etc.) but all of this produced negative results.
- **Data mixture**: MMLU epochs 1→3, GSM8K epochs 2→4 (confirmed best from sweeps). Epoch counts now configurable via `--mmlu-epochs` / `--gsm8k-epochs`. Might remove these in the future though.

Quality of life, footguns, minor fixes:

- **Hyperparameter inheritance**: SFT now inherits batch sizes and LRs from the pretrained checkpoint metadata by default (CLI overrides still work). Also saved `total_batch_size` to `base_train.py` checkpoint metadata.
- **GC management**: disabled Python GC after step 1 to avoid ~500ms pauses (manual collect every 5000 steps), same as base pretraining.
- **ChatCORE eval**: periodic eval during SFT (`--chatcore-every=200`) across all 6 tasks, logged to wandb.
- **MFU**: uses `get_peak_flops()` for actual GPU instead of hardcoded H100 value.
- Removed `--dry-run` and `--dtype` flags. All ranks now participate in checkpoint save.

---

## 2026-02-05: Auto Batch Size Scaling

### Background

So far, the `--total-batch-size` was hardcoded to be `2**19 = 524,288` ~= 0.5M tokens. This was the optimal setting for d12, but when I tried to re-tune it for d26 (GPT-2), I noticed that the optimal was closer to `2**20 = 1,048,576` ~= 1M tokens. This is to be expected - larger models prefer a higher optimal total batch size. However, we have to make sure that all settings of `--depth` get their own optimal batch size calculated in some principled way. Here, I referenced the "Power Lines" paper from Cerebras ([arXiv:2505.13738](https://arxiv.org/abs/2505.13738)) for a lot of related experimentation. In particular, they found that **Bopt ∝ D^0.383** (where D is the number of training tokens, not the number of parameters!). So the idea is to tune the optimal batch size on d12, and then extrapolate it with this power law to bigger models. The 0.383 exponent means batch size grows slowly: 10× more tokens only justifies ~2.4× bigger batch. For nanochat's compute-optimal training (D ∝ N via `--target-param-data-ratio`), this means deeper models naturally want larger batches.

### Implementation

Added `--total-batch-size=-1` (now the default) to auto-compute optimal batch:

```python
get_scaling_params = lambda m: m.num_scaling_params()['transformer_matrices'] + m.num_scaling_params()['lm_head']
if args.total_batch_size == -1:
    D_REF = args.target_param_data_ratio * get_scaling_params(build_model_meta(12))
    B_REF = 2**19
    args.total_batch_size = 2 ** round(math.log2(B_REF * (target_tokens / D_REF) ** 0.383))
```

Reference point: d=12 model with B=2^19 (empirically validated). The reference is computed dynamically so that if the architecture changes (e.g., different `--aspect-ratio`), the math automatically adjusts. However, if the model actually does change too much, one would also want to re-tune the optimal batch size for d=12.

### Results

With this formula, we currently get:

| Depth | Scaling Params | Target Tokens | Auto Batch |
|-------|---------------|---------------|------------|
| d=8   | 42M           | 0.44B         | 2^18 = 262K |
| d=10-16 | 70M-235M    | 0.7B-2.5B     | 2^19 = 524K |
| d=18-26 | 324M-918M   | 3.4B-9.6B     | 2^20 = 1.05M |
| d=32-50 | 1.7B-6.2B   | 17.6B-65.6B   | 2^21 = 2.1M |

In particular, this matches empirical observations that d26 prefers ~2^20 while d12 prefers ~2^19.

### Code Cleanup

Also refactored model initialization to use `build_model_meta(depth)` helper and `dataclasses.asdict()` for cleaner config handling.

### Useful references

- [Bergsma et al., Power Laws for Batch Size, Model Size, and Training Horizon](https://arxiv.org/abs/2505.13738)
- [McCandlish et al., An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162)
- [Brown et al., Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Merrill et al., The Batch Size–Critical Batch Size Myth](https://arxiv.org/abs/2505.23971)

### One more thing (batch size ramp)

Tried batch size ramping. The simplest implementation I could think of "tricks" the existing training loop by slicing each micro-batch into smaller pieces and calling optimizer.step() more frequently early in training (1/8 → 1/4 → 1/2 → full batch over the first x% of training, with sqrt LR scaling). Also required a torch.compile warmup phase to pre-compile all slice sizes and avoid recompilation spikes during training. While the idea is sound and small gains were observed, they weren't sufficient to justify the code complexity introduced (conditional slicing logic, warmup with state save/restore, etc.). Not merged for now.

---

## 2026-02-05: SwiGLU Activation (Negative Result)

Replaced ReLU² MLP activation with SwiGLU (inspired by [twitter](https://x.com/_xjdr/status/2019141521690567058)). SwiGLU uses three projections instead of two, so to match parameters and FLOPs we scale hidden_dim from 4× to 8/3×:

```python
# Old ReLU²: 2 matrices, 4x expansion
#   params: 2 × n × 4n = 8n²
#   flops:  2 × 2n × 4n = 16n² per token
self.c_fc   = Linear(n_embd, 4 * n_embd)
self.c_proj = Linear(4 * n_embd, n_embd)
x = c_proj(relu(c_fc(x)).square())

# New SwiGLU: 3 matrices, 8/3x expansion
#   params: 2 × n × (8n/3) + (8n/3) × n = 8n²  ✓ matches
#   flops:  3 × 2n × (8n/3) = 16n² per token   ✓ matches
hidden_dim = (8 * n_embd) // 3
self.w1 = Linear(n_embd, hidden_dim)  # gate
self.w2 = Linear(n_embd, hidden_dim)  # up
self.w3 = Linear(hidden_dim, n_embd)  # down
x = w3(silu(w1(x)) * w2(x))
```

Tested at both d12 and d24 (GPT-2 scale). Worse on all measures — step efficiency, wall clock time, and FLOPs. ReLU² remains superior for nanochat. **Not adopted.**

---

## 2026-02-03: Flip Muon MLP LR Multiplier (PR #492)

Tested flipping the shape-based LR heuristic in Muon from boosting tall matrices (input projections like `c_fc`) to boosting wide matrices (output projections like `c_proj`). The original code applies `max(1, rows/cols)^0.5`, giving ~2x LR to `c_fc`. The flipped version gives ~2x LR to `c_proj` instead, which aligns with classical fan-in/fan-out scaling conventions. This was proposed in [PR #492](https://github.com/karpathy/nanochat/pull/492) and showed improvements in modded-nanogpt.

**Result:** Quick d12 experiment: slightly worse **Not adopted.**

---

## 2026-02-03: Skip AdamW Every Other Step

Inspired by modded-nanogpt, tried stepping AdamW only on odd iterations while Muon steps every iteration. The idea is that small AdamW params (embeddings, scalars, gates) don't need updates as frequently as the large weight matrices, and skipping saves both compute and communication.

Added `skip_adamw` parameter to `MuonAdamW.step()` and `DistMuonAdamW.step()` plus a matching `zero_grad(skip_adamw=...)` to let AdamW gradients accumulate over 2 steps. Used `lr *= 2**-0.5` (sqrt scaling) to compensate for the 2x effective batch size on AdamW params.

**Result:** for nanochat d12, we see ~2% faster tok/s, but each step is slightly worse in loss. On net, when plotting against wall clock time, it's slightly worse. **Not adopted.**

---

## 2026-02-02: FP8 Training with torchao

Integrated FP8 training using `torchao.float8` to accelerate Linear layer matmuls on H100 GPUs.

### Background

FP8 (8-bit floating point) uses H100's FP8 tensor cores for ~2x theoretical matmul throughput. The tradeoff is quantization overhead: computing scales and casting tensors to/from FP8. Still, as an example torchtitan (Meta's distributed training framework) reports 25-28% speedups with FP8 for some of their experiments.

**Previous attempt (Jan 2026):** FP8 on just `lm_head` following modded-nanogpt with custom ops → 1% speedup, +2GB memory. Failed due to fragile torch.compile interaction. But this experiment was also done on ~d12 scale back then instead of the bigger model that gets GPT-2 capability of approx d24.

**This attempt:** Use torchao's `convert_to_float8_training()` on ALL Linear layers, increase model size to d24. The core snippet is:

```python
from torchao.float8 import Float8LinearConfig, convert_to_float8_training
config = Float8LinearConfig.from_recipe_name("tensorwise")
convert_to_float8_training(model, config=config)
```

But in practice it's more involved (see base_train.py).

### Results

**Microbenchmark (d26 MLP, 65536x1664 @ 1664x6656):**

| Method | Forward | Fwd+Bwd | Speedup |
|--------|---------|---------|---------|
| BF16 + compile | 2.00ms | 4.79ms | 1.00x |
| FP8 rowwise + compile | 1.84ms | 4.55ms | 1.08x |
| FP8 tensorwise + compile | 1.45ms | 4.06ms | **1.38x** |
| FP8 rowwise (no compile) | 2.89ms | 21.86ms | 0.23x ❌ |

torch.compile is MANDATORY. Without it, FP8 is 4x slower due to unfused scaling ops.

**Full training (d26):**

| Config | tok/sec | vs baseline |
|--------|---------|-------------|
| BF16 baseline | 630K | 1.00x |
| FP8 rowwise | 564K | 0.90x ❌ |
| FP8 tensorwise | 740K | **1.17x** ✓ |

Memory usage also decreases quite a bit, by ~9GB (activations stored as FP8 instead of BF16).

Seeing 17% speedup is encouraging but we're still not done yet because each step is now in lower precision and less powerful individually, so to make up for the precision drop we have to train longer. Empirically, running some sweeps overnight on d24 scale, I saw that the actual speedup (when you match performance) is closer to 5%. It's possible that our LLMs at ~d24 scale are still too small to confidently enjoy the speedups that come from fp8 for bigger models.

### Key Learnings

For nanochat at approximate scale of interest (~GPT-2 capability, ~d24):

1. **Tensorwise >> Rowwise** - Rowwise computes per-row scales, overhead exceeds benefit. Tensorwise uses one scale per tensor.
2. **Filter small layers** - Layers with dims not divisible by 16 must be skipped (FP8 hardware requirement)
3. **Larger models benefit more** - d12 was still slower with FP8; d26+ shows gains. Therefore, in some depths there is a benefit to fp8 and in some there isn't. Keeping it configurable for now, passed in via kwargs and default off.
4. **The effective, capability-matched speedup is lower still** - because each step is of slightly lower precision/quality.

### Integration

Added `--fp8` flag to `base_train.py`, default recipe is "tensorwise", example of turning on:

```bash
torchrun --nproc_per_node=8 -m scripts.base_train --depth=24 --fp8
```

Uses tensorwise by default. Requires `torchao==0.15.0` (compatible with torch 2.9.1), which was added to dependencies.

**TLDR**: turning on fp8 for GPT-2 capability nanochat model gives approx +5% capability-matched speedup.

---

## 2026-01-29: Hyperball/MuonH Experiments (Negative Result)

Explored Hyperball optimization from [this post](https://psychedelic-sunstone-851.notion.site/Fantastic-Pretraining-Optimizers-and-Where-to-Find-Them-2-1-Hyperball-Optimization-2e924306e6f280e7a5ffee00eb40a0dd) (saved to `knowledge/muonh.md`). Constrains weights to sphere of radius R (initial norm): `W_{t+1} = R · Normalize(W_t - η·R · Normalize(u_t))`. Had to change a number of details in a branch, e.g. not use zero init for our projections (or the initial norm would be zero), keep track of the initial norm, adjust Muon -> MuonH for the update.

Experiments on d12:

| Experiment | Result |
|------------|--------|
| MuonH for matrix params | Worse than baseline |
| MuonH + LR sweep (2.5e-3 to 1e-2) | Still worse |
| Added learnable RMSNorm scales (paper says γ preserves expressivity) | Still worse |
| Various RMSNorm init tweaks, e.g. 0 at init to residual | Still worse |
| AdamH for lm_head (paper recommends this) | Broken - loss plateaus (see below) |
| AdamH + learnable output scales | Still worse |

Could not outperform the baseline implementation. The article doesn't go into too much detail on how AdamH is applied to `lm_head` exactly. The classifier layer has to be able to increase in magnitude to make more confident predictions over time. Tried a sensible version with added 0-D learnable scalar, and also with RMSNorms with per-channel learnable scalars both pre and post resnet blocks.

**Result:** This was not an out-of-the-box win for nanochat even with a mild attempt over a few hours at a bit of tuning and debugging. The idea itself is intuitively appealing. Might come back around later to try harder later.

---

## 2026-01-28: Reverted Bigram Hash Embeddings

Removed bigram embeddings (engram-lite) from the codebase. At larger scale (d25), the improvement was tiny and disappeared entirely when measured by wall clock time. It also bloated the VRAM used. The extra parameters and complexity aren't justified.

---

## 2026-01-27: Bigram Hash Embeddings (Engram-lite)

Explored N-gram memory modules inspired by the [DeepSeek Engram paper](https://arxiv.org/abs/2601.07372) and [modded-nanogpt PR #201](https://github.com/KellerJordan/modded-nanogpt/pull/201).

### Background

The Engram paper introduces "conditional memory" as a complement to MoE - using O(1) hash lookups to retrieve static N-gram patterns instead of reconstructing them through computation. Key insight: transformers waste early layers "simulating retrieval through computation" for patterns like named entities and formulaic phrases that could be simple table lookups.

### What We Tried

**1. Full Engram module with context-aware gating (paper design)**
```python
# Hash bigrams to retrieve embeddings, then gate with hidden state
e = embed(hash(prev_token, curr_token))
q = RMSNorm(h)           # hidden state as query
k = RMSNorm(W_k @ e)     # projected embedding as key
v = W_v @ e
α = sigmoid(q · k / √d)  # scalar gate per position
output = α * v
```
- Injected after block 1 (paper found early injection optimal)
- Slight improvement, but quite a bit of complexity added.

**2. Early-layer only injection**
- Only inject bigram signal in first 4 layers (where paper claims static pattern offloading helps most)
- **Result:** Actually hurt performance. The model seems to need uniform injection across all layers.

**3. Trigrams**
- Extended to hash both 2-grams and 3-grams, concatenating embeddings
- **Result:** No improvement over bigrams alone. Dilutes capacity from more frequent 2-gram patterns.

**4. Bigram-only with x0-style injection (modded-nanogpt engram-lite approach)**
- Simple hash: `(36313 * curr) XOR (27191 * prev) mod table_size`
- Zero-init embedding table, learned per-layer lambdas
- Add to residual at every layer: `x = resid_λ[i]*x + x0_λ[i]*x0 + bigram_λ[i]*x0_bigram`
- **Result:** This simple approach works and provides a consistent improvement.

TLDR The winning approach follows modded-nanogpt's "engram-lite", simply adding the following module and feeding its output into the residual branch (gated by a per-layer learnable \lambda) before every single block:

```python
class BigramEmbed(nn.Module):
    def __init__(self, vocab_size, embed_dim, table_multiplier=5):
        self.embed = nn.Embedding(vocab_size * table_multiplier, embed_dim)

    def forward(self, idx):
        h = (36313 * idx[:, 1:]) ^ (27191 * idx[:, :-1]) % (table_size - 1)
        return self.embed(h)
```

As for optimal hyperparameters:

- **Table size:** `vocab_size * 5` (~164K entries for 32K vocab). Swept a number of settings and 5 was optimal.
- **Injection:** Every layer via learned `bigram_lambdas` (init 0.1 was better than 0.0).
- **Normalization:** Also tried adding a `norm()` to the embeddings (mirroring the token embeddings), this was slightly worse.
- **Init:** Zero-init embedding, so starts as identity (tried small noisy init, it's worse)
- **Optimizer:** AdamW with same LR as token embeddings

### Key Learnings

1. **Gating didn't help at our scale.** The paper's context-aware gating mechanism (sigmoid dot-product gate) added parameters and complexity without improvement. modded-nanogpt found the same: "simple direct addition to the residual stream outperformed by a decent margin."

2. **Uniform injection beats early-only.** Despite the paper's finding that early layers benefit most, restricting injection to early layers hurt. The x0-style "add everywhere with learned lambda" pattern works better for our architecture/scale.

3. **Bigrams are sufficient.** Trigrams didn't help - the extra context doesn't pay for the diluted capacity.

4. **Scale matters.** The Engram paper's results are at 27B params with MoE. At our ~100M-1B scale, the simpler approach wins. The elaborate gating mechanism may become useful at larger scales where collision handling matters more.

### Parameters Added

For d12 model with `table_multiplier=5`:
- Bigram embedding: 32768 × 5 × 768 = ~126M params
- Per-layer lambdas: 12 scalars (negligible)

If you're keeping track, we now have *a lot* of parameters, a significant amount of them in embeddings (token embeddings, bigram embeddings, value embeddings). For example, for a d12 we now have:

```
Parameter counts:
wte                     : 25,165,824
bigram_embed            : 125,829,120
value_embeds            : 150,994,944
lm_head                 : 25,165,824
transformer_matrices    : 84,935,808
scalars                 : 36
total                   : 412,091,556
```

In other words, only about a quarter of parameters are now weight projections and the vast majority is embedding tables.

Still, on all axes (steps, wall clock time, flops), this somewhat parameter-bloated architecture beats the baseline and will now become the default.

After adding the engram-lite, I re-ran the scaling laws to determine the new optimal tokens:params ratio. I swept FLOPs in the range 1e18..1e19, exponentially strided in 4 settings (1e18, 2e18, 5e18, 1e19). I looked at a number of ways of determining the effective parameter count for the purposes of the scaling laws. The results looked like this:

```
Kaplan-style (all projections including lm_head and no embeddings)

Optimal configurations (from quadratic fits):
FLOPs        Eff Params      Tokens          Ratio      Val BPB
-----------------------------------------------------------------
1e+18        110,678,115     1,241,505,403   11.2       0.8972
2e+18        167,797,457     1,785,336,422   10.7       0.8616
5e+18        250,650,865     2,642,234,152   10.8       0.8293
1e+19        381,758,347     3,806,871,243   10.3       0.7999

N \propto C^0.54, D \propto C^0.49

Chinchilla-style (all parameters, period.)

Optimal configurations (from quadratic fits):
FLOPs        Eff Params      Tokens          Ratio      Val BPB
-----------------------------------------------------------------
1e+18        416,320,605     1,232,157,011   3.0        0.8974
2e+18        560,239,841     1,763,669,281   3.2        0.8616
5e+18        741,495,903     2,629,909,368   3.6        0.8291
1e+19        988,644,331     3,884,841,895   4.0        0.7999

N \propto C^0.37, D \propto C^0.50

Transformer-only-style (only the projections inside the transformer)

Optimal configurations (from quadratic fits):
FLOPs        Eff Params      Tokens          Ratio      Val BPB
-----------------------------------------------------------------
1e+18        80,259,665      1,315,639,547   17.2       0.8966
2e+18        131,488,566     1,864,134,141   14.5       0.8622
5e+18        220,985,474     2,595,328,843   12.1       0.8302
1e+19        401,213,504     3,328,704,512   8.5        0.7994

N \propto C^0.70, D \propto C^0.41
```

Clearly, the Kaplan-style ratios are most consistent and produce stable ~0.5 exponents for both params and tokens, meaning we can have a single fixed ratio of tokens:params for compute optimal models. This turns out to be about ~10.5, which now becomes the new default.

---

## 2026-01-19 to 2026-01-22: Optimizer Hyperparameter Sweep

Ran ~320 experiments across 6 rounds, scaling from d12→d16→d20 to find optimal optimizer hyperparameters. Added granular per-component control to `setup_optimizers()` — separate LRs and betas for embedding, unembedding, value_embeds, resid_lambdas, x0_lambdas, and Muon matrix params.

### What We Swept
- Learning rates for all 6 parameter groups
- Beta1/beta2 for all 5 AdamW groups
- Muon momentum (start/end), weight decay
- Hundreds of combinations (2-way, 3-way, 4-way, etc.)

### The Journey

**At d12**, found two independent improvement routes:
- **Route A:** emb_lr↑ (0.3→0.4), weight_decay↑ (0.1→0.15), matrix_lr↑ (0.02→0.025)
- **Route B:** x0_lr↓ (0.5→0.2), x0_beta1↑ (0.8→0.9+)

Both gave ~0.002 improvement, but combining them caused conflicts. Fine-tuning found wd=0.13, matrix_lr=0.027, emb_lr=0.38 helped slightly. Best d12 config: Route A + x0_beta1=0.95.

**At d16**, Route B became competitive with Route A. The routes still conflicted when combined.

**At d20** (target scale), everything changed:
- Fine-tuned values from d12 **actively hurt** performance
- Routes no longer conflicted
- Just `x0_beta1=0.96` alone captured nearly all the gains

### Final x0_beta1 Sweep at d20

| x0_beta1 | val/bpb | Δ vs baseline |
|----------|---------|---------------|
| **0.96** | **0.7971** | **-0.0007** |
| 0.94 | 0.7972 | -0.0006 |
| 0.90 | 0.7972 | -0.0006 |
| 0.97 | 0.7977 | -0.0001 |
| 0.98 | 0.8011 | +0.0033 💀 |

Flat plateau from 0.90-0.96, then sharp cliff at 0.97+.

### Key Learnings

1. **Hyperparameters are scale-dependent.** What works at d12 doesn't transfer to d20. The elaborate fine-tuning that won at d12 actively hurts at d20.

2. **Improvement magnitude shrinks with scale.** ~0.002 at d12 → ~0.0007 at d20. The baseline is already better-tuned for larger models.

3. **Sharp cliffs exist.** x0_beta1=0.98 is catastrophic while 0.96 is optimal.

4. **Don't over-tune on small proxies.** Validate at target scale before shipping.

### Final Recommendation

For production d20 runs, add one flag:
```
--x0-lambdas-beta1=0.96
```

Skip everything else discovered at smaller scales.

---

## 2026-01-18: More various experiments

- Tried Muon custom kernels for XXT and all the others. The improvement was there for targeted tests (~20%) but washed out completely to noise in an actual training run, especially because the Muon compute is split across all the workers. Abandoned due to complexity bloat.
- Fuse Q,K,V,O nn.Linear layers into a single QKVO Linear layer. ~Zero impact
- Tried the `sa_lambdas` that gate QKV and O. Slightly confused because of the use of rmsnorm, which erases the effect of any scalar multiplier. Helped a tiny bit (~1e-4 of loss), abandoned to control complexity.

---

## 2026-01-17: Various experiments

Modded-nanogpt uses [Value Embeddings](https://arxiv.org/abs/2410.17897) (VEs) in a funny U-shaped structure, 3 of them in total and with gates. I tried a large number of tweaks on this today:

- VEs at every layer, at alternating layers, U shaped, front and back. Alternating layers worked best, i.e. we end up with *a lot* more VEs than modded-nanogpt, at every other layer. It works better.
- Many parameters sharing ideas to reduce new parameter count, nothing here worked. All failed.
- Many ideas to reduce parameter count, the LLM hates all of them: low rank decompositions, projections. All failed.
- Gated yes or no and how much. Gate helps.

Long story short is that the models *love* Value Embeddings. It is a way to add a huge amount of capacity (parameters) to the model at almost zero cost of FLOPs, because these embeddings are simply added to the Values tensor. Any attempt to reduce the capacity of value embeddings (param sharing, low rank, projections) fail. The model wants many of them, and with all the capacity, and doing so wins across all x axes of steps, flops and wall clock. I re-ran the scaling laws and, because the models are now very parameter bloated, the optimal ratio has halved from 8 to 4! Way down lower than Chinchilla's 20 at this point.

Other experiments, looking at val/bpb as a function of all of steps, flops and wall clock time:

- Aspect ratio of 128 is worse than 64, I tried a sweep fixing FLOPs == 1e18 and 64 outperforms. The LLM prefers to be slightly thinner and longer.
- Head dim definitely prefers to be 128 instead of 64, i.e. fewer bigger heads
- Bunch of other random stuff like that.

Keeping all of this work on a private branch for now but hope to push shortly.

---

## 2026-01-17: Modded-nanogpt Ideas Sweep (Continued)

Continued testing ideas from modded-nanogpt.

| Idea | Result | Notes |
|------|--------|-------|
| Attention gates | No improvement | Per-head learnable gates on attention output. +1GB memory, decreased efficiency. |
| Batch size schedule | Abandoned | 8→16→24 with LR scaling. Made training script too bloated/complex, not worth cognitive overhead. |
| Value embeddings | Helps a lot | Experiments still ongoing, more on this later. |

---

## 2026-01-16: Flash Attention 3 Fallback to SDPA

Added automatic fallback from Flash Attention 3 to PyTorch's `scaled_dot_product_attention` (SDPA) for users without Hopper GPUs. This enables nanochat to run on older CUDA GPUs, CPU, and MPS (Apple Silicon).

### Implementation

Created `nanochat/flash_attention.py` - a unified interface that:
- Detects FA3 availability at import time (requires sm90+ / Hopper)
- Exports a `flash_attn` object matching FA3's API exactly (`flash_attn.flash_attn_func`, `flash_attn.flash_attn_with_kvcache`)
- Automatically routes to FA3 or SDPA based on hardware
- Handles tensor layout differences: FA3 uses (B, T, H, D), SDPA uses (B, H, T, D)
- Implements sliding window attention via explicit masks for SDPA
- Manages KV cache manually for SDPA (FA3 does it in-place)

### Changes to Existing Files

Changes to existing code were intentionally kept extremely minimal.

**gpt.py**: Only the import line changed and a comment

**engine.py**: Zero changes needed

**base_train.py**: Added status print and warnings:
- Prints whether FA3 or SDPA fallback is being used
- Warns about efficiency loss without FA3
- Warns about sliding window support if `--window-pattern` is not "L"

### Testing

Tests are split into two classes due to dtype/device constraints:

1. **TestFA3VsSDPA**: Comparison tests requiring Hopper GPU + bfloat16. Run both implementations on identical inputs and verify outputs match (max diff typically 0, at most ~0.004 for sliding window).

2. **TestSDPAOnly**: SDPA-only tests that run on any device with appropriate dtype. Verify forward pass, backward pass, and KV cache work correctly.

Added `_override_impl` mechanism for testing - can force 'fa3' or 'sdpa' to directly compare implementations.

### Notes

- SDPA fallback is significantly slower than FA3 especially in that it lacks the sliding window attention support
- Recommend `--window-pattern L` (full context) when using SDPA fallback

---

## 2026-01-16: Modded-nanogpt Ideas Sweep (Mostly Negative)

Tested several architectural ideas from modded-nanogpt to see if they transfer to nanochat. All of these did not help:

| Idea | Result | Notes |
|------|--------|-------|
| Half-truncated RoPE | No improvement | Only first half of head dims get RoPE (base 1024, linspace). Second half "stationary". |
| Asymmetric softcap | Slightly worse | `23 * sigmoid((x+5)/7.5)` vs our symmetric `15 * tanh(x/15)`. May only help with FP8. |
| Smear gate | Negligible | Blend each token with predecessor via learned gate. Tiny improvement not worth n_embd² params. |
| Backout | No improvement | Save activations at ~60% through network, subtract scaled version at end. |
| Skip connection | Slightly worse | Save at layer ~25%, add at layer ~50%. Also +2GB memory from storing activations. |

Value Embeddings do show promise. I need a more elaborate exploration of a few related ideas, which I leave for tomorrow.

---

## 2026-01-15: Olmo pretraining mix (Negative result)

I attempted to train on the Olmo 3 pretraining dataset [allenai/dolma3_mix-6T](https://huggingface.co/datasets/allenai/dolma3_mix-6T) instead of FineWeb-edu. I ran into a number of [errors and issues](https://huggingface.co/datasets/allenai/dolma3_mix-6T/discussions/2) trying to both download and process the dataset and then noticed some quality issues (e.g. some documents seem to be extremely short, like "5".). I managed to work around these with some sensible hacks (e.g. reject documents less than 100 characters in length) and tried to process the dataset exactly as FineWeb, re-trained the tokenizer and trained a d16 model. The CORE score decreased from 15.5 to 13.8, i.e. the result is quite a bit worse.

I am still looking to try the [DCLM dataset](https://arxiv.org/abs/2406.11794), which according to the paper should be better that FineWeb-edu. I do have some concerns that the same group both prepared the DCLM dataset *and* introduced the CORE score so I'm a bit hesitant in case there was some overfitting to CORE score adjacent data distribution.

Classifying as negative result and reverting back to FineWeb-edu for now.

---

## 2026-01-13: Varlen Attention (Negative Result)

Attempted to prevent attention from "leaking" across document boundaries using Flash Attention's `flash_attn_varlen_func`, similar to modded-nanogpt's approach.

### Background

With the BOS-aligned dataloader, multiple documents are packed into each row. Standard attention allows tokens to attend across document boundaries within a row. The hypothesis was that preventing this "leakage" via varlen attention might improve training.

### Approach: Compute cu_seqlens from inputs

- Find BOS positions: `(inputs.view(-1) == bos_token_id).nonzero()`
- Gotcha 1: Variable-length `cu_seqlens` caused torch.compile recompilation (25s/iter!) - fixed by padding to fixed size
- Gotcha 2: `nonzero()` inside compiled model hit recompile limit - fixed by moving computation outside compiled region

### Final Results (d16)

| Metric | Baseline | Varlen |
|--------|----------|--------|
| val_bpb | 0.85427 | 0.85407 |
| MFU | ~same | ~same |
| tok/sec | ~same | ~same |

Essentially identical. The 0.0002 bpb improvement is almost noise.

### Conclusion

Not worth the code complexity. The "leakage" across document boundaries within a row is not harmful - the model handles it fine. The BOS-aligned dataloader already provides the key benefit (every row starts with proper context). Not merging to master.

---

## 2026-01-13: BOS-Aligned Dataloader with Bin Packing

Redesigned the pretraining and midtraining dataloader to ensure every sequence starts with a BOS token, and explored bin-packing algorithms to minimize wasted tokens.

### Problem Statement

The original dataloader streams tokens into a flat buffer and reshapes into batches. This means some rows start mid-document (no BOS), which could confuse the model during training. We want every row to start with BOS and contain well-formed documents.

### Approach 1: Greedy-Crop BOS (Simple)

Each row is built independently:
- Start with a document (which has BOS prepended)
- Pack more documents until row is full
- If a document doesn't fit, **crop it** to fill remaining space (discard the rest)
- 100% utilization (no padding), but wastes cropped tokens

### Waste Analysis

Measured token waste empirically on real data (T=2048):
- **39.4% of tokens are cropped** (discarded when docs don't fit)
- **22.9% is the theoretical minimum** (tokens in docs longer than T+1 that can never fit)
- The extra ~16.5% comes from "unlucky" cropping when a long doc starts near the end of a row

### Bin Packing Algorithms Explored

| Algorithm | Util% | Crop% | Pad% | Notes |
|-----------|-------|-------|------|-------|
| Greedy-Crop (baseline) | 100% | 39.4% | 0% | Simple, no wasted compute |
| Greedy-Pad | 78% | 23.0% | 22% | Pads instead of crops - wastes compute |
| First-Fit Decreasing (FFD) | 99.7% | 23.0% | 0.3% | Near-optimal packing, minimal padding |
| **BestFit-Crop** | 100% | 34.6% | 0% | Smart cropping, no padding |

### BestFit-Crop Algorithm

A middle ground that maintains 100% utilization while reducing cropping:

1. Buffer N documents
2. For each row, greedily pick the **largest doc that fits entirely**
3. Repeat until nothing fits
4. When nothing fits, crop a doc to fill remaining space exactly

This avoids "unlucky" crops by searching the buffer for better-fitting documents.

**Results (T=2048):**
- Crop waste reduced from 39.4% → 34.6% (~12% relative improvement)
- Still achieves 100% utilization (no padding, every token trains)
- Slightly more rows than baseline (uses more documents per batch)

### Decision: Keep Two Implementations

1. Keep the original implementation which is very simple, efficient and has 100% token utilization in the batch (no padding with ignore tokens), but creates slightly more confusing token streams for the LLM because documents during training can start abruptly from the middle with no context. Note that this never happens at test time, where BOS is always present.

2. **`_bos_bestfit` (BestFit-Crop, new default)**: Slightly more complex but still keeps 100% token utilization in the batch (no padding), but at the cost of discarding documents when they don't fit. In practice, about 34% of tokens are discarded with this approach. This is ok because for most models we care about we have plenty of data without having to go to multiple epochs. One more subtle effect is that it does skew the data distribution a tiny bit because, reliably and necessarily, tokens at the tails of long documents will be discarded. However, this doesn't seem to impact actual downstream performance.

### Midtraining

The midtraining dataloader was also updated. Because conversations are on average a lot shorter than pretraining documents, only about 3.3% of tokens get cropped.

### NOTE: loss scale

Do note that switching to the BOS dataloader changes the validation loss and makes all previous experiments not comparable in absolute value of the loss, because we have a lot fewer "confusing" tokens in the train/val batches. All tokens can look back and find the BOS token and have the full context of that document to make predictions. Therefore, the loss appears lower but this is "fake" to some extent, and the expectation is that the vast majority of relative comparisons done so far would agree with those before and after this change.

---

## 2026-01-13: Number Token Split Pattern

Validated the `\p{N}{1,2}` pattern in `SPLIT_PATTERN` (tokenizer.py line 30), which I only guessed earlier and had a TODO for to validate. GPT-4 uses `\p{N}{1,3}` to group number sequences of up to 3 digits into tokens, but we suspected smaller vocab sizes benefit from grouping fewer digits per token.

**Results (d12, vocab=32K):**
| Pattern | val_bpb |
|---------|---------|
| `\p{N}{1,1}` | 0.969 |
| `\p{N}{1,2}` | **0.965** |
| `\p{N}{1,3}` | 0.972 |

**Conclusion:** `{1,2}` is optimal for vocab size 32K. Grouping 3 digits wastes tokens on rare 3-digit combinations; grouping 1 digit is too fine-grained and bloats token sequences. Keeping `{1,2}` as default.

---

## 2026-01-13: FP8 Training for lm_head

Attempted to use FP8 (8-bit floating point) for the lm_head layer to speed up the large vocab projection matmul. H100 GPUs have FP8 tensor cores that can theoretically provide ~2x speedup over BF16.

### Implementation Approaches Tried

**1. Dynamic Scaling (failed)**
- Compute `x.abs().max()` and `w.abs().max()` each forward to determine scales
- Problem: `.item()` calls cause graph breaks with torch.compile
- Tried `@torch._dynamo.allow_in_graph` pattern (like torchao.float8) - worked but no speedup
- Tried `torch.library.custom_op` with float scales - caused NaN gradients after first optimizer step
- Root cause: interaction between custom ops, dynamic scale computation, and torch.compile is fragile

**2. Static Scaling (partial success)**
- Pre-set scales at init time like modded-nanogpt: `x_scale=10/448, w_scale=0.1/448`
- `grad_scale` computed dynamically from batch size (safe since it's just `1/(B*T)/57344` due to the gradient expression of cross entropy). modded-nanogpt has a bug here probably because they set `grad_scale = 0.75/448`, but grads are in E5M2 so this should probably be `1/57344`, 1 being the amax of any individual element of cross entropy loss, and no normalization by B,T because they use sum reduction not mean reduction.
- Uses `torch.library.custom_op` with `@torch.compile` on inner kernels
- This works correctly - no NaNs, proper gradients

### Results (d12)

| Metric | BF16 Baseline | FP8 lm_head |
|--------|---------------|-------------|
| GPU Memory | 34 GB | 36 GB |
| tok/sec | baseline | ~1% faster |

### The Memory Mystery

FP8 *should* save memory since we store `x_f8` (1 byte) instead of `x` (2 bytes) for backward. But we see 2GB *increase*. Suspected causes:
- `torch.compile` on inner kernels creating extra buffers/specializations
- `torch._scaled_mm` internal workspace allocations
- Custom op registration machinery overhead

Tried saving original weight `w` (just a reference to parameter) instead of `w_f8` in backward, then re-quantizing on the spot during backward - didn't help. Still saw bump.

### Microbenchmark vs Reality

Raw microbenchmark showed promise:
- BF16 matmul: 16.95 ms
- FP8 matmul (static scales): 10.31 ms (1.64x faster)
- FP8 with dynamic scaling: 12.25 ms (1.38x faster)

But in full training, the ~1% tok/sec improvement doesn't justify the 2GB memory increase and the added code complexity and the need to tune scale factors for both x and w.

### Code Artifacts

See the branch `fp8_attempt_fail` for:

- `nanochat/fp8_static.py` - Static scaling implementation (working)
- `nanochat/fp8_dynamic.py` - Dynamic scaling implementation (torchao-style, working but slow)
- `gpt.py` imports `fp8_static.LinearFP8` and simply swaps it for `lm_head` in `gpt.py`.

### Open Questions

- Why does the custom op approach use more memory than vanilla BF16?
- Why is the bump in tok_per_sec so low? We should see ~1.6X speedup in both the forward pass and also (twice) in backward pass for the gradients. Granted, Amdahl's law is part of the solution because our vocab_size is only 32K so the final layer isn't a huge part of the profile but the expected speedup is still not fully realized.

**Conclusion:** Negative result for now. The implementation works correctly but provides marginal speedup with *increased* memory usage. I'm not understanding the torch.compile interaction here. The complexity of FP8 custom ops isn't justified for lm_head alone. TODO to study in more detail the way this is implemented in other libraries, e.g. torchao.

---

## 2026-01-12: Multi-Token Prediction (MTP)

Ported multi-token prediction from modded-nanogpt. Instead of predicting just the next token, predict the next n tokens at each position with weighted loss.

### Implementation

- Instead of calling the loss `n_predict` times, uses a fancy batched computation using `unfold` + `gather` + cross-entropy decomposition (`CE = logsumexp - logits[target]`)
- Schedule anneals from 3-token to 1-token prediction:
  - 0-33%: `[1.0, 0.5, 0.25→0]` (3rd token fades)
  - 33-67%: `[1.0, 0.5→0]` (2nd token fades)
  - 67-100%: `[1.0]` (standard next-token)
- Weights normalized to sum to 1

### Results (d12)

| Metric | Baseline | MTP |
|--------|----------|-----|
| GPU Memory | 34 GB | 47 GB |
| MFU | 41% | 40% |
| val/bpb (per step) | baseline | same/slightly worse |
| val/bpb (wall clock) | baseline | noticeably worse |

**Conclusion:** Negative result for nanochat. The extra memory and compute overhead from predicting multiple tokens doesn't pay off, in fact the results get worse. The auxiliary loss signal may help in other settings (larger models, different architectures?), but for our setup it's pure overhead at the moment.

---

## 2026-01-11: Sliding Window Attention

Added configurable sliding window attention, inspired by GPT-3's alternating short/long pattern.

**Pattern string configuration:**
- New `--window_pattern` CLI arg and `GPTConfig.window_pattern` field
- Pattern is tiled across layers (e.g., `SSSL` for 20 layers → `SSSLSSSLSSSLSSSLSSSL`)
- Final layer always forced to L (full context) regardless of pattern
- Short window = `sequence_len // 2`
- Long window = `sequence_len` (full context)
- All previous models so far have been simply `L` and checkpoint loading is modified accordingly to fill in this param for old models, see `_patch_missing_config_keys`

Quick experiments showed `SSSL` (every 4th layer is long) works well - provides a good balance between compute savings and model quality. This is now the default.

---

## 2026-01-11: Flash Attention 3 Integration

Replaced PyTorch's `scaled_dot_product_attention` (FA2) with Flash Attention 3 for training and inference.

### Changes Made

**1. FA3 via `kernels` package**
- Official FA3 is "beta" and requires building from source (painful)
- Using `kernels` package from HuggingFace Hub: `get_kernel('varunneal/flash-attention-3')`
- Loads pre-built wheels, works out of the box on H100

**2. Simplified attention code**
- FA3 uses `(B, T, H, D)` layout matching our projection output directly - no transpose needed
- Training: `flash_attn.flash_attn_func(q, k, v, causal=True)`
- Inference: `flash_attn.flash_attn_with_kvcache()` handles all cache cases in one call
- Removed 3 separate FA2 code paths (training, single-token, chunk inference)
- GQA handled automatically when n_kv_heads < n_heads

**3. Rewrote KVCache for FA3**
- Old format: `(num_layers, 2, B, H, T, D)` combined tensor
- New format: separate `k_cache` and `v_cache` of shape `(num_layers, B, T, H, D)`
- FA3 updates cache in-place during `flash_attn_with_kvcache`
- Position tracked via `cache_seqlens` tensor (int32, per batch element)
- Simpler API: `get_layer_cache()`, `advance()`, `reset()`, `prefill()`

### Results

- **~9% improvement in tok/sec** during training out of the box
- Benchmarks showed FA3 is 2x faster than FA2 at realistic training sizes (batch=32, seq=2048)
- FA3 supports sliding window via `window_size=(left, 0)`, which is huge and expected to give further improvements. This is ready to tune but keeping full context for now.

---

## 2026-01-11: Per-Layer Residual Scalars (x0 & resid lambdas)

Cherry-picked an idea from modded-nanogpt around learnable per-layer residual connections.

### Changes Made

**1. x0_lambdas (x0 residual connections)**
- Save initial normalized embedding as `x0` after `norm(wte(idx))`
- At each layer, blend x0 back in: `x = resid_lambdas[i] * x + x0_lambdas[i] * x0`
- Zero-initialized, so disabled at start; model learns which layers benefit from the shortcut
- Provides direct path from embedding to deep layers, helps preserve token information

**2. resid_lambdas (residual stream scaling)**
- Per-layer multiplicative scaling of the residual stream
- Initialized to 1.0 (neutral, standard transformer behavior)
- Allows model to learn to amplify/dampen residual at each layer

**3. DistAdamW small parameter handling**
- Added support for parameters with < 1024 elements (like the scalar lambdas)
- Small params use `all_reduce` instead of `reduce_scatter`/`all_gather`
- Fixes crash when param shape isn't divisible by world_size

### Key Finding: Different LR Sensitivity

The two scalar types need very different learning rates:
- **x0_lambdas (additive)**: Can use normal LR (~0.5). Adding a fraction of x0 is forgiving.
- **resid_lambdas (multiplicative)**: Needs ~100x smaller LR (~0.005). Multiplying the residual compounds through layers.

Implementation: `resid_params` gets `scalar_lr * 0.01`, `x0_params` gets full `scalar_lr`.

### Experiment Results

Swept `--scalar_lr` (controlling x0_lambdas) at multiple depths:

| Depth | Baseline (disabled) | Best scalar_lr | Best val_bpb | Δ bpb |
|-------|---------------------|----------------|--------------|-------|
| d8    | 1.0885              | 0.20           | 1.0782       | -0.0103 |
| d12   | 0.9770              | 0.60           | 0.9693       | -0.0077 |
| d16   | 0.9059              | 0.20           | 0.9002       | -0.0057 |
| d20   | 0.8565              | 0.10           | 0.8526       | -0.0039 |

**Observations:**
- Consistent improvement across all model sizes
- Optimal LR varies by depth; default of 0.5 is reasonable, but 0.6 is better for d12
- Adding resid_lambdas (with 0.01x LR) gives small additional improvement over x0 alone

### Meta Device Footgun

Important lesson: `__init__` runs in meta device context, so any tensor values set there are fake. Must initialize actual values in `init_weights()`. Added docstring warning to `__init__`.

### Summary

Added `--scalar_lr` (default 0.5) controlling learnable per-layer scalars. The formula `x = resid_lambdas[i] * x + x0_lambdas[i] * x0` gives the model control over residual scaling and direct shortcuts to the initial embedding. Solid improvement with essentially no compute overhead.

---

## 2026-01-10: Muon Optimizer Upgrades & Cautious Weight Decay

Cherry-picked improvements from NorMuon (modded-nanogpt) into our simpler Muon implementation. Decided against using NorMuon directly due to hard-coded architecture assumptions (expects 32 params split 10 attn + 22 mlp), parameter labeling requirements, and complexity.

### Changes Made

**1. Polar Express Orthogonalization**
- Replaced Newton-Schulz iteration with "Polar Express Sign Method" from [arxiv.org/pdf/2505.16932](https://arxiv.org/pdf/2505.16932)
- Uses 5 different coefficient tuples (one per iteration) instead of fixed coefficients
- Both methods kept in code for easy comparison (`zeropower_via_polar_express` vs `zeropower_via_newtonschulz5`)
- **Result:** No dramatic/noticeable difference in training, but keeping the new Polar Express as default.

**2. NorMuon Variance Reduction**
- Added per-neuron/column adaptive learning rate from NorMuon ([arxiv.org/pdf/2510.05491](https://arxiv.org/pdf/2510.05491))
- Maintains `second_momentum_buffer` with shape `[rows, 1]` or `[1, cols]` (whichever is smaller)
- Normalizes updates based on running per-row/col variance estimate (beta2=0.95)
- Memory overhead: ~1/max(rows, cols) per param, negligible
- **Result:** Led to a very small improvement, kept and enabled by default.

**3. Cautious Weight Decay**
- Only decays weights where `update * weight >= 0` (same sign) from [arxiv.org/abs/2411.16085](https://arxiv.org/abs/2411.16085)
- Standard WD always pulls toward zero; cautious WD skips decay when gradient is pushing weight away from zero
- **Implementation note:** Had to inline the logic rather than use a separate `@torch.compile` function. Passing changing float values (like `weight_decay` during scheduling) as function arguments triggers recompilation. Reading from `group["weight_decay"]` inside the step avoids this.
- **Result:** Solid improvements, especially the cautious version was better than standard wd.
- Now defaults to ON for Muon via the `weight_decay` param. AdamW still has no weight decay and is hardcoded to 0 weight decay, might try to re-tune this later.

**4. Weight decay schedule**
- Added a linear schedule to weight decay that is default on from 1.0 to 0.0 (i.e. start with max weight decay in the beginning of training, then ramp to 0 by the end). Worked better than a static setting in experiments. (modded-nanogpt has the same schedule but it is implemented in a more confusing way by multiplying twice by the learning rate, which is already wired up to a decay schedule).

### Weight Decay Scaling Experiments

Swept weight decay values at d8, d12, d16, d20 to find optimal values and scaling law.

**Optimal Values Found:**
| Depth | Width (channels) | Optimal WD |
|-------|------------------|------------|
| d8    | 512              | ~0.40      |
| d12   | 768              | ~0.22      |
| d16   | 1024             | ~0.10      |
| d20   | 1280             | ~0.08      |

**Scaling Law:**
- Fit power law: `WD = k / channels^α` in log-log space
- Found α ≈ 1.97 (approximately 2), meaning WD ∝ 1/width²

**Practical Formula:**
```
WD_target = WD_reference × (d_reference / d_target)²
```
Example: If d12 optimal is 0.22, then d20 optimal ≈ 0.22 × (12/20)² ≈ 0.08

**Reference:** Moonlight paper uses fixed WD=0.1 for their 15B MoE model. Our experiments indicated a scaling law where the optimal WD changed with depth, so we go along with the empirical scaling law.

### Summary

Muon was changed to use Polar Express, added NorMuon variance reduction, and cautious weight decay with schedule that ramps linearly to zero. All of these changes follow modded-nanogpt repo, but all of them were also validated piece by piece to yield improvements in nanochat with the exception of the Polar Express change which was in the noise. This is default on and configurable with `--weight_decay`, using simply 0.2 and ∝ 1/width² scaling. The kwarg `--weight_decay` is therefore changing as of this change. It used to configure AdamW via standard weight decay and now it becomes exclusively used in Muon (AdamW is hardcoded to 0.0), and it is scaled based on depth.

---

## 2026-01-08: exp_grad_clip - Gradient Clipping

**Hypothesis:** Gradient clipping may be unnecessary overhead. Tested L2 norm clipping at various thresholds (0.25, 0.5, 1.0, 2.0) and elementwise clipping.

**Results:**
- No benefit at any scale tested (d12, d20)
- All variants within noise (~0.9827 val_bpb)
- Grad norm never exceeds 1.0 naturally, so clipping is always inactive
- Clipping adds ~2% time overhead from the all-reduce

**Bug Found:** Original implementation clipped local gradients before sync. Since this codebase doesn't use DDP (gradient sync is in the optimizers), each rank was clipping based on its own local norm. Fixed on the branch with proper distributed all-reduce.

**Observation:** modded-nanogpt does not appear to clip either right now.

**Summary:** Deleted all grad-clip code paths. The code naturally produces well-behaved gradients. This improves a bit of MFU because we don't have to calculate and sync grad norms.
