# Pretraining data: status and runbook

Every pretraining byte comes from the network in one explicit, parallel, resumable fetch stage (run it in tmux);
tokenizer, compile and training read local files only and fail with the fetch command when one is missing.
Design, layout and commands: README, sections Data > Corpora, Fetching, Code corpora, General-domain corpora, Mixtures, Compiled rows.

## Implemented

| Piece | Where | Verified on this machine |
|---|---|---|
| Fetch stage | `pretrain/fetch.py` | one path for all six corpora: manifest once, then every missing val file and the first `--max-files` train files in parallel, a file appearing only once complete. `climbmix` 8 train + val files (0.79 GB) in 150 s, `txt360_v2_web` 8 + val (17 GB) in 166 s, `smol_pdfedu_dclm_fwedu` 2 + val in 131 s (earlier code, same downloads) |
| Stack-Edu from Software Heritage | `pretrain/code/swh.py` | manifest of 167,063,359 blobs in md5(document_id) order (2,105 train shards + 1 val shard of 16,823 whole repositories, 249.8 MB), built from the pinned revision in 399 s: the same rows and values as the earlier seed-42 build; fetched as original text: val and 1 train shard, 162,187 documents, every text re-encoding with its `src_encoding` to its blob's bytes |
| `refinecode_stackv2_reconstructed` via The Stack v2 join | `pretrain/code/swh.py` | metadata + 398 GiB of The Stack v2 downloaded and joined in 94 min (~484 GB of DuckDB spill at peak in an earlier build): 91.75% of 336,845,710 rows resolve; without 2,870,630 notebooks, 306,346,760 unique blobs in md5(document_id) order, 4,557 train shards + 1 val shard of 10,850 whole repositories (246.7 MB): the same rows and values as the earlier seed-42 build, including the `lang` tag of the 17 blobs RefineCode lists twice under one path (whole-row tie-break); 77,016,649 (25.1%) also in Stack-Edu, union 396,393,470; val and 1 train shard fetched as original text, 140,142 documents, every text re-encoding to its blob's bytes |
| The Stack v3 | `pretrain/code/stack_v3.py` | 92,162 train files, one per row group of 8,191 parts (9-14 row groups each), and 1 val row group; the manifest reads every part's footer, which the Hub rate-limits (HTTP 429, waited out by the client), and two builds wrote the same manifest.json; a row group read by byte range equals the same row group of the downloaded part in every column (12 of 12); used as published (its publisher's near-dedup; see LOG) |
| Training-time selection | `sources.read_training_documents`, `SAMPLING` | RefineCode Java kept 0.4461 (target 200/449 = 0.4454), HTML 0.1355 (64/474 = 0.1350), other languages whole; keyed on sha256(blob ID): over all 58.9M Java blobs 0.44547, within 1 sd; with `--without=stack_edu` 0 flagged documents kept |
| General-domain corpora | `pretrain/sources.py` | the registry's claims hold on the fetched files: ClimbMix ~252M chars / 56M tokens per shard, 1,024-document row groups; Smol-Data ~950M tokens per shard, FinePDFs-Edu / DCLM / FineWeb-Edu 0.47 / 0.32 / 0.21 of characters; TxT360 web-high-medium 1.14B tokens per shard, web 93% of characters (Common Crawl 66%, ClueWeb 19%, HPLT 8%), the same source shares in every row group sampled across chunk0 and chunk1. No empty document; exact duplicates 0.019% / 0.14% / 1 in 8.08M; dropped as eval-contaminated 0.07% / 0.27% / 0.14% of documents (tokens with a six-corpus vocab-32768 tokenizer) |
| Local-only reads, missing-file errors | `pretrain/sources.py` | the tokenizer stopped on `stack_v3` with the fetch command when its files held 674M of the 750M characters it needed |
| Lossless code segmentation | `pretrain/code/__init__.py` (`segment`), applied by `compile.py` | lossless by construction: a document's pieces cover it exactly, each later piece adding only a BOS. Packing crop at seq 2048 depends on the tokenizer: with a vocab-32768 tokenizer Stack-Edu 2.6% (30-32% crop-only), Stack v3 7.3%; with a vocab-8192 one Stack-Edu 3.6%, RefineCode 3.7%, Stack v3 16.2% (its val 1.7%), ClimbMix 14.8% (see Open 6); with a six-corpus vocab-32768 tokenizer Stack-Edu 2.7%, RefineCode 2.6%, Stack v3 14.5% (21 shards), 13.0% (all 430 shards of 38 parts; val 1.2%) |
| RefineCode without Stack-Edu | `compile.py --without`, `sources.mixture_without`, `sources.compiled_name`, `sources.compiled_mixture` | exactly the flagged documents dropped, through `read_training_documents`; compile names its output from the `--without` it applied; base_train and base_eval read a mixture's compiled data through the one `compiled_mixture` |
| Weighted mixtures | `pretrain/stream.py`, `base_train.py`, `base_eval.py` | 3-corpus run: planned shares 0.375 / 0.375 / 0.25 match the weights; 6-corpus run (all registered corpora, 1/6 each): planned shares 0.164-0.169; per-corpus bpb from base_eval |
| Mixture tokenizer + statistics | `nanochat/tokenizer.py` | composition per corpus and language; held-out bytes/token: prose 4.10, Python 3.68, C/C++ 3.31, Java 4.19, JS/TS 3.87, Go 3.12, Rust 3.60, other code 3.43; six-corpus tokenizer (vocab 32768, 2B chars, 1/6 each): prose 4.18, Python 3.64, C/C++ 3.43, Java 4.13, JS/TS 3.84, Go 3.15, Rust 3.57, other code 3.49 |

## Measured rates

| What | Result |
|---|---|
| Hub download, 8 files in parallel | 87-89 MiB/s |
| Software Heritage (obstore), one process | Python's default pool (24 threads) ~260-320 files/s, 64 threads 820, 128 threads 1,610: each request waits on latency, not bandwidth (~3 KB of text per file); while a 100 MB/s Hub download shares the 1 Gb/s link, a Stack-Edu shard takes 610-710 s |
| Software Heritage (obstore), 20 processes | default pool ~5,000-5,300 files/s (Stack-Edu and RefineCode in full, 473M blobs: ~26 h); 64 threads per process 11,128 files/s (~12 h), 120,000 blobs each (Open 8) |
| Stack v3, row groups by byte range | one row group (114 MB decoded) in 3.6 s; 20 in parallel in 9.2 s, as fast as downloading whole parts (38 parts, 430 row groups, in 175 s) |
| Rewritten fetch vs the earlier code | 8 units re-fetched (3 Stack-Edu, 3 RefineCode, 2 Stack v3; 609k documents): identical values, statistics and column types |
| Compile, 8 Stack-Edu shards (2 GB of code) | ~1 min on 20 CPUs |
| Compile, Stack v3 38 parts (430 shards + val, 17.8B tokens, six-corpus vocab-32768 tokenizer) | 6.5 min on 20 CPUs |

## Open

1. General-domain corpora are still cropped, not segmented (the spec allowed keeping their behavior): a document longer
   than a row keeps only its first row. At seq 2048 with a six-corpus vocab-32768 tokenizer and the earlier
   16000-document packing buffer this dropped 13.0% of ClimbMix's tokens, 48.7% of TxT360's and 59.8% of Smol-Data's
   (99th-percentile document: 19k, 50k and 82k characters). Packing over the whole file now crops only this floor
   (ClimbMix 11.4% against 11.3%, TxT360 36.8% against 36.7%). Segmenting them like code would keep those tokens; it
   changes their compiled rows.
2. Fetch budgets: fetch amounts should follow the training-token budget. Compile's `<split>.json` gives the tokens of
   every file; with the six-corpus vocab-32768 tokenizer a Stack-Edu shard holds 68-69M tokens, a Stack v3 shard
   21-130M, a ClimbMix shard 56M, a `smol_pdfedu_dclm_fwedu` file ~950M and a `txt360_v2_web` file 1.14B.
3. The tokenizer for comparable runs: train it once on the final mixture and keep it.
4. Stack v3 exact dedup against Stack-Edu and RefineCode is not applied: 139 and 107 exact copies among the fetched
   shards, measured by content sha1 (with verify.py, since removed) while Software Heritage text was still sanitized;
   Stack v3's text is PII-redacted by its publisher, so a redacted file does not match its original bytes.
5. `refinecode_stackv2_reconstructed` is RefineCode's released file selection (its dedup and 130+ filters decide which
   files are listed), not its training text: notebooks are excluded rather than converted (the Jupyter-structured
   conversion is unpublished), its copyright/PII removal is not applied (unreleased, and lexer-based header removal
   deleted code), the metadata covers only its files that are also in The Stack v2 (about half its raw code), and its
   code-related web data is absent. Java and HTML follow its published downsampling at read time. Equal-token
   comparisons against Stack-Edu and Stack v3 remain to be run. The 8.25% of its metadata that does not resolve is
   files absent from The Stack v2 at that path (on its Go and Rust partitions, a case-insensitive match recovers 22 of
   277,853 and 3 of 89,073; 10-15% have the same file name elsewhere in the repository), so no join recovers them. 0.057%
   of listed files resolve to two blobs (mostly a repository's `master` and `main` branches); both are kept.
6. Packing still discards code tokens that segmentation keeps. When no remaining piece of the file fits a row's remaining
   space, the shortest is cropped to fill it and the rest of it is dropped; once the short pieces are used up the
   remaining ones are near-row-length, so the cropped piece would often have fitted a later row whole. With a
   16000-document buffer at seq 2048: vocab 8192, 16.2% of Stack v3's train tokens (4 of its 21 shards crop 21-39%)
   and 3.6-3.7% of Stack-Edu and RefineCode, at seq 512 31-33%; six-corpus vocab 32768, 13.0% over all 430 shards of
   38 Stack v3 parts (per shard median 2.1%, max 45.3%). Packing over the whole file takes Stack v3's first part from
   12.30% to 11.75% (val 2.24% to 2.12%): the loss is this rule, not the lookahead. Returning the remainder to the
   file's pieces as a new piece (with its own BOS) would keep it; that changes the packing method and every compiled
   file, so it is left as a decision.
7. Val of a general-domain corpus is held out by file, not by document. ClimbMix and Smol-Data repeat some val
   documents exactly in their train files (boilerplate pages; one Smol-Data page 140 times in 2 train files): each
   ClimbMix train file holds exact copies of 0.006% of val documents, each Smol-Data file 0.084%, so a run over N train
   files has trained on at most N times that share of val (ClimbMix: 170 files ≤1.0%, all 6,542 ≤39%; Smol-Data: 20
   files ≤1.7%, all 99 ≤8.3%). TxT360 has none in 8 files. Compile could drop train documents whose exact text is in
   the corpus's val split; that changes the compiled train rows, so it is left as a decision.
8. Software Heritage fetch concurrency. A process fetches with Python's default thread pool, and the rate is bound by
   request latency, not the link: 20 processes x 64 threads fetch 11,128 files/s against 5,025 with the default pool,
   which would take a full Stack-Edu plus RefineCode fetch from ~26 h to ~12 h. Raising it means choosing a number of
   requests in flight (and staying under Software Heritage's throttling, which obstore retries), so it is left as a
   decision.
9. Compile memory on the ~1B-token general-domain files. A worker packs a whole raw file, so it holds the file's tokens
   (Smol-Data 1.75 GiB, TxT360 2.11 GiB of uint16), and decontaminates one published row group at a time (~12k
   documents there, against 1,024 in ClimbMix and the code files: up to 1.55 GiB). Its measured peak is 5.0 GiB,
   against 1.0-1.5 GiB for ClimbMix and code files, so 20 workers on such files could reach ~100 GiB of the 121 GiB.
   One train and one val file compile in 996 s (Smol-Data) and 901 s (TxT360): two workers, one document encoded at a
   time each. Not yet hit, since no run has compiled more than 2 of these files; bounding it means a worker count or a
   packing buffer, so it is left as a decision.
