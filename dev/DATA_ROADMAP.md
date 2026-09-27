# Pretraining data: status and runbook

Every pretraining byte comes from the network in one explicit, parallel, resumable fetch stage (run it in tmux);
tokenizer, compile and training read local files only and fail with the fetch command when one is missing.
Design, layout and commands: README, sections Data > Corpora, Fetching, Code corpora, General-domain corpora, Mixtures, Compiled rows.

## Implemented

| Piece | Where | Verified on this machine |
|---|---|---|
| Fetch stage, general-domain corpora | `pretrain/fetch.py` | `climbmix` 8 train + val files (0.79 GB) in 150 s, `txt360_v2_web` 8 + val (17 GB) in 166 s, `smol_pdfedu_dclm_fwedu` 2 + val in 131 s; each manifest records every pinned file's sha256 (its Hub LFS id) |
| Stack-Edu from Software Heritage | `pretrain/fetch_swh.py` | manifest of 167,063,359 blobs (2,105 train shards + 1 val shard of 16,337 whole repositories) built from the pinned revision in ~350 s, identical in three builds apart from the val fix; sha1 of every blob = its ID; re-fetched sanitized (before the val fix): 8 train shards and val, 717,849 documents, 58 dropped (unfetchable, or only a copyright header) |
| `refinecode_stackv2_reconstructed` via The Stack v2 join | `pretrain/fetch_swh.py` | metadata + 398 GiB of The Stack v2 downloaded and joined in ~86 min (~484 GB of DuckDB spill at peak): 91.75% of 336,845,710 rows resolve; without 2,870,630 notebooks, 306,346,760 unique blobs in 4,557 train shards + 1 val shard of 9,505 whole repositories; two builds identical except the `lang` tag of 17 blobs that RefineCode lists twice under one path with different tags, now chosen by a whole-row tie-break; 77,016,649 (25.1%) also in Stack-Edu (0 `in_stack_edu` flags disagree), union 396,393,470; 8 sanitized shards of ~67k files / 248 MB, 604,877 documents with val, 0 notebooks |
| The Stack v3 | `pretrain/fetch_stack_v3.py` | a part is ~21k repositories, 10-11 shards of ~140-170 MB; `--max-files` 8 then 20 fetched exactly 1 then 1 more part; 38 train parts (430 shards) + val, 10,754,153 documents, 11 GB, in 175 s; used as published (its publisher's near-dedup; see LOG) |
| Copyright/PII normalization of Software Heritage text | `pretrain/sanitize.py` | 17 edge cases; ~1.2 ms per file; on the fetched shards `<EMAIL>` in 3.1% (Stack-Edu) and 4.1% (RefineCode) of documents, `<IP>` 0.37% / 0.52%, `<SECRET>` 0.11% / 0.15%; 0.7% of characters changed |
| Training-time selection | `sources.read_training_documents`, `SAMPLING` | RefineCode Java kept 0.4461 (target 200/449 = 0.4454), HTML 0.1355 (64/474 = 0.1350), other languages whole; keyed on sha256(blob ID): over all 58.9M Java blobs 0.44547, within 1 sd; with `--without=stack_edu` 0 flagged documents kept |
| Verification | `pretrain/verify.py` | all fetched corpora pass in one DuckDB scan each (2-7 s): no empty document, every content_hash matches its text, every Stack v3 text is `size_bytes` long; over both whole manifests 0 of 306,346,760 RefineCode rows have an `in_stack_edu` flag that disagrees with the blob overlap; general-domain corpora in 10-12 s each: 21 of 21 fetched files (18.6 GB) have their pinned sha256 |
| General-domain corpora | `pretrain/sources.py`, `verify.py` | the registry's claims hold on the fetched files: ClimbMix ~252M chars / 56M tokens per shard, 1,024-document row groups; Smol-Data ~950M tokens per shard, FinePDFs-Edu / DCLM / FineWeb-Edu 0.47 / 0.32 / 0.21 of characters; TxT360 web-high-medium 1.14B tokens per shard, web 93% of characters (Common Crawl 66%, ClueWeb 19%, HPLT 8%), the same source shares in every row group sampled across chunk0 and chunk1. No empty document; exact duplicates 0.019% / 0.14% / 1 in 8.08M; dropped as eval-contaminated 0.07% / 0.27% / 0.14% of documents (tokens with a six-corpus vocab-32768 tokenizer) |
| Local-only reads, missing-file errors | `pretrain/sources.py` | the tokenizer stopped on `stack_v3` with the fetch command when its files held 674M of the 750M characters it needed |
| Lossless code segmentation | `pretrain/compile.py` | lossless by construction: a document's pieces cover it exactly, each later piece adding only a BOS. Packing crop at seq 2048 depends on the tokenizer: with a vocab-32768 tokenizer Stack-Edu 2.6% (30-32% crop-only), Stack v3 7.3%; with a vocab-8192 one Stack-Edu 3.6%, RefineCode 3.7%, Stack v3 16.2% (its val 1.7%), ClimbMix 14.8% (see Open 6); with a six-corpus vocab-32768 tokenizer Stack-Edu 2.7%, RefineCode 2.6%, Stack v3 14.5% (21 shards), 13.0% (all 430 shards of 38 parts; val 1.2%) |
| RefineCode without Stack-Edu | `compile.py --without`, `sources.mixture_without`, `sources.compiled_name`, `sources.compiled_mixture` | exactly the flagged documents dropped, through `read_training_documents`; compile names its output from the `--without` it applied; base_train and base_eval read a mixture's compiled data through the one `compiled_mixture` |
| Weighted mixtures | `pretrain/stream.py`, `base_train.py`, `base_eval.py` | 3-corpus run: planned shares 0.375 / 0.375 / 0.25 match the weights; 6-corpus run (all registered corpora, 1/6 each): planned shares 0.164-0.169; per-corpus bpb from base_eval |
| Mixture tokenizer + statistics | `nanochat/tokenizer.py` | composition per corpus and language; held-out bytes/token: prose 4.10, Python 3.68, C/C++ 3.31, Java 4.19, JS/TS 3.87, Go 3.12, Rust 3.60, other code 3.43; six-corpus tokenizer (vocab 32768, 2B chars, 1/6 each): prose 4.18, Python 3.64, C/C++ 3.43, Java 4.13, JS/TS 3.84, Go 3.15, Rust 3.57, other code 3.49 |

## Measured rates

| What | Result |
|---|---|
| Hub download, 8 files in parallel | 87-89 MiB/s |
| Software Heritage (obstore), one process, default thread pool (24 threads) | ~260-270 files/s, sanitized or not: a RefineCode shard of 67k files in ~260 s; while a 100 MB/s Hub download shares the 1 Gb/s link, a Stack-Edu shard takes 610-710 s |
| Software Heritage (obstore), 20 processes | ~5,300 files/s (the earlier httpx client, 7 processes x 64 connections: ~4,800) |
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
4. Stack v3 exact dedup against Stack-Edu and RefineCode by content_hash is measured by verify.py, not applied: 139
   and 107 exact copies among the fetched shards (content_hash is of the stored text, sanitized for both).
5. `refinecode_stackv2_reconstructed` is RefineCode's released file selection (its dedup and 130+ filters decide which
   files are listed), not its training text: notebooks are excluded rather than converted (the Jupyter-structured
   conversion is unpublished), copyright/PII normalization is an independent equivalent, the metadata covers only
   its files that are also in The Stack v2 (about half its raw code), and its code-related web data is absent. Java
   and HTML follow its published downsampling at read time. Equal-token comparisons against Stack-Edu and Stack v3
   remain to be run.
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
