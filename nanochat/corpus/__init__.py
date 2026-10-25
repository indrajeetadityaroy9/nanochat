"""
The pre-training corpus (docs/corpus.md), one module per stage, under paths.CORPUS; run order: docs/pretraining.md.

  schema.py     document categories: schema, split group, whether the content is code
  sources.py    the 18 sources, the corpus's file paths, curated texts and repository samples
  mixture.py    the run's tokens, its two stages and their groups' shares; a run's weights and rows per corpus
  parallel.py   spawned workers, a bounded ordered map, the memory budget
  prepare.py    fetch each source's Hub files and standardize them
  curate.py     Python 3 check with repair, exact deduplication, split, curated shards
  publish.py    the curated shards to the Hub (config.DATASET) and back
  tokenizer.py  load a vocabulary (ablations/tokenizer.py trains, decides and adopts it)
  compile.py    curated documents to tokens; the 32K stage's repository documents
  pack.py       each stage's training rows with FIM examples, and their weights
  layout.py     the compiled and packed data's paths and counts (torch-free, for spawned workers)
  loader.py     the packed data's micro-batches
  crate/        the Rust extension: SA-BPE trainer, repository samples, packing

  CORPUS/raw/<source>/<Hub path>                fetched Hub columns (prepare.fetch)
  CORPUS/standardized/<source>/<k>.parquet      a fetched file in its category's schema
  CORPUS/checked/<source>/<k>.parquet           a Python source's standardized file, every file parsing
  CORPUS/curated/<source>/<split>/<k>.parquet   curated shard k (the published dataset)
  CORPUS/tokenizer/tokenizer.{tiktoken,pattern} the corpus's tokenizer
  CORPUS/tokenizer-ablation/<arm>/, compiled/   the tokenizer ablation's arms and their data
  CORPUS/compiled/<corpus>-<data key>/<split>.* compiled split (.bin, .idx, .json) and packs (-T<seq_len>.pieces,
                                                .rows, .fim)
"""
