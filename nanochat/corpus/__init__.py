"""
The pretraining corpus, as its design document specifies (docs/corpus.md), one module per stage, under
paths.CORPUS:

  schema.py     the document categories: each one's schema, split group and whether its content is code
  sources.py    the sources (a Hub dataset, the files it reads, its category) and the corpus's files
  mixture.py    the run's tokens, its two stages and their groups' shares, the sources read in part
  parallel.py   spawned workers and a bounded, ordered map over them
  prepare.py    fetch each source's Hub files (of a source read in part, its extent), standardize them into its schema
  curate.py     the Python 3 check with repair, exact deduplication, the split, the curated shards
  publish.py    the curated shards to the Hub (config.DATASET), and back where the tokenizer is trained
  tokenizer.py  the corpus's tokenizer: its directory and loader (ablations/tokenizer.py trains, decides, adopts it)
  compile.py    the curated documents in tokens, and the 32K stage's repository documents per language
  pack.py       each stage's training rows (and FIM examples), holding exactly what it reads; the stages' weights
  loader.py     the compiled and packed data, and their micro-batches

Run in this order; everything up to publish is text, everything after it needs the tokenizer:

  python -m nanochat.corpus.prepare
  python -m nanochat.corpus.curate
  python -m nanochat.corpus.publish push
  python -m nanochat.corpus.publish pull
  python -m ablations.tokenizer vocabularies|intrinsic|data|adopt  (adopt: the decided arm to CORPUS/tokenizer)
  python -m nanochat.corpus.compile
  python -m nanochat.corpus.pack

  CORPUS/raw/<source>/<Hub path>                     the Hub files' columns the standardizer reads (prepare.fetch)
  CORPUS/standardized/<source>/<k>.parquet           fetched Hub file k of the source, in its category's schema
  CORPUS/checked/<source>/<k>.parquet                a Python source's standardized file k, every file parsing
  CORPUS/curated/<source>/<split>/<k>.parquet        the source's curated shard k of a split (published)
  CORPUS/README.md                                   the published dataset's card (publish.card)
  CORPUS/tokenizer/tokenizer.{tiktoken,pattern}      the corpus's tokenizer
  CORPUS/tokenizer-ablation/<arm>/, compiled/        the tokenizer ablation's arms and their compiled data
  CORPUS/compiled/<corpus>-<data key>/<split>.*      the compiled split (.bin, .idx, .json) and its packs
                                                     (<split>-T<seq_len>.pieces, .rows, .fim)
"""
