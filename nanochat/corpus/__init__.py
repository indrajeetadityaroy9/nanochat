"""
The pretraining corpus, built from the ground up as its design document specifies (normalization_doc.md), one module
per stage, under paths.CORPUS:

  sources.py    the sources: a Hub dataset each (read at its latest revision), the files it reads, its document category
  schema.py     the document categories' schemas and the split's modulus
  mixture.py    the run's tokens, its two stages and their groups' shares, the sources read in part
  prepare.py    fetch a source's Hub files (of a source read in part, its extent), standardize each into its schema
  curate.py     the Python 3 check with repair, exact deduplication, the split, the curated shards
  publish.py    the curated shards to the Hub (config.DATASET), and back where the tokenizer is trained
  tokenizer.py  the corpus's tokenizer, trained on its curated train splits at the file stage's shares of bytes
  compile.py    the curated documents in tokens, and the 32K stage's repository documents per language
  pack.py       each stage's training rows (and FIM examples), holding exactly what it reads; the stages' weights
  loader.py     the compiled and packed data, and their micro-batches

Run in this order; everything up to publish is text, everything after it needs the tokenizer:

  python -m nanochat.corpus.prepare <source> ...   (every source in mixture.READ)
  python -m nanochat.corpus.curate
  python -m nanochat.corpus.publish push           (prints the dataset's revision)
  python -m nanochat.corpus.publish pull --revision <sha>
  python -m nanochat.corpus.tokenizer
  python -m nanochat.corpus.compile
  python -m nanochat.corpus.pack

  CORPUS/raw/<source>/<Hub path>                     the Hub files' columns the standardizer reads (prepare.fetch)
  CORPUS/standardized/<source>/<k>.parquet           fetched Hub file k of the source, in its schema
  CORPUS/manifest.json                               each source's repo, revision, files and rows (prepare.record);
                                                     the code's commit and the curated shards' rows (publish.push)
  CORPUS/checked/<source>/<k>.parquet                a Python repository source's file k, every file parsing
  CORPUS/curated/<source>/<split>/<k>.parquet        the source's curated shard k of a split (published)
  CORPUS/README.md                                   the published dataset's card (publish.card)
  CORPUS/tokenizer/tokenizer.{tiktoken,pattern}      the corpus's tokenizer
  CORPUS/compiled/<corpus>-<data key>/<split>.*      the compiled split (.bin, .idx, .json) and its packs
                                                     (<split>-T<seq_len>.pieces, .rows, .fim)
  CORPUS/packed.json                                 each stage's seq_len, rows and corpus weights (pack.py)
"""
