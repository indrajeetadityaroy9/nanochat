"""
The pretraining corpus, built from the ground up, one module per stage, under paths.CORPUS:

  sources.py   the sources: a Hub dataset each (read at its latest revision), the files it reads, its document category
  schema.py    the document categories' schemas
  prepare.py   fetch every Hub file of a source, then standardize each into its category's schema

  CORPUS/raw/<source>/<Hub path>             the Hub files (prepare.fetch)
  CORPUS/standardized/<source>/<k>.parquet   Hub file k of the source, in its schema (prepare.standardize_file)
"""
