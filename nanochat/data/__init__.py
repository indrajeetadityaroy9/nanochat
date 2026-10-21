"""
The pretraining data of nanochat, one module per stage. The tokenizer (tokenizer.py) and its Rust crate bpe/ (the
SA-BPE trainer, arXiv 2604.14053; curation's deduplication and repository samples; the packer) are shared by the
stages, the loader and evaluation.

    HF Hub, Software Heritage (pinned)       sources.py: the sources, one mixture component each
        │  prepare.py, the only network step: fetch the code group whole and every other group up to its share of the
        │  longest run that repeats no code (config.MIXTURE); normalize every raw file to its category's schema
        │  (schema.py: repository file, notebook, text) with a local id, keeping Python 3 repository files
        ▼
    normalized/<source>/                     the Parquet store: one file per raw file
        │  prepare.py, curate: exact and near duplicates dropped across the repository corpora and across the
        │  document corpora (dedup.py: StarCoder2's MinHash LSH, arXiv 2402.19173), and the repository files
        │  GitHub Linguist (go-enry) marks vendored or generated; 1% val by repository / problem / document; a
        │  repository source's files partitioned by repository (DuckDB) and split into repository samples, the
        │  files that import one another in dependency order (DeepSeek-Coder, arXiv 2401.14196 §2.2)
        ▼
    curated/<source>/{train,val}/            the training documents
        │  tokenizer.py: SA-BPE trained on a sample of the curated train documents of every source
        │  compile.py: tokenize every document once (a repository sample: its files after their path comments),
        │  unpadded and without BOS (any sequence length)
        ▼
    <split>.bin tokens, <split>.idx document offsets
        │  pack.py, per sequence length: cut long documents between whole statements (repository samples) or at line
        │  ends, make train pieces of the repository corpora AST-FIM (arXiv 2506.00204) fill-in-the-middle examples
        │  (their sections encoded each on its own), BOS-aligned Best-Fit-Decreasing packing (no document cut to fill
        │  a row; padding instead), as an index of each row's pieces and the examples' tokens
        ▼
    <split>-T<seq_len>.pieces, .rows, .fim
        │  mmap'd in place from local NVMe; each row built on the fly
        ▼
    mixture batches, resumable              loader.py
        │  pinned memory, asynchronous copy
        ▼
    GPU

    python -m nanochat.data.prepare
    python -m nanochat.data.tokenizer
    python -m nanochat.data.compile
    python -m nanochat.data.pack --stage file
    python -m nanochat.data.pack --stage repo

Everything is stored under paths.DATA:
    raw/<source>/                          fetched files at their Hub paths (Stack-Edu: metadata with Software
                                           Heritage text)
    normalized/<source>/                   <k>.parquet normalized documents, one per raw file
    dedup/                                 deduplication's signature, id and flag files while curating
    shuffle/<source>/                      a repository source's files partitioned by repository while it is curated
    curated/<source>/{train,val}/          <k>.parquet training documents (a repository source: its files in sample
                                           order, one file per partition)
    compiled/<source>-<tok>/               {train,val}.bin tokens, .idx document offsets, .json documents per curated
                                           file (compile.py); {train,val}-T<seq_len>.pieces, .rows, .fim: the packed
                                           rows (pack.py)
    huggingface/                           the Hugging Face cache (Stack-Edu metadata, the evaluation benchmarks' files)
"""
