"""
All dataset and data-loading components of nanochat, for pretraining and post-training.

Pretraining follows a decoupled storage/compute streaming design:

    raw corpus (HF parquet, pinned)          sources.py
        │  tokenize + BOS-aligned best-fit packing, once, CPU-parallel
        ▼
    immutable token shards + index.json      compile.py, shards.py
        │  any fsspec store: s3:// (S3/MinIO), gs://, hf://, shared filesystem
        ▼
    node-local NVMe cache, mmap'd            stream.py (storage.py does locked, atomic fetches)
        │  deterministic elastic order, DataLoader workers, pinned memory
        ▼
    GPU

Post-training and evaluation data:
    tasks/          SFT, RL and chat-eval task datasets (pinned HF parquet)
    sft.py          SFT conversation packing loader
    eval_bundle.py  CORE benchmark data

Everything is stored under $NANOCHAT_DATA_DIR (default <base_dir>/data):
    raw/<dataset>/                         raw parquet at repo-relative paths
    compiled/<dataset>-T<seq_len>-<tok>/   {train,val}/index.json + shards (also the stream cache)
    tasks/<org>--<repo>/                   task parquet at repo-relative paths
    eval_bundle/                           CORE data
"""
