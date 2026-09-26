"""
All dataset and data-loading components of nanochat, for pretraining and post-training.

Pretraining:

    raw corpus (HF parquet, pinned)          sources.py
        │  drop eval-contaminated docs, tokenize, BOS-aligned best-fit packing, once, CPU-parallel
        ▼
    immutable token shards + index.json      compile.py, decontam.py, shards.py
        │  mmap'd in place from local NVMe
        ▼
    deterministic elastic row order          stream.py
        │  DataLoader workers, pinned memory
        ▼
    GPU

Post-training and evaluation data:
    tasks/          SFT, RL and chat-eval task datasets (pinned HF parquet)
    sft.py          SFT conversation packing loader
    eval_bundle.py  CORE benchmark data

Everything is stored under <base_dir>/data, each file downloaded once on first use (storage.py):
    <org>/<repo>/                          HF dataset files (raw corpora, task data) at repo-relative paths
    compiled/<dataset>-T<seq_len>-<tok>/   {train,val}/index.json + shards
    eval_bundle/                           CORE data
"""
