"""
All dataset and data-loading components of nanochat, for pretraining and post-training.

Pretraining:

    raw corpus (HF parquet, pinned)          sources.py
        │  drop eval-contaminated docs, tokenize, BOS-aligned best-fit packing, once, CPU-parallel
        ▼
    one file of packed token rows per split  compile.py, decontam.py
        │  mmap'd in place from local NVMe
        ▼
    deterministic elastic row order          stream.py
        │  pinned memory, asynchronous copy
        ▼
    GPU

Post-training and evaluation data:
    tasks/          SFT, RL and chat-eval task datasets (pinned HF parquet)
    sft.py          SFT conversation packing loader
    eval_bundle.py  CORE benchmark data

Everything is stored under <base_dir>/data, each file downloaded once on first use (storage.py):
    <org>/<repo>/                          HF dataset files (raw corpora, task data) at repo-relative paths
    compiled/<dataset>-T<seq_len>-<tok>/   {train,val}.bin: packed token rows
    eval_bundle/                           CORE data
"""
