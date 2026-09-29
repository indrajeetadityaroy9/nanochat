"""
All dataset and data-loading components of nanochat: one package per stage, and two modules they share.
    storage.py    the data root, <base_dir>/data; task and eval files are downloaded once per node, under a lock; the
                  atomic, durable JSON writer for manifests and statistics
    task.py       benchmark tasks: the Task base, one split of a pinned HF dataset repo, the multiple-choice prompt format

pretrain/
    HF Hub, Software Heritage (pinned)       fetch.py, code/ (swh.py, stack_v3.py): the only network step
        │  resumable, deterministic; text corpora's files as they are, code materialized as the sources' text
        ▼
    local corpus files + manifest.json       sources.py: the registry and local reads
        │  drop train copies of val docs, tokenize, split long documents at line ends, BOS-aligned lossless best-fit
        │  packing, once
        ▼
    one file of packed token rows per split  compile.py
        │  mmap'd in place from local NVMe
        ▼
    weighted, elastic row order              stream.py
        │  pinned memory, asynchronous copy
        ▼
    GPU

eval/         the CORE bundle (core.py) and the ARC, MMLU, GSM8K, HumanEval and MBPP test sets

Everything is stored under <base_dir>/data:
    raw/<corpus>/                          fetched corpora: manifest.json, then the files it lists
    compiled/<dataset>-T<seq_len>-<tok>/   {train,val}.bin: packed token rows; {train,val}.json: their statistics
    <org>/<repo>/                          task data at repo-relative paths (and metadata while a manifest is built)
    eval_bundle/                           CORE data
"""
