"""
All dataset and data-loading components of nanochat: one package per stage, and two modules they share.
    storage.py    the data root, <base_dir>/data, and downloads: each file once, on first use, under a lock
    task.py       chat tasks: the Task base, one split of a pinned HF dataset repo, the multiple-choice prompt format

pretrain/
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

posttrain/    SFT data: SmolTalk (smoltalk.py), the task mixture and its packing loader (sft.py)
eval/         the CORE bundle (core.py) and the chat benchmarks ARC, MMLU, GSM8K and HumanEval with their graders

SFT also trains on the MMLU and GSM8K train splits, RL on GSM8K's with its grader as the reward, and pretraining
decontaminates against the eval sets, so those stages import from eval/; eval/ imports from neither.

Everything is stored under <base_dir>/data:
    <org>/<repo>/                          HF dataset files (raw corpora, task data) at repo-relative paths
    compiled/<dataset>-T<seq_len>-<tok>/   {train,val}.bin: packed token rows
    eval_bundle/                           CORE data
"""
