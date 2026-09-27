"""
All dataset and data-loading components of nanochat: one package per stage, and two modules they share.
    storage.py    the data root, <base_dir>/data; task and eval files are downloaded once, on first use, under a lock
    task.py       chat tasks: the Task base, one split of a pinned HF dataset repo, the multiple-choice prompt format

pretrain/
    HF Hub, Software Heritage (pinned)       fetch.py, code/ (swh.py, stack_v3.py): the only network step
        │  resumable, deterministic; general-domain files as they are, code materialized as the sources' text
        ▼
    local corpus files + manifest.json       sources.py: the registry, local reads, training selection, mixtures
        │  select training documents (published downsampling, overlaps), drop eval-contaminated docs, tokenize,
        │  split long code files, BOS-aligned best-fit packing, once
        ▼
    one file of packed token rows per split  compile.py, decontam.py
        │  mmap'd in place from local NVMe
        ▼
    weighted, elastic row order              stream.py
        │  pinned memory, asynchronous copy
        ▼
    GPU

posttrain/    SFT data: SmolTalk (smoltalk.py), the task mixture and its packing loader (sft.py)
eval/         the CORE bundle (core.py) and the chat benchmarks ARC, MMLU, GSM8K and HumanEval with their graders

SFT also trains on the MMLU and GSM8K train splits, RL on GSM8K's with its grader as the reward, and pretraining
decontaminates against the eval sets, so those stages import from eval/; eval/ imports from neither.

Everything is stored under <base_dir>/data:
    raw/<corpus>/                          fetched corpora: manifest.json, then the files it lists
    compiled/<dataset>-T<seq_len>-<tok>/   {train,val}.bin: packed token rows; {train,val}.json: their statistics
    <org>/<repo>/                          task data at repo-relative paths (and metadata while a manifest is built)
    eval_bundle/                           CORE data
"""
