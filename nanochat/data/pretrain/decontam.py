"""
Benchmark decontamination: compile drops every document that shares an n-word sequence (default 13, the GPT-3
criterion) with an item of the evaluation sets nanochat reports, so their scores are not inflated by pretraining:
every CORE task (scripts/base_eval.py) and the ARC, MMLU, GSM8K and HumanEval test sets (data/eval/).

Words are lowercased alphanumeric runs, so case, punctuation and whitespace do not hide a match. Each word hashes to
64 bits (blake2b) and an n-gram combines its words' hashes by a polynomial rolling hash, so the eval n-grams, hashed
once, and every compile worker agree without sharing state. N-grams never span two documents or two eval items.
"""

import os
import re
import json
import hashlib
from itertools import chain

import numpy as np
import yaml

from nanochat.data.eval.core import get_eval_bundle_dir
from nanochat.data.eval.arc import ARC
from nanochat.data.eval.mmlu import MMLU
from nanochat.data.eval.gsm8k import GSM8K
from nanochat.data.eval.humaneval import HumanEval

WORD = re.compile(r"[a-z0-9]+")
BASE = np.uint64(0x100000001B3)  # odd 64-bit multiplier of the rolling hash (the FNV-1a prime); arithmetic wraps mod 2^64


def ngram_hashes(texts, n):
    """(hashes uint64, text index int64) of every n-word sequence that lies within one of the texts."""
    words = [WORD.findall(t.lower()) for t in texts]
    counts = np.fromiter(map(len, words), dtype=np.int64, count=len(words))
    flat_words = list(chain.from_iterable(words))
    num = max(len(flat_words) - n + 1, 0)
    # each distinct word hashed once per call
    table = {w: int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "little") for w in set(flat_words)}
    flat = np.fromiter(map(table.__getitem__, flat_words), dtype=np.uint64, count=len(flat_words))
    hashes = np.zeros(num, dtype=np.uint64)
    for j in range(n):
        hashes = hashes * BASE + flat[j:j + num]
    owner = np.repeat(np.arange(len(texts), dtype=np.int64), counts)
    within = owner[:num] == owner[n - 1:]  # the n-gram starts and ends in the same text
    return hashes[within], owner[:num][within]


def contaminated(texts, eval_hashes, n):
    """Boolean mask over texts: True where a text shares at least one n-word sequence with the eval sets. The texts'
    n-gram hashes are looked up in the sorted eval_hashes in sorted order, which keeps the binary searches local."""
    hashes, owner = ngram_hashes(texts, n)
    order = np.argsort(hashes)
    found = np.take(eval_hashes, np.searchsorted(eval_hashes, hashes[order]), mode="clip") == hashes[order]
    mask = np.zeros(len(texts), dtype=bool)
    mask[owner[order[found]]] = True
    return mask


def strings(obj):
    """Every string in a JSON-like eval item (questions, passages, choices, answers, message contents)."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from strings(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from strings(value)


def eval_tasks():
    """The items of each CORE task and benchmark test set, one list of texts (one string per item) per task."""
    bundle = get_eval_bundle_dir()
    with open(os.path.join(bundle, "core.yaml"), encoding="utf-8") as f:
        core_tasks = yaml.safe_load(f)["icl_tasks"]
    for task in core_tasks:
        with open(os.path.join(bundle, "eval_data", task["dataset_uri"]), encoding="utf-8") as f:
            yield [" ".join(strings(json.loads(line))) for line in f]
    for task in [ARC(subset="ARC-Easy", split="test"), ARC(subset="ARC-Challenge", split="test"),
                 MMLU(subset="all", split="test"), GSM8K(subset="main", split="test"), HumanEval()]:
        yield [" ".join(strings(task[i])) for i in range(len(task))]


def eval_ngrams(n):
    """Sorted unique hashes of every n-word sequence of the eval sets, hashed one task at a time."""
    return np.unique(np.concatenate([ngram_hashes(items, n)[0] for items in eval_tasks()]))
