"""
Benchmark decontamination of pretraining data. At compile time, every document that shares an n-word
sequence (default 13, the GPT-3 criterion) with an item of the evaluation sets nanochat reports is dropped,
so evaluation scores are not inflated by benchmark text seen in pretraining:
    CORE        every task of the DCLM eval bundle's core.yaml (scripts/base_eval.py)
    chat eval   ARC-Easy, ARC-Challenge, MMLU and GSM8K test splits, and HumanEval (scripts/chat_eval.py)

Words are lowercased alphanumeric runs, so case, punctuation and whitespace differences do not hide a match.
Each word is hashed to 64 bits with blake2b and n-grams combine the word hashes with a polynomial rolling
hash, so the index built once per compile and the check in every compile worker agree without sharing state.
N-grams never span two documents (or two eval items).
"""

import os
import json
import re
import hashlib
from itertools import chain

import numpy as np
import yaml

WORD = re.compile(r"[a-z0-9]+")
_BASE = np.uint64(0x100000001B3) # odd 64-bit multiplier of the rolling hash (the FNV-1a prime); arithmetic wraps mod 2^64
FILTER_BITS = 28 # bit filter in front of the index: 2^28 bits = 32 MiB


def _word_hash(word):
    return int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "little")


def ngram_hashes(texts, n):
    """(hashes uint64, text index int64) of every n-word sequence that lies within one of the texts."""
    words = [WORD.findall(t.lower()) for t in texts]
    counts = np.fromiter(map(len, words), dtype=np.int64, count=len(words))
    flat_words = list(chain.from_iterable(words))
    num = len(flat_words) - n + 1
    if num <= 0:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int64)
    table = {w: _word_hash(w) for w in set(flat_words)} # hash each distinct word once per call
    flat = np.fromiter(map(table.__getitem__, flat_words), dtype=np.uint64, count=len(flat_words))
    hashes = np.zeros(num, dtype=np.uint64)
    for j in range(n):
        hashes = hashes * _BASE + flat[j:j + num]
    owner = np.repeat(np.arange(len(texts), dtype=np.int64), counts)
    within = owner[:num] == owner[n - 1:] # the n-gram starts and ends in the same text
    return hashes[within], owner[:num][within]


class EvalIndex:
    """
    Set of eval n-gram hashes: sorted for exact binary search, behind a one-hash bit filter. A document
    n-gram whose filter bit is clear is certainly not an eval n-gram, which rejects all but ~2% of them
    with one lookup (binary-searching every n-gram of a corpus is the slow part otherwise).
    """

    def __init__(self, hashes):
        self.hashes = np.unique(hashes)
        self.bits = np.zeros(2**FILTER_BITS // 8, dtype=np.uint8)
        slots = self._slots(self.hashes)
        np.bitwise_or.at(self.bits, slots >> 3, (1 << (slots & 7)).astype(np.uint8))

    @staticmethod
    def _slots(hashes):
        return (hashes >> np.uint64(64 - FILTER_BITS)).astype(np.intp)

    def __len__(self):
        return len(self.hashes)

    def contains(self, hashes):
        """Boolean mask over hashes: which of them are eval n-grams."""
        slots = self._slots(hashes)
        found = ((self.bits[slots >> 3] >> (slots & 7).astype(np.uint8)) & 1).astype(bool)
        maybe = np.flatnonzero(found)
        if len(maybe):
            pos = np.minimum(np.searchsorted(self.hashes, hashes[maybe]), len(self.hashes) - 1)
            found[maybe] = self.hashes[pos] == hashes[maybe]
        return found

    def digest(self):
        """Identity of the index: compiled data records it, and a changed eval set forces a recompile."""
        return hashlib.blake2b(self.hashes.tobytes(), digest_size=8).hexdigest()


def contaminated(texts, index, n):
    """Boolean mask over texts: True where a text shares at least one n-word sequence with the EvalIndex."""
    mask = np.zeros(len(texts), dtype=bool)
    hashes, owner = ngram_hashes(texts, n)
    mask[owner[index.contains(hashes)]] = True
    return mask


def _strings(obj):
    """Every string in a JSON-like eval item (questions, passages, choices, answers, message contents)."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _strings(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _strings(value)


def eval_texts():
    """The text of every item of the CORE tasks and the chat-eval test sets, one string per item."""
    from nanochat.data.eval.core import get_eval_bundle_dir
    from nanochat.data.eval.arc import ARC
    from nanochat.data.eval.mmlu import MMLU
    from nanochat.data.eval.gsm8k import GSM8K
    from nanochat.data.eval.humaneval import HumanEval

    bundle = get_eval_bundle_dir()
    with open(os.path.join(bundle, "core.yaml"), encoding="utf-8") as f:
        core_tasks = yaml.safe_load(f)["icl_tasks"]
    for task in core_tasks:
        with open(os.path.join(bundle, "eval_data", task["dataset_uri"]), encoding="utf-8") as f:
            for line in f:
                yield " ".join(_strings(json.loads(line)))
    for task in [ARC(subset="ARC-Easy", split="test"), ARC(subset="ARC-Challenge", split="test"),
                 MMLU(subset="all", split="test"), GSM8K(subset="main", split="test"), HumanEval()]:
        for i in range(len(task)):
            yield " ".join(_strings(task[i]))


def build_eval_index(n, chunk_texts=10_000):
    """EvalIndex of every n-word sequence of the eval sets, hashed a chunk of items at a time to bound memory."""
    parts, texts = [], []
    for text in eval_texts():
        texts.append(text)
        if len(texts) == chunk_texts:
            parts.append(ngram_hashes(texts, n)[0])
            texts = []
    parts.append(ngram_hashes(texts, n)[0])
    return EvalIndex(np.concatenate(parts))
