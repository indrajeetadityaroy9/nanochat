"""
Test eval decontamination (nanochat/data/pretrain/decontam.py): which documents share an n-word sequence with an eval item.

python -m pytest tests/test_data_decontam.py -v
"""

import numpy as np

from nanochat.data.pretrain.decontam import contaminated, ngram_hashes

N = 13
EVAL = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May."


def eval_hashes(texts):
    return np.unique(ngram_hashes(texts, N)[0])


def test_overlap_survives_case_punctuation_and_whitespace():
    index = eval_hashes([EVAL])
    docs = [
        "Worked example. " + EVAL + " How many clips did she sell altogether?",
        "NATALIA SOLD CLIPS TO 48 OF HER FRIENDS IN APRIL -- and then\n\nshe sold half as many clips in May!",
        "An unrelated document about a quick brown fox that jumps over the lazy dog again and again and again.",
    ]
    assert contaminated(docs, index, N).tolist() == [True, True, False]


def test_overlap_shorter_than_n_is_kept():
    index = eval_hashes([EVAL])
    twelve = "natalia sold clips to 48 of her friends in april and then"
    assert contaminated([twelve + " went home to count the rest of them"], index, N).tolist() == [False]


def test_ngrams_never_span_two_texts():
    first, second = "natalia sold clips to 48 of her", "friends in april and then she" # 7 + 6 = 13 words of EVAL
    index = eval_hashes([EVAL])
    assert contaminated([first, second], index, N).tolist() == [False, False]
    assert contaminated([first + " " + second], index, N).tolist() == [True]
    # nor two eval items: split across items, the same 13 words give no eval n-gram
    assert eval_hashes([first, second]).size == 0
