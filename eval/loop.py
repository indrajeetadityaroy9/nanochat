"""The loop measure and stop rule of the natural-termination probe (eval/official_termination.py `generate`), shared
with the served loop stop (eval/official_loop_stop.py). Import-light (numpy only): `vllm serve`'s processes import it.

A response is stopped at a check, every CHUNK generated tokens from REPEAT_TAIL on, when its output holds no END and
its last REPEAT_TAIL tokens have `repetition` >= LOOP.
"""

from collections import Counter

import numpy as np

END = "</think>"
CHUNK = 2048  # generated tokens between checks (the probe's rounds)
REPEAT_TAIL, REPEAT_WINDOW, LOOP = 4096, 32, 0.99


def repetition(ids: list[int]) -> float:
    """The share of the last REPEAT_TAIL tokens inside a REPEAT_WINDOW-token window that recurs within them."""
    tail = ids[-REPEAT_TAIL:]
    windows = [tuple(tail[i:i + REPEAT_WINDOW]) for i in range(len(tail) - REPEAT_WINDOW + 1)]
    counts = Counter(windows)
    covered = np.zeros(len(tail), bool)
    for i, w in enumerate(windows):
        if counts[w] > 1:
            covered[i:i + REPEAT_WINDOW] = True
    return float(covered.mean()) if len(tail) else 0.0


def looping(output: list[int], decode) -> float | None:
    """The `repetition` of an output that the rule stops at a check (len(output) a positive multiple of CHUNK, at least
    REPEAT_TAIL), else None; decode: token ids -> text."""
    n = len(output)
    if n < REPEAT_TAIL or n % CHUNK:
        return None
    r = repetition(output)
    return r if r >= LOOP and END not in decode(output) else None


def stopped(ids: list[int], bos: int, decode) -> bool:
    """Whether a served response (its generated ids, the closing BOS included) was ended by the loop stop: it ends
    with the BOS forced at a check the rule stops (a BOS the model emits there itself would have been forced)."""
    return ids[-1:] == [bos] and looping(ids[:-1], decode) is not None
