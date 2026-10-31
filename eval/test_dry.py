"""eval/dry.py's penalty against App. A Eq. 1 of arXiv 2608.22761 on random token sequences (loops, then noise).
Run in the nanochat-eval image (eval/dry.py imports vLLM's logits-processor interface):

    python -m pytest eval/test_dry.py
"""

import numpy as np
import pytest
import torch

from eval.dry import DEFAULTS, MAX_MATCH, Dry


def dry_reference(x: list[int], breaker: np.ndarray, multiplier: float, base: float, allowed_length: int) -> dict:
    """Eq. 1's penalty per candidate token v: n_t(v) = max over i < t with x_{i+1} = v of the backward match length
    m(i, t) (x_i = x_t, matching through no breaker), at most MAX_MATCH."""
    t, n = len(x) - 1, {}
    for i in range(t):
        m = 0
        while m < MAX_MATCH and i - m >= 0 and x[i - m] == x[t - m] and not breaker[x[t - m]]:
            m += 1
        if m:
            n[x[i + 1]] = max(n.get(x[i + 1], 0), m)
    return {v: multiplier * base ** (m - allowed_length) for v, m in n.items()
            if m >= allowed_length and not breaker[v]}


@pytest.mark.parametrize("seed", range(20))
def test_dry_matches_eq_1(seed):
    rng = np.random.default_rng(seed)
    breaker = rng.random(12) < 0.2
    period = rng.integers(0, 12, rng.integers(1, 9)).tolist()
    x = (period * 40 + rng.integers(0, 12, 30).tolist())[:rng.integers(2, 300)]
    prompt, output = x[:len(x) // 3], x[len(x) // 3:]
    got = Dry(breaker, [99], **DEFAULTS)(prompt, output, torch.zeros(12, dtype=torch.float64))
    expected = np.zeros(12)
    for v, p in dry_reference(x, breaker, DEFAULTS["multiplier"], DEFAULTS["base"], DEFAULTS["allowed_length"]).items():
        expected[v] = -p
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-12)
