"""DRY ("Don't Repeat Yourself") sampling (arXiv 2608.22761 §3, App. A Eq. 1) as a vLLM V1 logits processor: a
candidate token v that would extend the context's suffix into a verbatim repeat of an earlier span has its logit
lowered by multiplier * base^(n - allowed_length), n the longest such match (the suffix and the earlier span free of
sequence breakers, n >= allowed_length, v no breaker). The context is the last `range` tokens of prompt + output
(App. H, Table 8: multiplier 0.8, base 1.75, allowed length 2, range 1,024); matches are measured up to MAX_MATCH
tokens (the penalty at MAX_MATCH, 0.8 * 1.75^62, already removes the token). Applied only while the output has not
emitted "</think>" (an answer copies code from its reasoning). Breakers: every token whose text holds a newline, a
colon, a quote or an asterisk (the paper's default set, §3.2).

A request takes it through SamplingParams(extra_args={"dry": {}}) (or with overrides of DEFAULTS); the engine
through LLM(logits_processors=[DryLogitsProcessor]).
"""

import numpy as np
import torch
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

DEFAULTS = {"multiplier": 0.8, "base": 1.75, "allowed_length": 2, "range": 1024}
MAX_MATCH, BREAK_CHARS = 64, ("\n", ":", '"', "'", "*")


def breakers() -> np.ndarray:
    """A boolean mask over the vocabulary: the tokens whose text holds a breaker character."""
    from corpus.tokenizer import get_tokenizer

    enc = get_tokenizer().enc
    mask = np.zeros(enc.n_vocab, bool)
    for t in range(enc.n_vocab):
        try:
            text = enc.decode_single_token_bytes(t).decode("utf-8", errors="replace")
        except KeyError:  # an id without a token
            text = ""
        mask[t] = any(c in text for c in BREAK_CHARS)
    return mask


class Dry:
    """One request's DRY state: the vocabulary's breaker mask and whether its output has closed its reasoning."""

    def __init__(self, breaker: np.ndarray, end: list[int], multiplier: float, base: float, allowed_length: int,
                 range: int):
        self.breaker, self.end = breaker, end
        self.multiplier, self.base, self.allowed, self.range = multiplier, base, allowed_length, range
        self.closed_at = None  # the output length by which "</think>" was seen
        self.scanned = None  # the output length scanned so far; None before the prompt is

    def closed(self, prompt: list[int], output: list[int]) -> bool:
        """Whether prompt + output holds "</think>" (a chunked generation's earlier rounds are in its prompt)."""
        if self.closed_at is None:
            k = len(self.end)
            if self.scanned is None:
                self.scanned = 0
                if any(prompt[i:i + k] == self.end for i in range(len(prompt) - k + 1)):
                    self.closed_at = 0
                    return True
            for i in range(max(0, self.scanned - k + 1), len(output) - k + 1):
                if output[i:i + k] == self.end:
                    self.closed_at = i + k
                    break
            self.scanned = len(output)
        return self.closed_at is not None

    def __call__(self, prompt: list[int], output: list[int], logits: torch.Tensor) -> torch.Tensor:
        if self.closed(prompt, output):
            return logits
        tail = output[-self.range:]
        if len(tail) < self.range:
            tail = list(prompt[max(0, len(prompt) - (self.range - len(tail))):]) + list(tail)
        x = np.asarray(tail, dtype=np.int64)
        t = len(x) - 1
        if t < 1 or self.breaker[x[t]]:
            return logits
        candidates = np.flatnonzero(x[:t] == x[t])  # i < t with x_i = x_t: m(i, t) >= 1
        length = np.ones(len(candidates), np.int64)
        alive = candidates
        for ell in range(1, MAX_MATCH):
            if t - ell < 0 or self.breaker[x[t - ell]]:
                break
            alive = alive[(alive - ell >= 0)]
            alive = alive[x[np.maximum(alive - ell, 0)] == x[t - ell]]
            if not len(alive):
                break
            length[np.searchsorted(candidates, alive)] += 1
        keep = length >= self.allowed
        if not keep.any():
            return logits
        tokens, n = x[candidates[keep] + 1], length[keep]
        best = {}
        for v, m in zip(tokens.tolist(), n.tolist()):
            if not self.breaker[v] and m > best.get(v, 0):
                best[v] = m
        if best:
            v = torch.tensor(list(best), device=logits.device)
            penalty = self.multiplier * self.base ** (np.array(list(best.values()), np.float64) - self.allowed)
            logits[v] -= torch.tensor(penalty, dtype=logits.dtype, device=logits.device)
        return logits


class DryLogitsProcessor(AdapterLogitsProcessor):
    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        super().__init__(vllm_config, device, is_pin_memory)
        from corpus.tokenizer import get_tokenizer

        self.breaker = breakers()
        self.end = get_tokenizer().enc.encode_ordinary("</think>")

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params):
        config = (params.extra_args or {}).get("dry")
        return None if config is None else Dry(self.breaker, self.end, **(DEFAULTS | config))
