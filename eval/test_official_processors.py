"""The served model's text processing on recorded inputs: the reasoning parser's split, the loop rule and its served
stop, and DRY against App. A Eq. 1 of arXiv 2608.22761. vLLM's V2 logits-processor state is stood in by the fields the
processors read (vllm/v1/worker/gpu/sample/logits_processor/interface.py). Run in the nanochat-eval image:

    python -m pytest eval/test_official_processors.py
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from corpus.tokenizer import get_tokenizer
from eval.dry import DEFAULTS, MAX_MATCH, Dry, DryLogitsProcessor
from eval.loop import CHUNK, END, LOOP, REPEAT_TAIL, looping, repetition, stopped
from eval.official_loop_stop import LOG, LoopStopLogitsProcessor
from eval.official_reasoning_parser import NanochatReasoningParser

# The last characters of a loop-stopped response (step 7676's probe, HumanEval/1: 20,480 tokens, repetition 1.0)
LOOP_TAIL = ("1101. '(', counter becomes 3.\n1102. ' ', skip.\n1103. ')', counter becomes 2.\n1104. '(', counter "
             "becomes 3.\n1105. ' ', skip.\n1106. ')', counter becomes 2.\n1107. '(', counter becomes 3.\n1108. ' ', "
             "skip.\n1109. ')', counter becomes 2.\n1200. '(', counter becomes 3.\n")
TOK = get_tokenizer()
BOS, V = TOK.bos_token_id, TOK.enc.n_vocab


def loop_ids() -> list[int]:
    ids = TOK.enc.encode_ordinary(LOOP_TAIL * 200)
    assert len(ids) > REPEAT_TAIL + CHUNK
    return ids


def prose_ids() -> list[int]:
    """Text that does not loop: this repository's official-tool sources."""
    return TOK.enc.encode_ordinary("".join(p.read_text() for p in sorted(Path(__file__).parent.glob("official*.py"))))


def test_reasoning_parser_splits_at_the_last_close():
    shim = SimpleNamespace(decode=TOK.enc.decode, encode=lambda text, add_special_tokens: TOK.enc.encode_ordinary(text))
    parser = NanochatReasoningParser(shim)
    reasoning = "So the code seems to handle all these cases.\n\nI think this should work.\n"
    answer = "```python\ndef f():\n    return 1\n```"
    text = f"{reasoning}{END}\n\n{answer}\n"
    assert parser.extract_reasoning(text, None) == (reasoning, answer)
    assert parser.extract_reasoning(f"a{END}b{END} c ", None) == (f"a{END}b", "c")
    assert parser.extract_reasoning(LOOP_TAIL, None) == (LOOP_TAIL, "")  # unclosed: all reasoning, empty answer
    ids = TOK.enc.encode_ordinary(text)
    assert parser.is_reasoning_end(ids) and not parser.is_reasoning_end(TOK.enc.encode_ordinary(LOOP_TAIL))
    assert TOK.enc.decode(parser.extract_content_ids(ids)) == f"\n\n{answer}\n"
    assert parser.extract_content_ids(TOK.enc.encode_ordinary(LOOP_TAIL)) == []


def test_loop_rule_on_recorded_loop():
    ids, prose, decode = loop_ids(), prose_ids(), TOK.enc.decode
    assert repetition(ids[:REPEAT_TAIL]) == 1.0 and repetition(prose[:REPEAT_TAIL]) < LOOP
    assert looping(ids[:REPEAT_TAIL], decode) == 1.0 and looping(ids[:REPEAT_TAIL + CHUNK], decode) == 1.0
    assert looping(ids[:REPEAT_TAIL - CHUNK], decode) is None  # before REPEAT_TAIL
    assert looping(ids[:REPEAT_TAIL + 1], decode) is None  # not at a check
    assert looping(prose[:REPEAT_TAIL], decode) is None
    end = TOK.enc.encode_ordinary(END)
    closed = end + ids[:REPEAT_TAIL - len(end)]
    assert looping(closed, decode) is None  # a closed response is never stopped
    assert stopped([*ids[:REPEAT_TAIL], BOS], BOS, decode)
    assert not stopped(ids[:REPEAT_TAIL], BOS, decode)
    assert not stopped([*ids[:REPEAT_TAIL + 1], BOS], BOS, decode)
    assert not stopped([*prose[:REPEAT_TAIL], BOS], BOS, decode)


def request_states(sequences: dict[int, tuple[list[int], int, int]], width: int) -> SimpleNamespace:
    """V2's per-slot buffers for {slot: (prompt + committed tokens, prompt length, prefill length)}."""
    slots = 1 + max(sequences)
    tokens, total = torch.zeros(slots, width, dtype=torch.long), torch.zeros(slots, dtype=torch.long)
    prompt, prefill = np.zeros(slots, np.int64), np.zeros(slots, np.int64)
    for slot, (ids, p, f) in sequences.items():
        tokens[slot, :len(ids)] = torch.tensor(ids)
        prompt[slot], prefill[slot], total[slot] = p, f, len(ids)
    return SimpleNamespace(all_token_ids=SimpleNamespace(gpu=tokens), prompt_len=SimpleNamespace(np=prompt),
                           prefill_len=SimpleNamespace(np=prefill), total_len=SimpleNamespace(gpu=total))


def context(slots: list[int], seq_lens: list[int], inputs: list[int]) -> SimpleNamespace:
    return SimpleNamespace(idx_mapping_np=np.array(slots), seq_lens_upper_bound_np=np.array(seq_lens),
                           expanded_idx_mapping=torch.tensor(slots), input_ids=torch.tensor(inputs))


def test_loop_stop_forces_bos_at_a_check(tmp_path, monkeypatch):
    monkeypatch.setenv(LOG, str(tmp_path / "stops.jsonl"))
    prompt, ids, prose, n = [BOS, 5, 6, 7], loop_ids(), prose_ids(), REPEAT_TAIL
    # slot 2 loops, its last token this step's input (not yet committed); slot 0 is prose; slot 1 loops but this is a
    # chunk inside its refill; slot 3 loops one token past a check
    states = request_states({2: (prompt + ids[:n - 1], 4, 4), 0: (prompt + prose[:n], 4, 4),
                             1: (prompt + ids[:n], 4, 4 + n + 8), 3: (prompt + ids[:n + 1], 4, 4)}, width=n + 16)
    processor = LoopStopLogitsProcessor(None, states)
    logits = torch.randn(4, V)
    before = logits.clone()
    processor.apply(logits, context([2, 0, 1, 3], [4 + n, 4 + n, 4 + n, 4 + n + 1], [ids[n - 1], prose[n - 1], 0, 0]))
    assert torch.equal(logits[1:], before[1:])
    assert logits[0, BOS] == 0 and torch.isinf(logits[0]).sum() == V - 1
    lines = (tmp_path / "stops.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == [{"tokens": n, "repetition": 1.0}]


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
    x = (period * 40 + rng.integers(0, 12, 30).tolist())[:rng.integers(2, 300)]  # a loop, then noise
    prompt, output = x[:len(x) // 3], x[len(x) // 3:]
    got = Dry(breaker, [99], **DEFAULTS)(prompt, output, torch.zeros(12, dtype=torch.float64))
    expected = np.zeros(12)
    for v, p in dry_reference(x, breaker, 0.8, 1.75, 2).items():
        expected[v] = -p
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-12)


def test_dry_range_and_close():
    breaker, x = np.zeros(12, bool), [1, 2, 3, 1, 2, 3, 1, 2]
    # the whole context: "1 2 3 1 2" recurs before the suffix, so 3 is penalized at n 5
    got = Dry(breaker, [10, 11], **DEFAULTS)(x[:2], x[2:], torch.zeros(12, dtype=torch.float64))
    assert got[3] == -0.8 * 1.75**3 and torch.count_nonzero(got) == 1
    # the last 5 tokens, "1 2 3 1 2": only "1 2" recurs, n 2
    got = Dry(breaker, [10, 11], **(DEFAULTS | {"range": 5}))(x[:2], x[2:], torch.zeros(12, dtype=torch.float64))
    assert got[3] == -0.8 and torch.count_nonzero(got) == 1
    for prompt, output in (([10, 11, *x[:2]], x[2:]), (x[:2], [*x[2:4], 10, 11, *x[4:]])):  # closed: no penalty
        got = Dry(breaker, [10, 11], **DEFAULTS)(prompt, output, torch.zeros(12, dtype=torch.float64))
        assert torch.count_nonzero(got) == 0


def test_dry_processor_mirrors_the_output():
    """Step by step (one token a step, then gaps read from the history; a preemption refilling 49 tokens; a chunk inside
    the refill, skipped): the mirrored output and the penalties equal Dry's on the full sequence; a request without
    "dry" is untouched."""
    ids = TOK.enc.encode_ordinary(LOOP_TAIL * 3)
    prompt, width = ids[:20], len(ids) + 4
    processor = DryLogitsProcessor(None, request_states({0: (prompt, 20, 20), 1: (prompt, 20, 20)}, width))
    assert processor.add_request(0, SimpleNamespace(extra_args={"dry": {}}))
    assert not processor.add_request(1, SimpleNamespace(extra_args=None))
    for k in [*range(40), *range(40, len(ids) - 20, 7)]:  # k generated tokens, the last this step's input
        committed = prompt + ids[20:20 + k - 1] if k else prompt
        refill = 20 + 49 if k >= 50 else 20
        processor.req_states = request_states({0: (committed, 20, refill), 1: (committed, 20, 20)}, width)
        last = ids[20 + k - 1] if k else 0
        if k == 47:  # a chunk inside the refill: not sampled, left alone
            processor.req_states.prefill_len.np[0] = 20 + 60
        logits = torch.zeros(2, V)
        processor.apply(logits, context([0, 1], [20 + k, 20 + k], [last, last]))
        if k == 47:
            assert torch.count_nonzero(logits) == 0
            continue
        expected = Dry(processor.breaker, processor.end, **DEFAULTS)(prompt, ids[20:20 + k], torch.zeros(V))
        assert processor.requests[0][2] == ids[20:20 + k]
        assert torch.equal(logits[0], expected) and torch.count_nonzero(logits[1]) == 0
