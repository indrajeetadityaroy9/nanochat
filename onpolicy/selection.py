"""
The rollouts' outcomes, the FTPO rows drawn from their loops and the DPO pairs drawn from their verified prompts
(docs/post-training.md §8), over one or several rollout directories (the waves of scripts/onpolicy.sh), written to
<out dir>. A response is a success when it closed its reasoning, ended with BOS and its answer passed the prompt's
tests (onpolicy/verify.py verdicts.jsonl, where a directory has one: rows need no verdicts); a loop when it never
closed and its tail repeats (onpolicy/rollouts.py `loop`).

- FTPO rows (ftpo.jsonl; Antidoom's unit, final-token preference of arXiv 2510.15061 §5): one per loop with an exact
  cycle (onpolicy/rollouts.py `cycle`: onset o, period p), at Antidoom's target (`target`): the first token of the
  cycle's first repeat, s + p, s the cycle's start slid left from o as far as each token equals the one p later once
  leading whitespace is stripped (github.com/Liquid4All/antidoom src/antidoom/generate.py:211-246: " So" and "So" are
  one key), moved past punctuation and whitespace and past a sentence end before a restart word as Antidoom does
  (generate.py:252-288): prefix_ids the prompt's ids and the response's ids[:target], rejected_id ids[target],
  `target` its offset from o + p (negative when the slide passed o). The chosen tokens are not chosen here: the
  trainer takes them from the reference model's distribution at that position (onpolicy/train.py). No success is
  required (a loop of a prompt never solved yields a row). Loops with no target before the response's end are counted
  and dropped; rows with an identical prefix are kept once (the first in prompt and sample order); rows whose prefix
  and rejected token exceed --max-length are dropped (counted). The rows are then rebalanced on their rejected token
  and source as Antidoom's trainer does (`rebalance`).
- DPO pairs (dpo.jsonl, `dpo_pairs`): outcome pairs, correct against incorrect rollouts of the same prompt
  (Phi-4-mini-reasoning's rollout preference learning, arXiv 2504.21233 §3.3, without its filter for incorrect
  responses with minor differences); chosen and rejected are separate samples with no shared history, so the pairs
  target correctness, while loop entry is FTPO's (DeLoop's shared-history contrast, arXiv 2610.19376 §4.1, at token
  level). Per verified prompt with a success and a failure, up to PAIRS_PER_PROMPT pairs: chosen the prompt's
  shortest success (fewest tokens; the lower sample index on a tie) through its closing BOS; rejected its failures,
  exact-cycle loops first (in sample order), then the other failures (in sample order). A loop's rejected side ends
  at its first repeat, ids[:onset + period + min(period, WINDOW_CAP)], with no BOS (WINDOW_CAP bounds a long period's
  share of its tokens), so it is not a stop that the pair lowers; its gradient falls almost entirely on the pre-cycle
  prefix, the repeat's tokens being near-certain. Another failure's rejected side is its whole response. Only
  verified prompts have successes, so unverified rollouts (the mix waves) yield no pairs. Pairs whose prompt and
  longer side exceed --max-length are dropped (counted).
- selection.json: per temperature and pooled, the outcomes, loop rates per response and per prompt, loop onsets and
  periods, and over the prompts whose responses are verified pass@1 and pass@n (unbiased, arXiv 2107.03374 Eq. 1),
  the share of prompts with a success and with mixed outcomes (a success and a loop) and the successes' lengths; per
  source (the prompt's `source`, onpolicy/rollouts.py); and the rows:
  loops with an exact cycle, duplicates, rows before and after rebalancing (and per source), the 20 most frequent
  rejected tokens (decoded) before and after, the rows of verified prompts never solved, and position, onset and
  period statistics; the pairs: their count, prompts, the dropped, and chosen, rejected and pair lengths.

The loops' targets are computed in a process per core of the CPU set (`loop_target`).

    python -m onpolicy.selection <out dir> [<rollouts dir> ...] [--max-length 32768]
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from math import comb
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from corpus.parallel import SPAWN
from corpus.tokenizer import get_tokenizer

MAX_LENGTH = 32768
# DPO pairs per prompt and the rejected side's cap past the first repeat
PAIRS_PER_PROMPT, WINDOW_CAP = 2, 256
# Antidoom's rebalancing (github.com/Liquid4All/antidoom @ bd6a126): rejected_regularisation_strength and
# max_train_examples of configs/default.yaml:91, :87, and its README's "set max_train_examples to at most 70% of the
# actual number of preference rows"; its seed (default.yaml:86)
STRENGTH, MAX_ROWS, KEEP, SEED = 0.3, 12000, 0.7, 3407
OUTCOMES = ("closed_ended", "closed_limit", "unclosed_limit", "unclosed_loop_stopped", "unclosed_ended")
# Antidoom's target adjustment (src/antidoom/generate.py @ bd6a126): its _RESTART_WORDS (:37-65)
RESTART_WORDS = frozenset({
    "actually", "after", "also", "alternatively", "because", "but", "finally", "first", "given", "hmm", "however", "in",
    "let", "looking", "maybe", "now", "okay", "perhaps", "second", "since", "so", "the", "then", "therefore", "this",
    "thus", "wait"})


def load(directories: list[Path]) -> tuple[dict[str, dict], list[dict]]:
    """The sampled prompts by id and the rollouts of every directory; a directory's verdicts (verdicts.jsonl, when
    written) give its responses' `verdict` and `success`, else both are None."""
    prompts, rows = {}, []
    for directory in directories:
        prompts |= {r["id"]: r for r in map(json.loads, (directory / "prompts.jsonl").read_text().splitlines())}
        part = pq.read_table(directory / "rollouts.parquet").to_pylist()
        path = directory / "verdicts.jsonl"
        lines = path.read_text().splitlines() if path.exists() else []
        verdicts = {(v["prompt_id"], v["sample"]): v["verdict"] for v in map(json.loads, lines)}
        for r in part:
            r["verdict"] = verdicts.get((r["prompt_id"], r["sample"]))
            r["success"] = None if r["verdict"] is None else r["closed"] and r["ended"] and r["verdict"] == "pass"
        rows += part
    return prompts, rows


def pass_at(n: int, c: int, k: int) -> float:
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def quantiles(values: list[float]) -> dict | None:
    return ({"n": len(values), "p10_p50_p90": np.percentile(values, [10, 50, 90]).tolist(), "max": max(values)}
            if values else None)


def summary(rows: list[dict]) -> dict:
    """Outcomes and rates over `rows` (one temperature, or pooled); the verdicts' statistics over its verified rows."""
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_id"]].append(r)
    loops = [r for r in rows if r["loop"]]
    exact = [r for r in loops if r["loop_period"] is not None]
    out = {
        "responses": len(rows), "prompts": len(by_prompt), "samples_per_prompt": sorted({len(rs) for rs in
                                                                                          by_prompt.values()}),
        "outcomes": {o: sum(r["outcome"] == o for r in rows) for o in OUTCOMES},
        "loop_rate_responses": len(loops) / len(rows),
        "loop_rate_prompts": sum(any(r["loop"] for r in rs) for rs in by_prompt.values()) / len(by_prompt),
        "loops_exact_cycle": len(exact),
        "loop_onset_tokens": quantiles([r["loop_onset"] for r in exact]),
        "loop_period_tokens": quantiles([r["loop_period"] for r in exact])}
    verified = {p: rs for p, rs in by_prompt.items() if all(r["verdict"] is not None for r in rs)}
    if not verified:
        return out
    vrows = [r for rs in verified.values() for r in rs]
    n = min(len(rs) for rs in verified.values())
    successes = [r for r in vrows if r["success"]]
    return out | {
        "verified_responses": len(vrows),
        "verdicts_closed_ended": {v: sum(r["verdict"] == v for r in vrows if r["closed"] and r["ended"])
                                  for v in ("pass", "pass_trivial", "fail", "timeout", "no_code")},
        "pass@1": len(successes) / len(vrows),
        f"pass@{n}": float(np.mean([pass_at(len(rs), sum(r["success"] for r in rs), n) for rs in verified.values()])),
        "prompts_solved": sum(any(r["success"] for r in rs) for rs in verified.values()) / len(verified),
        "prompts_mixed": sum(any(r["success"] for r in rs) and any(r["loop"] for r in rs)
                             for rs in verified.values()) / len(verified),
        "success_tokens": quantiles([r["tokens"] for r in successes])}


def ratios(counts: Counter, strength: float) -> dict[str, float]:
    """Antidoom's target shares per rejected token (src/antidoom/ftpo_data.py:105-114 `_regularised_ratios`): a count
    above the median is weighted by (median / count) ** strength, the others by 1; shares of the weighted counts."""
    median = float(np.median(list(counts.values())))
    weights = {t: 1.0 if c <= median or strength <= 0 else (median / c) ** strength for t, c in counts.items()}
    total = sum(weights[t] * c for t, c in counts.items())
    return {t: weights[t] * c / total for t, c in counts.items()}


def rebalance(rows: list[dict], target: int, strength: float, seed: int) -> list[dict]:
    """Antidoom's rejected-token normalisation (src/antidoom/ftpo_data.py:141-214 `_rejected_normalisation`), each
    row's source its `source`: each (rejected token, source) bucket shuffled; rows taken one at a time from the bucket
    whose token is furthest below its target share, a tie broken by the source furthest below its share of the rows
    (the first such bucket on a tie), while the token is under round(share * target); then, if short of `target`,
    without that cap; the selection shuffled."""
    if not rows:
        return []
    share = ratios(Counter(r["rejected"] for r in rows), strength)
    caps = {t: int(round(s * target)) for t, s in share.items()}
    pool = {s: c / len(rows) for s, c in Counter(r["source"] for r in rows).items()}
    buckets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        buckets[r["rejected"]][r["source"]].append(r)
    rng = np.random.default_rng(seed)
    for by_source in buckets.values():
        for bucket in by_source.values():
            rng.shuffle(bucket)
    selected, seen, taken = [], Counter(), Counter()
    for capped in (True, False):
        while len(selected) < target:
            best, best_key, denominator = None, None, max(len(selected), 1)
            for t, by_source in buckets.items():
                if capped and seen[t] >= caps[t]:
                    continue
                gap = share[t] - seen[t] / denominator
                for s, bucket in by_source.items():
                    key = (gap, pool[s] - taken[s] / denominator)
                    if bucket and (best_key is None or key > best_key):
                        best, best_key = (t, s), key
            if best is None:
                break
            t, s = best
            selected.append(buckets[t][s].pop())
            seen[t] += 1
            taken[s] += 1
    rng.shuffle(selected)
    return selected


def boundary_only(text: str) -> bool:
    """Antidoom's `_is_boundary_only` (generate.py:185-186): non-empty, no letter, digit or underscore."""
    return bool(text) and not any(ch.isalnum() or ch == "_" for ch in text)


def skip_boundary(text: Callable[[int], str], n: int, i: int, max_tokens: int = 4, max_chars: int = 12) -> int:
    """Antidoom's `_skip_boundary_only_tokens` (generate.py:189-204): past at most max_tokens boundary-only tokens
    whose texts total at most max_chars characters."""
    start, chars = i, 0
    while i < n and i - start < max_tokens:
        chars += len(text(i))
        if chars > max_chars or not boundary_only(text(i)):
            break
        i += 1
    return i


def normalised_word(text: str) -> str:
    """Antidoom's `_normalised_word` (generate.py:207-208)."""
    return re.sub(r"^[^\w_]+|[^\w_]+$", "", text.strip().lower())


def target(text: Callable[[int], str], ids: list[int], onset: int, period: int) -> int | None:
    """Antidoom's rejected index (`select_rejected_token_index_token_aware`, generate.py:232-288) from the exact cycle's
    onset o and period p, `text(i)` the text of token i: the seed s slid left from o while the token before it equals
    the one p later with leading whitespace stripped (`_slide_token_period_left`, `_lstrip_token_key_sequence`); s + p;
    past boundary-only tokens (at most 4, 12 characters); when that token is a word followed by at most 3
    boundary-only tokens (8 characters) holding a sentence end ".!?" and then a restart word, and the tokens before the
    cycle and before its repeat differ, the restart word; then the first token that is not whitespace. None past the
    response's end."""
    n, seed = len(ids), onset
    while seed > 0 and seed - 1 + period < n and text(seed - 1).lstrip() == text(seed - 1 + period).lstrip():
        seed -= 1
    i = skip_boundary(text, n, seed + period)
    if i >= n:
        return None
    word = normalised_word(text(i))
    if word and any(ch.isalnum() or ch == "_" for ch in word):
        punct_start = i + 1
        punct_end = skip_boundary(text, n, punct_start, max_tokens=3, max_chars=8)
        punct = "".join(text(j) for j in range(punct_start, punct_end))
        next_word = normalised_word(text(punct_end)) if punct_end < n else ""
        seed_prev, repeat_prev = seed - 1, seed + period - 1
        if (punct_end > punct_start and any(ch in punct for ch in ".!?") and seed_prev >= 0
                and ids[seed_prev] != ids[repeat_prev] and next_word in RESTART_WORDS):
            i = punct_end
    while i < n:
        if text(i).strip():
            assert text(i).lstrip() == text(i - period).lstrip(), "target left the repeat"
            return i
        i += 1
    return None


def loop_target(loop: tuple[list[int], int, int]) -> int | None:
    """`target` of a loop (ids, onset, period), its tokens' text by the tokenizer."""
    ids, onset, period = loop
    decode = get_tokenizer().enc.decode
    return target(lambda i: decode([ids[i]]), ids, onset, period)


def exact(r: dict) -> bool:
    """A loop whose tail is an exact cycle (onpolicy/rollouts.py `cycle`)."""
    return r["loop"] and r["loop_period"] is not None


def ftpo_rows(prompts: dict[str, dict], rows: list[dict], max_length: int) -> tuple[list[dict], dict]:
    """One row per loop with an exact cycle at its `target`, duplicates and over-long rows removed (in prompt and
    sample order)."""
    out, seen, counts = [], set(), Counter()
    loops = [r for r in sorted(rows, key=lambda r: (r["prompt_id"], r["sample"])) if exact(r)]
    with ProcessPoolExecutor(len(os.sched_getaffinity(0)), mp_context=SPAWN) as pool:
        positions = list(pool.map(loop_target, [(r["ids"], r["loop_onset"], r["loop_period"]) for r in loops],
                                  chunksize=16))
    for r, position in zip(loops, positions, strict=True):
        counts["loops_exact_cycle"] += 1
        ids, onset, period = r["ids"], r["loop_onset"], r["loop_period"]
        if position is None:
            counts["no_target"] += 1
            continue
        prefix = prompts[r["prompt_id"]]["ids"] + ids[:position]
        if tuple(prefix) in seen:
            counts["duplicates"] += 1
            continue
        seen.add(tuple(prefix))
        if len(prefix) + 1 > max_length:
            counts["dropped_too_long"] += 1
            continue
        out.append({"prompt_id": r["prompt_id"], "sample": r["sample"], "temperature": r["temperature"],
                    "onset": onset, "period": period, "target": position - onset - period, "prefix_ids": prefix,
                    "rejected_id": ids[position]})
    return out, dict(counts)


def rejected_side(r: dict) -> list[int]:
    """A loop through its first repeat, the repeat capped at WINDOW_CAP tokens."""
    return r["ids"][:r["loop_onset"] + r["loop_period"] + min(r["loop_period"], WINDOW_CAP)]


def dpo_pairs(prompts: dict[str, dict], rows: list[dict], max_length: int, bos: int) -> tuple[list[dict], int]:
    """The DPO pairs (module docstring) in prompt order and the number dropped for length."""
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_id"]].append(r)
    pairs, too_long = [], 0
    for pid, rs in sorted(by_prompt.items()):
        wins = sorted((r for r in rs if r["success"]), key=lambda r: (r["tokens"], r["sample"]))
        if not wins:
            continue
        best, context = wins[0], prompts[pid]["ids"]
        chosen = best["ids"] if best["ids"][-1:] == [bos] else [*best["ids"], bos]
        failures = sorted((r for r in rs if r["success"] is False), key=lambda r: (not exact(r), r["sample"]))
        for fail in failures[:PAIRS_PER_PROMPT]:
            side = rejected_side(fail) if exact(fail) else fail["ids"]
            if len(context) + max(len(chosen), len(side)) > max_length:
                too_long += 1
                continue
            pairs.append({"prompt_id": pid, "chosen_sample": best["sample"], "rejected_sample": fail["sample"],
                          "loop_onset": fail["loop_onset"] if exact(fail) else None,
                          "loop_period": fail["loop_period"] if exact(fail) else None,
                          "prompt_ids": context, "chosen_ids": chosen, "rejected_ids": side})
    return pairs, too_long


def select(out: Path, directories: list[Path], max_length: int) -> dict:
    prompts, rows = load(directories)
    tok = get_tokenizer()
    decode = tok.enc.decode
    source = {i: p["source"] for i, p in prompts.items()}
    candidates, counts = ftpo_rows(prompts, rows, max_length)
    for c in candidates:
        c["rejected"] = decode([c["rejected_id"]])  # Antidoom balances on the decoded token (`rejected_decoded`)
        c["source"] = source[c["prompt_id"]]
    kept = rebalance(candidates, min(MAX_ROWS, int(KEEP * len(candidates))), STRENGTH, SEED)
    with open(out / "ftpo.jsonl", "w") as f:
        f.writelines(json.dumps({k: v for k, v in c.items() if k not in ("rejected", "source")}) + "\n" for c in kept)
    pairs, too_long = dpo_pairs(prompts, rows, max_length, tok.bos_token_id)
    with open(out / "dpo.jsonl", "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in pairs)
    verified = {r["prompt_id"] for r in rows if r["verdict"] is not None}
    solved = {r["prompt_id"] for r in rows if r["success"]}
    temperatures = sorted({r["temperature"] for r in rows})
    sources = sorted(set(source.values()))
    report = {
        "all": summary(rows),
        "by_temperature": {str(t): summary([r for r in rows if r["temperature"] == t]) for t in temperatures},
        "by_source": {s: summary([r for r in rows if source[r["prompt_id"]] == s]) for s in sources},
        "ftpo": counts | {
            "max_length": max_length, "strength": STRENGTH, "rows_before_rebalance": len(candidates),
            "rows": len(kept),
            "rows_unsolved_prompts": sum(c["prompt_id"] in verified - solved for c in kept) if verified else None,
            "rows_by_temperature": {str(t): sum(c["temperature"] == t for c in kept) for t in temperatures},
            "rows_before_rebalance_by_source": dict(Counter(c["source"] for c in candidates)),
            "rows_by_source": dict(Counter(c["source"] for c in kept)),
            "rejected_top20_before": Counter(c["rejected"] for c in candidates).most_common(20),
            "rejected_top20": Counter(c["rejected"] for c in kept).most_common(20),
            "position_tokens": quantiles([c["onset"] + c["period"] + c["target"] for c in kept]),
            "rows_target_moved": sum(c["target"] != 0 for c in kept),
            "onset_tokens": quantiles([c["onset"] for c in kept]),
            "period_tokens": quantiles([c["period"] for c in kept]),
            "prefix_tokens": quantiles([len(c["prefix_ids"]) for c in kept])},
        "dpo": {"pairs": len(pairs), "prompts": len({p["prompt_id"] for p in pairs}), "dropped_too_long": too_long,
                "max_length": max_length, "per_prompt_cap": PAIRS_PER_PROMPT, "window_cap": WINDOW_CAP,
                "chosen_tokens": quantiles([len(p["chosen_ids"]) for p in pairs]),
                "rejected_tokens": quantiles([len(p["rejected_ids"]) for p in pairs]),
                "pair_length": quantiles([len(p["prompt_ids"]) + max(len(p["chosen_ids"]), len(p["rejected_ids"]))
                                          for p in pairs])}}
    (out / "selection.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path, help="where ftpo.jsonl, dpo.jsonl and selection.json go")
    parser.add_argument("rollouts", type=Path, nargs="*", help="rollout directories (default: out)")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()
    print(json.dumps(select(args.out, args.rollouts or [args.out], args.max_length), indent=1))
