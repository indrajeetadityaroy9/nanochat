"""
The rollouts' outcomes and the FTPO rows drawn from their loops (docs/post-training.md §8). A response is a success
when it closed its reasoning, ended with BOS and its answer passed the prompt's tests (onpolicy/verify.py
verdicts.jsonl); a loop when it never closed and its tail repeats (onpolicy/rollouts.py `loop`).

- FTPO rows (ftpo.jsonl; Antidoom's unit, final-token preference of arXiv 2510.15061 §5): one per loop with an exact
  cycle (onpolicy/rollouts.py `cycle`: onset o, period p), at the first token of the cycle's first repeat, o + p
  (Antidoom's target: the cycle's start slid left as far as tokens equal the ones p later, plus one period,
  github.com/Liquid4All/antidoom src/antidoom/generate.py:246-248): prefix_ids the prompt's ids and the response's
  ids[:o + p], rejected_id ids[o + p] (= ids[o]). The chosen tokens are not chosen here: the trainer takes them from
  the reference model's distribution at that position (onpolicy/train.py). No success is required (a loop of a prompt
  never solved yields a row). Rows with an identical prefix are kept once (the first in prompt and sample order);
  rows whose prefix and rejected token exceed --max-length are dropped (counted). The rows are then rebalanced on
  their rejected token as Antidoom's trainer does (`rebalance`).
- selection.json: per temperature and pooled, the outcomes, pass@1 and pass@n (unbiased, arXiv 2107.03374 Eq. 1),
  loop rates per response and per prompt, the share of prompts with a success and with mixed outcomes (a success and
  a loop), the length statistics of successes, of loop onsets and periods; per style; and the rows: loops with an
  exact cycle, duplicates, rows before and after rebalancing, the 20 most frequent rejected tokens (decoded) before
  and after, the rows of prompts never solved, and position, onset and period statistics.

    python -m onpolicy.selection <rollouts dir> [--max-length 32768]
"""
import argparse
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from corpus.tokenizer import get_tokenizer

MAX_LENGTH = 32768
# Antidoom's rebalancing (github.com/Liquid4All/antidoom @ bd6a126): rejected_regularisation_strength and
# max_train_examples of configs/default.yaml:91, :87, and its README's "set max_train_examples to at most 70% of the
# actual number of preference rows"; its seed (default.yaml:86)
STRENGTH, MAX_ROWS, KEEP, SEED = 0.3, 12000, 0.7, 3407
OUTCOMES = ("closed_ended", "closed_limit", "unclosed_limit", "unclosed_loop_stopped", "unclosed_ended")


def load(directory: Path) -> tuple[dict[str, dict], list[dict]]:
    """The sampled prompts by id and the rollouts with their verdicts and `success`."""
    prompts = {r["id"]: r for r in map(json.loads, (directory / "prompts.jsonl").read_text().splitlines())}
    rows = pq.read_table(directory / "rollouts.parquet").to_pylist()
    verdicts = {(v["prompt_id"], v["sample"]): v["verdict"]
                for v in map(json.loads, (directory / "verdicts.jsonl").read_text().splitlines())}
    for r in rows:
        r["verdict"] = verdicts[r["prompt_id"], r["sample"]]
        r["success"] = r["closed"] and r["ended"] and r["verdict"] == "pass"
    return prompts, rows


def pass_at(n: int, c: int, k: int) -> float:
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def quantiles(values: list[float]) -> dict | None:
    return ({"n": len(values), "p10_p50_p90": np.percentile(values, [10, 50, 90]).tolist(), "max": max(values)}
            if values else None)


def summary(rows: list[dict]) -> dict:
    """Outcomes and rates over `rows` (one temperature, or pooled)."""
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_id"]].append(r)
    sizes = {len(rs) for rs in by_prompt.values()}
    n = min(sizes)
    successes = [r for r in rows if r["success"]]
    loops = [r for r in rows if r["loop"]]
    exact = [r for r in loops if r["loop_period"] is not None]
    return {
        "responses": len(rows), "prompts": len(by_prompt), "samples_per_prompt": sorted(sizes),
        "outcomes": {o: sum(r["outcome"] == o for r in rows) for o in OUTCOMES},
        "verdicts_closed_ended": {v: sum(r["verdict"] == v for r in rows if r["closed"] and r["ended"])
                                  for v in ("pass", "pass_trivial", "fail", "timeout", "no_code")},
        "pass@1": len(successes) / len(rows),
        f"pass@{n}": float(np.mean([pass_at(len(rs), sum(r["success"] for r in rs), n) for rs in by_prompt.values()])),
        "loop_rate_responses": len(loops) / len(rows),
        "loop_rate_prompts": sum(any(r["loop"] for r in rs) for rs in by_prompt.values()) / len(by_prompt),
        "prompts_solved": sum(any(r["success"] for r in rs) for rs in by_prompt.values()) / len(by_prompt),
        "prompts_mixed": sum(any(r["success"] for r in rs) and any(r["loop"] for r in rs)
                             for rs in by_prompt.values()) / len(by_prompt),
        "success_tokens": quantiles([r["tokens"] for r in successes]),
        "loops_exact_cycle": len(exact),
        "loop_onset_tokens": quantiles([r["loop_onset"] for r in exact]),
        "loop_period_tokens": quantiles([r["loop_period"] for r in exact])}


def ratios(counts: Counter, strength: float) -> dict[str, float]:
    """Antidoom's target shares per rejected token (src/antidoom/ftpo_data.py:105-114 `_regularised_ratios`): a count
    above the median is weighted by (median / count) ** strength, the others by 1; shares of the weighted counts."""
    median = float(np.median(list(counts.values())))
    weights = {t: 1.0 if c <= median or strength <= 0 else (median / c) ** strength for t, c in counts.items()}
    total = sum(weights[t] * c for t, c in counts.items())
    return {t: weights[t] * c / total for t, c in counts.items()}


def rebalance(rows: list[dict], target: int, strength: float, seed: int) -> list[dict]:
    """Antidoom's rejected-token normalisation (src/antidoom/ftpo_data.py:141-214 `_rejected_normalisation`, one
    source): each rejected token's bucket shuffled; rows taken one at a time from the token furthest below its target
    share (the first such token on a tie), while under round(share * target); then, if short of `target`, without that
    cap; the selection shuffled."""
    if not rows:
        return []
    share = ratios(Counter(r["rejected"] for r in rows), strength)
    caps = {t: int(round(s * target)) for t, s in share.items()}
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["rejected"]].append(r)
    rng = np.random.default_rng(seed)
    for t in buckets:
        rng.shuffle(buckets[t])
    selected, seen = [], defaultdict(int)
    for capped in (True, False):
        while len(selected) < target:
            best, best_gap = None, None
            for t, available in buckets.items():
                if not available or (capped and seen[t] >= caps.get(t, 0)):
                    continue
                gap = share.get(t, 0.0) - seen[t] / max(len(selected), 1)
                if best_gap is None or gap > best_gap:
                    best, best_gap = t, gap
            if best is None:
                break
            selected.append(buckets[best].pop())
            seen[best] += 1
    rng.shuffle(selected)
    return selected


def ftpo_rows(prompts: dict[str, dict], rows: list[dict], max_length: int) -> tuple[list[dict], dict]:
    """One row per loop with an exact cycle, duplicates and over-long rows removed (in prompt and sample order)."""
    out, seen, counts = [], set(), Counter()
    for r in sorted(rows, key=lambda r: (r["prompt_id"], r["sample"])):
        if not r["loop"] or r["loop_period"] is None:
            continue
        counts["loops_exact_cycle"] += 1
        position = r["loop_onset"] + r["loop_period"]
        prefix = prompts[r["prompt_id"]]["ids"] + r["ids"][:position]
        if tuple(prefix) in seen:
            counts["duplicates"] += 1
            continue
        seen.add(tuple(prefix))
        if len(prefix) + 1 > max_length:
            counts["dropped_too_long"] += 1
            continue
        out.append({"prompt_id": r["prompt_id"], "sample": r["sample"], "temperature": r["temperature"],
                    "onset": r["loop_onset"], "period": r["loop_period"], "prefix_ids": prefix,
                    "rejected_id": r["ids"][position]})
    return out, dict(counts)


def select(directory: Path, max_length: int) -> dict:
    prompts, rows = load(directory)
    decode = get_tokenizer().enc.decode
    candidates, counts = ftpo_rows(prompts, rows, max_length)
    for c in candidates:
        c["rejected"] = decode([c["rejected_id"]])  # Antidoom balances on the decoded token (`rejected_decoded`)
    kept = rebalance(candidates, min(MAX_ROWS, int(KEEP * len(candidates))), STRENGTH, SEED)
    with open(directory / "ftpo.jsonl", "w") as f:
        f.writelines(json.dumps({k: v for k, v in c.items() if k != "rejected"}) + "\n" for c in kept)
    solved = {r["prompt_id"] for r in rows if r["success"]}
    temperatures = sorted({r["temperature"] for r in rows})
    styles = sorted({p["style"] for p in prompts.values()})
    report = {
        "all": summary(rows),
        "by_temperature": {str(t): summary([r for r in rows if r["temperature"] == t]) for t in temperatures},
        "by_style": {s: summary([r for r in rows if prompts[r["prompt_id"]]["style"] == s]) for s in styles},
        "ftpo": counts | {
            "max_length": max_length, "strength": STRENGTH, "rows_before_rebalance": len(candidates),
            "rows": len(kept), "rows_unsolved_prompts": sum(c["prompt_id"] not in solved for c in kept),
            "rows_by_temperature": {str(t): sum(c["temperature"] == t for c in kept) for t in temperatures},
            "rejected_top20_before": Counter(c["rejected"] for c in candidates).most_common(20),
            "rejected_top20": Counter(c["rejected"] for c in kept).most_common(20),
            "position_tokens": quantiles([c["onset"] + c["period"] for c in kept]),
            "onset_tokens": quantiles([c["onset"] for c in kept]),
            "period_tokens": quantiles([c["period"] for c in kept]),
            "prefix_tokens": quantiles([len(c["prefix_ids"]) for c in kept])}}
    (directory / "selection.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()
    print(json.dumps(select(args.rollouts, args.max_length), indent=1))
