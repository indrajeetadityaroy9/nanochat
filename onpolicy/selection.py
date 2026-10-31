"""
The rollouts' outcomes and the training sets drawn from them (docs/post-training.md §8). A response is a success when
it closed its reasoning, ended with BOS and its answer passed the prompt's tests (onpolicy/verify.py verdicts.jsonl);
a loop when it never closed and its tail repeats (onpolicy/rollouts.py `loop`).

- RSFT set (rsft.jsonl): per prompt with a success, its shortest success (fewest tokens; the lower sample index on a
  tie): the prompt's ids and the response's ids through its closing BOS.
- Preference pairs (pairs.jsonl): per prompt with a success and a loop whose cycle is exact (rollouts.py `cycle`), up
  to PAIRS_PER_PROMPT pairs, one per such loop in sample order: chosen the prompt's shortest success, rejected the
  loop through its first repeat, ids[:onset + period + min(period, WINDOW_CAP)]. The rejected side ends where the
  model has re-entered the cycle once (the first looping token is Antidoom's rejected token; DeLoop's rejected action
  is the first repeat of an earlier action, arXiv 2610.19376 §4.1), with no BOS, so the pair moves probability away
  from re-entering the cycle and not from a stop; WINDOW_CAP bounds a long period's share of the rejected tokens.
  Pairs whose prompt and longer side exceed --max-length are dropped (counted).
- selection.json: per temperature and pooled, the outcomes, pass@1 and pass@n (unbiased, arXiv 2107.03374 Eq. 1),
  loop rates per response and per prompt, the share of prompts with mixed outcomes (a success and a loop) and with a
  success at all, the length statistics of successes, of shortest successes, of loop onsets and periods and of the
  rejected sides; per style; and the sets' sizes.

    python -m onpolicy.selection <rollouts dir> [--max-length 16384]
"""
import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from corpus.tokenizer import get_tokenizer

PAIRS_PER_PROMPT, WINDOW_CAP, MAX_LENGTH = 2, 256, 16384
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


def rejected(r: dict) -> list[int]:
    return r["ids"][:r["loop_onset"] + r["loop_period"] + min(r["loop_period"], WINDOW_CAP)]


def summary(rows: list[dict]) -> dict:
    """Outcomes and rates over `rows` (one temperature, or pooled)."""
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_id"]].append(r)
    sizes = {len(rs) for rs in by_prompt.values()}
    n = min(sizes)
    successes = [r for r in rows if r["success"]]
    loops = [r for r in rows if r["loop"]]
    shortest = [min(r["tokens"] for r in rs if r["success"]) for rs in by_prompt.values()
                if any(x["success"] for x in rs)]
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
        "shortest_success_tokens": quantiles(shortest),
        "loops_exact_cycle": len(exact),
        "loop_onset_tokens": quantiles([r["loop_onset"] for r in exact]),
        "loop_period_tokens": quantiles([r["loop_period"] for r in exact]),
        "rejected_tokens": quantiles([len(rejected(r)) for r in exact])}


def select(directory: Path, max_length: int) -> dict:
    prompts, rows = load(directory)
    bos = get_tokenizer().bos_token_id
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_id"]].append(r)
    rsft, pairs, too_long, yield_by = [], [], 0, defaultdict(int)
    for pid, rs in sorted(by_prompt.items()):
        wins = sorted((r for r in rs if r["success"]), key=lambda r: (r["tokens"], r["sample"]))
        if not wins:
            continue
        best, context = wins[0], prompts[pid]["ids"]
        chosen = best["ids"] if best["ids"][-1:] == [bos] else [*best["ids"], bos]
        rsft.append({"prompt_id": pid, "sample": best["sample"], "temperature": best["temperature"],
                     "prompt_ids": context, "completion_ids": chosen})
        for loop in [r for r in rs if r["loop"] and r["loop_period"] is not None][:PAIRS_PER_PROMPT]:
            side = rejected(loop)
            if len(context) + max(len(chosen), len(side)) > max_length:
                too_long += 1
                continue
            yield_by[f"chosen_T{best['temperature']}_rejected_T{loop['temperature']}"] += 1
            pairs.append({"prompt_id": pid, "chosen_sample": best["sample"], "rejected_sample": loop["sample"],
                          "chosen_temperature": best["temperature"], "rejected_temperature": loop["temperature"],
                          "loop_onset": loop["loop_onset"], "loop_period": loop["loop_period"],
                          "prompt_ids": context, "chosen_ids": chosen, "rejected_ids": side})
    for name, items in (("rsft", rsft), ("pairs", pairs)):
        with open(directory / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(item) + "\n" for item in items)
    temperatures = sorted({r["temperature"] for r in rows})
    styles = sorted({p["style"] for p in prompts.values()})
    report = {
        "all": summary(rows),
        "by_temperature": {str(t): summary([r for r in rows if r["temperature"] == t]) for t in temperatures},
        "by_style": {s: summary([r for r in rows if prompts[r["prompt_id"]]["style"] == s]) for s in styles},
        "rsft": {"examples": len(rsft), "tokens": quantiles([len(x["completion_ids"]) for x in rsft]),
                 "by_temperature": {str(t): sum(x["temperature"] == t for x in rsft) for t in temperatures}},
        "pairs": {"pairs": len(pairs), "prompts": len({p["prompt_id"] for p in pairs}), "dropped_too_long": too_long,
                  "max_length": max_length, "per_prompt_cap": PAIRS_PER_PROMPT, "window_cap": WINDOW_CAP,
                  "by_temperatures": dict(sorted(yield_by.items())),
                  "chosen_tokens": quantiles([len(p["chosen_ids"]) for p in pairs]),
                  "rejected_tokens": quantiles([len(p["rejected_ids"]) for p in pairs]),
                  "pair_length": quantiles([len(p["prompt_ids"]) + max(len(p["chosen_ids"]), len(p["rejected_ids"]))
                                            for p in pairs])}}
    (directory / "selection.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()
    print(json.dumps(select(args.rollouts, args.max_length), indent=1))
