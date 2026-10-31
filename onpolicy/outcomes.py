"""
The KodCode val evaluation's outcomes (docs/post-training.md §5) for one rollouts directory (onpolicy/rollouts.py,
with its verdicts, onpolicy/verify.py), written to <dir>/outcomes.json: pooled, per temperature and per source (the
prompt's `source`), the outcomes, the loop rates per response and per prompt, the exact-cycle loops' onsets and
periods, and over the prompts whose responses are all verified pass@1 and pass@n (unbiased, arXiv 2107.03374 Eq. 1),
the shares of prompts with a success and with mixed outcomes (a success and a loop) and the successes' lengths. A
response is a success when it closed its reasoning, ended with BOS and its answer passed the prompt's tests; a loop
when it never closed and its tail repeats (onpolicy/rollouts.py).

    python -m onpolicy.outcomes <rollouts dir>
"""
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

OUTCOMES = ("closed_ended", "closed_limit", "unclosed_limit", "unclosed_loop_stopped", "unclosed_ended")


def load(directory: Path) -> tuple[dict[str, dict], list[dict]]:
    """The sampled prompts by id and the rollouts, each with its `verdict` and `success` (None when the directory has
    no verdicts.jsonl)."""
    prompts = {r["id"]: r for r in map(json.loads, (directory / "prompts.jsonl").read_text().splitlines())}
    rows = pq.read_table(directory / "rollouts.parquet").to_pylist()
    path = directory / "verdicts.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    verdicts = {(v["prompt_id"], v["sample"]): v["verdict"] for v in map(json.loads, lines)}
    for r in rows:
        r["verdict"] = verdicts.get((r["prompt_id"], r["sample"]))
        r["success"] = None if r["verdict"] is None else r["closed"] and r["ended"] and r["verdict"] == "pass"
    return prompts, rows


def pass_at(n: int, c: int, k: int) -> float:
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def quantiles(values: list[float]) -> dict | None:
    return ({"n": len(values), "p10_p50_p90": np.percentile(values, [10, 50, 90]).tolist(), "max": max(values)}
            if values else None)


def summary(rows: list[dict]) -> dict:
    """Outcomes and rates over `rows`; the verdicts' statistics over its verified prompts."""
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_id"]].append(r)
    loops = [r for r in rows if r["loop"]]
    exact = [r for r in loops if r["loop_period"] is not None]
    out = {
        "responses": len(rows), "prompts": len(by_prompt),
        "samples_per_prompt": sorted({len(rs) for rs in by_prompt.values()}),
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


def outcomes(directory: Path) -> dict:
    prompts, rows = load(directory)
    source = {i: p["source"] for i, p in prompts.items()}
    report = {"all": summary(rows),
              "by_temperature": {str(t): summary([r for r in rows if r["temperature"] == t])
                                 for t in sorted({r["temperature"] for r in rows})},
              "by_source": {s: summary([r for r in rows if source[r["prompt_id"]] == s])
                            for s in sorted(set(source.values()))}}
    (directory / "outcomes.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(outcomes(Path(sys.argv[1])), indent=1))
