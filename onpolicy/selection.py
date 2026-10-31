"""
The rollouts' outcomes, the commit rows drawn from their loops and the DPO pairs drawn from their verified prompts
(docs/post-training.md §8), over one or several rollout directories (the waves of scripts/onpolicy.sh), written to
<out dir>. A response is a success when it closed its reasoning, ended with BOS and its answer passed the prompt's
tests (onpolicy/verify.py verdicts.jsonl, where a directory has one: rows need no verdicts); a loop when it never
closed and its tail repeats (onpolicy/rollouts.py `loop`).

- Commit rows (commit.jsonl, onpolicy/train.py `commit`): one per loop with an exact cycle (onpolicy/rollouts.py
  `cycle`: onset o, period p) whose ids hold its cut (`cut`: the first line start at or after o + COPIES p, after one
  verbatim repeat): prefix_ids the prompt's ids and the response's ids[:cut]. No success is required (a loop of a
  prompt never solved yields a row: committing there passes 14.1% of KodCode val loops against none for the loop).
  Rows with an identical prefix are kept once (the first in prompt and sample order); rows whose prefix and END_THINK
  exceed --max-length are dropped (counted).
- DPO pairs (dpo.jsonl, `dpo_pairs`), of two kinds, both scored on correctness:
  - loop pairs (DeLoop's unit, arXiv 2610.19376 §4.1: x the history before the looping action, y- the looping
    action, y+ the first later action of the same rollout that breaks the loop, no label needed). These loops never
    break within their rollout, so y+ is a commit instead (onpolicy/rollouts.py, the rows with `resume_of`: END_THINK
    forced after the cut, then the model's own answer), kept only when its answer passes the tests; and x runs through
    the cut, one verbatim repeat past the loop's start: per exact-cycle loop of a verified prompt with a successful
    commit, the prompt is the prompt and the loop through its cut; chosen the shortest successful commit from there
    through its closing BOS; rejected the loop's next repeat, min(period, WINDOW_CAP) tokens. The shared history puts
    the pair's gradient on the tokens where the loop and its break diverge.
  - outcome pairs (rollout DPO: correct against incorrect rollouts of the same prompt, Phi-4-mini-reasoning §3.3): per
    verified prompt with a success and a failure not already in a loop pair, up to PAIRS_PER_PROMPT pairs: chosen the
    prompt's shortest success through its closing BOS; rejected its failures, exact-cycle loops first, then the
    others (each in sample order); a loop's side ends at its first repeat (ids[:onset + period + min(period,
    WINDOW_CAP)], no BOS), another failure's side is its whole response.
  Only verified prompts have successes, so unverified rollouts (the mix waves) yield no pairs. Pairs whose prompt and
  longer side exceed --max-length are dropped (counted).
- selection.json: per temperature and pooled, the outcomes, loop rates per response and per prompt, loop onsets and
  periods, and over the prompts whose responses are verified pass@1 and pass@n (unbiased, arXiv 2107.03374 Eq. 1),
  the share of prompts with a success and with mixed outcomes (a success and a loop) and the successes' lengths; per
  source (the prompt's `source`, onpolicy/rollouts.py; a KodCode row without one: kodcode/<style>); the commit rows
  (per source, onset, period and prefix statistics, the dropped); the pairs: their count, prompts, the commits and
  their successes, the dropped, and chosen, rejected and pair lengths.

    python -m onpolicy.selection <out dir> [<rollouts dir> ...] [--max-length 32768]
"""
import argparse
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from corpus.posttrain.sft import END_THINK
from corpus.tokenizer import get_tokenizer

MAX_LENGTH = 32768
# DPO pairs per prompt and the rejected side's cap past the cut
PAIRS_PER_PROMPT, WINDOW_CAP = 2, 256
# the cut: after one verbatim repeat, the loop established before it is broken (Word Salad Chopper chops after two
# consecutive near-repeated reasoning chunks, arXiv 2511.00536 §3.2); committing at o + 2p passes 24.8% of 802 KodCode
# val loops against 25.6% at the first repeat's start. At a line start (`cut`): there the commit opens as the model's
# own ends of reasoning do, while a mid-line cut makes it break the line first, 17.5 of the commit's 20 nats, and the
# line breaks it learned spread to normal text (the guard's NLL +5.5% at step 40 against -0.6% at step 30)
COPIES = 2
OUTCOMES = ("closed_ended", "closed_limit", "unclosed_limit", "unclosed_loop_stopped", "unclosed_ended")


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


def newline_ids() -> np.ndarray:
    """The tokens whose text ends in a newline: a line starts after each."""
    enc = get_tokenizer().enc
    return np.array([i for i in range(enc.n_vocab) if enc.decode([i]).endswith("\n")])


def cut(ids: list[int], onset: int, period: int, newline: np.ndarray) -> int | None:
    """A loop's cut: the first line start c (ids[c - 1] one of `newline`) in [onset + COPIES period, + period) within
    the ids; None when that copy of the cycle holds none."""
    lo = onset + COPIES * period
    hi = min(lo + period, len(ids) + 1)
    starts = np.flatnonzero(np.isin(np.asarray(ids[lo - 1:hi - 1]), newline)) if lo <= len(ids) else []
    return lo + int(starts[0]) if len(starts) else None


def commit_rows(prompts: dict[str, dict], rows: list[dict], max_length: int,
                commit_tokens: int) -> tuple[list[dict], dict]:
    """One row per loop with an exact cycle whose ids hold its cut (`cut`), duplicates and over-long rows removed (in
    prompt and sample order)."""
    out, seen, counts, newline = [], set(), Counter(), newline_ids()
    for r in sorted(rows, key=lambda r: (r["prompt_id"], r["sample"])):
        if not (r["loop"] and r["loop_period"] is not None):
            continue
        counts["loops_exact_cycle"] += 1
        c = cut(r["ids"], r["loop_onset"], r["loop_period"], newline)
        if c is None:
            counts["no_line_start_in_cut_copy"] += 1
            continue
        prefix = prompts[r["prompt_id"]]["ids"] + r["ids"][:c]
        if tuple(prefix) in seen:
            counts["duplicates"] += 1
            continue
        seen.add(tuple(prefix))
        if len(prefix) + commit_tokens > max_length:
            counts["dropped_too_long"] += 1
            continue
        out.append({"prompt_id": r["prompt_id"], "sample": r["sample"], "temperature": r["temperature"],
                    "onset": r["loop_onset"], "period": r["loop_period"], "cut": c, "prefix_ids": prefix})
    return out, dict(counts)


def rejected_side(r: dict) -> list[int]:
    """A loop through its first repeat, the repeat capped at WINDOW_CAP tokens."""
    return r["ids"][:r["loop_onset"] + r["loop_period"] + min(r["loop_period"], WINDOW_CAP)]


def dpo_pairs(prompts: dict[str, dict], rows: list[dict], resumed: list[dict], max_length: int,
              bos: int) -> tuple[list[dict], int]:
    """The DPO pairs (module docstring) in prompt order and the number dropped for length."""
    by_prompt, wins_from = defaultdict(list), defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_id"]].append(r)
    for r in resumed:
        if r["success"]:
            wins_from[(r["prompt_id"], r["resume_of"])].append(r)
    pairs, too_long = [], 0

    def add(kind: str, pid: str, prompt: list[int], chosen: dict, chosen_ids: list[int], rejected: dict,
            rejected_ids: list[int]) -> None:
        nonlocal too_long
        if len(prompt) + max(len(chosen_ids), len(rejected_ids)) > max_length:
            too_long += 1
            return
        pairs.append({"kind": kind, "prompt_id": pid, "chosen_sample": chosen["sample"],
                      "rejected_sample": rejected["sample"], "loop_onset": rejected["loop_onset"],
                      "loop_period": rejected["loop_period"], "prompt_ids": prompt, "chosen_ids": chosen_ids,
                      "rejected_ids": rejected_ids})

    for pid, rs in sorted(by_prompt.items()):
        context = prompts[pid]["ids"]
        exact = sorted((r for r in rs if r["success"] is False and r["loop"] and r["loop_period"] is not None),
                       key=lambda r: r["sample"])
        broken = set()
        for loop in exact:
            wins = sorted(wins_from[(pid, loop["sample"])], key=lambda r: (r["tokens"], r["sample"]))
            if not wins:
                continue
            best = wins[0]
            at = best["resume_from"]  # the loop's cut, where its commits start
            if best["ids"][:at] != loop["ids"][:at]:
                raise ValueError(f"{pid} sample {best['sample']} does not continue sample {loop['sample']}'s history")
            chosen = best["ids"][at:] if best["ids"][-1:] == [bos] else [*best["ids"][at:], bos]
            add("loop", pid, context + loop["ids"][:at], best, chosen, loop,
                loop["ids"][at:at + min(loop["loop_period"], WINDOW_CAP)])
            broken.add(loop["sample"])
        wins = sorted((r for r in rs if r["success"]), key=lambda r: (r["tokens"], r["sample"]))
        if not wins:
            continue
        best = wins[0]
        chosen = best["ids"] if best["ids"][-1:] == [bos] else [*best["ids"], bos]
        other = sorted((r for r in rs if r["success"] is False and not (r["loop"] and r["loop_period"] is not None)),
                       key=lambda r: r["sample"])
        for fail in [*(r for r in exact if r["sample"] not in broken), *other][:PAIRS_PER_PROMPT]:
            add("outcome", pid, context, best, chosen, fail,
                rejected_side(fail) if fail["loop"] and fail["loop_period"] is not None else fail["ids"])
    return pairs, too_long


def select(out: Path, directories: list[Path], max_length: int) -> dict:
    prompts, rows = load(directories)
    resumed = [r for r in rows if r.get("resume_of") is not None]  # the loops' commits: pairs only
    rows = [r for r in rows if r.get("resume_of") is None]
    tok = get_tokenizer()
    source = {i: p.get("source") or f"kodcode/{p['style']}" for i, p in prompts.items()}
    kept, counts = commit_rows(prompts, rows, max_length, len(tok.enc.encode_ordinary(END_THINK)))
    with open(out / "commit.jsonl", "w") as f:
        f.writelines(json.dumps(c) + "\n" for c in kept)
    pairs, too_long = dpo_pairs(prompts, rows, resumed, max_length, tok.bos_token_id)
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
        "commit": counts | {
            "max_length": max_length, "copies": COPIES, "rows": len(kept),
            "rows_unsolved_prompts": sum(c["prompt_id"] in verified - solved for c in kept) if verified else None,
            "rows_by_temperature": {str(t): sum(c["temperature"] == t for c in kept) for t in temperatures},
            "rows_by_source": dict(Counter(source[c["prompt_id"]] for c in kept)),
            "onset_tokens": quantiles([c["onset"] for c in kept]),
            "period_tokens": quantiles([c["period"] for c in kept]),
            "cut_tokens": quantiles([c["cut"] for c in kept]),
            "prefix_tokens": quantiles([len(c["prefix_ids"]) for c in kept])},
        "dpo": {"pairs": len(pairs), "prompts": len({p["prompt_id"] for p in pairs}), "dropped_too_long": too_long,
                "loop_pairs": sum(p["kind"] == "loop" for p in pairs),
                "outcome_pairs": sum(p["kind"] == "outcome" for p in pairs),
                "commits": len(resumed), "commit_successes": sum(bool(r["success"]) for r in resumed),
                "loops_committed": len({(r["prompt_id"], r["resume_of"]) for r in resumed}),
                "max_length": max_length, "per_prompt_cap": PAIRS_PER_PROMPT, "window_cap": WINDOW_CAP,
                "chosen_tokens": quantiles([len(p["chosen_ids"]) for p in pairs]),
                "rejected_tokens": quantiles([len(p["rejected_ids"]) for p in pairs]),
                "pair_length": quantiles([len(p["prompt_ids"]) + max(len(p["chosen_ids"]), len(p["rejected_ids"]))
                                          for p in pairs])}}
    (out / "selection.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path, help="where commit.jsonl, dpo.jsonl and selection.json go")
    parser.add_argument("rollouts", type=Path, nargs="*", help="rollout directories (default: out)")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()
    print(json.dumps(select(args.out, args.rollouts or [args.out], args.max_length), indent=1))
