"""The reasoning protocol's EvalPlus outputs for one checkpoint (scripts/official_reasoning.sh,
RUNS/<run>/official-<step>/evalplus/) as summary.json: per benchmark and mode (greedy: n 1, T 0; sampled: n 20, T 0.6,
top-p 0.95), how the samples end (eval/official_evalplus.py's <samples>.endings.jsonl) apart from how they score
(evalplus.evaluate's <samples>.eval_results.json): the share that closes "</think>", the finish reasons, the samples
ended by the server's loop stop (eval/official_loop_stop.py), pass@1 over every sample, and pass@1 over the samples
that closed (pooled: each problem weighted by its closed samples); "base" on HumanEval's or MBPP's tests, "plus" on
both those and EvalPlus's (base_status and plus_status "pass"). A sample joins its verdict by its problem and its
sanitized solution text. "loop_stops": the server's stops of EvalPlus's samples (from their token ids) and of
lcb_runner's (lcb_runner keeps no token ids: the lines of each shard's kept log, lcb/shards/<start>/loop-stops.jsonl,
over the shards of lcb/shards.txt).

    python -m eval.official_reasoning <official-<step> dir>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

MODES = {"greedy": "nanochat_openai_temp_0.0", "sampled": "nanochat_openai_temp_0.6"}
BENCHMARKS = {"humaneval_plus": "humaneval", "mbpp_plus": "mbpp"}
LOOP_STOPS = "loop-stops.jsonl"  # a LiveCodeBench shard's log of scripts/official_reasoning.sh's NANOCHAT_LOOP_STOP_LOG


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def samples(stem: Path) -> list[dict]:
    """Each sample's ending and verdicts, in the samples file's order (stem: the samples file less ".jsonl")."""
    rows, endings = jsonl(Path(f"{stem}.jsonl")), jsonl(Path(f"{stem}.endings.jsonl"))
    if [r["task_id"] for r in rows] != [e["task_id"] for e in endings]:
        raise ValueError(f"{stem}: the samples and their endings differ in their problems or order")
    verdicts = defaultdict(list)  # (problem, solution) -> its verdicts, in evaluate's order
    for task_id, results in json.loads(Path(f"{stem}.eval_results.json").read_text())["eval"].items():
        for r in results:
            verdicts[task_id, r["solution"]].append(r)
    out = []
    for row, ending in zip(rows, endings, strict=True):
        r = verdicts[row["task_id"], row["solution"]].pop(0)
        out.append(ending | {"base": r["base_status"] == "pass",
                            "plus": r["base_status"] == "pass" and r["plus_status"] == "pass"})
    return out


def cell(rows: list[dict]) -> dict:
    closed = [r for r in rows if r["closed"]]
    reasons = defaultdict(int)
    for r in rows:
        reasons[r["finish_reason"]] += 1
    return {"problems": len({r["task_id"] for r in rows}), "samples": len(rows), "closed": len(closed) / len(rows),
            "finish_reasons": dict(reasons), "loop_stopped": sum(r["loop_stopped"] for r in rows),
            "pass@1": {k: sum(r[k] for r in rows) / len(rows) for k in ("base", "plus")},
            "pass@1_closed": {k: sum(r[k] for r in closed) / len(closed) if closed else None for k in ("base", "plus")}}


if __name__ == "__main__":
    out = Path(sys.argv[1])
    summary = {benchmark: {mode: cell(samples(out / "evalplus" / dataset / identifier))
                           for mode, identifier in MODES.items()}
               for benchmark, dataset in BENCHMARKS.items()}
    shards = [line.split()[0] for line in (out / "lcb" / "shards.txt").read_text().splitlines()]
    summary["loop_stops"] = {
        "evalplus": sum(c["loop_stopped"] for modes in summary.values() for c in modes.values()),
        "lcb": sum(len(jsonl(out / "lcb" / "shards" / start / LOOP_STOPS)) for start in shards)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
