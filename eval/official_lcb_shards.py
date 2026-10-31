"""LiveCodeBench generation in resumable shards (scripts/official_reasoning.sh): lcb_runner writes a run's outputs only
once every problem is generated, so the window is split into shards generated one after another, each an
eval/official_lcb.py run with lcb_runner's own --start_date/--end_date and its outputs kept (a shard with outputs is
skipped on a rerun); lcb_runner's evaluation then scores their union as one run's outputs.

- plan: the shards of a window, a line "<start> <end>" each in <out file>: the calendar months of the window's
  contest dates, each boundary moved to the month's first day without a contest. lcb_runner's date filter keeps a
  problem with start <= contest_date <= end at midnight, both ends inclusive, and contest dates carry times (e.g.
  2025-03-01T18:30), so only a boundary on a day without problems splits them without overlap or gap; checked: every
  window problem falls in exactly one shard.
- evaluate: in the lcb directory, the union of shards/*/<lcb_runner's output file> (each window problem exactly once)
  through the steps of lcb_runner's runner/main.py after generation, with its --evaluate: the outputs sorted and
  saved, get_metrics, extract_instance_results, insert_output_evaluation; the same three files as one run's (output,
  _eval, _eval_all), so the metric and its per-problem results are lcb_runner's.

    python -m eval.official_lcb_shards plan <release> <start> <end> <out file>   (/opt/lcb's Python, /opt/LiveCodeBench)
    python -m eval.official_lcb_shards evaluate <main.py arguments of the window>    (in the lcb directory)
"""

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

SHARDS = "shards"  # <lcb directory>/shards/<shard start>/: one shard's lcb_runner working directory


def first_of_next_month(d: date) -> date:
    return date(d.year + d.month // 12, d.month % 12 + 1, 1)


def plan(release: str, start: str, end: str) -> list[tuple[date, date]]:
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset

    problems = load_code_generation_dataset(release, start_date=start, end_date=end)
    days = {p.contest_date.date() for p in problems}
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    bounds, month = [first], first_of_next_month(first)
    while month < last:
        boundary = month
        while boundary in days:
            boundary += timedelta(days=1)
        if boundary < last:
            bounds.append(boundary)
        month = first_of_next_month(month)
    bounds.append(last)
    shards = list(zip(bounds, bounds[1:]))
    for p in problems:  # lcb_runner's filter (benchmarks/code_generation.py) per shard
        found = [s for s in shards
                 if datetime.combine(s[0], datetime.min.time()) <= p.contest_date
                 <= datetime.combine(s[1], datetime.min.time())]
        if len(found) != 1:
            raise ValueError(f"{p.question_id} ({p.contest_date}) falls in {len(found)} shards")
    return shards


def evaluate() -> None:
    from lcb_runner.evaluation import extract_instance_results
    from lcb_runner.runner.parser import get_args
    from lcb_runner.runner.scenario_router import build_prompt_benchmark, get_metrics, sort_and_extract_save_results
    from lcb_runner.utils.path_utils import get_output_path

    from eval.official_lcb import MODEL, register

    register()
    args = get_args()
    benchmark, _ = build_prompt_benchmark(args)
    output_path = get_output_path(MODEL, args)
    save_results = [r for path in sorted(Path(SHARDS).glob(f"*/{output_path}"))
                    for r in json.loads(path.read_text())]
    ids = sorted(r["question_id"] for r in save_results)
    if ids != sorted(p.question_id for p in benchmark):
        raise ValueError(f"the shards' {len(ids)} problems ({len(set(ids))} distinct) are not the window's "
                         f"{len(benchmark)}")
    # runner/main.py from its outputs on, with --evaluate
    save_results, combined_results = sort_and_extract_save_results(args.scenario, save_results)
    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=4)
    metrics = get_metrics(args.scenario, args, benchmark, combined_results)
    graded = extract_instance_results(metrics[1])
    save_eval_results = [
        instance.insert_output_evaluation(outputs_list, extracted_list, graded_list, metadata=meta)
        for instance, (outputs_list, extracted_list), graded_list, meta in zip(
            benchmark, combined_results, graded, metrics[2])]
    with open(output_path.replace(".json", "_eval.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(output_path.replace(".json", "_eval_all.json"), "w") as f:
        json.dump(save_eval_results, f, indent=4)


if __name__ == "__main__":
    if sys.argv[1] == "plan":
        Path(sys.argv[5]).write_text("".join(f"{a.isoformat()} {b.isoformat()}\n" for a, b in plan(*sys.argv[2:5])))
    else:
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        evaluate()
