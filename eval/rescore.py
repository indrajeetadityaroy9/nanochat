"""
CPU replay of the scorers and checkers of a checkpoint's evaluation (eval/suite.py evaluate's record and its
sampled-completions sidecar) on its stored completions, without a model: every scorer reruns (EvalPlus base and plus,
MultiPL-E's judge, the infilling checks, parse outcomes, the filtered mode's verdicts) on the evaluation's code paths
and every aggregate is recomputed by the functions the evaluation uses. Copied from the record, as they need the model
or the packed corpus, and so not validated by an equal replay: the likelihood sums and sample weights (each context's
"corpora" and "weights", "solution"), the filtered searches' "attempts", and every "latency". Checks run against an
empty VerdictCache of their own (a temporary database), so every verdict is recomputed, none read from paths.VERDICTS.

    python -m eval.rescore <eval-json> [--out <path>]

writes the replay (suite.write_evaluation: a record and its sidecar) to --out, prints each field whose replay differs
from the evaluation, and exits 1 if any does. Records written before the sidecar lack "samples" and raise KeyError.
"""
import argparse
import json
import sys
import tempfile
import warnings
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from human_eval_infilling.data import read_problems

from corpus.mixture import FILE_LEVEL_STAGE, REPO_LEVEL_STAGE
from eval.bpb import pooled
from eval.cache import VerdictCache
from eval.code_eval import (
    FILTERED_MODE,
    Task,
    codegen_scores,
    codegen_scoring,
    filtered_record,
    fim_lines,
    infilling,
    parse_scores,
    reference_lines,
    sanitizers,
    tasks,
    vetted,
)
from eval.suite import read_evaluation, write_evaluation
from eval.verify import timed_workers, verdicts

STAGES = {"8k": FILE_LEVEL_STAGE, "32k": REPO_LEVEL_STAGE}


def search_record(stored: dict, accepted: list[bool]) -> dict:
    """A filtered record's search record: "accepted" replayed, "attempts" and "latency" stored. The chosen completion
    is the accepted candidate, else the greedy one, which the search tried first and rejected: its verdict is the
    stored "accepted"."""
    return {"attempts": stored["attempts"], "accepted": accepted, "latency": stored["latency"]}


def codegen(pool: ProcessPoolExecutor, runs: ThreadPoolExecutor, name: str, items: list[Task],
            found: list[list[tuple[str, str]]], stored: dict) -> dict:
    """An EvalPlus or MultiPL-E benchmark's {"greedy", "sampled", "filtered"} records rescored; found: its tasks'
    vetted examples."""
    check, language, score = codegen_scoring(pool, name)
    greedy, sampled, filtered = stored["greedy"], stored["sampled"], stored["filtered"]
    chosen = filtered["completions"]
    accepted = verdicts([(f, *check(t, c)) for f, t, c in zip(found, items, chosen, strict=True)], language=language,
                        pool=runs)
    return {"greedy": codegen_scores(score, items, "greedy", [[c] for c in greedy["completions"]], greedy["latency"]),
            "sampled": codegen_scores(score, items, "sampled", sampled["completions"], sampled["latency"]),
            "filtered": filtered_record(score, items, chosen, search_record(filtered, accepted))}


def fim(runs: ThreadPoolExecutor, key: str, found: dict[str, list[list[tuple[str, str]]]], stored: dict) -> dict:
    """An "<benchmark>/<mode>" infilling entry's records rescored: greedy, greedy_reference_lines (from the greedy
    middles) and, in FILTERED_MODE, filtered (on found[benchmark], its vetted examples)."""
    name, mode = key.split("/")
    problems = list(read_problems(name).values())
    middles = stored["greedy"]["completions"]
    out = {"greedy": infilling(name, problems, middles, {"latency": stored["greedy"]["latency"]})}
    lines = reference_lines(name, problems)
    if lines is not None:
        out["greedy_reference_lines"] = infilling(
            name, problems, [fim_lines(m, k) for m, k in zip(middles, lines, strict=True)], {})
    if mode == FILTERED_MODE:
        filtered = stored["filtered"]
        accepted = verdicts([(f, p["prompt"] + m + p["suffix"], [])
                             for f, p, m in zip(found[name], problems, filtered["completions"], strict=True)],
                            language="python", pool=runs)
        out["filtered"] = infilling(name, problems, filtered["completions"], search_record(filtered, accepted))
    return out


def rescore(stored: dict) -> dict:
    """The replay of an evaluation record, in its layout and key order."""
    likelihood = {key: None if stored[key] is None else pooled(stored[key]["corpora"], stored[key]["weights"], stage)
                  for key, stage in STAGES.items()}
    benchmarks = tasks()
    found = None if stored["codegen"] is None else vetted(benchmarks)  # the filtered records' examples
    with sanitizers() as pool, ThreadPoolExecutor(timed_workers()) as runs:
        parse = {name: parse_scores(pool, name, benchmarks[name], r["completions"], r["latency"])
                 for name, r in stored["parse"].items()}
        codes = None if stored["codegen"] is None else {
            name: codegen(pool, runs, name, benchmarks[name], found[name], records)
            for name, records in stored["codegen"].items()}
        fims = None if stored["fim"] is None else {key: fim(runs, key, found, records)
                                                   for key, records in stored["fim"].items()}
    return ({"run_name": stored["run_name"], "step": stored["step"]} | likelihood | {"solution": stored["solution"]}
            | {"parse": parse, "codegen": codes, "fim": fims})


def differences(stored: object, replayed: object, path: str = "") -> Iterator[str]:
    """Each leaf (or missing key, or list length) at which a replay differs from the stored record."""
    if isinstance(stored, dict) and isinstance(replayed, dict):
        for key in [*stored, *(k for k in replayed if k not in stored)]:
            if key not in stored or key not in replayed:
                yield f"{path}/{key}: only in the {'replay' if key not in stored else 'record'}"
            else:
                yield from differences(stored[key], replayed[key], f"{path}/{key}")
    elif isinstance(stored, list) and isinstance(replayed, list) and len(stored) == len(replayed):
        for i, (a, b) in enumerate(zip(stored, replayed, strict=True)):
            yield from differences(a, b, f"{path}[{i}]")
    elif stored != replayed:
        yield f"{path}: {stored!r:.100} != {replayed!r:.100}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay an evaluation record's scores on the CPU.")
    parser.add_argument("evaluation", type=Path, help="an eval-<step>.json")
    parser.add_argument("--out", type=Path, help="where to write the replay")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=SyntaxWarning)  # ast.parse's, on generated code (as run.py)
    stored = read_evaluation(args.evaluation)
    with tempfile.TemporaryDirectory() as tmp:
        VerdictCache.path = Path(tmp, "verdicts.sqlite")
        replayed = rescore(stored)
    if args.out is not None:
        write_evaluation(args.out, replayed)
    found = list(differences(stored, json.loads(json.dumps(replayed, allow_nan=False))))
    print(*found, f"{len(found)} fields differ", sep="\n")
    sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
