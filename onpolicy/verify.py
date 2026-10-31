"""
KodCode's tests on a response's answer (docs/post-training.md §8). The answer's code (`code`) is its last fenced block
tagged python that defines one of the names the tests import (else its last python block: a later block is often a
usage example or a test), else its last untagged block, dedented; a response with none fails. Instruct and complete
problems: the code as solution.py beside KodCode's pytest file after `from solution import *` (a share of the files
name the solution's functions and modules without importing them), passing when pytest exits 0 within
PYTEST_TIMEOUT. Online-judge problems: the code as a program per stdin case within CASE_TIMEOUT each, passing when
every case's stdout equals the expected one line by line, trailing whitespace and trailing blank lines ignored. A
prompt whose tests a wrong program (`stub`: its names returning None, or no output) passes has trivial tests; its
passes are 'pass_trivial', not successes. Each run is a fresh process in its own session and temporary directory,
under resource limits (address space MEMORY, file size FILE_SIZE, CPU seconds its timeout, no core files), its whole
process group killed at the end, with single-threaded BLAS and no network (the container's); the workers are
spawned, so no parent state (CUDA) is inherited. Run in a CPU-only container.

On the pool's own R1 answers (each passed KodCode's tests in KodCode's environment), `--known-good`: 2,995 of 3,000
train and 1,499 of 1,500 val rows pass; the 6 others are answers naming the function otherwise than the tests (2),
tests reading the environment (the home directory; bs4, absent here) and failing answers (2). Stubs pass 3 of 3,000
and 4 of 1,500 (tests of plotting or printing side effects).

    python -m onpolicy.verify <rollouts dir> [--workers N]   # the rollouts' answers: verdicts.jsonl
    python -m onpolicy.verify --known-good <split> <n>       # the pool's own R1 answers, a sample of n rows
"""
import argparse
import ast
import contextlib
import json
import os
import re
import resource
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

PYTEST_TIMEOUT, CASE_TIMEOUT = 30, 10  # seconds
MEMORY, FILE_SIZE = 16 << 30, 64 << 20  # bytes; torch's import alone reserves more than 4 GiB of address space
TEST_PREFIX = "from solution import *\n"
FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)
PYTHON_TAGS = ("python", "py", "python3")
ENV = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "LC_ALL": "C.UTF-8", "MPLBACKEND": "Agg"}


def code(answer: str, names: list[str] = ()) -> str | None:
    """The answer's last fenced block tagged python that defines (def or class) one of `names`, else its last python
    block, else its last untagged block; dedented (a fence inside a list item carries the item's indentation); None
    if it has none."""
    blocks = FENCE.findall(answer)
    python = [body for tag, body in blocks if tag.strip().lower() in PYTHON_TAGS]
    defining = [body for body in python
                if any(re.search(rf"^\s*(def|class)\s+{re.escape(n)}\b", body, re.M) for n in names)]
    found = defining or python or [body for tag, body in blocks if not tag.strip()]
    return textwrap.dedent(found[-1]) if found else None


def limits(seconds: int) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_SIZE, FILE_SIZE))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))


def run(command: list[str], cwd: str, timeout: int, stdin: str = "") -> tuple[int | None, str]:
    """(exit code or None at the timeout, stdout) of command in its own session; its process group killed after."""
    with tempfile.TemporaryFile("w+") as inp, tempfile.TemporaryFile("w+") as out:
        inp.write(stdin)
        inp.seek(0)
        process = subprocess.Popen(command, cwd=cwd, stdin=inp, stdout=out, stderr=subprocess.DEVNULL, text=True,
                                   env=ENV | {"HOME": cwd, "TMPDIR": cwd}, start_new_session=True,
                                   preexec_fn=lambda: limits(timeout))
        try:
            code_ = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            code_ = None
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        out.seek(0)
        return code_, out.read()


def lines(text: str) -> list[str]:
    out = [line.rstrip() for line in text.split("\n")]
    while out and not out[-1]:
        out.pop()
    return out


def check(task: tuple[str | None, str, str]) -> str:
    """'pass', 'fail', 'timeout' or 'no_code' for (code, test, kind)."""
    program, test, kind = task
    if program is None:
        return "no_code"
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "solution.py").write_text(program)
        if kind == "pytest":
            Path(directory, "test_solution.py").write_text(TEST_PREFIX + test)
            status, _ = run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
                             "test_solution.py"], directory, PYTEST_TIMEOUT)
            return "timeout" if status is None else "pass" if status == 0 else "fail"
        cases = ast.literal_eval(test)
        for stdin, expected in zip(cases["stdin"], cases["stdout"], strict=True):
            status, out = run([sys.executable, "solution.py"], directory, CASE_TIMEOUT, stdin)
            if status is None:
                return "timeout"
            if status != 0 or lines(out) != lines(expected):
                return "fail"
        return "pass"


def verify(tasks: list[tuple[str | None, str, str]], workers: int) -> list[str]:
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        return list(pool.map(check, tasks, chunksize=4))


def stub(row: dict) -> str:
    """A wrong program for a row: each name its tests import returning None (pytest), or one printing nothing."""
    return "".join(f"def {name}(*args, **kwargs):\n    return None\n" for name in row["names"]) or "pass\n"


def verify_rollouts(directory: Path, workers: int) -> None:
    """verdicts.jsonl for rollouts.parquet: one line per response, its verdict ('not_closed' for a response that
    never closed its reasoning; an answer cut at the limit is tested as it stands; 'pass_trivial' for a pass on a
    prompt whose stub passes)."""
    import pyarrow.parquet as pq

    prompts = {r["id"]: r for r in map(json.loads, (directory / "prompts.jsonl").read_text().splitlines())}
    rows = pq.read_table(directory / "rollouts.parquet",
                         columns=["prompt_id", "sample", "closed", "answer"]).to_pylist()
    start = time.time()
    closed = [r for r in rows if r["closed"]]
    tested = sorted({r["prompt_id"] for r in closed})
    verdicts = verify([(code(r["answer"], prompts[r["prompt_id"]]["names"]), prompts[r["prompt_id"]]["test"],
                        prompts[r["prompt_id"]]["kind"]) for r in closed]
                      + [(stub(prompts[p]), prompts[p]["test"], prompts[p]["kind"]) for p in tested], workers)
    trivial = {p for p, v in zip(tested, verdicts[len(closed):], strict=True) if v == "pass"}
    found = {(r["prompt_id"], r["sample"]): "pass_trivial" if v == "pass" and r["prompt_id"] in trivial else v
             for r, v in zip(closed, verdicts[:len(closed)], strict=True)}
    with open(directory / "verdicts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({"prompt_id": r["prompt_id"], "sample": r["sample"],
                                "verdict": found.get((r["prompt_id"], r["sample"]), "not_closed")}) + "\n")
    print(json.dumps({"responses": len(rows), "tested": len(closed), "trivial_prompts": len(trivial),
                      "seconds": round(time.time() - start)}))


def known_good(split: str, n: int, workers: int) -> dict:
    """The verifier on the pool's own R1 answers (each passed KodCode's tests in KodCode's environment): the verdicts
    of n rows drawn by onpolicy.prompts.sample, per style and per subset family (the subset before its first ':' or
    '_'); and on wrong programs (`stub`) for the same rows, whose passes are tests that accept anything."""
    from onpolicy.prompts import read_pool, sample

    rows = sample(read_pool(split), n, seed=1)
    start = time.time()
    verdicts = verify([(code(r["r1_answer"], r["names"]), r["test"], r["kind"]) for r in rows], workers)
    stubs = verify([(stub(r), r["test"], r["kind"]) for r in rows], workers)
    by: dict[str, dict[str, dict[str, int]]] = {"style": {}, "family": {}}
    for r, v in zip(rows, verdicts, strict=True):
        for kind, name in (("style", r["style"]), ("family", re.split(r"[:_]", r["subset"])[0])):
            counts = by[kind].setdefault(name, {})
            counts[v] = counts.get(v, 0) + 1
    return {"split": split, "rows": n, "pass": verdicts.count("pass") / n, "by_style": by["style"],
            "by_family": by["family"], "stub_pass": stubs.count("pass") / n,
            "seconds": round(time.time() - start), "workers": workers,
            "failures": [{"id": r["id"], "verdict": v} for r, v in zip(rows, verdicts, strict=True) if v != "pass"],
            "stub_passes": [r["id"] for r, v in zip(rows, stubs, strict=True) if v == "pass"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", type=Path, nargs="?")
    parser.add_argument("--known-good", nargs=2, metavar=("SPLIT", "N"))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 8))
    args = parser.parse_args()
    if args.known_good:
        print(json.dumps(known_good(args.known_good[0], int(args.known_good[1]), args.workers), indent=1))
    else:
        verify_rollouts(args.rollouts, args.workers)
