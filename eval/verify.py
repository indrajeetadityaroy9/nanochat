"""
Post-generation checks (syntax, basedpyright, g++ -fsyntax-only, the prompt's own examples) and the resampling search
of the filtered mode: a greedy candidate, then seeded samples until one passes (Codex, arXiv 2107.03374 §3.5;
AlphaCode, arXiv 2203.07814 §4.5).
"""

import ast
import contextlib
import dataclasses
import doctest
import fcntl
import functools
import hashlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import tokenize
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from evalplus.eval import query_maximum_memory_bytes

from eval.cache import VerdictCache
from inference.decoder import NativeDecoder, Request
from inference.syntax_constraint import syntax_error
from metrics import latency
from paths import PCH

# basedpyright rules kept on (every other rule off, typeCheckingMode "off"): errors that fail at run time
# (NameError, AttributeError, call TypeErrors) or, for reportCallIssue, unmatched overloads; on d16 filtered-mode pools
# they rejected 0 of 100 correct candidates (docs/evaluation.md §7). reportReturnType is off: it rejects 3 HumanEval+
# reference solutions and alone rejected no incorrect HumanEval+ candidate.
PYRIGHT_RULES = ("reportUndefinedVariable", "reportAttributeAccessIssue", "reportCallIssue")
# The configuration python_diagnostics writes (pyrightconfig.json), also the basedpyright verdicts' cache identity
PYRIGHT_CONFIG = {"typeCheckingMode": "off", **dict.fromkeys(PYRIGHT_RULES, "error")}
# A basedpyright process's limit: one process over the largest normal round (FIM multi-line, 5,815 programs) takes
# 27.6 s, a group (at most max(PYRIGHT_GROUP, ceil(N / timed_workers())) programs) a few seconds; one generated program
# held its 400-program batch > 47 min. A stalling program in a group of G costs (ceil(log2 G) + 1) timeouts.
PYRIGHT_TIMEOUT = 60
# The fewest programs a basedpyright process checks: a process starts in ≈ 0.4 s and then checks ≈ 2.7 ms per program
# (5,450 programs in 15.3 s), so a group of 100 spends about as long starting as checking; a call of N programs runs
# ceil(N / max(100, ceil(N / timed_workers()))) processes, one per timed-check core at most
PYRIGHT_GROUP = 100
# A failure of basedpyright itself in python_diagnostics' output (timeout, internal error): never kept in VerdictCache
PYRIGHT_FAILURE = "1:1: basedpyright: "
# MultiPL-E judge's per-step timeout (safe_subprocess/__init__.py:33, eval_cpp.py) and its compiler flags (eval_cpp.py)
CPP_TIMEOUT, CPP_FLAGS = 15, ("-std=c++17",)
# HumanEval's execution timeout (human_eval/evaluation.py:43)
EXAMPLES_TIMEOUT = 3
# Codex's temperature for samples selected by unit tests (arXiv 2107.03374 §3.3, Fig. 1: the pass@100 optimum)
TEMPERATURE = 0.8
# Codex's nucleus for all its sampling (arXiv 2107.03374 §3.2: top p 0.95), the filtered search's and the sampled mode's
TOP_P = 0.95
# Candidates per task, the greedy one included: pass@10 (Codex Table 1; EvalPlus, arXiv 2305.01210 Table 3)
ATTEMPTS = 10
# Cores of the process's starting affinity that the evaluation process tree keeps to, its timed children excepted
# (suite.evaluate): the decoder's host thread, its syntax masks (rayon) and the in-process checks of the filtered
# search, which used 13.0-15.5 cores (30-s means, at most 18.6) of a d24 benchmarks evaluation on the 64-core GH200
# host; timed checks run on the others (EvalPlus warns against an oversubscribed test bed, docs/execution.md)
DECODE_CORES = 16


@functools.cache
def cores() -> tuple[frozenset[int], frozenset[int]]:
    """(decode cores, timed-check cores): the last DECODE_CORES of the affinity the process starts with and the rest
    (both all of it when it has no more, as eval.rescore under a smaller cpuset). Called before suite.evaluate pins."""
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) <= DECODE_CORES:
        return frozenset(allowed), frozenset(allowed)
    return frozenset(allowed[-DECODE_CORES:]), frozenset(allowed[:-DECODE_CORES])


def timed(command: list) -> list:
    """command run on the timed-check cores (taskset, which execs it: same pid, same process group)."""
    return ["taskset", "--cpu-list", ",".join(map(str, sorted(cores()[1]))), *command]


def timed_workers() -> int:
    """Concurrent timed-check children: one per timed-check core."""
    return len(cores()[1])


@functools.cache
def timed_budget() -> threading.Semaphore:
    """The process's timed_workers() slots for timed-check children, one per timed-check core."""
    return threading.Semaphore(timed_workers())


@contextlib.contextmanager
def timed_slots(n: int = 1) -> Iterator[None]:
    """n of timed_budget()'s slots, held while the timed children they cover run: the scorer thread's evaluators (all
    slots) and the filtered searches' checks (one each) share them, so at most one timed child runs per timed-check
    core. Taken one at a time: the one caller taking several (the scorer thread, or eval.rescore's main thread) waits
    for single holders to finish, which never wait on slots themselves."""
    budget, taken = timed_budget(), 0
    try:
        for _ in range(n):
            budget.acquire()
            taken += 1
        yield
    finally:
        for _ in range(taken):
            budget.release()


# examples_pass's child: the program under evalplus.eval.utils reliability_guard, applied as EvalPlus's test runs do
# (evalplus/eval/__init__.py unsafe_execute); examples checked by value; EXAMPLES_DONE marks a run not ended by exit()
RUN_EXAMPLES = """
import ast, builtins, importlib, importlib.util, json, sys
from evalplus.eval import query_maximum_memory_bytes
from evalplus.eval.utils import reliability_guard
data = json.load(sys.stdin)
reliability_guard(maximum_memory_bytes=query_maximum_memory_bytes())
program = {"__name__": "program"}
exec(compile(data["program"], "program.py", "exec"), program)
for source, want in data["examples"]:
    tree, scope = ast.parse(source), dict(program)
    for name in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - scope.keys() - vars(builtins).keys():
        if importlib.util.find_spec(name):
            scope[name] = importlib.import_module(name)
    if want:
        assert eval(source, scope) == eval(want, scope)
    elif len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
        value = eval(source, scope)
        assert value is None or value is True
    else:
        exec(source, scope)
print("EXAMPLES_DONE")
"""


def compile_error(program: str) -> list[str]:
    """The compiler's error on a program ("line:col: SyntaxError: message", or the exception's type and message), []
    if it compiles: syntax errors, and 'return' outside function, duplicate argument and the like, which ast.parse (the
    syntax constraint's end) accepts and every run of the program raises. Parser overflow (MemoryError, RecursionError)
    and CPython 3.12.3's ValueError for f"{x:{y=}}" (python/cpython#121130) are failures."""
    try:
        compile(program, "program.py", "exec")
    except SyntaxError as e:
        return [f"{e.lineno}:{e.offset}: SyntaxError: {e.msg}"]
    except (MemoryError, RecursionError, ValueError) as e:
        return [f"{type(e).__name__}: {e}"]
    return []


@functools.cache
def verdict_cache(kind: Literal["basedpyright", "examples", "g++"], timeout: float | None = None) -> VerdictCache:
    """The VerdictCache of a check: its parameters, and for g++ the compiler (compiler()'s digest; cache.environment
    holds the Python packages', basedpyright and evalplus included)."""
    identity = {"basedpyright": lambda: PYRIGHT_CONFIG,
                "examples": lambda: [RUN_EXAMPLES, query_maximum_memory_bytes(), timeout],
                "g++": lambda: [compiler().name, *CPP_FLAGS, "-fsyntax-only"]}[kind]()
    return VerdictCache(kind, json.dumps(identity))


def python_diagnostics(programs: list[str], pool: ThreadPoolExecutor) -> list[list[str]]:
    """Each program's compile error or, if it compiles, basedpyright's PYRIGHT_RULES errors ("line:col: rule: message"),
    kept in VerdictCache. Each program sits in a directory of its own; basedpyright processes check disjoint groups of
    them concurrently, on pool (not the caller's). A program basedpyright cannot check fails: an exception inside its
    checker (e.g. RangeError on deep inference chains) aborts that process without a report, so the rest of its group
    are checked again without it; a run past PYRIGHT_TIMEOUT is split in halves until the program that stalls it is
    alone."""
    out = [compile_error(program) for program in programs]
    compiled = [i for i, found in enumerate(out) if not found]
    cache = verdict_cache("basedpyright")
    unchecked = []
    for i, found in zip(compiled, cache.get([programs[i] for i in compiled]), strict=True):
        if found is None:
            unchecked.append(i)
        else:
            out[i] = found
    paths = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i in unchecked:
            paths[i] = Path(tmp, str(i), "program.py")
            paths[i].parent.mkdir()
            paths[i].write_text(programs[i], encoding="utf-8")
        config = Path(tmp, "pyrightconfig.json")
        config.write_text(json.dumps(PYRIGHT_CONFIG), encoding="utf-8")

        def check(group: dict[int, Path]) -> list[dict]:
            """basedpyright's diagnostics of group's programs; a program it cannot check gets its failure in out."""
            while group:
                # its own session: on a timeout the whole group dies, the node process python -m basedpyright starts
                # included (killing the wrapper alone leaves node checking the stalling program)
                with timed_slots():
                    process = subprocess.Popen(timed([sys.executable, "-m", "basedpyright", "--outputjson",
                                                      "--pythonpath", sys.executable, "--project", config,
                                                      *group.values()]), cwd=tmp, stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE, text=True, start_new_session=True)
                    try:
                        # exit status 1 when it reports errors; the JSON report is on stdout either way
                        stdout, stderr = process.communicate(timeout=PYRIGHT_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.communicate()
                        stdout = None
                if stdout is None:  # timed out: the slot is free again before the halves take theirs
                    if len(group) == 1:
                        out[next(iter(group))].append(f"{PYRIGHT_FAILURE}timeout after {PYRIGHT_TIMEOUT} s")
                        return []
                    items = list(group.items())
                    return check(dict(items[:len(items) // 2])) + check(dict(items[len(items) // 2:]))
                if stdout.strip():
                    return json.loads(stdout)["generalDiagnostics"]
                # pyright.js sourceFile.check rethrows after its internal-error diagnostic; analyzeProgram then exits 2
                crash = re.search(r'internal error occurred while type checking file "([^"]+)": (.*)', stderr)
                if crash is None:
                    raise RuntimeError(f"basedpyright exited {process.returncode} without a report: {stderr}")
                i = int(Path(crash[1]).parent.name)
                out[i].append(f"{PYRIGHT_FAILURE}internal error: {crash[2]}")
                group = {k: p for k, p in group.items() if k != i}
            return []

        # one process per timed-check core at most, over disjoint contiguous groups of at least PYRIGHT_GROUP programs:
        # a program's diagnostics do not depend on its group (each in its own directory, none imports another)
        items = list(paths.items())
        size = max(PYRIGHT_GROUP, -(-len(items) // timed_workers()))
        reports = list(pool.map(check, [dict(items[k:k + size]) for k in range(0, len(items), size)]))
        for d in (d for report in reports for d in report):
            if d["severity"] == "error" and d.get("rule") in PYRIGHT_RULES:
                start = d["range"]["start"]
                out[int(Path(d["file"]).parent.name)].append(
                    f"{start['line'] + 1}:{start['character'] + 1}: {d['rule']}: {d['message']}")
    cache.put([(programs[i], out[i]) for i in unchecked if not any(d.startswith(PYRIGHT_FAILURE) for d in out[i])])
    return out


def solution_diagnostics(solution: str, entry_point: str) -> list[str]:
    """Certain failures of an EvalPlus solution its own checks miss: the solution's syntax error, and an entry function
    missing or without `return <value>` in its own body (every call returns None)."""
    unparsed = syntax_error(solution)
    if unparsed:
        return unparsed
    functions = [n for n in ast.parse(solution).body
                 if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == entry_point]
    if not functions:
        return [f"{entry_point}: not defined"]
    stack = list(functions[-1].body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Return) and node.value is not None:
            return []
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef):
            stack.extend(ast.iter_child_nodes(node))
    return [f"{entry_point}: returns no value"]


@functools.cache
def compiler() -> Path:
    """The directory of the C++ checks' `g++`: the system g++ with <bits/stdc++.h> forced first (-include) as a
    precompiled header, built once per compiler, flags and header under PCH/<digest of them> (the C++ verdict caches'
    compiler identity; a file lock serializes builders; the wrapper, written last, marks a finished build). Every
    MultiPL-E C++ prompt (558 of 558 at the pinned revision) begins "#include<assert.h>\\n#include<bits/stdc++.h>\\n",
    so the forced header declares nothing a judged program would not see: assert.h is re-read (it has no include
    guard) and bits/stdc++.h's headers are guarded. Same
    verdicts, a compile 4.0x and -fsyntax-only 5.3x faster [measured on 30 programs, 20-core aarch64; GCC manual,
    "Precompiled Headers": -include works with a precompiled header]."""
    gxx = shutil.which("g++")
    version = subprocess.run([gxx, "-v"], capture_output=True, text=True, check=True).stderr
    trace = subprocess.run([gxx, *CPP_FLAGS, "-x", "c++", "-fsyntax-only", "-H", "-"],
                           input="#include <bits/stdc++.h>\n", capture_output=True, text=True, check=True).stderr
    header = Path(next(line[2:] for line in trace.splitlines() if line.startswith(". ")))
    digest = hashlib.sha256(json.dumps([gxx, version, CPP_FLAGS]).encode() + header.read_bytes()).hexdigest()
    target = PCH / digest
    PCH.mkdir(parents=True, exist_ok=True)
    with open(PCH / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (target / "g++").exists():
            target.mkdir(exist_ok=True)
            shutil.copyfile(header, target / "stdc++.h")
            subprocess.run([gxx, *CPP_FLAGS, "-x", "c++-header", target / "stdc++.h", "-o", target / "stdc++.h.gch"],
                           check=True)
            wrapper = target / "g++.tmp"
            wrapper.write_text(f'#!/bin/sh\nexec {gxx} -include "$(dirname "$0")/stdc++.h" "$@"\n', encoding="utf-8")
            wrapper.chmod(0o755)
            wrapper.rename(target / "g++")
    return target


def cpp_diagnostics(program: str) -> list[str]:
    """`g++ -std=c++17 -fsyntax-only`'s error lines on the program (compiler()'s g++; [] if it compiles; ["timeout"]
    past CPP_TIMEOUT), kept in VerdictCache unless g++ timed out or failed other than by reporting errors (status 1)."""
    cache = verdict_cache("g++")
    [stored] = cache.get([program])
    if stored is not None:
        return stored
    with tempfile.TemporaryDirectory() as tmp, timed_slots():
        Path(tmp, "program.cpp").write_text(program, encoding="utf-8")
        try:
            result = subprocess.run(timed([compiler() / "g++", *CPP_FLAGS, "-fsyntax-only", "program.cpp"]), cwd=tmp,
                                    capture_output=True, text=True, timeout=CPP_TIMEOUT)
        except subprocess.TimeoutExpired:
            return ["timeout"]
    lines = result.stderr.splitlines()
    found = [] if result.returncode == 0 else (
        [line for line in lines if ": error: " in line or ": fatal error: " in line] or lines[-1:])
    if result.returncode in (0, 1):
        cache.put([(program, found)])
    return found


def examples(prompt: str) -> list[tuple[str, str]]:
    """The prompt's own examples as (source, want) pairs from its string literals: doctest `>>>` examples and Python
    `assert` lines. Tokenizing stops where the prompt ends mid-statement; a literal doctest rejects adds none."""
    found = []
    with contextlib.suppress(tokenize.TokenError, SyntaxError):
        for token in tokenize.generate_tokens(io.StringIO(prompt).readline):
            if token.type == tokenize.STRING:
                text = ast.literal_eval(token.string)
                with contextlib.suppress(ValueError):
                    found += [(e.source, e.want) for e in doctest.DocTestParser().get_examples(text)]
                found += [(line.strip(), "") for line in text.splitlines()
                          if line.lstrip().startswith("assert ") and not syntax_error(line.strip())]
    return found


def examples_pass(found: list[tuple[str, str]], program: str, *, timeout: float | None = EXAMPLES_TIMEOUT) -> bool:
    """Whether the program passes the examples (RUN_EXAMPLES) in a fresh isolated interpreter within timeout (on the
    timed-check cores, in a timed slot; untimed, timeout None, on the caller's); True without examples. Kept in
    VerdictCache unless the run timed out or a signal ended it."""
    if not found:
        return True
    data = json.dumps({"program": program, "examples": found})
    cache = verdict_cache("examples", timeout)
    [stored] = cache.get([data])
    if stored is not None:
        return stored
    command = [sys.executable, "-I", "-c", RUN_EXAMPLES]
    with tempfile.TemporaryDirectory() as tmp, contextlib.nullcontext() if timeout is None else timed_slots():
        try:
            result = subprocess.run(command if timeout is None else timed(command), input=data, cwd=tmp,
                                    capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
    passed = result.returncode == 0 and result.stdout.endswith("EXAMPLES_DONE\n")
    if result.returncode >= 0:
        cache.put([(data, passed)])
    return passed


def vetted_examples(tasks: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """Each (prompt, reference program)'s examples that the reference passes, each run alone and untimed (a trusted
    program): a prompt's wrong example (HumanEval/47's median 15.0, all three of HumanEval/116's) would reject every
    correct program."""
    found = [examples(prompt) for prompt, _ in tasks]
    runs = sorted({(e, reference) for (_, reference), es in zip(tasks, found, strict=True) for e in es})
    with ThreadPoolExecutor(os.cpu_count()) as pool:
        passed = dict(zip(runs, pool.map(lambda r: examples_pass([r[0]], r[1], timeout=None), runs), strict=True))
    return [[e for e in es if passed[e, reference]] for (_, reference), es in zip(tasks, found, strict=True)]


def verdicts(checks: list[tuple[list[tuple[str, str]], str, list[str]]], *, language: Literal["python", "cpp"],
             pool: ThreadPoolExecutor) -> list[bool]:
    """Whether each (vetted examples, program, known diagnostics) is accepted: no diagnostic (known or found) and the
    examples pass (run only without diagnostics; a run stops at the first failing example; C++ tasks have none). A
    program with a known diagnostic is rejected unchecked. Checks run on pool (not the caller's)."""
    unknown = [check for check in checks if not check[2]]
    programs = [program for _, program, _ in unknown]
    found = python_diagnostics(programs, pool) if language == "python" else list(pool.map(cpp_diagnostics, programs))
    passed = iter(pool.map(lambda c, d: not d and examples_pass(*c[:2]), unknown, found))
    return [not known and next(passed) for _, _, known in checks]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Candidate:
    """A task's candidates[0] is greedy; candidates[a] for a > 0 of task i a sample at TEMPERATURE from the TOP_P
    nucleus with seed ATTEMPTS i + a (the sampled mode's scheme: no two attempts of a search share a seed)."""
    ids: list[int]
    accepted: bool  # verdicts'


@dataclasses.dataclass(frozen=True, kw_only=True)
class Search:
    """A filtered search over tasks 0..count-1: request(i) builds a fresh constraint per call; check(i, ids) gives a
    candidate's (vetted examples, program, known diagnostics) for verdicts."""
    request: Callable[[int], Request]
    check: Callable[[int, list[int]], tuple[list[tuple[str, str]], str, list[str]]]
    language: Literal["python", "cpp"]
    count: int


def search(generator: NativeDecoder, searches: list[Search],
           done: Callable[[int, list[list[Candidate]], dict], None]) -> None:
    """Every search's candidates from one generate call; done(s, candidates, latency) once search s ends: each task's
    candidates in attempt order and metrics.latency over its kept candidates (its duration: the sum over rounds of
    the span from their first start to their last end). Requests come in attempt order (every search's attempt 0, then
    attempt 1, ...), each built when a decoder slot takes it, unless its task has an accepted candidate by then. Each
    search's verdicts run round by round on a thread of their own, round a as soon as its candidates of the tasks
    without an accepted candidate before a are decoded: the programs a serial search checks (a candidate's tokens
    depend on its request and attempt alone), with decoding as many rounds ahead as the verdicts lag. The searches'
    checks share one pool of timed_workers() threads."""
    accepted_at = [[None] * x.count for x in searches]  # a task's accepted attempt, set by its search's judge
    prepared, generated, timings, issued = {}, {}, {}, []  # (attempt, search, task) -> check future, ids, timing
    lock, failure = threading.Lock(), []

    def future(key: tuple[int, int, int]) -> Future:
        with lock:
            if key not in prepared:
                prepared[key] = Future()
                if failure:
                    prepared[key].set_exception(failure[0])
            return prepared[key]

    def requests() -> Iterator[Request]:
        for attempt in range(ATTEMPTS):
            for s, x in enumerate(searches):
                for i in range(x.count):
                    if accepted_at[s][i] is None:
                        issued.append((attempt, s, i))
                        yield (x.request(i) if attempt == 0 else dataclasses.replace(
                            x.request(i), temperature=TEMPERATURE, top_p=TOP_P, seed=ATTEMPTS * i + attempt))

    def prepare(key: tuple[int, int, int], ids: list[int]) -> None:
        try:
            check = searches[key[1]].check(key[2], ids)
        except BaseException as e:
            future(key).set_exception(e)
            raise
        future(key).set_result(check)

    def judge(s: int) -> None:
        x, pending, stamps, duration = searches[s], list(range(searches[s].count)), [], 0.0
        tried = [[] for _ in range(x.count)]
        for attempt in range(ATTEMPTS):
            accepted = verdicts([future((attempt, s, i)).result() for i in pending], language=x.language, pool=runs)
            kept = [timings[attempt, s, i] for i in pending]
            stamps += [(*timings[attempt, s, i], len(generated[attempt, s, i])) for i in pending]
            duration += max(end for _, _, end in kept) - min(start for start, _, _ in kept)
            for i, ok in zip(pending, accepted, strict=True):
                tried[i].append(Candidate(ids=generated[attempt, s, i], accepted=ok))
                if ok:
                    accepted_at[s][i] = attempt
            pending = [i for i in pending if accepted_at[s][i] is None]
            if not pending:
                break
        done(s, tried, latency(*zip(*stamps, strict=True), duration=duration))

    with (ThreadPoolExecutor(os.cpu_count()) as preparing, ThreadPoolExecutor(timed_workers()) as runs,
          ThreadPoolExecutor(len(searches)) as judges):
        judging = [judges.submit(judge, s) for s in range(len(searches))]
        try:
            for k, ids in generator.generate(requests()):
                key = issued[k]
                generated[key], timings[key] = ids, generator.timings[k]
                preparing.submit(prepare, key, ids)
        except BaseException as e:
            with lock:
                failure.append(e)
                for f in prepared.values():
                    if not f.done():
                        f.set_exception(e)
            raise
        for j in judging:
            j.result()


def chosen(candidates: list[Candidate]) -> Candidate:
    """The candidate a search returns: the accepted one (the last tried), else the greedy one."""
    return candidates[-1] if candidates[-1].accepted else candidates[0]
