"""
Post-generation checks (syntax, basedpyright, g++ -fsyntax-only, the prompt's own examples) and the resampling search
of the filtered mode: a greedy candidate, then seeded samples until one passes (Codex, arXiv 2107.03374 §3.5;
AlphaCode, arXiv 2203.07814 §4.5).
"""

import ast
import contextlib
import dataclasses
import doctest
import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import tokenize
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from inference.decoder import NativeDecoder, Request
from inference.syntax_constraint import syntax_error
from metrics import latency

# basedpyright rules kept on (every other rule off, typeCheckingMode "off"): errors that fail at run time
# (NameError, AttributeError, call TypeErrors) or, for reportCallIssue, unmatched overloads; on d16 filtered-mode pools
# they rejected 0 of 100 correct candidates (docs/evaluation.md §7). reportReturnType is off: it rejects 3 HumanEval+
# reference solutions and alone rejected no incorrect HumanEval+ candidate.
PYRIGHT_RULES = ("reportUndefinedVariable", "reportAttributeAccessIssue", "reportCallIssue")
# A basedpyright process's limit: the largest normal round (FIM multi-line, 5,815 programs) takes 27.6 s; one generated
# program held its 400-program batch > 47 min. A stalling program in a round of N costs (ceil(log2 N) + 1) timeouts.
PYRIGHT_TIMEOUT = 60
# MultiPL-E judge's per-step timeout (safe_subprocess/__init__.py:33, eval_cpp.py)
CPP_TIMEOUT = 15
# HumanEval's execution timeout (human_eval/evaluation.py:43)
EXAMPLES_TIMEOUT = 3
# Codex's temperature for samples selected by unit tests (arXiv 2107.03374 §3.3, Fig. 1: the pass@100 optimum)
TEMPERATURE = 0.8
# Candidates per task, the greedy one included: pass@10 (Codex Table 1; EvalPlus, arXiv 2305.01210 Table 3)
ATTEMPTS = 10
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
    """syntax_constraint.syntax_error, then the compiler's own errors on a parsed program ("line:col: SyntaxError:
    message", or the exception's type and message): 'return' outside function, duplicate argument and the like, which
    ast.parse (the syntax constraint's end) accepts and every run of the program raises."""
    found = syntax_error(program)
    if found:
        return found
    with warnings.catch_warnings(action="ignore"):
        try:
            compile(program, "program.py", "exec")
        except SyntaxError as e:
            return [f"{e.lineno}:{e.offset}: SyntaxError: {e.msg}"]
        except (MemoryError, RecursionError) as e:
            return [f"{type(e).__name__}: {e}"]
    return []


def python_diagnostics(programs: list[str]) -> list[list[str]]:
    """Each program's compile error or, if it compiles, basedpyright's PYRIGHT_RULES errors ("line:col: rule: message").
    One basedpyright process checks all, each program in a directory of its own (no program imports another). A
    program basedpyright cannot check fails: an exception inside its checker (e.g. RangeError on deep inference chains)
    aborts the whole run without a report, so the rest are checked again without it; a run past PYRIGHT_TIMEOUT is
    split in halves until the program that stalls it is alone."""
    out, paths = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, program in enumerate(programs):
            out.append(compile_error(program))
            if out[-1]:
                continue
            paths[i] = Path(tmp, str(i), "program.py")
            paths[i].parent.mkdir()
            paths[i].write_text(program, encoding="utf-8")
        config = Path(tmp, "pyrightconfig.json")
        config.write_text(json.dumps({"typeCheckingMode": "off", **dict.fromkeys(PYRIGHT_RULES, "error")}),
                          encoding="utf-8")

        def check(group: dict[int, Path]) -> list[dict]:
            """basedpyright's diagnostics of group's programs; a program it cannot check gets its failure in out."""
            while group:
                # its own session: on a timeout the whole group dies, the node process python -m basedpyright starts
                # included (killing the wrapper alone leaves node checking the stalling program)
                process = subprocess.Popen([sys.executable, "-m", "basedpyright", "--outputjson", "--pythonpath",
                                            sys.executable, "--project", config, *group.values()], cwd=tmp,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                           start_new_session=True)
                try:
                    # exit status 1 when it reports errors; the JSON report is on stdout either way
                    stdout, stderr = process.communicate(timeout=PYRIGHT_TIMEOUT)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()
                    if len(group) == 1:
                        out[next(iter(group))].append(f"1:1: basedpyright: timeout after {PYRIGHT_TIMEOUT} s")
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
                out[i].append(f"1:1: basedpyright: internal error: {crash[2]}")
                group = {k: p for k, p in group.items() if k != i}
            return []

        for d in check(paths):
            if d["severity"] == "error" and d.get("rule") in PYRIGHT_RULES:
                start = d["range"]["start"]
                out[int(Path(d["file"]).parent.name)].append(
                    f"{start['line'] + 1}:{start['character'] + 1}: {d['rule']}: {d['message']}")
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


def cpp_diagnostics(program: str) -> list[str]:
    """`g++ -std=c++17 -fsyntax-only`'s error lines on the program ([] if it compiles; ["timeout"] past CPP_TIMEOUT)."""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "program.cpp").write_text(program, encoding="utf-8")
        try:
            result = subprocess.run(["g++", "-std=c++17", "-fsyntax-only", "program.cpp"], cwd=tmp,
                                    capture_output=True, text=True, timeout=CPP_TIMEOUT)
        except subprocess.TimeoutExpired:
            return ["timeout"]
    if result.returncode == 0:
        return []
    lines = result.stderr.splitlines()
    return [line for line in lines if ": error: " in line or ": fatal error: " in line] or lines[-1:]


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
    """Whether the program passes the examples (RUN_EXAMPLES) in a fresh isolated interpreter within timeout;
    True without examples."""
    if not found:
        return True
    data = json.dumps({"program": program, "examples": found})
    with tempfile.TemporaryDirectory() as tmp:
        try:
            result = subprocess.run([sys.executable, "-I", "-c", RUN_EXAMPLES], input=data, cwd=tmp,
                                    capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0 and result.stdout.endswith("EXAMPLES_DONE\n")


def vetted_examples(tasks: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """Each (prompt, reference program)'s examples that the reference passes, each run alone and untimed (a trusted
    program): a prompt's wrong example (HumanEval/47's median 15.0, all three of HumanEval/116's) would reject every
    correct program."""
    found = [examples(prompt) for prompt, _ in tasks]
    runs = sorted({(e, reference) for (_, reference), es in zip(tasks, found, strict=True) for e in es})
    with ThreadPoolExecutor(os.cpu_count()) as pool:
        passed = dict(zip(runs, pool.map(lambda r: examples_pass([r[0]], r[1], timeout=None), runs), strict=True))
    return [[e for e in es if passed[e, reference]] for (_, reference), es in zip(tasks, found, strict=True)]


def verdicts(checks: list[tuple[list[tuple[str, str]], str, list[str]]], *,
             language: Literal["python", "cpp"]) -> list[bool]:
    """Whether each (vetted examples, program, known diagnostics) is accepted: no diagnostic (known or found) and, for
    Python, the examples pass (run only without diagnostics)."""
    programs = [program for _, program, _ in checks]
    with ThreadPoolExecutor(os.cpu_count()) as pool:
        found = python_diagnostics(programs) if language == "python" else list(pool.map(cpp_diagnostics, programs))
        return list(pool.map(lambda c, d: not c[2] and not d and (language == "cpp" or examples_pass(*c[:2])),
                             checks, found))


@dataclasses.dataclass(frozen=True, kw_only=True)
class Candidate:
    """A task's candidates[0] is greedy, candidates[a] for a > 0 a sample with seed a at TEMPERATURE."""
    ids: list[int]
    accepted: bool  # verdicts'


def search(generator: NativeDecoder, request: Callable[[int], Request],
           check: Callable[[int, list[int]], tuple[list[tuple[str, str]], str, list[str]]], *,
           language: Literal["python", "cpp"],
           count: int) -> tuple[list[list[Candidate]], dict]:
    """Each task's candidates in attempt order and metrics.latency over every round. Each round decodes the tasks
    without an accepted candidate (at most ATTEMPTS rounds); request(i) must build a fresh constraint per call."""
    tried = [[] for _ in range(count)]
    pending = list(range(count))
    stamps, duration = [], 0.0
    with ThreadPoolExecutor(os.cpu_count()) as pool:
        for attempt in range(ATTEMPTS):
            sampling = {"temperature": 0.0 if attempt == 0 else TEMPERATURE, "seed": attempt}
            generated, checks = {}, {}
            begin = time.perf_counter()
            for k, ids in generator.generate(dataclasses.replace(request(i), **sampling) for i in pending):
                generated[pending[k]], checks[pending[k]] = ids, pool.submit(check, pending[k], ids)
            duration += time.perf_counter() - begin
            stamps += [(*generator.timings[k], len(generated[i])) for k, i in enumerate(pending)]
            accepted = verdicts([checks[i].result() for i in pending], language=language)
            for i, ok in zip(pending, accepted, strict=True):
                tried[i].append(Candidate(ids=generated[i], accepted=ok))
            pending = [i for i in pending if not tried[i][-1].accepted]
    return tried, latency(*zip(*stamps, strict=True), duration=duration)


def chosen(candidates: list[Candidate]) -> Candidate:
    """The candidate a search returns: the accepted one (the last tried), else the greedy one."""
    return candidates[-1] if candidates[-1].accepted else candidates[0]
