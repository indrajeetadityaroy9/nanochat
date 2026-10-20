"""
Post-generation checks of a program and the resampling search that uses them (the "system" generation mode,
eval/code_eval.py and eval/fim_eval.py):
- python_diagnostics: Python 3 syntax errors (ast.parse), else the errors of a static type checker (basedpyright,
  pyproject.toml) limited to rules whose findings are definite runtime failures or contradict an annotation;
- cpp_diagnostics: the errors of `g++ -std=c++17 -fsyntax-only`, the MultiPL-E judge's compiler and standard
  (evaluation/src/eval_cpp.py) without code generation;
- examples_pass: the prompt's own examples, never a benchmark's hidden tests;
- search: per task a greedy candidate, then seeded samples, until one has no diagnostics and passes its prompt's
  examples (Codex, Chen et al., arXiv 2107.03374 §3.5: "evaluating the first sample which passes the 3 public unit
  tests"; AlphaCode's filtering by the problem's example tests, Li et al., arXiv 2203.07814 §4.5).
"""

import ast
import contextlib
import dataclasses
import doctest
import io
import json
import os
import subprocess
import sys
import tempfile
import tokenize
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from nanochat.inference.decoder import Constraint, NativeDecoder, Request
from nanochat.inference.syntax_constraint import python_syntax

# basedpyright rules (docs/configuration/config-files.md) whose errors on known types are runtime failures or
# annotation contradictions: undefined variables (NameError), attribute accesses (AttributeError), call expressions and
# their arguments (arity and keyword TypeErrors), argument types against parameter annotations, return types against
# the declared return annotation. Every other rule is off (typeCheckingMode "off"): style, unknown types and
# possibly-unbound or optional-member findings that a correct program may legitimately trigger.
PYRIGHT_RULES = ("reportUndefinedVariable", "reportAttributeAccessIssue", "reportCallIssue", "reportArgumentType",
                 "reportReturnType")
# the MultiPL-E judge's per-step timeout (safe_subprocess/__init__.py:33, eval_cpp.py: 15 s compile, 15 s run)
CPP_TIMEOUT = 15
# HumanEval's execution timeout (human_eval/execution.py check_correctness timeout=3.0; Codex §3.5: 3 seconds)
EXAMPLES_TIMEOUT = 3
# sampling temperature of the candidates after the greedy one: Codex's temperature for samples selected by unit tests
# (arXiv 2107.03374 Fig. 1, §3.3: T = 0.8 is the pass@100 optimum, higher temperatures are optimal for larger k)
TEMPERATURE = 0.8
# candidates per task, the greedy one included: pass@10, the intermediate budget Codex (Table 1) and EvalPlus
# (arXiv 2305.01210 Table 3) report
ATTEMPTS = 10
# examples_pass's child process: runs the program as a module named "program", then each example (stdin JSON source/want
# pairs) in a copy of its namespace plus the modules the example names but the program does not bind (MBPP+ prompt
# asserts call math.isclose, imported by the original MBPP's test_imports); a failing example raises (exit status 1).
# Values, not doctest's output text: HumanEval writes some wants in an equal but differently printed form ("21" for
# '21', a list split over lines; HumanEval/65, /113) and some examples as bare comparisons without output (/108, /145):
# an example with a want passes iff its value equals the want's; one without, iff an expression's value is None or
# True, a statement (assert) iff it runs.
RUN_EXAMPLES = """
import ast, builtins, importlib, importlib.util, json, sys
data = json.load(sys.stdin)
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
"""


def python_diagnostics(programs: list[str]) -> list[list[str]]:
    """Each program's syntax error ("line:col: SyntaxError: message") or, if it parses, basedpyright's PYRIGHT_RULES
    errors ("line:col: rule: message") against this interpreter's environment (its installed packages resolve). The
    programs that parse are checked by one basedpyright process (its start dominates a check), each as program.py in a
    directory of its own, so no program imports another."""
    out, paths = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, program in enumerate(programs):
            try:
                ast.parse(program)
            except SyntaxError as e:
                out.append([f"{e.lineno}:{e.offset}: SyntaxError: {e.msg}"])
                continue
            out.append([])
            paths[i] = Path(tmp, str(i), "program.py")
            paths[i].parent.mkdir()
            paths[i].write_text(program, encoding="utf-8")
        if not paths:
            return out
        config = Path(tmp, "pyrightconfig.json")
        config.write_text(json.dumps({"typeCheckingMode": "off", **dict.fromkeys(PYRIGHT_RULES, "error")}),
                          encoding="utf-8")
        # exit status 1 when it reports errors; the JSON report is on stdout either way
        report = subprocess.run([sys.executable, "-m", "basedpyright", "--outputjson", "--pythonpath", sys.executable,
                                 "--project", config, *paths.values()], cwd=tmp, capture_output=True, text=True)
    for d in json.loads(report.stdout)["generalDiagnostics"]:
        if d["severity"] == "error" and d.get("rule") in PYRIGHT_RULES:
            start = d["range"]["start"]
            out[int(Path(d["file"]).parent.name)].append(
                f"{start['line'] + 1}:{start['character'] + 1}: {d['rule']}: {d['message']}")
    return out


def cpp_diagnostics(program: str) -> list[str]:
    """The error lines of `g++ -std=c++17 -fsyntax-only` on the program ("program.cpp:line:col: error: ...";
    ["timeout"] past CPP_TIMEOUT)."""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "program.cpp").write_text(program, encoding="utf-8")
        try:
            result = subprocess.run(["g++", "-std=c++17", "-fsyntax-only", "program.cpp"], cwd=tmp,
                                    capture_output=True, text=True, timeout=CPP_TIMEOUT)
        except subprocess.TimeoutExpired:
            return ["timeout"]
    return [line for line in result.stderr.splitlines() if ": error: " in line]


def parses(source: str) -> bool:
    """Whether source is Python 3 syntax."""
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


def examples(prompt: str) -> list[tuple[str, str]]:
    """The prompt's own examples as (source, want) pairs, read from its string literals (docstrings): their doctest
    `>>>` examples (HumanEval) and their `assert` lines that are Python (EvalPlus MBPP+ prompts: the task description
    and one assert in a docstring; HumanEval/149 writes `assert f(x) => y` prose). A literal that doctest rejects
    (inconsistent indentation of an example, doctest.py _check_prefix) contributes no `>>>` examples, as
    `python -m doctest` would run none of it. A prompt is a program's beginning, possibly mid-statement: the literals
    are those tokenize yields before it stops (TokenError at an open bracket or string, IndentationError at a partial
    dedent)."""
    found = []
    with contextlib.suppress(tokenize.TokenError, SyntaxError):
        for token in tokenize.generate_tokens(io.StringIO(prompt).readline):
            if token.type == tokenize.STRING:
                text = ast.literal_eval(token.string)
                with contextlib.suppress(ValueError):
                    found += [(e.source, e.want) for e in doctest.DocTestParser().get_examples(text)]
                found += [(line.strip(), "") for line in text.splitlines()
                          if line.lstrip().startswith("assert ") and parses(line.strip())]
    return found


def examples_pass(prompt: str, program: str, *, language: Literal["python", "cpp"]) -> bool:
    """Whether the program passes the prompt's examples (RUN_EXAMPLES: by value), run in a fresh isolated interpreter
    in an empty directory within EXAMPLES_TIMEOUT. True without examples; C++ prompts have none here."""
    found = examples(prompt) if language == "python" else []
    if not found:
        return True
    data = json.dumps({"program": program, "examples": found})
    with tempfile.TemporaryDirectory() as tmp:
        try:
            result = subprocess.run([sys.executable, "-I", "-c", RUN_EXAMPLES], input=data, cwd=tmp,
                                    capture_output=True, text=True, timeout=EXAMPLES_TIMEOUT)
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0


def verdicts(checks: list[tuple[str, str]], *, language: Literal["python", "cpp"]) -> list[tuple[list[str], bool]]:
    """Each (prompt, program)'s (diagnostics, whether it passes the prompt's examples), the examples run only without
    diagnostics: search's check of its candidates. Python's diagnostics come from one python_diagnostics call, C++'s
    and the examples from a thread pool (each a child process)."""
    with ThreadPoolExecutor(os.cpu_count()) as pool:
        if language == "python":
            diagnostics = python_diagnostics([program for _, program in checks])
        else:
            diagnostics = list(pool.map(cpp_diagnostics, [program for _, program in checks]))
        passed = list(pool.map(lambda c, d: not d and examples_pass(*c, language=language), checks, diagnostics))
    return list(zip(diagnostics, passed, strict=True))


def syntax_constraint(prefix: str, *, suffix: str = "") -> Constraint:
    """The complete Python 3 syntax constraint of a generation after prefix (and before suffix, FIM), of the process's
    PythonSyntax (python_syntax); the decoder computes the masks of all its slots' syntax constraints in one batch."""
    return python_syntax().constraint(prefix, suffix=suffix)


@dataclass(frozen=True, kw_only=True)
class Candidate:
    """A generated candidate: attempt 0 is greedy, attempt a > 0 a sample with seed a at TEMPERATURE."""
    attempt: int
    ids: list[int]
    diagnostics: list[str]
    examples: bool  # the prompt's examples pass (checked only without diagnostics)

    @property
    def accepted(self) -> bool:
        return not self.diagnostics and self.examples


def search(decoder: NativeDecoder, request: Callable[[int], Request],
           program: Callable[[int, list[int]], tuple[str, str]], *, language: Literal["python", "cpp"],
           count: int) -> list[list[Candidate]]:
    """The candidates of tasks 0..count-1 in attempt order: request(i) is task i's greedy request (a fresh constraint
    each call), program(i, ids) the (prompt, program) that verdicts checks for generated ids. Each round decodes every
    task without an accepted candidate yet (attempt a: request(i) at temperature 0 for a = 0, TEMPERATURE with seed a
    after), at most ATTEMPTS rounds, then checks the round's candidates at once (verdicts)."""
    tried = [[] for _ in range(count)]
    pending = list(range(count))
    for attempt in range(ATTEMPTS):
        sampling = {"temperature": 0.0 if attempt == 0 else TEMPERATURE, "seed": attempt}
        generated = {}
        for k, ids in decoder.run(dataclasses.replace(request(i), **sampling) for i in pending):
            generated[pending[k]] = ids
        checks = verdicts([program(i, generated[i]) for i in pending], language=language)
        for i, (diagnostics, passed) in zip(pending, checks, strict=True):
            tried[i].append(Candidate(attempt=attempt, ids=generated[i], diagnostics=diagnostics, examples=passed))
        pending = [i for i in pending if not tried[i][-1].accepted]
    return tried


def chosen(candidates: list[Candidate]) -> Candidate:
    """The candidate a search returns: the accepted one (the last tried), else the greedy one."""
    return candidates[-1] if candidates[-1].accepted else candidates[0]
