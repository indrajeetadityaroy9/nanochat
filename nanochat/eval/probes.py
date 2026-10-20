"""
The pre-registered "Evaluate & Break" probes: 25 hand-written inputs (none from a benchmark), each with a programmatic,
deterministic check (parsing, execution in a fresh process with a timeout, g++, importlib, byte comparison), run under
the generation configurations:
- greedy: left-to-right greedy decoding of BOS + the prompt;
- healed: greedy with token healing (decoder.heal);
- fim: the prompt as a PSM fill-in-the-middle prefix (decoder.fim_prompt) with the probe's suffix, empty for a
  left-to-right probe;
- fim_spm: the SPM order of an infill probe (both orders are trained, data/pack.py);
- constrained: greedy under the complete Python 3 syntax constraint (verify.syntax_constraint), PSM for an infill probe;
  Python probes only;
- system: code_eval.system_complete, the benchmarks' system mode: healed greedy (Python also syntax-constrained), then
  verify.search's checks and seeded resampling (eval/verify.py); left-to-right probes only.
Left-to-right completions end at BOS, at a special token (MultiPL-E automodel.py:99-101) or at their language's stop
sequences (EvalPlus's for Python, MultiPL-E's for C++), within EvalPlus's budget; middles end at BOS. A ValueError of
the decoder (context guard) or of the syntax constraint (prefix not Python 3) is the configuration's refusal: the
expected outcome of M3, a failure elsewhere. Results go to PROBES/<tag>.json and .md (a snapshot's:
<tag>-step<step>.json and .md).

    python -m nanochat.eval.probes --model-tag=<tag> [--step=<snapshot step>]
"""

import argparse
import ast
import dataclasses
import importlib.util
import itertools
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
from evalplus.provider.utility import EOS, extra_eos_for_direct_completion

from nanochat.config import FILE_STAGE, REPO_STAGE
from nanochat.data.sources import COMMENTS
from nanochat.data.tokenizer import Tokenizer, get_tokenizer
from nanochat.eval.code_eval import StopAt, Task, completion, system_complete
from nanochat.eval.verify import CPP_TIMEOUT, EXAMPLES_TIMEOUT, chosen, python_diagnostics, syntax_constraint
from nanochat.inference.decoder import NativeDecoder, Request, fim_prompt, heal
from nanochat.model.gpt import GPT
from nanochat.paths import PROBES
from nanochat.train.checkpoint import checkpoint_name, load_model

# PROBES/<name>.json: schema, run, step, seconds, results (a list of probe x configuration records)
SCHEMA = 1
Configuration = Literal["greedy", "healed", "fim", "fim_spm", "constrained", "system"]
CONFIGURATIONS = typing.get_args(Configuration)
FIM_MODES = {"fim": "psm", "fim_spm": "spm"}
# EvalPlus's budget for a base model (evalplus/provider/base.py:13 max_new_tokens)
MAX_TOKENS = 768
# EvalPlus's stops for direct completion (provider/utility.py EOS + extra_eos_for_direct_completion), MultiPL-E's
# humaneval-cpp stop_tokens
STOPS = {"python": (*EOS, *extra_eos_for_direct_completion("humaneval")), "cpp": ("\n}",)}
# a C++ completion stops before the function's closing "\n}", which MultiPL-E's tests restore (their first line "}")
CPP_CLOSE = "\n}\n"
# the context the M2 helper sits behind: three file-level contexts, inside the repo-level stage only
FAR = 3 * FILE_STAGE.seq_len
# M3's context: beyond the repo-level stage's by one file-level context (~40K tokens)
BEYOND = REPO_STAGE.seq_len + FILE_STAGE.seq_len


@dataclass(frozen=True, kw_only=True)
class Generation:
    """A probe's generated text: the completion (or middle) cut at its first stop, the program it forms (prompt +
    completion + suffix; C++: + CPP_CLOSE), the generated token count and how generation ended."""
    completion: str
    program: str
    tokens: int
    end: Literal["bos", "stop", "special", "budget"]
    attempts: int = 1  # system: candidates tried


@dataclass(frozen=True, kw_only=True)
class Probe:
    """A probe: its prompt (an infill probe's prefix), an infill probe's suffix, the code its check runs after the
    program, and its check (the failure diagnostic of a generation, "" when it passes; None: the decoder must refuse)."""
    id: str
    hypothesis: str
    language: Literal["python", "cpp"]
    prompt: str
    check: Callable[["Probe", Generation], str] | None
    tests: str = ""
    suffix: str | None = None


def configurations(probe: Probe) -> tuple[Configuration, ...]:
    """The configurations a probe runs under (module docstring)."""
    if probe.suffix is not None:
        return "fim", "fim_spm", "constrained"
    if probe.language == "python":
        return "greedy", "healed", "fim", "constrained", "system"
    return "greedy", "healed", "fim", "system"


# ---- checks

def execute(files: dict[str, str], commands: list[list[str]], *, timeout: float) -> str:
    """Run commands in turn in a fresh directory holding files: "" if each exits 0, else the failing one's first
    compiler error line or last output line (a Python exception), or its timeout."""
    with tempfile.TemporaryDirectory() as tmp:
        for path, text in files.items():
            file = pathlib.Path(tmp, path)
            # inside the fresh temp directory; files may share a subdirectory
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(text, encoding="utf-8")
        for command in commands:
            try:
                done = subprocess.run(command, cwd=tmp, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                return f"{command[0]}: timed out after {timeout} s"
            if done.returncode:
                lines = (done.stderr + done.stdout).strip().splitlines() or [f"exit status {done.returncode}"]
                return next((line for line in lines if ": error: " in line), lines[-1])
    return ""


def syntax_error(program: str) -> str:
    try:
        ast.parse(program)
    except SyntaxError as error:
        return f"SyntaxError: {error.msg} (line {error.lineno})"
    return ""


def runs(probe: Probe, generation: Generation) -> str:
    """The program parses (Python) and runs its probe's tests: Python in a fresh interpreter, C++ compiled by g++
    -std=c++17 (MultiPL-E's judge) and run."""
    if probe.language == "cpp":
        return execute({"main.cpp": generation.program + probe.tests},
                       [["g++", "-std=c++17", "-o", "main", "main.cpp"], ["./main"]], timeout=CPP_TIMEOUT)
    return syntax_error(generation.program) or execute({"main.py": generation.program + "\n\n" + probe.tests},
                                                       [[sys.executable, "main.py"]], timeout=EXAMPLES_TIMEOUT)


def parses(probe: Probe, generation: Generation) -> str:
    return syntax_error(generation.program)


def clean(probe: Probe, generation: Generation) -> str:
    """verify.python_diagnostics finds nothing (syntax, undefined names, annotation contradictions) and the tests run."""
    return "; ".join(python_diagnostics([generation.program])[0]) or runs(probe, generation)


def indents(text: str) -> list[str]:
    """The leading whitespace of each non-blank line of text."""
    return [line[:len(line) - len(line.lstrip())] for line in text.splitlines() if line.strip()]


def body_indented(probe: Probe, generation: Generation) -> str:
    """D1: the body's first line indented by 4 spaces (the docstring's), then the tests run."""
    first = indents(generation.completion)[:1]
    if first != ["    "]:
        return f"first body line indented {first!r}"
    return runs(probe, generation)


def import_resolves(probe: Probe, generation: Generation) -> str:
    """D2: the completed `import` statement (the prompt's last line, the completion's first) names a module importlib
    finds."""
    line = (probe.prompt.splitlines()[-1] + generation.completion).split("\n")[0]
    if syntax_error(line):
        return f"not an import statement: {line!r}"
    names = [alias.name for node in ast.parse(line).body for alias in getattr(node, "names", [])]
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    return f"no module {missing}" if missing or not names else ""


def cpp_syntax(probe: Probe, generation: Generation) -> str:
    """D5: g++ -std=c++17 -fsyntax-only accepts the program and a test main."""
    return execute({"main.cpp": generation.program + probe.tests},
                   [["g++", "-std=c++17", "-fsyntax-only", "main.cpp"]], timeout=CPP_TIMEOUT)


def ends_and_parses(probe: Probe, generation: Generation) -> str:
    """D6: generation ends by BOS or a stop sequence within the budget, and the program parses."""
    return "ran to the budget" if generation.end == "budget" else syntax_error(generation.program)


def function_only(probe: Probe, generation: Generation) -> str:
    """D7: the stops end the completion with the function: within the budget, the program one def, its tests run."""
    if generation.end == "budget":
        return "ran to the budget"
    error = syntax_error(generation.program)
    if error:
        return error
    extra = [type(node).__name__ for node in ast.parse(generation.program).body[1:]]
    return f"top-level statements after the function: {extra}" if extra else runs(probe, generation)


def middle_ends(probe: Probe, generation: Generation) -> str:
    """D9: the middle ends by BOS and prefix + middle + suffix parses."""
    return f"ended by {generation.end}" if generation.end != "bos" else syntax_error(generation.program)


def nothing_to_fill(probe: Probe, generation: Generation) -> str:
    """D10: a whitespace-only middle ended by BOS."""
    if generation.end != "bos":
        return f"ended by {generation.end}"
    return f"middle {generation.completion!r}" if generation.completion.strip() else ""


def plain_text(probe: Probe, generation: Generation) -> str:
    """D11: no special token generated; the program parses and its tests run."""
    return "generated a special token" if generation.end == "special" else runs(probe, generation)


def uses(helper: str, arity: int) -> Callable[[Probe, Generation], str]:
    """M1, M2, M4: the program calls helper with its arity and the repository (its files written out) runs the tests."""
    def check(probe: Probe, generation: Generation) -> str:
        error = syntax_error(generation.program)
        if error:
            return error
        calls = [len(node.args) + len(node.keywords) for node in ast.walk(ast.parse(generation.program))
                 if isinstance(node, ast.Call) and helper in (getattr(node.func, "id", None),
                                                              getattr(node.func, "attr", None))]
        if arity not in calls:
            return f"{helper} called with argument counts {calls}, not {arity}"
        return execute({**unpack(generation.program), "main.py": probe.tests}, [[sys.executable, "main.py"]],
                       timeout=EXAMPLES_TIMEOUT)
    return check


def kept_indentation(unit: str) -> Callable[[Probe, Generation], str]:
    """M5, M6: every indentation of the completion repeats unit, its first line two levels (a loop body), and the
    program parses and runs the tests."""
    def check(probe: Probe, generation: Generation) -> str:
        found = indents(generation.completion)
        if found[:1] != [2 * unit] or any(i.replace(unit, "") for i in found):
            return f"indentation {sorted(set(found))!r}"
        return runs(probe, generation)
    return check


def python3(probe: Probe, generation: Generation) -> str:
    """M7: the completed function (its def line and the completion) parses as Python 3."""
    header = probe.prompt.splitlines()[-1]
    return syntax_error(header + "\n" + generation.completion)


def round_trip(probe: Probe, generation: Generation) -> str:
    """M8: the prompt's bytes survive encode/decode, the generated bytes are valid UTF-8, and the tests run."""
    enc = get_tokenizer().enc
    if enc.decode(enc.encode_ordinary(probe.prompt)) != probe.prompt:
        return "prompt round trip differs"
    return "invalid UTF-8 generated" if "\ufffd" in generation.completion else runs(probe, generation)


# ---- repository-format context (nanochat/data/sources.py texts and documents)

def repository(files: list[tuple[str, str]]) -> str:
    """Files (path, text) as training serializes a repository sample (sources.py texts, documents): each file a line
    comment with its path, then its text with one line end, concatenated in order."""
    return "".join(f"{COMMENTS['Python']} {path}\n{text.removesuffix('\n')}\n" for path, text in files)


def unpack(program: str) -> dict[str, str]:
    """The files {path: text} of a repository-format program (inverse of repository)."""
    parts = re.split(rf"^{re.escape(COMMENTS['Python'])} (\S+\.py)\n", program, flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


class Entity(NamedTuple):
    """A record kind of the synthetic filler modules: class and module names, a text, an integer and a real field."""
    cls: str
    snake: str
    plural: str
    text: str
    count: str
    measure: str


ENTITIES = [Entity(*e) for e in (
    ("Invoice", "invoice", "invoices", "customer", "number", "amount"),
    ("Shipment", "shipment", "shipments", "carrier", "parcels", "weight_kg"),
    ("Reading", "reading", "readings", "sensor", "sequence", "value"),
    ("Booking", "booking", "bookings", "guest", "nights", "rate"),
    ("Ticket", "ticket", "tickets", "assignee", "priority", "hours"),
    ("Course", "course", "courses", "title", "credits", "fee"),
    ("Track", "track", "tracks", "artist", "seconds", "rating"),
    ("Vehicle", "vehicle", "vehicles", "plate", "seats", "mileage"),
    ("Recipe", "recipe", "recipes", "name", "servings", "calories"),
    ("Account", "account", "accounts", "owner", "pin_attempts", "balance"),
    ("Parcel", "parcel", "parcels", "address", "items", "postage"),
    ("Sample", "sample", "samples", "label", "replicate", "concentration"),
    ("Order", "order", "orders", "buyer", "quantity", "total"),
    ("Employee", "employee", "employees", "department", "grade", "salary"),
    ("Flight", "flight", "flights", "origin", "gate", "duration_h"),
    ("Patient", "patient", "patients", "ward", "age", "temperature"),
    ("Listing", "listing", "listings", "city", "bedrooms", "price"),
    ("Loan", "loan", "loans", "borrower", "term_months", "principal"),
    ("Fixture", "fixture", "fixtures", "venue", "round", "attendance"),
    ("Battery", "battery", "batteries", "model", "cycles", "capacity_ah"),
    ("Plot", "plot", "plots", "crop", "rows", "yield_t"),
    ("Device", "device", "devices", "hostname", "port", "uptime_h"),
    ("Article", "article", "articles", "author", "words", "score"),
    ("Review", "review", "reviews", "reviewer", "stars", "helpfulness"),
)]

# the filler modules of an entity, dependencies first (str.format fields: package and Entity's)
TEMPLATES = {
    "model": '''\
"""{cls} records of the {package} service."""

from dataclasses import dataclass, field


@dataclass
class {cls}:
    {text}: str
    {count}: int
    {measure}: float = 0.0
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {{"{text}": self.{text}, "{count}": self.{count}, "{measure}": self.{measure}, "tags": list(self.tags)}}

    @classmethod
    def from_dict(cls, data: dict) -> "{cls}":
        return cls(str(data["{text}"]), int(data["{count}"]), float(data.get("{measure}", 0.0)),
                   list(data.get("tags", [])))


def validate_{snake}(item: {cls}) -> list[str]:
    """The problems of item, empty when it is valid."""
    problems = []
    if not item.{text}.strip():
        problems.append("{text} is empty")
    if item.{count} < 0:
        problems.append("{count} is negative")
    if item.{measure} < 0:
        problems.append("{measure} is negative")
    return problems
''',
    "store": '''\
"""In-memory storage of {plural}."""

from {package}.{snake}_model import {cls}


class {cls}Store:
    """{plural} keyed by the id add() assigns."""

    def __init__(self) -> None:
        self.items: dict[int, {cls}] = {{}}
        self.next_id = 1

    def add(self, item: {cls}) -> int:
        item_id = self.next_id
        self.items[item_id] = item
        self.next_id += 1
        return item_id

    def get(self, item_id: int) -> {cls} | None:
        return self.items.get(item_id)

    def remove(self, item_id: int) -> bool:
        return self.items.pop(item_id, None) is not None

    def find(self, {text}: str) -> list[{cls}]:
        return [item for item in self.items.values() if item.{text} == {text}]

    def tagged(self, tag: str) -> list[{cls}]:
        return [item for item in self.items.values() if tag in item.tags]
''',
    "report": '''\
"""Summaries of {plural} for the {package} dashboards."""

import csv
import io
import statistics

from {package}.{snake}_model import {cls}


def total_{measure}(items: list[{cls}]) -> float:
    return sum(item.{measure} for item in items)


def mean_{count}(items: list[{cls}]) -> float:
    return statistics.fmean(item.{count} for item in items) if items else 0.0


def by_{text}(items: list[{cls}]) -> dict[str, list[{cls}]]:
    groups: dict[str, list[{cls}]] = {{}}
    for item in items:
        groups.setdefault(item.{text}, []).append(item)
    return groups


def to_csv(items: list[{cls}]) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["{text}", "{count}", "{measure}"])
    for item in items:
        writer.writerow([item.{text}, item.{count}, f"{{item.{measure}:.2f}}"])
    return out.getvalue()
''',
}


def filler(tokenizer: Tokenizer, packages: list[str], *, tokens: int) -> list[tuple[str, str]]:
    """Synthetic but realistic repository files (TEMPLATES of ENTITIES in each package, dependencies first) until
    their serialization holds at least `tokens` tokens."""
    files, count = [], 0
    for package, entity, (kind, template) in itertools.product(packages, ENTITIES, TEMPLATES.items()):
        path = f"{package.replace('.', '/')}/{entity.snake}_{kind}.py"
        files.append((path, template.format(package=package, **entity._asdict())))
        count += len(tokenizer.enc.encode_ordinary(repository(files[-1:])))
        if count >= tokens:
            return files
    raise ValueError(f"{len(files)} filler files hold {count} < {tokens} tokens")


# ---- the probes

def probes(tokenizer: Tokenizer) -> list[Probe]:
    """The 25 probes (local PROBES.md: D decoding/prompting, M model/training, S software system)."""
    units = ("pantry/units.py", '''\
"""Mass units of the pantry inventory."""

GRAMS_PER_UNIT = {"g": 1.0, "kg": 1000.0, "oz": 28.349523125, "lb": 453.59237}


def to_grams(quantity: float, unit: str) -> float:
    """quantity, measured in unit, in grams."""
    return quantity * GRAMS_PER_UNIT[unit]
''')
    report = ("pantry/report.py", '''\
"""Weight reports of the pantry inventory."""

from pantry.units import to_grams


def total_grams(items: list[tuple[str, float, str]]) -> float:
    """Total weight in grams of items given as (name, quantity, unit)."""
''')
    distance = ("geo/distance.py", '''\
"""Great-circle geometry on the Earth's surface."""

import math

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two points given in degrees."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))
''')
    routes = ("geo/routes.py", '''\
"""Lengths of routes given as waypoint lists."""

from geo.distance import haversine_km


def route_length_km(points: list[tuple[float, float]]) -> float:
    """Length in kilometres of the path through points, (latitude, longitude) pairs in degrees, in order."""
''')
    pricing = ("shop/pricing.py", '''\
"""Price rules of the shop."""


def apply_discount(price: float, percent: float) -> float:
    """price reduced by percent per cent."""
    return price * (1 - percent / 100)
''')
    return [
        Probe(id="D1", hypothesis='prompt ends after a docstring\'s closing """ + newline', language="python",
              check=body_indented, prompt='''\
def run_lengths(text: str) -> list[tuple[str, int]]:
    """Split text into runs of one repeated character, as (character, run length) pairs.

    >>> run_lengths("aaabcc")
    [('a', 3), ('b', 1), ('c', 2)]
    >>> run_lengths("")
    []
    """
''', tests='''\
assert run_lengths("aaabcc") == [("a", 3), ("b", 1), ("c", 2)]
assert run_lengths("") == []
assert run_lengths("zz z") == [("z", 2), (" ", 1), ("z", 1)]
'''),
        Probe(id="D2", hypothesis="prompt ends mid-identifier (import num)", language="python", check=import_resolves,
              prompt='"""Rolling statistics of sensor readings."""\n\nimport num'),
        Probe(id="D3", hypothesis="prompt ends in partial indentation (3 of 4 spaces)", language="python",
              check=parses, prompt='''\
def clamp(value: float, low: float, high: float) -> float:
    """value limited to the closed interval [low, high]."""
   '''),
        Probe(id="D4", hypothesis="prompt ends at a def line without its newline", language="python", check=runs,
              prompt='''\
# digit_sum(n): the sum of the decimal digits of a non-negative integer n, e.g. digit_sum(4096) == 19.
def digit_sum(n: int) -> int:''', tests='''\
assert digit_sum(4096) == 19
assert digit_sum(0) == 0
assert digit_sum(7) == 7
'''),
        Probe(id="D5", hypothesis="C++ prompt ends at { without newline (MultiPL-E strip)", language="cpp",
              check=cpp_syntax, prompt='''\
#include <string>

// squeeze("aaabccdd") == "abcd": s with every run of equal adjacent characters replaced by one character.
std::string squeeze(const std::string& s) {''', tests='''\
int main() { return squeeze("aab") == "ab" ? 0 : 1; }
'''),
        Probe(id="D6", hypothesis="greedy repetition loop on a long literal list", language="python",
              check=ends_and_parses, prompt='''\
# The powers of two from 2**0 through 2**20, written out.
POWERS_OF_TWO = [
'''),
        Probe(id="D7", hypothesis="base model runs past the function", language="python", check=function_only,
              prompt='''\
def interleave(first: list, second: list) -> list:
    """Alternate the items of first and second, starting with first; the rest of the longer list follows in order."""
''', tests='''\
assert interleave([1, 2, 3], ["x"]) == [1, "x", 2, 3]
assert interleave([], [5, 6]) == [5, 6]
'''),
        Probe(id="D8", hypothesis="FIM single-line infill: PSM and SPM disagree", language="python", check=runs,
              prompt='''\
def normalize(weights: dict[str, float]) -> dict[str, float]:
    """weights scaled to sum to 1."""
    total = sum(weights.values())
''', suffix="    return scaled\n", tests='''\
assert normalize({"a": 1.0, "b": 3.0}) == {"a": 0.25, "b": 0.75}
'''),
        Probe(id="D9", hypothesis="FIM multi-line middle never ends", language="python", check=middle_ends,
              prompt='''\
def histogram(values: list[int], width: int) -> dict[int, int]:
    """Counts of values per bucket [k * width, (k + 1) * width), keyed by the bucket's lower bound."""
    counts: dict[int, int] = {}
''', suffix="    return dict(sorted(counts.items()))\n"),
        Probe(id="D10", hypothesis="FIM with nothing to fill", language="python", check=nothing_to_fill,
              prompt="def double(x: int) -> int:\n    return 2 * x\n", suffix="\n\nassert double(21) == 42\n"),
        Probe(id="D11", hypothesis="special-token text (<fim_prefix>) in ordinary code", language="python",
              check=plain_text, prompt='''\
MARKERS = ("<fim_prefix>", "<fim_suffix>", "<fim_middle>")


def strip_markers(text: str) -> str:
    """text with every occurrence of each of MARKERS removed."""
''', tests='''\
assert strip_markers("a<fim_prefix>b<fim_middle>c") == "abc"
assert strip_markers("<fim_suffix>") == ""
'''),
        Probe(id="M1", hypothesis="cross-file use, near (< 8K tokens)", language="python",
              check=uses("to_grams", 2), prompt=repository([units, report]), tests='''\
from pantry.report import total_grams

assert abs(total_grams([("rice", 2, "kg"), ("salt", 500, "g")]) - 2500) < 1e-9
assert total_grams([]) == 0
'''),
        Probe(id="M2", hypothesis=f"cross-file use, far (>= {FAR} tokens back)", language="python",
              check=uses("haversine_km", 4),
              prompt=repository([distance, *filler(tokenizer, ["geo", "geo.legacy"], tokens=FAR), routes]), tests='''\
from geo.distance import haversine_km
from geo.routes import route_length_km

points = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
assert abs(route_length_km(points) - haversine_km(0, 0, 0, 1) - haversine_km(0, 1, 1, 1)) < 1e-9
assert route_length_km([(5.0, 5.0)]) == 0
'''),
        Probe(id="M3", hypothesis=f"input beyond the trained context (>= {BEYOND} tokens)", language="python",
              check=None, prompt=repository([*filler(tokenizer, ["billing", "fleet", "clinic"], tokens=BEYOND),
                                             ("app/main.py", "def main() -> None:\n")])),
        Probe(id="M4", hypothesis="dependency order reversed (helper file after the caller's)", language="python",
              check=uses("apply_discount", 2), prompt=repository([("shop/cart.py", '''\
"""Shopping cart totals."""

from shop.pricing import apply_discount


def cart_total(lines: list[tuple[float, int]], percent: float) -> float:
    """Total of lines given as (unit price, count), every unit discounted by percent per cent."""
''')]), suffix='''

def cart_count(lines: list[tuple[float, int]]) -> int:
    return sum(count for _, count in lines)
''' + repository([pricing]), tests='''\
from shop.cart import cart_total

assert abs(cart_total([(10.0, 2), (5.0, 1)], 10.0) - 22.5) < 1e-9
'''),
        Probe(id="M5", hypothesis="2-space indentation style", language="python", check=kept_indentation("  "),
              prompt='''\
def is_even(n):
  return n % 2 == 0


def count_by_parity(numbers):
  """(evens, odds): how many of numbers are even and how many odd."""
  evens = 0
  odds = 0
  for n in numbers:
''', tests='''\
assert count_by_parity([1, 2, 3, 4, 6]) == (3, 2)
assert count_by_parity([]) == (0, 0)
'''),
        Probe(id="M6", hypothesis="tab indentation", language="python", check=kept_indentation("\t"), prompt='''\
def clean(word):
\treturn word.strip(".,;:!?").lower()


def unique_words(text):
\t"""The distinct words of text, cleaned, in order of first appearance."""
\tseen = []
\tfor word in text.split():
''', tests='''\
assert unique_words("The cat, the hat. A cat!") == ["the", "cat", "hat", "a"]
'''),
        Probe(id="M7", hypothesis="Python 2 input (print statements)", language="python", check=python3, prompt='''\
# Print a small inventory table.
def show(rows):
    for name, count in rows:
        print "%-12s %5d" % (name, count)
    print "-" * 18


def show_total(rows):
'''),
        Probe(id="M8", hypothesis="non-ASCII identifiers and strings", language="python", check=round_trip,
              prompt='''\
# Température moyenne de chaque ville (°C).
relevés = {"Zürich": [3.5, 4.5], "São Paulo": [22.0, 25.0], "東京": [9.0, 11.0]}


def moyenne(ville: str) -> float:
    """La moyenne des relevés de la ville « ville »."""
''', tests='''\
assert moyenne("東京") == 10.0
assert moyenne("Zürich") == 4.0
'''),
        Probe(id="M9", hypothesis="C++ templates", language="cpp", check=runs, prompt='''\
#include <string>
#include <vector>

// The items of values for which keep(item) is true, in their original order.
template <typename T, typename Predicate>
std::vector<T> keep_if(const std::vector<T>& values, Predicate keep) {
''', tests='''\
int main() {
    std::vector<int> evens = keep_if(std::vector<int>{1, 2, 3, 4}, [](int x) { return x % 2 == 0; });
    std::vector<std::string> words = keep_if(std::vector<std::string>{"a", "abc", "de"},
                                             [](const std::string& w) { return w.size() > 1; });
    return evens == std::vector<int>{2, 4} && words == std::vector<std::string>{"abc", "de"} ? 0 : 1;
}
'''),
        Probe(id="S1", hypothesis="undefined name (missing import)", language="python", check=clean, prompt='''\
def hypotenuses(legs: list[tuple[float, float]]) -> list[float]:
    """The hypotenuse of each right triangle with legs (a, b), rounded to 3 decimals.

    >>> hypotenuses([(3.0, 4.0), (1.0, 1.0)])
    [5.0, 1.414]
    """
''', tests='''\
assert hypotenuses([(3.0, 4.0), (1.0, 1.0)]) == [5.0, 1.414]
assert hypotenuses([(5.0, 12.0)]) == [13.0]
'''),
        Probe(id="S2", hypothesis="return type contradicts the annotation", language="python", check=clean,
              prompt='''\
def word_count(text: str) -> str:
    """The number of whitespace-separated words in text, as a decimal string.

    >>> word_count("to be or not")
    '4'
    """
''', tests='''\
assert word_count("to be or not") == "4"
assert word_count("") == "0"
'''),
        Probe(id="S3", hypothesis="wrong answer the prompt's examples catch", language="python", check=runs,
              prompt='''\
def second_largest(values: list[int]) -> int:
    """The second largest distinct value in values (which holds at least two distinct values).

    >>> second_largest([4, 9, 9, 2])
    4
    >>> second_largest([1, 5])
    1
    """
''', tests='''\
assert second_largest([4, 9, 9, 2]) == 4
assert second_largest([1, 5]) == 1
assert second_largest([7, 7, 3, 3, 1]) == 3
'''),
        Probe(id="S4", hypothesis="C++ compile error (const map operator[])", language="cpp", check=runs, prompt='''\
#include <map>
#include <string>

// The count stored for key in counts, or 0 when counts has no entry for key.
int count_or_zero(const std::map<std::string, int>& counts, const std::string& key) {
''', tests='''\
int main() {
    std::map<std::string, int> counts{{"a", 2}};
    return count_or_zero(counts, "a") == 2 && count_or_zero(counts, "b") == 0 ? 0 : 1;
}
'''),
        Probe(id="S5", hypothesis="unbalanced brackets in a long nested expression", language="python", check=parses,
              prompt='''\
def default_settings() -> dict:
    """The service's default settings tree."""
    return {"server": {"hosts": [("alpha", [8080, 8081]), ("beta", [9090])], "limits": {"cpu": [1, 2, {"burst": [4, 8'''),
    ]


# ---- generation

def task(probe: Probe) -> Task:
    """The probe as a code_eval task: its prompt and its language's stop sequences (none for an infill probe)."""
    return Task(probe.id, probe.prompt, None, list(STOPS[probe.language]) if probe.suffix is None else [], None)


def program(probe: Probe, completion: str) -> str:
    """The program a completion forms: prompt + completion + the infill suffix (Python) or CPP_CLOSE (C++)."""
    return probe.prompt + completion + ((probe.suffix or "") if probe.language == "python" else CPP_CLOSE)


def request(tokenizer: Tokenizer, probe: Probe, configuration: Configuration) -> tuple[Request, bytes]:
    """The greedy decoder request of a probe under a configuration other than system, and the prompt bytes its
    generation repeats (the token healing drops)."""
    ids = [tokenizer.bos_token_id, *tokenizer.enc.encode_ordinary(probe.prompt)]
    healed, constraint = b"", None
    match configuration:
        case "healed":
            healed = tokenizer.token_bytes[ids[-1]]
            ids, constraint = heal(tokenizer, ids)
        case "fim" | "fim_spm":
            ids = fim_prompt(tokenizer, probe.prompt, probe.suffix or "", mode=FIM_MODES[configuration])
        case "constrained":
            constraint = syntax_constraint(probe.prompt, suffix=probe.suffix or "")
            if probe.suffix is not None:
                ids = fim_prompt(tokenizer, probe.prompt, probe.suffix, mode="psm")
    stop = None if probe.suffix is not None else StopAt(tokenizer, STOPS[probe.language], healed=healed)
    return Request(prompt=ids, max_tokens=MAX_TOKENS, stop=stop, constraint=constraint), healed


def generation(tokenizer: Tokenizer, probe: Probe, ids: list[int], *, healed: bytes, budget: int) -> Generation:
    """The Generation of a probe's generated ids (code_eval.completion: cut at the first special token and the first
    stop sequence, the healed bytes dropped) and how it ended."""
    completed = completion(tokenizer, task(probe), ids, special="stop", healed=healed)
    kept = list(itertools.takewhile(lambda t: t < tokenizer.bos_token_id, ids))
    whole = len(tokenizer.enc.decode_bytes(kept)[len(healed):].decode("utf-8", "replace"))
    end = ("stop" if len(completed) < whole else "special" if len(kept) < len(ids) else
           "budget" if len(ids) == budget else "bos")
    return Generation(completion=completed, program=program(probe, completed), tokens=len(ids), end=end)


def generate(decoder: NativeDecoder, tokenizer: Tokenizer, probe: Probe, configuration: Configuration) -> Generation:
    """A probe's generation under a configuration; system: code_eval.system_complete (verify.search over healed,
    syntax-constrained candidates), its chosen candidate."""
    if configuration != "system":
        req, healed = request(tokenizer, probe, configuration)
        ((_, ids),) = decoder.run([req])
        return generation(tokenizer, probe, ids, healed=healed, budget=MAX_TOKENS)
    _, (candidates,) = system_complete(decoder, tokenizer, [task(probe)], [probe.prompt], [MAX_TOKENS],
                                       special="stop", program=lambda t, text: program(probe, text),
                                       language=probe.language)
    healed = tokenizer.token_bytes[tokenizer.enc.encode_ordinary(probe.prompt)[-1]]
    # system_complete's budget: one more token, the healed one regenerated
    best = generation(tokenizer, probe, chosen(candidates).ids, healed=healed, budget=MAX_TOKENS + 1)
    return dataclasses.replace(best, attempts=len(candidates))


def evaluate(decoder: NativeDecoder, tokenizer: Tokenizer, probe: Probe, configuration: Configuration) -> dict:
    """A probe x configuration result: passed, the check's diagnostic, tokens generated, how generation ended (or
    "refused": a ValueError of the decoder's context guard or of the syntax constraint), attempts, completion, seconds."""
    start = time.perf_counter()
    try:
        g = generate(decoder, tokenizer, probe, configuration)
    except ValueError as error:
        outcome = {"passed": probe.check is None, "diagnostic": str(error), "tokens": 0, "end": "refused",
                   "attempts": 0, "completion": ""}
    else:
        diagnostic = "no refusal" if probe.check is None else probe.check(probe, g)
        outcome = {"passed": not diagnostic, "diagnostic": diagnostic, "tokens": g.tokens, "end": g.end,
                   "attempts": g.attempts, "completion": g.completion}
    return {"probe": probe.id, "configuration": configuration, **outcome, "seconds": time.perf_counter() - start}


def report(tag: str, step: int, probe_list: list[Probe], results: list[dict], seconds: float) -> str:
    """Markdown: the probe list, a probe x configuration table (pass/fail, end, tokens; "-": not run) and the
    diagnostics of every result that has one."""
    cells = {(r["probe"], r["configuration"]): r for r in results}
    lines = [f"# Probes of {tag} (step {step}), {seconds:.0f} s", "", "| probe | hypothesis | " +
             " | ".join(CONFIGURATIONS) + " |", "|---|---|" + "---|" * len(CONFIGURATIONS)]
    for p in probe_list:
        row = []
        for c in CONFIGURATIONS:
            r = cells.get((p.id, c))
            row.append("-" if r is None else
                       f"{'pass' if r['passed'] else 'FAIL'} {r['end']} {r['tokens']}" +
                       (f" x{r['attempts']}" if c == "system" else ""))
        lines.append(f"| {p.id} | {p.hypothesis} | " + " | ".join(row) + " |")
    passed = sum(r["passed"] for r in results)
    lines += ["", f"{passed}/{len(results)} passed.", "", "## Diagnostics", ""]
    lines += [f"- {r['probe']} {r['configuration']}: {' '.join(r['diagnostic'].split())}" for r in results
              if r["diagnostic"]]
    return "\n".join(lines) + "\n"


def run(model: GPT, tokenizer: Tokenizer) -> tuple[list[Probe], list[dict], float]:
    """Every probe under each of its configurations: (the probes, their results, seconds)."""
    probe_list = probes(tokenizer)
    # the longest request (a FIM prompt is the longest: its sentinels; healing shortens the prompt by the token it adds
    # to the budget), within the model's context: a longer request is refused by the decoder's context guard either way
    longest = max(len(fim_prompt(tokenizer, p.prompt, p.suffix or "", mode="psm")) for p in probe_list)
    decoder = NativeDecoder(model, bos_token_id=tokenizer.bos_token_id,
                            max_len=min(longest + MAX_TOKENS - 1, model.config.sequence_len))
    start = time.perf_counter()
    results = [evaluate(decoder, tokenizer, p, c) for p in probe_list for c in configurations(p)]
    return probe_list, results, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--step", type=int, default=None, help="a snapshot's step (default: the newest checkpoint)")
    args = parser.parse_args()
    tag = args.model_tag
    tokenizer = get_tokenizer()
    model, step = load_model(tag, step=args.step, device=torch.device("cuda"))
    probe_list, results, seconds = run(model, tokenizer)
    name = checkpoint_name(tag, step=args.step)
    (PROBES / f"{name}.json").write_text(json.dumps({"schema": SCHEMA, "run": tag, "step": step, "seconds": seconds,
                                                     "results": results}, allow_nan=False), encoding="utf-8")
    text = report(tag, step, probe_list, results, seconds)
    (PROBES / f"{name}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
