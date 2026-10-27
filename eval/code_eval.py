"""
Code generation of a base model, scored by each benchmark's official evaluator in a fresh process: EvalPlus
(arXiv 2305.01210), MultiPL-E C++ (arXiv 2208.08227), HumanEval-X C++ references (arXiv 2303.17568) and HumanEval
infilling (arXiv 2207.14255), in greedy, sampled (arXiv 2107.03374 §2.1) and filtered modes.
"""

import codecs
import functools
import hashlib
import itertools
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Literal, NamedTuple

import human_eval_infilling
import pyarrow.parquet as pq
import torch
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.provider.utility import EOS, extra_eos_for_direct_completion
from evalplus.sanitize import sanitize
from huggingface_hub import hf_hub_download
from human_eval_infilling.data import read_problems

from corpus.tokenizer import Tokenizer
from eval.cache import VerdictCache
from eval.verify import (
    TOP_P,
    Candidate,
    Search,
    chosen,
    compiler,
    search,
    solution_diagnostics,
    timed,
    timed_slots,
    timed_workers,
    vetted_examples,
)
from inference.decoder import NativeDecoder, Request, fim_prompt, heal_tokens
from inference.syntax_constraint import SyntaxConstraint
from metrics import completion_scores, latency, parse_outcome, parse_shares, pass_at_k, summary
from model.modeling import CausalLM

# HF dataset revisions: MultiPL-E's, its generator's pin (multipl_e/completions.py DATASET_REVISION)
MULTIPL_E = {"repo_id": "nuprl/MultiPL-E", "revision": "3a9e8d226c127392ce81dca29d09f33f0ce3247d"}
HUMANEVAL_X = {"repo_id": "zai-org/humaneval-x", "revision": "62c78627f3072a1454fa0cb0184737cafe5e4198"}
EVALPLUS = {"humaneval_plus": ("humaneval", get_human_eval_plus), "mbpp_plus": ("mbpp", get_mbpp_plus)}
MULTIPL_E_BENCHMARKS = ("humaneval_cpp", "mbpp_cpp")
INFILLING = ("single-line", "multi-line", "random-span")
FIM_MODES = ("psm", "spm")  # the two orders the packer trains (arXiv 2207.14255 App. D)
FILTERED_MODE = "spm"
# Sampled mode: Codex (arXiv 2107.03374 §2.1 unbiased pass@k; §3.3, Fig. 5: T = 0.2 for pass@1; top p: verify.TOP_P);
# MultiPL-E's pass@1 (arXiv 2208.08227: T = 0.2; completions.py --top-p 0.95); StarCoder2 (arXiv 2402.19173 §7.1.2)
N_SAMPLE, TEMPERATURE = 20, 0.2
# EvalPlus's most new tokens of a completion (evalplus/provider/base.py:13 max_new_tokens)
EVALPLUS_MAX_NEW_TOKENS = 768


class Task(NamedTuple):
    task_id: str
    prompt: str
    canonical_solution: str | None
    stopwords: list[str] | None
    entry_point: str | None  # EvalPlus: the entry point; MultiPL-E: its tests


def tasks() -> dict[str, list[Task]]:
    """{benchmark: [Task]}; humaneval_x_cpp holds reference solutions only."""
    out = {}
    for name, (dataset, load) in EVALPLUS.items():
        stopwords = EOS + extra_eos_for_direct_completion(dataset)
        out[name] = [Task(task_id, t["prompt"].strip() + "\n", t["canonical_solution"], stopwords, t["entry_point"])
                     for task_id, t in load().items()]
    for name in MULTIPL_E_BENCHMARKS:
        file = f"{name.replace('_', '-')}/test-00000-of-00001.parquet"
        rows = pq.read_table(hf_hub_download(filename=file, repo_type="dataset", **MULTIPL_E)).to_pylist()
        out[name] = [Task(r["name"], r["prompt"], None, r["stop_tokens"], r["tests"]) for r in rows]
    with open(hf_hub_download(filename="data/cpp/data/humaneval.jsonl", repo_type="dataset", **HUMANEVAL_X),
              encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    out["humaneval_x_cpp"] = [Task(r["task_id"], r["prompt"], r["canonical_solution"], None, None) for r in rows]
    return out


class KeywordsStoppingCriteria:
    """NativeDecoder stopping criteria: true once the generated text holds a stop sequence. healed: the bytes
    decoder.heal_tokens dropped, which the first tokens re-emit; they belong to the prompt and are not searched."""

    def __init__(self, tokenizer: Tokenizer, stopwords: list[str], *, healed: bytes):
        self.token_bytes, self.stopwords, self.longest = tokenizer.token_bytes, stopwords, max(map(len, stopwords))
        self.utf8, self.text, self.n, self.searched = codecs.getincrementaldecoder("utf-8")("replace"), "", 0, 0
        self.skip = len(healed)

    def __call__(self, ids: list[int]) -> bool:
        data = b"".join(self.token_bytes[t] for t in ids[self.n:])
        cut = min(self.skip, len(data))
        self.skip -= cut
        self.text += self.utf8.decode(data[cut:])
        self.n, start, self.searched = len(ids), max(0, self.searched - self.longest + 1), len(self.text)
        return any(s in self.text[start:] for s in self.stopwords)


def fim_lines(middle: str, lines: int) -> str:
    """The middle cut to its first `lines` lines, each ending in "\\n" as the reference middles do."""
    parts = middle.split("\n")
    if len(parts) > lines:
        return "\n".join(parts[:lines]) + "\n"
    return middle if middle == "" or middle.endswith("\n") else middle + "\n"


def completion(tokenizer: Tokenizer, task: Task, ids: list[int], *, special: Literal["skip", "stop"],
               healed: bytes) -> str:
    """Completion text of generated ids, cut before the first stop sequence, healed bytes dropped. Special tokens
    (ids >= BOS) are "skip"ped (EvalPlus) or "stop" the completion (MultiPL-E automodel.py:99-101)."""
    bos = tokenizer.bos_token_id
    ids = [t for t in ids if t < bos] if special == "skip" else list(itertools.takewhile(lambda t: t < bos, ids))
    text = tokenizer.enc.decode_bytes(ids)[len(healed):].decode("utf-8", "replace")  # as enc.decode
    return text[:min([j for j in (text.find(s) for s in task.stopwords) if j >= 0], default=len(text))]


def decoder(model: CausalLM, tokenizer: Tokenizer, *, max_length: int, prompt_store: int) -> NativeDecoder:
    """A NativeDecoder for requests of at most max_length positions, keeping up to prompt_store prompt positions,
    sized by free device memory after empty_cache."""
    torch.cuda.empty_cache()
    return NativeDecoder(model, bos_token_id=tokenizer.bos_token_id, max_length=max_length, prompt_store=prompt_store)


def generate_groups(generator: NativeDecoder, groups: list[list[Request]],
                    done: Callable[[int, list[list[int]], dict], None]) -> None:
    """One generator.generate call over every group's independent requests; done(g, ids, latency) once group g ends."""
    requests = [r for group in groups for r in group]
    bounds = [0, *itertools.accumulate(map(len, groups))]
    owner = [g for g, group in enumerate(groups) for _ in group]
    out, left = [[] for _ in requests], list(map(len, groups))
    for j, ids in generator.generate(requests):
        out[j], g = ids, owner[j]
        left[g] -= 1
        if not left[g]:
            span = range(bounds[g], bounds[g + 1])
            starts, firsts, ends = zip(*(generator.timings[k] for k in span), strict=True)
            done(g, out[bounds[g]:bounds[g + 1]],
                 latency(starts, firsts, ends, [len(out[k]) for k in span], duration=max(ends) - min(starts)))


def pass_scores(passed: list[list[bool]]) -> dict[str, dict]:
    """metrics.summary of each problem's pass@k estimate, k 1 and 10 when every problem has at least 10 samples."""
    return {f"pass@{k}": summary(pass_at_k(passed, k)) for k in (1, 10) if min(map(len, passed)) >= k}


class Filtered(NamedTuple):
    """A filtered-mode benchmark (filtered): its verify.Search and the text of a candidate's ids."""
    search: Search
    text: Callable[[int, list[int]], str]

    def record(self, tried: list[list[Candidate]], timing: dict) -> tuple[list[str], dict]:
        """Each task's chosen text and the search record of verify.search's candidates and latency."""
        return [self.text(i, chosen(c).ids) for i, c in enumerate(tried)], {
            "attempts": [len(c) for c in tried], "accepted": [c[-1].accepted for c in tried], "latency": timing}


def filtered(request: Callable[[int], Request], text: Callable[[int, list[int]], str],
             check: Callable[[int, str], tuple[str, list[str]]], *, found: list[list[tuple[str, str]]],
             language: Literal["python", "cpp"]) -> Filtered:
    """The filtered mode of tasks 0..len(found)-1. check(i, text) returns the checked program and its known
    diagnostics; found[i]: task i's vetted examples."""
    return Filtered(Search(request=request, check=lambda i, ids: (found[i], *check(i, text(i, ids))),
                           language=language, count=len(found)), text)


def filtered_completions(tokenizer: Tokenizer, items: list[Task], prompts: list[list[int]], budgets: list[int], *,
                         special: Literal["skip", "stop"], check: Callable[[Task, str], tuple[str, list[str]]],
                         found: list[list[tuple[str, str]]], language: Literal["python", "cpp"]) -> Filtered:
    """filtered of left-to-right prompts, healed (one more token of budget; Python under the syntax constraint);
    found: each task's vetted examples (vetted)."""
    dropped = [heal_tokens(tokenizer, p)[2] for p in prompts]

    def request(i: int) -> Request:
        input_ids, constraint, _ = heal_tokens(tokenizer, prompts[i])  # a fresh constraint per candidate
        if language == "python":
            constraint = SyntaxConstraint(tokenizer, tokenizer.enc.decode_bytes(input_ids[1:]), suffix=b"",
                                          healing=constraint)
        return Request(input_ids=input_ids, max_new_tokens=budgets[i] + 1, constraint=constraint,
                       stopping_criteria=KeywordsStoppingCriteria(tokenizer, items[i].stopwords, healed=dropped[i]))

    return filtered(request, lambda i, ids: completion(tokenizer, items[i], ids, special=special, healed=dropped[i]),
                    lambda i, text: check(items[i], text), found=found, language=language)


def filtered_middles(tokenizer: Tokenizer, problems: list[dict], prompts: list[list[int]], *, max_new_tokens: int,
                     found: list[list[tuple[str, str]]]) -> Filtered:
    """filtered of FIM prompts under the syntax constraint of prefix and suffix, checked on the prefix's vetted
    examples (vetted)."""
    return filtered(lambda i: Request(input_ids=prompts[i], max_new_tokens=max_new_tokens,
                                      constraint=SyntaxConstraint(tokenizer, problems[i]["prompt"].encode(),
                                                                  suffix=problems[i]["suffix"].encode())),
                    lambda i, ids: tokenizer.enc.decode(ids),
                    lambda i, middle: (problems[i]["prompt"] + middle + problems[i]["suffix"], []),
                    found=found, language="python")


def sanitizer_init() -> None:
    """A sanitize worker: SyntaxWarnings ignored, and EvalPlus's syntax_check counting a ValueError of the parser as
    a syntax error: CPython 3.12.3's compile() raises it for an f-string debug expression in a format spec, f"{x:{y=}}"
    (python/cpython#121130, fixed in 3.12.5), which this interpreter therefore cannot run."""
    warnings.filterwarnings("ignore", "", SyntaxWarning)
    module = sys.modules[sanitize.__module__]  # evalplus/sanitize.py (the package's `sanitize` is the function)
    check = module.syntax_check

    def syntax_check(code: str, verbose: bool = False) -> bool:
        try:
            return check(code, verbose)
        except ValueError:
            return False

    module.syntax_check = syntax_check


def sanitizers() -> ProcessPoolExecutor:
    """EvalPlus sanitize workers (sanitizer_init), one per CPU, from a forkserver, which imports the caller's main
    module: it must not evaluate at import time. The caller must shut the pool down: an open pool blocks a spawned
    process's exit."""
    return ProcessPoolExecutor(os.cpu_count(), mp_context=multiprocessing.get_context("forkserver"),
                               initializer=sanitizer_init)


def evalplus_source(task: Task, text: str) -> str:
    """The program EvalPlus sanitizes: prompt + completion, tabs replaced by four spaces."""
    return task.prompt + text.replace("\t", "    ")


def evalplus_check(pool: ProcessPoolExecutor, task: Task, text: str) -> tuple[str, list[str]]:
    """(solution, verify.solution_diagnostics) of a completion text (evalplus_source), sanitized by `pool`."""
    solution = pool.submit(sanitize, evalplus_source(task, text), task.entry_point).result()
    return solution, solution_diagnostics(solution, task.entry_point)


def multipl_e_check(task: Task, text: str) -> tuple[str, list[str]]:
    """(program, []) of a completion: prompt + completion + the closing "}" its stop token cut."""
    return task.prompt + text + "\n}", []


# `python -m evalplus.evaluate` in a fresh isolated process, one worker per timed-check core (argv[4], verify.timed),
# ended by os._exit: evaluate.py's stucking_checker thread would otherwise hold the exit up to 20 s
EVALPLUS_EVALUATE = """
import os, sys
from evalplus.evaluate import evaluate
evaluate(dataset=sys.argv[1], samples=sys.argv[2], output_file=sys.argv[3], parallel=int(sys.argv[4]))
sys.stdout.flush()
os._exit(0)
"""


def evalplus(pool: ProcessPoolExecutor, name: str, items: list[Task], samples: list[list[str]]) -> dict:
    """EvalPlus's evaluation of each task's completions: per-sample statuses and scores (base and plus pass@k)."""
    dataset = EVALPLUS[name][0]
    pairs = [(t, s) for t, task_samples in zip(items, samples, strict=True) for s in task_samples]
    with tempfile.TemporaryDirectory() as tmp:
        path, scored = Path(tmp, "samples.jsonl"), Path(tmp, "results.json")
        solutions = pool.map(sanitize, [evalplus_source(t, s) for t, s in pairs], [t.entry_point for t, _ in pairs])
        with path.open("w", encoding="utf-8") as f:
            for (t, _), solution in zip(pairs, solutions, strict=True):
                f.write(json.dumps({"task_id": t.task_id, "solution": solution}) + "\n")
        with timed_slots(timed_workers()):
            subprocess.run(timed([sys.executable, "-I", "-c", EVALPLUS_EVALUATE, dataset, path, scored,
                                  str(timed_workers())]), cwd=tmp, check=True)
        result = json.loads(scored.read_text(encoding="utf-8"))
    evaluated = [result["eval"][t.task_id] for t in items]
    base_status = [[v["base_status"] for v in vs] for vs in evaluated]
    plus_status = [[v["plus_status"] for v in vs] for vs in evaluated]
    base, plus = evalplus_passed(base_status, plus_status)
    return {"scores": {"base": pass_scores(base), "plus": pass_scores(plus)},
            "ids": [t.task_id for t in items], "base_status": base_status, "plus_status": plus_status}


def evalplus_passed(base_status: list[list[str]], plus_status: list[list[str]]) -> tuple[list[list[bool]], ...]:
    """Each sample's base and plus outcome (evalplus/evaluate.py's pass@k outcomes: base passes on base_status "pass",
    plus on base_status == plus_status == "pass")."""
    return ([[b == "pass" for b in bs] for bs in base_status],
            [[b == p == "pass" for b, p in zip(bs, ps, strict=True)]
             for bs, ps in zip(base_status, plus_status, strict=True)])


@functools.cache
def multipl_e_cache() -> VerdictCache:
    """The VerdictCache of MultiPL-E's judge: its sources (main.py, containerized_eval.py, eval_cpp.py,
    safe_subprocess) and its compiler (verify.compiler's digest)."""
    src = Path(os.environ["MULTIPL_E"], "evaluation", "src")
    sources = [(src / f).read_text(encoding="utf-8") for f in ("main.py", "containerized_eval.py", "eval_cpp.py",
                                                                 "safe_subprocess/__init__.py")]
    return VerdictCache("multipl-e", json.dumps([sources, compiler().name]))


def judged(result: dict) -> bool:
    """Whether a MultiPL-E result is the program's own: not a time-out (status Timeout, or exit code -1 for a killed
    compile), not SIGKILL, and a failed compile only by g++'s error exit (1)."""
    if result["status"] == "Timeout" or result["exit_code"] in (-1, -9):
        return False
    return result["status"] != "SyntaxError" or result["exit_code"] == 1


def multipl_e(items: list[Task], samples: list[list[str]]) -> dict:
    """MultiPL-E's evaluation of each problem's completions: per-completion statuses and scores (pass@k of status "OK",
    which eval_cpp.py sets only for a program that ran and exited 0, as pass_k.py counts). One main.py --file process
    per problem with completions not in VerdictCache (keyed by main.py's program), one judge thread each, in a timed
    slot; its `g++` verify.compiler()'s (first on PATH); no -I (the scripts import siblings)."""
    main = Path(os.environ["MULTIPL_E"], "evaluation", "src", "main.py")
    cache = multipl_e_cache()
    programs = [[t.prompt + s + "\n" + t.entry_point for s in task_samples]  # main.py cached_eval_script's
                for t, task_samples in zip(items, samples, strict=True)]
    stored = cache.get([p for ps in programs for p in ps])
    bounds = [0, *itertools.accumulate(map(len, programs))]
    results = [stored[bounds[j]:bounds[j + 1]] for j in range(len(items))]
    with tempfile.TemporaryDirectory() as tmp:
        files = {}
        for t, task_samples, rs in zip(items, samples, results, strict=True):
            todo = [s for s, r in zip(task_samples, rs, strict=True) if r is None]
            if todo:
                files[t.task_id] = Path(tmp, f"{t.task_id}.json")
                files[t.task_id].write_text(json.dumps({"name": t.task_id, "language": "cpp", "prompt": t.prompt,
                                                        "tests": t.entry_point, "completions": todo}),
                                            encoding="utf-8")
        env = os.environ | {"PATH": f"{compiler()}{os.pathsep}{os.environ['PATH']}"}

        def run(file: Path) -> None:
            with timed_slots():
                subprocess.run(timed([sys.executable, main, "--file", file, "--output-dir", tmp, "--max-workers", "1"]),
                               cwd=tmp, env=env, check=True)

        with ThreadPoolExecutor(timed_workers()) as pool:
            list(pool.map(run, files.values()))
        new = []
        for t, ps, rs in zip(items, programs, results, strict=True):
            if t.task_id in files:
                judge = iter(json.loads(Path(tmp, f"{t.task_id}.results.json").read_text(encoding="utf-8"))["results"])
                for j, r in enumerate(rs):
                    if r is None:
                        rs[j] = {k: v for k, v in next(judge).items() if k in ("status", "exit_code")}
                        if judged(rs[j]):
                            new.append((ps[j], rs[j]))
        cache.put(new)
    status = [[r["status"] for r in rs] for rs in results]
    passed = [[s == "OK" for s in ss] for ss in status]  # eval_cpp.py: "OK" iff the program ran and exited 0
    return {"scores": pass_scores(passed), "ids": [t.task_id for t in items], "status": status}


# human_eval_infilling check_correctness (3 s) in a fork pool of argv[4] workers, in a fresh isolated process on the
# timed-check cores. The CLI's modules are imported first: a forked sample's run-time imports depend on the loaded
# modules (reliability_guard).
INFILLING_EVALUATE = """
import json, multiprocessing, os, sys
from concurrent.futures import ProcessPoolExecutor
import fire
import human_eval_infilling.evaluation
from human_eval_infilling.data import read_problems
from human_eval_infilling.execution import check_correctness
problems = read_problems(sys.argv[1])
with open(sys.argv[2], encoding="utf-8") as f:
    samples = json.load(f)
def check(sample):
    return check_correctness(problems[sample["task_id"]], sample["completion"], 3.0)["result"]
with ProcessPoolExecutor(int(sys.argv[4]), mp_context=multiprocessing.get_context("fork")) as pool:
    result = list(pool.map(check, samples))
with open(sys.argv[3], "w", encoding="utf-8") as f:
    json.dump(result, f)
"""


@functools.cache
def infilling_cache() -> VerdictCache:
    """The VerdictCache of human_eval_infilling's check_correctness as INFILLING_EVALUATE runs it. The package is a
    checkout on sys.path, not a distribution (cache.environment cannot see it), so its modules and the problem files
    read_problems reads (<package>/../data) are keyed by their SHA-256."""
    package = Path(human_eval_infilling.__file__).parent
    files = sorted([*package.glob("*.py"), *package.parent.glob("data/*.jsonl.gz")])
    return VerdictCache("infilling", json.dumps([INFILLING_EVALUATE,
                                                 {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}]))


def infilling(name: str, problems: list[dict], middles: list[str], record: dict) -> dict:
    """Official evaluation of one middle per problem (results not "timed out" kept in VerdictCache): per-problem
    results and scores against the reference middle (pass@1 among them), metrics.summary clustered by HumanEval problem
    (arXiv 2411.00640); then the given record (latency, search record)."""
    samples = [{"task_id": p["task_id"], "completion": middle} for p, middle in zip(problems, middles, strict=True)]
    keys = [json.dumps([name, s["task_id"], s["completion"]]) for s in samples]
    cache = infilling_cache()
    result = cache.get(keys)
    todo = [j for j, r in enumerate(result) if r is None]
    if todo:
        with tempfile.TemporaryDirectory() as tmp:
            path, scored = Path(tmp, "samples.json"), Path(tmp, "results.json")
            path.write_text(json.dumps([samples[j] for j in todo]), encoding="utf-8")
            with timed_slots(timed_workers()):
                subprocess.run(timed([sys.executable, "-I", "-c", INFILLING_EVALUATE, name, path, scored,
                                      str(timed_workers())]), cwd=tmp, check=True)
            for j, r in zip(todo, json.loads(scored.read_text(encoding="utf-8")), strict=True):
                result[j] = r
        cache.put([(keys[j], result[j]) for j in todo if result[j] != "timed out"])
    passed = [r == "passed" for r in result]
    clusters = ["/".join(p["task_id"].split("/")[1:3]) for p in problems]
    scores = completion_scores(middles, [p["canonical_solution"] for p in problems]) | {"pass@1": passed}
    return {"scores": {k: summary(v, clusters) for k, v in scores.items()},
            "ids": [s["task_id"] for s in samples], "completions": middles, "result": result} | record


class CodegenRun(NamedTuple):
    """An EvalPlus or MultiPL-E benchmark's prompts and evaluator (codegen_runs)."""
    items: list[Task]
    prompts: list[list[int]]  # BOS and the prompt's tokens
    budgets: list[int]  # max_new_tokens per task
    special: Literal["skip", "stop"]
    check: Callable[[Task, str], tuple[str, list[str]]]
    language: Literal["python", "cpp"]
    score: Callable[[list[Task], list[list[str]]], dict]


def codegen_scoring(pool: ProcessPoolExecutor, name: str) -> tuple[Callable[[Task, str], tuple[str, list[str]]],
                                                                  Literal["python", "cpp"],
                                                                  Callable[[list[Task], list[list[str]]], dict]]:
    """An EvalPlus or MultiPL-E benchmark's (check, language, score) of CodegenRun."""
    if name in EVALPLUS:
        return functools.partial(evalplus_check, pool), "python", functools.partial(evalplus, pool, name)
    return multipl_e_check, "cpp", multipl_e


def codegen_runs(tokenizer: Tokenizer, benchmarks: dict[str, list[Task]],
                 pool: ProcessPoolExecutor) -> dict[str, CodegenRun]:
    """The CodegenRun of each EvalPlus and MultiPL-E benchmark."""
    bos = tokenizer.bos_token_id
    runs = {}
    for name in EVALPLUS:
        items = benchmarks[name]
        runs[name] = CodegenRun(items, [[bos, *tokenizer.enc.encode_ordinary(t.prompt)] for t in items],
                                [EVALPLUS_MAX_NEW_TOKENS] * len(items), "skip", *codegen_scoring(pool, name))
    for name in MULTIPL_E_BENCHMARKS:
        items = benchmarks[name]
        prompts = [[bos, *tokenizer.enc.encode_ordinary(t.prompt.strip())] for t in items]
        # At most 1024 tokens of prompt and completion, BOS included (automodel.py:67); the longest prompt is 600 tokens
        # under the corpus tokenizer
        runs[name] = CodegenRun(items, prompts, [1024 - len(p) for p in prompts], "stop",
                                *codegen_scoring(pool, name))
    return runs


def codegen_record(tokenizer: Tokenizer, run: CodegenRun, mode: Literal["greedy", "sampled"],
                   generated: list[list[int]], timing: dict) -> dict:
    """generation's record of a benchmark's greedy or sampled ids (a task's samples in seed order)."""
    n = 1 if mode == "greedy" else N_SAMPLE
    texts = [completion(tokenizer, run.items[j // n], ids, special=run.special, healed=b"")
             for j, ids in enumerate(generated)]
    return codegen_scores(run.score, run.items, mode, [texts[n * i:n * (i + 1)] for i in range(len(run.items))],
                          timing)


def codegen_scores(score: Callable[[list[Task], list[list[str]]], dict], items: list[Task],
                   mode: Literal["greedy", "sampled"], samples: list[list[str]], timing: dict) -> dict:
    """codegen_record of each task's completion texts: the evaluator's scores, the completions (greedy: one per task;
    sampled: each task's N_SAMPLE in seed order) and latency."""
    scores = score(items, samples)
    if mode == "greedy":
        return scores | {"completions": [text for text, in samples], "latency": timing}
    return scores | {"n_sample": N_SAMPLE, "completions": samples, "latency": timing}


def generation(model: CausalLM, tokenizer: Tokenizer, benchmarks: dict[str, list[Task]], *,
               pool: ProcessPoolExecutor, scorers: ThreadPoolExecutor) -> dict[str, dict[str, Future[dict]]]:
    """Futures of {benchmark: {"greedy", "sampled"}} records of the EvalPlus and MultiPL-E benchmarks. One generate
    call runs every greedy request, then the samples; sample k of problem i has seed N_SAMPLE i + k."""
    runs = codegen_runs(tokenizer, benchmarks, pool)
    keys = [(name, mode) for mode in ("greedy", "sampled") for name in runs]
    groups = []
    for name, mode in keys:
        run = runs[name]
        rows = list(zip(run.items, run.prompts, run.budgets, strict=True))
        if mode == "greedy":
            groups.append([Request(input_ids=p, max_new_tokens=m,
                                   stopping_criteria=KeywordsStoppingCriteria(tokenizer, t.stopwords, healed=b""))
                           for t, p, m in rows])
        else:
            groups.append([Request(input_ids=p, max_new_tokens=m,
                                   stopping_criteria=KeywordsStoppingCriteria(tokenizer, t.stopwords, healed=b""),
                                   temperature=TEMPERATURE, top_p=TOP_P, seed=N_SAMPLE * i + k)
                           for i, (t, p, m) in enumerate(rows) for k in range(N_SAMPLE)])
    generator = decoder(model, tokenizer,
                        max_length=max(len(r.input_ids) + r.max_new_tokens - 1 for group in groups for r in group),
                        prompt_store=sum(len(p) for run in runs.values() for p in run.prompts))
    results = {name: dict.fromkeys(("greedy", "sampled")) for name in runs}

    def done(g: int, generated: list[list[int]], timing: dict) -> None:
        name, mode = keys[g]
        results[name][mode] = scorers.submit(codegen_record, tokenizer, runs[name], mode, generated, timing)

    generate_groups(generator, groups, done)
    return results


def filtered_record(score: Callable[[list[Task], list[list[str]]], dict], items: list[Task], completions: list[str],
                    record: dict) -> dict:
    """filtered_generation's record: evaluator scores, chosen completions and their search record."""
    return score(items, [[c] for c in completions]) | {"completions": completions} | record


def vetted(benchmarks: dict[str, list[Task]]) -> dict[str, list[list[tuple[str, str]]]]:
    """verify.vetted_examples of each EvalPlus benchmark's tasks (prompt + canonical solution) and each INFILLING
    benchmark's problems (prefix + reference middle + suffix), in task order; none for the MultiPL-E tasks."""
    return ({name: vetted_examples([(t.prompt, evalplus_source(t, t.canonical_solution)) for t in benchmarks[name]])
             for name in EVALPLUS}
            | {name: [[] for _ in benchmarks[name]] for name in MULTIPL_E_BENCHMARKS}
            | {name: vetted_examples([(p["prompt"], p["prompt"] + p["canonical_solution"] + p["suffix"])
                                      for p in read_problems(name).values()]) for name in INFILLING})


def parse_record(pool: ProcessPoolExecutor, tokenizer: Tokenizer, name: str, items: list[Task], dropped: list[bytes],
                 generated: list[list[int]], timing: dict) -> dict:
    """parse_generation's record of generated ids (parse_scores)."""
    return parse_scores(pool, name, items, [completion(tokenizer, t, ids, special="skip", healed=d)
                                            for t, ids, d in zip(items, generated, dropped, strict=True)], timing)


def parse_scores(pool: ProcessPoolExecutor, name: str, items: list[Task], completions: list[str],
                 timing: dict) -> dict:
    """parse_record of completion texts: parse_outcome shares, EvalPlus's scores, outcomes, completions and latency."""
    outcomes = [parse_outcome(t.prompt + c) for t, c in zip(items, completions, strict=True)]
    return (parse_shares(outcomes) | evalplus(pool, name, items, [[c] for c in completions])
            | {"outcomes": outcomes, "completions": completions, "latency": timing})


def parse_generation(model: CausalLM, tokenizer: Tokenizer, benchmarks: dict[str, list[Task]], *,
                     pool: ProcessPoolExecutor, scorers: ThreadPoolExecutor) -> dict[str, Future[dict]]:
    """Futures of HumanEval+'s and MBPP+'s parse records: one greedy unconstrained completion of each healed prompt,
    TokEval's protocol (BOS, healing, greedy; arXiv 2608.18062 App. C.1)."""
    bos = tokenizer.bos_token_id
    healed = {name: [heal_tokens(tokenizer, [bos, *tokenizer.enc.encode_ordinary(t.prompt)]) for t in benchmarks[name]]
              for name in EVALPLUS}
    budget = EVALPLUS_MAX_NEW_TOKENS + 1  # and a token for the bytes heal_tokens drops
    generator = decoder(model, tokenizer,
                        max_length=max(len(h) + budget - 1 for prompts in healed.values() for h, _, _ in prompts),
                        prompt_store=0)  # each healed prompt is requested once
    names = list(healed)
    results = dict.fromkeys(names)

    def done(g: int, generated: list[list[int]], timing: dict) -> None:
        name = names[g]
        results[name] = scorers.submit(parse_record, pool, tokenizer, name, benchmarks[name],
                                       [d for _, _, d in healed[name]], generated, timing)

    generate_groups(generator, [[Request(input_ids=h, max_new_tokens=budget, constraint=c,
                                         stopping_criteria=KeywordsStoppingCriteria(tokenizer, t.stopwords, healed=d))
                                 for t, (h, c, d) in zip(benchmarks[name], healed[name], strict=True)]
                                for name in names], done)
    return results


class FimRun(NamedTuple):
    """An INFILLING benchmark's problems and prompts (fim_runs)."""
    problems: list[dict]
    max_new_tokens: int  # the longest reference middle in tokens
    prompts: dict[str, list[list[int]]]
    lines: list[int] | None  # the reference middles' line counts (fim_lines): single-line 1, multi-line each one's


def fim_runs(tokenizer: Tokenizer) -> dict[str, FimRun]:
    """The FimRun of each INFILLING benchmark; random-span has no line counts."""
    runs = {}
    for name in INFILLING:
        problems = list(read_problems(name).values())
        runs[name] = FimRun(problems,
                            max(len(tokenizer.enc.encode_ordinary(p["canonical_solution"])) for p in problems),
                            {mode: [fim_prompt(tokenizer, p["prompt"], p["suffix"], mode=mode) for p in problems]
                             for mode in FIM_MODES},
                            reference_lines(name, problems))
    return runs


def reference_lines(name: str, problems: list[dict]) -> list[int] | None:
    """An INFILLING benchmark's FimRun.lines: single-line 1, multi-line each reference middle's, random-span none."""
    return {"single-line": [1] * len(problems),
            "multi-line": [p["canonical_solution"].count("\n") for p in problems]}.get(name)


def fim_generation(model: CausalLM, tokenizer: Tokenizer, *,
                   scorers: ThreadPoolExecutor) -> dict[str, dict[str, Future[dict]]]:
    """Futures of {"<benchmark>/<mode>": {"greedy", "greedy_reference_lines"}} infilling records in both FIM orders;
    greedy_reference_lines (single- and multi-line) cuts the same middles by fim_lines. One generate call runs all."""
    runs = fim_runs(tokenizer)
    keys = [(name, mode) for name in INFILLING for mode in FIM_MODES]
    groups = [[Request(input_ids=p, max_new_tokens=runs[name].max_new_tokens) for p in runs[name].prompts[mode]]
              for name, mode in keys]
    generator = decoder(model, tokenizer,
                        max_length=max(len(r.input_ids) + r.max_new_tokens - 1 for group in groups for r in group),
                        prompt_store=0)
    results = {f"{name}/{mode}": {} for name, mode in keys}

    def done(g: int, generated: list[list[int]], timing: dict) -> None:
        name, mode = keys[g]
        run, record = runs[name], results[f"{name}/{mode}"]
        middles = [tokenizer.enc.decode(ids) for ids in generated]
        record["greedy"] = scorers.submit(infilling, name, run.problems, middles, {"latency": timing})
        if run.lines is not None:
            record["greedy_reference_lines"] = scorers.submit(
                infilling, name, run.problems,
                [fim_lines(m, k) for m, k in zip(middles, run.lines, strict=True)], {})

    generate_groups(generator, groups, done)
    return results


def filtered_generation(model: CausalLM, tokenizer: Tokenizer, benchmarks: dict[str, list[Task]], *,
                        pool: ProcessPoolExecutor, scorers: ThreadPoolExecutor,
                        found: dict[str, list[list[tuple[str, str]]]]
                        ) -> tuple[dict[str, Future[dict]], dict[str, Future[dict]]]:
    """Futures of each EvalPlus and MultiPL-E benchmark's filtered record and of {"<benchmark>/<FILTERED_MODE>":
    infilling record of filtered_middles' middles}: one verify.search over all, each record scored once its search
    ends. found: vetted's examples."""
    runs, fims = codegen_runs(tokenizer, benchmarks, pool), fim_runs(tokenizer)
    searches = ({name: filtered_completions(tokenizer, run.items, run.prompts, run.budgets, special=run.special,
                                            check=run.check, language=run.language, found=found[name])
                 for name, run in runs.items()}
                | {f"{name}/{FILTERED_MODE}": filtered_middles(tokenizer, run.problems, run.prompts[FILTERED_MODE],
                                                               max_new_tokens=run.max_new_tokens, found=found[name])
                   for name, run in fims.items()})
    # every request fits the decoder's max_length, the longest of either kind (a slot count changes no tokens)
    lengths = ([(p, m) for run in runs.values() for p, m in zip(run.prompts, run.budgets, strict=True)]
               + [(p, run.max_new_tokens) for run in fims.values() for p in run.prompts[FILTERED_MODE]])
    generator = decoder(model, tokenizer, max_length=max(len(p) + m - 1 for p, m in lengths),
                        prompt_store=sum(len(p) for p, _ in lengths))
    keys, results = list(searches), {}

    def done(s: int, tried: list[list[Candidate]], timing: dict) -> None:
        key = keys[s]
        texts, record = searches[key].record(tried, timing)
        if key in runs:
            results[key] = scorers.submit(filtered_record, runs[key].score, runs[key].items, texts, record)
        else:
            name = key.removesuffix(f"/{FILTERED_MODE}")
            results[key] = scorers.submit(infilling, name, fims[name].problems, texts, record)

    search(generator, [searches[key].search for key in keys], done)
    return {key: results[key] for key in runs}, {key: results[key] for key in keys if key not in runs}
