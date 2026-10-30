"""
Code generation diagnostics of a base model, scored by each benchmark's official evaluator in a fresh process: the
parse measure on HumanEval+ and MBPP+ (EvalPlus, arXiv 2305.01210) and the filtered mode on EvalPlus and MultiPL-E C++
(arXiv 2208.08227); HumanEval-X C++ (arXiv 2303.17568) holds reference solutions for the likelihoods (eval/bpb.py).
The benchmarks' standard protocols run in their official tools (scripts/official_eval.sh).
"""

import codecs
import functools
import itertools
import json
import multiprocessing
import os
import sys
import tempfile
import warnings
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Literal, NamedTuple

import pyarrow.parquet as pq
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.provider.utility import EOS, extra_eos_for_direct_completion
from evalplus.sanitize import sanitize
from huggingface_hub import hf_hub_download

from corpus.tokenizer import Tokenizer
from eval.cache import VerdictCache
from eval.verify import (
    Candidate,
    Search,
    chosen,
    compiler,
    run_group,
    search,
    solution_diagnostics,
    timed,
    timed_slots,
    timed_workers,
    vetted_examples,
)
from inference.decoder import NativeDecoder, Request, heal_tokens
from inference.syntax_constraint import SyntaxConstraint
from metrics import latency, parse_outcome, parse_shares, pass_cell
from model.modeling import CausalLM

# HF dataset revisions: MultiPL-E's, its generator's pin (multipl_e/completions.py DATASET_REVISION)
MULTIPL_E = {"repo_id": "nuprl/MultiPL-E", "revision": "3a9e8d226c127392ce81dca29d09f33f0ce3247d"}
HUMANEVAL_X = {"repo_id": "zai-org/humaneval-x", "revision": "62c78627f3072a1454fa0cb0184737cafe5e4198"}
# {benchmark: (EvalPlus dataset, loader)}; a record's scores name the base-test column by the dataset and the
# plus-test column by the benchmark (humaneval and humaneval_plus, mbpp and mbpp_plus, as the papers name them)
EVALPLUS = {"humaneval_plus": ("humaneval", get_human_eval_plus), "mbpp_plus": ("mbpp", get_mbpp_plus)}
MULTIPL_E_BENCHMARKS = ("humaneval_cpp", "mbpp_cpp")
# HumanEval infilling's benchmarks (arXiv 2207.14255; corpus/decontaminate.py's texts, scripts/official_eval.sh's runs)
INFILLING = ("single-line", "multi-line", "random-span")
# The filtered mode's record key: AlphaCode's n@k (arXiv 2203.07814 §2.2), one submission chosen from up to
# verify.ATTEMPTS = 10 candidates by example tests
FILTERED = "1@10"
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


def completion(tokenizer: Tokenizer, task: Task, ids: list[int], *, special: Literal["skip", "stop"],
               healed: bytes) -> str:
    """Completion text of generated ids, cut before the first stop sequence, healed bytes dropped. Special tokens
    (ids >= BOS) are "skip"ped (EvalPlus) or "stop" the completion (MultiPL-E automodel.py:99-101)."""
    bos = tokenizer.bos_token_id
    ids = [t for t in ids if t < bos] if special == "skip" else list(itertools.takewhile(lambda t: t < bos, ids))
    text = tokenizer.enc.decode_bytes(ids)[len(healed):].decode("utf-8", "replace")  # as enc.decode
    return text[:min([j for j in (text.find(s) for s in task.stopwords) if j >= 0], default=len(text))]


def truncated(tokenizer: Tokenizer, task: Task, ids: list[int], *, budget: int, healed: bytes) -> bool:
    """Whether generation ended at its token budget with no stop sequence in its text (the decoder returns no BOS and
    checks the budget before the stop sequences): a completion cut off mid-program, failing by length, not syntax."""
    data = tokenizer.enc.decode_bytes([t for t in ids if t < tokenizer.bos_token_id])[len(healed):]
    text = data.decode("utf-8", "replace")
    return len(ids) == budget and not any(s in text for s in task.stopwords)


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
    """pass@1 as metrics.pass_cell (per-problem means; Wilson interval for one sample a problem)."""
    return {"pass@1": pass_cell(passed)}


class Filtered(NamedTuple):
    """A filtered-mode benchmark (filtered): its verify.Search and the text of a candidate's ids."""
    search: Search
    text: Callable[[int, list[int]], str]

    def record(self, tried: list[list[Candidate]], timing: dict) -> tuple[list[str], dict]:
        """Each task's chosen text and the search record of verify.search's candidates (the chosen one's generated
        tokens among it) and latency."""
        return [self.text(i, chosen(c).ids) for i, c in enumerate(tried)], {
            "attempts": [len(c) for c in tried], "accepted": [c[-1].accepted for c in tried],
            "new_tokens": [len(chosen(c).ids) for c in tried], "latency": timing}


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
    """EvalPlus's evaluation of each task's completions (sanitized evalplus_source): per-sample statuses and pass@1
    of the base tests, under the dataset's name (humaneval, mbpp), and of base and plus tests, under the benchmark's
    (humaneval_plus, mbpp_plus)."""
    dataset = EVALPLUS[name][0]
    pairs = [(t, s) for t, task_samples in zip(items, samples, strict=True) for s in task_samples]
    with tempfile.TemporaryDirectory() as tmp:
        path, scored = Path(tmp, "samples.jsonl"), Path(tmp, "results.json")
        solutions = pool.map(sanitize, [evalplus_source(t, s) for t, s in pairs], [t.entry_point for t, _ in pairs])
        with path.open("w", encoding="utf-8") as f:
            for (t, _), solution in zip(pairs, solutions, strict=True):
                f.write(json.dumps({"task_id": t.task_id, "solution": solution}) + "\n")
        with timed_slots(timed_workers()):
            run_group(timed([sys.executable, "-I", "-c", EVALPLUS_EVALUATE, dataset, path, scored,
                             str(timed_workers())]), cwd=tmp).check_returncode()
        result = json.loads(scored.read_text(encoding="utf-8"))
    evaluated = [result["eval"][t.task_id] for t in items]
    base_status = [[v["base_status"] for v in vs] for vs in evaluated]
    plus_status = [[v["plus_status"] for v in vs] for vs in evaluated]
    base = [[b == "pass" for b in bs] for bs in base_status]  # evalplus/evaluate.py's pass@k outcomes
    plus = [[b == p == "pass" for b, p in zip(bs, ps, strict=True)]
            for bs, ps in zip(base_status, plus_status, strict=True)]
    return {"scores": {dataset: pass_scores(base), name: pass_scores(plus)},
            "ids": [t.task_id for t in items], "base_status": base_status, "plus_status": plus_status}


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
    """MultiPL-E's evaluation of each problem's completions: per-completion statuses and pass@1 of status "OK" (as
    pass_k.py counts). One main.py --file process per problem with uncached completions, on verify.compiler()'s g++;
    no -I (the scripts import siblings)."""
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
                run_group(timed([sys.executable, main, "--file", file, "--output-dir", tmp, "--max-workers", "1"]),
                          cwd=tmp, env=env).check_returncode()

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


class CodegenRun(NamedTuple):
    """An EvalPlus or MultiPL-E benchmark's prompts and evaluator (codegen_runs)."""
    items: list[Task]
    prompts: list[list[int]]  # BOS and the prompt's tokens
    budgets: list[int]  # max_new_tokens per task
    special: Literal["skip", "stop"]
    check: Callable[[Task, str], tuple[str, list[str]]]
    language: Literal["python", "cpp"]
    score: Callable[[list[Task], list[list[str]]], dict]


def codegen_runs(tokenizer: Tokenizer, benchmarks: dict[str, list[Task]],
                 pool: ProcessPoolExecutor) -> dict[str, CodegenRun]:
    """The CodegenRun of each EvalPlus and MultiPL-E benchmark."""
    bos = tokenizer.bos_token_id
    runs = {}
    for name in EVALPLUS:
        items = benchmarks[name]
        runs[name] = CodegenRun(items, [[bos, *tokenizer.enc.encode_ordinary(t.prompt)] for t in items],
                                [EVALPLUS_MAX_NEW_TOKENS] * len(items), "skip", functools.partial(evalplus_check, pool),
                                "python", functools.partial(evalplus, pool, name))
    for name in MULTIPL_E_BENCHMARKS:
        items = benchmarks[name]
        prompts = [[bos, *tokenizer.enc.encode_ordinary(t.prompt.strip())] for t in items]
        # At most 1024 tokens of prompt and completion, BOS included (automodel.py:67); the longest prompt is 600 tokens
        # under the corpus tokenizer
        runs[name] = CodegenRun(items, prompts, [1024 - len(p) for p in prompts], "stop", multipl_e_check, "cpp",
                                multipl_e)
    return runs


def filtered_record(score: Callable[[list[Task], list[list[str]]], dict], items: list[Task], completions: list[str],
                    record: dict) -> dict:
    """filtered_generation's record: evaluator scores, chosen completions and their search record."""
    return score(items, [[c] for c in completions]) | {"completions": completions} | record


def vetted(benchmarks: dict[str, list[Task]]) -> dict[str, list[list[tuple[str, str]]]]:
    """verify.vetted_examples of each EvalPlus benchmark's tasks (prompt + canonical solution), in task order; none for
    the MultiPL-E tasks."""
    return ({name: vetted_examples([(t.prompt, evalplus_source(t, t.canonical_solution)) for t in benchmarks[name]])
             for name in EVALPLUS}
            | {name: [[] for _ in benchmarks[name]] for name in MULTIPL_E_BENCHMARKS})


def parse_record(pool: ProcessPoolExecutor, tokenizer: Tokenizer, name: str, items: list[Task], dropped: list[bytes],
                 generated: list[list[int]], timing: dict, budget: int) -> dict:
    """parse_generation's record of generated ids: each completion's outcome ("truncated" when cut at the budget, else
    parse_outcome) and their shares, EvalPlus's scores, outcomes, completions, generated tokens, budget and latency."""
    completions = [completion(tokenizer, t, ids, special="skip", healed=d)
                   for t, ids, d in zip(items, generated, dropped, strict=True)]
    outcomes = ["truncated" if truncated(tokenizer, t, ids, budget=budget, healed=d) else parse_outcome(t.prompt + c)
                for t, ids, d, c in zip(items, generated, dropped, completions, strict=True)]
    return (parse_shares(outcomes) | evalplus(pool, name, items, [[c] for c in completions])
            | {"outcomes": outcomes, "completions": completions, "new_tokens": [len(ids) for ids in generated],
               "max_new_tokens": budget, "latency": timing})


def parse_generation(model: CausalLM, tokenizer: Tokenizer, benchmarks: dict[str, list[Task]], *,
                     pool: ProcessPoolExecutor, scorers: ThreadPoolExecutor) -> dict[str, Future[dict]]:
    """Futures of HumanEval+'s and MBPP+'s parse records: one greedy unconstrained completion of each healed prompt,
    TokEval's protocol (BOS, healing, greedy; arXiv 2608.18062 App. C.1)."""
    bos = tokenizer.bos_token_id
    healed = {name: [heal_tokens(tokenizer, [bos, *tokenizer.enc.encode_ordinary(t.prompt)]) for t in benchmarks[name]]
              for name in EVALPLUS}
    budget = EVALPLUS_MAX_NEW_TOKENS + 1  # and a token for the bytes heal_tokens drops
    generator = NativeDecoder(model, bos_token_id=tokenizer.bos_token_id,
                              max_length=max(len(h) + budget - 1 for prompts in healed.values() for h, _, _ in prompts),
                              prompt_store=0)  # each healed prompt is requested once
    names = list(healed)
    results = dict.fromkeys(names)

    def done(g: int, generated: list[list[int]], timing: dict) -> None:
        name = names[g]
        results[name] = scorers.submit(parse_record, pool, tokenizer, name, benchmarks[name],
                                       [d for _, _, d in healed[name]], generated, timing, budget)

    generate_groups(generator, [[Request(input_ids=h, max_new_tokens=budget, constraint=c,
                                         stopping_criteria=KeywordsStoppingCriteria(tokenizer, t.stopwords, healed=d))
                                 for t, (h, c, d) in zip(benchmarks[name], healed[name], strict=True)]
                                for name in names], done)
    return results


def filtered_generation(model: CausalLM, tokenizer: Tokenizer, benchmarks: dict[str, list[Task]], *,
                        pool: ProcessPoolExecutor, scorers: ThreadPoolExecutor,
                        found: dict[str, list[list[tuple[str, str]]]]) -> dict[str, Future[dict]]:
    """Futures of each EvalPlus and MultiPL-E benchmark's 1@10 record (FILTERED): one verify.search over all, each
    record scored once its search ends. found: vetted's examples."""
    runs = codegen_runs(tokenizer, benchmarks, pool)
    searches = {name: filtered_completions(tokenizer, run.items, run.prompts, run.budgets, special=run.special,
                                           check=run.check, language=run.language, found=found[name])
                for name, run in runs.items()}
    # every request fits the decoder's max_length, the longest (a slot count changes no tokens)
    lengths = [(p, m) for run in runs.values() for p, m in zip(run.prompts, run.budgets, strict=True)]
    generator = NativeDecoder(model, bos_token_id=tokenizer.bos_token_id,
                              max_length=max(len(p) + m - 1 for p, m in lengths),
                              prompt_store=sum(len(p) for p, _ in lengths))
    keys, results = list(searches), {}

    def done(s: int, tried: list[list[Candidate]], timing: dict) -> None:
        texts, record = searches[keys[s]].record(tried, timing)
        results[keys[s]] = scorers.submit(filtered_record, runs[keys[s]].score, runs[keys[s]].items, texts, record)

    search(generator, [searches[key].search for key in keys], done)
    return results
