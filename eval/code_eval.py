"""
Code generation of a base model, scored by each benchmark's official evaluator in a fresh process: EvalPlus
(arXiv 2305.01210), MultiPL-E C++ (arXiv 2208.08227), HumanEval-X C++ references (arXiv 2303.17568) and HumanEval
infilling (arXiv 2207.14255), in greedy, sampled (arXiv 2107.03374 §2.1) and filtered modes.
"""

import codecs
import csv
import functools
import io
import itertools
import json
import multiprocessing
import os
import statistics
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Literal, NamedTuple

import pyarrow.parquet as pq
import torch
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.provider.utility import EOS, extra_eos_for_direct_completion
from evalplus.sanitize import sanitize
from huggingface_hub import hf_hub_download
from human_eval_infilling.data import read_problems

from corpus.tokenizer import Tokenizer
from eval.verify import chosen, search, solution_diagnostics
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
# Sampled mode: Codex (arXiv 2107.03374 §2.1 unbiased pass@k; §3.2 top p 0.95; §3.3, Fig. 5: T = 0.2 for pass@1);
# MultiPL-E's pass@1 (arXiv 2208.08227: T = 0.2; completions.py --top-p 0.95); StarCoder2 (arXiv 2402.19173 §7.1.2)
N_SAMPLE, TEMPERATURE, TOP_P = 20, 0.2, 0.95
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


def filtered(generator: NativeDecoder, request: Callable[[int], Request], text: Callable[[int, list[int]], str],
             check: Callable[[int, str], tuple[str, str, list[str]]], *, language: Literal["python", "cpp"],
             count: int) -> tuple[list[str], dict]:
    """The filtered mode of tasks 0..count-1 (verify.search): each task's chosen text and the search record.
    check(i, text) returns verify.verdicts' (prompt, program, diagnostics)."""
    tried, timing = search(generator, request, lambda i, ids: check(i, text(i, ids)), language=language, count=count)
    return [text(i, chosen(c).ids) for i, c in enumerate(tried)], {
        "attempts": [len(c) for c in tried], "accepted": [c[-1].accepted for c in tried], "latency": timing}


def filtered_completions(generator: NativeDecoder, tokenizer: Tokenizer, items: list[Task], prompts: list[list[int]],
                         budgets: list[int], *, special: Literal["skip", "stop"],
                         check: Callable[[Task, str], tuple[str, str, list[str]]],
                         language: Literal["python", "cpp"]) -> tuple[list[str], dict]:
    """filtered of left-to-right prompts, healed (one more token of budget; Python under the syntax constraint)."""
    dropped = [heal_tokens(tokenizer, p)[2] for p in prompts]

    def request(i: int) -> Request:
        input_ids, constraint, _ = heal_tokens(tokenizer, prompts[i])  # a fresh constraint per candidate
        if language == "python":
            constraint = SyntaxConstraint(tokenizer, tokenizer.enc.decode_bytes(input_ids[1:]), suffix=b"",
                                          healing=constraint)
        return Request(input_ids=input_ids, max_new_tokens=budgets[i] + 1, constraint=constraint,
                       stopping_criteria=KeywordsStoppingCriteria(tokenizer, items[i].stopwords, healed=dropped[i]))

    return filtered(generator, request,
                    lambda i, ids: completion(tokenizer, items[i], ids, special=special, healed=dropped[i]),
                    lambda i, text: check(items[i], text), language=language, count=len(items))


def filtered_middles(generator: NativeDecoder, tokenizer: Tokenizer, problems: list[dict], prompts: list[list[int]], *,
                     max_new_tokens: int) -> tuple[list[str], dict]:
    """filtered of FIM prompts under the syntax constraint of prefix and suffix, checked on the prefix's examples."""
    return filtered(generator, lambda i: Request(
                        input_ids=prompts[i], max_new_tokens=max_new_tokens,
                        constraint=SyntaxConstraint(tokenizer, problems[i]["prompt"].encode(),
                                                    suffix=problems[i]["suffix"].encode())),
                    lambda i, ids: tokenizer.enc.decode(ids),
                    lambda i, middle: (problems[i]["prompt"], problems[i]["prompt"] + middle + problems[i]["suffix"],
                                       []),
                    language="python", count=len(problems))


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


def evalplus_check(pool: ProcessPoolExecutor, task: Task, text: str) -> tuple[str, str, list[str]]:
    """(prompt, solution, verify.solution_diagnostics) of a completion text (evalplus_source), sanitized by `pool`."""
    source = evalplus_source(task, text)
    solution = pool.submit(sanitize, source, task.entry_point).result()
    return task.prompt, solution, solution_diagnostics(source, solution, task.entry_point)


def multipl_e_check(task: Task, text: str) -> tuple[str, str, list[str]]:
    """(prompt, program, []) of a completion: prompt + completion + the closing "}" its stop token cut."""
    return task.prompt, task.prompt + text + "\n}", []


# `python -m evalplus.evaluate` in a fresh isolated process on all CPUs, ended by os._exit: evaluate.py's
# stucking_checker thread would otherwise hold the exit up to 20 s
EVALPLUS_EVALUATE = """
import os, sys
from evalplus.evaluate import evaluate
evaluate(dataset=sys.argv[1], samples=sys.argv[2], output_file=sys.argv[3], parallel=os.cpu_count())
sys.stdout.flush()
os._exit(0)
"""


def evalplus(pool: ProcessPoolExecutor, name: str, items: list[Task], samples: list[list[str]]) -> dict:
    """EvalPlus's evaluation of each task's completions: base and plus pass@1, per-sample statuses and scores."""
    dataset = EVALPLUS[name][0]
    pairs = [(t, s) for t, task_samples in zip(items, samples, strict=True) for s in task_samples]
    with tempfile.TemporaryDirectory() as tmp:
        path, scored = Path(tmp, "samples.jsonl"), Path(tmp, "results.json")
        solutions = pool.map(sanitize, [evalplus_source(t, s) for t, s in pairs], [t.entry_point for t, _ in pairs])
        with path.open("w", encoding="utf-8") as f:
            for (t, _), solution in zip(pairs, solutions, strict=True):
                f.write(json.dumps({"task_id": t.task_id, "solution": solution}) + "\n")
        subprocess.run([sys.executable, "-I", "-c", EVALPLUS_EVALUATE, dataset, path, scored], cwd=tmp, check=True)
        result = json.loads(scored.read_text(encoding="utf-8"))
    evaluated = [result["eval"][t.task_id] for t in items]
    base_status = [[v["base_status"] for v in vs] for vs in evaluated]
    plus_status = [[v["plus_status"] for v in vs] for vs in evaluated]
    # evalplus/evaluate.py's pass@k outcomes: base_status "pass"; plus, base_status == plus_status == "pass"
    base = [[b == "pass" for b in bs] for bs in base_status]
    plus = [[b == p == "pass" for b, p in zip(bs, ps, strict=True)] for bs, ps in zip(base_status, plus_status,
                                                                                       strict=True)]
    return {"base pass@1": result["pass_at_k"]["base"]["pass@1"], "plus pass@1": result["pass_at_k"]["plus"]["pass@1"],
            "scores": {"base": pass_scores(base), "plus": pass_scores(plus)},
            "ids": [t.task_id for t in items], "base_status": base_status, "plus_status": plus_status}


def multipl_e(items: list[Task], samples: list[list[str]]) -> dict:
    """MultiPL-E's evaluation of each problem's completions: pass_k.py pass@1, per-completion statuses and scores.
    One main.py --file process per problem, one judge thread each; no -I (the scripts import siblings)."""
    root = Path(os.environ["MULTIPL_E"])
    main = root / "evaluation" / "src" / "main.py"
    with tempfile.TemporaryDirectory() as tmp:
        files = [Path(tmp, f"{t.task_id}.json") for t in items]
        for file, t, task_samples in zip(files, items, samples, strict=True):
            file.write_text(json.dumps({"name": t.task_id, "language": "cpp", "prompt": t.prompt,
                                        "tests": t.entry_point,
                                        "completions": task_samples}), encoding="utf-8")
        with ThreadPoolExecutor(os.cpu_count()) as pool:
            list(pool.map(lambda file: subprocess.run([sys.executable, main, "--file", file, "--output-dir", tmp,
                                                       "--max-workers", "1"], cwd=tmp, check=True), files))
        # pass_k.py's CSV holds one k = 1 row at the temperature it assumes for files without one (0.2)
        out = subprocess.run([sys.executable, root / "pass_k.py", tmp], cwd=tmp, capture_output=True, text=True,
                             check=True).stdout
        [row] = [r for r in csv.DictReader(io.StringIO(out)) if r["Pass@k"] == "1"]
        pass_at_1 = float(row["Estimate"])
        results = [json.loads(Path(tmp, f"{t.task_id}.results.json").read_text(encoding="utf-8"))["results"]
                   for t in items]
    passed = [[r["status"] == "OK" and r["exit_code"] == 0 for r in rs] for rs in results]
    return {"pass@1": pass_at_1, "scores": pass_scores(passed), "ids": [t.task_id for t in items],
            "status": [[r["status"] for r in rs] for rs in results]}


# human_eval_infilling check_correctness (3 s) in a fork pool, in a fresh isolated process. The CLI's modules are
# imported first: a forked sample's run-time imports depend on the loaded modules (reliability_guard).
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
with ProcessPoolExecutor(os.cpu_count(), mp_context=multiprocessing.get_context("fork")) as pool:
    result = list(pool.map(check, samples))
with open(sys.argv[3], "w", encoding="utf-8") as f:
    json.dump(result, f)
"""


def infilling(name: str, problems: list[dict], middles: list[str]) -> dict:
    """Official evaluation of one middle per problem: pass@1, per-problem results and scores against the reference
    middle, metrics.summary clustered by HumanEval problem (arXiv 2411.00640)."""
    samples = [{"task_id": p["task_id"], "completion": middle} for p, middle in zip(problems, middles, strict=True)]
    with tempfile.TemporaryDirectory() as tmp:
        path, scored = Path(tmp, "samples.json"), Path(tmp, "results.json")
        path.write_text(json.dumps(samples), encoding="utf-8")
        subprocess.run([sys.executable, "-I", "-c", INFILLING_EVALUATE, name, path, scored], cwd=tmp, check=True)
        result = json.loads(scored.read_text(encoding="utf-8"))
    passed = [r == "passed" for r in result]
    clusters = ["/".join(p["task_id"].split("/")[1:3]) for p in problems]
    scores = completion_scores(middles, [p["canonical_solution"] for p in problems]) | {"pass@1": passed}
    return {"pass@1": statistics.fmean(passed),
            "scores": {k: summary(v, clusters) for k, v in scores.items()},
            "ids": [s["task_id"] for s in samples], "completions": middles, "result": result}


class CodegenRun(NamedTuple):
    """An EvalPlus or MultiPL-E benchmark's prompts and evaluator (codegen_runs)."""
    items: list[Task]
    prompts: list[list[int]]  # BOS and the prompt's tokens
    budgets: list[int]  # max_new_tokens per task
    special: Literal["skip", "stop"]
    check: Callable[[Task, str], tuple[str, str, list[str]]]
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


def codegen_record(tokenizer: Tokenizer, run: CodegenRun, mode: Literal["greedy", "sampled"],
                   generated: list[list[int]], timing: dict) -> dict:
    """generation's record of a benchmark's greedy or sampled ids (a task's samples in seed order)."""
    n = 1 if mode == "greedy" else N_SAMPLE
    texts = [completion(tokenizer, run.items[j // n], ids, special=run.special, healed=b"")
             for j, ids in enumerate(generated)]
    scores = run.score(run.items, [texts[n * i:n * (i + 1)] for i in range(len(run.items))])
    if mode == "greedy":
        return scores | {"completions": texts, "latency": timing}
    return scores | {"n_sample": N_SAMPLE, "latency": timing}


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


def filtered_record(run: CodegenRun, completions: list[str], record: dict) -> dict:
    """filtered_generation's record: evaluator scores, chosen completions and their search record."""
    return run.score(run.items, [[c] for c in completions]) | {"completions": completions} | record


def filtered_generation(model: CausalLM, tokenizer: Tokenizer, benchmarks: dict[str, list[Task]], *,
                        pool: ProcessPoolExecutor, scorers: ThreadPoolExecutor) -> dict[str, Future[dict]]:
    """Futures of each EvalPlus and MultiPL-E benchmark's filtered record; one benchmark searched at a time."""
    runs = codegen_runs(tokenizer, benchmarks, pool)
    generator = decoder(model, tokenizer, max_length=max(len(p) + m - 1 for run in runs.values()
                                                         for p, m in zip(run.prompts, run.budgets, strict=True)),
                        prompt_store=sum(len(p) for run in runs.values() for p in run.prompts))
    results = {}
    for name, run in runs.items():
        completions, record = filtered_completions(generator, tokenizer, run.items, run.prompts, run.budgets,
                                                   special=run.special, check=run.check, language=run.language)
        results[name] = scorers.submit(filtered_record, run, completions, record)
    return results


def parse_record(pool: ProcessPoolExecutor, tokenizer: Tokenizer, name: str, items: list[Task], dropped: list[bytes],
                 generated: list[list[int]], timing: dict) -> dict:
    """parse_generation's record: parse_outcome shares, EvalPlus's scores, outcomes, completions and latency."""
    completions = [completion(tokenizer, t, ids, special="skip", healed=d)
                   for t, ids, d in zip(items, generated, dropped, strict=True)]
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
                        prompt_store=sum(len(h) for prompts in healed.values() for h, _, _ in prompts))
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
        lines = {"single-line": [1] * len(problems),
                 "multi-line": [p["canonical_solution"].count("\n") for p in problems]}.get(name)
        runs[name] = FimRun(problems,
                            max(len(tokenizer.enc.encode_ordinary(p["canonical_solution"])) for p in problems),
                            {mode: [fim_prompt(tokenizer, p["prompt"], p["suffix"], mode=mode) for p in problems]
                             for mode in FIM_MODES},
                            lines)
    return runs


def infilling_record(name: str, problems: list[dict], middles: list[str], record: dict) -> dict:
    """infilling(name, problems, middles) and the given record (latency, search record)."""
    return infilling(name, problems, middles) | record


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
        record["greedy"] = scorers.submit(infilling_record, name, run.problems, middles, {"latency": timing})
        if run.lines is not None:
            record["greedy_reference_lines"] = scorers.submit(
                infilling_record, name, run.problems,
                [fim_lines(m, k) for m, k in zip(middles, run.lines, strict=True)], {})

    generate_groups(generator, groups, done)
    return results


def fim_filtered_generation(model: CausalLM, tokenizer: Tokenizer, *,
                            scorers: ThreadPoolExecutor) -> dict[str, Future[dict]]:
    """Futures of {"<benchmark>/<FILTERED_MODE>": infilling record of filtered_middles' middles}."""
    runs = fim_runs(tokenizer)
    prompts = [(p, run.max_new_tokens) for run in runs.values() for p in run.prompts[FILTERED_MODE]]
    generator = decoder(model, tokenizer, max_length=max(len(p) + m - 1 for p, m in prompts),
                        prompt_store=sum(len(p) for p, _ in prompts))
    results = {}
    for name, run in runs.items():
        middles, record = filtered_middles(generator, tokenizer, run.problems, run.prompts[FILTERED_MODE],
                                           max_new_tokens=run.max_new_tokens)
        results[f"{name}/{FILTERED_MODE}"] = scorers.submit(infilling_record, name, run.problems, middles, record)
    return results
