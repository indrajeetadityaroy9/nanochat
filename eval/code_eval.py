"""
Code generation of a base model, scored by each benchmark's official evaluator in a fresh process: EvalPlus
(arXiv 2305.01210), MultiPL-E C++ (arXiv 2208.08227), HumanEval-X C++ references (arXiv 2303.17568) and HumanEval
infilling (arXiv 2207.14255), in greedy, sampled (arXiv 2107.03374 §2.1) and filtered modes; and the reasoning
protocol of a post-trained model (docs/post-training.md §11.4) on the EvalPlus and MultiPL-E benchmarks.
"""

import codecs
import functools
import hashlib
import itertools
import json
import multiprocessing
import os
import re
import sys
import tempfile
import warnings
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Literal, NamedTuple

import human_eval_infilling
import numpy as np
import pyarrow.parquet as pq
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.provider.utility import EOS, extra_eos_for_direct_completion
from evalplus.sanitize import sanitize
from huggingface_hub import hf_hub_download
from human_eval_infilling.data import read_problems

from corpus.tokenizer import Tokenizer
from eval import livecodebench
from eval.cache import VerdictCache
from eval.verify import (
    TOP_P,
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
from inference.decoder import NativeDecoder, Request, fim_prompt, heal_tokens
from inference.syntax_constraint import SyntaxConstraint
from metrics import (
    completion_scores,
    latency,
    line_exact_match,
    parse_outcome,
    parse_shares,
    pass_at_k,
    pass_cell,
    summary,
)
from model.modeling import CausalLM

# HF dataset revisions: MultiPL-E's, its generator's pin (multipl_e/completions.py DATASET_REVISION)
MULTIPL_E = {"repo_id": "nuprl/MultiPL-E", "revision": "3a9e8d226c127392ce81dca29d09f33f0ce3247d"}
HUMANEVAL_X = {"repo_id": "zai-org/humaneval-x", "revision": "62c78627f3072a1454fa0cb0184737cafe5e4198"}
# The original MBPP (arXiv 2108.07732) as lm-evaluation-harness's `mbpp` task reads it (tasks/mbpp/mbpp.yaml:2-3:
# config full); SantaCoder's FIM task (arXiv 2301.03988 §5.4; bigcode-evaluation-harness santacoder_fim.py:55)
MBPP = {"repo_id": "google-research-datasets/mbpp", "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c"}
SANTACODER_FIM = {"repo_id": "bigcode/santacoder-fim-task", "revision": "def3907ab0e63fc18334665533d89770444babc0"}
# {benchmark: (EvalPlus dataset, loader)}; a record's scores name the base-test column by the dataset and the
# plus-test column by the benchmark (humaneval and humaneval_plus, mbpp and mbpp_plus, as the papers name them)
EVALPLUS = {"humaneval_plus": ("humaneval", get_human_eval_plus), "mbpp_plus": ("mbpp", get_mbpp_plus)}
MULTIPL_E_BENCHMARKS = ("humaneval_cpp", "mbpp_cpp")
# MBPP 3-shot (lm-evaluation-harness d6de816 tasks/mbpp/mbpp.yaml, metadata version 1.0): the test split's 500
# problems, the few-shot problems (utils.py:51-86 list_fewshot_samples, the prompt split's first three), greedy
# (mbpp.yaml:17), until "[DONE]" (:15-16), max_gen_toks 256 (models/huggingface.py:628-629 default); HF evaluate's
# code_eval timeout, 3 s (metrics/code_eval/code_eval.py:155)
MBPP_3SHOT, MBPP_SHOTS, MBPP_STOP, MBPP_MAX_NEW_TOKENS, MBPP_TIMEOUT = "mbpp_3shot", (2, 3, 4), "[DONE]", 256, 3.0
INFILLING = ("single-line", "multi-line", "random-span")
FIM_MODES = ("psm", "spm")  # the two orders the packer trains (arXiv 2207.14255 §3, App. D)
FILTERED_MODE = "spm"
# The filtered mode's record key: AlphaCode's n@k (arXiv 2203.07814 §2.2), one submission chosen from up to
# verify.ATTEMPTS = 10 candidates by example tests
FILTERED = "1@10"
# SantaCoder FIM: the Python rows; at most 128 new tokens a middle. The harness bounds prompt and middle together by
# max_length_generation, 512 positions by default (bigcode-evaluation-harness main.py:90-95); under this tokenizer
# some prompts exceed 512 tokens, so the bound is on the middle alone
SANTACODER_LANGUAGE, SANTACODER_MAX_NEW_TOKENS = "py", 128
# Sampled mode: Codex (arXiv 2107.03374 §2.1 unbiased pass@k; §3.3, Fig. 5: T = 0.2 for pass@1; top p: verify.TOP_P);
# MultiPL-E's pass@1 (arXiv 2208.08227 §IV-B: T = 0.2; completions.py --top-p 0.95); StarCoder2 (arXiv 2402.19173
# §7.1.2). pass@1 only: pass@10 is published at T = 0.8 (Codex §3.3; MultiPL-E §IV-B, pass_k.py)
N_SAMPLE, TEMPERATURE = 20, 0.2
# EvalPlus's most new tokens of a completion (evalplus/provider/base.py:13 max_new_tokens)
EVALPLUS_MAX_NEW_TOKENS = 768
# Reasoning protocol (docs/post-training.md §11.4): EvalPlus's instruction for chat models (evalplus/codegen.py
# instruction_prefix; "C++" for "Python" in MultiPL-E's), the post-training format's opening of the reasoning and
# its end (§11.1); samples at DeepSeek-R1's evaluation setting (arXiv 2501.12948 v1 §3: T 0.6, top-p 0.95, pass@1
# the mean over a problem's samples), N_SAMPLE a problem (LiveCodeBench: eval/livecodebench.py)
REASONING_INSTRUCTION = ("Please provide a self-contained {} script that solves the following problem in a markdown "
                         "code block:")
THINK, END_THINK = "\n\n<think>\n", "</think>"
REASONING_TEMPERATURE, REASONING_TOP_P = 0.6, 0.95
# A response's outcome, the first that holds: it ran to its budget (no BOS); it has no END_THINK; no code block
# follows its last END_THINK; answered (the benchmark's extraction of that block)
REASONING_OUTCOMES = ("budget", "no_think_end", "no_block", "answered")
# A fenced code block (CommonMark's backtick fences): an opening line of ``` and an info string, the code, a closing
# line of ``` alone
FENCED_BLOCK = re.compile(r"^```[^`\n]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)


class Task(NamedTuple):
    task_id: str
    prompt: str
    canonical_solution: str | None
    stopwords: list[str] | None
    entry_point: str | None  # EvalPlus: the entry point; MultiPL-E, MBPP 3-shot: its tests


def mbpp_rows(split: Literal["test", "prompt"]) -> list[dict]:
    """The original MBPP's rows of a split (config full) at the pinned revision."""
    file = f"full/{split}-00000-of-00001.parquet"
    return pq.read_table(hf_hub_download(filename=file, repo_type="dataset", **MBPP)).to_pylist()


def mbpp_text(row: dict) -> str:
    """lm-evaluation-harness's MBPP doc_to_text (tasks/mbpp/mbpp.yaml:7)."""
    tests = "\n".join(row["test_list"][:3])
    return (f"You are an expert Python programmer, and here is your task: {row['text']} Your code should pass these "
            f"tests:\n\n{tests}\n[BEGIN]\n")


def mbpp_shots() -> str:
    """The 3-shot context before a problem's text as lm-evaluation-harness renders it without a chat template
    (api/task.py:933-1042 fewshot_context, :1044-1107 build_qa_turn): each shot's doc_to_text, its doc_to_target (code +
    "\\n[DONE]", mbpp.yaml:8) with target_delimiter "" (:9), and the fewshot_delimiter "\\n\\n" (config/task.py:113);
    no description or system prompt."""
    rows = {r["task_id"]: r for r in mbpp_rows("prompt")}
    return "".join(f"{mbpp_text(rows[i])}{rows[i]['code']}\n{MBPP_STOP}\n\n" for i in MBPP_SHOTS)


def tasks() -> dict[str, list[Task]]:
    """{benchmark: [Task]}; humaneval_x_cpp holds reference solutions only; mbpp_3shot's prompt is the problem's text
    alone (codegen_runs prepends mbpp_shots) and its tests are the test_setup_code, if any, and the three asserts."""
    out = {}
    for name, (dataset, load) in EVALPLUS.items():
        stopwords = EOS + extra_eos_for_direct_completion(dataset)
        out[name] = [Task(task_id, t["prompt"].strip() + "\n", t["canonical_solution"], stopwords, t["entry_point"])
                     for task_id, t in load().items()]
    for name in MULTIPL_E_BENCHMARKS:
        file = f"{name.replace('_', '-')}/test-00000-of-00001.parquet"
        rows = pq.read_table(hf_hub_download(filename=file, repo_type="dataset", **MULTIPL_E)).to_pylist()
        out[name] = [Task(r["name"], r["prompt"], None, r["stop_tokens"], r["tests"]) for r in rows]
    out[MBPP_3SHOT] = [Task(str(r["task_id"]), mbpp_text(r), r["code"], [MBPP_STOP],
                            (r["test_setup_code"] + "\n" if r["test_setup_code"].strip() else "")
                            + "\n".join(r["test_list"]))
                       for r in mbpp_rows("test")]
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


def pass_scores(passed: list[list[bool]], ks: tuple[int, ...] = (1,)) -> dict[str, dict]:
    """pass@1 as metrics.pass_cell (per-problem means; Wilson interval for one sample a problem) and, each other k in
    ks, metrics.summary of each problem's unbiased pass@k estimate (every problem having at least k samples)."""
    return {f"pass@{k}": pass_cell(passed) if k == 1 else summary(pass_at_k(passed, k)) for k in ks
            if min(map(len, passed)) >= k}


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


def chat_source(task: Task, text: str) -> str:
    """The program EvalPlus sanitizes of a chat model's answer: the answer alone (not prefixed by the prompt), tabs
    replaced by four spaces (evalplus/codegen.py, provider/hf.py)."""
    return text.replace("\t", "    ")


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


def evalplus(pool: ProcessPoolExecutor, name: str, items: list[Task], samples: list[list[str]], *,
             source: Callable[[Task, str], str] = evalplus_source, ks: tuple[int, ...] = (1,)) -> dict:
    """EvalPlus's evaluation of each task's completions (sanitized source(task, completion)): per-sample statuses and
    scores (pass@k, k in ks) of the base tests, under the dataset's name (humaneval, mbpp), and of base and plus tests,
    under the benchmark's (humaneval_plus, mbpp_plus)."""
    dataset = EVALPLUS[name][0]
    pairs = [(t, s) for t, task_samples in zip(items, samples, strict=True) for s in task_samples]
    with tempfile.TemporaryDirectory() as tmp:
        path, scored = Path(tmp, "samples.jsonl"), Path(tmp, "results.json")
        solutions = pool.map(sanitize, [source(t, s) for t, s in pairs], [t.entry_point for t, _ in pairs])
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
    return {"scores": {dataset: pass_scores(base, ks), name: pass_scores(plus, ks)},
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


def multipl_e(items: list[Task], samples: list[list[str]], *, ks: tuple[int, ...] = (1,)) -> dict:
    """MultiPL-E's evaluation of each problem's completions: per-completion statuses and pass@k (k in ks) of status
    "OK" (as pass_k.py counts). One main.py --file process per problem with uncached completions, on
    verify.compiler()'s g++; no -I (the scripts import siblings)."""
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
    return {"scores": pass_scores(passed, ks), "ids": [t.task_id for t in items], "status": status}


# HF evaluate's code_eval (metrics/code_eval/execute.py:28-86 check_correctness, unsafe_execute at a5db2ad, the
# evaluator of lm-evaluation-harness's mbpp, tasks/mbpp/utils.py:8) on each program (argv[1]) with timeout argv[2],
# in a fork pool of argv[4] workers, in a fresh isolated process on the timed-check cores. Its helpers are
# human-eval's, as human_eval_infilling.execution has them (reliability_guard, time_limit, swallow_io and
# create_tempdir equal to execute.py's up to annotations and IOError = OSError).
MBPP_EVALUATE = """
import json, multiprocessing, sys
from concurrent.futures import ProcessPoolExecutor
from human_eval_infilling.execution import TimeoutException, create_tempdir, reliability_guard, swallow_io, time_limit
def unsafe_execute(check_program, result, timeout):
    with create_tempdir():
        import os
        import shutil
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        reliability_guard()
        try:
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(check_program, exec_globals)
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except BaseException as e:
            result.append(f"failed: {e}")
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir
def check(program):
    timeout = float(sys.argv[2])
    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=unsafe_execute, args=(program, result, timeout))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        result.append("timed out")
    return result[0]
with open(sys.argv[1], encoding="utf-8") as f:
    programs = json.load(f)
with ProcessPoolExecutor(int(sys.argv[4]), mp_context=multiprocessing.get_context("fork")) as pool:
    result = list(pool.map(check, programs))
with open(sys.argv[3], "w", encoding="utf-8") as f:
    json.dump(result, f)
"""


@functools.cache
def mbpp_cache() -> VerdictCache:
    """The VerdictCache of MBPP_EVALUATE: its source and human_eval_infilling's execution module (by SHA-256; the
    package is a checkout on sys.path that cache.environment cannot see)."""
    execution = Path(human_eval_infilling.__file__).parent / "execution.py"
    return VerdictCache("mbpp", json.dumps([MBPP_EVALUATE, MBPP_TIMEOUT,
                                            hashlib.sha256(execution.read_bytes()).hexdigest()]))


def mbpp(items: list[Task], samples: list[list[str]]) -> dict:
    """MBPP 3-shot's evaluation of each problem's completions as lm-evaluation-harness scores its mbpp task
    (utils.py:18-29 pass_at_1 through HF evaluate's code_eval, code_eval.py:172: program = completion + "\\n" +
    tests), with the problem's test_setup_code before its asserts (task 367's, the one test problem with any; the
    harness omits it): per-sample results ("passed", "failed: <exception>" or "timed out"; all but "timed out" kept
    in VerdictCache) and pass@1."""
    programs = [s + "\n" + t.entry_point for t, task_samples in zip(items, samples, strict=True) for s in task_samples]
    cache = mbpp_cache()
    result = cache.get(programs)
    todo = [j for j, r in enumerate(result) if r is None]
    if todo:
        with tempfile.TemporaryDirectory() as tmp:
            path, scored = Path(tmp, "programs.json"), Path(tmp, "results.json")
            path.write_text(json.dumps([programs[j] for j in todo]), encoding="utf-8")
            with timed_slots(timed_workers()):
                run_group(timed([sys.executable, "-I", "-c", MBPP_EVALUATE, path, str(MBPP_TIMEOUT), scored,
                                 str(timed_workers())]), cwd=tmp).check_returncode()
            for j, r in zip(todo, json.loads(scored.read_text(encoding="utf-8")), strict=True):
                result[j] = r
        cache.put([(programs[j], result[j]) for j in todo if result[j] != "timed out"])
    bounds = [0, *itertools.accumulate(map(len, samples))]
    status = [result[bounds[i]:bounds[i + 1]] for i in range(len(items))]
    return {"scores": pass_scores([[r == "passed" for r in rs] for rs in status]), "ids": [t.task_id for t in items],
            "status": status}


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
                run_group(timed([sys.executable, "-I", "-c", INFILLING_EVALUATE, name, path, scored,
                                 str(timed_workers())]), cwd=tmp).check_returncode()
            for j, r in zip(todo, json.loads(scored.read_text(encoding="utf-8")), strict=True):
                result[j] = r
        cache.put([(keys[j], result[j]) for j in todo if result[j] != "timed out"])
    passed = [r == "passed" for r in result]
    clusters = ["/".join(p["task_id"].split("/")[1:3]) for p in problems]
    scores = completion_scores(middles, [p["canonical_solution"] for p in problems]) | {"pass@1": passed}
    return {"scores": {k: summary(v, clusters) for k, v in scores.items()},
            "ids": [s["task_id"] for s in samples], "completions": middles, "result": result} | record


class CodegenRun(NamedTuple):
    """An EvalPlus, MultiPL-E or MBPP 3-shot benchmark's prompts and evaluator (codegen_runs)."""
    items: list[Task]
    prompts: list[list[int]]  # BOS and the prompt's tokens
    budgets: list[int]  # max_new_tokens per task
    special: Literal["skip", "stop"]
    check: Callable[[Task, str], tuple[str, list[str]]] | None  # the 1@10 mode's (none for greedy-only MBPP 3-shot)
    language: Literal["python", "cpp"]
    score: Callable[[list[Task], list[list[str]]], dict]


def codegen_runs(tokenizer: Tokenizer, benchmarks: dict[str, list[Task]],
                 pool: ProcessPoolExecutor) -> dict[str, CodegenRun]:
    """The CodegenRun of each EvalPlus and MultiPL-E benchmark and of MBPP 3-shot."""
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
    items, shots = benchmarks[MBPP_3SHOT], mbpp_shots()
    # BOS first, as in every prompt of this model; lm-evaluation-harness's HF model encodes the context with the
    # tokenizer's special tokens by default (models/huggingface.py:1076-1091 tok_batch_encode), a BOS for a tokenizer
    # that adds one
    runs[MBPP_3SHOT] = CodegenRun(items, [[bos, *tokenizer.enc.encode_ordinary(shots + t.prompt)] for t in items],
                                  [MBPP_MAX_NEW_TOKENS] * len(items), "skip", None, "python", mbpp)
    return runs


def codegen_record(tokenizer: Tokenizer, run: CodegenRun, mode: Literal["greedy", "sampled"],
                   generated: list[list[int]], timing: dict) -> dict:
    """generation's record of a benchmark's greedy or sampled ids: the evaluator's scores, the completions and their
    generated tokens (greedy: one per task; sampled: each task's N_SAMPLE in seed order), each task's token budget and
    latency."""
    n = 1 if mode == "greedy" else N_SAMPLE
    texts = [completion(tokenizer, run.items[j // n], ids, special=run.special, healed=b"")
             for j, ids in enumerate(generated)]
    tokens = [len(ids) for ids in generated]
    samples = [texts[n * i:n * (i + 1)] for i in range(len(run.items))]
    scores = run.score(run.items, samples)
    common = {"max_new_tokens": run.budgets, "latency": timing}
    if mode == "greedy":
        return scores | {"completions": texts, "new_tokens": tokens} | common
    return scores | {"n_sample": N_SAMPLE, "completions": samples,
                     "new_tokens": [tokens[n * i:n * (i + 1)] for i in range(len(run.items))]} | common


def generation(model: CausalLM, tokenizer: Tokenizer, benchmarks: dict[str, list[Task]], *,
               pool: ProcessPoolExecutor, scorers: ThreadPoolExecutor) -> dict[str, dict[str, Future[dict]]]:
    """Futures of {benchmark: {"greedy", "sampled"}} records of the EvalPlus and MultiPL-E benchmarks and of
    {MBPP_3SHOT: {"greedy"}}. Two decoder passes: the EvalPlus and MultiPL-E requests, every greedy one then the
    samples (sample k of problem i seeded N_SAMPLE i + k), their prompts in the prompt store; then MBPP 3-shot's, whose
    longer prompts would otherwise set the first pass's max_length (fewer slots) and each prompt is requested once
    (no store). A request's tokens depend on neither (§6 of docs/evaluation.md)."""
    runs = codegen_runs(tokenizer, benchmarks, pool)
    keys = [(name, "greedy") for name in runs] + [(name, "sampled") for name in runs if name != MBPP_3SHOT]
    results = {name: dict.fromkeys(mode for n, mode in keys if n == name) for name in runs}
    for part in ([key for key in keys if key[0] != MBPP_3SHOT], [key for key in keys if key[0] == MBPP_3SHOT]):
        groups = []
        for name, mode in part:
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
        sampled = {name for name, mode in part if mode == "sampled"}
        generator = NativeDecoder(model, bos_token_id=tokenizer.bos_token_id,
                                  max_length=max(len(r.input_ids) + r.max_new_tokens - 1 for g in groups for r in g),
                                  prompt_store=sum(len(p) for name in sampled for p in runs[name].prompts))

        def done(g: int, generated: list[list[int]], timing: dict, part: list = part) -> None:
            name, mode = part[g]
            results[name][mode] = scorers.submit(codegen_record, tokenizer, runs[name], mode, generated, timing)

        generate_groups(generator, groups, done)
        del generator  # its native memory goes with it (NativeDecoder's finalizer) before the next pass sizes slots
    return results


def reasoning_prompt(task: Task, language: Literal["python", "cpp"]) -> str:
    """The reasoning protocol's prompt text (BOS not included): the user turn of EvalPlus's chat prompt
    (evalplus/provider/utility.py make_raw_chat_prompt: REASONING_INSTRUCTION, the benchmark's prompt stripped in a
    bare ``` fence), stripped as the SFT renderer strips every prompt, then THINK. EvalPlus's prefilled assistant turn
    is left out: a reasoning model's response opens with its reasoning."""
    instruction = REASONING_INSTRUCTION.format({"python": "Python", "cpp": "C++"}[language])
    return f"{instruction}\n```\n{task.prompt.strip()}\n```{THINK}"


def last_block(text: str) -> str:
    """The code of the last fenced block (FENCED_BLOCK) of text, "" if none."""
    blocks = FENCED_BLOCK.findall(text)
    return blocks[-1] if blocks else ""


def reasoning_answer(text: str, *, cut: bool, extract: Callable[[str], str] = last_block) -> tuple[str, str]:
    """(REASONING_OUTCOMES outcome, answer) of a response's text: the answer is extract of the text after the last
    END_THINK, "" unless the outcome is "answered" (an empty extraction is "no_block"). cut: the response ran to its
    budget."""
    if cut:
        return "budget", ""
    _, end, after = text.rpartition(END_THINK)
    if not end:
        return "no_think_end", ""
    answer = extract(after)
    return ("answered", answer) if answer else ("no_block", "")


def multipl_e_answers(items: list[Task], answers: list[list[str]]) -> dict:
    """multipl_e of reasoning answers, each judged as its prompt's #include lines, "#define main reasoning_main", the
    answer, "\\n", "#undef main" and MultiPL-E's tests without their first line, the "}" closing the prompt's function:
    the answer holds the whole function, so neither the prompt's comment and signature nor that "}" are added, and a
    main of its own (a self-contained program) becomes an uncalled function, the tests' main the program's."""
    headers = [t._replace(prompt="".join(line for line in t.prompt.splitlines(keepends=True)
                                         if line.startswith("#include")) + "#define main reasoning_main\n",
                          entry_point="#undef main\n" + t.entry_point.removeprefix("}\n")) for t in items]
    return multipl_e(headers, answers, ks=(1, N_SAMPLE))


class ReasoningRun(NamedTuple):
    """A benchmark's reasoning prompts, samples per problem, answer extraction and evaluator (reasoning_runs)."""
    items: list  # Task, or livecodebench's problems
    prompts: list[list[int]]  # BOS and the prompt text's tokens
    budgets: list[int]  # max_new_tokens per task: the context less the prompt
    n: int  # samples per problem
    extract: Callable[[str], str]  # the answer of the text after the last END_THINK
    score: Callable[[list, list[list[str]]], dict]


def reasoning_runs(tokenizer: Tokenizer, benchmarks: dict[str, list[Task]], problems: list, pool: ProcessPoolExecutor,
                   *, context: int) -> dict[str, ReasoningRun]:
    """The ReasoningRun of each EvalPlus and MultiPL-E benchmark (N_SAMPLE a problem, the last fenced block; Python
    answers scored by EvalPlus as a chat model's, chat_source, C++ by multipl_e_answers) and of LiveCodeBench's
    problems (livecodebench.problems(); eval/livecodebench.py: its prompt, samples, extraction and scorer) at a model's
    context (positions of prompt and response, BOS included)."""
    bos = tokenizer.bos_token_id

    def encoded(texts: list[str]) -> list[list[int]]:
        return [[bos, *tokenizer.enc.encode_ordinary(text)] for text in texts]

    runs = {}
    for name in (*EVALPLUS, *MULTIPL_E_BENCHMARKS):
        language = "python" if name in EVALPLUS else "cpp"
        items = benchmarks[name]
        prompts = encoded([reasoning_prompt(t, language) for t in items])
        score = (functools.partial(evalplus, pool, name, source=chat_source, ks=(1, N_SAMPLE))
                 if language == "python" else multipl_e_answers)
        runs[name] = ReasoningRun(items, prompts, [context - len(p) for p in prompts], N_SAMPLE, last_block, score)
    prompts = encoded([livecodebench.prompt_text(p) + THINK for p in problems])
    runs[livecodebench.NAME] = ReasoningRun(
        problems, prompts, [context - len(p) for p in prompts], livecodebench.N_SAMPLE, livecodebench.answer,
        functools.partial(livecodebench.score, ks=(1, livecodebench.N_SAMPLE)))
    return runs


def reasoning_record(tokenizer: Tokenizer, run: ReasoningRun, generated: list[list[int]], timing: dict) -> dict:
    """reasoning_generation's record of a benchmark's ids (each problem's run.n in seed order): the evaluator's
    statuses and pass@1 (the mean over a problem's samples) and pass@n of the answers, each response's outcome and
    their shares, the responses' generated tokens (BOS not counted) and their mean, median, 90th percentile and
    maximum, the responses (special tokens skipped), each problem's budget and latency."""
    bos, n = tokenizer.bos_token_id, run.n
    texts = [tokenizer.enc.decode([t for t in ids if t < bos]) for ids in generated]
    tokens = [len(ids) for ids in generated]
    graded = [reasoning_answer(text, cut=m == run.budgets[j // n], extract=run.extract)
              for j, (text, m) in enumerate(zip(texts, tokens, strict=True))]
    outcomes = [o for o, _ in graded]

    def per_task(values: list) -> list[list]:
        return [values[n * i:n * (i + 1)] for i in range(len(run.items))]

    lengths = np.array(tokens)
    return run.score(run.items, per_task([a for _, a in graded])) | {
        "n_sample": n, "outcome_shares": {k: outcomes.count(k) / len(outcomes) for k in REASONING_OUTCOMES},
        "outcomes": per_task(outcomes),
        "response_tokens": {"mean": float(lengths.mean()), "median": float(np.median(lengths)),
                            "p90": float(np.percentile(lengths, 90)), "max": int(lengths.max())},
        "completions": per_task(texts), "new_tokens": per_task(tokens), "max_new_tokens": run.budgets,
        "latency": timing}


def reasoning_generation(model: CausalLM, tokenizer: Tokenizer, benchmarks: dict[str, list[Task]], problems: list, *,
                         pool: ProcessPoolExecutor, scorers: ThreadPoolExecutor) -> dict[str, Future[dict]]:
    """Futures of each reasoning benchmark's record (docs/post-training.md §11.4; problems: LiveCodeBench's): run.n
    responses a problem at REASONING_TEMPERATURE from the nucleus REASONING_TOP_P, each until BOS or the model's
    context (no stop sequence), sample k of problem i seeded run.n i + k; one generate call runs all."""
    runs = reasoning_runs(tokenizer, benchmarks, problems, pool, context=model.config.max_position_embeddings)
    names = list(runs)
    groups = [[Request(input_ids=p, max_new_tokens=m, temperature=REASONING_TEMPERATURE, top_p=REASONING_TOP_P,
                       seed=run.n * i + k)
               for i, (p, m) in enumerate(zip(run.prompts, run.budgets, strict=True)) for k in range(run.n)]
              for run in runs.values()]
    generator = NativeDecoder(model, bos_token_id=tokenizer.bos_token_id,
                              max_length=max(len(r.input_ids) + r.max_new_tokens - 1 for g in groups for r in g),
                              prompt_store=sum(len(p) for run in runs.values() for p in run.prompts))
    results = dict.fromkeys(names)

    def done(g: int, generated: list[list[int]], timing: dict) -> None:
        results[names[g]] = scorers.submit(reasoning_record, tokenizer, runs[names[g]], generated, timing)

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
                            {"single-line": [1] * len(problems),
                             "multi-line": [p["canonical_solution"].count("\n") for p in problems]}.get(name))
    return runs


def santacoder_problems() -> list[dict]:
    """SantaCoder FIM's Python rows (bigcode/santacoder-fim-task at the pinned revision, language SANTACODER_LANGUAGE)
    in dataset order."""
    file = "data/train-00000-of-00001-b6ec1fdd66018baa.parquet"
    rows = pq.read_table(hf_hub_download(filename=file, repo_type="dataset", **SANTACODER_FIM)).to_pylist()
    return [r for r in rows if r["language"] == SANTACODER_LANGUAGE]


def santacoder(problems: list[dict], middles: list[str], record: dict) -> dict:
    """SantaCoder FIM's score of one middle per row: exact match of the stripped middle and reference
    (santacoder_fim.py:112, metrics.line_exact_match), metrics.summary clustered by HumanEval problem (a row's name up
    to ".json-L"); the per-row matches, ids, middles, then the given record."""
    matched = [line_exact_match(m, p["canonical_solution"]) for p, m in zip(problems, middles, strict=True)]
    clusters = [p["name"].rsplit(".json-L", 1)[0] for p in problems]
    return {"scores": {"exact_match": summary(matched, clusters)}, "ids": [p["name"] for p in problems],
            "exact_match": matched, "completions": middles} | record


def fim_generation(model: CausalLM, tokenizer: Tokenizer, *,
                   scorers: ThreadPoolExecutor) -> dict[str, dict[str, Future[dict]]]:
    """Futures of {"<benchmark>/<mode>": {"greedy", "greedy_reference_lines"}} infilling records in both FIM orders;
    greedy_reference_lines (single- and multi-line) cuts the same middles by fim_lines; and {"santacoder_fim/psm":
    {"greedy"}}: SantaCoder FIM's Python rows in its prefix-suffix-middle prompt (santacoder_fim.py:78-80, here
    decoder.fim_prompt's psm), greedy, at most SANTACODER_MAX_NEW_TOKENS new tokens. A middle ends at BOS (the
    model's end of text, as the task's stop words end at <|endoftext|>, :62) or another special token. One
    generate call runs all."""
    runs = fim_runs(tokenizer)
    keys = [(name, mode) for name in INFILLING for mode in FIM_MODES]
    groups = [[Request(input_ids=p, max_new_tokens=runs[name].max_new_tokens) for p in runs[name].prompts[mode]]
              for name, mode in keys]
    santa = santacoder_problems()
    santa_prompts = [fim_prompt(tokenizer, p["prompt"], p["suffix"], mode="psm") for p in santa]
    groups.append([Request(input_ids=p, max_new_tokens=SANTACODER_MAX_NEW_TOKENS) for p in santa_prompts])
    generator = NativeDecoder(model, bos_token_id=tokenizer.bos_token_id,
                              max_length=max(len(r.input_ids) + r.max_new_tokens - 1 for g in groups for r in g),
                              prompt_store=0)
    results = {f"{name}/{mode}": {} for name, mode in keys} | {"santacoder_fim/psm": {}}

    def done(g: int, generated: list[list[int]], timing: dict) -> None:
        # a FIM sentinel (ids > BOS) ends the middle too, as the first special token ends a MultiPL-E completion
        middles = [tokenizer.enc.decode(list(itertools.takewhile(lambda t: t < tokenizer.bos_token_id, ids)))
                   for ids in generated]
        tokens = {"new_tokens": [len(ids) for ids in generated]}
        if g == len(keys):
            results["santacoder_fim/psm"]["greedy"] = scorers.submit(
                santacoder, santa, middles, tokens | {"max_new_tokens": SANTACODER_MAX_NEW_TOKENS, "latency": timing})
            return
        name, mode = keys[g]
        run, record = runs[name], results[f"{name}/{mode}"]
        record["greedy"] = scorers.submit(infilling, name, run.problems, middles,
                                          tokens | {"max_new_tokens": run.max_new_tokens, "latency": timing})
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
    """Futures of each EvalPlus and MultiPL-E benchmark's 1@10 record (FILTERED) and of {"<benchmark>/<FILTERED_MODE>":
    infilling record of filtered_middles' middles}: one verify.search over all, each record scored once its search
    ends. found: vetted's examples."""
    runs = {name: run for name, run in codegen_runs(tokenizer, benchmarks, pool).items() if run.check is not None}
    fims = fim_runs(tokenizer)
    searches = ({name: filtered_completions(tokenizer, run.items, run.prompts, run.budgets, special=run.special,
                                            check=run.check, language=run.language, found=found[name])
                 for name, run in runs.items()}
                | {f"{name}/{FILTERED_MODE}": filtered_middles(tokenizer, run.problems, run.prompts[FILTERED_MODE],
                                                               max_new_tokens=run.max_new_tokens, found=found[name])
                   for name, run in fims.items()})
    # every request fits the decoder's max_length, the longest of either kind (a slot count changes no tokens)
    lengths = ([(p, m) for run in runs.values() for p, m in zip(run.prompts, run.budgets, strict=True)]
               + [(p, run.max_new_tokens) for run in fims.values() for p in run.prompts[FILTERED_MODE]])
    generator = NativeDecoder(model, bos_token_id=tokenizer.bos_token_id,
                              max_length=max(len(p) + m - 1 for p, m in lengths),
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
