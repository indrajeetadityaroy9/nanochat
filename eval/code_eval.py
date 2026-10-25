"""
Code generation of a base model, scored by the benchmarks' official evaluators.

Benchmarks:
- humaneval_plus (164) and mbpp_plus (378): EvalPlus (arXiv 2305.01210; the evalplus package pinned in
  pyproject.toml, data releases HumanEvalPlus v0.1.10 / MbppPlus v0.2.0). Prompt, budget, stop sequences and
  post-processing as EvalPlus's codegen for a base (direct-completion) model: prompt.strip() + "\\n"
  (evalplus/codegen.py:71), at most EVALPLUS_MAX_NEW_TOKENS new tokens (provider/base.py:13), stops EOS +
  extra_eos_for_direct_completion(dataset) (provider/vllm.py:41-42), special tokens skipped in decoding
  (provider/hf.py:96 skip_special_tokens), the completion cut at its first stop and its tabs replaced by four spaces
  (provider/hf.py:104-105, vllm.py:76), solution sanitize(prompt + completion, entry point) (codegen.py:79-82); scored
  by evalplus.evaluate (EVALPLUS_EVALUATE; base and plus pass@1, its estimate_pass_at_k).
- humaneval_cpp (161) and mbpp_cpp (397): MultiPL-E (arXiv 2208.08227), each row's prompt, tests and stop_tokens, from
  nuprl/MultiPL-E at the dataset revision MultiPL-E's generator pins (multipl_e/completions.py DATASET_REVISION
  @3025a53). Generation as MultiPL-E's Hugging Face generator (automodel.py @3025a53): from prompt.strip() (:119), at
  most 1024 tokens of prompt and completion (:67 generate(max_length=...); :56 truncates the prompt), the completion
  ending at the first special token (:99-101) and at its first stop token (multipl_e/completions.py:216
  stop_at_stop_token); judged by the evaluator of nuprl/MultiPL-E @3025a53 (its checkout at $MULTIPL_E,
  docker/Dockerfile): evaluation/src/main.py
  (program = prompt + completion + "\\n" + tests, main.py:34; g++ -std=c++17 then the binary, 15 s each,
  eval_cpp.py:9-36, safe_subprocess/__init__.py:33), pass@1 by its pass_k.py (status "OK" and exit code 0).
- humaneval_x_cpp (164; HumanEval-X, arXiv 2303.17568): C++ reference solutions, which MultiPL-E lacks.
- fill-in-the-middle: openai/human-eval-infilling (arXiv 2207.14255; installed in the image,
  docker/Dockerfile), single-line, multi-line and random-span infilling of the 164 HumanEval solutions. Each problem's
  prefix and suffix form a PSM or SPM prompt (inference.decoder.fim_prompt); the model writes the middle until BOS (its
  end-of-middle target in training), at most the benchmark's longest reference middle in tokens, scored as written
  (the package's protocol: no post-processing). The references' middles are whole lines, the model's (trained on
  character-level and AST spans) rarely end at a line end; "greedy_reference_lines" scores the same greedy
  single-line and multi-line middles cut to the reference's lines (fim_lines: the first line, the first k lines of a
  k-line reference), the post-processing of arXiv 2505.18789 (Table 1, Fig. 4), which reads the reference's line
  count, so it is a separate metric, not the benchmark's pass@1. FIM prompts are not healed: they end with a sentinel
  (PSM) or with the prefix encoded on its own (SPM), the training format, whose character-level spans train mid-token
  boundaries (corpus/crate/pack.rs random_span). Scored by the package's
  check_correctness per middle (its 3 s timeout) in a worker process per CPU (INFILLING_EVALUATE); pass@1 by its
  estimate_pass_at_k.

Scores: pass@1 by three generation modes, each scored by the benchmark's evaluator. "greedy": one greedy completion per
problem (the pass@1 of DeepSeek-Coder, arXiv 2401.14196 §4.1, and EvalPlus's leaderboard; FIM in PSM and SPM order).
"sampled" (EvalPlus, MultiPL-E): pass@1 by Codex's unbiased estimator over N_SAMPLE samples per problem at
TEMPERATURE from the nucleus TOP_P (the temperature and nucleus of Codex and MultiPL-E), each benchmark's
evaluator computing the estimate. "filtered" (FIM in FILTERED_MODE order only): filtered
pass@1 (Codex's filtered pass@k) over at most verify.ATTEMPTS samples, verify.search: the complete Python 3 syntax
constraint (inference/syntax_constraint.py; FIM: given the prefix and suffix, the middle ending at BOS, uncut),
left-to-right prompts healed (decoder.heal_tokens), the greedy candidate then seeded samples until one has no
diagnostics (verify.python_diagnostics, solution_diagnostics, cpp_diagnostics) and passes its prompt's own examples,
scored by the same evaluators. Selection reads only the prompt, but the prompts' examples overlap the scored base tests
(verify.py), so it favours base over plus pass@1. Every evaluator runs as a fresh process (exec), never a fork of this
torch/CUDA process.
Each record also holds "scores": metrics.summary (standard errors, Miller arXiv 2411.00640) of its per-problem scores,
the evaluator's pass@1 their mean: pass@1 and, with N_SAMPLE samples, pass@10 (Codex's estimator, metrics.pass_at_k);
FIM's also CrossCodeEval's line exact match, exact match, edit similarity and identifier match against the reference
middle (metrics.completion_scores), clustered by HumanEval problem. "latency": metrics.latency of the decoding.

Generation and scoring: requests (inference.decoder.Request) in, token ids out, through an Engine (a verify.Generator
factory; suite.native_engine: decoder.NativeDecoder), so the generation functions do not depend on the engine. A
pass's independent requests share one generate call (generate_groups: parse_generation's two benchmarks, generation's
greedy and sampled requests of every benchmark, fim_generation's requests of every benchmark in both orders); the
searches (filtered_generation, fim_filtered_generation) run a benchmark at a time. Each benchmark's mode is scored on
the caller's scorer thread as soon as its requests are generated, while the GPU works on: the functions return the
futures of the records (suite.evaluate orders the GPU work and collects them).
"""

import codecs
import csv
import functools
import io
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
from typing import Literal, NamedTuple, Protocol

import pyarrow.parquet as pq
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.provider.utility import EOS, extra_eos_for_direct_completion
from evalplus.sanitize import sanitize
from huggingface_hub import hf_hub_download
from human_eval_infilling.data import read_problems
from human_eval_infilling.evaluation import estimate_pass_at_k

from corpus.tokenizer import Tokenizer
from eval.verify import Generator, chosen, search, solution_diagnostics
from inference.decoder import Request, fim_prompt, heal_tokens
from inference.syntax_constraint import SyntaxConstraint
from metrics import completion_scores, latency, parse_outcome, parse_shares, pass_at_k, summary

# HF dataset revisions: MultiPL-E's, its generator's pin (multipl_e/completions.py DATASET_REVISION)
MULTIPL_E = {"repo_id": "nuprl/MultiPL-E", "revision": "3a9e8d226c127392ce81dca29d09f33f0ce3247d"}
HUMANEVAL_X = {"repo_id": "zai-org/humaneval-x", "revision": "62c78627f3072a1454fa0cb0184737cafe5e4198"}
EVALPLUS = {"humaneval_plus": ("humaneval", get_human_eval_plus), "mbpp_plus": ("mbpp", get_mbpp_plus)}
MULTIPL_E_BENCHMARKS = ("humaneval_cpp", "mbpp_cpp")
INFILLING = ("single-line", "multi-line", "random-span")
FIM_MODES = ("psm", "spm")  # the two orders the packer trains (arXiv 2207.14255 App. D)
# the FIM order of the filtered mode
FILTERED_MODE = "spm"
# The sampled mode (Codex, arXiv 2107.03374 §2.1: pass@k's unbiased estimator over n samples per problem; §3.2: nucleus
# sampling at top p 0.95 for all its sampling; §3.3, Fig. 5: T = 0.2 the pass@1 optimum): MultiPL-E's pass@1
# (arXiv 2208.08227: T = 0.2, its pass@1 stable from about 20 samples; completions.py --top-p 0.95). StarCoder2
# (arXiv 2402.19173 §7.1.2) samples MultiPL-E at T = 0.2, top-p 0.95.
N_SAMPLE, TEMPERATURE, TOP_P = 20, 0.2, 0.95
# EvalPlus's most new tokens of a completion (evalplus/provider/base.py:13 max_new_tokens)
EVALPLUS_MAX_NEW_TOKENS = 768


class Task(NamedTuple):
    task_id: str
    prompt: str
    canonical_solution: str | None  # the reference solution
    stopwords: list[str] | None  # the benchmark's stop sequences
    entry_point: str | None  # EvalPlus: the entry point; MultiPL-E: its tests


def tasks() -> dict[str, list[Task]]:
    """{benchmark: [Task]} (module docstring); humaneval_x_cpp holds reference solutions only."""
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
    """NativeDecoder.generate's stopping criteria: true once the generated text holds a stop sequence (incremental
    UTF-8 decoding; search starts where a stop sequence could still begin). healed: the bytes of the prompt token that
    decoder.heal_tokens dropped (b"" for an unhealed prompt), which the generation's first tokens re-emit
    (decoder.Reemit, possibly over several tokens); they belong to the prompt and are not searched."""

    def __init__(self, tokenizer: Tokenizer, stopwords: list[str], *, healed: bytes):
        self.token_bytes, self.stopwords, self.longest = tokenizer.token_bytes, stopwords, max(map(len, stopwords))
        self.utf8, self.text, self.n, self.searched = codecs.getincrementaldecoder("utf-8")("replace"), "", 0, 0
        self.skip = len(healed)  # the healed bytes not yet generated

    def __call__(self, ids: list[int]) -> bool:
        data = b"".join(self.token_bytes[t] for t in ids[self.n:])
        cut = min(self.skip, len(data))
        self.skip -= cut
        self.text += self.utf8.decode(data[cut:])
        self.n, start, self.searched = len(ids), max(0, self.searched - self.longest + 1), len(self.text)
        return any(s in self.text[start:] for s in self.stopwords)


def fim_lines(middle: str, lines: int) -> str:
    """The middle cut to its first `lines` lines, each kept line ending in "\\n" as the references' middles do (one
    written up to BOS or the budget gets one)."""
    parts = middle.split("\n")
    if len(parts) > lines:
        return "\n".join(parts[:lines]) + "\n"
    return middle if middle == "" or middle.endswith("\n") else middle + "\n"


def completion(tokenizer: Tokenizer, task: Task, ids: list[int], *, special: Literal["skip", "stop"],
               healed: bytes) -> str:
    """The completion text of generated ids, cut before the task's first stop sequence (MultiPL-E stop_at_stop_token;
    EvalPlus vLLM stop strings). Special tokens (ids >= BOS, corpus/tokenizer.py) are "skip"ped (EvalPlus
    skip_special_tokens) or "stop" the completion (MultiPL-E automodel.py:99-101); healed (KeywordsStoppingCriteria)
    bytes dropped."""
    bos = tokenizer.bos_token_id
    ids = [t for t in ids if t < bos] if special == "skip" else list(itertools.takewhile(lambda t: t < bos, ids))
    text = tokenizer.enc.decode_bytes(ids)[len(healed):].decode("utf-8", "replace")  # as enc.decode
    return text[:min([j for j in (text.find(s) for s in task.stopwords) if j >= 0], default=len(text))]


class Engine(Protocol):
    """A verify.Generator factory (suite.native_engine): a generator of requests of at most max_length positions
    (len(input_ids) + max_new_tokens - 1), given prompt_store, the prompt positions it may keep for requests that
    repeat a prompt (an engine without a prompt store ignores it)."""

    def __call__(self, *, max_length: int, prompt_store: int) -> Generator: ...


def generate_groups(generator: Generator, groups: list[list[Request]],
                    done: Callable[[int, list[list[int]], dict], None]) -> None:
    """One generator.generate call over every group's requests, in group order (the requests are independent: a
    request's tokens do not depend on the others, decoder.NativeDecoder), calling done(g, ids, timing) as soon as group
    g's requests are all generated: their generated ids in request order and metrics.latency of them
    (generator.timings; duration: from the group's first start to its last end, the call's other requests
    interleaved)."""
    bounds = [0, *itertools.accumulate(map(len, groups))]
    owner = [g for g, group in enumerate(groups) for _ in group]
    out, left = [None] * bounds[-1], list(map(len, groups))
    for i, ids in generator.generate([r for group in groups for r in group]):
        out[i], g = ids, owner[i]
        left[g] -= 1
        if not left[g]:
            starts, firsts, ends, n = zip(*[(*generator.timings[j], len(out[j]))
                                            for j in range(bounds[g], bounds[g + 1])], strict=True)
            done(g, out[bounds[g]:bounds[g + 1]], latency(starts, firsts, ends, n, duration=max(ends) - min(starts)))


def pass_scores(passed: list[list[bool]]) -> dict[str, dict]:
    """metrics.summary of each problem's pass@k estimate (metrics.pass_at_k over its samples; one sample: whether it
    passes), k 1 and 10 when every problem has at least 10 samples."""
    return {f"pass@{k}": summary(pass_at_k(passed, k)) for k in (1, 10) if min(map(len, passed)) >= k}


def filtered(generator: Generator, request: Callable[[int], Request], text: Callable[[int, list[int]], str],
             check: Callable[[int, str], tuple[str, str, list[str]]], *, language: Literal["python", "cpp"],
             count: int) -> tuple[list[str], dict]:
    """The filtered mode of tasks 0..count-1: verify.search over request(i)'s candidates, each checked as check(i, its
    text(i, ids)) (the (prompt, program, diagnostics) verify.verdicts runs); the text of each task's verify.chosen
    candidate and which one it is per task: the candidates tried and whether the last was accepted (else the greedy
    one is scored), with the search's latency."""
    tried, timing = search(generator, request, lambda i, ids: check(i, text(i, ids)), language=language, count=count)
    return [text(i, chosen(c).ids) for i, c in enumerate(tried)], {
        "attempts": [len(c) for c in tried], "accepted": [c[-1].accepted for c in tried], "latency": timing}


def filtered_completions(generator: Generator, tokenizer: Tokenizer, items: list[Task], prompts: list[list[int]],
                         budgets: list[int], *, special: Literal["skip", "stop"],
                         check: Callable[[Task, str], tuple[str, str, list[str]]],
                         language: Literal["python", "cpp"]) -> tuple[list[str], dict]:
    """filtered of left-to-right prompts (BOS and the prompt's tokens): candidates of the healed prompt
    (decoder.heal_tokens; one more token of budget, the dropped token's bytes re-emitted: Reemit, or the syntax
    constraint of the healed prompt's bytes owing them for Python tasks); each candidate checked as check(task,
    completion): the program its evaluator scores."""
    dropped = [heal_tokens(tokenizer, p)[2] for p in prompts]

    def request(i: int) -> Request:
        input_ids, constraint, _ = heal_tokens(tokenizer, prompts[i])  # a fresh constraint per candidate
        if language == "python":
            constraint = SyntaxConstraint(tokenizer, tokenizer.enc.decode_bytes(input_ids[1:]), suffix=b"",
                                          owed=dropped[i])
        return Request(input_ids=input_ids, max_new_tokens=budgets[i] + 1, constraint=constraint,
                       stopping_criteria=KeywordsStoppingCriteria(tokenizer, items[i].stopwords, healed=dropped[i]))

    return filtered(generator, request,
                    lambda i, ids: completion(tokenizer, items[i], ids, special=special, healed=dropped[i]),
                    lambda i, text: check(items[i], text), language=language, count=len(items))


def filtered_middles(generator: Generator, tokenizer: Tokenizer, problems: list[dict], prompts: list[list[int]], *,
                     max_new_tokens: int) -> tuple[list[str], dict]:
    """filtered of FIM prompts: candidates under the syntax constraint of the problem's prefix and suffix (the middle
    ending at BOS where prefix + middle + suffix parses), each checked as prefix + middle + suffix against the examples
    of its prompt (the prefix)."""
    return filtered(generator, lambda i: Request(
                        input_ids=prompts[i], max_new_tokens=max_new_tokens,
                        constraint=SyntaxConstraint(tokenizer, problems[i]["prompt"].encode(),
                                                    suffix=problems[i]["suffix"].encode())),
                    lambda i, ids: tokenizer.enc.decode(ids),
                    lambda i, middle: (problems[i]["prompt"], problems[i]["prompt"] + middle + problems[i]["suffix"],
                                       []),
                    language="python", count=len(problems))


def sanitizers() -> ProcessPoolExecutor:
    """evalplus_solution's sanitize workers, a process per CPU (sanitize parses every line span of a program), from a
    forkserver (never a fork of this torch/CUDA process), SyntaxWarnings of generated code ignored. Each worker imports
    the main module (run.py, whose top level is light for this reason). The caller shuts the pool down
    (suite.evaluate's `with`): in a spawned process (run.py's phases), multiprocessing's exit handler joins the
    pool's workers before concurrent.futures' exit handler would stop them, so a pool left open never lets the process
    exit."""
    return ProcessPoolExecutor(os.cpu_count(), mp_context=multiprocessing.get_context("forkserver"),
                               initializer=warnings.filterwarnings, initargs=("ignore", "", SyntaxWarning))


def evalplus_solution(pool: ProcessPoolExecutor, task: Task, text: str) -> str:
    """The solution EvalPlus scores for a completion text: its tabs replaced by four spaces, sanitize(prompt +
    completion, entry point), computed by a sanitizers process of `pool` (callers parallelize by threads)."""
    return pool.submit(sanitize, task.prompt + text.replace("\t", "    "), task.entry_point).result()


def evalplus_check(pool: ProcessPoolExecutor, task: Task, text: str) -> tuple[str, str, list[str]]:
    """(prompt, solution, verify.solution_diagnostics) of a completion text, the solution evalplus_solution's."""
    solution = evalplus_solution(pool, task, text)
    return task.prompt, solution, solution_diagnostics(task.prompt + text.replace("\t", "    "), solution,
                                                       task.entry_point)


def multipl_e_check(task: Task, text: str) -> tuple[str, str, list[str]]:
    """(prompt, program, no diagnostics yet) of a completion text: the judge's program (prompt + completion + "\\n" +
    tests, main.py:34) up to the tests' first line, the function's closing "}" (the stop token "\\n}" its completion is
    cut before)."""
    return task.prompt, task.prompt + text + "\n}", []


# `python -m evalplus.evaluate` as a function call (its entry point, Fire(evaluate)) in a fresh isolated process, all
# CPUs testing (its default: half), ended by os._exit once evaluate has written its output file: its stall warner
# (evaluate.py stucking_checker, a non-daemon thread sleeping 20 s per check) would hold the exit up to 20 s
EVALPLUS_EVALUATE = """
import os, sys
from evalplus.evaluate import evaluate
evaluate(dataset=sys.argv[1], samples=sys.argv[2], output_file=sys.argv[3], parallel=os.cpu_count())
sys.stdout.flush()
os._exit(0)
"""


def evalplus(pool: ProcessPoolExecutor, name: str, items: list[Task], samples: list[list[str]]) -> dict:
    """EvalPlus's evaluation of each task's completions (evalplus_solution's solutions, as many per task as it has):
    {"base pass@1", "plus pass@1" (its estimate_pass_at_k over each task's samples, plus: base and plus tests passed),
    "scores": {"base", "plus": pass_scores of the same outcomes}, "ids", "base_status", "plus_status" (per task, per
    sample)}."""
    dataset = EVALPLUS[name][0]
    pairs = [(t, s) for t, task_samples in zip(items, samples, strict=True) for s in task_samples]
    with tempfile.TemporaryDirectory() as tmp:
        path, scored = Path(tmp, "samples.jsonl"), Path(tmp, "results.json")
        with path.open("w", encoding="utf-8") as f, ThreadPoolExecutor(os.cpu_count()) as threads:
            for (t, _), solution in zip(pairs, threads.map(lambda pair: evalplus_solution(pool, *pair), pairs),
                                        strict=True):
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
    """MultiPL-E's evaluation of each problem's completions: {"pass@1" (pass_k.py: Codex's estimator over each
    problem's completions), "scores": pass_scores of pass_k.py's outcomes (status "OK" and exit code 0), "ids",
    "status" (per problem, per completion)}. Each problem is a completions file as
    multipl_e/completions.py writes it, judged by its own main.py --file process (--dir runs problems serially),
    os.cpu_count() at once, each with one judge thread (--max-workers 1): main.py's default of cpu_count() - 1 threads
    per file puts up to 20 g++ per core, and a compile past its wall-clock limit scores "SyntaxError" (eval_cpp.py).
    The scripts import siblings, so no -I."""
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
        # pass_k.py's CSV (Dataset,Pass@k,Estimate,NumProblems,MinCompletions,MaxCompletions): at temperature 0.2,
        # which it takes when a file has none, as ours, one row, k = 1; none when a directory mixes temperatures
        out = subprocess.run([sys.executable, root / "pass_k.py", tmp], cwd=tmp, capture_output=True, text=True,
                             check=True).stdout
        rows = [r for r in csv.DictReader(io.StringIO(out)) if r["Pass@k"] == "1"]
        if len(rows) != 1:
            raise RuntimeError(f"pass_k.py printed {len(rows)} pass@1 rows: {out!r}")
        pass_at_1 = float(rows[0]["Estimate"])
        results = [json.loads(Path(tmp, f"{t.task_id}.results.json").read_text(encoding="utf-8"))["results"]
                   for t in items]
    passed = [[r["status"] == "OK" and r["exit_code"] == 0 for r in rs] for rs in results]
    return {"pass@1": pass_at_1, "scores": pass_scores(passed), "ids": [t.task_id for t in items],
            "status": [[r["status"] for r in rs] for rs in results]}


# human_eval_infilling's evaluation (evaluation.py evaluate_functional_correctness: check_correctness per sample at its
# default timeout, 3 s) across a fork pool of worker processes, in a fresh isolated process: its CLI's thread pool forks
# a Manager and a Process per sample from one interpreter, so its rate is bound by that interpreter's GIL (4 threads or
# 64, about the same). The CLI's modules are imported first: a sample runs in a fork of this process, and what it
# imports at run time depends on the modules already loaded (reliability_guard disables what an import would call).
INFILLING_EVALUATE = """
import json, multiprocessing, sys
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
with ProcessPoolExecutor(int(sys.argv[3]), mp_context=multiprocessing.get_context("fork")) as pool:
    result = list(pool.map(check, samples))
with open(sys.argv[4], "w", encoding="utf-8") as f:
    json.dump(result, f)
"""


def infilling(name: str, problems: list[dict], middles: list[str]) -> dict:
    """Official evaluation of one middle per problem: {"pass@1", "scores", "ids", "completions", "result"} (per problem
    "passed", "failed: <exception>" or "timed out", human_eval_infilling/execution.py), a worker process per CPU
    (INFILLING_EVALUATE). scores: metrics.summary of each per-problem score, clustered by the HumanEval problem a
    task derives from (task id "<Benchmark>/HumanEval/<n>/<span>"; Miller, arXiv 2411.00640: clustered questions),
    "pass@1" (whether it passed) and metrics.completion_scores against the reference middle (canonical_solution)."""
    samples = [{"task_id": p["task_id"], "completion": middle} for p, middle in zip(problems, middles, strict=True)]
    with tempfile.TemporaryDirectory() as tmp:
        path, scored = Path(tmp, "samples.json"), Path(tmp, "results.json")
        path.write_text(json.dumps(samples), encoding="utf-8")
        subprocess.run([sys.executable, "-I", "-c", INFILLING_EVALUATE, name, path, str(os.cpu_count()), scored],
                       cwd=tmp, check=True)
        result = json.loads(scored.read_text(encoding="utf-8"))
    passed = [r == "passed" for r in result]
    clusters = ["/".join(p["task_id"].split("/")[1:3]) for p in problems]
    scores = completion_scores(middles, [p["canonical_solution"] for p in problems]) | {"pass@1": passed}
    return {"pass@1": float(estimate_pass_at_k(1, passed, 1).mean()),
            "scores": {k: summary(v, clusters) for k, v in scores.items()},
            "ids": [s["task_id"] for s in samples], "completions": middles, "result": result}


class CodegenRun(NamedTuple):
    """An EvalPlus or MultiPL-E benchmark's prompts and evaluator (codegen_runs)."""
    items: list[Task]
    prompts: list[list[int]]  # BOS and the prompt's tokens
    budgets: list[int]  # max_new_tokens per task
    special: Literal["skip", "stop"]  # completion's treatment of special tokens
    check: Callable[[Task, str], tuple[str, str, list[str]]]  # filtered_completions' check
    language: Literal["python", "cpp"]
    score: Callable[[list[Task], list[list[str]]], dict]  # the evaluator of each task's completions


def codegen_runs(tokenizer: Tokenizer, benchmarks: dict[str, list[Task]],
                 pool: ProcessPoolExecutor) -> dict[str, CodegenRun]:
    """The CodegenRun of each EvalPlus and MultiPL-E benchmark (module docstring), EvalPlus's solutions sanitized by
    `pool` (sanitizers)."""
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
        # MultiPL-E: at most 1024 tokens of prompt and completion (multipl_e/completions.py --max-tokens,
        # automodel.py:67 generate(max_length=...)), the BOS among them
        runs[name] = CodegenRun(items, prompts, [1024 - len(p) for p in prompts], "stop", multipl_e_check, "cpp",
                                multipl_e)
    return runs


def codegen_record(tokenizer: Tokenizer, run: CodegenRun, mode: Literal["greedy", "sampled"],
                   generated: list[list[int]], timing: dict) -> dict:
    """generation's record of a benchmark's greedy or sampled requests (generated: their ids in request order, a task's
    greedy one or its N_SAMPLE samples in seed order): its evaluator's scores of their completions, the greedy
    completions, and the decoding's latency."""
    n = 1 if mode == "greedy" else N_SAMPLE
    texts = [completion(tokenizer, run.items[j // n], ids, special=run.special, healed=b"")
             for j, ids in enumerate(generated)]
    scores = run.score(run.items, [texts[n * i:n * (i + 1)] for i in range(len(run.items))])
    if mode == "greedy":
        return scores | {"completions": texts, "latency": timing}
    return scores | {"n_sample": N_SAMPLE, "latency": timing}


def generation(engine: Engine, tokenizer: Tokenizer, benchmarks: dict[str, list[Task]], *,
               pool: ProcessPoolExecutor, scorers: ThreadPoolExecutor) -> dict[str, dict[str, Future[dict]]]:
    """{benchmark: {"greedy": its evaluator's scores of the greedy completions, the completions and the decoding's
    "latency", "sampled": of N_SAMPLE samples per problem (module docstring)}} of the EvalPlus and MultiPL-E benchmarks
    (codegen_runs; EvalPlus's solutions sanitized by `pool`): futures, each codegen_record on `scorers` as soon as its
    requests are generated. One generate call runs every benchmark's greedy requests, then their samples
    (generate_groups). Sample k of problem i draws with seed N_SAMPLE i + k."""
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
    # the prompt store: each prompt, which its greedy request and its samples share
    generator = engine(max_length=max(len(r.input_ids) + r.max_new_tokens - 1 for group in groups for r in group),
                       prompt_store=sum(len(p) for run in runs.values() for p in run.prompts))
    results = {name: dict.fromkeys(("greedy", "sampled")) for name in runs}

    def done(g: int, generated: list[list[int]], timing: dict) -> None:
        name, mode = keys[g]
        results[name][mode] = scorers.submit(codegen_record, tokenizer, runs[name], mode, generated, timing)

    generate_groups(generator, groups, done)
    return results


def filtered_record(run: CodegenRun, completions: list[str], record: dict) -> dict:
    """filtered_generation's record of a benchmark: its evaluator's scores of the chosen completions, the completions
    and their search record."""
    return run.score(run.items, [[c] for c in completions]) | {"completions": completions} | record


def filtered_generation(engine: Engine, tokenizer: Tokenizer, benchmarks: dict[str, list[Task]], *,
                        pool: ProcessPoolExecutor, scorers: ThreadPoolExecutor) -> dict[str, Future[dict]]:
    """{benchmark: its evaluator's scores of filtered_completions' completions, the completions and their search
    record} of the EvalPlus and MultiPL-E benchmarks (codegen_runs): futures, each filtered_record on `scorers` once
    its search ends, while the next benchmark's search runs."""
    runs = codegen_runs(tokenizer, benchmarks, pool)
    # the longest request: prompt + budget - 1 (filtered_completions' healed prompt is a token shorter and its budget a
    # token longer); the prompt store: the healed prompts, which a search's rounds repeat
    generator = engine(max_length=max(len(p) + m - 1 for run in runs.values()
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
    """parse_generation's record of a benchmark's generated ids (dropped: the bytes heal_tokens dropped from each
    prompt): their completions' metrics.parse_outcome shares, EvalPlus's scores, the outcomes, the completions and the
    decoding's latency."""
    completions = [completion(tokenizer, t, ids, special="skip", healed=d)
                   for t, ids, d in zip(items, generated, dropped, strict=True)]
    outcomes = [parse_outcome(t.prompt + c) for t, c in zip(items, completions, strict=True)]
    return (parse_shares(outcomes) | evalplus(pool, name, items, [[c] for c in completions])
            | {"outcomes": outcomes, "completions": completions, "latency": timing})


def parse_generation(engine: Engine, tokenizer: Tokenizer, benchmarks: dict[str, list[Task]], *,
                     pool: ProcessPoolExecutor, scorers: ThreadPoolExecutor) -> dict[str, Future[dict]]:
    """{benchmark: {"parses", "indentation", "syntax" (shares of its problems: how prompt + completion compiles,
    parse_outcome), "outcomes", "completions", "latency" (generate_groups)} | EvalPlus's base and plus pass@1 of the
    completions (evalplus), sanitized by `pool` (sanitizers)} of HumanEval+ and MBPP+: one greedy completion per
    problem, unconstrained, of the healed prompt (decoder.heal_tokens), TokEval's protocol (BOS, healing, greedy), so
    the failures a vocabulary's line-break tokens cause (SyntaxError when punctuation fuses with a line break,
    IndentationError when a line break fuses with the indentation after it, in arXiv 2608.18062's models) show where
    pass@1 is near zero. One generate call runs both benchmarks; each one's parse_record, CPU work, runs on `scorers`
    as soon as its completions exist, while the GPU decodes on (and the caller's GPU work runs): the futures of the
    records."""
    bos = tokenizer.bos_token_id
    healed = {name: [heal_tokens(tokenizer, [bos, *tokenizer.enc.encode_ordinary(t.prompt)]) for t in benchmarks[name]]
              for name in EVALPLUS}
    budget = EVALPLUS_MAX_NEW_TOKENS + 1  # and a token for the bytes heal_tokens drops
    generator = engine(max_length=max(len(h) + budget - 1 for prompts in healed.values() for h, _, _ in prompts),
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
    prompts: dict[str, list[list[int]]]  # per FIM_MODES order
    lines: list[int] | None  # the reference middles' line counts (fim_lines): single-line 1, multi-line each one's


def fim_runs(tokenizer: Tokenizer) -> dict[str, FimRun]:
    """The FimRun of each INFILLING benchmark (module docstring); random-span has no line counts."""
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


def fim_generation(engine: Engine, tokenizer: Tokenizer, *,
                   scorers: ThreadPoolExecutor) -> dict[str, dict[str, Future[dict]]]:
    """{"<benchmark>/<mode>": {"greedy": infilling(...) of the greedy middles as generated (the package's protocol) and
    the decoding's "latency", and for single-line and multi-line "greedy_reference_lines": of the same middles cut to
    the reference's lines (fim_lines)}} of the INFILLING benchmarks in each FIM_MODES order: futures, each on
    `scorers` as soon as its middles are generated. One generate call runs every benchmark's requests in both orders
    (generate_groups)."""
    runs = fim_runs(tokenizer)
    keys = [(name, mode) for name in INFILLING for mode in FIM_MODES]
    groups = [[Request(input_ids=p, max_new_tokens=runs[name].max_new_tokens) for p in runs[name].prompts[mode]]
              for name, mode in keys]
    # no prompt repeats in the pass: no prompt store
    generator = engine(max_length=max(len(r.input_ids) + r.max_new_tokens - 1 for group in groups for r in group),
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


def fim_filtered_generation(engine: Engine, tokenizer: Tokenizer, *,
                            scorers: ThreadPoolExecutor) -> dict[str, Future[dict]]:
    """{"<benchmark>/<FILTERED_MODE>": infilling(...) of filtered_middles' middles and their search record} of the
    INFILLING benchmarks: futures, each on `scorers` once its search ends, while the next benchmark's search runs."""
    runs = fim_runs(tokenizer)
    prompts = [(p, run.max_new_tokens) for run in runs.values() for p in run.prompts[FILTERED_MODE]]
    # the prompt store: the prompts, which a search's rounds repeat
    generator = engine(max_length=max(len(p) + m - 1 for p, m in prompts), prompt_store=sum(len(p) for p, _ in prompts))
    results = {}
    for name, run in runs.items():
        middles, record = filtered_middles(generator, tokenizer, run.problems, run.prompts[FILTERED_MODE],
                                           max_new_tokens=run.max_new_tokens)
        results[f"{name}/{FILTERED_MODE}"] = scorers.submit(infilling_record, name, run.problems, middles, record)
    return results
