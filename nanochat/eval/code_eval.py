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
  by `python -m evalplus.evaluate` (base and plus pass@1, its estimate_pass_at_k).
- humaneval_cpp (161) and mbpp_cpp (397): MultiPL-E (arXiv 2208.08227), each row's prompt, tests and
  stop_tokens. Generation as MultiPL-E's Hugging Face generator (automodel.py @3025a53): from prompt.strip() (:119), at
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
  end-of-middle target in training), at most the benchmark's longest reference middle in tokens. The references'
  middles are whole lines, the model's (trained on character-level and AST spans) rarely end at a line end, so
  single-line and multi-line middles are cut to the reference's lines (fim_lines: the first line, the first k lines
  of a k-line reference; random-span unchanged), decoding stopping there (LinesStop), the post-processing of
  arXiv 2505.18789 (Table 1, Fig. 4: single-line one line, multi-line the reference's line count; its Table 1 also
  truncates a random-span middle's overlap with the prefix and suffix, which is not done here). FIM prompts are not
  healed: they end with a sentinel (PSM) or with the prefix encoded on its own (SPM), the training format, whose
  character-level spans train mid-token boundaries (corpus/crate/pack.rs random_span). Scored by the package's entry
  point (`python -m human_eval_infilling.evaluate_functional_correctness`: 3 s timeout, 4 workers); pass@1 by its
  estimate_pass_at_k.

Scores: pass@1 by three generation modes, each scored by the benchmark's evaluator. "greedy": one greedy completion per
problem (the pass@1 of DeepSeek-Coder, arXiv 2401.14196 §4.1, and EvalPlus's leaderboard; FIM in PSM and SPM order).
"sampled" (EvalPlus, MultiPL-E): pass@1 by Codex's unbiased estimator over SAMPLES samples per problem at
SAMPLE_TEMPERATURE from the nucleus TOP_P (the temperature and nucleus of Codex and MultiPL-E), each benchmark's
evaluator computing the estimate. "filtered" (FIM in FILTERED_MODE order only): filtered
pass@1 (Codex's filtered pass@k) over at most verify.ATTEMPTS samples, verify.search: the complete Python 3 syntax
constraint (inference/syntax_constraint.py; FIM: given the prefix and suffix), left-to-right prompts healed
(decoder.heal), the greedy candidate then seeded samples until one has no diagnostics (verify.python_diagnostics,
solution_diagnostics, cpp_diagnostics) and passes its prompt's own examples, scored by the same evaluators. Selection
reads only the prompt, but the prompts' examples overlap the scored base tests (verify.py), so it favours base over
plus pass@1. Every evaluator runs as a fresh process (exec), never a fork of this torch/CUDA process.
"""

import codecs
import dataclasses
import functools
import itertools
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Literal, NamedTuple

import pyarrow.parquet as pq
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.provider.utility import EOS, extra_eos_for_direct_completion
from evalplus.sanitize import sanitize
from huggingface_hub import hf_hub_download
from human_eval_infilling.data import read_problems
from human_eval_infilling.evaluation import estimate_pass_at_k

from nanochat.corpus.tokenizer import Tokenizer
from nanochat.eval.verify import chosen, search, solution_diagnostics
from nanochat.inference.decoder import AllOf, NativeDecoder, Request, fim_prompt, heal
from nanochat.inference.syntax_constraint import SyntaxConstraint
from nanochat.model.gpt import GPT

# HF dataset revisions
MULTIPL_E = {"repo_id": "nuprl/MultiPL-E", "revision": "28441b6024e71d4a1c1c0f6bf171c935cd5a43f2"}
HUMANEVAL_X = {"repo_id": "zai-org/humaneval-x", "revision": "62c78627f3072a1454fa0cb0184737cafe5e4198"}
EVALPLUS = {"humaneval_plus": ("humaneval", get_human_eval_plus), "mbpp_plus": ("mbpp", get_mbpp_plus)}
MULTIPL_E_BENCHMARKS = ("humaneval_cpp", "mbpp_cpp")
INFILLING = ("single-line", "multi-line", "random-span")
FIM_MODES = ("psm", "spm")  # the two orders the packer trains (arXiv 2207.14255 App. D)
# the FIM filtered mode's order: at d8 SPM beat PSM on all three benchmarks (greedy pass@1)
FILTERED_MODE = "spm"
# The sampled mode (Codex, arXiv 2107.03374 §2.1: pass@k's unbiased estimator over n samples per problem; §3.2: nucleus
# sampling at top p 0.95 for all its sampling; §3.3, Fig. 5: T = 0.2 the pass@1 optimum): MultiPL-E's pass@1
# (arXiv 2208.08227: T = 0.2, its pass@1 stable from about 20 samples; completions.py --top-p 0.95). StarCoder2
# (arXiv 2402.19173 §7.1.2) samples MultiPL-E at T = 0.2, top-p 0.95.
SAMPLES, SAMPLE_TEMPERATURE, TOP_P = 20, 0.2, 0.95
# EvalPlus's most new tokens of a completion (evalplus/provider/base.py:13 max_new_tokens)
EVALPLUS_MAX_NEW_TOKENS = 768


class Task(NamedTuple):
    id: str
    prompt: str
    solution: str | None  # the reference solution
    stops: list[str] | None  # the benchmark's stop sequences
    entry: str | None  # EvalPlus: the entry point; MultiPL-E: its tests


def tasks() -> dict[str, list[Task]]:
    """{benchmark: [Task]} (module docstring); humaneval_x_cpp holds reference solutions only."""
    out = {}
    for name, (dataset, load) in EVALPLUS.items():
        stops = EOS + extra_eos_for_direct_completion(dataset)
        out[name] = [Task(task_id, t["prompt"].strip() + "\n", t["canonical_solution"], stops, t["entry_point"])
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


class StopAt:
    """NativeDecoder.run's stop rule: true once the generated text holds a stop sequence (incremental UTF-8 decoding;
    search starts where a stop sequence could still begin). healed: the bytes of the prompt token that decoder.heal
    dropped (b"" for an unhealed prompt), which the generation's first tokens re-emit (decoder.Reemit, possibly over
    several tokens); they belong to the prompt and are not searched."""

    def __init__(self, tokenizer: Tokenizer, stops: list[str], *, healed: bytes):
        self.token_bytes, self.stops, self.longest = tokenizer.token_bytes, stops, max(map(len, stops))
        self.utf8, self.text, self.n, self.searched = codecs.getincrementaldecoder("utf-8")("replace"), "", 0, 0
        self.skip = len(healed)  # the healed bytes not yet generated

    def __call__(self, ids: list[int]) -> bool:
        data = b"".join(self.token_bytes[t] for t in ids[self.n:])
        cut = min(self.skip, len(data))
        self.skip -= cut
        self.text += self.utf8.decode(data[cut:])
        self.n, start, self.searched = len(ids), max(0, self.searched - self.longest + 1), len(self.text)
        return any(s in self.text[start:] for s in self.stops)


class LinesStop:
    """NativeDecoder.run's stop rule of a FIM middle cut to `lines` lines (fim_lines): true once the generated bytes
    hold that many newlines (a newline byte is never part of a longer UTF-8 character)."""

    def __init__(self, tokenizer: Tokenizer, *, lines: int):
        self.token_bytes, self.lines, self.newlines, self.n = tokenizer.token_bytes, lines, 0, 0

    def __call__(self, ids: list[int]) -> bool:
        self.newlines += sum(self.token_bytes[t].count(b"\n") for t in ids[self.n:])
        self.n = len(ids)
        return self.newlines >= self.lines


def fim_lines(middle: str, lines: int | None) -> str:
    """The middle cut to its first `lines` lines (None: unchanged), each kept line ending in "\\n" as the references'
    middles do (one written up to BOS or the budget gets one)."""
    if lines is None:
        return middle
    parts = middle.split("\n")
    if len(parts) > lines:
        return "\n".join(parts[:lines]) + "\n"
    return middle if middle == "" or middle.endswith("\n") else middle + "\n"


def completion(tokenizer: Tokenizer, task: Task, ids: list[int], *, special: Literal["skip", "stop"],
               healed: bytes) -> str:
    """The completion text of generated ids, cut before the task's first stop sequence (MultiPL-E stop_at_stop_token;
    EvalPlus vLLM stop strings). Special tokens (ids >= BOS, corpus/tokenizer.py) are "skip"ped (EvalPlus
    skip_special_tokens) or "stop" the completion (MultiPL-E automodel.py:99-101); healed (StopAt) bytes dropped."""
    bos = tokenizer.bos_token_id
    ids = [t for t in ids if t < bos] if special == "skip" else list(itertools.takewhile(lambda t: t < bos, ids))
    text = tokenizer.enc.decode_bytes(ids)[len(healed):].decode("utf-8", "replace")  # as enc.decode
    return text[:min([j for j in (text.find(s) for s in task.stops) if j >= 0], default=len(text))]


def in_order(decoder: NativeDecoder, requests: list[Request]) -> list[list[int]]:
    """The generated ids of each request, in request order."""
    out = [None] * len(requests)
    for i, ids in decoder.run(requests):
        out[i] = ids
    return out


def filtered(decoder: NativeDecoder, request: Callable[[int], Request], text: Callable[[int, list[int]], str],
             check: Callable[[int, str], tuple[str, str, list[str]]], *, language: Literal["python", "cpp"],
             count: int) -> tuple[list[str], dict[str, list]]:
    """The filtered mode of tasks 0..count-1: verify.search over request(i)'s candidates, each checked as check(i, its
    text(i, ids)) (the (prompt, program, diagnostics) verify.verdicts runs); the text of each task's verify.chosen
    candidate and which one it is per task: the candidates tried and whether the last was accepted (else the greedy
    one is scored)."""
    tried = search(decoder, request, lambda i, ids: check(i, text(i, ids)), language=language, count=count)
    return [text(i, chosen(c).ids) for i, c in enumerate(tried)], {
        "attempts": [len(c) for c in tried], "accepted": [c[-1].accepted for c in tried]}


def filtered_completions(decoder: NativeDecoder, tokenizer: Tokenizer, items: list[Task], prompts: list[list[int]],
                         budgets: list[int], *, special: Literal["skip", "stop"],
                         check: Callable[[Task, str], tuple[str, str, list[str]]],
                         language: Literal["python", "cpp"]) -> tuple[list[str], dict[str, list]]:
    """filtered of left-to-right prompts (BOS and the prompt's tokens): candidates of the healed prompt
    (decoder.heal; one more token of budget, the dropped token's bytes re-emitted), Python tasks also under the syntax
    constraint of the healed prompt's text; each candidate checked as check(task, completion): the program its
    evaluator scores."""
    dropped = [heal(tokenizer, p)[2] for p in prompts]

    def request(i: int) -> Request:
        prompt, constraint, _ = heal(tokenizer, prompts[i])  # a fresh constraint per candidate
        if language == "python":
            constraint = AllOf(constraint, SyntaxConstraint(tokenizer.enc.decode(prompt[1:]), suffix=""))
        return Request(prompt=prompt, max_tokens=budgets[i] + 1, constraint=constraint,
                       stop=StopAt(tokenizer, items[i].stops, healed=dropped[i]))

    return filtered(decoder, request,
                    lambda i, ids: completion(tokenizer, items[i], ids, special=special, healed=dropped[i]),
                    lambda i, text: check(items[i], text), language=language, count=len(items))


def middle_request(tokenizer: Tokenizer, prompt: list[int], lines: int | None, *, max_tokens: int) -> Request:
    """The greedy request of a FIM prompt, stopping at its middle's last kept line (LinesStop; None: uncut)."""
    return Request(prompt=prompt, max_tokens=max_tokens,
                   stop=None if lines is None else LinesStop(tokenizer, lines=lines))


def filtered_middles(decoder: NativeDecoder, tokenizer: Tokenizer, problems: list[dict], prompts: list[list[int]],
                     lines: list[int | None], *, max_tokens: int) -> tuple[list[str], dict[str, list]]:
    """filtered of FIM prompts: candidates under the syntax constraint of the problem's prefix and suffix, each middle
    cut to its lines (fim_lines) and checked as prefix + middle + suffix against the examples of its prompt (the
    prefix)."""
    return filtered(decoder, lambda i: dataclasses.replace(
                        middle_request(tokenizer, prompts[i], lines[i], max_tokens=max_tokens),
                        constraint=SyntaxConstraint(problems[i]["prompt"], suffix=problems[i]["suffix"])),
                    lambda i, ids: fim_lines(tokenizer.enc.decode(ids), lines[i]),
                    lambda i, middle: (problems[i]["prompt"], problems[i]["prompt"] + middle + problems[i]["suffix"],
                                       []),
                    language="python", count=len(problems))


@functools.cache
def sanitizers() -> ProcessPoolExecutor:
    """evalplus_check's sanitize workers, a process per CPU (sanitize parses every line span of a program: run
    serially, the evaluation's bottleneck), from a forkserver (never a fork of this torch/CUDA process), SyntaxWarnings
    of generated code ignored; started once per process."""
    return ProcessPoolExecutor(os.cpu_count(), mp_context=multiprocessing.get_context("forkserver"),
                               initializer=warnings.filterwarnings, initargs=("ignore", "", SyntaxWarning))


def evalplus_solution(task: Task, text: str) -> str:
    """The solution EvalPlus scores for a completion text: its tabs replaced by four spaces, sanitize(prompt +
    completion, entry point), computed by a sanitizers process (callers parallelize by threads)."""
    return sanitizers().submit(sanitize, task.prompt + text.replace("\t", "    "), task.entry).result()


def evalplus_check(task: Task, text: str) -> tuple[str, str, list[str]]:
    """(prompt, solution, verify.solution_diagnostics) of a completion text, the solution evalplus_solution's."""
    solution = evalplus_solution(task, text)
    return task.prompt, solution, solution_diagnostics(task.prompt + text.replace("\t", "    "), solution, task.entry)


def multipl_e_check(task: Task, text: str) -> tuple[str, str, list[str]]:
    """(prompt, program, no diagnostics yet) of a completion text: the judge's program (prompt + completion + "\\n" +
    tests, main.py:34) up to the tests' first line, the function's closing "}" (the stop token "\\n}" its completion is
    cut before)."""
    return task.prompt, task.prompt + text + "\n}", []


def evalplus(name: str, items: list[Task], samples: list[list[str]]) -> dict:
    """EvalPlus's evaluation of each task's completions (evalplus_solution's solutions, as many per task as it has):
    {"base pass@1", "plus pass@1" (its estimate_pass_at_k over each task's samples, plus: base and plus tests passed),
    "ids", "base_status", "plus_status" (per task, per sample)}."""
    dataset = EVALPLUS[name][0]
    pairs = [(t, s) for t, task_samples in zip(items, samples, strict=True) for s in task_samples]
    with tempfile.TemporaryDirectory() as tmp:
        path, scored = Path(tmp, "samples.jsonl"), Path(tmp, "results.json")
        with path.open("w", encoding="utf-8") as f, ThreadPoolExecutor(os.cpu_count()) as pool:
            for (t, _), solution in zip(pairs, pool.map(lambda pair: evalplus_solution(*pair), pairs), strict=True):
                f.write(json.dumps({"task_id": t.id, "solution": solution}) + "\n")
        subprocess.run([sys.executable, "-I", "-m", "evalplus.evaluate", "--dataset", dataset, "--samples", path,
                        "--output_file", scored], cwd=tmp, check=True)
        result = json.loads(scored.read_text(encoding="utf-8"))
    evaluated = [result["eval"][t.id] for t in items]
    return {"base pass@1": result["pass_at_k"]["base"]["pass@1"], "plus pass@1": result["pass_at_k"]["plus"]["pass@1"],
            "ids": [t.id for t in items], "base_status": [[v["base_status"] for v in vs] for vs in evaluated],
            "plus_status": [[v["plus_status"] for v in vs] for vs in evaluated]}


def multipl_e(items: list[Task], samples: list[list[str]]) -> dict:
    """MultiPL-E's evaluation of each problem's completions: {"pass@1" (pass_k.py: Codex's estimator over each
    problem's completions), "ids", "status" (per problem, per completion)}. Each problem is a completions file as
    multipl_e/completions.py writes it, judged by its own main.py --file process (--dir runs problems serially),
    os.cpu_count() at once; the scripts import siblings, so no -I."""
    root = Path(os.environ["MULTIPL_E"])
    main = root / "evaluation" / "src" / "main.py"
    with tempfile.TemporaryDirectory() as tmp:
        files = [Path(tmp, f"{t.id}.json") for t in items]
        for file, t, task_samples in zip(files, items, samples, strict=True):
            file.write_text(json.dumps({"name": t.id, "language": "cpp", "prompt": t.prompt, "tests": t.entry,
                                        "completions": task_samples}), encoding="utf-8")
        with ThreadPoolExecutor(os.cpu_count()) as pool:
            list(pool.map(lambda file: subprocess.run([sys.executable, main, "--file", file, "--output-dir", tmp],
                                                      cwd=tmp, check=True), files))
        # pass_k.py's pass@1 line (one line: pass_k.py takes temperature 0.2 when a file has none, as ours):
        # Dataset,Pass@k,Estimate,NumProblems,MinCompletions,MaxCompletions
        pass_at_1 = float(subprocess.run([sys.executable, root / "pass_k.py", "--suppress-header", tmp],
                                         cwd=tmp, capture_output=True, text=True, check=True).stdout.split(",")[2])
        status = [[r["status"] for r in json.loads(Path(tmp, f"{t.id}.results.json").read_text(encoding="utf-8"))
                   ["results"]] for t in items]
    return {"pass@1": pass_at_1, "ids": [t.id for t in items], "status": status}


def infilling(name: str, problems: list[dict], middles: list[str]) -> dict:
    """Official evaluation of one middle per problem: {"pass@1", "ids", "completions", "result"} (per problem "passed",
    "failed: <exception>" or "timed out", human_eval_infilling/execution.py)."""
    with tempfile.TemporaryDirectory() as tmp:
        samples = Path(tmp, "samples.jsonl")
        with samples.open("w", encoding="utf-8") as f:
            for p, middle in zip(problems, middles, strict=True):
                f.write(json.dumps({"task_id": p["task_id"], "completion": middle}) + "\n")
        subprocess.run([sys.executable, "-I", "-m", "human_eval_infilling.evaluate_functional_correctness", name,
                        samples], cwd=tmp, check=True)
        with Path(tmp, "samples.jsonl_results.jsonl").open(encoding="utf-8") as f:
            scored = [json.loads(line) for line in f]
    return {"pass@1": float(estimate_pass_at_k(1, [s["passed"] for s in scored], 1).mean()),
            "ids": [s["task_id"] for s in scored], "completions": middles, "result": [s["result"] for s in scored]}


def generation(model: GPT, tokenizer: Tokenizer, benchmarks: dict[str, list[Task]]) -> dict[str, dict[str, dict]]:
    """{benchmark: {"greedy": its evaluator's scores of the greedy completions and the completions, "sampled": of
    SAMPLES samples per problem (module docstring), "filtered": of filtered_completions' completions, the completions
    and their search record}} of the EvalPlus and MultiPL-E benchmarks. Sample k of problem i draws with seed
    SAMPLES i + k."""
    bos = tokenizer.bos_token_id
    runs = {}
    for name in (*EVALPLUS, *MULTIPL_E_BENCHMARKS):
        items = benchmarks[name]
        if name in EVALPLUS:
            prompts = [[bos, *tokenizer.enc.encode_ordinary(t.prompt)] for t in items]
            budgets = [EVALPLUS_MAX_NEW_TOKENS] * len(items)
            runs[name] = items, prompts, budgets, "skip", evalplus_check, "python", functools.partial(evalplus, name)
        else:
            prompts = [[bos, *tokenizer.enc.encode_ordinary(t.prompt.strip())] for t in items]
            # MultiPL-E: at most 1024 tokens of prompt and completion (multipl_e/completions.py --max-tokens,
            # automodel.py:67 generate(max_length=...)), the BOS among them
            budgets = [1024 - len(p) for p in prompts]
            runs[name] = items, prompts, budgets, "stop", multipl_e_check, "cpp", multipl_e
    # the longest request: prompt + budget - 1; filtered_completions' healed prompt is a token shorter and its budget
    # a token longer, the same length
    max_len = max(len(p) + m - 1 for _, prompts, budgets, *_ in runs.values()
                  for p, m in zip(prompts, budgets, strict=True))
    # the prompt store: the prompts of the greedy and sampled passes and the healed ones of the filtered pass
    # (decode.cu keeps what fits after its slots)
    prompt_store = 2 * sum(len(p) for _, prompts, *_ in runs.values() for p in prompts)
    decoder = NativeDecoder(model, bos_token_id=bos, max_len=max_len, prompt_store=prompt_store)
    results = {}
    for name, (items, prompts, budgets, special, check, language, score) in runs.items():
        generated = in_order(decoder, [Request(prompt=p, max_tokens=m, stop=StopAt(tokenizer, t.stops, healed=b""))
                                       for t, p, m in zip(items, prompts, budgets, strict=True)])
        greedy = [completion(tokenizer, t, ids, special=special, healed=b"")
                  for t, ids in zip(items, generated, strict=True)]
        sampled = in_order(decoder, [Request(prompt=p, max_tokens=m, stop=StopAt(tokenizer, t.stops, healed=b""),
                                             temperature=SAMPLE_TEMPERATURE, top_p=TOP_P, seed=SAMPLES * i + k)
                                     for i, (t, p, m) in enumerate(zip(items, prompts, budgets, strict=True))
                                     for k in range(SAMPLES)])
        samples = [[completion(tokenizer, t, ids, special=special, healed=b"")
                    for ids in sampled[SAMPLES * i:SAMPLES * (i + 1)]] for i, t in enumerate(items)]
        chosen_completions, record = filtered_completions(decoder, tokenizer, items, prompts, budgets,
                                                          special=special, check=check, language=language)
        results[name] = {
            "greedy": score(items, [[c] for c in greedy]) | {"completions": greedy},
            "sampled": score(items, samples) | {"samples": SAMPLES},
            "filtered": score(items, [[c] for c in chosen_completions]) | {"completions": chosen_completions} | record}
    return results


def parse_outcome(source: str) -> Literal["parses", "indentation", "syntax"]:
    """How a program fails to compile: IndentationError (TabError among them) checked before any other SyntaxError,
    the classification of TokEval's companion post (cimeister.github.io/blog/code_tokenizer_ablations, "How the failing
    generations fail"; linked from arXiv 2608.18062 App. F.3, footnote 15); the parser's overflow (MemoryError,
    RecursionError) is a SyntaxError, as syntax_constraint.syntax_error counts it."""
    with warnings.catch_warnings(action="ignore"):
        try:
            compile(source, "<completion>", "exec")
        except IndentationError:
            return "indentation"
        except (SyntaxError, MemoryError, RecursionError):
            return "syntax"
    return "parses"


def parse_generation(model: GPT, tokenizer: Tokenizer, benchmarks: dict[str, list[Task]]) -> dict[str, dict]:
    """{benchmark: {"parses", "indentation", "syntax" (shares of its problems: how prompt + completion compiles,
    parse_outcome), "outcomes", "completions"} | EvalPlus's base and plus pass@1 of the completions} of HumanEval+ and
    MBPP+: one greedy completion per problem, unconstrained, of the healed prompt (decoder.heal), TokEval's protocol
    (BOS, healing, greedy), so the failures a vocabulary's line-break tokens cause (SyntaxError when punctuation fuses
    with a line break, IndentationError when a line break fuses with the indentation after it, in arXiv 2608.18062's
    models) show where pass@1 is near zero."""
    bos = tokenizer.bos_token_id
    runs = {name: [[bos, *tokenizer.enc.encode_ordinary(t.prompt)] for t in benchmarks[name]] for name in EVALPLUS}
    budget = EVALPLUS_MAX_NEW_TOKENS + 1  # and a token for the bytes heal drops
    max_len = max(len(p) - 1 + budget for prompts in runs.values() for p in prompts)
    decoder = NativeDecoder(model, bos_token_id=bos, max_len=max_len,
                            prompt_store=sum(len(p) for prompts in runs.values() for p in prompts))
    results = {}
    for name, prompts in runs.items():
        items = benchmarks[name]
        healed = [heal(tokenizer, p) for p in prompts]
        generated = in_order(decoder, [Request(prompt=h, max_tokens=budget, constraint=c,
                                               stop=StopAt(tokenizer, t.stops, healed=d))
                                       for t, (h, c, d) in zip(items, healed, strict=True)])
        completions = [completion(tokenizer, t, ids, special="skip", healed=d)
                       for t, ids, (_, _, d) in zip(items, generated, healed, strict=True)]
        outcomes = [parse_outcome(t.prompt + c) for t, c in zip(items, completions, strict=True)]
        shares = {kind: outcomes.count(kind) / len(outcomes) for kind in ("parses", "indentation", "syntax")}
        results[name] = (shares | evalplus(name, items, [[c] for c in completions])
                         | {"outcomes": outcomes, "completions": completions})
    return results


def fim_generation(model: GPT, tokenizer: Tokenizer) -> dict[str, dict[str, dict]]:
    """{"<benchmark>/<mode>": {"greedy": infilling(...) of greedy middles cut to their lines (fim_lines)}} of the
    INFILLING benchmarks in each FIM_MODES order, the FILTERED_MODE entry also {"filtered": that of filtered_middles'
    middles and their search record}."""
    runs = {}
    for name in INFILLING:
        problems = list(read_problems(name).values())
        max_tokens = max(len(tokenizer.enc.encode_ordinary(p["canonical_solution"])) for p in problems)
        prompts = {mode: [fim_prompt(tokenizer, p["prompt"], p["suffix"], mode=mode) for p in problems]
                   for mode in FIM_MODES}
        # the reference middles' lines (fim_lines): single-line 1, multi-line each reference's, random-span uncut
        lines = [{"single-line": 1, "multi-line": p["canonical_solution"].count("\n")}.get(name) for p in problems]
        runs[name] = problems, max_tokens, prompts, lines
    max_len = max(len(prompt) + max_tokens - 1 for _, max_tokens, prompts, _ in runs.values()
                  for mode_prompts in prompts.values() for prompt in mode_prompts)
    # the prompt store: every prompt the greedy and filtered passes run (decode.cu keeps what fits after its slots)
    prompt_store = sum(len(prompt) for _, _, prompts, _ in runs.values() for mode_prompts in prompts.values()
                       for prompt in mode_prompts)
    decoder = NativeDecoder(model, bos_token_id=tokenizer.bos_token_id, max_len=max_len, prompt_store=prompt_store)
    results = {}
    for name, (problems, max_tokens, prompts, lines) in runs.items():
        for mode in FIM_MODES:
            generated = in_order(decoder, [middle_request(tokenizer, prompt, k, max_tokens=max_tokens)
                                         for prompt, k in zip(prompts[mode], lines, strict=True)])
            middles = [fim_lines(tokenizer.enc.decode(ids), k) for ids, k in zip(generated, lines, strict=True)]
            results[f"{name}/{mode}"] = {"greedy": infilling(name, problems, middles)}
        middles, record = filtered_middles(decoder, tokenizer, problems, prompts[FILTERED_MODE], lines,
                                           max_tokens=max_tokens)
        results[f"{name}/{FILTERED_MODE}"]["filtered"] = infilling(name, problems, middles) | record
    return results
