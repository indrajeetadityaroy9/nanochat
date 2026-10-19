"""
Code generation and reference-solution loss of a base model, scored by the benchmarks' official evaluators.

Benchmarks:
- humaneval_plus (164) and mbpp_plus (378): EvalPlus (Liu et al., arXiv 2305.01210; the evalplus package pinned in
  pyproject.toml, data releases HumanEvalPlus v0.1.10 / MbppPlus v0.2.0). Prompt, budget, stop sequences and
  post-processing as EvalPlus's codegen for a base (direct-completion) model: prompt.strip() + "\\n"
  (evalplus/codegen.py:71), at most 768 new tokens (provider/base.py:13), stops EOS +
  extra_eos_for_direct_completion(dataset) (provider/vllm.py:41-42), special tokens skipped in decoding
  (provider/hf.py:96 skip_special_tokens), the completion cut at its first stop and its tabs replaced by four spaces
  (provider/hf.py:104-105, vllm.py:76), solution sanitize(prompt + completion, entry point) (codegen.py:79-82); scored
  by `python -m evalplus.evaluate` (base and plus pass@1, its estimate_pass_at_k).
- humaneval_cpp (161) and mbpp_cpp (397): MultiPL-E (Cassano et al., arXiv 2208.08227), each row's prompt, tests and
  stop_tokens. Generation as MultiPL-E's Hugging Face generator (automodel.py @3025a53): from prompt.strip() (:119), at
  most 1024 tokens of prompt and completion (:56, 67 generate(max_length=...)), the completion ending at the first
  special token (:99-101) and at its first stop token (multipl_e/completions.py:216 stop_at_stop_token); judged by the
  evaluator of nuprl/MultiPL-E @3025a53 (its checkout at $MULTIPL_E, docker/Dockerfile): evaluation/src/main.py
  (program = prompt + completion + "\\n" + tests, main.py:34; g++ -std=c++17 then the binary, 15 s each,
  eval_cpp.py:9-36, safe_subprocess/__init__.py:33), pass@1 by its pass_k.py (status "OK" and exit code 0).
- humaneval_x_cpp (164; HumanEval-X, Zheng et al., arXiv 2303.17568): C++ reference solutions, which MultiPL-E lacks.

Scores:
- solution bits: bits and bytes (loss_eval.evaluate) of each reference solution given its prompt (BOS, prompt, solution;
  scored from the first token starting at or after the prompt's end) on humaneval_plus, mbpp_plus and humaneval_x_cpp;
  their bits per byte track pass@1 across checkpoints where corpus loss does not (Xing et al., arXiv 2604.07769).
- pass@1 of one greedy completion per problem.
Every evaluator runs as a fresh process (exec), never a fork of this torch/CUDA process.
"""

import bisect
import codecs
import itertools
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import pyarrow.parquet as pq
import torch
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.provider.utility import EOS, extra_eos_for_direct_completion
from evalplus.sanitize import sanitize
from huggingface_hub import hf_hub_download

from nanochat.data.loader import PretrainingBatches
from nanochat.eval.loss_eval import evaluate
from nanochat.inference.decoder import NativeDecoder

# HF dataset revisions
MULTIPL_E = {"repo_id": "nuprl/MultiPL-E", "revision": "28441b6024e71d4a1c1c0f6bf171c935cd5a43f2"}
HUMANEVAL_X = {"repo_id": "zai-org/humaneval-x", "revision": "62c78627f3072a1454fa0cb0184737cafe5e4198"}
EVALPLUS = {"humaneval_plus": ("humaneval", get_human_eval_plus), "mbpp_plus": ("mbpp", get_mbpp_plus)}
MULTIPL_E_BENCHMARKS = ("humaneval_cpp", "mbpp_cpp")
SOLUTION_BENCHMARKS = ("humaneval_plus", "mbpp_plus", "humaneval_x_cpp")


class Task(NamedTuple):
    id: str
    prompt: str
    solution: str   # the reference solution
    stops: list     # the benchmark's stop sequences
    entry: str      # EvalPlus: the entry point; MultiPL-E: its tests


def tasks():
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


def solution_batches(tokenizer, examples, *, batch_rows, seq_len, device):
    """Micro-batches (inputs, targets, cu_seqlens) of (prompt, solution) examples, batch_rows x seq_len (one compiled
    shape), one example and one attention segment per row: BOS + the tokens of prompt + solution, targets padding before
    the first token starting at or after the prompt's end (decode_with_offsets); a token straddling it is not scored."""
    bos = tokenizer.bos_token_id
    seqs = []
    for prompt, solution in examples:
        ids = tokenizer.enc.encode_ordinary(prompt + solution)
        _, starts = tokenizer.enc.decode_with_offsets(ids)
        seqs.append(([bos, *ids], bisect.bisect_left(starts, len(prompt)) + 1))  # + 1: the BOS before
    cu_seqlens = torch.arange(0, batch_rows * seq_len + 1, seq_len, dtype=torch.int32, device=device)
    batches = []
    for g in range(0, len(seqs), batch_rows):
        x = torch.full((batch_rows, seq_len), bos, dtype=torch.int32)
        y = torch.full((batch_rows, seq_len), PretrainingBatches.PAD, dtype=torch.int64)
        for r, (ids, first) in enumerate(seqs[g:g + batch_rows]):
            x[r, :len(ids) - 1] = torch.tensor(ids[:-1])
            y[r, first - 1:len(ids) - 1] = torch.tensor(ids[first:])
        batches.append((x.to(device), y.to(device), cu_seqlens))
    return batches


@torch.no_grad()
def solution_bits(losses, tokenizer, benchmarks, *, batch_rows, seq_len, device):
    """{benchmark: bits and bytes (loss_eval.evaluate) of its reference solutions} for SOLUTION_BENCHMARKS."""
    token_bytes = tokenizer.get_token_bytes(device=device)
    results = {}
    for name in SOLUTION_BENCHMARKS:
        batches = solution_batches(tokenizer, [(t.prompt, t.solution) for t in benchmarks[name]],
                                   batch_rows=batch_rows, seq_len=seq_len, device=device)
        results[name] = evaluate(losses, batches, token_bytes, steps=len(batches), num_segments=batch_rows)
    return results


class StopAt:
    """NativeDecoder.run's stop rule: true once the generated text holds a stop sequence (incremental UTF-8 decoding;
    search starts where a stop sequence could still begin)."""

    def __init__(self, tokenizer, stops):
        self.token_bytes, self.stops, self.longest = tokenizer.token_bytes, stops, max(map(len, stops))
        self.utf8, self.text, self.n, self.searched = codecs.getincrementaldecoder("utf-8")("replace"), "", 0, 0

    def __call__(self, ids):
        self.text += "".join(self.utf8.decode(self.token_bytes[t]) for t in ids[self.n:])
        self.n, start, self.searched = len(ids), max(0, self.searched - self.longest + 1), len(self.text)
        return any(s in self.text[start:] for s in self.stops)


def complete(decoder, tokenizer, items, prompts, budgets, *, special):
    """Greedy completions of `prompts` within `budgets` new tokens, each cut before its task's first stop sequence
    (MultiPL-E stop_at_stop_token; EvalPlus vLLM stop strings). Special tokens (ids >= BOS, data/tokenizer.py) are
    "skip"ped (EvalPlus skip_special_tokens) or "stop" the completion (MultiPL-E automodel.py:99-101)."""
    bos = tokenizer.bos_token_id
    requests = (([bos, *tokenizer.enc.encode_ordinary(p)], m, StopAt(tokenizer, t.stops))
                for t, p, m in zip(items, prompts, budgets, strict=True))
    out = [None] * len(items)
    for i, ids in decoder.run(requests):
        ids = [t for t in ids if t < bos] if special == "skip" else list(itertools.takewhile(lambda t: t < bos, ids))
        text = tokenizer.enc.decode(ids)
        out[i] = text[:min([j for j in (text.find(s) for s in items[i].stops) if j >= 0], default=len(text))]
    return out


def evalplus(name, items, completions):
    """EvalPlus's evaluation of the sanitized solutions: {"base pass@1", "plus pass@1", "ids", "completions",
    "base_status", "plus_status"}."""
    dataset = EVALPLUS[name][0]
    with tempfile.TemporaryDirectory() as tmp:
        samples, scored = os.path.join(tmp, "samples.jsonl"), os.path.join(tmp, "results.json")
        with open(samples, "w", encoding="utf-8") as f:
            for t, c in zip(items, completions, strict=True):
                solution = sanitize(t.prompt + c.replace("\t", "    "), entrypoint=t.entry)
                f.write(json.dumps({"task_id": t.id, "solution": solution}) + "\n")
        subprocess.run([sys.executable, "-I", "-m", "evalplus.evaluate", "--dataset", dataset, "--samples", samples,
                        "--output_file", scored], cwd=tmp, check=True)
        with open(scored, encoding="utf-8") as f:
            result = json.load(f)
    verdicts = [result["eval"][t.id][0] for t in items]
    return {"base pass@1": result["pass_at_k"]["base"]["pass@1"], "plus pass@1": result["pass_at_k"]["plus"]["pass@1"],
            "ids": [t.id for t in items], "completions": completions,
            "base_status": [v["base_status"] for v in verdicts], "plus_status": [v["plus_status"] for v in verdicts]}


def multipl_e(items, completions):
    """MultiPL-E's evaluation of one completion per problem: {"pass@1" (pass_k.py), "ids", "completions", "status"}.
    Each problem is a completions file as multipl_e/completions.py writes it, judged by its own main.py --file process
    (--dir runs problems serially), os.cpu_count() at once; the scripts import siblings, so no -I."""
    root = os.environ["MULTIPL_E"]
    main = os.path.join(root, "evaluation", "src", "main.py")
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for t, c in zip(items, completions, strict=True):
            files.append(os.path.join(tmp, f"{t.id}.json"))
            with open(files[-1], "w", encoding="utf-8") as f:
                json.dump({"name": t.id, "language": "cpp", "prompt": t.prompt, "tests": t.entry, "completions": [c]}, f)
        with ThreadPoolExecutor(os.cpu_count()) as pool:
            list(pool.map(lambda file: subprocess.run([sys.executable, main, "--file", file, "--output-dir", tmp],
                                                      cwd=tmp, check=True), files))
        # pass_k.py's one CSV line: Dataset,Pass@k,Estimate,NumProblems,MinCompletions,MaxCompletions
        pass_at_1 = float(subprocess.run([sys.executable, os.path.join(root, "pass_k.py"), "--suppress-header", tmp],
                                         cwd=tmp, capture_output=True, text=True, check=True).stdout.split(",")[2])
        status = []
        for t in items:
            with open(os.path.join(tmp, f"{t.id}.results.json"), encoding="utf-8") as f:
                status.append(json.load(f)["results"][0]["status"])
    return {"pass@1": pass_at_1, "ids": [t.id for t in items], "completions": completions, "status": status}


def generation(model, tokenizer, benchmarks):
    """{benchmark: official evaluation} of greedy completions of the EvalPlus and MultiPL-E benchmarks."""
    decoder, results = NativeDecoder(model, bos_token_id=tokenizer.bos_token_id), {}
    for name in EVALPLUS:
        items = benchmarks[name]
        # EvalPlus: at most 768 new tokens (evalplus/provider/base.py:13 max_new_tokens)
        completions = complete(decoder, tokenizer, items, [t.prompt for t in items], [768] * len(items), special="skip")
        results[name] = evalplus(name, items, completions)
    for name in MULTIPL_E_BENCHMARKS:
        items = benchmarks[name]
        prompts = [t.prompt.strip() for t in items]
        # MultiPL-E: at most 1024 tokens of prompt and completion (multipl_e/completions.py --max-tokens,
        # automodel.py:67 generate(max_length=...)), the BOS among them
        budgets = [1024 - 1 - len(tokenizer.enc.encode_ordinary(p)) for p in prompts]
        completions = complete(decoder, tokenizer, items, prompts, budgets, special="stop")
        results[name] = multipl_e(items, completions)
    return results
