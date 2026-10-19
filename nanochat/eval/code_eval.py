"""
Code generation and reference-solution loss of a base model, scored by the benchmarks' official evaluators.

Benchmarks:
- humaneval_plus (164) and mbpp_plus (378): EvalPlus (Liu et al., arXiv 2305.01210; the evalplus package pinned in
  pyproject.toml, data releases HumanEvalPlus v0.1.10 / MbppPlus v0.2.0). Prompt, stop sequences and post-processing
  as EvalPlus's codegen for a base (direct-completion) model: prompt.strip() + "\\n" (evalplus/codegen.py:71), stops
  EOS + extra_eos_for_direct_completion(dataset) (evalplus/provider/vllm.py:41-42), the completion cut at its first
  stop and its tabs replaced by four spaces (evalplus/provider/hf.py:104-105, vllm.py:76), solution sanitize(prompt +
  completion, entry point) (codegen.py:79-82); scored by `python -m evalplus.evaluate` (base and plus pass@1, its
  estimate_pass_at_k).
- humaneval_cpp (161) and mbpp_cpp (397): MultiPL-E (Cassano et al., arXiv 2208.08227), each row's prompt, tests and
  stop_tokens; a completion ends at its first stop token (multipl_e/completions.py:216 stop_at_stop_token); judged by
  the evaluator of nuprl/MultiPL-E @3025a53 (its checkout at $MULTIPL_E, docker/Dockerfile): evaluation/src/main.py
  (program = prompt + completion + "\\n" + tests, main.py:34; g++ -std=c++17 then the binary, 15 s each,
  eval_cpp.py:9-36, safe_subprocess/__init__.py:33), pass@1 by its pass_k.py (status "OK" and exit code 0).
- humaneval_x_cpp (164; HumanEval-X, Zheng et al., arXiv 2303.17568): C++ reference solutions, which MultiPL-E lacks.

Scores:
- solution bpb: bits per byte of each reference solution given its prompt (BOS, prompt, solution; scored from the
  first token that begins at or after the prompt's end), on humaneval_plus, mbpp_plus (EvalPlus canonical solutions
  after the generation prompt) and humaneval_x_cpp. Xing et al. (arXiv 2604.07769): it tracks pass@1 across
  checkpoints where corpus loss does not.
- pass@1 of one greedy completion per problem, bounded by its evaluator's default budget: EvalPlus max_new_tokens 768
  (evalplus/provider/base.py:13), MultiPL-E --max-tokens 1024 (multipl_e/completions.py:68-73).
Every evaluator runs as a fresh process (exec), never as a fork of this torch/CUDA process.
"""

import codecs
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
import pyarrow.parquet as pq
import torch
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.provider.utility import EOS, extra_eos_for_direct_completion
from evalplus.sanitize import sanitize
from huggingface_hub import hf_hub_download

from nanochat.data.loader import PretrainingBatches
from nanochat.eval.loss_eval import evaluate_bpb
from nanochat.inference.decoder import NativeDecoder

MULTIPL_E = ("nuprl/MultiPL-E", "28441b6024e71d4a1c1c0f6bf171c935cd5a43f2")  # HF dataset revision
HUMANEVAL_X = ("zai-org/humaneval-x", "62c78627f3072a1454fa0cb0184737cafe5e4198")
EVALPLUS = {"humaneval_plus": ("humaneval", get_human_eval_plus), "mbpp_plus": ("mbpp", get_mbpp_plus)}


class Task(NamedTuple):
    id: str
    prompt: str
    solution: str   # the reference solution
    stops: list     # the benchmark's stop sequences
    entry: str      # EvalPlus: the entry point; MultiPL-E: its tests


def hub_file(dataset, file):
    repo, revision = dataset
    return hf_hub_download(repo, file, repo_type="dataset", revision=revision)


def tasks():
    """{benchmark: [Task]} (module docstring); humaneval_x_cpp holds reference solutions only."""
    out = {}
    for name, (dataset, load) in EVALPLUS.items():
        stops = EOS + extra_eos_for_direct_completion(dataset)
        out[name] = [Task(task_id, t["prompt"].strip() + "\n", t["canonical_solution"], stops, t["entry_point"])
                     for task_id, t in load().items()]
    for name in ("humaneval_cpp", "mbpp_cpp"):
        rows = pq.read_table(hub_file(MULTIPL_E, f"{name.replace('_', '-')}/test-00000-of-00001.parquet")).to_pylist()
        out[name] = [Task(r["name"], r["prompt"], None, r["stop_tokens"], r["tests"]) for r in rows]
    with open(hub_file(HUMANEVAL_X, "data/cpp/data/humaneval.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    out["humaneval_x_cpp"] = [Task(r["task_id"], r["prompt"], r["canonical_solution"], None, None) for r in rows]
    return out


SOLUTION_BENCHMARKS = ("humaneval_plus", "mbpp_plus", "humaneval_x_cpp")
# generated benchmarks and their evaluator's default max new tokens (module docstring)
GENERATION_BUDGETS = {"humaneval_plus": 768, "mbpp_plus": 768, "humaneval_cpp": 1024, "mbpp_cpp": 1024}


def solution_batches(tokenizer, examples, batch_rows, seq_len, device):
    """Micro-batches (inputs, targets, cu_seqlens) of (prompt, solution) examples, each batch_rows x seq_len (one shape:
    one compiled graph), one example per row and one attention segment per row: BOS then prompt + solution tokenized
    together (as in training text), targets the loader's padding but from the first token that begins at or after the
    prompt's end, so the scored targets' bytes are the solution's."""
    bos = tokenizer.bos_token_id
    seqs = []
    for prompt, solution in examples:
        ids = [bos, *tokenizer.enc.encode_ordinary(prompt + solution)]
        ends = np.cumsum([0] + [len(tokenizer.token_bytes[t]) for t in ids[1:]])  # byte offset after each token
        first = int(np.searchsorted(ends, len(prompt.encode()), side="left")) + 1  # the first token starting at/after the prompt end
        seqs.append((ids, first))
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
def solution_bpb(losses, tokenizer, device, batch_rows, seq_len, benchmarks):
    """{benchmark: bits per byte of all its reference solutions pooled} of SOLUTION_BENCHMARKS under losses (GPT.losses),
    by loss_eval.evaluate_bpb."""
    token_bytes = tokenizer.get_token_bytes(device=device)
    results = {}
    for name in SOLUTION_BENCHMARKS:
        batches = solution_batches(tokenizer, [(t.prompt, t.solution) for t in benchmarks[name]], batch_rows, seq_len,
                                   device)
        results[name] = evaluate_bpb(losses, batches, len(batches), token_bytes, batch_rows)
    return results


class StopAt:
    """NativeDecoder.run's stop rule: true once the generated text holds a stop sequence (each call decodes only the
    new ids, UTF-8 incrementally, and searches only from where a sequence could still begin)."""

    def __init__(self, tokenizer, stops):
        self.token_bytes, self.stops, self.longest = tokenizer.token_bytes, stops, max(map(len, stops))
        self.utf8, self.text, self.n, self.searched = codecs.getincrementaldecoder("utf-8")("replace"), "", 0, 0

    def __call__(self, ids):
        self.text += "".join(self.utf8.decode(self.token_bytes[t]) for t in ids[self.n:])
        self.n, start, self.searched = len(ids), max(0, self.searched - self.longest + 1), len(self.text)
        return any(s in self.text[start:] for s in self.stops)


def complete(decoder, tokenizer, items, max_tokens):
    """Greedy completions of the tasks' prompts, each cut before its first stop sequence (MultiPL-E's
    stop_at_stop_token; vLLM's stop strings in EvalPlus)."""
    out = [None] * len(items)
    requests = (([tokenizer.bos_token_id, *tokenizer.enc.encode_ordinary(t.prompt)], max_tokens, StopAt(tokenizer, t.stops))
                for t in items)
    for i, ids in decoder.run(requests):
        text = tokenizer.enc.decode(ids)
        out[i] = text[:min([j for j in (text.find(s) for s in items[i].stops) if j >= 0], default=len(text))]
    return out


def evalplus(name, items, completions):
    """EvalPlus's evaluation of the sanitized solutions (module docstring): {"base pass@1", "plus pass@1" (EvalPlus's
    pass_at_k), "ids", "completions", "base_status", "plus_status"}."""
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
    """MultiPL-E's evaluation (module docstring) of one completion per problem: {"pass@1" (pass_k.py), "ids",
    "completions", "status" (eval_cpp.py's)}. Each problem a completions file as multipl_e/completions.py writes it
    (the dataset row's fields main.py reads, "completions"), judged and scored by the repository's scripts, which import
    their siblings from the script's directory (so not isolated: no -I). main.py's --max-workers runs one problem's
    completions in parallel and its --dir walks the problems one at a time, so with one completion per problem each
    problem is its own main.py --file process, os.cpu_count() of them at once."""
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
    """{benchmark: its official evaluation} of greedy completions of GENERATION_BUDGETS's benchmarks."""
    decoder, results = NativeDecoder(model, bos_token_id=tokenizer.bos_token_id), {}
    for name, max_tokens in GENERATION_BUDGETS.items():
        items = benchmarks[name]
        completions = complete(decoder, tokenizer, items, max_tokens)
        results[name] = evalplus(name, items, completions) if name in EVALPLUS else multipl_e(items, completions)
    return results
