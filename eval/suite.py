"""
A checkpoint's evaluation (RUNS/<run_name>/eval-<step>.json: likelihoods, the parse measure, the generation
benchmarks and the reasoning protocol) and a run's record (its non-train log records and evaluations).
"""
import contextlib
import gc
import gzip
import hashlib
import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import psutil
import torch

from corpus.tokenizer import Tokenizer
from eval.bpb import likelihood
from eval.cache import environment
from eval.code_eval import (
    FILTERED,
    filtered_generation,
    fim_generation,
    generation,
    parse_generation,
    reasoning_generation,
    sanitizers,
    tasks,
    vetted,
)
from eval.verify import cores
from metrics import evalplus_average
from paths import ROOT, RUNS
from train.checkpoint import load_model

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event

# The checkout's code and configuration trees (beside its top-level .py, .toml and .lock files)
SOURCE_TREES = ("ablations", "conf", "corpus", "docker", "eval", "inference", "model", "posttrain", "scripts", "train")


def pin(cpus: frozenset[int]) -> None:
    """Every thread of this process and of its descendants (and what they start later) to cpus. A process or thread
    can end meanwhile (psutil raises FileNotFoundError while /proc/<pid>/stat exists, psutil#2418)."""
    for process in [psutil.Process(), *psutil.Process().children(recursive=True)]:
        with contextlib.suppress(psutil.NoSuchProcess, FileNotFoundError):
            for thread in process.threads():
                with contextlib.suppress(ProcessLookupError):
                    os.sched_setaffinity(thread.id, cpus)


def evaluation_path(run_name: str, step: int) -> Path:
    return RUNS / run_name / f"eval-{step}.json"


def source_digest() -> str:
    """SHA-256 over the checkout's code and configuration: each file's path relative to ROOT and its bytes, in path
    order (bytecode caches excluded), so a record names the code that produced it; a checkout's digest is computed the
    same way."""
    files = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in (".py", ".toml", ".lock")]
    files += [p for tree in SOURCE_TREES for p in (ROOT / tree).rglob("*")
              if p.is_file() and "__pycache__" not in p.parts]
    digest = hashlib.sha256()
    for p in sorted(files):
        digest.update(p.relative_to(ROOT).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return digest.hexdigest()


def write_atomically(path: Path, data: bytes) -> None:
    """path holds data or its previous state, never a part (run.job takes an existing eval-<step>.json as done)."""
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_evaluation(path: Path, evaluation: dict) -> None:
    """Write an evaluation: the record at path with "samples" {"file", "sha256"} (null without generation benchmarks)
    in place of every sampled and reasoning record's "completions", which go to eval-<step>.samples.json.gz beside it,
    gzip-compressed JSON {benchmark: each task's sampled completions} and {"reasoning/<benchmark>": each task's
    responses}. NaN and infinities raise."""
    record, completions, samples = dict(evaluation), {}, None
    if evaluation["codegen"] is not None:
        sampled = {name: records["sampled"] for name, records in evaluation["codegen"].items() if "sampled" in records}
        record["codegen"] = {name: records | ({"sampled": {k: v for k, v in sampled[name].items()
                                                           if k != "completions"}} if name in sampled else {})
                             for name, records in evaluation["codegen"].items()}
        completions |= {name: r["completions"] for name, r in sampled.items()}
    if evaluation["reasoning"] is not None:
        record["reasoning"] = {name: {k: v for k, v in r.items() if k != "completions"}
                               for name, r in evaluation["reasoning"].items()}
        completions |= {f"reasoning/{name}": r["completions"] for name, r in evaluation["reasoning"].items()}
    if completions:
        data = gzip.compress(json.dumps(completions, allow_nan=False).encode(), mtime=0)
        sidecar = path.with_name(f"{path.stem}.samples.json.gz")
        write_atomically(sidecar, data)
        samples = {"file": sidecar.name, "sha256": hashlib.sha256(data).hexdigest()}
    write_atomically(path, json.dumps(record | {"samples": samples}, allow_nan=False).encode())


def resolved(records: dict) -> dict:
    """records with each future (code_eval's scoring) replaced by its result, in the same key order."""
    return {k: v.result() if isinstance(v, Future) else resolved(v) for k, v in records.items()}


def evaluate(tokenizer: Tokenizer, *, run_name: str, step: int, generate: Literal["benchmarks", "parse", "reasoning"],
             previous: "Event | None", released: "Event") -> None:
    """Write evaluation_path(run_name, step) of the checkpoint; with generate "parse", codegen, fim and
    evalplus_average are null; "reasoning" adds to "benchmarks" the reasoning protocol (docs/post-training.md §11.4),
    null in the other modes. "evalplus_average": metrics.evalplus_average of each EvalPlus mode.
    The process tree keeps to verify.cores()'s decode cores until the last GPU phase ends, then frees the GPU, moves
    to the timed-check cores and sets `released` while scoring finishes. previous: set once the previous checkpoint's
    evaluation has ended; official scoring and timed checks wait for it, so they never share the CPU with another
    checkpoint's (timed verdicts depend on load). "provenance": source digest (taken at the start), environment
    (eval/cache.py), start and end times (Unix seconds)."""
    started, source = time.time(), source_digest()
    decode, timed = cores()  # the starting affinity's split, taken before the pin
    os.environ["RAYON_NUM_THREADS"] = str(len(decode))  # rayon sizes its pool once, at first use
    pin(decode)
    benchmarks = tasks()
    model = load_model(run_name, step, device=torch.device("cuda"))
    full = generate != "parse"
    codegen = fim = reasoning = None
    with sanitizers() as pool, ThreadPoolExecutor(1) as scorers, ThreadPoolExecutor(1) as background:
        if previous is not None:
            scorers.submit(previous.wait)
        # the filtered searches' vetted examples, CPU work (untimed reference runs) during the GPU phases below
        found = background.submit(vetted, benchmarks) if full else None
        parse = parse_generation(model, tokenizer, benchmarks, pool=pool, scorers=scorers)
        if full:
            codegen = generation(model, tokenizer, benchmarks, pool=pool, scorers=scorers)
            fim = fim_generation(model, tokenizer, scorers=scorers)
        if generate == "reasoning":
            reasoning = reasoning_generation(model, tokenizer, benchmarks, pool=pool, scorers=scorers)
        results = likelihood(model, tokenizer, benchmarks, device=torch.device("cuda"))
        if full:
            if previous is not None:
                previous.wait()
            filtered, fim_filtered = filtered_generation(model, tokenizer, benchmarks, pool=pool, scorers=scorers,
                                                         found=found.result())
            for name, record in filtered.items():
                codegen[name][FILTERED] = record
            for key, record in fim_filtered.items():
                fim[key][FILTERED] = record
        del model
        torch._dynamo.reset()  # compiled graphs and their guards' references
        gc.collect()  # the decoders' native memory goes with their objects (NativeDecoder's finalizer)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"GPU released: allocated {torch.cuda.memory_allocated()}, reserved {torch.cuda.memory_reserved()}, "
              f"free {free} of {total} bytes", flush=True)
        pin(timed)
        released.set()
        results |= {"parse": resolved(parse), "codegen": None if codegen is None else resolved(codegen),
                    "fim": None if fim is None else resolved(fim),
                    "reasoning": None if reasoning is None else resolved(reasoning)}
    results["evalplus_average"] = None if codegen is None else {
        mode: evalplus_average(results["codegen"]["humaneval_plus"][mode], results["codegen"]["mbpp_plus"][mode])
        for mode in ("greedy", "sampled", FILTERED)}
    provenance = {"source_sha256": source, "environment": json.loads(environment()), "started": started,
                  "finished": time.time()}
    write_evaluation(evaluation_path(run_name, step),
                     {"run_name": run_name, "step": step} | results | {"provenance": provenance})


def record(run_name: str, log: list[dict]) -> dict:
    """A run's non-train log records and its checkpoints' evaluations in step order (a diverged run has none)."""
    paths = (RUNS / run_name).glob("eval-*.json")
    return {"run_name": run_name, "records": [r for r in log if r["kind"] != "train"],
            "evaluations": sorted((json.loads(p.read_text(encoding="utf-8")) for p in paths), key=lambda r: r["step"])}
