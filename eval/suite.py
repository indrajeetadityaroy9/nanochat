"""
A checkpoint's evaluation (RUNS/<run_name>/eval-<step>.json: likelihoods, the parse measure and the generation
benchmarks) and a run's record (its non-train log records and evaluations).
"""
import contextlib
import gc
import gzip
import hashlib
import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import psutil
import torch

from corpus.tokenizer import Tokenizer
from eval.bpb import likelihood
from eval.code_eval import filtered_generation, fim_generation, generation, parse_generation, sanitizers, tasks, vetted
from eval.verify import cores
from paths import RUNS
from train.checkpoint import load_model

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event


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


def write_evaluation(path: Path, evaluation: dict) -> None:
    """Write an evaluation: the record at path with "samples" {"file", "sha256"} (null without generation benchmarks)
    in place of every sampled record's "completions", which go to eval-<step>.samples.json.gz beside it,
    gzip-compressed JSON {benchmark: each task's completions}. NaN and infinities raise."""
    record, samples = dict(evaluation), None
    if evaluation["codegen"] is not None:
        record["codegen"] = {name: records | {"sampled": {k: v for k, v in records["sampled"].items()
                                                          if k != "completions"}}
                             for name, records in evaluation["codegen"].items()}
        data = gzip.compress(json.dumps({name: records["sampled"]["completions"]
                                         for name, records in evaluation["codegen"].items()},
                                        allow_nan=False).encode(), mtime=0)
        sidecar = path.with_name(f"{path.stem}.samples.json.gz")
        sidecar.write_bytes(data)
        samples = {"file": sidecar.name, "sha256": hashlib.sha256(data).hexdigest()}
    path.write_text(json.dumps(record | {"samples": samples}, allow_nan=False), encoding="utf-8")


def resolved(records: dict) -> dict:
    """records with each future (code_eval's scoring) replaced by its result, in the same key order."""
    return {k: v.result() if isinstance(v, Future) else resolved(v) for k, v in records.items()}


def evaluate(tokenizer: Tokenizer, *, run_name: str, step: int, generate: Literal["benchmarks", "parse"],
             previous: "Event | None", released: "Event") -> None:
    """Write evaluation_path(run_name, step) of the checkpoint, loaded onto the GPU; with generate "parse", codegen and
    fim are null. The process tree keeps to verify.cores()'s decode cores (rayon's pool sized to them), its timed
    checks to the timed-check cores (verify.timed). After the last GPU phase the model, decoders and compiled graphs are
    dropped, the allocator's cache emptied, the process tree moved to the timed-check cores and `released` set, while
    scoring finishes. previous: set once the previous checkpoint's evaluation has ended; the official scoring and the
    filtered searches' timed checks wait for it, so they never share the CPU with another checkpoint's. Results do not
    depend on the GPU order except timed checks, which heavy CPU load can fail."""
    decode, timed = cores()  # the starting affinity's split, taken before the pin
    os.environ["RAYON_NUM_THREADS"] = str(len(decode))  # rayon sizes its pool once, at first use
    pin(decode)
    benchmarks = tasks()
    model = load_model(run_name, step, device=torch.device("cuda"))
    codegen = fim = None
    with sanitizers() as pool, ThreadPoolExecutor(1) as scorers, ThreadPoolExecutor(1) as background:
        if previous is not None:
            scorers.submit(previous.wait)
        # the filtered searches' vetted examples, CPU work (untimed reference runs) during the GPU phases below
        found = background.submit(vetted, benchmarks) if generate == "benchmarks" else None
        parse = parse_generation(model, tokenizer, benchmarks, pool=pool, scorers=scorers)
        if generate == "benchmarks":
            codegen = generation(model, tokenizer, benchmarks, pool=pool, scorers=scorers)
            fim = fim_generation(model, tokenizer, scorers=scorers)
        results = likelihood(model, tokenizer, benchmarks, device=torch.device("cuda"))
        if generate == "benchmarks":
            if previous is not None:
                previous.wait()
            filtered, fim_filtered = filtered_generation(model, tokenizer, benchmarks, pool=pool, scorers=scorers,
                                                         found=found.result())
            for name, record in filtered.items():
                codegen[name]["filtered"] = record
            for key, record in fim_filtered.items():
                fim[key]["filtered"] = record
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
                    "fim": None if fim is None else resolved(fim)}
    write_evaluation(evaluation_path(run_name, step), {"run_name": run_name, "step": step} | results)


def record(run_name: str, log: list[dict]) -> dict:
    """A run's non-train log records and its checkpoints' evaluations in step order (a diverged run has none)."""
    paths = (RUNS / run_name).glob("eval-*.json")
    return {"run_name": run_name, "records": [r for r in log if r["kind"] != "train"],
            "evaluations": sorted((json.loads(p.read_text(encoding="utf-8")) for p in paths), key=lambda r: r["step"])}
