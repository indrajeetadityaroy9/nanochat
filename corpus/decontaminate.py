"""
Benchmark decontamination of the curated train splits: a row whose text shares a 10-gram of whitespace-separated words
with a scored benchmark's prompt or reference program is dropped (Qwen2.5-Coder's rule, arXiv 2409.12186 §5; the
10-gram filter of DeepSeek-Coder, arXiv 2401.14196 §2.4). Val splits are untouched (held out, never trained). A shard
with a dropped row is rewritten with its row groups (a repository source's: whole samples) less those rows; RECORD
lists every dropped row, written once every shard is matched and before any is rewritten. Shards in parallel, a process
per CPU. Runs once, on curate's shards.

python -m corpus.decontaminate
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from human_eval_infilling.data import read_problems

from corpus.parallel import SPAWN
from corpus.schema import COMPRESSION
from corpus.sources import CURATED, SOURCES, curated_shards
from native import corpus_crate

N = 10
# The dropped rows, beside the shards (publish.push uploads it with them)
RECORD = CURATED / "decontamination.json"
GRAMS = None  # a worker's corpus_crate().Grams of the benchmark texts (worker_init)


def benchmark_texts() -> dict[str, str]:
    """{benchmark/task: text} of every scored task: its prompt and reference solution (MultiPL-E's prompt alone, it has
    none), and each HumanEval infilling program (prompt, reference middle and suffix; its three benchmarks' problems
    are cut from one program per HumanEval task). Imported here: the shard workers need neither torch nor eval."""
    from eval.code_eval import INFILLING, tasks

    texts = {f"{name}/{t.task_id}": t.prompt + (t.canonical_solution or "") for name, items in tasks().items()
             for t in items}
    for name in INFILLING:
        for p in read_problems(name).values():
            task = "infilling/" + "/".join(p["task_id"].split("/")[1:3])
            texts[task] = p["prompt"] + p["canonical_solution"] + p["suffix"]
    return texts


def shard_matches(grams, path: Path) -> list[tuple[int, list[int]]]:
    """(row, the matched benchmark texts' indices) of each row of a shard sharing an N-gram with a benchmark text."""
    found, row = [], 0
    file = pq.ParquetFile(path)
    for g in range(file.num_row_groups):
        contents = file.read_row_group(g, columns=["content"])["content"].to_pylist()
        found += [(row + i, m) for i, m in enumerate(grams.matches(contents)) if m]
        row += len(contents)
    return found


def drop_rows(path: Path, rows: list[int]) -> None:
    """The shard without rows (indices into the file), each row group kept, atomically."""
    file = pq.ParquetFile(path)
    tmp = path.with_name(f"{path.name}.tmp")
    drop, start = np.asarray(rows), 0
    with pq.ParquetWriter(tmp, file.schema_arrow, compression=COMPRESSION) as writer:
        for g in range(file.num_row_groups):
            table = file.read_row_group(g)
            keep = ~np.isin(np.arange(start, start + table.num_rows), drop)
            start += table.num_rows
            if keep.any():
                writer.write_table(table.filter(keep), row_group_size=int(keep.sum()))
    tmp.replace(path)


def worker_init(texts: list[str]) -> None:
    """The worker's GRAMS; one matching thread each, the pool running a shard per CPU."""
    global GRAMS
    os.environ["RAYON_NUM_THREADS"] = "1"  # read when the crate builds rayon's pool, at its import
    GRAMS = corpus_crate().Grams(texts, n=N)


def worker_matches(path: Path) -> list[tuple[int, list[int]]]:
    """shard_matches of a shard with the worker's GRAMS."""
    return shard_matches(GRAMS, path)


def main() -> None:
    names, texts = zip(*benchmark_texts().items(), strict=True)
    corpus_crate()  # built once here, before the workers load it
    with RECORD.open("x") as record:  # once: a second run finds nothing and would record nothing
        shards = [(source, path) for source in SOURCES for path in curated_shards(source, "train")]
        with ProcessPoolExecutor(mp_context=SPAWN, initializer=worker_init, initargs=(list(texts),)) as pool:
            found = list(pool.map(worker_matches, [path for _, path in shards]))
            dropped = {source: [] for source in SOURCES}
            for (source, path), rows in zip(shards, found, strict=True):
                dropped[source] += [[int(path.stem), row, tests] for row, tests in rows]
            # each dropped row: [shard, its row in the shard before decontamination, indices into "benchmarks"];
            # written before any shard is rewritten, so a run stopped while rewriting leaves the record whole
            json.dump({"n": N, "benchmarks": names, "dropped": dropped}, record)
            record.flush()
            rewritten = [(path, [row for row, _ in rows]) for (_, path), rows in zip(shards, found, strict=True) if rows]
            list(pool.map(drop_rows, *zip(*rewritten, strict=True)))
        for source, rows in dropped.items():
            print(f"{source}: {len(rows):,} train rows dropped", flush=True)


if __name__ == "__main__":
    main()
