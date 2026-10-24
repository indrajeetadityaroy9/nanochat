"""
Curate the sources the mixture reads (mixture.READ) from their standardized files (prepare.py), in the steps of the
design document (normalization_doc.md §4):

1. check: every Python file, a repository's, a synthetic one or a function, must parse with ast.parse (Python 3.12).
   A file that does not is repaired by the smallest set of steps that makes it parse (repaired): Stack v3's
   placeholders <EMAIL>, <KEY>, <NAME> and <PASSWORD> become the bare word; leading-indentation tabs are expanded to 8
   columns, Python 2's rule (Python 2 Language Reference §2.1.8); every 2to3 fixer of fissix (the maintained fork of
   lib2to3, which Python 3.12 lacks) is applied. The repaired text is kept and its id recomputed; a file that never
   parses is dropped (checked_path).
2. dedup: one row per id over every source and both splits, the first in mixture.READ's order, then file and row; an
   empty text is dropped.
3. split: a group is val when md5_number(group) % SPLIT = 0 (schema.py); a repository's group is its lowercased name.
4. shards (curated_dir): a repository source's files as the partitions of that residue (partition 0 val, 1..SPLIT-1
   train), each sorted by (repository, path, id) and cut into import-connected samples by the data crate's
   repository_samples (SAMPLE_FILE); any other source's train rows in id order, a uniform order as id is a hash, over
   SPLIT shards (1% of the source each, as a repository partition), and its val rows in one shard.

python -m nanochat.corpus.curate
"""

import ast
import collections
import functools
import hashlib
import itertools
import multiprocessing
import os
import re
import shutil
import sys
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from nanochat.corpus import mixture
from nanochat.corpus.prepare import COMPRESSION, sha1
from nanochat.corpus.schema import CONVERSATION, FUNCTION, ISSUE, REPOSITORY_FILE, SAMPLE_FILE, SPLIT, SYNTHETIC_FILE
from nanochat.corpus.sources import SOURCES, checked_path, curated_dir, standardized_path
from nanochat.native import data_crate
from nanochat.paths import CORPUS

SPAWN = multiprocessing.get_context("spawn")
# The id of an empty text
EMPTY = hashlib.sha1(b"").hexdigest()
# Half the machine's memory: the lower end of DuckDB's guidance, as some operations bypass the buffer manager's limit
# (duckdb.org/docs/lts/guides/troubleshooting/oom_errors) and other jobs share the unified memory.
MEMORY = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2
# Row groups in flight: one per core, so a core finishing one takes up the next
WINDOW = os.cpu_count()
PLACEHOLDER = re.compile(r"<(EMAIL|KEY|NAME|PASSWORD)>")
LEADING = re.compile(r"^[ \t]+", re.MULTILINE)
# The recursion limit of the measurement the check reproduces (normalization_doc.md §4.2): deep enough for real
# files, shallow enough that ast.parse does not overflow the C stack
RECURSION = 6_000


def residue(group: str) -> str:
    """DuckDB's residue of a group column, md5_number(group) % SPLIT: 0 holds the group out as val."""
    return f"md5_number({group}) % {SPLIT}"


def database() -> duckdb.DuckDBPyConnection:
    """DuckDB in MEMORY, spilling to CORPUS/spill, without insertion order (the guide cited at MEMORY)."""
    return duckdb.connect(config={"memory_limit": f"{MEMORY}B", "temp_directory": str(CORPUS / "spill"),
                                  "preserve_insertion_order": False})


def in_order(pool: ProcessPoolExecutor, fn: Callable, tasks: Iterable) -> Iterator:
    """fn of each task, WINDOW in flight, in order: no more results wait in memory than the window holds."""
    pending: collections.deque[Future] = collections.deque()
    for task in tasks:
        pending.append(pool.submit(fn, task))
        if len(pending) == WINDOW:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


# ---- 1. check

def parses(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except (SyntaxError, ValueError, MemoryError, RecursionError):  # ValueError: a null byte
        return False


@functools.cache
def fixers():
    """fissix's tool with every fixer of fissix.fixes, made once per process."""
    from fissix import refactor
    return refactor.RefactoringTool(refactor.get_fixers_from_package("fissix.fixes"))


def two_to_three(text: str) -> str | None:
    """text through every fixer of fissix, or None when its Python 2 grammar rejects it."""
    try:
        return str(fixers().refactor_string(text if text.endswith("\n") else text + "\n", "<file>"))
    except Exception:  # fissix raises its grammar's parse and tokenize errors and more on what it cannot read
        return None


def bare_placeholders(text: str) -> str:
    return PLACEHOLDER.sub(lambda m: m.group(1), text)


def expanded_tabs(text: str) -> str:
    return LEADING.sub(lambda m: m.group(0).expandtabs(8), text)


def repaired(text: str) -> str | None:
    """text if it parses, else its smallest repair that parses, else None: of the steps that change it (placeholders,
    tabs, 2to3, in that order), every subset by size and then in step order, its steps applied in step order; so a file
    that 2to3 alone repairs keeps its tabs and placeholders."""
    if parses(text):
        return text
    steps = [step for step in (bare_placeholders, expanded_tabs) if step(text) != text] + [two_to_three]
    for size in range(1, len(steps) + 1):
        for subset in itertools.combinations(steps, size):
            result: str | None = text
            for step in subset:
                result = step(result)
                if result is None:  # 2to3's grammar rejects it
                    break
            if result is not None and parses(result):
                return result
    return None


def check_row_group(task: tuple[str, int, int]) -> pa.Table:
    """Row group g of a Python source's standardized file k: the files that parse after repair, repaired, ids
    recomputed."""
    source, k, g = task
    sys.setrecursionlimit(RECURSION)
    table = pq.ParquetFile(standardized_path(source, k)).read_row_group(g)
    texts = [repaired(text) for text in table["content"].to_pylist()]
    table = table.filter(pa.array([text is not None for text in texts], pa.bool_()))
    content = pa.chunked_array([pa.array([text for text in texts if text is not None], pa.string())])
    table = table.set_column(table.schema.get_field_index("content"), "content", content)
    return table.set_column(table.schema.get_field_index("id"), "id", sha1(content)).cast(SOURCES[source].schema)


def check(source: str, files: int, pool: ProcessPoolExecutor) -> None:
    """checked_path of each of a Python source's standardized files, written whole, then renamed into place (an
    existing file is complete and kept). The row groups of the files left are one stream, so a file of few row groups
    does not leave the pool idle while its slowest group finishes."""
    groups = {k: pq.ParquetFile(standardized_path(source, k)).num_row_groups for k in range(files)
              if not checked_path(source, k).exists()}
    tasks = [(source, k, g) for k, n in groups.items() for g in range(n)]
    writers: dict[int, pq.ParquetWriter] = {}
    for (_, k, g), table in zip(tasks, in_order(pool, check_row_group, tasks), strict=True):
        path = checked_path(source, k)
        partial = path.with_name(path.name + ".partial")
        if g == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            writers[k] = pq.ParquetWriter(partial, SOURCES[source].schema, compression=COMPRESSION)
        writers[k].write_table(table)
        if g == groups[k] - 1:
            writers.pop(k).close()
            partial.rename(path)


# ---- 2. dedup

def duplicates(paths: dict[str, list[Path]]) -> dict[str, pa.Table]:
    """{source: (file, row) of each row to drop}: every row that is not the first of its id (the first in the order of
    paths' sources, then file and row), and every empty text."""
    union = " UNION ALL ".join(
        f"SELECT id, {i} AS source, file_index AS file, file_row_number AS row "
        f"FROM read_parquet({[str(p) for p in files]}, file_row_number = true)"
        for i, files in enumerate(paths.values()))
    with database() as con:
        drops = con.execute(f"SELECT source, file, row FROM (SELECT id, source, file, row, row_number() OVER "
                            f"(PARTITION BY id ORDER BY source, file, row) AS n FROM ({union})) "
                            f"WHERE n > 1 OR id = '{EMPTY}'").fetch_arrow_table()
    return {source: drops.filter(pc.equal(drops["source"], i)).select(["file", "row"])
            for i, source in enumerate(paths)}


# ---- 3 and 4. split and shards

def shuffle_dir(source: str) -> Path:
    return CORPUS / "shuffle" / source


def shuffle(source: str, paths: list[Path], drops: pa.Table) -> None:
    """A repository source's rows without the duplicates, as Hive partitions shuffle/<source>/part=<p>/ by the residue
    of the repository (its name lowercased: GitHub's names are case-insensitive) (DuckDB COPY ... PARTITION_BY, which
    creates the partition directories, not their parent)."""
    shuffle_dir(source).parent.mkdir(exist_ok=True)
    with database() as con:
        con.register("drops", drops)
        con.execute(f"COPY (SELECT n.repo_name, n.relative_path, n.content, n.id, {residue('lower(n.repo_name)')} AS "
                    f"part FROM read_parquet({[str(p) for p in paths]}, file_row_number = true) AS n "
                    f"ANTI JOIN drops AS d ON d.file = n.file_index AND d.row = n.file_row_number) "
                    f"TO '{shuffle_dir(source)}' (FORMAT parquet, COMPRESSION {COMPRESSION}, PARTITION_BY (part))")


def partition_row_groups(source: str, part: int) -> list[pq.RowGroupMetaData]:
    """The row-group metadata of the files of a shuffle partition."""
    return [md.row_group(g) for md in map(pq.read_metadata, (shuffle_dir(source) / f"part={part}").iterdir())
            for g in range(md.num_row_groups)]


def curate_partition(task: tuple[str, int]) -> None:
    """One partition as a curated shard (partition 0 is val), as nanochat/data/curate.py makes it: its files sorted by
    (repository, path, id) into one chunk, the repository its lowercased name, so a repository is a zero-copy slice
    (large_string: 64-bit offsets hold a partition beyond 2 GiB), cut into samples (repository_samples) and written in
    sample order, one row group per write of whole samples, each but the last at least as long as the partition's
    longest row group."""
    source, part = task
    length = max(row_group.num_rows for row_group in partition_row_groups(source, part))
    table = pq.read_table(shuffle_dir(source) / f"part={part}",
                          schema=pa.schema([("repo_name", pa.string()), ("relative_path", pa.string()),
                                            ("content", pa.large_string()), ("id", pa.string())]))
    table = table.append_column("repository", pc.utf8_lower(table["repo_name"]))
    table = table.sort_by([(key, "ascending") for key in ("repository", "relative_path", "id")]).combine_chunks()
    paths, contents = table["relative_path"], table["content"]
    runs = [0, *pc.run_end_encode(table["repository"].chunk(0)).run_ends.to_numpy()]
    groups = [np.add(group, a) for a, b in itertools.pairwise(runs)
              for group in data_crate().repository_samples(paths.slice(a, b - a).to_pylist(),
                                                           contents.slice(a, b - a).to_pylist(),
                                                           grammar=SOURCES[source].language)]
    order, samples = np.concatenate(groups), np.repeat(np.arange(len(groups)), [len(group) for group in groups])
    cuts = [0]
    for start in np.flatnonzero(np.diff(samples, prepend=-1)).tolist():
        if start - cuts[-1] >= length:
            cuts.append(start)
    columns = table.select(["repo_name", "relative_path", "content"])
    split = "val" if part == 0 else "train"
    with pq.ParquetWriter(curated_dir(source, split) / f"{part}.parquet", SAMPLE_FILE,
                          compression=COMPRESSION) as writer:
        for a, b in itertools.pairwise([*cuts, len(order)]):
            sample = columns.take(order[a:b]).append_column("sample", pa.array(samples[a:b]))
            writer.write_table(sample.cast(SAMPLE_FILE), row_group_size=b - a)


def shard_documents(source: str, paths: list[Path], drops: pa.Table) -> None:
    """A document source's rows without the dropped ones: val rows in one shard, train rows in id order over SPLIT
    shards."""
    spec = SOURCES[source]
    group = {CONVERSATION: "n.problem_id", ISSUE: "lower(n.repo_name)", FUNCTION: "lower(n.repo_name)"}.get(
        spec.schema, "n.id")
    columns = ", ".join(f"n.{name}" for name in spec.schema.names)
    with database() as con:
        con.register("drops", drops)
        con.execute(f"CREATE TABLE kept AS SELECT {columns}, {residue(group)} = 0 AS val "
                    f"FROM read_parquet({[str(p) for p in paths]}, file_row_number = true) AS n "
                    f"ANTI JOIN drops AS d ON d.file = n.file_index AND d.row = n.file_row_number")
        names = ", ".join(spec.schema.names)
        con.execute(f"COPY (SELECT {names} FROM kept WHERE val ORDER BY id) "
                    f"TO '{curated_dir(source, 'val')}/0.parquet' (FORMAT parquet, COMPRESSION {COMPRESSION})")
        con.execute(f"CREATE TABLE train AS SELECT {names}, ntile({SPLIT}) OVER (ORDER BY id) - 1 AS shard "
                    f"FROM kept WHERE NOT val")
        for k in range(SPLIT):
            con.execute(f"COPY (SELECT {names} FROM train WHERE shard = {k} ORDER BY id) "
                        f"TO '{curated_dir(source, 'train')}/{k}.parquet' (FORMAT parquet, COMPRESSION {COMPRESSION})")


def curate() -> None:
    """Every read source's curated shards, from its standardized files (the Python repository sources' checked ones)."""
    data_crate()  # built once here, before the workers load it
    files = {source: len(list(standardized_path(source, 0).parent.iterdir())) for source in mixture.READ}
    python = [s for s in mixture.READ if SOURCES[s].schema in (REPOSITORY_FILE, SYNTHETIC_FILE, FUNCTION)
              and SOURCES[s].language == "Python"]
    with ProcessPoolExecutor(mp_context=SPAWN) as pool:
        for source in python:
            check(source, files[source], pool)
    paths = {source: [(checked_path if source in python else standardized_path)(source, k)
                      for k in range(files[source])] for source in mixture.READ}
    drops = duplicates(paths)
    for source in mixture.READ:
        print(f"{source}: {sum(pq.read_metadata(p).num_rows for p in paths[source]):,} rows, "
              f"{drops[source].num_rows:,} duplicates", flush=True)
        for split in ("train", "val"):
            curated_dir(source, split).mkdir(parents=True)
        if SOURCES[source].schema is REPOSITORY_FILE:
            shuffle(source, paths[source], drops[source])
            # a partition worker holds its partition twice at peak (the table and its sorted copy)
            largest = max(sum(row_group.total_byte_size for row_group in partition_row_groups(source, part))
                          for part in range(SPLIT))
            with ProcessPoolExecutor(min(os.cpu_count(), MEMORY // (2 * largest)), mp_context=SPAWN) as pool:
                list(pool.map(curate_partition, [(source, part) for part in range(SPLIT)]))
            shutil.rmtree(shuffle_dir(source))
        else:
            shard_documents(source, paths[source], drops[source])


if __name__ == "__main__":
    curate()
