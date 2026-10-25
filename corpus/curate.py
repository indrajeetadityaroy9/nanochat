"""
Curate every source from its standardized files:

1. check: every Python file (a category whose content is code) must parse with ast.parse (Python 3.12); one that
   does not is repaired by the smallest set of steps that makes it parse (repaired): Stack v3's placeholders <EMAIL>,
   <KEY>, <NAME>, <PASSWORD> become the bare word; leading tabs expand to 8 columns (Python 2 Language Reference
   §2.1.8); fissix's 2to3 fixers. The repaired text is kept with its id recomputed; a file that never parses is dropped.
2. dedup: one row per id over every source and split, the first in SOURCES order, then file and row; empty
   texts dropped.
3. split: val iff md5_number(group) % SPLIT = 0 (schema.py).
4. shards: a repository source's files by that residue's partitions (0 val, 1..SPLIT-1 train), each sorted by
   (repository, path, id) and cut into import-connected samples (the crate's repository_samples; SAMPLE_FILE); any
   other source's train rows in id order (a uniform order, id being a hash) over SPLIT shards, its val rows in one.

Every output, rows, order and row groups, is a function of the standardized files.

python -m corpus.curate
"""

import ast
import functools
import hashlib
import itertools
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fissix import refactor

from corpus.parallel import MEMORY, SPAWN, ordered
from corpus.schema import COMPRESSION, REPOSITORY_FILE, SAMPLE_FILE, SPLIT, sha1
from corpus.sources import (
    SOURCES,
    checked_path,
    curated_dir,
    footer_rows,
    repository_samples,
    sample_starts,
    standardized_path,
)
from native import corpus_crate
from paths import CORPUS

# The id of an empty text
EMPTY = hashlib.sha1(b"").hexdigest()
# Row groups in flight: four per core, an engineering choice: the cores keep working while the oldest one, whose
# result is written next, runs long (a row group's time follows its slowest files), at the memory of the results held
WINDOW = 4 * os.cpu_count()
# The fewest files of a repository shard's row group (the last excepted), whole samples each, so a reader of row
# groups never splits a sample; 2^16, an engineering choice: a row group is compile's task, so smaller groups spread a
# shard over more workers and larger ones cost fewer tasks
SAMPLE_ROWS = 65_536
PLACEHOLDER = re.compile(r"<(EMAIL|KEY|NAME|PASSWORD)>")
LEADING = re.compile(r"^[ \t]+", re.MULTILINE)
# ast.parse's recursion limit, an engineering choice above Python's default 1,000: higher parses more deeply nested
# files, too high overflows the C stack before RecursionError
RECURSION = 6_000


def residue(group: str) -> str:
    """DuckDB's residue of a group, md5_number(group) % SPLIT: 0 holds the group out as val."""
    return f"md5_number({group}) % {SPLIT}"


def database() -> duckdb.DuckDBPyConnection:
    """DuckDB in MEMORY, spilling to CORPUS/spill, without insertion order (the guide cited at parallel.MEMORY)."""
    return duckdb.connect(config={"memory_limit": f"{MEMORY}B", "temp_directory": str(CORPUS / "spill"),
                                  "preserve_insertion_order": False})


# ---- 1. check

def parses(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except (SyntaxError, ValueError, MemoryError, RecursionError):  # ValueError: a null byte
        return False


@functools.cache
def fixers() -> refactor.RefactoringTool:
    """fissix's tool with every fixer of fissix.fixes, made once per process."""
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
    schema = SOURCES[source].category.schema
    table = pq.ParquetFile(standardized_path(source, k)).read_row_group(g)
    texts = pa.array([repaired(text) for text in table["content"].to_pylist()], pa.string())
    content = texts.drop_null()
    table = table.filter(texts.is_valid())
    table = table.set_column(schema.get_field_index("content"), schema.field("content"), content)
    return table.set_column(schema.get_field_index("id"), schema.field("id"), sha1(content))


def check(source: str, files: int, pool: ProcessPoolExecutor) -> None:
    """checked_path of each of a Python source's standardized files (written to .partial, then renamed; an existing
    file is kept), the remaining files' row groups as one stream so no file's slowest group idles the pool."""
    groups = {k: pq.ParquetFile(standardized_path(source, k)).num_row_groups for k in range(files)
              if not checked_path(source, k).exists()}
    tasks = [(source, k, g) for k, n in groups.items() for g in range(n)]
    writers: dict[int, pq.ParquetWriter] = {}
    for (_, k, g), table in zip(tasks, ordered(pool, check_row_group, tasks, WINDOW), strict=True):
        path = checked_path(source, k)
        partial = path.with_name(path.name + ".partial")
        if g == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            writers[k] = pq.ParquetWriter(partial, SOURCES[source].category.schema, compression=COMPRESSION)
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
                            f"WHERE n > 1 OR id = '{EMPTY}'").to_arrow_table()
    return {source: drops.filter(pc.equal(drops["source"], i)).select(["file", "row"])
            for i, source in enumerate(paths)}


# ---- 3 and 4. split and shards

def shuffle_dir(source: str) -> Path:
    return CORPUS / "shuffle" / source


def undropped(paths: list[Path]) -> str:
    """A DuckDB FROM clause of the rows of paths that the registered table drops does not list, as n."""
    return (f"read_parquet({[str(p) for p in paths]}, file_row_number = true) AS n "
            f"ANTI JOIN drops AS d ON d.file = n.file_index AND d.row = n.file_row_number")


def shuffle(source: str, paths: list[Path], drops: pa.Table) -> None:
    """A repository source's rows without the duplicates, as Hive partitions shuffle/<source>/part=<p>/ by the residue
    of its repository (DuckDB COPY ... PARTITION_BY, which creates the partition directories, not their parent)."""
    shuffle_dir(source).parent.mkdir(exist_ok=True)
    with database() as con:
        con.register("drops", drops)
        con.execute(f"COPY (SELECT n.repo_name, n.relative_path, n.content, n.id, {residue(REPOSITORY_FILE.group)} "
                    f"AS part FROM {undropped(paths)}) "
                    f"TO '{shuffle_dir(source)}' (FORMAT parquet, COMPRESSION {COMPRESSION}, PARTITION_BY (part))")


def curate_partition(task: tuple[str, int]) -> None:
    """One partition as a curated shard (partition 0 is val): its files sorted by (lowercased repository, path, id),
    cut into samples (sources.repository_samples) and written in sample order, a row group per SAMPLE_ROWS files or
    more of whole samples."""
    source, part = task
    table = pq.read_table(shuffle_dir(source) / f"part={part}",
                          schema=pa.schema([("repo_name", pa.string()), ("relative_path", pa.string()),
                                            ("content", pa.large_string()), ("id", pa.string())]))
    table, _, groups = repository_samples(table, ("relative_path", "id"), SOURCES[source].language, one_per_path=False)
    order, samples = np.concatenate(groups), np.repeat(np.arange(len(groups)), [len(group) for group in groups])
    cuts = [0]
    for start in sample_starts(samples):
        if start - cuts[-1] >= SAMPLE_ROWS:
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
    category = SOURCES[source].category
    columns = ", ".join(f"n.{name}" for name in category.schema.names)
    names = ", ".join(category.schema.names)
    with database() as con:
        con.register("drops", drops)
        con.execute(f"CREATE TABLE kept AS SELECT {columns}, {residue(category.group)} = 0 AS val "
                    f"FROM {undropped(paths)}")
        con.execute(f"COPY (SELECT {names} FROM kept WHERE val ORDER BY id) "
                    f"TO '{curated_dir(source, 'val')}/0.parquet' (FORMAT parquet, COMPRESSION {COMPRESSION})")
        con.execute(f"CREATE TABLE train AS SELECT {names}, ntile({SPLIT}) OVER (ORDER BY id) - 1 AS shard "
                    f"FROM kept WHERE NOT val")
        for k in range(SPLIT):
            con.execute(f"COPY (SELECT {names} FROM train WHERE shard = {k} ORDER BY id) "
                        f"TO '{curated_dir(source, 'train')}/{k}.parquet' (FORMAT parquet, COMPRESSION {COMPRESSION})")


def main() -> None:
    """Every source's curated shards, from its standardized files (a Python source's checked ones)."""
    corpus_crate()  # built once here, before the workers load it
    files = {source: len(list(standardized_path(source, 0).parent.iterdir())) for source in SOURCES}
    python = [source for source, spec in SOURCES.items() if spec.category.code and spec.language == "Python"]
    with ProcessPoolExecutor(mp_context=SPAWN) as pool:
        for source in python:
            check(source, files[source], pool)
    paths = {source: [(checked_path if source in python else standardized_path)(source, k)
                      for k in range(files[source])] for source in SOURCES}
    drops = duplicates(paths)
    for source, spec in SOURCES.items():
        print(f"{source}: {footer_rows(paths[source]):,} rows, "
              f"{drops[source].num_rows:,} duplicates", flush=True)
        for split in ("train", "val"):
            curated_dir(source, split).mkdir(parents=True)
        if spec.category == REPOSITORY_FILE:
            shuffle(source, paths[source], drops[source])
            # a partition worker holds its partition twice at peak (the table and its sorted copy)
            largest = max(sum(md.row_group(g).total_byte_size for md in map(pq.read_metadata, part.iterdir())
                              for g in range(md.num_row_groups)) for part in shuffle_dir(source).iterdir())
            with ProcessPoolExecutor(min(os.cpu_count(), MEMORY // (2 * largest)), mp_context=SPAWN) as pool:
                list(pool.map(curate_partition, [(source, part) for part in range(SPLIT)]))
            shutil.rmtree(shuffle_dir(source))
        else:
            shard_documents(source, paths[source], drops[source])


if __name__ == "__main__":
    main()
