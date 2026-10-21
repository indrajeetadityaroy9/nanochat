"""
Curate the normalized sources (prepare.py) into the training documents:

  curated/<source>/{train,val}/<k>.parquet    the training documents

Curate deduplicates every source (dedup.py) and holds out 1% of each source's groups as val (SPLIT): a repository (by
name, so it is val in every repository source), a codeforces problem, otherwise a document; deduplication keeps one copy
of a text across all splits. A source of documents keeps its remaining rows in file order. A repository-file source
becomes repository samples (DeepSeek-Coder arXiv 2401.14196 §2.2): DuckDB partitions its remaining files by the residue
of the repository name, then each partition splits each repository into its samples (data crate repository_samples:
files that import one another, dependencies first).

Run by prepare.py, after the sources are normalized.
"""

import itertools
import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from nanochat.data import dedup
from nanochat.data.schema import REPOSITORY_FILE, SAMPLE_FILE
from nanochat.data.sources import SOURCES, curated_dir, normalized_path
from nanochat.native import data_crate
from nanochat.paths import DATA

# Workers start without the parent's threads (DuckDB's, obstore's, go-enry's Go runtime): Go does not support fork
# without exec, and a forked child's calls into Go block.
SPAWN = multiprocessing.get_context("spawn")
SPLIT = 100  # a group is val when its residue is 0: the experiment's 1% of each source's groups
# Half the machine's memory: the lower end of DuckDB's 50-60% of system memory, as some operations bypass the buffer
# manager's limit (duckdb.org/docs/lts/guides/troubleshooting/oom_errors) and other jobs share the unified memory.
MEMORY = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2


def residue(group: str) -> str:
    """DuckDB's residue of a group column, md5_number(group) % SPLIT: 0 holds the group out as val."""
    return f"md5_number({group}) % {SPLIT}"


def shuffle_dir(source: str) -> Path:
    return DATA / "shuffle" / source


def shuffle(source: str, paths: list[Path], masks: list[np.ndarray]) -> None:
    """A repository-file source's normalized files (paths) without the rows deduplication drops (masks: each file's
    kept rows), as Hive partitions shuffle/<source>/part=<p>/ by the residue of the val group, the repository name
    (DuckDB COPY ... PARTITION_BY). file_index is a file's position in paths; file_row_number its row's position."""
    rows = [np.flatnonzero(~mask) for mask in masks]
    drops = pa.table({"file": np.repeat(np.arange(len(rows)), list(map(len, rows))), "row": np.concatenate(rows)})
    # DuckDB in MEMORY, spilling to the data store, without insertion order (the troubleshooting guide cited at MEMORY)
    with duckdb.connect(config={"memory_limit": f"{MEMORY}B", "temp_directory": str(DATA / "shuffle" / "spill"),
                                "preserve_insertion_order": False}) as con:
        con.register("drops", drops)
        con.execute(f"COPY (SELECT n.repo_name, n.relative_path, n.content, n.id, {residue('n.val_group')} AS part "
                    f"FROM read_parquet({[str(path) for path in paths]}, file_row_number = true) AS n "
                    f"ANTI JOIN drops AS d ON d.file = n.file_index AND d.row = n.file_row_number) "
                    f"TO '{shuffle_dir(source)}' (FORMAT parquet, PARTITION_BY (part))")


def partition_row_groups(source: str, part: int) -> list[pq.RowGroupMetaData]:
    """The row-group metadata of the files of a shuffle partition."""
    return [md.row_group(g) for md in map(pq.read_metadata, (shuffle_dir(source) / f"part={part}").iterdir())
            for g in range(md.num_row_groups)]


def curate_partition(task: tuple[str, int]) -> None:
    """One shuffle partition as a curated shard (partition 0 is val). Files are sorted by (repository, path, id) into
    one chunk, so a repository is a zero-copy slice (large_string: 64-bit offsets hold a partition beyond 2 GiB),
    split into samples (repository_samples) and written in sample order (SAMPLE_FILE), in row groups of whole samples,
    each but the last at least as long as the shuffle's longest row group."""
    source, part = task
    length = max(row_group.num_rows for row_group in partition_row_groups(source, part))
    table = pq.read_table(shuffle_dir(source) / f"part={part}",
                          schema=pa.schema([("repo_name", pa.string()), ("relative_path", pa.string()),
                                            ("content", pa.large_string()), ("id", pa.string())]))
    table = table.sort_by([(key, "ascending") for key in ("repo_name", "relative_path", "id")]).combine_chunks()
    paths, contents = table["relative_path"], table["content"]
    groups = [np.add(group, a)
              for a, b in itertools.pairwise([0, *pc.run_end_encode(table["repo_name"].chunk(0)).run_ends.to_numpy()])
              for group in data_crate().repository_samples(paths.slice(a, b - a).to_pylist(),
                                                           contents.slice(a, b - a).to_pylist(),
                                                           grammar=SOURCES[source].grammar)]
    order = np.concatenate(groups)
    samples = np.repeat(np.arange(len(groups)), [len(group) for group in groups])
    cuts = [0]
    for start in np.flatnonzero(np.diff(samples, prepend=-1)).tolist():
        if start - cuts[-1] >= length:
            cuts.append(start)
    columns = table.select(["repo_name", "relative_path", "content"])
    split = "val" if part == 0 else "train"
    with pq.ParquetWriter(curated_dir(source, split) / f"{part}.parquet", SAMPLE_FILE) as writer:
        for a, b in itertools.pairwise([*cuts, len(order)]):
            sample = columns.take(order[a:b]).append_column("sample", pa.array(samples[a:b]))
            writer.write_table(sample.cast(SAMPLE_FILE))


def split_file(task: tuple[str, int, np.ndarray]) -> None:
    """One normalized file of a source of documents as its curated shard of each split: the rows deduplication keeps
    (keep), row group by row group in file order (DuckDB on one thread: the pool's processes are the parallelism)."""
    source, k, keep = task
    spec, path = SOURCES[source], normalized_path(source, k)
    with duckdb.connect(config={"threads": 1}) as con:
        val = con.execute(f"SELECT {residue('val_group')} = 0 AS val "
                          f"FROM read_parquet('{path}', file_row_number = true) "
                          f"ORDER BY file_row_number").fetchnumpy()["val"]
    held, trained = val & keep, ~val & keep
    pf = pq.ParquetFile(path)
    with (pq.ParquetWriter(curated_dir(source, "train") / f"{k}.parquet", spec.schema) as train,
          pq.ParquetWriter(curated_dir(source, "val") / f"{k}.parquet", spec.schema) as held_out):
        start = 0
        for g in range(pf.num_row_groups):
            table = pf.read_row_group(g, columns=spec.schema.names)
            end = start + table.num_rows
            held_out.write_table(table.filter(pa.array(held[start:end])))
            train.write_table(table.filter(pa.array(trained[start:end])))
            start = end


def curate(normalized: dict[str, int]) -> None:
    """The curated shards of every source from its normalized files ({source: number of normalized files}) after
    deduplication over every source (dedup.kept). The crate is built once here, before the workers load it. A
    partition worker holds its partition twice at peak (the table and its sorted copy), so as many partitions run at
    once as fit twice in MEMORY."""
    data_crate()
    paths = {source: [normalized_path(source, k) for k in range(normalized[source])] for source in SOURCES}
    with ProcessPoolExecutor(mp_context=SPAWN) as pool:
        keep = dedup.kept(paths, pool)
    # the repository sources' shuffles (DuckDB creates their partition directories and its spill directory only)
    (DATA / "shuffle").mkdir()
    for source, spec in SOURCES.items():
        for split in ("train", "val"):
            curated_dir(source, split).mkdir(parents=True)
        if spec.schema is REPOSITORY_FILE:
            shuffle(source, paths[source], keep[source])
            largest = max(sum(row_group.total_byte_size for row_group in partition_row_groups(source, part))
                          for part in range(SPLIT))
            with ProcessPoolExecutor(min(os.cpu_count(), MEMORY // (2 * largest)), mp_context=SPAWN) as pool:
                list(pool.map(curate_partition, [(source, part) for part in range(SPLIT)]))
            shutil.rmtree(shuffle_dir(source))
        else:
            with ProcessPoolExecutor(mp_context=SPAWN) as pool:
                list(pool.map(split_file, [(source, k, mask) for k, mask in enumerate(keep[source])]))
