"""
Prepare every pretraining source (sources.py): fetch, normalize into the Parquet store, curate.

  raw/<source>/                               the Hub files at their Hub paths (Stack-Edu: with Software Heritage text)
  normalized/<source>/<k>.parquet             one file per raw file, in its category's schema (schema.py) + val group
  curated/<source>/{train,val}/<k>.parquet    the training documents

Fetch, the only network step, downloads a source's Hub files in sorted order at its pinned revision. Stack-Edu's Hub
files are metadata: its raw file adds each blob's text, fetched from Software Heritage's public S3 bucket by blob ID
(obstore) and decoded with its src_encoding (The Stack v2 dataset card, download_contents); a blob the bucket does not
hold or that does not decode has no text.

The extent of each source follows from EXTENT, the groups of config.MIXTURE at Qwen2.5-Coder's code:text:math = 7:2:1
(arXiv 2409.12186 §3.1.2): the first group, code, is taken whole, and its tokens over its share are the horizon; every
later group takes its sources in order, and each source's Hub files in order, until it holds its share of the horizon
(ClimbMix, the text group, is a shuffled crawl). A mixture whose code share is at least 0.7 (config.MIXTURE: 0.87)
reads at most that much text and math per code token.
A file's tokens are its content column's uncompressed Parquet bytes over the o200k_base (tiktoken) bytes per
token of its source's first normalized row group, before deduplication (the pipeline's tokenizer is trained on the
curated store). A file already normalized is kept (stage writes each file whole), so a source is fetched and normalized
once.

Normalize maps each raw file to its category's schema in parallel, with a local id (the sha1 of the text). A row needs
text; a repository file's text is kept as its language reads it (source_text), and a Python repository file must then
parse as Python 3 (ast.parse drops Python 2 and broken files; it runs after source_text, as ast.parse on a str rejects
the leading U+FEFF that CPython's source decoding strips). A codeforces_cots document is its conversation: the R1
prompt followed by its generation.

Curate deduplicates every source (dedup.py) and holds out 1% of each source's groups as val (SPLIT): a repository (by
name, so it is val in every repository source), a codeforces problem, otherwise a document; deduplication keeps one copy
of a text across all splits. A source of documents keeps its remaining rows in file order. A repository-file source
becomes repository samples (DeepSeek-Coder arXiv 2401.14196 §2.2): DuckDB partitions its remaining files by the residue
of the repository name, then each partition splits each repository into its samples (bpe repository_samples: files that
import one another, dependencies first).

python -m nanochat.data.prepare
"""

import ast
import asyncio
import fnmatch
import functools
import gzip
import hashlib
import itertools
import multiprocessing
import os
import resource
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor

import duckdb
import numpy as np
import obstore
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import hf_hub_download, list_repo_files
from obstore.store import S3Store

from nanochat.config import MIXTURE
from nanochat.data import dedup
from nanochat.data.schema import REPOSITORY_FILE, SAMPLE_FILE
from nanochat.data.sources import EDU, SOURCES, column_bytes, curated_dir
from nanochat.data.tokenizer import bpe
from nanochat.paths import data_dir

# Workers start without the parent's threads (DuckDB's, obstore's, go-enry's Go runtime): Go does not support fork
# without exec, and a forked child's calls into Go block.
SPAWN = multiprocessing.get_context("spawn")


def normalized_root():
    return os.path.join(data_dir(), "normalized")


# ---- fetch

@functools.cache
def swh_store():
    """This process's client of Software Heritage's bucket, created once so its connections are reused across row
    groups; the file-descriptor limit is raised to its hard limit, as a row group's requests are in flight at once."""
    resource.setrlimit(resource.RLIMIT_NOFILE, (resource.getrlimit(resource.RLIMIT_NOFILE)[1],) * 2)
    return S3Store("softwareheritage", region="us-east-1", skip_signature=True)


def blob_texts(task):
    """A Stack-Edu metadata row group with each blob's text (null without one), all requests in flight at once."""
    meta, group = task
    rows = pq.ParquetFile(meta).read_row_group(group)
    store = swh_store()

    async def blob(blob_id, encoding):
        try:
            return gzip.decompress(await (await obstore.get_async(store, "content/" + blob_id)).bytes_async()).decode(encoding)
        except (FileNotFoundError, UnicodeDecodeError):  # not in the bucket, or not text in its src_encoding
            return None

    async def texts():
        return await asyncio.gather(*map(blob, rows["blob_id"].to_pylist(), rows["src_encoding"].to_pylist()))

    return rows.append_column("text", pa.array(asyncio.run(texts()), pa.string()))


def fetch(source, file):
    """The local path of one of the source's Hub files. Stack-Edu's blobs are fetched by a pool that lives for that file
    only, so its workers hold no memory beside normalize's."""
    spec = SOURCES[source]
    directory = os.path.join(data_dir(), "raw", source)
    if spec.repo != EDU[0]:
        return hf_hub_download(spec.repo, file, repo_type="dataset", revision=spec.revision, local_dir=directory)
    path = os.path.join(directory, file)
    meta = hf_hub_download(spec.repo, file, repo_type="dataset", revision=spec.revision)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with (ProcessPoolExecutor(mp_context=SPAWN) as pool,
          pq.ParquetWriter(path, pq.read_schema(meta).append(pa.field("text", pa.string()))) as writer):
        for table in pool.map(blob_texts, [(meta, g) for g in range(pq.ParquetFile(meta).num_row_groups)]):
            writer.write_table(table)
    return path


@functools.cache
def bytes_per_token(source):
    """The UTF-8 bytes per o200k_base (tiktoken) token of the source's first normalized row group."""
    texts = pq.ParquetFile(normalized_path(source, 0)).read_row_group(0, columns=["content"])["content"]
    count = sum(map(len, tiktoken.get_encoding("o200k_base").encode_ordinary_batch(texts.to_pylist())))
    return pc.sum(pc.binary_length(texts)).as_py() / count


def tokens(source, k):
    """The o200k tokens of a source's normalized file k: its content bytes at the source's bytes per token."""
    return column_bytes(normalized_path(source, k), "content") / bytes_per_token(source)


# ---- normalize: one function per source family, raw table -> category schema plus the row's val group

def sha1(texts):
    """The hex sha1 of each text's UTF-8 bytes."""
    return pa.array([hashlib.sha1(b).hexdigest() for b in texts.cast(pa.large_binary()).to_pylist()], pa.string())


def source_text(texts):
    """Source files' texts as Python and C++ read them: CR LF and a lone CR read as LF, a leading U+FEFF dropped.
    Python Language Reference §2.1.2 "Physical lines" and §2.1.4 "Encoding declarations"; C++23 [lex.phases] phase 1
    (CR LF and a CR not followed by LF become one new-line) and phase 2 (a first character U+FEFF is deleted)."""
    return pc.replace_substring(pc.replace_substring(pc.replace_substring_regex(texts, "^\ufeff", ""), "\r\n", "\n"), "\r", "\n")


def normalized_schema(spec):
    """A source's normalized schema: its category's schema and the row's val group."""
    return spec.schema.append(pa.field("val_group", pa.string()))


def with_group(columns, spec, group):
    return pa.table(columns | {"val_group": group}, schema=normalized_schema(spec))


def stack_edu(t, spec):
    n = t.num_rows
    return with_group({"id": sha1(t["text"]), "repo_name": t["repo_name"], "relative_path": t["path"], "content": t["text"],
                       "language": pa.repeat(spec.language, n), "int_score": t["int_score"],
                       "quality_score": pa.nulls(n), "category": pa.nulls(n)}, spec, t["repo_name"])


def ultradata(t, spec):
    n = t.num_rows
    return with_group({"id": sha1(t["content"]), "repo_name": t["repo_name"], "relative_path": t["relative_path"], "content": t["content"],
                       "language": pa.repeat(spec.language, n), "int_score": pa.nulls(n),
                       "quality_score": t["quality_score"], "category": t["category"]}, spec, t["repo_name"])


def notebook(t, spec):
    ids = sha1(t["text"])
    return with_group({"id": ids, "content": t["text"], "language": pa.repeat(spec.language, t.num_rows)}, spec, ids)


def text(t, spec):
    ids = sha1(t[spec.text])
    url = t["url"] if "url" in t.column_names else pa.nulls(t.num_rows)
    return with_group({"id": ids, "url": url, "content": t[spec.text]}, spec, ids)


def conversation(t, spec):
    """A codeforces_cots row as one document: the R1 prompt (ending with a line end) followed by the generation; its
    group is the problem's id, shared across subsets, so every generation of a problem is in one split."""
    content = pc.binary_join_element_wise(t["prompt"], t[spec.text], "")
    return with_group({"id": sha1(content), "url": pa.nulls(t.num_rows), "content": content}, spec, t["id"])


NORMALIZE = {
    "stack_edu_python": stack_edu, "stack_edu_cpp": stack_edu,
    "ultradata_stack_v3_l2_python": ultradata, "ultradata_stack_v3_l2_cpp": ultradata,
    "ultradata_code_l2_python": ultradata, "ultradata_code_l2_cpp": ultradata,
    "kaggle_notebooks": notebook, "codeforces_cots": conversation, "climbmix": text, "opc_fineweb_math": text,
    "ultradata_math_l2": text,
}


def parses_as_python3(text):
    """Whether text parses as Python 3. ast.parse raises SyntaxError or ValueError for invalid source (Python Library
    Reference, compile(); UnicodeDecodeError is a ValueError), RecursionError or MemoryError for too deep nesting."""
    try:
        ast.parse(text)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False


def normalized_path(source, k):
    return os.path.join(normalized_root(), source, f"{k}.parquet")


def stage(task):
    """Normalize one raw file into its normalized file, renamed into place once complete (an existing normalized file
    is whole): rows with text, a repository file's text as its language reads it (source_text), and a Python
    repository file only if it parses as Python 3."""
    source, k, path = task
    spec = SOURCES[source]
    warnings.filterwarnings("ignore", category=SyntaxWarning)  # invalid escapes in otherwise valid code
    partial = normalized_path(source, k) + ".partial"
    pf = pq.ParquetFile(path)
    with pq.ParquetWriter(partial, normalized_schema(spec)) as writer:
        for g in range(pf.num_row_groups):
            raw = pf.read_row_group(g)
            texts = raw.filter(pc.is_valid(raw[spec.text]))
            if spec.grammar is not None:
                texts = texts.set_column(texts.schema.get_field_index(spec.text), spec.text, source_text(texts[spec.text]))
            table = NORMALIZE[source](texts, spec)
            if spec.grammar == "Python":
                table = table.filter(pa.array([parses_as_python3(t) for t in table["content"].to_pylist()], pa.bool_()))
            writer.write_table(table)
    os.replace(partial, normalized_path(source, k))


# ---- curate: deduplication (dedup.py), the val split, repository files as repository samples

SPLIT = 100  # a group is val when its residue is 0: the experiment's 1% of each source's groups


def residue(group):
    """DuckDB's residue of a group column, md5_number(group) % SPLIT: 0 holds the group out as val."""
    return f"md5_number({group}) % {SPLIT}"


# Half the machine's memory: the lower end of DuckDB's 50-60% of system memory, as some operations bypass the buffer
# manager's limit (duckdb.org/docs/lts/guides/troubleshooting/oom_errors) and other jobs share the unified memory.
MEMORY = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2
# The groups' shares of the store (module docstring)
EXTENT = {"code": 0.7, "text": 0.2, "math": 0.1}


def shuffle_dir(source):
    return os.path.join(data_dir(), "shuffle", source)


def shuffle(source, paths, masks):
    """A repository-file source's normalized files (paths) without the rows deduplication drops (masks: each file's
    kept rows), as Hive partitions shuffle/<source>/part=<p>/ by the residue of the val group, the repository name
    (DuckDB COPY ... PARTITION_BY). file_index is a file's position in paths; file_row_number its row's position."""
    rows = [np.flatnonzero(~mask) for mask in masks]
    drops = pa.table({"file": np.repeat(np.arange(len(rows)), list(map(len, rows))), "row": np.concatenate(rows)})
    os.makedirs(os.path.join(data_dir(), "shuffle"), exist_ok=True)  # DuckDB creates the spill and partition directories only
    # DuckDB in MEMORY, spilling to the data store, without insertion order (the troubleshooting guide cited at MEMORY)
    with duckdb.connect(config={"memory_limit": f"{MEMORY}B", "temp_directory": os.path.join(data_dir(), "shuffle", "spill"),
                                "preserve_insertion_order": False}) as con:
        con.register("drops", drops)
        con.execute(f"COPY (SELECT n.repo_name, n.relative_path, n.content, n.id, {residue('n.val_group')} AS part "
                    f"FROM read_parquet({paths}, file_row_number = true) AS n "
                    f"ANTI JOIN drops AS d ON d.file = n.file_index AND d.row = n.file_row_number) "
                    f"TO '{shuffle_dir(source)}' (FORMAT parquet, PARTITION_BY (part))")


def partition_row_groups(source, part):
    """The row-group metadata of the files of a shuffle partition."""
    directory = os.path.join(shuffle_dir(source), f"part={part}")
    return [md.row_group(g) for md in (pq.read_metadata(os.path.join(directory, f)) for f in os.listdir(directory))
            for g in range(md.num_row_groups)]


def curate_partition(task):
    """One shuffle partition as a curated shard (partition 0 is val). Files are sorted by (repository, path, id) into
    one chunk, so a repository is a zero-copy slice (large_string: 64-bit offsets hold a partition beyond 2 GiB), split
    into samples (repository_samples) and written in sample order (SAMPLE_FILE), in row groups of whole samples, each but
    the last at least as long as the shuffle's longest row group."""
    source, part = task
    length = max(row_group.num_rows for row_group in partition_row_groups(source, part))
    table = pq.read_table(os.path.join(shuffle_dir(source), f"part={part}"),
                          schema=pa.schema([("repo_name", pa.string()), ("relative_path", pa.string()),
                                            ("content", pa.large_string()), ("id", pa.string())]))
    table = table.sort_by([(key, "ascending") for key in ("repo_name", "relative_path", "id")]).combine_chunks()
    paths, contents = table["relative_path"], table["content"]
    groups = [np.add(group, a)
              for a, b in itertools.pairwise([0, *pc.run_end_encode(table["repo_name"].chunk(0)).run_ends.to_numpy()])
              for group in bpe().repository_samples(paths.slice(a, b - a).to_pylist(), contents.slice(a, b - a).to_pylist(),
                                                    grammar=SOURCES[source].grammar)]
    order = np.concatenate(groups)
    samples = np.repeat(np.arange(len(groups)), [len(group) for group in groups])
    cuts = [0]
    for start in np.flatnonzero(np.diff(samples, prepend=-1)).tolist():
        if start - cuts[-1] >= length:
            cuts.append(start)
    columns = table.select(["repo_name", "relative_path", "content"])
    with pq.ParquetWriter(os.path.join(curated_dir(source, "val" if part == 0 else "train"), f"{part}.parquet"), SAMPLE_FILE) as writer:
        for a, b in itertools.pairwise(cuts + [len(order)]):
            writer.write_table(columns.take(order[a:b]).append_column("sample", pa.array(samples[a:b])).cast(SAMPLE_FILE))


def split_file(task):
    """One normalized file of a source of documents as its curated shard of each split: the rows deduplication keeps
    (keep), row group by row group in file order (DuckDB on one thread: the pool's processes are the parallelism)."""
    source, k, keep = task
    spec, path = SOURCES[source], normalized_path(source, k)
    with duckdb.connect(config={"threads": 1}) as con:
        val = con.execute(f"SELECT {residue('val_group')} = 0 AS val FROM read_parquet('{path}', file_row_number = true) "
                          f"ORDER BY file_row_number").fetchnumpy()["val"]
    held, trained = val & keep, ~val & keep
    pf = pq.ParquetFile(path)
    with (pq.ParquetWriter(os.path.join(curated_dir(source, "train"), f"{k}.parquet"), spec.schema) as train,
          pq.ParquetWriter(os.path.join(curated_dir(source, "val"), f"{k}.parquet"), spec.schema) as held_out):
        start = 0
        for g in range(pf.num_row_groups):
            table = pf.read_row_group(g, columns=spec.schema.names)
            end = start + table.num_rows
            held_out.write_table(table.filter(pa.array(held[start:end])))
            train.write_table(table.filter(pa.array(trained[start:end])))
            start = end


def curate(normalized):
    """The curated shards of every source from its normalized files ({source: number of normalized files}) after
    deduplication over every source (dedup.kept). The crate is built once here, before the workers load it. A
    partition worker holds its partition twice at peak (the table and its sorted copy), so as many partitions run at
    once as fit twice in MEMORY."""
    bpe()
    paths = {source: [normalized_path(source, k) for k in range(normalized[source])] for source in SOURCES}
    with ProcessPoolExecutor(mp_context=SPAWN) as pool:
        keep = dedup.kept(paths, pool)
    for source, spec in SOURCES.items():
        for split in ("train", "val"):
            os.makedirs(curated_dir(source, split))
        if spec.schema is REPOSITORY_FILE:
            shuffle(source, paths[source], keep[source])
            largest = max(sum(row_group.total_byte_size for row_group in partition_row_groups(source, part)) for part in range(SPLIT))
            with ProcessPoolExecutor(min(os.cpu_count(), MEMORY // (2 * largest)), mp_context=SPAWN) as pool:
                list(pool.map(curate_partition, [(source, part) for part in range(SPLIT)]))
            shutil.rmtree(shuffle_dir(source))
        else:
            with ProcessPoolExecutor(mp_context=SPAWN) as pool:
                list(pool.map(split_file, [(source, k, mask) for k, mask in enumerate(keep[source])]))


def prepare():
    """Fetch, normalize and curate every source: config.MIXTURE's first group whole (its files normalized in parallel),
    every later group in Hub order until it holds its EXTENT share of the horizon (module docstring)."""
    normalized, horizon = {}, None
    for group, (_, sources) in MIXTURE.items():
        share = EXTENT[group]
        held = 0
        for source in sources:
            spec = SOURCES[source]
            files = sorted(f for f in list_repo_files(spec.repo, repo_type="dataset", revision=spec.revision)
                           if any(fnmatch.fnmatchcase(f, glob) for glob in spec.files.values()))
            os.makedirs(os.path.join(normalized_root(), source), exist_ok=True)
            if horizon is None:
                with ProcessPoolExecutor(mp_context=SPAWN) as pool:
                    list(pool.map(stage, [(source, k, fetch(source, file)) for k, file in enumerate(files)
                                          if not os.path.exists(normalized_path(source, k))]))
                count = len(files)
                held += sum(tokens(source, k) for k in range(count))
            else:
                count = 0
                while count < len(files) and held < share * horizon:
                    if not os.path.exists(normalized_path(source, count)):
                        stage((source, count, fetch(source, files[count])))
                    held += tokens(source, count)
                    count += 1
            normalized[source] = count
            print(f"{source}: {count} of {len(files)} files")
        horizon = horizon or held / share
    curate(normalized)


if __name__ == "__main__":
    prepare()
