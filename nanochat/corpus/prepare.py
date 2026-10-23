"""
Prepare sources (sources.py): fetch Hub files a source's globs match at the dataset's latest revision (its main's
commit, resolved once per source, so the listing and every file are one snapshot), then standardize each into its
category's schema (schema.py), one standardized file per fetched Hub file, and record the source in the manifest. A
fetch reads only the columns the source's standardizer reads (columns), by ranged reads of the Hub's Parquet file,
several files at once. A source the mixture reads whole is fetched whole; one it reads in part (mixture.IN_PART), only
the Hub files that hold its share of the draw (extent).

Stack-Edu's Hub files are metadata: its raw file adds each blob's text, fetched from Software Heritage's public S3
bucket by blob ID (obstore) and decoded with its src_encoding (The Stack v2 dataset card, download_contents); a blob
the bucket does not hold or that does not decode has no text.

Standardize keeps the rows with text, gives each a local id (the sha1 of the document's text) and keeps a repository
file's text as its language reads it (standardize_file); of Codeforces-CoTs it keeps the finished traces only.

python -m nanochat.corpus.prepare <source> ...
"""

import argparse
import asyncio
import fnmatch
import functools
import gzip
import hashlib
import itertools
import json
import multiprocessing
import resource
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import obstore
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, dataset_info, hf_hub_download, list_repo_files
from obstore.store import S3Store

from nanochat.corpus import mixture
from nanochat.corpus.schema import CONVERSATION, EXERCISE, ISSUE, NOTEBOOK, REPOSITORY_FILE, SPLIT
from nanochat.corpus.sources import EDU, SOURCES, Source, raw_dir, standardized_path
from nanochat.paths import CORPUS

SPAWN = multiprocessing.get_context("spawn")
# Parquet codec of the files prepare writes: zstd, smaller than pyarrow's default snappy at the same speed (measured
# on an UltraData-Code L3 raw file)
COMPRESSION = "zstd"
# Hub files fetched at once: throughput rises from one connection to a few and falls beyond (measured; the corpus
# design document, normalization_doc.md §1)
FETCHES = 4


@functools.cache
def revision(source: str) -> str:
    """The latest revision of the source's dataset, its main's commit, resolved once per process."""
    return dataset_info(SOURCES[source].repo).sha


@functools.cache
def hub_files(source: str) -> list[str]:
    """The source's Hub files, every file its globs match at its revision, sorted."""
    spec = SOURCES[source]
    return sorted(f for f in list_repo_files(spec.repo, repo_type="dataset", revision=revision(source))
                  if any(fnmatch.fnmatchcase(f, glob) for glob in spec.files.values()))


def text_bytes(source: str, file: str) -> int:
    """The text bytes of one of the source's Hub files, from its Parquet footer: the text column's uncompressed size
    less each value's 4-byte length prefix."""
    spec = SOURCES[source]
    with HfFileSystem().open(f"datasets/{spec.repo}@{revision(source)}/{file}", "rb") as remote:
        md = pq.ParquetFile(remote).metadata
    column = md.schema.to_arrow_schema().get_field_index(spec.text)
    chunks = sum(md.row_group(g).column(column).total_uncompressed_size for g in range(md.num_row_groups))
    return chunks - 4 * md.num_rows


@functools.cache
def supply(source: str) -> dict[str, int]:
    """{Hub file: its text bytes} of every Hub file of the source."""
    files = hub_files(source)
    with ThreadPoolExecutor(FETCHES) as pool:
        return dict(zip(files, pool.map(functools.partial(text_bytes, source), files), strict=True))


def extent(source: str) -> list[str]:
    """The Hub files prepare fetches: all of them in sorted order, or, of a source the mixture reads in part, the
    fewest in the order of their paths' sha1 (a uniformly random order whose prefixes nest as the extent grows) whose
    text bytes hold the group's fraction of the source. The fraction is the group's draw over its sources' tokens at
    production's bytes per token, widened by mixture.SPREAD for a retrained tokenizer's compression and over the train
    split's share of groups (schema.SPLIT)."""
    if source not in mixture.PART:
        return hub_files(source)
    group = mixture.PART[source]
    tokens = sum(sum(supply(s).values()) / mixture.BYTES_PER_TOKEN[s] for s, g in mixture.PART.items() if g == group)
    need = (mixture.draw(group) / tokens * (1 + mixture.SPREAD) * SPLIT / (SPLIT - 1)
            * sum(supply(source).values()))
    order = sorted(hub_files(source), key=lambda f: hashlib.sha1(f.encode()).digest())
    held = itertools.accumulate(supply(source)[f] for f in order)
    return order[:next(k + 1 for k, total in enumerate(held) if total >= need)]


@functools.cache
def swh_store() -> S3Store:
    """This process's client of Software Heritage's bucket, created once so its connections are reused across row
    groups; the file-descriptor limit is raised to its hard limit, as a row group's requests are in flight at once.
    A request retries for up to 15 minutes, its waits doubling from a second to a minute: with obstore's default
    backoff, a reset connection exhausted its retries at once and failed the source."""
    resource.setrlimit(resource.RLIMIT_NOFILE, (resource.getrlimit(resource.RLIMIT_NOFILE)[1],) * 2)
    return S3Store("softwareheritage", region="us-east-1", skip_signature=True,
                   retry_config={"max_retries": 15, "retry_timeout": timedelta(minutes=15),
                                 "backoff": {"init_backoff": timedelta(seconds=1), "max_backoff": timedelta(minutes=1),
                                             "base": 2}})


def blob_texts(task: tuple[Path, int]) -> pa.Table:
    """A Stack-Edu metadata row group with each blob's text (null without one), all requests in flight at once."""
    meta, group = task
    rows = pq.ParquetFile(meta).read_row_group(group)
    store = swh_store()

    async def blob(blob_id: str, encoding: str) -> str | None:
        try:
            response = await obstore.get_async(store, "content/" + blob_id)
            return gzip.decompress(await response.bytes_async()).decode(encoding)
        except (FileNotFoundError, UnicodeDecodeError):  # not in the bucket, or not text in its src_encoding
            return None

    async def texts() -> list[str | None]:
        return await asyncio.gather(*map(blob, rows["blob_id"].to_pylist(), rows["src_encoding"].to_pylist()))

    return rows.append_column("text", pa.array(asyncio.run(texts()), pa.string()))


def fetch(source: str, file: str) -> Path:
    """The local path of one of the source's Hub files at the revision: a Parquet file of the columns its standardizer
    reads, or for Stack-Edu its metadata and each blob's text, written whole, then renamed into place (a writer closed
    by an error still writes a valid footer, so only the rename marks a file complete; an existing file is kept).
    Stack-Edu's blobs are fetched by a pool that lives for that file only, so its workers hold no memory beside
    standardize's."""
    spec = SOURCES[source]
    path = raw_dir(source) / file
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if spec.repo != EDU:
        read = columns(source)
        with HfFileSystem().open(f"datasets/{spec.repo}@{revision(source)}/{file}", "rb") as remote:
            hub = pq.ParquetFile(remote)
            with pq.ParquetWriter(partial, pa.schema([hub.schema_arrow.field(c) for c in read]),
                                  compression=COMPRESSION) as writer:
                for g in range(hub.num_row_groups):
                    writer.write_table(hub.read_row_group(g, columns=read))
    else:
        meta = hf_hub_download(spec.repo, file, repo_type="dataset", revision=revision(source))
        with (ProcessPoolExecutor(mp_context=SPAWN) as pool,
              pq.ParquetWriter(partial, pq.read_schema(meta).append(pa.field("text", pa.string())),
                               compression=COMPRESSION) as writer):
            for table in pool.map(blob_texts, [(meta, g) for g in range(pq.ParquetFile(meta).num_row_groups)]):
                writer.write_table(table)
    partial.rename(path)
    return path


# ---- standardize: one function per category, raw table -> its schema

def sha1(texts: pa.ChunkedArray) -> pa.Array:
    """The hex sha1 of each text's UTF-8 bytes."""
    return pa.array([hashlib.sha1(b).hexdigest() for b in texts.cast(pa.large_binary()).to_pylist()], pa.string())


def lines(texts: pa.ChunkedArray) -> pa.ChunkedArray:
    """texts with CR LF and a lone CR as LF and a leading U+FEFF dropped: a source file as Python and C++ read it
    (Python Language Reference §2.1.2 "Physical lines" and §2.1.4 "Encoding declarations"; C++23 [lex.phases] phases 1
    and 2), and any other text, whose CR and leading byte-order mark come from its platform and encoding."""
    texts = pc.replace_substring_regex(texts, "^\ufeff", "")
    return pc.replace_substring(pc.replace_substring(texts, "\r\n", "\n"), "\r", "\n")


def repository_file(t: pa.Table, spec: Source) -> pa.Table:
    content = lines(t["content"])
    return pa.table({"id": sha1(content), "repo_name": t["repo_name"], "relative_path": t["relative_path"],
                     "content": content, "language": pa.repeat(spec.language, t.num_rows)}, schema=REPOSITORY_FILE)


def stack_edu(t: pa.Table, spec: Source) -> pa.Table:
    """A Stack-Edu file: its metadata path, which begins with "/", taken relative to the repository's root."""
    content = lines(t["text"])
    return pa.table({"id": sha1(content), "repo_name": t["repo_name"],
                     "relative_path": pc.utf8_ltrim(t["path"], characters="/"), "content": content,
                     "language": pa.repeat(spec.language, t.num_rows)}, schema=REPOSITORY_FILE)


def exercise(t: pa.Table, spec: Source) -> pa.Table:
    content = lines(t["content"])
    return pa.table({"id": sha1(content), "content": content, "language": pa.repeat(spec.language, t.num_rows)},
                    schema=EXERCISE)


def text(t: pa.Table, spec: Source) -> pa.Table:
    content = lines(t[spec.text])
    return pa.table({"id": sha1(content), "content": content}, schema=spec.schema)


def conversation(t: pa.Table, spec: Source) -> pa.Table:
    """A finished codeforces_cots trace (finish_reason "stop"; a truncated one mostly ends before its code): the R1
    prompt (ending with a line end) followed by its generation, and the problem's id."""
    t = t.filter(pc.equal(t["finish_reason"], "stop"))
    content = lines(pc.binary_join_element_wise(t["prompt"], t[spec.text], ""))
    return pa.table({"id": sha1(content), "problem_id": t["id"], "content": content}, schema=CONVERSATION)


def issue(t: pa.Table, spec: Source) -> pa.Table:
    content = lines(t["text"])
    return pa.table({"id": sha1(content), "repo_name": t["repo_name"], "content": content}, schema=ISSUE)


def notebook(t: pa.Table, spec: Source) -> pa.Table:
    content = lines(t["text"])
    return pa.table({"id": sha1(content), "content": content, "language": pa.repeat(spec.language, t.num_rows)},
                    schema=NOTEBOOK)


STANDARDIZE = {"stack_edu_python": stack_edu, "stack_edu_cpp": stack_edu,
               "ultradata_stack_v3_l2_python": repository_file, "ultradata_stack_v3_l2_cpp": repository_file,
               "ultradata_code_l2_python": repository_file, "ultradata_code_l2_cpp": repository_file,
               "ultradata_code_l3_python": exercise, "ultradata_code_l3_cpp": exercise,
               "ultradata_math_l1": text, "ultradata_math_l3_qa": text, "ultradata_math_l3_conversation": text,
               "ultradata_math_l3_multi_style": text, "ultradata_math_l3_textbook_exercise": text,
               "codeforces_cots": conversation, "github_issues": issue, "kaggle_notebooks": notebook,
               "climbmix": text, "opc_fineweb_math": text}


def columns(source: str) -> list[str]:
    """The Hub columns a source's standardizer reads (Stack-Edu's metadata is fetched whole)."""
    spec = SOURCES[source]
    return {repository_file: ["repo_name", "relative_path", "content"], exercise: ["content"], text: [spec.text],
            conversation: ["id", "prompt", "generation", "finish_reason"], issue: ["repo_name", "text"],
            notebook: ["text"]}[STANDARDIZE[source]]


def standardize_file(task: tuple[str, int, Path]) -> None:
    """Standardize one raw file into its standardized file: the rows with text, each text's line ends normalized
    (lines)."""
    source, k, path = task
    spec = SOURCES[source]
    pf = pq.ParquetFile(path)
    with pq.ParquetWriter(standardized_path(source, k), spec.schema, compression=COMPRESSION) as writer:
        for g in range(pf.num_row_groups):
            raw = pf.read_row_group(g)
            writer.write_table(STANDARDIZE[source](raw.filter(pc.is_valid(raw[spec.text])), spec))


def record(source: str, entry: dict) -> None:
    """The source's entry in CORPUS/manifest.json, which is rewritten whole."""
    path = CORPUS / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"sources": {}}
    manifest["sources"][source] = entry
    path.write_text(json.dumps(manifest, indent=1))


def prepare(sources: list[str]) -> None:
    """Fetch each source's extent at its dataset's latest revision, standardize its files in parallel, and record it."""
    for source in sources:
        spec, files = SOURCES[source], extent(source)
        standardized_path(source, 0).parent.mkdir(parents=True)
        with ThreadPoolExecutor(1 if spec.repo == EDU else FETCHES) as pool:
            paths = list(pool.map(functools.partial(fetch, source), files))
        with ProcessPoolExecutor(mp_context=SPAWN) as pool:
            list(pool.map(standardize_file, [(source, k, path) for k, path in enumerate(paths)]))
        rows = sum(pq.read_metadata(standardized_path(source, k)).num_rows for k in range(len(files)))
        record(source, {"repo": spec.repo, "revision": revision(source), "hub_files": len(hub_files(source)),
                        "files": files, "rows": rows})
        print(f"{source}: {len(files)} of {len(hub_files(source))} files, {rows:,} rows, "
              f"{spec.repo}@{revision(source)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m nanochat.corpus.prepare",
                                     description="fetch and standardize sources of the corpus")
    parser.add_argument("sources", nargs="+", choices=SOURCES)
    prepare(parser.parse_args().sources)
