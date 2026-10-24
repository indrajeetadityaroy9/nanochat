"""
Prepare every source (sources.py): fetch the Hub files it reads at its revision, standardize each into its category's
schema (schema.py), one standardized file per fetched Hub file, and record the source in CORPUS/manifest.json.

The manifest pins each source: its revision is its dataset's latest when first prepared, and a source is prepared
again only when its extent differs from the recorded files (a changed mixture); its standardized and checked files are
then rebuilt, while fetched files are kept. A fetch reads only the columns the standardizer reads, by ranged reads of
the Hub's Parquet file, several files at once. A source the mixture reads whole is fetched whole; one it reads in part
(mixture.IN_PART), only the Hub files that hold its share of the draw (extent).

Stack-Edu's Hub files are metadata: its raw file adds each blob's text, fetched from Software Heritage's public S3
bucket by blob id (obstore) and decoded with its src_encoding (The Stack v2 dataset card, download_contents); a blob
the bucket does not hold or that does not decode has no text.

Standardization keeps the rows with text, normalizes each text's line ends and byte-order marks (lines), gives each
document its sha1 as id, and of Codeforces-CoTs keeps the finished traces only.

python -m nanochat.corpus.prepare
"""

import asyncio
import fnmatch
import functools
import gzip
import hashlib
import itertools
import json
import resource
import shutil
from collections.abc import Callable
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
from nanochat.corpus.parallel import SPAWN
from nanochat.corpus.schema import CONVERSATION, FUNCTION, REPOSITORY_FILE, SPLIT
from nanochat.corpus.sources import (
    EDU,
    MANIFEST,
    SOURCES,
    Source,
    checked_path,
    raw_dir,
    standardized_path,
    text_bytes,
)

# Parquet codec of the files the corpus writes: zstd, smaller than pyarrow's default snappy at the same speed (measured
# on an UltraData-Code L3 raw file)
COMPRESSION = "zstd"
# Hub files fetched at once: throughput rises from one connection to a few and falls beyond (measured,
# normalization_doc.md §1)
FETCHES = 4


def manifest() -> dict[str, dict]:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"sources": {}}


@functools.cache
def revision(source: str) -> str:
    """The source's pinned revision: the manifest's, else its dataset's latest (its main's commit)."""
    recorded = manifest()["sources"].get(source)
    return recorded["revision"] if recorded else dataset_info(SOURCES[source].repo).sha


@functools.cache
def hub_files(source: str) -> list[str]:
    """The source's Hub files, every file its globs match at its revision, sorted."""
    spec = SOURCES[source]
    return sorted(f for f in list_repo_files(spec.repo, repo_type="dataset", revision=revision(source))
                  if any(fnmatch.fnmatchcase(f, glob) for glob in spec.files.values()))


def hub_text_bytes(source: str, file: str) -> int:
    """The text bytes of one of the source's Hub files, from its Parquet footer."""
    spec = SOURCES[source]
    with HfFileSystem().open(f"datasets/{spec.repo}@{revision(source)}/{file}", "rb") as remote:
        return text_bytes(pq.ParquetFile(remote).metadata, spec.text)


@functools.cache
def file_bytes(source: str) -> dict[str, int]:
    """{Hub file: its text bytes} of every Hub file of the source."""
    files = hub_files(source)
    with ThreadPoolExecutor(FETCHES) as pool:
        return dict(zip(files, pool.map(functools.partial(hub_text_bytes, source), files), strict=True))


def extent(source: str) -> list[str]:
    """The Hub files prepare fetches: all of them in sorted order, or, of a source the mixture reads in part, the
    fewest in the order of their paths' sha1 (a uniformly random order whose prefixes nest as the extent grows) whose
    text bytes hold the group's fraction of the source. The fraction is the group's draw over its sources' tokens at
    production's bytes per token, widened by mixture.SPREAD for a retrained tokenizer's compression and over the train
    split's share of groups (schema.SPLIT)."""
    if source not in mixture.PART:
        return hub_files(source)
    group = mixture.PART[source]
    tokens = sum(sum(file_bytes(s).values()) / mixture.BYTES_PER_TOKEN[s]
                 for s, g in mixture.PART.items() if g == group)
    need = (mixture.draw(group) / tokens * (1 + mixture.SPREAD) * SPLIT / (SPLIT - 1)
            * sum(file_bytes(source).values()))
    order = sorted(hub_files(source), key=lambda f: hashlib.sha1(f.encode()).digest())
    held = itertools.accumulate(file_bytes(source)[f] for f in order)
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

    async def fetched() -> list[str | None]:
        return await asyncio.gather(*map(blob, rows["blob_id"].to_pylist(), rows["src_encoding"].to_pylist()))

    return rows.append_column("text", pa.array(asyncio.run(fetched()), pa.string()))


def fetch(source: str, file: str) -> Path:
    """The local path of one of the source's Hub files at its revision: a Parquet file of the columns its standardizer
    reads, or for Stack-Edu its metadata and each blob's text, written whole, then renamed into place (a writer closed
    by an error still writes a valid footer, so only the rename marks a file complete; an existing file is kept).
    Stack-Edu's blobs are fetched by a pool that lives for that file only."""
    spec = SOURCES[source]
    path = raw_dir(source) / file
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if spec.repo != EDU:
        read = columns(spec)
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


# ---- standardize: a raw table into its category's schema

def sha1(texts: pa.ChunkedArray) -> pa.Array:
    """The hex sha1 of each text's UTF-8 bytes."""
    return pa.array([hashlib.sha1(b).hexdigest() for b in texts.cast(pa.large_binary()).to_pylist()], pa.string())


def lines(texts: pa.ChunkedArray) -> pa.ChunkedArray:
    """texts with CR LF and a lone CR as LF and every leading U+FEFF dropped: a source file as Python and C++ read it
    (Python Language Reference §2.1.2 "Physical lines" and §2.1.4 "Encoding declarations"; C++23 [lex.phases] phases 1
    and 2), and any other text, whose CR and leading byte-order marks come from its platform and encoding. A file can
    begin with more than one mark, encoded again with one without its first dropped: both languages drop one mark and
    reject a second."""
    texts = pc.replace_substring_regex(texts, "^\ufeff+", "")
    return pc.replace_substring(pc.replace_substring(texts, "\r\n", "\n"), "\r", "\n")


def document(spec: Source, content: pa.ChunkedArray, **fields: pa.ChunkedArray) -> pa.Table:
    """A table of the source's category: content with its line ends normalized (lines), its sha1 as id, the given
    fields, and the source's language where the category records one."""
    content = lines(content)
    fields |= {"id": sha1(content), "content": content}
    if "language" in spec.category.schema.names:
        fields["language"] = pa.repeat(spec.language, len(content))
    return pa.table({name: fields[name] for name in spec.category.schema.names}, schema=spec.category.schema)


def repository_file(t: pa.Table, spec: Source) -> pa.Table:
    return document(spec, t["content"], repo_name=t["repo_name"], relative_path=t["relative_path"])


def stack_edu(t: pa.Table, spec: Source) -> pa.Table:
    """A Stack-Edu file: its metadata path, which begins with "/", taken relative to the repository's root."""
    return document(spec, t["text"], repo_name=t["repo_name"],
                    relative_path=pc.utf8_ltrim(t["path"], characters="/"))


def conversation(t: pa.Table, spec: Source) -> pa.Table:
    """A finished Codeforces-CoTs trace (finish_reason "stop"; a truncated one mostly ends before its code): the R1
    prompt (ending with a line end) followed by its generation, and the problem's id."""
    t = t.filter(pc.equal(t["finish_reason"], "stop"))
    return document(spec, pc.binary_join_element_wise(t["prompt"], t[spec.text], ""), problem_id=t["id"])


def function(t: pa.Table, spec: Source) -> pa.Table:
    """A CodeSearchNet function: its code, which holds its docstring, and its repository."""
    return document(spec, t["code"], repo_name=t["repo"])


def text_only(t: pa.Table, spec: Source) -> pa.Table:
    """An exercise, a synthetic file, a notebook or a web document: its text alone."""
    return document(spec, t[spec.text])


def standardizer(spec: Source) -> Callable[[pa.Table, Source], pa.Table]:
    if spec.repo == EDU:
        return stack_edu
    return {REPOSITORY_FILE: repository_file, CONVERSATION: conversation, FUNCTION: function}.get(
        spec.category, text_only)


def columns(spec: Source) -> list[str]:
    """The Hub columns a source's standardizer reads (Stack-Edu's metadata is fetched whole)."""
    return {repository_file: ["repo_name", "relative_path", "content"],
            conversation: ["id", "prompt", "generation", "finish_reason"], function: ["repo", "code"],
            text_only: [spec.text]}[standardizer(spec)]


def standardize(task: tuple[str, int, Path]) -> None:
    """One raw file into its standardized file: the rows with text, in the source's category's schema."""
    source, k, path = task
    spec = SOURCES[source]
    pf = pq.ParquetFile(path)
    with pq.ParquetWriter(standardized_path(source, k), spec.category.schema, compression=COMPRESSION) as writer:
        for g in range(pf.num_row_groups):
            raw = pf.read_row_group(g)
            writer.write_table(standardizer(spec)(raw.filter(pc.is_valid(raw[spec.text])), spec))


def prepare(source: str, files: list[str]) -> None:
    """Fetch the source's files, standardize them in parallel into a fresh standardized directory (its checked files,
    derived from them, removed), and record the source."""
    spec = SOURCES[source]
    shutil.rmtree(standardized_path(source, 0).parent, ignore_errors=True)
    shutil.rmtree(checked_path(source, 0).parent, ignore_errors=True)
    standardized_path(source, 0).parent.mkdir(parents=True)
    with ThreadPoolExecutor(1 if spec.repo == EDU else FETCHES) as pool:
        paths = list(pool.map(functools.partial(fetch, source), files))
    with ProcessPoolExecutor(mp_context=SPAWN) as pool:
        list(pool.map(standardize, [(source, k, path) for k, path in enumerate(paths)]))
    rows = sum(pq.read_metadata(standardized_path(source, k)).num_rows for k in range(len(files)))
    record = manifest()
    record["sources"][source] = {"repo": spec.repo, "revision": revision(source), "hub_files": len(hub_files(source)),
                                 "files": files, "rows": rows}
    MANIFEST.write_text(json.dumps(record, indent=1))
    print(f"{source}: {len(files)} of {len(hub_files(source))} files, {rows:,} rows, "
          f"{spec.repo}@{revision(source)}", flush=True)


def main() -> None:
    for source in SOURCES:
        files = extent(source)
        if manifest()["sources"].get(source, {}).get("files") != files:
            prepare(source, files)


if __name__ == "__main__":
    main()
