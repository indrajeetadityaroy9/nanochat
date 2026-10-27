"""
Prepare every source: fetch the Hub files of its extent (only the standardizer's columns; Stack-Edu's blobs from
Software Heritage's public S3 bucket, The Stack v2 dataset card), then standardize each into its category's schema.

python -m corpus.prepare
"""

import asyncio
import fnmatch
import functools
import gzip
import hashlib
import itertools
import resource
import shutil
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import obstore
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download, list_repo_files
from obstore.store import S3Store

from corpus import mixture
from corpus.parallel import SPAWN
from corpus.schema import COMPRESSION, CONVERSATION, FUNCTION, REPOSITORY_FILE, SPLIT, sha1
from corpus.sources import (
    EDU,
    SOURCES,
    Source,
    checked_path,
    footer_rows,
    raw_dir,
    standardized_path,
    text_bytes,
)

# The highest throughput of 1, 2, 4 and 8 files in flight
FETCHES = 4


@functools.cache
def hub_files(source: str) -> list[str]:
    """The source's Hub files, every file its globs match, sorted."""
    spec = SOURCES[source]
    return sorted(f for f in list_repo_files(spec.repo, repo_type="dataset")
                  if any(fnmatch.fnmatchcase(f, glob) for glob in spec.files.values()))


@functools.cache
def file_bytes(source: str) -> dict[str, int]:
    """{Hub file: its text bytes, from its Parquet footer} of every Hub file of the source."""
    spec, files = SOURCES[source], hub_files(source)

    def footer_bytes(file: str) -> int:
        with HfFileSystem().open(f"datasets/{spec.repo}/{file}", "rb") as remote:
            return text_bytes(pq.ParquetFile(remote).metadata, spec.text)

    with ThreadPoolExecutor(FETCHES) as pool:
        return dict(zip(files, pool.map(footer_bytes, files), strict=True))


def extent(source: str) -> list[str]:
    """The Hub files prepare fetches: all, sorted; of a source read in part, the fewest in the SHA-1 order of their
    paths whose text bytes hold its draw widened by mixture.SPREAD and SPLIT / (SPLIT - 1)."""
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
    """This process's client of Software Heritage's bucket, with the file-descriptor limit raised to its hard limit."""
    resource.setrlimit(resource.RLIMIT_NOFILE, (resource.getrlimit(resource.RLIMIT_NOFILE)[1],) * 2)
    return S3Store("softwareheritage", region="us-east-1", skip_signature=True)


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
    """The local path of one of the source's Hub files."""
    spec = SOURCES[source]
    path = raw_dir(source) / file
    path.parent.mkdir(parents=True, exist_ok=True)
    if spec.repo != EDU:
        read = columns(spec)
        with HfFileSystem().open(f"datasets/{spec.repo}/{file}", "rb") as remote:
            hub = pq.ParquetFile(remote)
            with pq.ParquetWriter(path, pa.schema([hub.schema_arrow.field(c) for c in read]),
                                  compression=COMPRESSION) as writer:
                for g in range(hub.num_row_groups):
                    writer.write_table(hub.read_row_group(g, columns=read))
    else:
        meta = hf_hub_download(spec.repo, file, repo_type="dataset")
        with (ProcessPoolExecutor(mp_context=SPAWN) as pool,
              pq.ParquetWriter(path, pq.read_schema(meta).append(pa.field("text", pa.string())),
                               compression=COMPRESSION) as writer):
            for table in pool.map(blob_texts, [(meta, g) for g in range(pq.ParquetFile(meta).num_row_groups)]):
                writer.write_table(table)
    return path


def lines(texts: pa.ChunkedArray) -> pa.ChunkedArray:
    """texts with CR LF and a lone CR as LF and every leading U+FEFF dropped (Python Language Reference §2.1.2, §2.1.4;
    C++23 [lex.phases] phases 1-2)."""
    texts = pc.replace_substring_regex(texts, "^\ufeff+", "")
    return pc.replace_substring(pc.replace_substring(texts, "\r\n", "\n"), "\r", "\n")


def document(spec: Source, content: pa.ChunkedArray, **fields: pa.ChunkedArray) -> pa.Table:
    """A table of the source's category: content normalized (lines), its sha1 as id, the given fields and language."""
    content = lines(content)
    fields |= {"id": sha1(content), "content": content}
    if "language" in spec.category.schema.names:
        fields["language"] = pa.repeat(spec.language, len(content))
    return pa.table({name: fields[name] for name in spec.category.schema.names}, schema=spec.category.schema)


def repository_file(t: pa.Table, spec: Source) -> pa.Table:
    return document(spec, t["content"], repo_name=t["repo_name"], relative_path=t["relative_path"])


def stack_edu(t: pa.Table, spec: Source) -> pa.Table:
    """A Stack-Edu file, its path without leading "/"."""
    return document(spec, t["text"], repo_name=t["repo_name"],
                    relative_path=pc.utf8_ltrim(t["path"], characters="/"))


def conversation(t: pa.Table, spec: Source) -> pa.Table:
    """A finished Codeforces-CoTs trace (finish_reason "stop"): prompt followed by generation, and the problem's id."""
    t = t.filter(pc.equal(t["finish_reason"], "stop"))
    return document(spec, pc.binary_join_element_wise(t["prompt"], t[spec.text], ""), problem_id=t["id"])


def function(t: pa.Table, spec: Source) -> pa.Table:
    """A CodeSearchNet function (code with its docstring) and its repository."""
    return document(spec, t["code"], repo_name=t["repo"])


def text_only(t: pa.Table, spec: Source) -> pa.Table:
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
    standardized = standardizer(spec)
    pf = pq.ParquetFile(path)
    with pq.ParquetWriter(standardized_path(source, k), spec.category.schema, compression=COMPRESSION) as writer:
        for g in range(pf.num_row_groups):
            raw = pf.read_row_group(g)
            writer.write_table(standardized(raw.filter(pc.is_valid(raw[spec.text])), spec))


def prepare(source: str, files: list[str]) -> None:
    """Fetch and standardize the source's files into a fresh standardized directory (its checked files removed)."""
    spec = SOURCES[source]
    directory = standardized_path(source, 0).parent
    shutil.rmtree(directory, ignore_errors=True)
    shutil.rmtree(checked_path(source, 0).parent, ignore_errors=True)
    directory.mkdir(parents=True)
    with ThreadPoolExecutor(1 if spec.repo == EDU else FETCHES) as pool:
        paths = list(pool.map(functools.partial(fetch, source), files))
    with ProcessPoolExecutor(mp_context=SPAWN) as pool:
        list(pool.map(standardize, [(source, k, path) for k, path in enumerate(paths)]))
    rows = footer_rows(standardized_path(source, k) for k in range(len(files)))
    print(f"{source}: {len(files)} of {len(hub_files(source))} files of {spec.repo}, {rows:,} rows", flush=True)


def main() -> None:
    for source in SOURCES:
        prepare(source, extent(source))


if __name__ == "__main__":
    main()
