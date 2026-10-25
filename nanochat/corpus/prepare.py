"""
Prepare every source: fetch the Hub files it reads, then standardize each into its category's schema, one
standardized file per fetched file. Fetching reads only the standardizer's columns (ranged reads, several files at
once) and keeps fetched files; standardized and checked files are rebuilt. A source read in part (mixture.IN_PART)
fetches only the files that hold its share of the draw (extent). Stack-Edu's Hub files are metadata: each blob's text
comes from Software Heritage's public S3 bucket by blob id, decoded with its src_encoding (The Stack v2 dataset card);
a blob the bucket lacks or that does not decode has no text. Standardization keeps rows with text, normalizes line
ends and byte-order marks (lines), sets id to the text's SHA-1, and of Codeforces-CoTs keeps finished traces only.

python -m nanochat.corpus.prepare
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
from datetime import timedelta
from pathlib import Path

import obstore
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download, list_repo_files
from obstore.store import S3Store

from nanochat.corpus import mixture
from nanochat.corpus.parallel import SPAWN
from nanochat.corpus.schema import COMPRESSION, CONVERSATION, FUNCTION, REPOSITORY_FILE, SPLIT, sha1
from nanochat.corpus.sources import (
    EDU,
    SOURCES,
    Source,
    checked_path,
    footer_rows,
    raw_dir,
    standardized_path,
    text_bytes,
)

# Hub files fetched at once: an engineering choice, more overlapping more transfers at more concurrent Hub connections
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
    """The Hub files prepare fetches: all, sorted; of a source read in part, the fewest in the order of their paths'
    SHA-1 (a uniform order whose prefixes nest as the extent grows) whose text bytes hold the group's fraction of the
    source: its draw over its sources' tokens at mixture.BYTES_PER_TOKEN, widened by mixture.SPREAD and by the train
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
    """This process's client of Software Heritage's bucket, created once so connections are reused; the file-descriptor
    limit raised to its hard limit (a row group's requests are in flight at once); retries for up to 15 minutes."""
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
    """The local path of one of the source's Hub files: its standardizer's columns (Stack-Edu: metadata and each blob's
    text), written to a .partial file and renamed, so only a complete file exists; an existing file is kept."""
    spec = SOURCES[source]
    path = raw_dir(source) / file
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if spec.repo != EDU:
        read = columns(spec)
        with HfFileSystem().open(f"datasets/{spec.repo}/{file}", "rb") as remote:
            hub = pq.ParquetFile(remote)
            with pq.ParquetWriter(partial, pa.schema([hub.schema_arrow.field(c) for c in read]),
                                  compression=COMPRESSION) as writer:
                for g in range(hub.num_row_groups):
                    writer.write_table(hub.read_row_group(g, columns=read))
    else:
        meta = hf_hub_download(spec.repo, file, repo_type="dataset")
        with (ProcessPoolExecutor(mp_context=SPAWN) as pool,
              pq.ParquetWriter(partial, pq.read_schema(meta).append(pa.field("text", pa.string())),
                               compression=COMPRESSION) as writer):
            for table in pool.map(blob_texts, [(meta, g) for g in range(pq.ParquetFile(meta).num_row_groups)]):
                writer.write_table(table)
    partial.rename(path)
    return path


# ---- standardize: a raw table into its category's schema

def lines(texts: pa.ChunkedArray) -> pa.ChunkedArray:
    """texts with CR LF and a lone CR as LF and every leading U+FEFF dropped, as Python and C++ read a source file
    (Python Language Reference §2.1.2, §2.1.4; C++23 [lex.phases] phases 1-2). A file may begin with several marks
    (re-encoded once per mark); both languages drop one and reject a second."""
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
    standardized = standardizer(spec)
    pf = pq.ParquetFile(path)
    with pq.ParquetWriter(standardized_path(source, k), spec.category.schema, compression=COMPRESSION) as writer:
        for g in range(pf.num_row_groups):
            raw = pf.read_row_group(g)
            writer.write_table(standardized(raw.filter(pc.is_valid(raw[spec.text])), spec))


def prepare(source: str, files: list[str]) -> None:
    """Fetch the source's files and standardize them in parallel into a fresh standardized directory (its checked
    files, derived from them, removed)."""
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
