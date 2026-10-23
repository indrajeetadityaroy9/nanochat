"""
Prepare sources (sources.py): fetch each Hub file a source's globs match, in sorted order at the dataset's latest
revision (its main's commit, resolved once per source, so the listing and every file are one snapshot), then
standardize each into its category's schema (schema.py), one standardized file per Hub file.

Stack-Edu's Hub files are metadata: its raw file adds each blob's text, fetched from Software Heritage's public S3
bucket by blob ID (obstore) and decoded with its src_encoding (The Stack v2 dataset card, download_contents); a blob
the bucket does not hold or that does not decode has no text.

Standardize keeps the rows with text, gives each a local id (the sha1 of the document's text) and keeps a repository
file's text as its language reads it (standardize_file); it selects nothing else, and every source is taken whole.

python -m nanochat.corpus.prepare <source> ...
"""

import argparse
import asyncio
import fnmatch
import functools
import gzip
import hashlib
import multiprocessing
import resource
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import obstore
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import dataset_info, hf_hub_download, list_repo_files
from obstore.store import S3Store

from nanochat.corpus.schema import CONVERSATION, EXERCISE, ISSUE, NOTEBOOK, REPOSITORY_FILE
from nanochat.corpus.sources import EDU, SOURCES, Source, raw_dir, standardized_path

SPAWN = multiprocessing.get_context("spawn")


def hub_files(source: str, revision: str) -> list[str]:
    """The source's Hub files, every file its globs match at the revision, sorted."""
    spec = SOURCES[source]
    return sorted(f for f in list_repo_files(spec.repo, repo_type="dataset", revision=revision)
                  if any(fnmatch.fnmatchcase(f, glob) for glob in spec.files.values()))


@functools.cache
def swh_store() -> S3Store:
    """This process's client of Software Heritage's bucket, created once so its connections are reused across row
    groups; the file-descriptor limit is raised to its hard limit, as a row group's requests are in flight at once."""
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

    async def texts() -> list[str | None]:
        return await asyncio.gather(*map(blob, rows["blob_id"].to_pylist(), rows["src_encoding"].to_pylist()))

    return rows.append_column("text", pa.array(asyncio.run(texts()), pa.string()))


def fetch(source: str, file: str, revision: str) -> Path:
    """The local path of one of the source's Hub files, downloaded at the revision. Stack-Edu's blobs are fetched by a
    pool that lives for that file only, so its workers hold no memory beside standardize's."""
    spec = SOURCES[source]
    if spec.repo != EDU:
        return Path(hf_hub_download(spec.repo, file, repo_type="dataset", revision=revision,
                                    local_dir=raw_dir(source)))
    path = raw_dir(source) / file
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = hf_hub_download(spec.repo, file, repo_type="dataset", revision=revision)
    with (ProcessPoolExecutor(mp_context=SPAWN) as pool,
          pq.ParquetWriter(path, pq.read_schema(meta).append(pa.field("text", pa.string()))) as writer):
        for table in pool.map(blob_texts, [(meta, g) for g in range(pq.ParquetFile(meta).num_row_groups)]):
            writer.write_table(table)
    return path


# ---- standardize: one function per category, raw table -> its schema

def sha1(texts: pa.ChunkedArray) -> pa.Array:
    """The hex sha1 of each text's UTF-8 bytes."""
    return pa.array([hashlib.sha1(b).hexdigest() for b in texts.cast(pa.large_binary()).to_pylist()], pa.string())


def repository_file(t: pa.Table, spec: Source) -> pa.Table:
    return pa.table({"id": sha1(t["content"]), "repo_name": t["repo_name"], "relative_path": t["relative_path"],
                     "content": t["content"], "language": pa.repeat(spec.language, t.num_rows)},
                    schema=REPOSITORY_FILE)


def stack_edu(t: pa.Table, spec: Source) -> pa.Table:
    """A Stack-Edu file: its metadata path, which begins with "/", taken relative to the repository's root."""
    return pa.table({"id": sha1(t["text"]), "repo_name": t["repo_name"],
                     "relative_path": pc.utf8_ltrim(t["path"], characters="/"), "content": t["text"],
                     "language": pa.repeat(spec.language, t.num_rows)}, schema=REPOSITORY_FILE)


def exercise(t: pa.Table, spec: Source) -> pa.Table:
    return pa.table({"id": sha1(t["full_content"]), "language": pa.repeat(spec.language, t.num_rows)}
                    | {name: t[name] for name in ("task", "analysis", "solution", "test", "content", "content_format",
                                                   "full_content", "full_content_format")}
                    | {"seed": t["raw_content"]}, schema=EXERCISE)


def text(t: pa.Table, spec: Source) -> pa.Table:
    return pa.table({"id": sha1(t[spec.text]), "content": t[spec.text]}, schema=spec.schema)


def conversation(t: pa.Table, spec: Source) -> pa.Table:
    """A codeforces_cots row: the R1 prompt (ending with a line end) followed by its generation, and the problem's
    id."""
    content = pc.binary_join_element_wise(t["prompt"], t[spec.text], "")
    return pa.table({"id": sha1(content), "problem_id": t["id"], "content": content}, schema=CONVERSATION)


def issue(t: pa.Table, spec: Source) -> pa.Table:
    return pa.table({"id": sha1(t["text"]), "repo_name": t["repo_name"], "content": t["text"]}, schema=ISSUE)


def notebook(t: pa.Table, spec: Source) -> pa.Table:
    return pa.table({"id": sha1(t["text"]), "content": t["text"], "language": pa.repeat(spec.language, t.num_rows)},
                    schema=NOTEBOOK)


STANDARDIZE = {"stack_edu_python": stack_edu, "stack_edu_cpp": stack_edu,
               "ultradata_stack_v3_l2_python": repository_file, "ultradata_stack_v3_l2_cpp": repository_file,
               "ultradata_code_l2_python": repository_file, "ultradata_code_l2_cpp": repository_file,
               "ultradata_code_l3_python": exercise, "ultradata_code_l3_cpp": exercise,
               "ultradata_math_l1": text, "ultradata_math_l3_qa": text, "ultradata_math_l3_conversation": text,
               "ultradata_math_l3_multi_style": text, "ultradata_math_l3_textbook_exercise": text,
               "codeforces_cots": conversation, "github_issues": issue, "kaggle_notebooks": notebook,
               "climbmix": text, "opc_fineweb_math": text}


def standardize_file(task: tuple[str, int, Path]) -> None:
    """Standardize one raw file into its standardized file: the rows with text; a repository file's text as Python and
    C++ read source files, CR LF and a lone CR as LF and a leading U+FEFF dropped (Python Language Reference §2.1.2
    "Physical lines" and §2.1.4 "Encoding declarations"; C++23 [lex.phases] phases 1 and 2)."""
    source, k, path = task
    spec = SOURCES[source]
    pf = pq.ParquetFile(path)
    with pq.ParquetWriter(standardized_path(source, k), spec.schema) as writer:
        for g in range(pf.num_row_groups):
            raw = pf.read_row_group(g)
            texts = raw.filter(pc.is_valid(raw[spec.text]))
            if spec.schema is REPOSITORY_FILE:
                lines = pc.replace_substring_regex(texts[spec.text], "^\ufeff", "")
                lines = pc.replace_substring(pc.replace_substring(lines, "\r\n", "\n"), "\r", "\n")
                texts = texts.set_column(texts.schema.get_field_index(spec.text), spec.text, lines)
            writer.write_table(STANDARDIZE[source](texts, spec))


def prepare(sources: list[str]) -> None:
    """Fetch every Hub file of each source at its dataset's latest revision, then standardize its files in parallel."""
    for source in sources:
        revision = dataset_info(SOURCES[source].repo).sha
        files = hub_files(source, revision)
        standardized_path(source, 0).parent.mkdir(parents=True)
        paths = [fetch(source, file, revision) for file in files]
        with ProcessPoolExecutor(mp_context=SPAWN) as pool:
            list(pool.map(standardize_file, [(source, k, path) for k, path in enumerate(paths)]))
        print(f"{source}: {len(files)} files at {SOURCES[source].repo}@{revision}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m nanochat.corpus.prepare",
                                     description="fetch and standardize sources of the corpus")
    parser.add_argument("sources", nargs="+", choices=SOURCES)
    prepare(parser.parse_args().sources)
