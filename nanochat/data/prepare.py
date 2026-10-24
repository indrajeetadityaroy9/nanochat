"""
Prepare every pretraining source (sources.py): fetch, normalize into the Parquet store, then curate (curate.py).

  raw/<source>/                               the Hub files at their Hub paths (Stack-Edu: with Software Heritage text)
  normalized/<source>/<k>.parquet             one file per raw file, in its category's schema (schema.py) + val group

Fetch, the only network step, downloads a source's Hub files in sorted order at main. Stack-Edu's Hub
files are metadata: its raw file adds each blob's text, fetched from Software Heritage's public S3 bucket by blob ID
(obstore) and decoded with its src_encoding (The Stack v2 dataset card, download_contents); a blob the bucket does not
hold or that does not decode has no text.

The extent of each source follows from EXTENT, the groups of config.MIXTURE at Qwen2.5-Coder's code:text:math = 7:2:1
(arXiv 2409.12186 §3.1.2): the first group, code, is taken whole, and its tokens over its share are the horizon; every
later group takes its sources in order, and each source's Hub files in order, until it holds its share of the horizon
(ClimbMix, the text group, is a shuffled crawl). A mixture whose code share is at least 0.7 (config.MIXTURE: 0.87)
reads at most that much text and math per code token; pack.py keeps of each later group only what a run draws.
A file's tokens are its content column's uncompressed Parquet bytes over the o200k_base (tiktoken) bytes per
token of its source's first normalized row group, before deduplication (the pipeline's tokenizer is trained on the
curated store).

Normalize maps each raw file to its category's schema in parallel, with a local id (the sha1 of the text). A row needs
text; a repository file's text is kept as its language reads it (line ends, a leading U+FEFF; normalize_file), and a
repository file must then parse as Python 3 (ast.parse drops Python 2 and broken files; it runs after, as ast.parse on
a str rejects the leading U+FEFF that CPython's source decoding strips). A codeforces_cots document is its
conversation: the R1 prompt followed by its generation.

python -m nanochat.data.prepare
"""

import ast
import asyncio
import fnmatch
import functools
import gzip
import hashlib
import resource
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import obstore
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import hf_hub_download, list_repo_files
from obstore.store import S3Store

from nanochat.config import MIXTURE
from nanochat.data.curate import SPAWN, curate
from nanochat.data.sources import EDU, SOURCES, Source, column_bytes, normalized_path
from nanochat.paths import DATA

# ---- fetch

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


def fetch(source: str, file: str) -> Path:
    """The local path of one of the source's Hub files. Stack-Edu's blobs are fetched by a pool that lives for that file
    only, so its workers hold no memory beside normalize's."""
    spec = SOURCES[source]
    directory = DATA / "raw" / source
    if spec.repo != EDU:
        return Path(hf_hub_download(spec.repo, file, repo_type="dataset", local_dir=directory))
    path = directory / file
    meta = hf_hub_download(spec.repo, file, repo_type="dataset")
    with (ProcessPoolExecutor(mp_context=SPAWN) as pool,
          pq.ParquetWriter(path, pq.read_schema(meta).append(pa.field("text", pa.string()))) as writer):
        for table in pool.map(blob_texts, [(meta, g) for g in range(pq.ParquetFile(meta).num_row_groups)]):
            writer.write_table(table)
    return path


@functools.cache
def bytes_per_token(source: str) -> float:
    """The UTF-8 bytes per o200k_base (tiktoken) token of the source's first normalized row group."""
    texts = pq.ParquetFile(normalized_path(source, 0)).read_row_group(0, columns=["content"])["content"]
    count = sum(map(len, tiktoken.get_encoding("o200k_base").encode_ordinary_batch(texts.to_pylist())))
    return pc.sum(pc.binary_length(texts)).as_py() / count


def tokens(source: str, k: int) -> float:
    """The o200k tokens of a source's normalized file k: its content bytes at the source's bytes per token."""
    return column_bytes(normalized_path(source, k), "content") / bytes_per_token(source)


# ---- normalize: one function per source family, raw table -> category schema plus the row's val group

def sha1(texts: pa.ChunkedArray) -> pa.Array:
    """The hex sha1 of each text's UTF-8 bytes."""
    return pa.array([hashlib.sha1(b).hexdigest() for b in texts.cast(pa.large_binary()).to_pylist()], pa.string())


def normalized_schema(spec: Source) -> pa.Schema:
    """A source's normalized schema: its category's schema and the row's val group."""
    return spec.schema.append(pa.field("val_group", pa.string()))


def with_group(columns: dict[str, pa.ChunkedArray | pa.Array], spec: Source, group: pa.ChunkedArray) -> pa.Table:
    return pa.table(columns | {"val_group": group}, schema=normalized_schema(spec))


def stack_edu(t: pa.Table, spec: Source) -> pa.Table:
    n = t.num_rows
    return with_group({"id": sha1(t["text"]), "repo_name": t["repo_name"], "relative_path": t["path"],
                       "content": t["text"], "language": pa.repeat(spec.language, n), "int_score": t["int_score"],
                       "quality_score": pa.nulls(n), "category": pa.nulls(n)}, spec, t["repo_name"])


def ultradata(t: pa.Table, spec: Source) -> pa.Table:
    n = t.num_rows
    return with_group({"id": sha1(t["content"]), "repo_name": t["repo_name"], "relative_path": t["relative_path"],
                       "content": t["content"], "language": pa.repeat(spec.language, n), "int_score": pa.nulls(n),
                       "quality_score": t["quality_score"], "category": t["category"]}, spec, t["repo_name"])


def notebook(t: pa.Table, spec: Source) -> pa.Table:
    ids = sha1(t["text"])
    return with_group({"id": ids, "content": t["text"], "language": pa.repeat(spec.language, t.num_rows)}, spec, ids)


def text(t: pa.Table, spec: Source) -> pa.Table:
    ids = sha1(t[spec.text])
    url = t["url"] if "url" in t.column_names else pa.nulls(t.num_rows)
    return with_group({"id": ids, "url": url, "content": t[spec.text]}, spec, ids)


def conversation(t: pa.Table, spec: Source) -> pa.Table:
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


def parses_as_python3(text: str) -> bool:
    """Whether text parses as Python 3. ast.parse raises SyntaxError or ValueError for invalid source (Python Library
    Reference, compile(); UnicodeDecodeError is a ValueError), RecursionError or MemoryError for too deep nesting."""
    try:
        ast.parse(text)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False


def normalize_file(task: tuple[str, int, Path]) -> None:
    """Normalize one raw file into its normalized file: rows with text, a repository file's text as its language reads
    it, and a Python repository file only if it parses as Python 3."""
    source, k, path = task
    spec = SOURCES[source]
    warnings.filterwarnings("ignore", category=SyntaxWarning)  # invalid escapes in otherwise valid code
    pf = pq.ParquetFile(path)
    with pq.ParquetWriter(normalized_path(source, k), normalized_schema(spec)) as writer:
        for g in range(pf.num_row_groups):
            raw = pf.read_row_group(g)
            texts = raw.filter(pc.is_valid(raw[spec.text]))
            if spec.grammar is not None:
                # as Python and C++ read source files: CR LF and a lone CR read as LF, a leading U+FEFF dropped (Python
                # Language Reference §2.1.2 "Physical lines" and §2.1.4 "Encoding declarations"; C++23 [lex.phases]
                # phase 1: CR LF and a CR not followed by LF become one new-line; phase 2: a first U+FEFF is deleted)
                lines = pc.replace_substring_regex(texts[spec.text], "^\ufeff", "")
                lines = pc.replace_substring(pc.replace_substring(lines, "\r\n", "\n"), "\r", "\n")
                texts = texts.set_column(texts.schema.get_field_index(spec.text), spec.text, lines)
            table = NORMALIZE[source](texts, spec)
            if spec.grammar == "Python":
                table = table.filter(pa.array([parses_as_python3(t) for t in table["content"].to_pylist()], pa.bool_()))
            writer.write_table(table)


# The groups' shares of the store (module docstring)
EXTENT = {"code": 0.7, "text": 0.2, "math": 0.1}


def prepare() -> None:
    """Fetch, normalize and curate every source: config.MIXTURE's first group whole (its files normalized in parallel),
    every later group in Hub order until it holds its EXTENT share of the horizon (module docstring)."""
    normalized, horizon = {}, None
    for group, (_, sources) in MIXTURE.items():
        share = EXTENT[group]
        held = 0
        for source in sources:
            spec = SOURCES[source]
            files = sorted(f for f in list_repo_files(spec.repo, repo_type="dataset")
                           if any(fnmatch.fnmatchcase(f, glob) for glob in spec.files.values()))
            normalized_path(source, 0).parent.mkdir(parents=True)
            for directory in sorted({(DATA / "raw" / source / file).parent for file in files}):
                directory.mkdir(parents=True)  # the Hub paths' directories, where fetch writes
            if horizon is None:
                with ProcessPoolExecutor(mp_context=SPAWN) as pool:
                    list(pool.map(normalize_file, [(source, k, fetch(source, file)) for k, file in enumerate(files)]))
                count = len(files)
                held += sum(tokens(source, k) for k in range(count))
            else:
                count = 0
                while count < len(files) and held < share * horizon:
                    normalize_file((source, count, fetch(source, files[count])))
                    held += tokens(source, count)
                    count += 1
            normalized[source] = count
            print(f"{source}: {count} of {len(files)} files")
        horizon = horizon or held / share
    curate(normalized)


if __name__ == "__main__":
    prepare()
