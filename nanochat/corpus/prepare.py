"""
Prepare sources (sources.py): fetch each Hub file a source's globs match, in sorted order at its pinned revision, then
standardize each into its category's schema (schema.py), one standardized file per Hub file.

Standardize keeps the rows with text, gives each a local id (the sha1 of the document's text) and keeps a repository
file's text as its language reads it (standardize_file); it selects nothing else, and every source is taken whole.

python -m nanochat.corpus.prepare <source> ...
"""

import argparse
import fnmatch
import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

from nanochat.corpus.schema import EXERCISE, REPOSITORY_FILE
from nanochat.corpus.sources import SOURCES, Source, raw_dir, standardized_path

SPAWN = multiprocessing.get_context("spawn")


def hub_files(source: str) -> list[str]:
    """The source's Hub files, every file its globs match at its revision, sorted."""
    spec = SOURCES[source]
    return sorted(f for f in list_repo_files(spec.repo, repo_type="dataset", revision=spec.revision)
                  if any(fnmatch.fnmatchcase(f, glob) for glob in spec.files.values()))


def fetch(source: str, file: str) -> Path:
    """The local path of one of the source's Hub files, downloaded at its pinned revision."""
    spec = SOURCES[source]
    return Path(hf_hub_download(spec.repo, file, repo_type="dataset", revision=spec.revision,
                                local_dir=raw_dir(source)))


# ---- standardize: one function per category, raw table -> its schema

def sha1(texts: pa.ChunkedArray) -> pa.Array:
    """The hex sha1 of each text's UTF-8 bytes."""
    return pa.array([hashlib.sha1(b).hexdigest() for b in texts.cast(pa.large_binary()).to_pylist()], pa.string())


def repository_file(t: pa.Table, spec: Source) -> pa.Table:
    return pa.table({"id": sha1(t["content"]), "repo_name": t["repo_name"], "relative_path": t["relative_path"],
                     "content": t["content"], "language": pa.repeat(spec.language, t.num_rows),
                     "category": t["category"], "algo_rel_score": t["algo_rel_score"],
                     "quality_score": t["quality_score"]}, schema=REPOSITORY_FILE)


def exercise(t: pa.Table, spec: Source) -> pa.Table:
    return pa.table({"id": sha1(t["full_content"]), "language": pa.repeat(spec.language, t.num_rows)}
                    | {name: t[name] for name in ("task", "analysis", "solution", "test", "content", "content_format",
                                                   "full_content", "full_content_format")}
                    | {"seed": t["raw_content"]}, schema=EXERCISE)


def text(t: pa.Table, spec: Source) -> pa.Table:
    return pa.table({"id": sha1(t[spec.text]), "content": t[spec.text]}, schema=spec.schema)


STANDARDIZE = {"ultradata_code_l2_python": repository_file, "ultradata_code_l2_cpp": repository_file,
               "ultradata_code_l3_python": exercise, "ultradata_code_l3_cpp": exercise, "climbmix": text}


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
    """Fetch every Hub file of each source, then standardize its files in parallel."""
    for source in sources:
        files = hub_files(source)
        standardized_path(source, 0).parent.mkdir(parents=True)
        paths = [fetch(source, file) for file in files]
        with ProcessPoolExecutor(mp_context=SPAWN) as pool:
            list(pool.map(standardize_file, [(source, k, path) for k, path in enumerate(paths)]))
        print(f"{source}: {len(files)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m nanochat.corpus.prepare",
                                     description="fetch and standardize sources of the corpus")
    parser.add_argument("sources", nargs="+", choices=SOURCES)
    prepare(parser.parse_args().sources)
