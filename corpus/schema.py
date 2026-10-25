"""
Document categories. A standardized file holds one category's rows; id is the SHA-1 of the text.

REPOSITORY_FILE  a repository's source file, with its path and language
EXERCISE         a synthetic exercise as published: the task, then the solution
SYNTHETIC_FILE   a source file an LLM wrote from a repository file (rewritten, or transpiled)
FUNCTION         a function with its docstring (CodeSearchNet), with its repository
CONVERSATION     a Codeforces-CoTs record: the prompt, then the generation, with the problem
NOTEBOOK         a Kaggle notebook in Markdown
TEXT             a web document

A category's group decides the split: val iff md5_number(group) % SPLIT = 0 (curate.py); repository names lowercased,
as GitHub's are case-insensitive. `code`: the content is one whole program, which must parse when it is Python.
SAMPLE_FILE: a repository source's curated rows, cut into import-connected samples, numbered by `sample` per shard.
"""

import hashlib
from dataclasses import dataclass

import pyarrow as pa

SPLIT = 100  # 1% of groups held out
# Parquet codec of the files the corpus writes, the published dataset's (publish.DATASET)
COMPRESSION = "zstd"


def strings(*names: str) -> list[pa.Field]:
    return [pa.field(name, pa.string(), nullable=False) for name in names]


def sha1(texts: pa.Array | pa.ChunkedArray) -> pa.Array:
    """The hex sha1 of each text's UTF-8 bytes: a row's id."""
    return pa.array([hashlib.sha1(b).hexdigest() for b in texts.cast(pa.large_binary()).to_pylist()], pa.string())


@dataclass(frozen=True)
class Category:
    name: str  # keeps categories with equal schemas, groups and flags distinct
    schema: pa.Schema
    group: str  # the split's group, a DuckDB expression over the schema's columns
    code: bool


REPOSITORY_FILE = Category("repository file", pa.schema(strings("id", "repo_name", "relative_path", "content",
                                                                "language")), "lower(repo_name)", code=True)
EXERCISE = Category("exercise", pa.schema(strings("id", "content", "language")), "id", code=False)
SYNTHETIC_FILE = Category("synthetic file", pa.schema(strings("id", "content", "language")), "id", code=True)
FUNCTION = Category("function", pa.schema(strings("id", "repo_name", "content", "language")), "lower(repo_name)",
                    code=True)
CONVERSATION = Category("conversation", pa.schema(strings("id", "problem_id", "content")), "problem_id", code=False)
NOTEBOOK = Category("notebook", pa.schema(strings("id", "content", "language")), "id", code=False)
TEXT = Category("text", pa.schema(strings("id", "content")), "id", code=False)

SAMPLE_FILE = pa.schema([*strings("repo_name", "relative_path", "content"),
                         pa.field("sample", pa.int64(), nullable=False)])
