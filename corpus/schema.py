"""
Document categories: each a schema of non-null strings, the group deciding the split, and whether content is code.
"""

import hashlib
from dataclasses import dataclass

import pyarrow as pa

SPLIT = 100  # 1% of groups held out
# the published dataset's codec
COMPRESSION = "zstd"


def strings(*names: str) -> list[pa.Field]:
    return [pa.field(name, pa.string(), nullable=False) for name in names]


def sha1(texts: pa.Array | pa.ChunkedArray) -> pa.Array:
    return pa.array([hashlib.sha1(b).hexdigest() for b in texts.cast(pa.large_binary()).to_pylist()], pa.string())


@dataclass(frozen=True)
class Category:
    name: str  # keeps categories with equal schemas, groups and flags distinct
    schema: pa.Schema
    group: str  # a DuckDB expression over the schema's columns
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
