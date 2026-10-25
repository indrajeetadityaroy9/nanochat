"""
The document categories of the corpus's sources (docs/corpus.md §2). Every standardized file holds one
category's rows in its schema; id is the sha1 of the document's text, computed locally.

REPOSITORY_FILE  a source file of a code repository, with its path in the repository and its language
EXERCISE         a synthetic programming exercise as its publisher serializes it: the task (a code comment, unfenced
                 text or Markdown), then the solution
SYNTHETIC_FILE   a source file an LLM wrote from a repository file (rewritten in its language, or transpiled)
FUNCTION         a function of a repository file with its docstring (CodeSearchNet), with its repository
CONVERSATION     a Codeforces-CoTs record: the R1 prompt, then its generation (reasoning and solution), with the problem
NOTEBOOK         a Kaggle notebook in Markdown: its title, its datasets' description, its code as a script
TEXT             a web document

A category fixes the split's group: a document is val when md5_number(group) % SPLIT = 0 (curate.py), a repository
by its lowercased name, GitHub's names being case-insensitive. Its `code` flag marks a content that is one whole
program text in its source's language, which must parse when that language is Python (curate.check).

SAMPLE_FILE is a repository source's curated layout: its files cut into import-connected samples, a sample's files
consecutive rows in its dependency order, numbered by `sample` within their shard.
"""

from dataclasses import dataclass

import pyarrow as pa

SPLIT = 100  # 1% of groups held out, production's split (docs/corpus.md §4.4)


def strings(*names: str) -> list[pa.Field]:
    return [pa.field(name, pa.string(), nullable=False) for name in names]


@dataclass(frozen=True)
class Category:
    name: str
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
