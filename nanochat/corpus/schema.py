"""
The document categories of the corpus's sources: every standardized file holds one. id is computed locally, the sha1
of the document's text; no publisher's identifier is kept as id.

REPOSITORY_FILE: a source file of a code repository, with its path in the repository and its language (Linguist
name).
EXERCISE: a synthetic programming exercise in its parts (task, analysis, solution, test), the publisher's two
renderings of it (content: the task and the solution; full_content: all four parts, the document's text) with their
formats, and the source file it was written from (seed).
CONVERSATION: a codeforces_cots record as one document, the R1 prompt (the problem and its instructions) followed by
its generation (a reasoning trace and a solution), with its problem's id (problem_id), which every generation of the
problem shares.
TEXT: a web document.
"""

import pyarrow as pa

REPOSITORY_FILE = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("repo_name", pa.string(), nullable=False),
    pa.field("relative_path", pa.string(), nullable=False),
    pa.field("content", pa.string(), nullable=False),
    pa.field("language", pa.string(), nullable=False),
])

EXERCISE = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("language", pa.string(), nullable=False),
    pa.field("task", pa.string()),
    pa.field("analysis", pa.string()),
    pa.field("solution", pa.string()),
    pa.field("test", pa.string()),
    pa.field("content", pa.string()),
    pa.field("content_format", pa.string()),
    pa.field("full_content", pa.string(), nullable=False),
    pa.field("full_content_format", pa.string()),
    pa.field("seed", pa.string()),
])

TEXT = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("content", pa.string(), nullable=False),
])

CONVERSATION = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("problem_id", pa.string(), nullable=False),
    pa.field("content", pa.string(), nullable=False),
])
