"""
The document categories of the pretraining sources: every normalized file holds one of the first three schemas plus
each row's val group (prepare.with_group), and the curated files of the repository-file sources hold SAMPLE_FILE.

id, in every normalized schema, is computed locally: the sha1 of the document's text. No publisher's identifier is
kept as id (codeforces_cots's problem id is its val group).

REPOSITORY_FILE: a source file of a code repository, with its path in the repository, its language (Linguist name) and
the publisher's selection scores.
NOTEBOOK: a notebook as one document, with its language.
TEXT: a web or math document, or a codeforces_cots conversation (its R1 prompt and generation).
SAMPLE_FILE: a curated repository file and its repository sample (prepare.curate): the files of a sample are
consecutive rows, in the sample's order.
"""

import pyarrow as pa

REPOSITORY_FILE = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("repo_name", pa.string(), nullable=False),
    pa.field("relative_path", pa.string(), nullable=False),
    pa.field("content", pa.string(), nullable=False),
    pa.field("language", pa.string(), nullable=False),
    pa.field("int_score", pa.int64()),        # Stack-Edu's educational score
    pa.field("quality_score", pa.float64()),  # UltraData's
    pa.field("category", pa.string()),        # UltraData's
])

NOTEBOOK = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("content", pa.string(), nullable=False),
    pa.field("language", pa.string(), nullable=False),
])

TEXT = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("url", pa.string()),
    pa.field("content", pa.string(), nullable=False),
])

SAMPLE_FILE = pa.schema([
    pa.field("repo_name", pa.string(), nullable=False),
    pa.field("relative_path", pa.string(), nullable=False),
    pa.field("content", pa.string(), nullable=False),
    pa.field("sample", pa.int64(), nullable=False),  # the sample's number in its curated file
])
