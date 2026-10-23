"""
The document categories of the corpus's sources: every standardized file holds one. id is computed locally, the sha1
of the document's text; no publisher's identifier is kept as id.

REPOSITORY_FILE: a source file of a code repository, with its path in the repository and its language (Linguist
name).
EXERCISE: a synthetic programming exercise as the publisher serializes its task and solution (content: the task as a
code comment, unfenced text or Markdown, then the solution), with its language.
CONVERSATION: a codeforces_cots record as one document, the R1 prompt (the problem and its instructions) followed by
its generation (a reasoning trace and a solution), with its problem's id (problem_id), which every generation of the
problem shares.
ISSUE: a GitHub issue thread as one document (its title, the question and the answers, each by an anonymized user,
and its status), with its repository.
NOTEBOOK: a Kaggle notebook as one document, in Markdown (its title, the description of the datasets it reads, its
code converted to a script in code blocks), with the code's language.
TEXT: a web document.
SAMPLE_FILE: a curated repository file and its import-connected sample (curate.py): the files of a sample are
consecutive rows, in the sample's dependency order.

The split (curate.py): a document's group is val when md5_number(group) % SPLIT is 0. The group is the repository
(REPOSITORY_FILE, ISSUE), the problem (CONVERSATION), else the document.
"""

import pyarrow as pa

SPLIT = 100  # production's split (nanochat/data/curate.py): 1% of groups held out

REPOSITORY_FILE = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("repo_name", pa.string(), nullable=False),
    pa.field("relative_path", pa.string(), nullable=False),
    pa.field("content", pa.string(), nullable=False),
    pa.field("language", pa.string(), nullable=False),
])

EXERCISE = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("content", pa.string(), nullable=False),
    pa.field("language", pa.string(), nullable=False),
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

ISSUE = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("repo_name", pa.string(), nullable=False),
    pa.field("content", pa.string(), nullable=False),
])

NOTEBOOK = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("content", pa.string(), nullable=False),
    pa.field("language", pa.string(), nullable=False),
])

SAMPLE_FILE = pa.schema([
    pa.field("repo_name", pa.string(), nullable=False),
    pa.field("relative_path", pa.string(), nullable=False),
    pa.field("content", pa.string(), nullable=False),
    pa.field("sample", pa.int64(), nullable=False),  # the sample's number in its curated file
])
