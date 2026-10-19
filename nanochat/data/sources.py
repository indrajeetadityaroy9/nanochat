"""
The pretraining sources: one mixture component each, a pinned Hub dataset and the files it reads, in the schema of its
document category (schema.py), and the training documents of their curated files.

files maps a language to the glob of the Hub files holding it (Linguist names for code; the language of a text corpus;
for codeforces_cots, the language of the R1 solutions each subset holds). text names the column of a fetched file that
holds a document's text (Stack-Edu's Hub files are metadata: prepare.fetch adds `text` from Software Heritage; a
codeforces_cots document is its `prompt` followed by this `generation`).
"""

import itertools
import os
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.data.schema import NOTEBOOK, REPOSITORY_FILE, TEXT
from nanochat.paths import data_dir

# The line comment of each repository-file language: Python Language Reference §2.1.3 "Comments", C++23 [lex.comment]
COMMENTS = {"Python": "#", "C++": "//"}


@dataclass(frozen=True)
class Source:
    schema: pa.Schema
    repo: str
    revision: str
    files: dict
    text: str

    @property
    def language(self):
        """The language of a source of one language (every source but codeforces_cots)."""
        (language,) = self.files
        return language

    @property
    def grammar(self):
        """The language whose tree-sitter grammar (data/bpe) parses a repository-file source, None for other sources:
        its files are grouped into repository samples by their imports (prepare.curate) and cut between statements
        into fill-in-the-middle examples (pack.py); prepare.py keeps them with canonical line ends
        (prepare.source_text) and keeps Python files only if they parse as Python 3."""
        return self.language if self.schema is REPOSITORY_FILE else None


EDU = ("HuggingFaceTB/stack-edu", "eeec5caac5cc3758a18f1d3ba4416837a9ba814c")  # metadata; text from Software Heritage
L2 = ("openbmb/UltraData-Stack-v3-L2", "d6bdb3e707cf17bbfb64067cd4c4ed84b0809baf")
CODE = ("openbmb/UltraData-Code", "525c9ac8639c5032633392d096e54db8c84a005e")

SOURCES = {
    "stack_edu_python": Source(REPOSITORY_FILE, *EDU, {"Python": "Python/*.parquet"}, "text"),
    "stack_edu_cpp": Source(REPOSITORY_FILE, *EDU, {"C++": "Cpp/*.parquet"}, "text"),
    "ultradata_stack_v3_l2_python": Source(REPOSITORY_FILE, *L2, {"Python": "data/UltraData-Stack-v3-L2/python/*.parquet"}, "content"),
    "ultradata_stack_v3_l2_cpp": Source(REPOSITORY_FILE, *L2, {"C++": "data/UltraData-Stack-v3-L2/cpp/*.parquet"}, "content"),
    "ultradata_code_l2_python": Source(REPOSITORY_FILE, *CODE, {"Python": "data/UltraData-Code-L2/py/*.parquet"}, "content"),
    "ultradata_code_l2_cpp": Source(REPOSITORY_FILE, *CODE, {"C++": "data/UltraData-Code-L2/cpp/*.parquet"}, "content"),
    "kaggle_notebooks": Source(NOTEBOOK, "HuggingFaceTB/issues-kaggle-notebooks", "ef882ad1ed8274340e8fc9bac087c903f2f75396",
                               {"Python": "kaggle/*.parquet"}, "text"),
    "codeforces_cots": Source(TEXT, "open-r1/codeforces-cots", "39ac85c150806230473c70ad72c31f6232fe3f41",
                              {"Python": "solutions_py_decontaminated/*.parquet",
                               "C++": "solutions_short_and_long_decontaminated/*.parquet"}, "generation"),
    "climbmix": Source(TEXT, "karpathy/climbmix-400b-shuffle", "915333b4f8b8684f39aeaafea600fea6f43fb703",
                       {"English": "shard_*.parquet"}, "text"),
    "opc_fineweb_math": Source(TEXT, "OpenCoder-LLM/opc-fineweb-math-corpus", "858b10c748e6c95e0cbc5ebd38543b1c4699857b",
                               {"English": "data/train-*.parquet"}, "text"),
    "ultradata_math_l2": Source(TEXT, "openbmb/UltraData-Math", "fe10db8efd35597fd7fcff8ff576b5ec4ea5ff87",
                                {"English": "data/UltraData-Math-L2-preview/*.parquet"}, "content"),
}


def curated_dir(source, split):
    """<data_dir>/curated/<source>/<split>: the source's curated shards of a split (prepare.curate)."""
    return os.path.join(data_dir(), "curated", source, split)


def curated_files(source, split):
    """Local paths of the source's curated shards of a split in the order of their number k (<k>.parquet: a
    repository-file source's partition, any other source's normalized file; prepare.curate)."""
    directory = curated_dir(source, split)
    return [os.path.join(directory, f) for f in sorted(os.listdir(directory), key=lambda f: int(f.removesuffix(".parquet")))]


def column_bytes(path, column):
    """The uncompressed bytes of a column of a parquet file, from its footer."""
    md = pq.read_metadata(path)
    j = md.schema.names.index(column)
    return sum(md.row_group(g).column(j).total_uncompressed_size for g in range(md.num_row_groups))


def texts(source, table):
    """The text of each row of a curated table: its content; for a file of a repository sample, the content after a
    line comment with the file's path (DeepSeek-Coder arXiv 2401.14196 §2.2: "a comment indicating the file's path is
    added at the beginning of each file"), with a line end appended when it does not end with one, as a non-empty source
    file is read (C++23 [lex.phases] phase 2; Python Language Reference §2.1.2: the end of input terminates the last line)."""
    contents = table["content"].to_pylist()
    spec = SOURCES[source]
    if spec.schema is not REPOSITORY_FILE:
        return contents
    comment = COMMENTS[spec.grammar]
    return [f"{comment} {path}\n{text.removesuffix('\n')}\n" for path, text in zip(table["relative_path"].to_pylist(), contents)]


def documents(source, table):
    """The training documents of a curated row group: a repository sample, its files' texts (consecutive rows of one
    sample, which a row group holds whole) concatenated in the sample's order; any other row's text."""
    rows = texts(source, table)
    if SOURCES[source].schema is not REPOSITORY_FILE:
        return rows
    starts = np.flatnonzero(np.diff(table["sample"].to_numpy(), prepend=-1)).tolist()
    return ["".join(rows[a:b]) for a, b in itertools.pairwise(starts + [len(rows)])]
