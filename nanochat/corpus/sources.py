"""
The corpus's sources (docs/corpus.md §1) and its file paths under paths.CORPUS. A source: a Hub dataset, its files per
language (a Linguist name for code), its category (schema.py) and the column holding a document's text. SOURCES is in
deduplication's keep-first order (curate.py).
- Stack-Edu (arXiv 2502.02737): The Stack v2's educational files, by Software Heritage blob id (prepare.fetch).
- UltraData-Stack-v3-L2: The Stack v3's files that UltraData's L2 classifiers select.
- UltraData-Code (arXiv 2610.23960): L2, selected source files; L3, exercises written from L2's algorithmic files.
- Nemotron-Pretraining-Code-v2 (arXiv 2512.20848 §2.2.2): every Python file of NVIDIA's GitHub corpus rewritten by
  Qwen3-32B, and Python files transpiled into C++.
- CodeXGLUE code-to-text (arXiv 2102.04664): CodeSearchNet's functions with their docstrings; its train split only.
- UltraData-Math L3 (arXiv 2602.09003): synthesized mathematics in four formats.
- Codeforces-CoTs: DeepSeek-R1's reasoning traces and Python or C++ solutions to Codeforces problems.
- issues-kaggle-notebooks: StarCoder2's Kaggle notebooks (arXiv 2402.19173 §2.4.2) as text.
- ClimbMix: English web text.
"""

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.corpus.schema import (
    CONVERSATION,
    EXERCISE,
    FUNCTION,
    NOTEBOOK,
    REPOSITORY_FILE,
    SYNTHETIC_FILE,
    TEXT,
    Category,
)
from nanochat.paths import CORPUS

# The line comment of each repository-file language: Python Language Reference §2.1.3 "Comments", C++23 [lex.comment]
COMMENTS = {"Python": "#", "C++": "//"}


@dataclass(frozen=True)
class Source:
    category: Category
    repo: str
    files: dict[str, str]  # language -> glob of the Hub files holding it
    text: str  # the fetched column holding the document's text

    @property
    def language(self) -> str:
        (language,) = self.files
        return language


EDU = "HuggingFaceTB/stack-edu"
STACK_V3 = "openbmb/UltraData-Stack-v3-L2"
CODE = "openbmb/UltraData-Code"
NEMOTRON = "nvidia/Nemotron-Pretraining-Code-v2"
MATH = "openbmb/UltraData-Math"

SOURCES = {
    "stack_edu_python": Source(REPOSITORY_FILE, EDU, {"Python": "Python/*.parquet"}, "text"),
    "stack_edu_cpp": Source(REPOSITORY_FILE, EDU, {"C++": "Cpp/*.parquet"}, "text"),
    "ultradata_stack_v3_l2_python": Source(REPOSITORY_FILE, STACK_V3,
                                           {"Python": "data/UltraData-Stack-v3-L2/python/*.parquet"}, "content"),
    "ultradata_stack_v3_l2_cpp": Source(REPOSITORY_FILE, STACK_V3,
                                        {"C++": "data/UltraData-Stack-v3-L2/cpp/*.parquet"}, "content"),
    "ultradata_code_l2_python": Source(REPOSITORY_FILE, CODE, {"Python": "data/UltraData-Code-L2/py/*.parquet"},
                                       "content"),
    "ultradata_code_l2_cpp": Source(REPOSITORY_FILE, CODE, {"C++": "data/UltraData-Code-L2/cpp/*.parquet"}, "content"),
    "ultradata_code_l3_python": Source(EXERCISE, CODE, {"Python": "data/UltraData-Code-L3/py/*.parquet"}, "content"),
    "ultradata_code_l3_cpp": Source(EXERCISE, CODE, {"C++": "data/UltraData-Code-L3/cpp/*.parquet"}, "content"),
    "nemotron_rewriting_python": Source(SYNTHETIC_FILE, NEMOTRON,
                                        {"Python": "Synthetic-Code/synthetic-rewriting/*.parquet"}, "content"),
    "nemotron_transpilation_cpp": Source(SYNTHETIC_FILE, NEMOTRON,
                                         {"C++": "Synthetic-Code/synthetic-transpilation/*.parquet"}, "content"),
    "codexglue_python": Source(FUNCTION, "google/code_x_glue_ct_code_to_text", {"Python": "python/train-*.parquet"},
                               "code"),
    "ultradata_math_l3_qa": Source(TEXT, MATH, {"English": "data/UltraData-Math-L3/QA-Synthetic/*.parquet"},
                                   "content"),
    "ultradata_math_l3_conversation": Source(
        TEXT, MATH, {"English": "data/UltraData-Math-L3/Conversation-Synthetic/*.parquet"}, "content"),
    "ultradata_math_l3_multi_style": Source(
        TEXT, MATH, {"English": "data/UltraData-Math-L3/Multi-Style-Synthetic/*.parquet"}, "content"),
    "ultradata_math_l3_textbook_exercise": Source(
        TEXT, MATH, {"English": "data/UltraData-Math-L3/Textbook-Exercise-Synthetic/*.parquet"}, "content"),
    "codeforces_cots": Source(CONVERSATION, "open-r1/codeforces-cots",
                              {"Python": "solutions_py_decontaminated/*.parquet",
                               "C++": "solutions_short_and_long_decontaminated/*.parquet"}, "generation"),
    "kaggle_notebooks": Source(NOTEBOOK, "HuggingFaceTB/issues-kaggle-notebooks", {"Python": "kaggle/*.parquet"},
                               "text"),
    "climbmix": Source(TEXT, "karpathy/climbmix-400b-shuffle", {"English": "shard_*.parquet"}, "text"),
}


def raw_dir(source: str) -> Path:
    """The source's fetched Hub files at their Hub paths (prepare.fetch)."""
    return CORPUS / "raw" / source


def standardized_path(source: str, k: int) -> Path:
    """The source's fetched Hub file k (prepare.extent's order) in its category's schema (prepare.standardize)."""
    return CORPUS / "standardized" / source / f"{k}.parquet"


def checked_path(source: str, k: int) -> Path:
    """A Python source's standardized file k with every file parsing as Python 3, repaired where it did not
    (curate.check)."""
    return CORPUS / "checked" / source / f"{k}.parquet"


def curated_dir(source: str, split: str) -> Path:
    """The source's curated shards of a split, <k>.parquet (curate.py)."""
    return CORPUS / "curated" / source / split


def curated_shards(source: str, split: str) -> list[Path]:
    """The source's curated shards of a split in the order of their number k."""
    return sorted(curated_dir(source, split).iterdir(), key=lambda path: int(path.stem))


def text_bytes(metadata: pq.FileMetaData, column: str) -> int:
    """The bytes of a Parquet file's string column, from its footer: the column's uncompressed size less each value's
    4-byte length prefix."""
    index = metadata.schema.to_arrow_schema().get_field_index(column)
    return sum(metadata.row_group(g).column(index).total_uncompressed_size
               for g in range(metadata.num_row_groups)) - 4 * metadata.num_rows


def texts(source: str, table: pa.Table) -> list[str]:
    """Each curated row's text: its content; a repository file's is preceded by a line comment with its path
    (DeepSeek-Coder, arXiv 2401.14196 §2.2) and ends with a line end (C++23 [lex.phases] phase 2; Python Language
    Reference §2.1.2). A path's line breaks are escaped (\\r, \\n) so the comment stays one line; a C++ file ending in
    a backslash gets an empty line, so phase 2 does not splice the next file's comment onto it."""
    contents = table["content"].to_pylist()
    spec = SOURCES[source]
    if spec.category != REPOSITORY_FILE:
        return contents
    comment = COMMENTS[spec.language]
    bodies = [text.removesuffix("\n") for text in contents]
    return [f"{comment} {path.replace('\r', '\\r').replace('\n', '\\n')}\n{body}\n"
            + ("\n" if spec.language == "C++" and body.endswith("\\") else "")
            for path, body in zip(table["relative_path"].to_pylist(), bodies, strict=True)]
