"""
The corpus's sources: a Hub dataset each, read at its latest revision (prepare.prepare), the files it reads, its
document category (schema.py) and the column of a fetched file that holds a document's text.

files maps a language to the glob of the Hub files holding it (Linguist names for code; the language of a text corpus;
for codeforces_cots, the language of the R1 solutions each subset holds). Stack-Edu (arXiv 2502.02737) holds The Stack
v2's educational files by Software Heritage blob ID, their text fetched from Software Heritage (prepare.fetch).
UltraData-Stack-v3-L2 holds The Stack v3's files that UltraData's L2 classifiers select. UltraData-Code
(arXiv 2610.23960) releases L2, selected source files, and L3, exercises written from L2's algorithmic files; both per
language. UltraData-Math (arXiv 2602.09003) releases L1, Common Crawl's math pages cleaned and deduplicated, one folder
per crawl snapshot, and L3, documents synthesized in four formats, a source each. Codeforces-CoTs holds DeepSeek-R1's
reasoning traces and solutions to Codeforces problems: of its decontaminated subsets, the Python solutions and the C++
solutions (short and long). issues-kaggle-notebooks holds StarCoder2's GitHub issues and Kaggle notebooks
(arXiv 2402.19173) reformatted as text, a source each. opc-fineweb-math-corpus holds the math pages OpenCoder's
fastText classifiers recalled from FineWeb (arXiv 2411.04905). Nemotron-Pretraining-Code-v2 (arXiv 2512.20848 §2.2)
holds, of its synthetic code, every Python file of NVIDIA's GitHub corpus rewritten by Qwen3-32B (SGCR and SCOR prompts,
arXiv 2505.02881, and NVIDIA's own) and Python files transpiled into C++, a source each. CodeXGLUE's code-to-text
(arXiv 2102.04664) holds CodeSearchNet's GitHub functions with their docstrings (arXiv 1909.09436); its Python train
split is read, not the benchmark's validation and test splits.
"""

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from nanochat.corpus.schema import (
    CONVERSATION,
    EXERCISE,
    FUNCTION,
    ISSUE,
    NOTEBOOK,
    REPOSITORY_FILE,
    SYNTHETIC_FILE,
    TEXT,
)
from nanochat.paths import CORPUS

# The line comment of each repository-file language: Python Language Reference §2.1.3 "Comments", C++23 [lex.comment]
COMMENTS = {"Python": "#", "C++": "//"}


@dataclass(frozen=True)
class Source:
    schema: pa.Schema
    repo: str
    files: dict[str, str]
    text: str

    @property
    def language(self) -> str:
        (language,) = self.files
        return language


EDU = "HuggingFaceTB/stack-edu"
STACK_V3 = "openbmb/UltraData-Stack-v3-L2"
CODE = "openbmb/UltraData-Code"
MATH = "openbmb/UltraData-Math"
ISSUES_KAGGLE = "HuggingFaceTB/issues-kaggle-notebooks"
NEMOTRON = "nvidia/Nemotron-Pretraining-Code-v2"
CODEXGLUE = "google/code_x_glue_ct_code_to_text"

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
    "codexglue_python": Source(FUNCTION, CODEXGLUE, {"Python": "python/train-*.parquet"}, "code"),
    "ultradata_math_l1": Source(TEXT, MATH, {"English": "data/UltraData-Math-L1/*/*.parquet"}, "content"),
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
    "github_issues": Source(ISSUE, ISSUES_KAGGLE, {"English": "issues/*.parquet"}, "text"),
    "kaggle_notebooks": Source(NOTEBOOK, ISSUES_KAGGLE, {"Python": "kaggle/*.parquet"}, "text"),
    "climbmix": Source(TEXT, "karpathy/climbmix-400b-shuffle", {"English": "shard_*.parquet"}, "text"),
    "opc_fineweb_math": Source(TEXT, "OpenCoder-LLM/opc-fineweb-math-corpus", {"English": "data/train-*.parquet"},
                               "text"),
}


def raw_dir(source: str) -> Path:
    """CORPUS/raw/<source>: the source's Hub files at their Hub paths, the columns standardization reads
    (prepare.fetch)."""
    return CORPUS / "raw" / source


def standardized_path(source: str, k: int) -> Path:
    """CORPUS/standardized/<source>/<k>.parquet: the source's fetched Hub file k in the order prepare.extent gives, in
    its schema (prepare.standardize_file)."""
    return CORPUS / "standardized" / source / f"{k}.parquet"


def checked_path(source: str, k: int) -> Path:
    """CORPUS/checked/<source>/<k>.parquet: a Python repository source's standardized file k with every file parsing as
    Python 3, repaired where it did not (curate.check)."""
    return CORPUS / "checked" / source / f"{k}.parquet"


def curated_dir(source: str, split: str) -> Path:
    """CORPUS/curated/<source>/<split>: the source's curated shards of a split, <k>.parquet (curate.py)."""
    return CORPUS / "curated" / source / split


def texts(source: str, table: pa.Table) -> list[str]:
    """The text of each row of a curated table: its content; for a repository file, the content after a line comment
    with the file's path (DeepSeek-Coder arXiv 2401.14196 §2.2: "a comment indicating the file's path is added at the
    beginning of each file"), with a line end appended when it does not end with one, as a non-empty source file is
    read (C++23 [lex.phases] phase 2; Python Language Reference §2.1.2: the end of input terminates the last line). A
    path's line breaks are written as the escapes \\r and \\n, so the comment stays one line; a C++ file whose last
    line ends in a backslash gets an empty line after it, as phase 2 splices a backslash and the line end after it,
    which would join the next file's comment to that line."""
    contents = table["content"].to_pylist()
    spec = SOURCES[source]
    if spec.schema is not REPOSITORY_FILE:
        return contents
    comment = COMMENTS[spec.language]
    bodies = [text.removesuffix("\n") for text in contents]
    return [f"{comment} {path.replace('\r', '\\r').replace('\n', '\\n')}\n{body}\n"
            + ("\n" if spec.language == "C++" and body.endswith("\\") else "")
            for path, body in zip(table["relative_path"].to_pylist(), bodies, strict=True)]
