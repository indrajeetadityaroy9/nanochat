"""
The corpus's sources (SOURCES, in deduplication's keep-first order) and its file paths under paths.CORPUS.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from corpus.schema import (
    CONVERSATION,
    EXERCISE,
    FUNCTION,
    NOTEBOOK,
    REPOSITORY_FILE,
    SYNTHETIC_FILE,
    TEXT,
    Category,
)
from native import corpus_crate
from paths import CORPUS

# Python Language Reference §2.1.3, C++23 [lex.comment]
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
    """The source's fetched Hub files at their Hub paths."""
    return CORPUS / "raw" / source


def standardized_path(source: str, k: int) -> Path:
    """The source's fetched Hub file k (prepare.extent's order), standardized."""
    return CORPUS / "standardized" / source / f"{k}.parquet"


def checked_path(source: str, k: int) -> Path:
    """A Python source's standardized file k after the Python 3 check (curate.check)."""
    return CORPUS / "checked" / source / f"{k}.parquet"


def curated_dir(source: str, split: str) -> Path:
    """The source's curated shards of a split, <k>.parquet."""
    return CORPUS / "curated" / source / split


def curated_shards(source: str, split: str) -> list[Path]:
    return sorted(curated_dir(source, split).iterdir(), key=lambda path: int(path.stem))


def text_bytes(metadata: pq.FileMetaData, column: str) -> int:
    """Text bytes of a Parquet string column estimated from its footer: uncompressed size less 4-byte length prefixes
    (page headers, dictionary and level data still counted)."""
    index = metadata.schema.to_arrow_schema().get_field_index(column)
    return sum(metadata.row_group(g).column(index).total_uncompressed_size
               for g in range(metadata.num_row_groups)) - 4 * metadata.num_rows


def footer_rows(paths: Iterable[Path]) -> int:
    """The rows of Parquet files, from their footers."""
    return sum(pq.read_metadata(path).num_rows for path in paths)


def texts(source: str, table: pa.Table) -> list[str]:
    """Each curated row's text; a repository file's after a line comment with its path (arXiv 2401.14196 §2.2), ending
    with a line end, a C++ file ending in a backslash with one more (C++23 [lex.phases] phase 2)."""
    contents = table["content"].to_pylist()
    spec = SOURCES[source]
    if spec.category != REPOSITORY_FILE:
        return contents
    comment = COMMENTS[spec.language]
    bodies = [text.removesuffix("\n") for text in contents]
    return [f"{comment} {path.replace('\r', '\\r').replace('\n', '\\n')}\n{body}\n"
            + ("\n" if spec.language == "C++" and body.endswith("\\") else "")
            for path, body in zip(table["relative_path"].to_pylist(), bodies, strict=True)]


def repository_samples(table: pa.Table, keys: tuple[str, ...], language: str, *, one_per_path: bool
                       ) -> tuple[pa.Table, list[int], list[list[int]]]:
    """Repository files sorted by lowercased repository, then `keys` (with one_per_path, the first row per path).
    Returns the table in one chunk, each repository's end row and its import-connected samples (topological_sort)."""
    table = table.append_column("repository", pc.utf8_lower(table["repo_name"]))
    table = table.sort_by([(key, "ascending") for key in ("repository", *keys)]).combine_chunks()
    if one_per_path:
        paths = pc.binary_join_element_wise(table["repository"], table["relative_path"], "\0").chunk(0)
        first = pc.not_equal(paths[1:], paths[:-1]).to_numpy(zero_copy_only=False)
        table = table.filter(np.concatenate([[True], first])).combine_chunks()
    ends = pc.run_end_encode(table["repository"].chunk(0)).run_ends.to_numpy().tolist()
    samples = corpus_crate().topological_sort(table["relative_path"].to_pylist(), table["content"].to_pylist(),
                                             ends, grammar=language)
    return table, ends, samples


def sample_starts(samples: np.ndarray) -> list[int]:
    """The first row of each sample, given each row's sample number."""
    return np.flatnonzero(np.diff(samples, prepend=-1)).tolist()
