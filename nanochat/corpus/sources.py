"""
The corpus's sources: a pinned Hub dataset each, the files it reads, its document category (schema.py) and the column
of a fetched file that holds a document's text.

files maps a language to the glob of the Hub files holding it (Linguist names for code; the language of a text
corpus). UltraData-Code (arXiv 2610.23960) releases L2, selected source files, and L3, exercises written from L2's
algorithmic files; both per language.
"""

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from nanochat.corpus.schema import EXERCISE, REPOSITORY_FILE, TEXT
from nanochat.paths import CORPUS


@dataclass(frozen=True)
class Source:
    schema: pa.Schema
    repo: str
    revision: str
    files: dict[str, str]
    text: str

    @property
    def language(self) -> str:
        (language,) = self.files
        return language


CODE = ("openbmb/UltraData-Code", "525c9ac8639c5032633392d096e54db8c84a005e")
CLIMBMIX = ("karpathy/climbmix-400b-shuffle", "915333b4f8b8684f39aeaafea600fea6f43fb703")

SOURCES = {
    "ultradata_code_l2_python": Source(REPOSITORY_FILE, *CODE, {"Python": "data/UltraData-Code-L2/py/*.parquet"},
                                       "content"),
    "ultradata_code_l2_cpp": Source(REPOSITORY_FILE, *CODE, {"C++": "data/UltraData-Code-L2/cpp/*.parquet"}, "content"),
    "ultradata_code_l3_python": Source(EXERCISE, *CODE, {"Python": "data/UltraData-Code-L3/py/*.parquet"},
                                       "full_content"),
    "ultradata_code_l3_cpp": Source(EXERCISE, *CODE, {"C++": "data/UltraData-Code-L3/cpp/*.parquet"}, "full_content"),
    "climbmix": Source(TEXT, *CLIMBMIX, {"English": "shard_*.parquet"}, "text"),
}


def raw_dir(source: str) -> Path:
    """CORPUS/raw/<source>: the source's Hub files at their Hub paths (prepare.fetch)."""
    return CORPUS / "raw" / source


def standardized_path(source: str, k: int) -> Path:
    """CORPUS/standardized/<source>/<k>.parquet: Hub file k of the source in sorted order, in its schema
    (prepare.standardize_file)."""
    return CORPUS / "standardized" / source / f"{k}.parquet"
