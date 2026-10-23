"""
The corpus's sources: a pinned Hub dataset each, the files it reads, its document category (schema.py) and the column
of a fetched file that holds a document's text.

files maps a language to the glob of the Hub files holding it (Linguist names for code; the language of a text
corpus). Stack-Edu (arXiv 2502.02737) holds The Stack v2's educational files by Software Heritage blob ID, their text
fetched from Software Heritage (prepare.fetch). UltraData-Stack-v3-L2 holds The Stack v3's files that UltraData's L2
classifiers select. UltraData-Code (arXiv 2610.23960) releases L2, selected source files, and L3, exercises written
from L2's algorithmic files; both per language. UltraData-Math (arXiv 2602.09003) releases L1, Common Crawl's math
pages cleaned and deduplicated, one folder per crawl snapshot, and L3, documents synthesized in four formats, a source
each.
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


EDU = ("HuggingFaceTB/stack-edu", "eeec5caac5cc3758a18f1d3ba4416837a9ba814c")
STACK_V3 = ("openbmb/UltraData-Stack-v3-L2", "d6bdb3e707cf17bbfb64067cd4c4ed84b0809baf")
CODE = ("openbmb/UltraData-Code", "525c9ac8639c5032633392d096e54db8c84a005e")
MATH = ("openbmb/UltraData-Math", "fe10db8efd35597fd7fcff8ff576b5ec4ea5ff87")
CLIMBMIX = ("karpathy/climbmix-400b-shuffle", "915333b4f8b8684f39aeaafea600fea6f43fb703")

SOURCES = {
    "stack_edu_python": Source(REPOSITORY_FILE, *EDU, {"Python": "Python/*.parquet"}, "text"),
    "stack_edu_cpp": Source(REPOSITORY_FILE, *EDU, {"C++": "Cpp/*.parquet"}, "text"),
    "ultradata_stack_v3_l2_python": Source(REPOSITORY_FILE, *STACK_V3,
                                           {"Python": "data/UltraData-Stack-v3-L2/python/*.parquet"}, "content"),
    "ultradata_stack_v3_l2_cpp": Source(REPOSITORY_FILE, *STACK_V3,
                                        {"C++": "data/UltraData-Stack-v3-L2/cpp/*.parquet"}, "content"),
    "ultradata_code_l2_python": Source(REPOSITORY_FILE, *CODE, {"Python": "data/UltraData-Code-L2/py/*.parquet"},
                                       "content"),
    "ultradata_code_l2_cpp": Source(REPOSITORY_FILE, *CODE, {"C++": "data/UltraData-Code-L2/cpp/*.parquet"}, "content"),
    "ultradata_code_l3_python": Source(EXERCISE, *CODE, {"Python": "data/UltraData-Code-L3/py/*.parquet"},
                                       "full_content"),
    "ultradata_code_l3_cpp": Source(EXERCISE, *CODE, {"C++": "data/UltraData-Code-L3/cpp/*.parquet"}, "full_content"),
    "ultradata_math_l1": Source(TEXT, *MATH, {"English": "data/UltraData-Math-L1/*/*.parquet"}, "content"),
    "ultradata_math_l3_qa": Source(TEXT, *MATH, {"English": "data/UltraData-Math-L3/QA-Synthetic/*.parquet"},
                                   "content"),
    "ultradata_math_l3_conversation": Source(
        TEXT, *MATH, {"English": "data/UltraData-Math-L3/Conversation-Synthetic/*.parquet"}, "content"),
    "ultradata_math_l3_multi_style": Source(
        TEXT, *MATH, {"English": "data/UltraData-Math-L3/Multi-Style-Synthetic/*.parquet"}, "content"),
    "ultradata_math_l3_textbook_exercise": Source(
        TEXT, *MATH, {"English": "data/UltraData-Math-L3/Textbook-Exercise-Synthetic/*.parquet"}, "content"),
    "climbmix": Source(TEXT, *CLIMBMIX, {"English": "shard_*.parquet"}, "text"),
}


def raw_dir(source: str) -> Path:
    """CORPUS/raw/<source>: the source's Hub files at their Hub paths (prepare.fetch)."""
    return CORPUS / "raw" / source


def standardized_path(source: str, k: int) -> Path:
    """CORPUS/standardized/<source>/<k>.parquet: Hub file k of the source in sorted order, in its schema
    (prepare.standardize_file)."""
    return CORPUS / "standardized" / source / f"{k}.parquet"
