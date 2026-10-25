"""
The compiled and packed data on disk (compile.py, pack.py), without torch, so the spawned workers that read them
import none: their paths, dtypes, shard bounds, grammars and row counts.
"""

import json
from pathlib import Path

import numpy as np

from corpus import mixture
from corpus.sources import SOURCES
from corpus.tokenizer import Tokenizer

TOKEN_DTYPE = np.uint16  # compiled tokens, as the corpus crate reads them (corpus/crate pack.rs: &[u16])
# a pack index entry of the corpus crate (corpus/crate pack.rs pack_shard): token range [start, end) of the compiled
# split, then the range of the piece's FIM example in the split's FIM tokens (empty for a piece read left to right)
PIECE_COLUMNS = 4


def split_path(corpus: str, split: str, tokenizer: Tokenizer) -> Path:
    """The stem of a compiled split's files (compile.py: .bin tokens, .idx document offsets, .json documents per
    curated file), in <tokenizer.compiled>/<corpus>-<tokenizer data key>/."""
    return tokenizer.compiled / f"{corpus}-{tokenizer.data_key}" / split


def packed_path(corpus: str, split: str, tokenizer: Tokenizer, seq_len: int) -> Path:
    """The stem of a packed split's files (pack.py: .pieces, .rows, .fim), the compiled split's at the row length."""
    return split_path(corpus, split, tokenizer).with_name(f"{split}-T{seq_len}")


def grammar(corpus: str) -> str | None:
    """The tree-sitter grammar of a corpus of repository code (its statement cuts and AST middles), else None."""
    if corpus in mixture.REPOSITORY_SOURCES:
        return SOURCES[corpus].language
    return {name: language for language, name in mixture.REPOSITORIES.items()}.get(corpus)


def fim_packed(corpus: str, split: str) -> bool:
    """Whether a split is packed with fill-in-the-middle examples: the train split of repository code (pack.py)."""
    return grammar(corpus) is not None and split == "train"


def shard_bounds(corpus: str, split: str, tokenizer: Tokenizer) -> np.ndarray:
    """Each curated shard's first document in a compiled split, then the number of documents (compiled <split>.json:
    each shard's documents, in order)."""
    files = json.loads(split_path(corpus, split, tokenizer).with_suffix(".json").read_text())["files"]
    return np.cumsum([0] + [file["documents"] for file in files])


def packed_rows(corpus: str, split: str, tokenizer: Tokenizer, seq_len: int) -> int:
    """The rows of a packed split (its .rows: each row's first piece, then the number of pieces)."""
    rows = packed_path(corpus, split, tokenizer, seq_len).with_suffix(".rows")
    return len(np.memmap(rows, dtype=np.int64, mode="r")) - 1


def stage_rows(tokenizer: Tokenizer, stage: mixture.Stage) -> dict[str, int]:
    """{corpus: its packed train rows} of the stage's corpora."""
    return {corpus: packed_rows(corpus, "train", tokenizer, stage.seq_len) for corpus in stage.sources}
