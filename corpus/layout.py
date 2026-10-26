"""
Paths, dtypes, shard bounds, grammars and row counts of the tokenized and packed data, torch-free for spawned workers.
"""

import json
from pathlib import Path

import numpy as np

from corpus import mixture
from corpus.sources import SOURCES
from corpus.tokenizer import Tokenizer

TOKEN_DTYPE = np.uint16  # pack.rs reads &[u16]
# pack.rs pack_shard's index entry: [start, end) in the tokenized split, then the FIM example's range in .fim
PIECE_COLUMNS = 4


def split_path(corpus: str, split: str, tokenizer: Tokenizer) -> Path:
    """The stem of a tokenized split's files (.bin, .idx, .json), in <tokenizer.tokenized>/<corpus>-<data key>/."""
    return tokenizer.tokenized / f"{corpus}-{tokenizer.data_key}" / split


def packed_path(corpus: str, split: str, tokenizer: Tokenizer, seq_len: int) -> Path:
    """The stem of a packed split's files (.pieces, .rows, .fim)."""
    return split_path(corpus, split, tokenizer).with_name(f"{split}-T{seq_len}")


def grammar(corpus: str) -> str | None:
    """The tree-sitter grammar of a corpus of repository code, else None."""
    if corpus in mixture.REPOSITORY_SOURCES:
        return SOURCES[corpus].language
    return {name: language for language, name in mixture.REPOSITORIES.items()}.get(corpus)


def fim_packed(corpus: str, split: str) -> bool:
    """Whether a split is packed with FIM examples: the train split of repository code."""
    return grammar(corpus) is not None and split == "train"


def shard_bounds(corpus: str, split: str, tokenizer: Tokenizer) -> np.ndarray:
    """Each curated shard's first document in a tokenized split, then the number of documents."""
    files = json.loads(split_path(corpus, split, tokenizer).with_suffix(".json").read_text())["files"]
    return np.cumsum([0] + [file["documents"] for file in files])


def packed_rows(corpus: str, split: str, tokenizer: Tokenizer, seq_len: int) -> int:
    """The rows of a packed split."""
    rows = packed_path(corpus, split, tokenizer, seq_len).with_suffix(".rows")
    return len(np.memmap(rows, dtype=np.int64, mode="r")) - 1


def stage_rows(tokenizer: Tokenizer, stage: mixture.Stage) -> dict[str, int]:
    return {corpus: packed_rows(corpus, "train", tokenizer, stage.seq_len) for corpus in stage.sources}
