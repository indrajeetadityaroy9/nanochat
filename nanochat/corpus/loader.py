"""
The corpus's compiled and packed data on disk (compile.py, pack.py), and their reading by training.
"""

from pathlib import Path

import numpy as np

from nanochat.data.tokenizer import Tokenizer

TOKEN_DTYPE = np.uint16  # compiled tokens, as the data crate reads them (data/crate pack.rs: &[u16])
# a pack index entry of the data crate (data/crate pack.rs pack_shard): token range [start, end) of the compiled split,
# then the range of the piece's FIM example in the split's FIM tokens (empty for a piece read left to right)
PIECE_COLUMNS = 4


def split_path(corpus: str, split: str, tokenizer: Tokenizer) -> Path:
    """The stem of a compiled split's files (compile.py: .bin tokens, .idx document offsets, .json documents per
    curated file), in <tokenizer.compiled>/<corpus>-<tokenizer data key>/."""
    return tokenizer.compiled / f"{corpus}-{tokenizer.data_key}" / split


def packed_path(corpus: str, split: str, tokenizer: Tokenizer, seq_len: int) -> Path:
    """The stem of a packed split's files (pack.py: .pieces, .rows, .fim), the compiled split's at the row length."""
    return split_path(corpus, split, tokenizer).with_name(f"{split}-T{seq_len}")
