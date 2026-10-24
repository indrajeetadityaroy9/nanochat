"""
The corpus's tokenizer: the vocabulary the tokenizer ablation decides and adopts (ablations/tokenizer.py adopt) into
CORPUS/tokenizer (tokenizer.tiktoken, tokenizer.pattern), with production's vocabulary size and special tokens
(nanochat/data/tokenizer.py Tokenizer), encoded by tiktoken. get_tokenizer loads it, or any vocabulary directory (an
ablation arm's), once per process; a vocabulary's compiled and packed data are in its directory's sibling compiled/,
keyed by its data key (loader.split_path), so compile.py and pack.py serve the corpus's tokenizer and every arm alike.
"""

import base64
import functools
from pathlib import Path

from nanochat.data.tokenizer import Tokenizer
from nanochat.paths import CORPUS

TOKENIZER = CORPUS / "tokenizer"


@functools.cache
def get_tokenizer(directory: Path = TOKENIZER) -> Tokenizer:
    """The vocabulary in `directory`, its compiled and packed data in directory.parent / "compiled"."""
    with open(directory / "tokenizer.tiktoken", "rb") as f:
        mergeable = [base64.b64decode(line.split()[0]) for line in f]  # line i: token i, base64, and its rank i
    with open(directory / "tokenizer.pattern", newline="") as f:
        return Tokenizer(mergeable, f.read(), name=directory.name, compiled=directory.parent / "compiled")
