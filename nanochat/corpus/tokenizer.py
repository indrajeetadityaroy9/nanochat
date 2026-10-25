"""
The corpus's tokenizer in CORPUS/tokenizer (tokenizer.tiktoken, tokenizer.pattern), adopted by ablations/tokenizer.py:
the mergeable tokens, then SPECIAL_TOKENS, encoded by tiktoken. A vocabulary's compiled and packed data live in its
directory's sibling compiled/, keyed by its data key, so compile.py and pack.py serve every arm alike.
"""

import base64
import functools
import hashlib
import json
from pathlib import Path

import numpy as np
import tiktoken

from nanochat.paths import CORPUS

TOKENIZER = CORPUS / "tokenizer"
# The tokenizer ablation's arm vocabularies, ABLATION/<arm>, and their data, ABLATION/compiled
ABLATION = CORPUS / "tokenizer-ablation"

BOS = "<|bos|>"  # starts every piece of a packed row (loader.py); as a target it ends a document or a FIM middle
# Fill-in-the-middle sentinels (StarCoder's names; arXiv 2207.14255 §3): a FIM example is
# <fim_prefix> prefix <fim_suffix> suffix <fim_middle> middle (PSM) or <fim_prefix> <fim_suffix> suffix <fim_middle>
# prefix middle (SPM), its middle ended by BOS (loader.py)
FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE = "<fim_prefix>", "<fim_suffix>", "<fim_middle>"
FIM_TOKENS = (FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE)
SPECIAL_TOKENS = (BOS, *FIM_TOKENS)  # ids after the mergeable tokens, in this order

# The reference split pattern (the sa-bpe arm): cl100k's with digit runs of one or two (karpathy/nanochat's) and no
# line break after a punctuation run, so no token joins "):" to its line break (docs/tokenizer-ablation.md)
SPLIT_PATTERN = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++|\s*[\r\n]"""
                 r"""|\s+(?!\S)|\s+""")


def tiktoken_file(mergeable: list[bytes]) -> str:
    """tiktoken's file format: each mergeable token in base64 and its rank."""
    return "".join(f"{base64.b64encode(token).decode()} {rank}\n" for rank, token in enumerate(mergeable))


class Tokenizer:
    """A vocabulary: its mergeable tokens' bytes in rank order, then SPECIAL_TOKENS, encoded by tiktoken with its split
    pattern; `compiled` the directory of its compiled and packed data."""

    def __init__(self, mergeable: list[bytes], pattern: str, *, compiled: Path) -> None:
        self.compiled, self.pattern = compiled, pattern
        self.bos_token_id = len(mergeable)
        self.special_tokens = {token: len(mergeable) + i for i, token in enumerate(SPECIAL_TOKENS)}
        # every id's bytes (a special token's as its text) and its length in document bytes (0 for a special token)
        self.token_bytes = [*mergeable, *(token.encode() for token in SPECIAL_TOKENS)]
        self.token_lengths = np.array([len(b) for b in mergeable] + [0] * len(SPECIAL_TOKENS), dtype=np.int64)
        self.vocab_size = len(self.token_bytes)
        # the documents' tokens' identity, keying compiled and packed data: SHA-256 of the pattern, BOS's id and the
        # tiktoken file (the FIM sentinels after BOS do not enter it)
        identity = json.dumps([pattern, {BOS: self.bos_token_id}, tiktoken_file(mergeable)])
        self.data_key = hashlib.sha256(identity.encode()).hexdigest()
        self.enc = tiktoken.Encoding(self.data_key, pat_str=pattern, special_tokens=self.special_tokens,
                                     mergeable_ranks={token: rank for rank, token in enumerate(mergeable)})


def save(directory: Path, mergeable: list[bytes], pattern: str) -> None:
    """A vocabulary's tokenizer.tiktoken and tokenizer.pattern, which get_tokenizer loads."""
    (directory / "tokenizer.tiktoken").write_text(tiktoken_file(mergeable), newline="")
    (directory / "tokenizer.pattern").write_text(pattern, newline="")


@functools.cache
def get_tokenizer(directory: Path = TOKENIZER) -> Tokenizer:
    """The vocabulary in `directory`, its compiled and packed data in directory.parent / "compiled"."""
    with open(directory / "tokenizer.tiktoken", "rb") as f:
        mergeable = [base64.b64decode(line.split()[0]) for line in f]  # line i: token i, base64, and its rank i
    with open(directory / "tokenizer.pattern", newline="") as f:
        return Tokenizer(mergeable, f.read(), compiled=directory.parent / "compiled")
