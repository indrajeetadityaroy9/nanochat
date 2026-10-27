"""
The corpus's tokenizer (CORPUS/tokenizer: tokenizer.tiktoken, tokenizer.pattern), encoded by tiktoken; the tokenized
and packed data are keyed by its data key.
"""

import base64
import functools
import hashlib
import json

import numpy as np
import tiktoken

from paths import CORPUS

TOKENIZER = CORPUS / "tokenizer"

BOS = "<|bos|>"
# FIM sentinels named as in deepseek-ai/deepseek-coder-6.7b-base tokenizer.json (arXiv 2401.14196 §3.1.2)
FIM_BEGIN, FIM_HOLE, FIM_END = ("<\uff5cfim\u2581begin\uff5c>", "<\uff5cfim\u2581hole\uff5c>",
                                "<\uff5cfim\u2581end\uff5c>")
FIM_TOKENS = (FIM_BEGIN, FIM_HOLE, FIM_END)
SPECIAL_TOKENS = (BOS, *FIM_TOKENS)  # ids after the mergeable tokens, in this order

# cl100k's with digit runs of one or two (karpathy/nanochat) and no line break after a punctuation run
SPLIT_PATTERN = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++|\s*[\r\n]"""
                 r"""|\s+(?!\S)|\s+""")


def tiktoken_file(mergeable: list[bytes]) -> str:
    """tiktoken's file format: each mergeable token in base64 and its rank."""
    return "".join(f"{base64.b64encode(token).decode()} {rank}\n" for rank, token in enumerate(mergeable))


class Tokenizer:
    """A vocabulary: its mergeable tokens' bytes in rank order, then SPECIAL_TOKENS, encoded by tiktoken."""

    def __init__(self, mergeable: list[bytes], pattern: str) -> None:
        self.pattern, self.mergeable = pattern, mergeable
        self.bos_token_id = len(mergeable)
        self.special_tokens = {token: len(mergeable) + i for i, token in enumerate(SPECIAL_TOKENS)}
        # a special token's bytes are its text; its document length 0
        self.token_bytes = [*mergeable, *(token.encode() for token in SPECIAL_TOKENS)]
        self.token_lengths = np.array([len(b) for b in mergeable] + [0] * len(SPECIAL_TOKENS), dtype=np.int64)
        self.vocab_size = len(self.token_bytes)
        # SHA-256 of [pattern, {BOS: id}, tiktoken file]: the FIM sentinels after BOS do not enter it
        identity = json.dumps([pattern, {BOS: self.bos_token_id}, tiktoken_file(mergeable)])
        self.data_key = hashlib.sha256(identity.encode()).hexdigest()
        self.enc = tiktoken.Encoding(self.data_key, pat_str=pattern, special_tokens=self.special_tokens,
                                     mergeable_ranks={token: rank for rank, token in enumerate(mergeable)})


@functools.cache
def get_tokenizer() -> Tokenizer:
    """The corpus's tokenizer, from TOKENIZER."""
    with open(TOKENIZER / "tokenizer.tiktoken", "rb") as f:
        mergeable = [base64.b64decode(line.split()[0]) for line in f]
    with open(TOKENIZER / "tokenizer.pattern", newline="") as f:
        return Tokenizer(mergeable, f.read())
