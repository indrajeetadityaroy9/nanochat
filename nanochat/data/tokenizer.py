"""
BPE tokenizer in the style of GPT-4: merges trained by source-attributed BPE (SA-BPE, F·ln R; arXiv 2604.14053)
within the chunks of SPLIT_PATTERN in the Rust crate nanochat/data/bpe, encoding by OpenAI's tiktoken, whose
regex engine (fancy-regex) the trainer shares.

Trained on every curated source's train split in about equal byte shares (main), writing
TOKENIZER/tokenizer.tiktoken and tokenizer.pattern. Equal byte shares minimize the expected compute per byte
over a uniform prior on mixtures, compute per byte being linear in the byte mixture (our derivation, following
arXiv 2407.13623, Eq. 5).

python -m nanochat.data.tokenizer
"""

import base64
import functools
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType
from urllib.parse import urlsplit

import numpy as np
import pyarrow.parquet as pq
import tiktoken
import torch

from nanochat.config import VOCAB_SIZE
from nanochat.data.schema import REPOSITORY_FILE, TEXT
from nanochat.data.sources import SOURCES, column_bytes, curated_files, texts
from nanochat.native import crate_module
from nanochat.paths import TOKENIZER

BOS = "<|bos|>"  # starts every piece of a packed row (data/loader.py); as a target it ends a document or a FIM middle
# Fill-in-the-middle sentinels (StarCoder's names; arXiv 2207.14255 §3): a FIM example is
# <fim_prefix> prefix <fim_suffix> suffix <fim_middle> middle (PSM) or <fim_prefix> <fim_suffix> suffix <fim_middle>
# prefix middle (SPM), its middle ended by BOS (data/loader.py)
FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE = "<fim_prefix>", "<fim_suffix>", "<fim_middle>"
FIM_TOKENS = (FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE)
SPECIAL_TOKENS = (BOS, *FIM_TOKENS)  # ids after the mergeable tokens, in this order

# The split pattern is GPT-4's (cl100k) with \p{N}{1,2} instead of \p{N}{1,3}, as in karpathy/nanochat (its
# measurement at a 32K vocabulary: groups of 2 digits beat 1 and 3), and with no line break in a run of punctuation:
# a vocabulary with tokens fusing punctuation and a following line break ("):\n") scored MBPP pass@1 0.075 on average
# against 0.226 without (arXiv 2608.18062 §5, 18 models of nanochat's d24 architecture on math and code), and such
# tokens bias the token boundary of prompts that end without one (DeepSeek-V3, arXiv 2412.19437 §4.1, which splits
# them at random in training instead). A run of whitespace up to its last line break is one chunk (trailing spaces and
# blank lines join the break), so a line break joins the whitespace before it, never the indentation after it.
SPLIT_PATTERN = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++|\s*[\r\n]"""
                 r"""|\s+(?!\S)|\s+""")


def bpe() -> ModuleType:
    """The data crate's Python module (train: the tokenizer trainer; sketch, duplicates: data/dedup.py;
    repository_samples: data/curate.py; pack: data/pack.py; native.crate_module)."""
    return crate_module(Path(__file__).parent / "bpe")


class Tokenizer:
    """A vocabulary, the bytes of its mergeable tokens in rank order, then its special tokens (SPECIAL_TOKENS),
    encoded by OpenAI's tiktoken with the split pattern it was trained with."""

    def __init__(self, mergeable: list[bytes], pattern: str) -> None:
        self.pattern = pattern
        self.bos_token_id = len(mergeable)
        self.special_tokens = {name: len(mergeable) + i for i, name in enumerate(SPECIAL_TOKENS)}
        # every id's bytes, a special token's as its text (as tiktoken decodes it)
        self.token_bytes = [*mergeable, *(name.encode() for name in SPECIAL_TOKENS)]
        self.vocab_size = len(self.token_bytes)
        # tiktoken's file format: each token in base64 and its rank
        self.serialized = "".join(f"{base64.b64encode(token).decode()} {rank}\n"
                                  for rank, token in enumerate(mergeable))
        # identity of the documents' tokens, keying compiled and packed data (loader.split_path): SHA-256 of the split
        # pattern, BOS (its id) and the tiktoken file; the special tokens after BOS do not enter it, though packs hold
        # the FIM sentinels' ids
        identity = json.dumps([pattern, {BOS: self.bos_token_id}, self.serialized])
        self.data_key = hashlib.sha256(identity.encode()).hexdigest()
        self.enc = tiktoken.Encoding(self.data_key, pat_str=pattern, special_tokens=self.special_tokens,
                                     mergeable_ranks={token: rank for rank, token in enumerate(mergeable)})

    def get_token_bytes(self, *, device: torch.device | str) -> torch.Tensor:
        """Byte length of every token id, as an int32 tensor of shape (vocab_size,), used for vocab-size-invariant bits
        per byte. Special tokens count 0 bytes."""
        lengths = torch.tensor([len(b) for b in self.token_bytes], dtype=torch.int32)
        lengths[self.bos_token_id:] = 0
        return lengths.to(device)


@functools.cache
def get_tokenizer() -> Tokenizer:
    """The trained tokenizer of TOKENIZER, loaded once per process."""
    with open(TOKENIZER / "tokenizer.tiktoken", "rb") as f:
        mergeable = [base64.b64decode(line.split()[0]) for line in f]  # line i: token i, base64, and its rank i
    with open(TOKENIZER / "tokenizer.pattern", newline="") as f:
        return Tokenizer(mergeable, f.read())


# ---- train the tokenizer on the curated sources

def sample(task: tuple[str, Path, int, int, int]) -> tuple[list[str], list[str]]:
    """Row group g of a curated train file: the texts (sources.texts) and source keys of its rows at the source's
    sample step, `start` the source's rows before the group."""
    source, path, g, start, step = task
    group = pq.ParquetFile(path).read_row_group(g)
    batch = group.take(np.arange(-start % step, group.num_rows, step))
    schema = SOURCES[source].schema
    if schema is REPOSITORY_FILE:
        names = ["repo:" + name for name in batch["repo_name"].to_pylist()]
    elif schema is TEXT:
        names = [f"host:{urlsplit(url).hostname}" if url else "doc:" + id
                 for url, id in zip(batch["url"].to_pylist(), batch["id"].to_pylist(), strict=True)]
    else:
        names = ["doc:" + id for id in batch["id"].to_pylist()]
    return texts(source, batch), names


def main() -> None:
    """TOKENIZER/tokenizer.tiktoken and tokenizer.pattern: config.VOCAB_SIZE - len(SPECIAL_TOKENS) - 256 SA-BPE merges
    within the chunks of SPLIT_PATTERN (merge i makes token 256 + i), trained on about equal byte shares of every
    source's curated train split: a systematic sample, every step-th row in file order as its text (sources.texts),
    step = the split's content bytes // the smallest split's (the smallest split whole; a share is between 1 and 2
    times the smallest split's bytes); row groups are read in parallel, the sample kept in file order. A repository
    sample's files are its rows: they split into the chunks of SPLIT_PATTERN the concatenated sample does, as each
    file's text ends with a line break and the next begins with its path comment. Each row's source key names its
    authorship unit, whose count is SA-BPE's R: a file's repository (one name across the corpora that hold the
    repository), a web page's host, else the document itself (a notebook, a text without a URL, a codeforces_cots
    conversation)."""
    sizes = {source: sum(column_bytes(path, "content") for path in curated_files(source, "train"))
             for source in SOURCES}
    tasks = []
    for source in SOURCES:
        step, at = sizes[source] // min(sizes.values()), 0
        for path in curated_files(source, "train"):
            md = pq.read_metadata(path)
            for g in range(md.num_row_groups):
                tasks.append((source, path, g, at, step))
                at += md.row_group(g).num_rows
    docs, sources, keys = [], [], {}  # keys: source key -> source id
    with ProcessPoolExecutor() as pool:
        for batch, names in pool.map(sample, tasks):
            docs += batch
            sources += [keys.setdefault(name, len(keys)) for name in names]
    mergeable = [bytes([i]) for i in range(256)]
    for left, right in bpe().train(docs, sources, pattern=SPLIT_PATTERN,
                                   n_merges=VOCAB_SIZE - len(SPECIAL_TOKENS) - 256):
        mergeable.append(mergeable[left] + mergeable[right])
    tokenizer = Tokenizer(mergeable, SPLIT_PATTERN)
    for name, content in (("tokenizer.tiktoken", tokenizer.serialized), ("tokenizer.pattern", SPLIT_PATTERN)):
        (TOKENIZER / name).write_text(content, newline="")
    print(f"Tokenizer data key: {tokenizer.data_key}")


if __name__ == "__main__":
    main()
