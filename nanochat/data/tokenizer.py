"""
BPE tokenizer in the style of GPT-4: merges trained by source-attributed BPE (SA-BPE, F·ln R; Chizhov et al. 2026, arXiv
2604.14053) within the chunks of SPLIT_PATTERN in the Rust crate nanochat/data/bpe, encoding by OpenAI's tiktoken, whose
regex engine (fancy-regex) the trainer shares; and the prefix-token targets of the vocabulary (prefix_targets).

Trained on every curated source's train split in about equal byte shares (main), writing
<base_dir>/tokenizer/tokenizer.tiktoken and tokenizer.pattern. Equal byte shares minimize the expected compute per byte
over a uniform prior on mixtures, compute per byte being linear in the byte mixture (our derivation, following Tao et
al. 2024, arXiv 2407.13623, Eq. 5).

python -m nanochat.data.tokenizer
"""

import base64
import functools
import glob
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from urllib.parse import urlsplit

import numpy as np
import pyarrow.parquet as pq
import tiktoken
import torch

from nanochat.config import VOCAB_SIZE
from nanochat.data.schema import REPOSITORY_FILE, TEXT
from nanochat.data.sources import SOURCES, column_bytes, curated_files, texts
from nanochat.native import native_library
from nanochat.paths import base_dir

BOS = "<|bos|>"  # starts every piece of a packed row (data/loader.py); as a target it ends a document or a FIM middle
# Fill-in-the-middle sentinels (StarCoder's names; Bavarian et al. arXiv 2207.14255 §3): a FIM example is
# <fim_prefix> prefix <fim_suffix> suffix <fim_middle> middle (PSM) or <fim_prefix> <fim_suffix> suffix <fim_middle>
# prefix middle (SPM), its middle ended by BOS (data/loader.py)
FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE = "<fim_prefix>", "<fim_suffix>", "<fim_middle>"
FIM_TOKENS = (FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE)
SPECIAL_TOKENS = (BOS, *FIM_TOKENS)  # ids after the mergeable tokens, in this order

# The split pattern is GPT-4's (cl100k) with \p{N}{1,2} instead of \p{N}{1,3}, as in karpathy/nanochat (its
# measurement at a 32K vocabulary: groups of 2 digits beat 1 and 3). A run of punctuation keeps the line breaks that
# follow it ("):\n" is one chunk), and a run of whitespace up to its last line break is one chunk (trailing spaces and
# blank lines join the break), so a chunk may span lines.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


@functools.cache
def bpe():
    """The crate's Python module (train: the tokenizer trainer; sketch, duplicates: data/dedup.py; repository_samples:
    data/prepare.py curate; pack: data/pack.py), built by cargo once per version of its sources (native.native_library)."""
    crate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe")
    manifest = os.path.join(crate, "Cargo.toml")
    target = os.path.join(base_dir(), "cache", "native", "bpe-target")

    def build(path):
        subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path", manifest, "--target-dir", target])
        shutil.copyfile(os.path.join(target, "release", "libnanochat_bpe.so"), path)

    sources = sorted(glob.glob(os.path.join(crate, "*.rs"))) + [os.path.join(crate, name) for name in ("Cargo.toml", "Cargo.lock")]
    path = native_library("bpe", sources, build)
    spec = importlib.util.spec_from_file_location("nanochat_bpe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tokenizer:
    """A vocabulary, the bytes of its mergeable tokens in rank order, then its special tokens (SPECIAL_TOKENS), encoded by
    OpenAI's tiktoken with the split pattern it was trained with."""

    def __init__(self, mergeable, pattern):
        self.pattern = pattern
        self.bos_token_id = len(mergeable)
        self.special_tokens = {name: len(mergeable) + i for i, name in enumerate(SPECIAL_TOKENS)}
        self.token_bytes = [*mergeable, *(name.encode() for name in SPECIAL_TOKENS)]  # every id's bytes, a special as its text (as tiktoken decodes it)
        self.vocab_size = len(self.token_bytes)
        self.serialized = "".join(f"{base64.b64encode(token).decode()} {rank}\n" for rank, token in enumerate(mergeable))  # tiktoken's file format
        # data_key: identity of the documents' tokens, keying compiled and packed data (loader.split_path): SHA-256 of
        # the split pattern, BOS (its id) and the tiktoken file; the special tokens after BOS do not enter it, though
        # packs hold the FIM sentinels' ids. fingerprint: identity of the model's vocabulary, recorded in checkpoints:
        # SHA-256 of the split pattern, every special token and the tiktoken file.
        self.data_key = hashlib.sha256(json.dumps([pattern, {BOS: self.bos_token_id}, self.serialized]).encode()).hexdigest()
        self.fingerprint = hashlib.sha256(json.dumps([pattern, self.special_tokens, self.serialized]).encode()).hexdigest()
        self.enc = tiktoken.Encoding(self.fingerprint, pat_str=pattern, special_tokens=self.special_tokens,
                                     mergeable_ranks={token: rank for rank, token in enumerate(mergeable)})

    @classmethod
    def train(cls, texts, sources, *, vocab_size):
        """The tokenizer of vocab_size - len(SPECIAL_TOKENS) - 256 SA-BPE merges within the chunks of SPLIT_PATTERN on
        documents texts, document d from source id sources[d]; the special tokens take the ids after the merges."""
        mergeable = [bytes([i]) for i in range(256)]
        for left, right in bpe().train(texts, sources, pattern=SPLIT_PATTERN, n_merges=vocab_size - len(SPECIAL_TOKENS) - 256):
            mergeable.append(mergeable[left] + mergeable[right])
        return cls(mergeable, SPLIT_PATTERN)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        with open(os.path.join(tokenizer_dir, "tokenizer.tiktoken"), "rb") as f:
            mergeable = [base64.b64decode(line.split()[0]) for line in f]  # line i: token i, base64, and its rank i
        with open(os.path.join(tokenizer_dir, "tokenizer.pattern"), newline="") as f:
            return cls(mergeable, f.read())

    def get_token_bytes(self, *, device):
        """Byte length of every token id, as an int32 tensor of shape (vocab_size,), used for vocab-size-invariant bits per
        byte. Special tokens count 0 bytes."""
        lengths = torch.tensor([len(b) for b in self.token_bytes], dtype=torch.int32)
        lengths[self.bos_token_id:] = 0
        return lengths.to(device)

    def save(self, tokenizer_dir):
        """tokenizer.tiktoken: the mergeable tokens in tiktoken's format; tokenizer.pattern: the split pattern."""
        os.makedirs(tokenizer_dir, exist_ok=True)
        for name, content in (("tokenizer.tiktoken", self.serialized), ("tokenizer.pattern", self.pattern)):
            with open(os.path.join(tokenizer_dir, name), "w", newline="") as f:
                f.write(content)


@functools.cache
def get_tokenizer():
    """The trained tokenizer of <base_dir>/tokenizer, loaded once per process."""
    return Tokenizer.from_directory(os.path.join(base_dir(), "tokenizer"))


def prefix_targets(tokenizer, *, rows):
    """Prefix-token targets (modded-nanogpt's prefix-token loss; ops/softcap_ce.py's weighted target columns): (rows,)
    int64, rows >= the vocabulary (the lm head's padded size); for each mergeable token, the id of the token whose
    bytes are the longest proper prefix of its bytes. -1 (ignored) for single bytes, for tokens whose bytes are all
    whitespace (bytes.isspace: on code an indentation or line structure, whose prefix is a one-shorter indentation), for
    the special tokens and for the padding ids; the last entry is one of these, so an ignored target (-1) indexes it and
    stays -1."""
    mergeable = tokenizer.token_bytes[:tokenizer.bos_token_id]
    ids = {b: i for i, b in enumerate(mergeable)}
    table = torch.full((rows,), -1, dtype=torch.long)
    table[:len(mergeable)] = torch.tensor([-1 if b.isspace() else next((ids[b[:n]] for n in range(len(b) - 1, 0, -1) if b[:n] in ids), -1)
                                           for b in mergeable])
    return table


# ---- training sample

def keyed_documents(source, batch):
    """(text, source key) of every row of a curated batch (sources.texts). A repository sample's files are its rows: they
    split into the chunks of SPLIT_PATTERN the concatenated sample does, as each file's text ends with a line break and
    the next begins with its path comment. The key names the row's authorship unit, whose count is SA-BPE's R: a file's
    repository (one name across the corpora that hold the repository), a web page's host, else the document itself (a
    notebook, a text without a URL, a codeforces_cots conversation)."""
    schema = SOURCES[source].schema
    if schema is REPOSITORY_FILE:
        keys = ["repo:" + name for name in batch["repo_name"].to_pylist()]
    elif schema is TEXT:
        keys = [f"host:{urlsplit(url).hostname}" if url else "doc:" + id for url, id in zip(batch["url"].to_pylist(), batch["id"].to_pylist())]
    else:
        keys = ["doc:" + id for id in batch["id"].to_pylist()]
    return list(zip(texts(source, batch), keys))


def sample(source, *, step):
    """(text, source key) of a systematic sample of a source's curated train split: every step-th row, in file order."""
    picks, at = [], 0
    for path in curated_files(source, "train"):
        shard = pq.ParquetFile(path)
        for g in range(shard.num_row_groups):
            batch = shard.read_row_group(g)
            picks += keyed_documents(source, batch.take(np.arange(-at % step, batch.num_rows, step)))
            at += batch.num_rows
    return picks


# ---- train the tokenizer on the curated sources

def main():
    """The tokenizer of config.VOCAB_SIZE trained on about equal byte shares of every source's curated train split:
    every step-th row, step = the split's content bytes // the smallest split's (the smallest split whole; a share is
    between 1 and 2 times the smallest split's bytes)."""
    sizes = {source: sum(column_bytes(path, "content") for path in curated_files(source, "train")) for source in SOURCES}
    picks = [pick for source in SOURCES for pick in sample(source, step=sizes[source] // min(sizes.values()))]
    keys = {}  # source key -> source id
    sources = [keys.setdefault(key, len(keys)) for _, key in picks]
    tokenizer = Tokenizer.train([doc for doc, _ in picks], sources, vocab_size=VOCAB_SIZE)
    tokenizer.save(os.path.join(base_dir(), "tokenizer"))
    print(f"Tokenizer fingerprint: {tokenizer.fingerprint}, data key: {tokenizer.data_key}")


if __name__ == "__main__":
    main()
