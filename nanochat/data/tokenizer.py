"""
BPE tokenizer in the style of GPT-4: merges trained by source-attributed BPE (SA-BPE, F·ln R; arXiv 2604.14053 §4.1,
whose log base is unstated: ln as in its repository's scoring.py; the base changes no argmax) within the chunks of
SPLIT_PATTERN by the data crate (data/crate train.rs), encoding by OpenAI's tiktoken, whose regex engine (fancy-regex)
the trainer shares.

Trained on every curated source's train split in about equal byte shares (main, SAMPLE_SHARES), writing
TOKENIZER/tokenizer.tiktoken and tokenizer.pattern. Our argument for equal byte shares: tokens per byte are linear in
the byte mixture, so their expectation over any exchangeable prior on mixtures equals their value at the equal mixture,
which a tokenizer trained on the equal mixture minimizes on the assumption that a tokenizer trained on a mixture
compresses that mixture best (arXiv 2402.01035 §2.3, qualitatively); compute is proportional to tokens
(arXiv 2407.13623 Eq. 5, its only part in the argument). arXiv 2610.12144 §7 recommends proportional
tokenizer-training weights instead, which the corpus's tokenizer takes (nanochat/corpus/tokenizer.py), and the
tokenizer ablation (ablations/tokenizer.py) now trains its arms on the corpus; get_tokenizer loads production's or a
vocabulary in TOKENIZER_ABLATION/<name>.

python -m nanochat.data.tokenizer
"""

import base64
import functools
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import numpy as np
import pyarrow.parquet as pq
import tiktoken

from nanochat.config import MIXTURE, VOCAB_SIZE
from nanochat.data.schema import REPOSITORY_FILE, TEXT
from nanochat.data.sources import SOURCES, column_bytes, curated_files, texts
from nanochat.native import data_crate
from nanochat.paths import DATA, TOKENIZER, TOKENIZER_ABLATION

BOS = "<|bos|>"  # starts every piece of a packed row (data/loader.py); as a target it ends a document or a FIM middle
# Fill-in-the-middle sentinels (StarCoder's names; arXiv 2207.14255 §3): a FIM example is
# <fim_prefix> prefix <fim_suffix> suffix <fim_middle> middle (PSM) or <fim_prefix> <fim_suffix> suffix <fim_middle>
# prefix middle (SPM), its middle ended by BOS (data/loader.py)
FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE = "<fim_prefix>", "<fim_suffix>", "<fim_middle>"
FIM_TOKENS = (FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE)
SPECIAL_TOKENS = (BOS, *FIM_TOKENS)  # ids after the mergeable tokens, in this order

# The split pattern is GPT-4's (cl100k) with \p{N}{1,2} instead of \p{N}{1,3}, as in karpathy/nanochat (its
# measurement: groups of 2 digits beat 1 and 3), and with no line break in a run of punctuation: a vocabulary with
# tokens fusing punctuation and a following line break ("):\n") scored lower MBPP pass@1 than one without
# (arXiv 2608.18062 Appendix F.3, models of nanochat's d24 architecture on math and code),
# and such tokens bias the token boundary of prompts that end without one (DeepSeek-V3, arXiv 2412.19437 §4.1, which
# splits them at random in training instead). A run of whitespace up to its last line break is one chunk (trailing
# spaces and blank lines join the break), so a line break joins the whitespace before it, never the indentation after
# it.
SPLIT_PATTERN = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++|\s*[\r\n]"""
                 r"""|\s+(?!\S)|\s+""")
# The byte shares of production's tokenizer's sample (sample)
SAMPLE_SHARES: Literal["equal", "mixture"] = "equal"


def tiktoken_file(mergeable: list[bytes]) -> str:
    """tiktoken's file format: each mergeable token in base64 and its rank."""
    return "".join(f"{base64.b64encode(token).decode()} {rank}\n" for rank, token in enumerate(mergeable))


class Tokenizer:
    """A vocabulary, the bytes of its mergeable tokens in rank order, then its special tokens (SPECIAL_TOKENS),
    encoded by OpenAI's tiktoken with the split pattern it was trained with; `name` is get_tokenizer's, `compiled` the
    directory of its compiled and packed data (loader.split_path)."""

    def __init__(self, mergeable: list[bytes], pattern: str, *, name: str, compiled: Path) -> None:
        self.name, self.compiled = name, compiled
        self.pattern = pattern
        self.bos_token_id = len(mergeable)
        self.special_tokens = {token: len(mergeable) + i for i, token in enumerate(SPECIAL_TOKENS)}
        # every id's bytes, a special token's as its text (as tiktoken decodes it), and its byte length in the
        # documents (a special token stands for no text: 0), for bits per byte
        self.token_bytes = [*mergeable, *(token.encode() for token in SPECIAL_TOKENS)]
        self.token_lengths = np.array([len(b) for b in mergeable] + [0] * len(SPECIAL_TOKENS), dtype=np.int64)
        self.vocab_size = len(self.token_bytes)
        # identity of the documents' tokens, keying compiled and packed data (loader.split_path): SHA-256 of the split
        # pattern, BOS (its id) and the tiktoken file; the special tokens after BOS do not enter it, though packs hold
        # the FIM sentinels' ids
        identity = json.dumps([pattern, {BOS: self.bos_token_id}, tiktoken_file(mergeable)])
        self.data_key = hashlib.sha256(identity.encode()).hexdigest()
        self.enc = tiktoken.Encoding(self.data_key, pat_str=pattern, special_tokens=self.special_tokens,
                                     mergeable_ranks={token: rank for rank, token in enumerate(mergeable)})


def save(directory: Path, mergeable: list[bytes], pattern: str) -> None:
    """A vocabulary's tokenizer.tiktoken and tokenizer.pattern, which get_tokenizer loads."""
    (directory / "tokenizer.tiktoken").write_text(tiktoken_file(mergeable), newline="")
    (directory / "tokenizer.pattern").write_text(pattern, newline="")


@functools.cache
def get_tokenizer(name: str = "production") -> Tokenizer:
    """A trained tokenizer, loaded once per process: production's (TOKENIZER, its data in DATA/compiled) or a tokenizer
    ablation arm's (TOKENIZER_ABLATION/<name>, every arm's data in TOKENIZER_ABLATION/compiled)."""
    directory, compiled = ((TOKENIZER, DATA / "compiled") if name == "production"
                           else (TOKENIZER_ABLATION / name, TOKENIZER_ABLATION / "compiled"))
    with open(directory / "tokenizer.tiktoken", "rb") as f:
        mergeable = [base64.b64decode(line.split()[0]) for line in f]  # line i: token i, base64, and its rank i
    with open(directory / "tokenizer.pattern", newline="") as f:
        return Tokenizer(mergeable, f.read(), name=name, compiled=compiled)


# ---- train the tokenizer on the curated sources

def sample_row_group(task: tuple[str, Path, int, int, int]) -> tuple[list[str], list[str]]:
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


def sample(*, shares: Literal["equal", "mixture"]) -> tuple[list[str], list[int]]:
    """The training sample (texts, source ids): of every source's curated train split, about equal byte shares
    ("equal") or config.MIXTURE's group shares of the same bytes, each group's sources at their natural shares
    ("mixture": the training mixture's shares of rows, taken as bytes); a systematic sample, every step-th row in file
    order as its text (sources.texts), step = the split's content bytes // the smallest split's ("equal": the smallest
    split whole; a share is between 1 and 2 times the smallest split's bytes) or one step per group, its content bytes
    over its share of len(SOURCES) smallest splits ("mixture"); row groups are read in parallel, the sample kept in
    file order. A repository sample's files are its rows: they split into the chunks of SPLIT_PATTERN the concatenated
    sample does, as each file's text ends with a line break and the next begins with its path comment. Each row's
    source id names its authorship unit, whose count is SA-BPE's R: a file's repository (one name across the corpora
    that hold the repository), a web page's host (urlsplit().hostname: lower-cased, without its port; a www. host and
    its bare host are distinct), else the document itself (a notebook, a text without a URL, a codeforces_cots
    conversation)."""
    sizes = {source: sum(column_bytes(path, "content") for path in curated_files(source, "train"))
             for source in SOURCES}
    smallest = min(sizes.values())
    if shares == "equal":
        steps = {source: sizes[source] // smallest for source in SOURCES}
    else:
        steps = {}
        for share, corpora in MIXTURE.values():
            steps |= dict.fromkeys(corpora, int(sum(sizes[c] for c in corpora) / (share * len(SOURCES) * smallest)))
    tasks = []
    for source in SOURCES:
        step, at = steps[source], 0
        for path in curated_files(source, "train"):
            md = pq.read_metadata(path)
            for g in range(md.num_row_groups):
                tasks.append((source, path, g, at, step))
                at += md.row_group(g).num_rows
    docs, sources, keys = [], [], {}  # keys: source key -> source id
    with ProcessPoolExecutor() as pool:
        for batch, names in pool.map(sample_row_group, tasks):
            docs += batch
            sources += [keys.setdefault(name, len(keys)) for name in names]
    return docs, sources


def merges(docs: list[str], sources: list[int], *, pattern: str, score: Literal["sa-bpe", "frequency"]
           ) -> list[bytes]:
    """The mergeable tokens, bytes in rank order: the 256 bytes, then config.VOCAB_SIZE - len(SPECIAL_TOKENS) - 256
    merges trained on the sample within the chunks of `pattern` (merge i makes token 256 + i), by SA-BPE's F·ln R or
    by frequency alone (the crate's train)."""
    mergeable = [bytes([i]) for i in range(256)]
    for left, right in data_crate().train(docs, sources, pattern=pattern,
                                          n_merges=VOCAB_SIZE - len(SPECIAL_TOKENS) - 256, score=score):
        mergeable.append(mergeable[left] + mergeable[right])
    return mergeable


def main() -> None:
    """TOKENIZER/tokenizer.tiktoken and tokenizer.pattern: SA-BPE merges within the chunks of SPLIT_PATTERN, trained on
    the sample at SAMPLE_SHARES."""
    docs, sources = sample(shares=SAMPLE_SHARES)
    save(TOKENIZER, merges(docs, sources, pattern=SPLIT_PATTERN, score="sa-bpe"), SPLIT_PATTERN)
    print(f"Tokenizer data key: {get_tokenizer().data_key}")


if __name__ == "__main__":
    main()
