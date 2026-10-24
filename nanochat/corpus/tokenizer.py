"""
The corpus's tokenizer (normalization_doc.md §9): production's algorithm, vocabulary size, split pattern and special
tokens (nanochat/data/tokenizer.py: SA-BPE's F·ln R within the chunks of SPLIT_PATTERN, trained by the data crate,
encoded by tiktoken), trained on this corpus's curated train splits at the file stage's shares of bytes: "tokenizers
should be trained on the data mix that they are expected to see during training/inference" (arXiv 2402.01035 §2.3),
and leaving a source out of tokenizer training or weighting sources equally raises their bits per byte
(arXiv 2610.12144). Writes CORPUS/tokenizer/tokenizer.tiktoken and tokenizer.pattern; get_tokenizer loads it, its
compiled and packed data in CORPUS/compiled.

python -m nanochat.corpus.tokenizer
"""

import base64
import functools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from nanochat.config import VOCAB_SIZE
from nanochat.corpus import mixture
from nanochat.corpus.parallel import SPAWN
from nanochat.corpus.sources import SOURCES, curated_shards, text_bytes, texts
from nanochat.data import tokenizer as production
from nanochat.native import data_crate
from nanochat.paths import CORPUS

TOKENIZER = CORPUS / "tokenizer"
# The sample's bytes: production's tokenizer sample's (its curated splits' Parquet footers)
SAMPLE_BYTES = 13.38e9


@functools.cache
def get_tokenizer() -> production.Tokenizer:
    """The corpus's tokenizer, loaded once per process, its data in CORPUS/compiled."""
    with open(TOKENIZER / "tokenizer.tiktoken", "rb") as f:
        mergeable = [base64.b64decode(line.split()[0]) for line in f]  # line i: token i, base64, and its rank i
    with open(TOKENIZER / "tokenizer.pattern", newline="") as f:
        return production.Tokenizer(mergeable, f.read(), name="corpus", compiled=CORPUS / "compiled")


def train_bytes(source: str) -> int:
    """The text bytes of the source's curated train split (Parquet footers)."""
    return sum(text_bytes(pq.read_metadata(path), "content") for path in curated_shards(source, "train"))


def sample_row_group(task: tuple[str, Path, int, int, float]) -> tuple[list[str], list[str]]:
    """Row group g of a curated train shard: the texts (sources.texts) and authorship units of its rows that the
    source's systematic sample takes, `start` the source's rows before the group: row i when floor((i + 1) * fraction)
    exceeds floor(i * fraction). A row naming its repository (a repository file, a function) has that repository as its
    unit, its name lowercased (one unit across the sources that hold it); any other document is its own."""
    source, path, g, start, fraction = task
    group = pq.ParquetFile(path).read_row_group(g)
    rows = np.arange(start, start + group.num_rows)
    batch = group.take(np.flatnonzero(np.floor((rows + 1) * fraction) > np.floor(rows * fraction)))
    if "repo_name" in batch.column_names:
        units = ["repo:" + name.lower() for name in batch["repo_name"].to_pylist()]
    else:
        units = ["doc:" + id for id in batch["id"].to_pylist()]
    return texts(source, batch), units


def sample() -> tuple[list[str], list[int]]:
    """The training sample (texts, unit ids): of each file-stage source's curated train split, its share of the stage's
    bytes as its share of SAMPLE_BYTES; the share of bytes is its share of the stage's tokens (Stage.shares, tokens
    counted at production's bytes per token) at its bytes per token. A systematic sample in shard order, row groups
    read in parallel, the sample kept in order."""
    sources = mixture.FILE_STAGE.sources
    sizes = {s: train_bytes(s) for s in sources}
    shares = mixture.FILE_STAGE.shares({s: sizes[s] / mixture.BYTES_PER_TOKEN[s] for s in sources},
                                       mixture.FILE_STAGE.tokens)
    stage_bytes = sum(shares[s] * mixture.BYTES_PER_TOKEN[s] for s in sources)
    tasks = []
    for s in sources:
        fraction, at = shares[s] * mixture.BYTES_PER_TOKEN[s] / stage_bytes * SAMPLE_BYTES / sizes[s], 0
        print(f"{s}: {fraction:.4f} of its {sizes[s] / 1e9:.2f} GB", flush=True)
        for path in curated_shards(s, "train"):
            md = pq.read_metadata(path)
            for g in range(md.num_row_groups):
                tasks.append((s, path, g, at, fraction))
                at += md.row_group(g).num_rows
    docs, units, ids = [], [], {}
    with ProcessPoolExecutor(mp_context=SPAWN) as pool:
        for batch, names in pool.map(sample_row_group, tasks):
            docs += batch
            units += [ids.setdefault(name, len(ids)) for name in names]
    return docs, units


def main() -> None:
    """Train, save and load the corpus's tokenizer; report each source's bytes per token on its curated val split
    under it and production's (mixture.BYTES_PER_TOKEN), and each group read in part's train tokens under it over its
    draw (the train split's text bytes at the val split's bytes per token): below 1, pack.fit runs out of its
    shards."""
    docs, units = sample()
    print(f"sample: {len(docs):,} documents, {sum(map(len, map(str.encode, docs))) / 1e9:.2f} GB, {max(units) + 1:,} "
          f"authorship units", flush=True)
    mergeable = [bytes([i]) for i in range(256)]
    for left, right in data_crate().train(docs, units, pattern=production.SPLIT_PATTERN,
                                          n_merges=VOCAB_SIZE - len(production.SPECIAL_TOKENS) - 256, score="sa-bpe"):
        mergeable.append(mergeable[left] + mergeable[right])
    TOKENIZER.mkdir(parents=True)
    production.save(TOKENIZER, mergeable, production.SPLIT_PATTERN)
    corpus = get_tokenizer()
    print(f"data key {corpus.data_key}")
    bytes_per_token = {}
    for source in SOURCES:
        val = [text for path in curated_shards(source, "val") for text in texts(source, pq.read_table(path))]
        bytes_per_token[source] = sum(len(text.encode()) for text in val) / sum(
            map(len, corpus.enc.encode_ordinary_batch(val)))
        print(f"{source}: {bytes_per_token[source]:.3f} bytes per token, production's "
              f"{mixture.BYTES_PER_TOKEN[source]:.3f}", flush=True)
    for group in mixture.IN_PART:
        held = sum(train_bytes(s) / bytes_per_token[s] for s, g in mixture.PART.items() if g == group)
        print(f"{group}: train tokens over the draw {held / mixture.draw(group):.4f}", flush=True)


if __name__ == "__main__":
    main()
