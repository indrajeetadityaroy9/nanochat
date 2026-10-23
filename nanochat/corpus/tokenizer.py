"""
The corpus's tokenizer (the design document, normalization_doc.md §9): production's algorithm, vocabulary size, split
pattern and special tokens (nanochat/data/tokenizer.py: SA-BPE's F·ln R within the chunks of SPLIT_PATTERN, trained by
the data crate, encoded by tiktoken), trained on this corpus's curated train splits at the file stage's shares
(mixture.FILE_STAGE): "tokenizers should be trained on the data mix that they are expected to see during
training/inference" (arXiv 2402.01035 §2.3), and leaving a source out of tokenizer training or weighting sources
equally raises their bits per byte (arXiv 2610.12144). Writes CORPUS/tokenizer/tokenizer.tiktoken and tokenizer.pattern;
get_tokenizer loads it, its compiled and packed data in CORPUS/compiled.

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
from nanochat.corpus.schema import REPOSITORY_FILE
from nanochat.corpus.sources import SOURCES, curated_dir, texts
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


def train_files(source: str) -> list[Path]:
    """The source's curated train shards in the order of their number k."""
    return sorted(curated_dir(source, "train").iterdir(), key=lambda path: int(path.stem))


def content_bytes(path: Path) -> int:
    """The uncompressed bytes of a curated file's content column, from its footer."""
    md = pq.read_metadata(path)
    column = md.schema.to_arrow_schema().get_field_index("content")
    return sum(md.row_group(g).column(column).total_uncompressed_size for g in range(md.num_row_groups))


def sample_row_group(task: tuple[str, Path, int, int, int]) -> tuple[list[str], list[str]]:
    """Row group g of a curated train shard: the texts (sources.texts) and authorship units of its rows at the source's
    sample step, `start` the source's rows before the group. A repository file's unit is its repository (one name
    across the sources that hold it); any other document's is itself."""
    source, path, g, start, step = task
    group = pq.ParquetFile(path).read_row_group(g)
    batch = group.take(np.arange(-start % step, group.num_rows, step))
    if SOURCES[source].schema is REPOSITORY_FILE:
        units = ["repo:" + name for name in batch["repo_name"].to_pylist()]
    else:
        units = ["doc:" + id for id in batch["id"].to_pylist()]
    return texts(source, batch), units


def sample() -> tuple[list[str], list[int]]:
    """The training sample (texts, unit ids): of each file-stage source's curated train split, its share of the stage's
    tokens (mixture.source_shares, tokens counted at production's bytes per token) taken as its share of SAMPLE_BYTES; a
    systematic sample, every step-th row in shard order (step = the split's content bytes over its part of the sample),
    row groups read in parallel, the sample kept in order."""
    sources = [s for _, members in mixture.FILE_STAGE.values() for s in members]
    sizes = {s: sum(content_bytes(path) for path in train_files(s)) for s in sources}
    shares = mixture.source_shares(mixture.FILE_STAGE, mixture.FILE_TOKENS,
                                   {s: sizes[s] / mixture.BYTES_PER_TOKEN[s] for s in sources})
    tasks = []
    for s in sources:
        step, at = max(1, int(sizes[s] / (shares[s] * SAMPLE_BYTES))), 0
        for path in train_files(s):
            md = pq.read_metadata(path)
            for g in range(md.num_row_groups):
                tasks.append((s, path, g, at, step))
                at += md.row_group(g).num_rows
    docs, units, ids = [], [], {}
    with ProcessPoolExecutor() as pool:
        for batch, names in pool.map(sample_row_group, tasks):
            docs += batch
            units += [ids.setdefault(name, len(ids)) for name in names]
    return docs, units


def main() -> None:
    """Train, save and load the corpus's tokenizer; report each source's bytes per token on its curated val split
    under it and under production's."""
    docs, units = sample()
    print(f"sample: {len(docs):,} documents, {sum(map(len, map(str.encode, docs))) / 1e9:.2f} GB, {max(units) + 1:,} "
          f"authorship units", flush=True)
    mergeable = [bytes([i]) for i in range(256)]
    for left, right in data_crate().train(docs, units, pattern=production.SPLIT_PATTERN,
                                          n_merges=VOCAB_SIZE - len(production.SPECIAL_TOKENS) - 256, score="sa-bpe"):
        mergeable.append(mergeable[left] + mergeable[right])
    TOKENIZER.mkdir(parents=True)
    production.save(TOKENIZER, mergeable, production.SPLIT_PATTERN)
    corpus, reference = get_tokenizer(), production.get_tokenizer()
    print(f"data key {corpus.data_key}")
    for source in mixture.READ:
        val = [text for path in sorted(curated_dir(source, "val").iterdir())
               for text in texts(source, pq.read_table(path))]
        size = sum(len(text.encode()) for text in val)
        new, old = (size / sum(map(len, t.enc.encode_ordinary_batch(val))) for t in (corpus, reference))
        print(f"{source}: {new:.3f} bytes per token, production's {old:.3f}", flush=True)


if __name__ == "__main__":
    main()
