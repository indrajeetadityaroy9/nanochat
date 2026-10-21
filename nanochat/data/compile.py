"""
Compile the prepared sources (prepare.py) into tokens: per source and split, in compiled/<source>-<tokenizer data key>/,
the tokens of every curated document in file order, without padding or BOS (<split>.bin, loader.TOKEN_DTYPE), the
offset of each document's first token followed by the total (<split>.idx, int64), and the number of documents of each
curated file (<split>.json; pack.py packs each file's documents together). The data does not depend on the sequence
length: pack.py groups the documents into training rows of one.

A document is a repository sample (its files' texts, sources.texts: each after a comment with its path, in dependency
order; curate.py) or the content of a notebook or text (a codeforces_cots conversation included). Row groups are
the unit of parallel work; the parent writes their tokens in task order, so the output does not depend on the number
of workers.

python -m nanochat.data.compile
"""

import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from nanochat.data.loader import TOKEN_DTYPE, split_path
from nanochat.data.schema import REPOSITORY_FILE
from nanochat.data.sources import SOURCES, curated_files, texts
from nanochat.data.tokenizer import Tokenizer, get_tokenizer


def compile_row_group(task: tuple[str, Path, int]) -> tuple[np.ndarray, np.ndarray]:
    """(tokens, document lengths) of one row group; task = (source, curated shard, row group index). A repository
    sample's files are consecutive rows of one sample, which a row group holds whole, concatenated in the sample's
    order. The tokenizer comes from get_tokenizer's per-process cache, not the task: it is too large to send per
    task."""
    source, path, row_group = task
    table = pq.ParquetFile(path).read_row_group(row_group)
    documents = texts(source, table)
    if SOURCES[source].schema is REPOSITORY_FILE:
        starts = np.flatnonzero(np.diff(table["sample"].to_numpy(), prepend=-1)).tolist()
        documents = ["".join(documents[a:b]) for a, b in itertools.pairwise([*starts, len(documents)])]
    # a special token's text encodes as ordinary text (compiled data hold no special id); uint32, no Python int each.
    # A row group may hold no document: curate.split_file writes each normalized row group's surviving rows.
    ids = [get_tokenizer().enc.encode_to_numpy(text, disallowed_special=()) for text in documents]
    lengths = np.array([len(doc) for doc in ids], dtype=np.int64)
    return np.concatenate([np.zeros(0, TOKEN_DTYPE), *ids]).astype(TOKEN_DTYPE), lengths


def compile_split(source: str, split: str, tokenizer: Tokenizer) -> None:
    """Compile a source's curated shards of a split into <split>.bin and <split>.idx, with <split>.json beside them."""
    files = curated_files(source, split)
    path = split_path(source, split, tokenizer)
    row_groups = {file: pq.read_metadata(file).num_row_groups for file in files}
    tasks = [(source, file, i) for file, count in row_groups.items() for i in range(count)]
    print(f"Compiling {len(files)} {split} files ({len(tasks):,} row groups) of '{source}' into {path}.bin")
    per_file, offset = [], 0
    with (ProcessPoolExecutor() as pool, open(path.with_suffix(".bin"), "wb") as bin_file,
          open(path.with_suffix(".idx"), "wb") as idx_file):
        idx_file.write(np.zeros(1, np.int64))
        results = pool.map(compile_row_group, tasks)  # in task order, whichever worker finishes first
        for count in row_groups.values():
            documents_in_file = 0
            for tokens, lengths in itertools.islice(results, count):
                bin_file.write(tokens)
                idx_file.write(offset + np.cumsum(lengths))
                offset += len(tokens)
                documents_in_file += len(lengths)
            per_file.append({"documents": documents_in_file})
    path.with_suffix(".json").write_text(json.dumps({"files": per_file}))


if __name__ == "__main__":
    tokenizer = get_tokenizer()
    for source in SOURCES:
        # a new output directory per source and tokenizer: compile does not resume
        split_path(source, "val", tokenizer).parent.mkdir(parents=True)
        for split in ("val", "train"):
            compile_split(source, split, tokenizer)
