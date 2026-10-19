"""
Compile the prepared sources (prepare.py) into tokens: per source and split, in compiled/<source>-<tokenizer data key>/,
the tokens of every curated document in file order, without padding or BOS (<split>.bin, loader.TOKEN_DTYPE), the
offset of each document's first token followed by the total (<split>.idx, int64), and the number of documents of each
curated file (<split>.json; pack.py packs each file's documents together). The data does not depend on the sequence
length: pack.py groups the documents into training rows of one.

A document is sources.documents: a repository sample (its files, each after a comment with its path, in dependency
order; prepare.curate) or the content of a notebook or text (a codeforces_cots conversation included). Row groups are
the unit of parallel work; the parent writes their tokens in task order, so the output does not depend on the number
of workers.

python -m nanochat.data.compile
"""

import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq

from nanochat.data.loader import TOKEN_DTYPE, split_path
from nanochat.data.sources import SOURCES, curated_files, documents
from nanochat.data.tokenizer import get_tokenizer
from nanochat.paths import write_json


def compile_row_group(task):
    """(tokens, document lengths) of one row group; task = (source, curated shard, row group index). The tokenizer
    comes from get_tokenizer's per-process cache, not the task: it is too large to send per task."""
    source, path, row_group = task
    # a special token's text encodes as ordinary text (compiled data hold no special id); uint32, no Python int per token
    ids = [get_tokenizer().enc.encode_to_numpy(text, disallowed_special=())
           for text in documents(source, pq.ParquetFile(path).read_row_group(row_group))]
    lengths = np.array([len(doc) for doc in ids], dtype=np.int64)
    return np.frombuffer(b"".join(ids), np.uint32).astype(TOKEN_DTYPE), lengths


def compile_split(source, split, tokenizer):
    """Compile a source's curated shards of a split into <split>.bin and <split>.idx, with <split>.json beside them."""
    files = curated_files(source, split)
    path = split_path(source, split, tokenizer)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row_groups = {file: pq.read_metadata(file).num_row_groups for file in files}
    tasks = [(source, file, i) for file, count in row_groups.items() for i in range(count)]
    print(f"Compiling {len(files)} {split} files ({len(tasks):,} row groups) of '{source}' into {path}.bin")
    per_file, offset = [], 0
    with ProcessPoolExecutor() as pool, open(path + ".bin", "wb") as bin_file, open(path + ".idx", "wb") as idx_file:
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
    write_json(path + ".json", {"files": per_file})


if __name__ == "__main__":
    tokenizer = get_tokenizer()
    for source in SOURCES:
        for split in ("val", "train"):
            compile_split(source, split, tokenizer)
