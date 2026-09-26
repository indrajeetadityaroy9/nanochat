"""
Compile a raw corpus into packed token rows, once, in parallel: one file per split of rows of seq_len + 1 tokens,
which training memory-maps in place (stream.py), so it does no tokenization or packing at all.

Documents sharing an n-word sequence (--decontam-ngram) with a CORE or chat-eval item are dropped before tokenization
(decontam.py), so benchmark scores are not inflated by leaked test data.

Packing is BOS-aligned best-fit: every row starts with a document's BOS, the longest buffered document that fits goes
in next, and when none fits the shortest one is cropped to fill the row exactly (no padding). A fuller buffer
(--buffer-docs) finds more exact fits: on ClimbMix 22% of tokens are cropped at 1000 documents and 12% at 16000, against
an 11% floor set by documents longer than a row. Each raw file is packed on its own and the rows are written in raw
file order, so the output does not depend on the number of worker processes.

The tokenizer must exist first (python -m nanochat.tokenizer): compiled data is keyed by its fingerprint.

python -m nanochat.data.pretrain.compile --dataset=climbmix --seq-len=2048 --max-files=170
"""

import os
import shutil
import argparse
import itertools
import tempfile
import multiprocessing as mp
from bisect import bisect_left, bisect_right, insort
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

from nanochat.tokenizer import get_tokenizer
from nanochat.data.storage import fetch_repo_file
from nanochat.data.pretrain.sources import DEFAULT_DATASET, DATASETS, list_raw_files, read_texts
from nanochat.data.pretrain.stream import compiled_path, token_dtype
from nanochat.data.pretrain.decontam import build_eval_index, contaminated


def pack(docs, row_len, buffer_docs):
    """
    Rows of exactly row_len tokens from an iterator of documents, BOS-aligned best-fit: the longest buffered document
    that fits goes in next, and when none fits the shortest one is cropped to fill the row; among equal lengths the
    earliest document goes first. When the documents run out, the unfinished row is dropped.
    """
    buffer = [] # sorted by length; insort keeps arrival order among equal lengths
    while True:
        row, remaining = [], row_len
        while remaining:
            for doc in itertools.islice(docs, buffer_docs - len(buffer)):
                insort(buffer, doc, key=len)
            if not buffer:
                return
            fits = bisect_right(buffer, remaining, key=len)
            if fits == 0: # nothing fits: crop the shortest document to fill the row
                row.append(buffer.pop(0)[:remaining])
                break
            doc = buffer.pop(bisect_left(buffer, len(buffer[fits - 1]), key=len))
            row.append(doc)
            remaining -= len(doc)
        yield np.concatenate(row)

# -----------------------------------------------------------------------------
# Worker processes: one raw file per task

_config = {}

def _init_worker(config):
    _config.update(config, tokenizer=get_tokenizer())

def _compile_file(filename):
    """Pack one raw file into a temporary file of rows; returns its path and the file's document counts."""
    c = _config
    bos = c["tokenizer"].get_bos_token_id()
    pf = pq.ParquetFile(fetch_repo_file(c["repo"], c["revision"], filename))
    stats = Counter()
    def documents():
        for rg_idx in range(pf.num_row_groups):
            texts = read_texts(pf, rg_idx)
            dropped = contaminated(texts, c["eval_index"], c["ngram"])
            stats["contaminated"] += int(dropped.sum())
            for text in itertools.compress(texts, ~dropped):
                tokens = c["tokenizer"].encode(text, prepend=bos)
                stats["docs"] += 1
                stats["tokens"] += len(tokens)
                yield np.array(tokens, dtype=c["dtype"]) # the packing buffer holds 2-4 bytes per token, not Python ints
    with tempfile.NamedTemporaryFile(dir=c["dir"], delete=False) as f:
        for row in pack(documents(), c["row_len"], c["buffer_docs"]):
            f.write(row)
    return f.name, stats

# -----------------------------------------------------------------------------

def compile_split(dataset, split, tokenizer, eval_index, *, seq_len, max_files, buffer_docs, decontam_ngram):
    """Pack the first max_files raw files of a split into its compiled file."""
    spec = DATASETS[dataset]
    files = list_raw_files(dataset, split)[:max_files]
    path = compiled_path(dataset, split, seq_len, tokenizer)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    config = {"repo": spec.repo, "revision": spec.revision, "dir": os.path.dirname(path), "dtype": token_dtype(tokenizer),
              "row_len": seq_len + 1, "buffer_docs": buffer_docs, "eval_index": eval_index, "ngram": decontam_ngram}
    print(f"Compiling {len(files)} {split} files of '{dataset}' into {path}")
    stats = Counter()
    with mp.Pool(initializer=_init_worker, initargs=(config,)) as pool, tempfile.NamedTemporaryFile(dir=config["dir"], delete=False) as out:
        for part, part_stats in pool.imap(_compile_file, files): # in raw file order, whichever worker finishes first
            with open(part, "rb") as f:
                shutil.copyfileobj(f, out)
            os.remove(part)
            stats += part_stats
    os.replace(out.name, path) # the compiled file appears only once it is complete
    rows = os.path.getsize(path) // (config["row_len"] * config["dtype"].itemsize)
    print(f"{split}: {rows:,} rows from {stats['docs']:,} docs ({1 - rows * config['row_len'] / stats['tokens']:.1%} of tokens cropped, "
          f"{stats['contaminated']:,} docs dropped as eval-contaminated)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile a raw corpus into packed token rows")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help=f"registered corpus in nanochat/data/pretrain/sources.py (default: {DEFAULT_DATASET})")
    parser.add_argument("--seq-len", type=int, default=2048, help="training sequence length; rows hold seq_len + 1 tokens (default: 2048)")
    parser.add_argument("--max-files", type=int, required=True, help="compile the first N raw files of each split")
    parser.add_argument("--buffer-docs", type=int, default=16000, help="documents buffered for best-fit packing (default: 16000)")
    parser.add_argument("--decontam-ngram", type=int, default=13, help="drop documents sharing an n-word sequence with a CORE or chat-eval item (default: 13)")
    args = parser.parse_args()
    tokenizer = get_tokenizer()
    eval_index = build_eval_index(args.decontam_ngram)
    for split in ["train", "val"]:
        compile_split(args.dataset, split, tokenizer, eval_index, seq_len=args.seq_len, max_files=args.max_files,
                      buffer_docs=args.buffer_docs, decontam_ngram=args.decontam_ngram)
