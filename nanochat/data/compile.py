"""
Compile a raw corpus into pre-tokenized, pre-packed token shards, once, in parallel: Megatron .bin/.idx
pairs holding one BOS-aligned training row per sequence (format: shards.py).
Training then reads the fixed-size rows in place (stream.py) and does no tokenization or packing at all.

Packing is BOS-aligned best-fit: every row starts with a document's BOS, the longest buffered
document that fits goes in next, and when none fits the shortest one is cropped to fill the row
exactly (no padding). The buffer is large (--buffer-docs, 16000) because a fuller buffer finds more exact fits:
on ClimbMix it halves the tokens lost to cropping (22% at 1000 documents, 12% at 16000, against an 11% floor
set by documents longer than a row). Work is split by raw file and each file is packed on its own, so the output
is identical for any number of workers, and an interrupted compile resumes where it stopped.
Shards are written to the node's compiled directory (<data_dir>/compiled/), which training reads.

Documents sharing a 13-word sequence with a CORE or chat-eval item are dropped before tokenization
(decontam.py, --decontam-ngram), so benchmark scores are not inflated by leaked test data.

The tokenizer must exist first (python -m nanochat.tokenizer): compiled data is keyed by its fingerprint.

python -m nanochat.data.compile --dataset=climbmix --seq-len=2048 --max-files=170
"""

import os
import json
import math
import time
import argparse
import multiprocessing as mp
from bisect import bisect_left, bisect_right, insort

import numpy as np
import pyarrow.parquet as pq

from nanochat.tokenizer import get_tokenizer
from nanochat.data.sources import DEFAULT_DATASET, DATASETS, list_raw_files, read_texts
from nanochat.data.storage import fetch_repo_file
from nanochat.data.shards import FORMAT, compiled_name, get_compiled_dir, token_dtype, write_shard, write_index
from nanochat.data.decontam import build_eval_index, contaminated


class BestFitPacker:
    """BOS-aligned best-fit packing of tokenized documents into rows of exactly row_len tokens."""

    def __init__(self, docs, row_len, buffer_docs):
        self.docs = iter(docs)
        self.row_len = row_len
        self.buffer_docs = buffer_docs
        self.keys = [] # (length, arrival) of the buffered documents, sorted
        self.buffer = {} # arrival -> tokens
        self.arrivals = 0

    def _refill(self):
        while len(self.keys) < self.buffer_docs:
            doc = next(self.docs, None)
            if doc is None:
                return
            insort(self.keys, (len(doc), self.arrivals))
            self.buffer[self.arrivals] = doc
            self.arrivals += 1

    def fill(self, row):
        """Fill row completely; returns False if the documents run out first (the row is then discarded)."""
        pos = 0
        while pos < self.row_len:
            self._refill()
            if not self.keys:
                return False
            remaining = self.row_len - pos
            i = bisect_right(self.keys, (remaining, math.inf)) - 1
            if i >= 0:
                # the longest document that fits entirely, earliest arrival among equal lengths
                i = bisect_left(self.keys, (self.keys[i][0], -1))
                length, arrival = self.keys.pop(i)
                row[pos:pos + length] = self.buffer.pop(arrival)
                pos += length
            else:
                # nothing fits: crop the shortest document (earliest arrival among equals) to fill the row
                _, arrival = self.keys.pop(0)
                row[pos:] = self.buffer.pop(arrival)[:remaining]
                pos = self.row_len
        return True

# -----------------------------------------------------------------------------
# Worker processes: one raw file per task

_worker = {}

def _init_worker(config):
    _worker.update(config, tokenizer=get_tokenizer())

def _compile_file(job):
    """Pack one raw file into shards <file index>-<part>.{bin,idx}; returns (file index, part manifest)."""
    file_index, filename = job
    cfg = _worker
    part_path = os.path.join(cfg["split_dir"], "parts", f"{file_index:05d}.json")
    if os.path.exists(part_path): # compiled by an earlier (interrupted) run
        with open(part_path) as f:
            return file_index, json.load(f)

    tokenizer = cfg["tokenizer"]
    bos = tokenizer.get_bos_token_id()
    raw_path = fetch_repo_file(cfg["spec"].repo, cfg["spec"].revision, filename)
    stats = {"docs": 0, "tokens": 0, "contaminated": 0}
    dtype = np.dtype(cfg["dtype"])
    def documents():
        pf = pq.ParquetFile(raw_path)
        for rg_idx in range(pf.num_row_groups):
            texts = read_texts(pf, rg_idx)
            if cfg["decontam_ngram"]:
                dropped = contaminated(texts, cfg["eval_index"], cfg["decontam_ngram"])
                stats["contaminated"] += int(dropped.sum())
                texts = [text for text, drop in zip(texts, dropped) if not drop]
            for i in range(0, len(texts), 256):
                for doc in tokenizer.encode(texts[i:i + 256], prepend=bos, num_threads=1):
                    stats["docs"] += 1
                    stats["tokens"] += len(doc)
                    yield np.array(doc, dtype=dtype) # the packing buffer holds 2-4 bytes per token, not Python ints

    packer = BestFitPacker(documents(), cfg["row_len"], cfg["buffer_docs"])
    buf = np.empty((cfg["shard_rows"], cfg["row_len"]), dtype=dtype)
    shards = []
    def flush(num_rows):
        prefix = os.path.join(cfg["split_dir"], f"{file_index:05d}-{len(shards):03d}")
        entry = write_shard(prefix, buf[:num_rows])
        shards.append(entry)
    num_rows = 0
    while packer.fill(buf[num_rows]):
        num_rows += 1
        if num_rows == len(buf):
            flush(num_rows)
            num_rows = 0
    if num_rows:
        flush(num_rows)

    part = {"file": filename, **stats, "shards": shards}
    with open(part_path + ".tmp", "w") as f:
        json.dump(part, f)
    os.replace(part_path + ".tmp", part_path)
    if cfg["delete_raw"]:
        os.remove(raw_path)
    return file_index, part

# -----------------------------------------------------------------------------

def compile_split(dataset, split, tokenizer, *, seq_len, max_files, workers, shard_mb, buffer_docs, decontam_ngram, delete_raw):
    """Compile one split of a corpus; returns its published index. decontam_ngram=0 keeps every document."""
    spec = DATASETS[dataset]
    fingerprint = tokenizer.fingerprint()
    split_dir = os.path.join(get_compiled_dir(dataset, seq_len, fingerprint), split)
    files = list_raw_files(dataset, split)
    if split == "train" and max_files > 0:
        files = files[:max_files]
    dtype = token_dtype(tokenizer.get_vocab_size())
    row_len = seq_len + 1
    shard_rows = max(1, int(shard_mb * 2**20) // (row_len * dtype.itemsize))
    eval_index = build_eval_index(decontam_ngram) if decontam_ngram > 0 else None
    decontam = {"ngram": decontam_ngram, "eval_index": eval_index.digest()} if eval_index is not None else None

    # parts from an interrupted run are only reused if they were made with the same settings
    settings = {"format": FORMAT, "repo": spec.repo, "revision": spec.revision, "row_len": row_len, "dtype": dtype.str, "shard_rows": shard_rows,
                "buffer_docs": buffer_docs, "decontamination": decontam}
    os.makedirs(os.path.join(split_dir, "parts"), exist_ok=True)
    settings_path = os.path.join(split_dir, "parts", "settings.json")
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            assert json.load(f) == settings, f"{split_dir} was compiled with different settings: delete it to recompile"
    else:
        with open(settings_path, "w") as f:
            json.dump(settings, f)

    config = {"spec": spec, "split_dir": split_dir, "row_len": row_len, "dtype": dtype.str,
              "shard_rows": shard_rows, "buffer_docs": buffer_docs, "decontam_ngram": decontam_ngram, "eval_index": eval_index,
              "delete_raw": delete_raw}
    workers = min(workers, len(files)) # the unit of parallelism is a raw file
    print(f"Compiling {len(files)} {split} files of '{dataset}' into {split_dir} ({workers} workers, {shard_rows:,} rows per shard)")
    if eval_index is not None:
        print(f"Decontamination: dropping documents that share a {decontam_ngram}-word sequence with the eval sets ({len(eval_index):,} n-grams)")
    t0 = time.time()
    by_index = {}
    with mp.Pool(workers, initializer=_init_worker, initargs=(config,)) as pool:
        for done, (file_index, part) in enumerate(pool.imap_unordered(_compile_file, enumerate(files)), start=1):
            by_index[file_index] = part
            rows = sum(s["rows"] for s in part["shards"])
            print(f"[{done}/{len(files)}] {part['file']}: {part['docs']:,} docs -> {rows:,} rows ({time.time() - t0:.0f}s)")
    parts = [by_index[i] for i in range(len(files))] # shards in raw file order: stream order is independent of workers

    meta = {
        "split": split,
        "dtype": dtype.str,
        "seq_len": seq_len,
        "row_len": row_len,
        "tokenizer": {"fingerprint": fingerprint, "vocab_size": tokenizer.get_vocab_size(), "bos_id": tokenizer.get_bos_token_id()},
        "source": {"dataset": dataset, "repo": spec.repo, "revision": spec.revision, "files": len(files)},
        "packing": {"algorithm": "bos_best_fit_crop", "buffer_docs": buffer_docs},
        "decontamination": None if decontam is None else {**decontam, "docs_dropped": sum(part["contaminated"] for part in parts)},
    }
    index = write_index(split_dir, meta, [s for part in parts for s in part["shards"]])
    tokens_in = sum(part["tokens"] for part in parts)
    tokens_kept = index["rows"] * row_len
    print(f"{split}: {index['rows']:,} rows = {tokens_kept:,} tokens in {len(index['shards'])} shards "
          f"from {sum(part['docs'] for part in parts):,} docs ({100 * (1 - tokens_kept / tokens_in):.1f}% of tokens cropped, "
          f"{sum(part['contaminated'] for part in parts):,} docs dropped as eval-contaminated, {time.time() - t0:.0f}s)")
    return index


def default_workers():
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile a raw corpus into pre-tokenized, pre-packed token shards")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help=f"registered corpus in nanochat/data/sources.py (default: {DEFAULT_DATASET})")
    parser.add_argument("--seq-len", type=int, default=2048, help="training sequence length; rows hold seq_len + 1 tokens (default: 2048)")
    parser.add_argument("--splits", type=str, default="train,val", help="comma-separated splits to compile (default: train,val)")
    parser.add_argument("--max-files", type=int, default=-1, help="compile only the first N raw train files (-1 = all)")
    parser.add_argument("--workers", type=int, default=default_workers(), help="compile processes (default: all CPUs available to this process)")
    parser.add_argument("--shard-mb", type=float, default=64, help="target shard size in MiB (default: 64)")
    parser.add_argument("--buffer-docs", type=int, default=16000, help="documents buffered for best-fit packing; more finds more exact fits (default: 16000)")
    parser.add_argument("--decontam-ngram", type=int, default=13, help="drop documents sharing an n-word sequence with a CORE or chat-eval item (0 = keep all; default: 13)")
    parser.add_argument("--delete-raw", action="store_true", help="delete each raw file once it is compiled (bounds disk use at scale)")
    args = parser.parse_args()
    tokenizer = get_tokenizer()
    print(f"Tokenizer fingerprint: {tokenizer.fingerprint()} | vocab: {tokenizer.get_vocab_size():,} | compiled name: {compiled_name(args.dataset, args.seq_len, tokenizer.fingerprint())}")
    for split in args.splits.split(","):
        compile_split(args.dataset, split, tokenizer, seq_len=args.seq_len, max_files=args.max_files, workers=args.workers,
                      shard_mb=args.shard_mb, buffer_docs=args.buffer_docs, decontam_ngram=args.decontam_ngram, delete_raw=args.delete_raw)
