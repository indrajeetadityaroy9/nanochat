"""
Compile a fetched corpus into packed token rows: per split, one file of rows of seq_len + 1 tokens that training
memory-maps (stream.py).

Each document of the raw files, in order, is
- dropped, in train, if its exact text is a val document (text_hashes): a corpus holds val out by file or by repository
  and can repeat a val document elsewhere, so val measures held-out text only once train drops its copies;
- tokenized after a BOS and split losslessly into pieces of at most a row, each led by the BOS, cut after the last
  token of the window that ends a line (segment);
- packed by Best-Fit-Decreasing over all pieces of its raw file (pack, Ding et al. 2024): no piece is cut and no token
  discarded; the end of a row that no piece fits is padding (stream.pad_id), neither context nor target in training.

Val compiles first. Row groups, not files, are the unit of parallel work, so every CPU is busy however many files a
split has. The parent packs each file once its row groups are done and writes the rows in raw file order, so the output
does not depend on the number of workers. <split>.json beside <split>.bin records the settings and the statistics per
file and in total. Compiled data is keyed by the tokenizer's fingerprint, so the tokenizer must exist first.

python -m nanochat.data.pretrain.compile --dataset=climbmix --seq-len=2048 --max-files=170
python -m nanochat.data.pretrain.compile --dataset=stack_edu --max-files=100
"""

import os
import hashlib
import argparse
import itertools
import tempfile
from bisect import bisect_left, insort
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq

from nanochat.tokenizer import get_tokenizer
from nanochat.data.storage import write_json
from nanochat.data.pretrain.sources import DEFAULT_DATASET, raw_dir, require_raw_files
from nanochat.data.pretrain.stream import compiled_path, token_dtype, pad_id


def text_hashes(texts):
    """64-bit blake2b hash of each text's exact contents."""
    return np.fromiter((int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "little") for t in texts),
                       dtype=np.uint64, count=len(texts))


def segment(doc, row_len, ends_line):
    """Pieces of at most row_len tokens that cover a BOS-prefixed document exactly, each starting with its BOS: a piece
    ends after the last token of its window that ends a line (ends_line, over the vocabulary), or with the window when
    none does."""
    if len(doc) <= row_len:
        return [doc]
    pieces, start = [], 1
    while len(doc) - start > row_len - 1:
        window = doc[start:start + row_len - 1]
        breaks = np.flatnonzero(ends_line[window])
        end = start + (breaks[-1] + 1 if breaks.size else len(window))
        pieces.append(np.concatenate((doc[:1], doc[start:end])))
        start = end
    pieces.append(np.concatenate((doc[:1], doc[start:])))
    return pieces


def pack(pieces, row_len, pad):
    """Rows of row_len tokens from pieces of at most row_len tokens by Best-Fit-Decreasing ("Fewer Truncations Improve
    Language Modeling", Ding et al. 2024): longest first, the earliest among equal lengths, each piece goes into the row
    with the least room that holds it, else starts a row, so no piece is cut; what a row's pieces leave is filled with
    pad. Rows wait in one list per room, so placing a piece moves no other row."""
    rows, waiting, rooms = [], defaultdict(list), []  # rooms: the sorted room sizes some unfilled row has
    for piece in sorted(pieces, key=len, reverse=True):  # a stable sort: equal lengths keep document order
        if (i := bisect_left(rooms, len(piece))) < len(rooms):
            room = rooms[i]
            row = waiting[room].pop()
            if not waiting[room]:
                del rooms[i]
        else:
            room, row = row_len, len(rows)
            rows.append([])
        rows[row].append(piece)
        if room := room - len(piece):
            if not waiting[room]:
                insort(rooms, room)
            waiting[room].append(row)
    for row in rows:
        yield np.concatenate(row + [np.full(row_len - sum(map(len, row)), pad, dtype=row[0].dtype)])


_config = {}  # the split's settings, sent once per worker: the tokenizer is too large to send per task


def _init_worker(config):
    _config.update(config)


def _compile_row_group(task):
    """(pieces in document order, statistics) of one row group; task = (raw file, row group index)."""
    path, row_group = task
    tokenizer = _config["tokenizer"]
    texts = pq.ParquetFile(path).read_row_group(row_group, columns=["text"]).column("text").to_pylist()
    val_copy = np.isin(text_hashes(texts), _config["val_hashes"])
    pieces, segmented = [], 0
    for text in itertools.compress(texts, ~val_copy):
        doc = np.array(tokenizer.encode(text, prepend=tokenizer.get_bos_token_id()), dtype=_config["dtype"])  # 2-4 bytes a token, not a Python int
        parts = segment(doc, _config["row_len"], _config["ends_line"])
        segmented += len(parts) > 1
        pieces += parts
    return pieces, Counter(documents=len(texts), dropped_val_copies=int(val_copy.sum()), segmented=segmented,
                           pieces=len(pieces), tokens=sum(map(len, pieces)))


def compile_split(dataset, split, tokenizer, val_hashes, *, seq_len, max_files):
    """Compile the first max_files raw files of a split (all when None) into <split>.bin, with <split>.json beside it,
    dropping the documents whose text hash is in val_hashes."""
    files = require_raw_files(dataset, split, max_files)
    path = compiled_path(dataset, split, seq_len, tokenizer)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row_len = seq_len + 1
    ends_line = np.array([t.endswith(b"\n") for t in tokenizer.enc.decode_tokens_bytes(range(tokenizer.get_vocab_size()))])
    config = dict(tokenizer=tokenizer, val_hashes=val_hashes, row_len=row_len, dtype=token_dtype(tokenizer), ends_line=ends_line)
    row_groups = [pq.read_metadata(file).num_row_groups for file in files]
    tasks = [(file, i) for file, count in zip(files, row_groups) for i in range(count)]
    print(f"Compiling {len(files)} {split} files ({len(tasks):,} row groups) of '{dataset}' into {path}")
    per_file, total = [], Counter()
    with (ProcessPoolExecutor(initializer=_init_worker, initargs=(config,)) as pool,
          tempfile.NamedTemporaryFile(dir=os.path.dirname(path), delete_on_close=False) as out):  # removed if compile fails
        results = pool.map(_compile_row_group, tasks)  # in task order, whichever worker finishes first
        for file, count in zip(files, row_groups):
            pieces, stats = [], Counter()
            for part, part_stats in itertools.islice(results, count):
                pieces += part
                stats.update(part_stats)
            rows = 0
            for row in pack(pieces, row_len, pad_id(tokenizer)):
                out.write(row)
                rows += 1
            stats.update(rows=rows, padding=rows * row_len - stats["tokens"])
            total.update(stats)
            per_file.append({"file": os.path.relpath(file, raw_dir(dataset)), **stats})
        out.close()
        os.replace(out.name, path)  # the compiled file appears only once complete
    settings = dict(dataset=dataset, seq_len=seq_len, tokenizer=tokenizer.fingerprint(), val_documents=len(val_hashes))
    write_json(os.path.splitext(path)[0] + ".json", {"settings": settings, "files": per_file, "total": total})
    print(f"{split}: {total['rows']:,} rows ({total['padding'] / (total['rows'] * row_len):.3%} padding) from "
          f"{total['documents'] - total['dropped_val_copies']:,} docs ({total['dropped_val_copies']:,} dropped as copies of "
          f"val documents; {total['segmented']:,} segmented into {total['pieces']:,} pieces)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile a raw corpus into packed token rows")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help=f"registered corpus in nanochat/data/pretrain/sources.py (default: {DEFAULT_DATASET})")
    parser.add_argument("--seq-len", type=int, default=2048, help="training sequence length; rows hold seq_len + 1 tokens (default: 2048)")
    parser.add_argument("--max-files", type=int, required=True, help="compile the first N raw train files (and every val file)")
    args = parser.parse_args()
    tokenizer = get_tokenizer()
    val_files = require_raw_files(args.dataset, "val")
    compile_split(args.dataset, "val", tokenizer, np.empty(0, np.uint64), seq_len=args.seq_len, max_files=None)
    val_hashes = np.unique(np.concatenate([text_hashes(pq.read_table(f, columns=["text"]).column("text").to_pylist()) for f in val_files]))
    compile_split(args.dataset, "train", tokenizer, val_hashes, seq_len=args.seq_len, max_files=args.max_files)
