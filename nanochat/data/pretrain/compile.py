"""
Compile a fetched corpus into packed token rows: per split, one file of rows of seq_len + 1 tokens that training
memory-maps (stream.py).

Each document of the raw files, in order, is
- dropped if it shares a --decontam-ngram word sequence with a CORE or benchmark item (decontam.py);
- tokenized after a BOS and, if code, split losslessly into pieces of at most a row (code.segment);
- packed by best fit over all documents of its raw file (pack), so only documents longer than a row are cropped.

Row groups, not files, are the unit of parallel work, so every CPU is busy however many files a split has. The parent
packs each file once its row groups are done and writes the rows in raw file order, so the output does not depend on
the number of workers. <split>.json beside <split>.bin records the settings and the statistics per file and in total.
Compiled data is keyed by the tokenizer's fingerprint, so the tokenizer must exist first.

python -m nanochat.data.pretrain.compile --dataset=climbmix --seq-len=2048 --max-files=170
python -m nanochat.data.pretrain.compile --dataset=stack_edu --max-files=100
"""

import os
import hashlib
import argparse
import itertools
import tempfile
from bisect import bisect_right
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq

from nanochat.tokenizer import get_tokenizer
from nanochat.data.storage import write_json
from nanochat.data.pretrain.sources import DEFAULT_DATASET, DATASETS, CodeCorpus, raw_dir, require_raw_files
from nanochat.data.pretrain.code import segment
from nanochat.data.pretrain.stream import compiled_path, token_dtype
from nanochat.data.pretrain.decontam import eval_ngrams, contaminated


def pack(docs, row_len):
    """Rows of exactly row_len tokens, by best fit: the longest remaining document that fits goes next, the earliest
    among equal lengths; when none fits, the shortest is cropped to fill the row and its rest is discarded. The
    unfinished last row is dropped. Documents wait in one queue per length, so taking one moves no other."""
    fitting, longer = defaultdict(deque), []  # at most a row long, one queue per length; longer ones only get cropped
    for doc in docs:
        (fitting[len(doc)] if len(doc) <= row_len else longer).append(doc)
    lengths = sorted(fitting)  # those with a document left
    longer = deque(sorted(longer, key=len))  # stable: arrival order among equal lengths

    def take(i):
        """The earliest document of lengths[i]."""
        queue = fitting[lengths[i]]
        if len(queue) == 1:
            del lengths[i]
        return queue.popleft()

    while True:
        row, remaining = [], row_len
        while remaining:
            if fits := bisect_right(lengths, remaining):
                doc = take(fits - 1)
                row.append(doc)
                remaining -= len(doc)
            elif lengths or longer:  # none fits: crop the shortest
                row.append((take(0) if lengths else longer.popleft())[:remaining])
                break
            else:
                return
        yield np.concatenate(row)


_config = {}  # the split's settings, sent once per worker: the tokenizer and eval n-grams are too large to send per task


def _init_worker(config):
    _config.update(config)


def _compile_row_group(task):
    """(pieces in document order, statistics) of one row group; task = (raw file, row group index)."""
    path, row_group = task
    tokenizer, ends_line = _config["tokenizer"], _config["ends_line"]
    texts = pq.ParquetFile(path).read_row_group(row_group, columns=["text"]).column("text").to_pylist()
    dropped = contaminated(texts, _config["eval_hashes"], _config["ngram"])
    pieces, segmented = [], 0
    for text in itertools.compress(texts, ~dropped):
        doc = np.array(tokenizer.encode(text, prepend=tokenizer.get_bos_token_id()), dtype=_config["dtype"])  # 2-4 bytes a token, not a Python int
        parts = [doc] if ends_line is None else segment(doc, _config["row_len"], ends_line)
        segmented += len(parts) > 1
        pieces += parts
    return pieces, Counter(documents=len(texts), dropped_contaminated=int(dropped.sum()), segmented=segmented,
                           pieces=len(pieces), tokens=sum(map(len, pieces)))


def compile_split(dataset, split, tokenizer, eval_hashes, *, ngram, seq_len, max_files):
    """Compile the first max_files raw files of a split (all when None) into <split>.bin, with <split>.json beside it."""
    files = require_raw_files(dataset, split, max_files)
    path = compiled_path(dataset, split, seq_len, tokenizer)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row_len, ends_line = seq_len + 1, None
    if isinstance(DATASETS[dataset], CodeCorpus):  # segment cuts code after tokens that end a line
        ends_line = np.array([t.endswith(b"\n") for t in tokenizer.enc.decode_tokens_bytes(range(tokenizer.get_vocab_size()))])
    config = dict(tokenizer=tokenizer, eval_hashes=eval_hashes, ngram=ngram, row_len=row_len, dtype=token_dtype(tokenizer),
                  ends_line=ends_line)
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
            for row in pack(pieces, row_len):
                out.write(row)
                rows += 1
            stats.update(rows=rows, tokens_cropped=stats["tokens"] - rows * row_len)
            total.update(stats)
            per_file.append({"file": os.path.relpath(file, raw_dir(dataset)), **stats})
        out.close()
        os.replace(out.name, path)  # the compiled file appears only once complete
    settings = dict(dataset=dataset, seq_len=seq_len, tokenizer=tokenizer.fingerprint(), decontam_ngram=ngram,
                    eval_ngrams=hashlib.sha256(eval_hashes.tobytes()).hexdigest())
    write_json(os.path.splitext(path)[0] + ".json", {"settings": settings, "files": per_file, "total": total})
    print(f"{split}: {total['rows']:,} rows from {total['documents'] - total['dropped_contaminated']:,} docs "
          f"({total['dropped_contaminated']:,} dropped as eval-contaminated, {total['segmented']:,} segmented into "
          f"{total['pieces']:,} pieces, {total['tokens_cropped'] / total['tokens']:.1%} of tokens cropped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile a raw corpus into packed token rows")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help=f"registered corpus in nanochat/data/pretrain/sources.py (default: {DEFAULT_DATASET})")
    parser.add_argument("--seq-len", type=int, default=2048, help="training sequence length; rows hold seq_len + 1 tokens (default: 2048)")
    parser.add_argument("--max-files", type=int, required=True, help="compile the first N raw train files (and every val file)")
    parser.add_argument("--decontam-ngram", type=int, default=13, help="drop documents sharing an n-word sequence with a CORE or benchmark item (default: 13)")
    args = parser.parse_args()
    tokenizer = get_tokenizer()
    eval_hashes = eval_ngrams(args.decontam_ngram)
    for split, max_files in [("train", args.max_files), ("val", None)]:
        compile_split(args.dataset, split, tokenizer, eval_hashes, ngram=args.decontam_ngram, seq_len=args.seq_len, max_files=max_files)
