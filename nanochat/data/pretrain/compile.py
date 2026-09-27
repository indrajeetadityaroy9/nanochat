"""
Compile a fetched corpus into packed token rows, once: per split, one file of rows of seq_len + 1 tokens that training
memory-maps (stream.py), so training does no tokenization or packing.

Each document of the raw files, in order:
- is read only if training reads it (sources.read_training_documents);
- is dropped if it shares a --decontam-ngram word sequence with a CORE or chat-eval item (decontam.py);
- is tokenized and, if code, split losslessly into pieces of at most a row (code.segment);
- is packed into BOS-aligned rows by best fit over all documents of its raw file (pack), so only documents longer than
  a row force cropping.

Raw files are packed in parallel, one per worker, and written in raw file order, so the output does not depend on the
number of workers. <split>.json beside <split>.bin records the settings and per-file and total statistics. The
tokenizer must exist first: compiled data is keyed by its fingerprint.

python -m nanochat.data.pretrain.compile --dataset=climbmix --seq-len=2048 --max-files=170
python -m nanochat.data.pretrain.compile --dataset=refinecode_stackv2_reconstructed --max-files=100
"""

import os
import shutil
import hashlib
import argparse
import itertools
import tempfile
from bisect import bisect_left, bisect_right
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from nanochat.tokenizer import get_tokenizer
from nanochat.data.storage import write_json
from nanochat.data.pretrain.sources import DEFAULT_DATASET, DATASETS, CodeCorpus, raw_dir, require_raw_files, read_training_documents
from nanochat.data.pretrain.code import SAMPLING, segment
from nanochat.data.pretrain.stream import compiled_path, token_dtype
from nanochat.data.pretrain.decontam import eval_ngrams, contaminated


def pack(docs, row_len):
    """Rows of exactly row_len tokens from a file's documents, by best fit: the longest remaining document that fits
    goes next (the earliest among equal lengths); when none fits, the shortest is cropped to fill the row and its rest is
    discarded. The unfinished last row is dropped."""
    docs = sorted(docs, key=len)  # stable: arrival order among equal lengths
    while True:
        row, remaining = [], row_len
        while remaining:
            if not docs:
                return
            fits = bisect_right(docs, remaining, key=len)
            if fits == 0:  # nothing fits: crop the shortest
                row.append(docs.pop(0)[:remaining])
                break
            doc = docs.pop(bisect_left(docs, len(docs[fits - 1]), key=len))
            row.append(doc)
            remaining -= len(doc)
        yield np.concatenate(row)


# -----------------------------------------------------------------------------
# Workers: one raw file per task; the split's configuration (tokenizer, eval n-grams) is set once per worker

_config = {}


def _init_worker(config):
    _config.update(config)


def _compile_file(path):
    """Pack one raw file into a temporary file of rows; return its path and statistics."""
    tokenizer, row_len, ends_line = _config["tokenizer"], _config["row_len"], _config["ends_line"]
    bos = tokenizer.get_bos_token_id()
    n = dict.fromkeys(["documents", "dropped_unselected", "dropped_contaminated", "segmented", "pieces", "tokens"], 0)

    def pieces():
        for table, listed in read_training_documents(_config["dataset"], path, ["text"]):
            texts = table.column("text").to_pylist()
            dropped = contaminated(texts, _config["eval_ngrams"], _config["ngram"])
            n["documents"] += listed
            n["dropped_unselected"] += listed - len(texts)
            n["dropped_contaminated"] += int(dropped.sum())
            for text in itertools.compress(texts, ~dropped):
                doc = np.array(tokenizer.encode(text, prepend=bos), dtype=_config["dtype"])  # 2-4 bytes per token in the packing buffer, not Python ints
                parts = [doc] if ends_line is None else segment(doc, row_len, ends_line)
                n["segmented"] += len(parts) > 1
                n["pieces"] += len(parts)
                n["tokens"] += sum(map(len, parts))
                yield from parts

    rows = 0
    with tempfile.NamedTemporaryFile(dir=_config["dir"], delete=False) as f:
        for row in pack(pieces(), row_len):
            f.write(row)
            rows += 1
    return f.name, n | {"rows": rows, "tokens_cropped": n["tokens"] - rows * row_len}


def compile_split(dataset, split, tokenizer, eval_hashes, *, ngram, seq_len, max_files):
    """Pack the first max_files raw files of a split (all when None) into its compiled file, with <split>.json
    beside it."""
    files = require_raw_files(dataset, split, max_files)
    path = compiled_path(dataset, split, seq_len, tokenizer)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ends_line = None
    if isinstance(DATASETS[dataset], CodeCorpus):
        ends_line = np.array([t.endswith(b"\n") for t in tokenizer.enc.decode_tokens_bytes(list(range(tokenizer.get_vocab_size())))])
    config = {"dataset": dataset, "tokenizer": tokenizer, "dir": os.path.dirname(path), "dtype": token_dtype(tokenizer),
              "row_len": seq_len + 1, "eval_ngrams": eval_hashes, "ngram": ngram, "ends_line": ends_line}
    print(f"Compiling {len(files)} {split} files of '{dataset}' into {path}")
    per_file = []
    with ProcessPoolExecutor(initializer=_init_worker, initargs=(config,)) as pool, tempfile.NamedTemporaryFile(dir=config["dir"], delete=False) as out:
        for file, (part, stats) in zip(files, pool.map(_compile_file, files)):  # in raw file order, whichever worker finishes first
            with open(part, "rb") as f:
                shutil.copyfileobj(f, out)
            os.remove(part)
            per_file.append({"file": os.path.relpath(file, raw_dir(dataset)), **stats})
    os.replace(out.name, path)  # the compiled file appears only once it is complete
    t = {key: sum(entry[key] for entry in per_file) for key in per_file[0] if key != "file"}
    settings = {"dataset": dataset, "sampling": SAMPLING.get(dataset), "seq_len": seq_len,
                "tokenizer": tokenizer.fingerprint(), "decontam_ngram": ngram, "eval_ngrams": hashlib.sha256(eval_hashes.tobytes()).hexdigest()}
    write_json(os.path.splitext(path)[0] + ".json", {"settings": settings, "files": per_file, "total": t})
    print(f"{split}: {t['rows']:,} rows from {t['documents'] - t['dropped_unselected'] - t['dropped_contaminated']:,} docs "
          f"({t['dropped_unselected']:,} not selected for training, {t['dropped_contaminated']:,} dropped as eval-contaminated, "
          f"{t['segmented']:,} segmented into {t['pieces']:,} pieces, {t['tokens_cropped'] / t['tokens']:.1%} of tokens cropped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile a raw corpus into packed token rows")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help=f"registered corpus in nanochat/data/pretrain/sources.py (default: {DEFAULT_DATASET})")
    parser.add_argument("--seq-len", type=int, default=2048, help="training sequence length; rows hold seq_len + 1 tokens (default: 2048)")
    parser.add_argument("--max-files", type=int, required=True, help="compile the first N raw train files (and every val file)")
    parser.add_argument("--decontam-ngram", type=int, default=13, help="drop documents sharing an n-word sequence with a CORE or chat-eval item (default: 13)")
    args = parser.parse_args()
    tokenizer = get_tokenizer()
    eval_hashes = eval_ngrams(args.decontam_ngram)
    for split, max_files in [("train", args.max_files), ("val", None)]:
        compile_split(args.dataset, split, tokenizer, eval_hashes, ngram=args.decontam_ngram, seq_len=args.seq_len, max_files=max_files)
