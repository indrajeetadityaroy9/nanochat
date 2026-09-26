"""
Token shards in Megatron-LM's indexed dataset format (.bin/.idx, "MMIDIDX" v1): pre-tokenized,
pre-packed training rows, readable with zero copies.

A compiled split is a directory:
    index.json            metadata and the ordered shard list; the source of truth, published last
    <name>.bin/<name>.idx one shard: a Megatron .bin/.idx pair

Each Megatron sequence (and document) is one packed training row of row_len = seq_len + 1 tokens
(inputs row[:-1], targets row[1:]); rows start with a document's BOS, and later documents in a row
start with their own BOS. The pairs are the files Megatron-LM, NeMo and NeMo Curator produce and
read (megatron.core.datasets.indexed_dataset.IndexedDataset), so the data interoperates with that
tooling, while nanochat reads them with the small mmap reader below instead of importing all of
megatron.core. Shards are immutable; index.json records a blake2b digest of each shard's rows, so a
resumed run can check that it continues on exactly the same data (index_id).

Compiled data is keyed by corpus, sequence length and tokenizer fingerprint:
    <base_dir>/data/compiled/<dataset>-T<seq_len>-<fingerprint>/{train,val}/
"""

import os
import json
import struct
import hashlib

import numpy as np

from nanochat.data.storage import get_data_dir

FORMAT = "nanochat-megatron-rows-v2"
INDEX_FILE = "index.json"

# Megatron .idx layout: header, version, dtype code, sequence count, document count, then int32 sequence
# lengths, int64 sequence byte offsets and int64 document boundaries (sequence indices, starting at 0)
_IDX_HEADER = b"MMIDIDX\x00\x00"
_IDX_VERSION = 1
_DTYPE_CODES = {np.dtype("<u2"): 8, np.dtype("<i4"): 4} # Megatron DType.uint16, DType.int32


def compiled_name(dataset, seq_len, fingerprint):
    return f"{dataset}-T{seq_len}-{fingerprint}"


def get_compiled_dir(dataset, seq_len, fingerprint):
    return os.path.join(get_data_dir(), "compiled", compiled_name(dataset, seq_len, fingerprint))


def token_dtype(vocab_size):
    """Megatron's token dtypes: uint16 when the vocabulary fits (2 bytes per token), else int32."""
    return np.dtype("<u2") if vocab_size <= 2**16 else np.dtype("<i4")


def _write_atomic(path, data):
    with open(path + ".tmp", "wb") as f:
        f.write(data)
    os.replace(path + ".tmp", path)


def write_shard(prefix, rows):
    """Write a (rows, row_len) token array as a Megatron .bin/.idx pair, one sequence per row; returns its index entry."""
    num_rows, row_len = rows.shape
    lengths = np.full(num_rows, row_len, dtype="<i4")
    offsets = np.arange(num_rows, dtype="<i8") * (row_len * rows.dtype.itemsize)
    documents = np.arange(num_rows + 1, dtype="<i8")
    idx = (_IDX_HEADER + struct.pack("<Q", _IDX_VERSION) + struct.pack("<B", _DTYPE_CODES[rows.dtype])
           + struct.pack("<Q", num_rows) + struct.pack("<Q", len(documents))
           + lengths.tobytes() + offsets.tobytes() + documents.tobytes())
    _write_atomic(prefix + ".bin", memoryview(rows).cast("B"))
    _write_atomic(prefix + ".idx", idx)
    return {"name": os.path.basename(prefix), "rows": int(num_rows), "digest": hashlib.blake2b(rows, digest_size=16).hexdigest()}


def open_shard(prefix):
    """Zero-copy (rows, row_len) view of a Megatron .bin/.idx pair of packed rows, backed by the page cache."""
    with open(prefix + ".idx", "rb") as f:
        assert f.read(len(_IDX_HEADER)) == _IDX_HEADER, f"{prefix}.idx is not a Megatron index"
        version, code, num_seqs, _num_docs = struct.unpack("<QBQQ", f.read(25))
        assert version == _IDX_VERSION, f"Unsupported Megatron index version {version}"
        lengths = np.frombuffer(f.read(4 * num_seqs), dtype="<i4")
        offsets = np.frombuffer(f.read(8 * num_seqs), dtype="<i8")
    dtype = next(dt for dt, c in _DTYPE_CODES.items() if c == code)
    row_len = int(lengths[0])
    # packed rows: all sequences have one length and are laid out back to back, so the .bin is a 2D array
    assert (lengths == row_len).all() and (offsets == np.arange(num_seqs) * row_len * dtype.itemsize).all(), f"{prefix} does not hold packed rows"
    return np.memmap(prefix + ".bin", dtype=dtype, mode="r", shape=(num_seqs, row_len))


def write_index(split_dir, meta, shards):
    """Publish a split: index.json lists the shards in stream order and is written atomically, last."""
    index = {"format": FORMAT, **meta, "rows": sum(s["rows"] for s in shards), "shards": shards}
    _write_atomic(os.path.join(split_dir, INDEX_FILE), json.dumps(index, indent=1).encode())
    return index


def check_index(index):
    assert index.get("format") == FORMAT, f"Unsupported compiled data format {index.get('format')!r}, expected {FORMAT!r}: recompile"
    assert index["rows"] > 0 and index["shards"], "Compiled split has no rows"
    return index


def index_id(index):
    """Identity of a split's exact contents and order (hash of the ordered shard digests)."""
    return hashlib.blake2b("".join(s["digest"] for s in index["shards"]).encode(), digest_size=8).hexdigest()
