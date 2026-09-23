"""
Test the pretraining data pipeline on a toy local corpus (no network): raw split selection,
compilation into packed token shards, and deterministic, elastic streaming.

python -m pytest tests/test_data_pretrain.py -v
"""

import os
import shutil
from collections import Counter

import numpy as np
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS, get_tokenizer, get_tokenizer_dir
from nanochat.data.sources import DATASETS, DatasetSpec, list_raw_files
from nanochat.data.compile import BestFitPacker, compile_split
from nanochat.data.shards import compiled_name, get_compiled_dir, open_shard
from nanochat.data.stream import DeviceBatches, TokenStream, load_index

SEQ_LEN = 16
WORDS = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "data", "stream", "shard", "token"]


def write_corpus(raw_dir, rng):
    """Toy corpus: 3 train files and 1 val file of documents with widely varying lengths, several row groups each."""
    for path in ["train/a.parquet", "train/b.parquet", "train/c.parquet", "val/v.parquet"]:
        docs = [" ".join(rng.choice(WORDS, size=rng.randint(1, 40))) for _ in range(60)]
        os.makedirs(os.path.join(raw_dir, os.path.dirname(path)), exist_ok=True)
        pq.write_table(pa.table({"text": docs}), os.path.join(raw_dir, path), row_group_size=20)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Env + registered local corpus + trained tokenizer, shared by the tests of this module."""
    root = tmp_path_factory.mktemp("pretrain")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("NANOCHAT_BASE_DIR", str(root / "base"))
        mp.setenv("NANOCHAT_DATA_DIR", str(root / "data"))
        mp.setitem(DATASETS, "toy", DatasetSpec(repo=None, train_files="train/*.parquet", val_files="val/*.parquet"))
        write_corpus(str(root / "data" / "raw" / "toy"), np.random.RandomState(0))
        texts = [" ".join(WORDS)] * 20
        RustBPETokenizer.train_from_iterator(iter(texts), 256 + len(SPECIAL_TOKENS) + 20).save(get_tokenizer_dir())
        yield root


def compile_toy(split="train", workers=2):
    return compile_split("toy", split, get_tokenizer(), seq_len=SEQ_LEN, max_files=-1, workers=workers,
                         shard_mb=0.0005, buffer_docs=8, remote=None, delete_raw=False)


@pytest.fixture(scope="module")
def compiled(corpus):
    index = compile_toy("train")
    compile_toy("val")
    local_dir = os.path.join(get_compiled_dir("toy", SEQ_LEN, get_tokenizer().fingerprint()), "train")
    return index, local_dir


def all_rows(index, local_dir):
    return [tuple(row) for sh in index["shards"] for row in open_shard(os.path.join(local_dir, sh["name"]))]


def stream_rows(stream, num_batches):
    it = iter(stream)
    return [tuple(row) for _ in range(num_batches) for row in next(it).numpy()]


def test_split_selection(corpus):
    assert list_raw_files("toy", "train") == ["train/a.parquet", "train/b.parquet", "train/c.parquet"]
    assert list_raw_files("toy", "val") == ["val/v.parquet"]


def test_best_fit_packer():
    # longest fitting doc first; when nothing fits, the shortest doc is cropped; a partial last row is dropped
    docs = [[1] * 4, [2] * 7, [3] * 3, [4] * 12, [5] * 2]
    packer = BestFitPacker(docs, row_len=10, buffer_docs=3)
    rows = []
    row = np.zeros(10, dtype=np.int64)
    while packer.fill(row):
        rows.append(row.tolist())
    assert rows == [[2] * 7 + [3] * 3, [1] * 4 + [5] * 2 + [4] * 4]


def test_compiled_rows_start_with_bos(compiled):
    index, local_dir = compiled
    bos = get_tokenizer().get_bos_token_id()
    rows = all_rows(index, local_dir)
    assert len(rows) == index["rows"] and all(len(r) == SEQ_LEN + 1 for r in rows)
    assert all(r[0] == bos for r in rows)


def test_compile_is_deterministic_and_resumable(compiled):
    index, local_dir = compiled
    # resume: all parts are reused (no shard rewritten) and the same index is published
    mtimes = {sh["name"]: os.path.getmtime(os.path.join(local_dir, sh["name"] + ".bin")) for sh in index["shards"]}
    os.remove(os.path.join(local_dir, "index.json"))
    assert compile_toy(workers=1)["shards"] == index["shards"]
    assert all(os.path.getmtime(os.path.join(local_dir, n + ".bin")) == t for n, t in mtimes.items())
    # a fresh compile with a different number of workers produces identical shards
    shutil.rmtree(local_dir)
    assert compile_toy(workers=3)["shards"] == index["shards"]


def test_epoch_covers_every_row_once(compiled):
    index, local_dir = compiled
    stream = TokenStream(index, local_dir, batch_rows=1, block_shards=2)
    rows = stream_rows(stream, index["rows"])
    assert Counter(rows) == Counter(all_rows(index, local_dir))
    assert rows != all_rows(index, local_dir) # shuffled


def test_global_batches_are_independent_of_world_size(compiled):
    # every optimizer step of 8 rows covers the same rows on 1, 2 or 4 ranks (with 4, 2, 1 micro-steps)
    index, local_dir = compiled
    def steps(world_size, num_steps=3, rows_per_step=8, batch_rows=2, start_row=0):
        accum = rows_per_step // (world_size * batch_rows)
        per_rank = [stream_rows(TokenStream(index, local_dir, batch_rows=batch_rows, rank=r, world_size=world_size, start_row=start_row, block_shards=2), num_steps * accum) for r in range(world_size)]
        per_micro = batch_rows * accum
        return [Counter(row for rows in per_rank for row in rows[s * per_micro:(s + 1) * per_micro]) for s in range(num_steps)]
    reference = steps(1)
    assert steps(2) == reference and steps(4) == reference
    # resuming after 1 step on a different world size continues with the same steps
    assert steps(4, num_steps=2, start_row=8) == reference[1:]


def test_workers_preserve_order(compiled):
    index, local_dir = compiled
    def batches(num_workers):
        stream = TokenStream(index, local_dir, batch_rows=3, block_shards=2)
        it = iter(DeviceBatches(stream, torch.device("cpu"), num_workers))
        return [next(it) for _ in range(12)]
    serial, parallel = batches(0), batches(2)
    for (x0, y0), (x1, y1) in zip(serial, parallel):
        assert torch.equal(x0, x1) and torch.equal(y0, y1)
    x, y = serial[0]
    assert x.dtype == torch.int32 and y.dtype == torch.int64 and x.shape == y.shape == (3, SEQ_LEN)
    assert torch.equal(x[:, 1:].long(), y[:, :-1]) # targets are inputs shifted by one


def test_remote_stream_with_bounded_cache(compiled, tmp_path):
    index, local_dir = compiled
    remote = str(tmp_path / "remote" / "train")
    shutil.copytree(local_dir, remote)
    cache = str(tmp_path / "cache")
    block_bytes = 2 * max(sh["bytes"] for sh in index["shards"])
    streamed = TokenStream(load_index(cache, remote), cache, remote, batch_rows=4, block_shards=2, cache_bytes=3 * block_bytes)
    reference = TokenStream(index, local_dir, batch_rows=4, block_shards=2)
    num_batches = 2 * index["rows"] // 4 # two epochs
    assert stream_rows(streamed, num_batches) == stream_rows(reference, num_batches)
    cached = [f.removesuffix(".bin") for f in os.listdir(cache) if f.endswith(".bin")]
    assert len(cached) <= 8 < len(index["shards"]) # 3 live blocks of 2 shards, plus one in-flight prefetch
    # a corrupted remote shard is rejected on fetch
    victim = next(sh["name"] for sh in index["shards"] if sh["name"] not in cached)
    with open(os.path.join(remote, victim + ".bin"), "r+b") as f:
        f.write(b"\xff\xff")
    corrupted = TokenStream(index, str(tmp_path / "cache2"), remote, batch_rows=4, block_shards=len(index["shards"]))
    with pytest.raises(IOError, match="Checksum mismatch"):
        stream_rows(corrupted, index["rows"] // 4 + 1)
