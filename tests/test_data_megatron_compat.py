"""
Test that nanochat's token shards are genuine Megatron-LM indexed datasets: shards written by nanochat
read identically through megatron.core's IndexedDataset, and packed rows written by Megatron's
IndexedDatasetBuilder read identically through nanochat. Skipped where megatron.core cannot be imported
(it needs Linux with triton, as in NGC containers).

python -m pytest tests/test_data_megatron_compat.py -v
"""

import numpy as np
import pytest
import torch

from nanochat.data.shards import open_shard, token_dtype, write_shard

indexed_dataset = pytest.importorskip("megatron.core.datasets.indexed_dataset")


def random_rows(vocab_size, num_rows=37, row_len=17):
    return np.random.RandomState(0).randint(0, vocab_size, size=(num_rows, row_len)).astype(token_dtype(vocab_size))


@pytest.mark.parametrize("vocab_size", [32768, 100_000]) # uint16 and int32 tokens
def test_nanochat_shard_reads_in_megatron(tmp_path, vocab_size):
    rows = random_rows(vocab_size)
    prefix = str(tmp_path / "shard")
    write_shard(prefix, rows)
    dataset = indexed_dataset.IndexedDataset(prefix)
    assert len(dataset) == len(rows)
    assert all(np.array_equal(dataset[i], rows[i]) for i in range(len(rows)))
    assert np.array_equal(dataset.document_indices, np.arange(len(rows) + 1)) # one document per row


@pytest.mark.parametrize("vocab_size", [32768, 100_000])
def test_megatron_shard_reads_in_nanochat(tmp_path, vocab_size):
    rows = random_rows(vocab_size)
    prefix = str(tmp_path / "shard")
    builder = indexed_dataset.IndexedDatasetBuilder(prefix + ".bin", dtype=rows.dtype.type)
    for row in rows:
        builder.add_item(torch.from_numpy(row.astype(np.int64)))
        builder.end_document()
    builder.finalize(prefix + ".idx")
    assert np.array_equal(open_shard(prefix), rows)
