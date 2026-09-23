"""
Test the Task container machinery: slicing views, mixtures, the HubDataset
parquet wrapper and split selection in load_hub_dataset (in-memory, no network).

python -m pytest tests/test_data_tasks.py -v
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from nanochat.data.tasks import common
from nanochat.data.tasks.common import Task, TaskMixture, HubDataset, render_mc


class ToyTask(Task):
    """A trivial task: example i is just {'i': i, 'tag': tag}."""

    def __init__(self, n=10, tag="a", **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.tag = tag

    def num_examples(self):
        return self.n

    def get_example(self, index):
        return {"i": index, "tag": self.tag}


def test_task_full():
    task = ToyTask(n=10)
    assert len(task) == 10
    assert task[0] == {"i": 0, "tag": "a"}
    assert task[9] == {"i": 9, "tag": "a"}


def test_task_slicing():
    # a view of [5, 10) has 5 examples and maps logical to physical indices
    task = ToyTask(n=10, start=5, stop=10)
    assert len(task) == 5
    assert task[0]["i"] == 5
    # step slicing uses ceil division for the length
    task = ToyTask(n=10, start=0, stop=10, step=3) # 0, 3, 6, 9
    assert len(task) == 4
    assert [task[i]["i"] for i in range(4)] == [0, 3, 6, 9]


def test_mixture_covers_all_examples_deterministically():
    mixture = TaskMixture([ToyTask(n=3, tag="a"), ToyTask(n=5, tag="b")])
    assert len(mixture) == 8
    examples = [mixture[i] for i in range(8)]
    # every example appears exactly once
    keys = sorted((ex["tag"], ex["i"]) for ex in examples)
    assert keys == [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1), ("b", 2), ("b", 3), ("b", 4)]
    # the shuffle is deterministic: a second instance yields the same order
    mixture2 = TaskMixture([ToyTask(n=3, tag="a"), ToyTask(n=5, tag="b")])
    assert examples == [mixture2[i] for i in range(8)]
    # and the tasks are actually interleaved, not concatenated
    assert [ex["tag"] for ex in examples] != ["a"] * 3 + ["b"] * 5


def test_mixture_oversampling():
    # passing a task twice doubles its examples
    mixture = TaskMixture([ToyTask(n=3), ToyTask(n=3)])
    assert len(mixture) == 6


def test_hub_dataset_rows():
    table = pa.table({"x": list(range(100)), "y": [str(i) for i in range(100)]})
    ds = HubDataset(table)
    assert len(ds) == 100
    assert ds[7] == {"x": 7, "y": "7"}


def test_hub_dataset_shuffle_matches_numpy():
    # the shuffle must reproduce datasets.Dataset.shuffle(seed) exactly,
    # which is a np.random.default_rng(seed) permutation
    table = pa.table({"x": list(range(100))})
    ds = HubDataset(table).shuffle(seed=42)
    perm = np.random.default_rng(42).permutation(100)
    assert [ds[i]["x"] for i in range(100)] == [int(p) for p in perm]
    # shuffling returns a view; the original order is untouched
    assert HubDataset(table)[0] == {"x": 0}


def test_load_hub_dataset_reads_the_split_in_sorted_file_order(tmp_path, monkeypatch):
    # row order must not depend on the order of the repo listing: seeded shuffles and
    # start/stop views select rows by position
    shards = {
        "data/train-00001-of-00002.parquet": [2, 3],
        "data/test-00000-of-00001.parquet": [9],
        "data/train-00000-of-00002.parquet": [0, 1],
    }
    for filename, rows in shards.items():
        (tmp_path / "data").mkdir(exist_ok=True)
        pq.write_table(pa.table({"x": rows}), tmp_path / filename)
    monkeypatch.setenv("NANOCHAT_DATA_DIR", str(tmp_path / "root"))
    monkeypatch.setattr(common, "list_repo_files", lambda repo, revision, cache_dir: ["README.md", *shards])
    monkeypatch.setattr(common, "fetch_repo_file", lambda repo, revision, filename, local_dir: str(tmp_path / filename))
    ds = common.load_hub_dataset("org/repo", "0" * 40, "data/train-*.parquet")
    assert [ds[i]["x"] for i in range(len(ds))] == [0, 1, 2, 3]
    with pytest.raises(AssertionError, match="validation"):
        common.load_hub_dataset("org/repo", "0" * 40, "data/validation-*.parquet")


def test_render_mc_letter_binding():
    query = render_mc("What is 1+1?", ("A", "B"), ("1", "2"))
    # the letter must directly follow '=' with no whitespace, so that the
    # prompt token for "A" matches the assistant's bare "A" response token
    assert "=A\n" in query and "=B\n" in query
