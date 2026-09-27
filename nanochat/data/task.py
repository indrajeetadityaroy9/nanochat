"""
Benchmark tasks, read by pretraining decontamination (pretrain/decontam.py): a Task is a sliceable dataset of a
benchmark's items as conversations. Tasks read one split of a pinned HF dataset repo (load_hub_dataset), and
multiple-choice tasks share one prompt format (render_mc).
"""

import os
import fnmatch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from nanochat.data.storage import fetch, get_data_dir


class HubDataset:
    """
    Minimal stand-in for a HuggingFace datasets Dataset: wraps a pyarrow
    Table and offers lazy row access and a seeded shuffle.
    """

    def __init__(self, table, permutation=None):
        self.table = table
        self.permutation = permutation

    def __len__(self):
        return self.table.num_rows

    def shuffle(self, seed):
        # matches datasets.Dataset.shuffle(seed=seed) exactly, row order comes out identical
        permutation = np.random.default_rng(seed).permutation(len(self))
        return HubDataset(self.table, permutation)

    def __getitem__(self, index):
        physical_index = index if self.permutation is None else int(self.permutation[index])
        row = {column: self.table[column][physical_index].as_py() for column in self.table.column_names}
        return row


def load_hub_dataset(repo, revision, files):
    """
    Minimal stand-in for HuggingFace datasets.load_dataset: one split of a HF dataset repo at a
    pinned commit. files is a glob over repo-relative paths selecting the split's parquet shards, read in sorted order.
    Each is downloaded once per node into <data_dir>/<org>/<repo>/ (storage.fetch): under torchrun one rank downloads
    while the others wait, and a file already there is not requested again.
    """
    filenames = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=revision) if fnmatch.fnmatchcase(f, files))
    assert filenames, f"No files of {repo}@{revision} match {files!r}"
    local_dir = os.path.join(get_data_dir(), repo)
    tables = [pq.read_table(fetch(os.path.join(local_dir, f), lambda _: hf_hub_download(repo, f, repo_type="dataset", revision=revision, local_dir=local_dir)))
              for f in filenames]
    return HubDataset(pa.concat_tables(tables))


class Task:
    """
    Base class of a Task. Allows for lightweight slicing of the underlying dataset.
    """

    def __init__(self, start=0, stop=None, step=1):
        # allows a lightweight logical view over a dataset
        assert start >= 0, f"Start must be non-negative, got {start}"
        assert stop is None or stop >= start, f"Stop should be greater than or equal to start, got {stop} and {start}"
        assert step >= 1, f"Step must be strictly positive, got {step}"
        self.start = start
        self.stop = stop # could be None here
        self.step = step

    def num_examples(self):
        raise NotImplementedError

    def get_example(self, index):
        raise NotImplementedError

    def __len__(self):
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        num = (span + step - 1) // step # ceil_div(span, step)
        assert num >= 0, f"Negative number of examples???: {num}" # prevent footguns
        return num

    def __getitem__(self, index: int):
        assert isinstance(index, int), f"Index must be an integer, got {type(index)}"
        physical_index = self.start + index * self.step
        conversation = self.get_example(physical_index)
        return conversation


def render_mc(question, letters, choices):
    """
    The common multiple choice rendering format we will use.

    Note two important design decisions:
    1)
    Bigger models don't care as much, but smaller models prefer to have
    the letter *after* the choice, which results in better binding.
    2)
    There is no whitespace between the delimiter (=) and the letter.
    This is actually critical because the tokenizer has different token ids
    for " A" vs. "A". The assistant responses will be just the letter itself,
    i.e. "A", so it is important that here in the prompt it is the exact same
    token, i.e. "A" with no whitespace before it. Again, bigger models don't care
    about this too much, but smaller models do care about some of these details.
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query
