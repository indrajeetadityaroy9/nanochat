"""
The Stack v3 train split (HuggingFaceCode/stack-v3-train): one GitHub repository per row, its files' text inline, in
parts of whole repositories.

The manifest orders the parts by md5 of their path: the first part's first row group is val, and every row group of the
others is a train file, read from the Hub by its byte range. A file is its row group's repositories flattened to one row
per source file, in repository order: the file's fields (content as text, content_id, the sha1 of its original bytes,
as document_id, file_path as path), then its repository's (repo_path as repository).
"""

import os
import hashlib
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from nanochat.data.storage import write_json
from nanochat.data.pretrain.sources import DATASETS, raw_dir


def open_part(name, part):
    spec = DATASETS[name]
    return pq.ParquetFile(HfFileSystem().open(f"datasets/{spec.repo}@{spec.revision}/{part}", cache_type="none"))


def build_manifest(name):
    spec = DATASETS[name]
    parts = sorted((p for p in HfApi().list_repo_files(spec.repo, repo_type="dataset", revision=spec.revision) if p.startswith("data/")),
                   key=lambda p: hashlib.md5(p.encode()).hexdigest())
    with ThreadPoolExecutor() as pool:
        groups = list(pool.map(lambda p: open_part(name, p).num_row_groups, parts[1:]))
    stems = [os.path.basename(p).removesuffix(".parquet") for p in parts]
    write_json(os.path.join(raw_dir(name), "manifest.json"), {
        "train": [f"train/{s}-{g}.parquet" for s, n in zip(stems[1:], groups) for g in range(n)],
        "val": [f"val/{stems[0]}-0.parquet"],
    })
    print(f"{name}: manifest of {sum(groups)} train row groups of {len(parts) - 1} parts, 1 val row group", flush=True)


def table(name, file):
    part, group = os.path.basename(file).removesuffix(".parquet").rsplit("-", 1)
    repos = open_part(name, f"data/{part}.parquet").read_row_group(int(group))
    files = repos["files"].combine_chunks()
    flat = pa.Table.from_struct_array(pc.list_flatten(files))
    repo = repos.drop_columns(["files"]).take(pc.list_parent_indices(files))
    return pa.Table.from_arrays(flat.columns + repo.columns, flat.column_names + repo.column_names).rename_columns(
        {"content": "text", "content_id": "document_id", "file_path": "path", "repo_path": "repository"})
