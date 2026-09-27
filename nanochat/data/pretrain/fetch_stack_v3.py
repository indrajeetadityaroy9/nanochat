"""
The Stack v3 train split (HuggingFaceCode/stack-v3-train): one GitHub repository per row, its files' text inline.

The manifest orders the 8192 hash-partitioned parts by a seeded hash: the first is val, the rest are the train units.
A unit is one part, and each of its row groups (whole repositories, files in repository order) becomes one shard of the
document schema; the val unit keeps only its first row group. content_id, the sha1 of a file's original bytes, is kept
as document_id; content_hash is the sha1 of the published, PII-redacted text.
"""

import os
import hashlib
import tempfile

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from nanochat.data.storage import get_data_dir, write_json
from nanochat.data.pretrain.sources import DATASETS, DOCUMENT_COLUMNS, SEED, raw_dir

EXTRA_COLUMNS = ["commit_id", "file_timestamp", "is_vendor", "size_bytes", "stars", "forks", "is_fork"]


def build_manifest(name):
    spec = DATASETS[name]
    parts = [p for p in HfApi().list_repo_files(spec.repo, repo_type="dataset", revision=spec.revision)
             if p.startswith("data/") and p.endswith(".parquet")]
    ordered = sorted(parts, key=lambda p: hashlib.md5(f"{p}{SEED}".encode()).hexdigest())
    unit = lambda split, p: f"{split}/{os.path.basename(p).removesuffix('.parquet')}"
    write_json(os.path.join(raw_dir(name), "manifest.json"), {
        "repo": spec.repo, "revision": spec.revision,
        "train": [unit("train", p) for p in ordered[1:]], "val": [unit("val", ordered[0])],
    })
    print(f"{name}: manifest of {len(parts)} parts, 1 held out for val", flush=True)


def documents(name, repos):
    """The files of a row group of repositories as a table of the document schema."""
    files = repos.column("files").combine_chunks()
    owner = pc.list_parent_indices(files)
    flat = pc.list_flatten(files)
    field = lambda f: pc.struct_field(flat, f)
    of_repo = lambda c: pc.take(repos.column(c), owner)
    github = pc.take(repos.column("github_metadata"), owner)
    text = field("content")
    return pa.table({
        "text": text,
        "source_dataset": pa.repeat(name, len(flat)),
        "language": field("language"),
        "document_id": field("content_id"),
        "content_hash": pa.array([hashlib.sha1(t.encode()).hexdigest() for t in text.to_pylist()]),
        "repository": of_repo("repo_path"),
        "repository_id": of_repo("repo_id"),
        "path": field("file_path"),
        "in_stack_edu": pa.nulls(len(flat), pa.bool_()),
        "in_refinecode": pa.nulls(len(flat), pa.bool_()),
        "license_type": field("license_type"),
        "detected_licenses": field("detected_licenses"),
        "commit_id": of_repo("commit_id"),
        "file_timestamp": field("file_timestamp"),
        "is_vendor": field("is_vendor"),
        "size_bytes": field("size_bytes"),
        "stars": pc.struct_field(github, "stars"),
        "forks": pc.struct_field(github, "forks"),
        "is_fork": pc.struct_field(github, "is_fork"),
    }).select(DOCUMENT_COLUMNS + EXTRA_COLUMNS)


def shards(name, unit):
    """The unit's shards, one per row group of its part: (file, table of the document schema, dropped files: none)."""
    spec = DATASETS[name]
    split, stem = unit.split("/")
    with tempfile.TemporaryDirectory(dir=get_data_dir()) as tmp:
        part = hf_hub_download(spec.repo, f"data/{stem}.parquet", repo_type="dataset", revision=spec.revision, local_dir=tmp)
        with pq.ParquetFile(part) as source:
            for group in range(source.num_row_groups) if split == "train" else [0]:
                yield f"{unit}-{group}.parquet", documents(name, source.read_row_group(group)), 0
