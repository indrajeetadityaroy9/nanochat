"""
Pretraining corpora: the registry, their files on the Hub, and their local files.

A corpus's Hub parquet files (hub_files) are its TextCorpus.train_files, which include val, or its CodeCorpus.files,
listed at the pinned commit, sorted. open_hub_file resolves one once, a request on the Hub's resolver rate limit (not
its API limit), and reads it from the storage URL it redirects to by HTTP range requests: the footer, then the row
groups read.

A corpus's manifest.json lists the files of each split in order (fetch.py writes it): the pinned Hub parquet files of a
text corpus, or the zstd parquet files fetch materializes for a code corpus (code/). The local reads here take the
files under <data_dir>/raw/<name>/ in that order, up to the first missing one, and fail with the fetch command when a
corpus, or enough of its files, has not been fetched.
"""

import os
import json
import itertools
from fnmatch import fnmatchcase
from dataclasses import dataclass

import fsspec
import pyarrow.parquet as pq
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

from nanochat.data.storage import get_data_dir


@dataclass
class CodeCorpus:
    repo: str         # Hub dataset repo the corpus is built from (metadata only for Stack-Edu)
    revision: str     # pinned commit
    files: str        # glob over repo paths of the parquet files it is built from (`*` also matches `/`)


@dataclass
class TextCorpus:
    repo: str         # Hub dataset repo
    revision: str     # pinned commit
    train_files: str  # glob over repo paths of the train files (`*` also matches `/`)
    val_files: str    # glob of the held-out val file(s), excluded from train


DATASETS = {
    # Code corpora, materialized by fetch (code/swh.py, code/stack_v3.py)
    # Stack-Edu (2025): 167M educational files of The Stack v2 in 15 languages; text from Software Heritage by blob ID
    "stack_edu": CodeCorpus(
        repo="HuggingFaceTB/stack-edu",
        revision="eeec5caac5cc3758a18f1d3ba4416837a9ba814c",
        files="*/train-*.parquet",
    ),
    # The Stack v3 (2026) train split: 173M GitHub repositories (August 2025), near-deduplicated and filtered by its
    # publisher, one repository per row with its PII-redacted text inline; ODC-BY. Fetch keeps its source code only
    # (code/stack_v3.py)
    "stack_v3": CodeCorpus(
        repo="HuggingFaceCode/stack-v3-train",
        revision="1f61b735bc0a5698345ce2196730f24bfa467f33",
        files="data/*.parquet",
    ),
    # Text corpora: the Hub parquet files as published, `text` column; val is held out by file
    # OpenCoder (2024): code-related pages recalled from FineWeb by fastText; 510 files of ~198k documents, each mixing
    # Common Crawl dumps; MIT
    "opc_fineweb_code": TextCorpus(
        repo="OpenCoder-LLM/opc-fineweb-code-corpus",
        revision="9e8e48e666c226294d6f9e6c2e13f2c84c1c06f3",
        train_files="data/train-*.parquet",
        val_files="data/train-00509-of-00510.parquet",
    ),
    # OpenCoder (2024): math-related pages recalled from FineWeb by fastText; 37 files of ~142k documents; ODC-BY
    "opc_fineweb_math": TextCorpus(
        repo="OpenCoder-LLM/opc-fineweb-math-corpus",
        revision="858b10c748e6c95e0cbc5ebd38543b1c4699857b",
        train_files="data/train-*.parquet",
        val_files="data/train-00036-of-00037.parquet",
    ),
    # NVIDIA ClimbMix (2025), repackaged and shuffled: 6543 shards of ~250M chars in 1024-document row groups
    # (card: MIT; upstream nvidia/Nemotron-ClimbMix: CC-BY-NC-4.0)
    "climbmix": TextCorpus(
        repo="karpathy/climbmix-400b-shuffle",
        revision="915333b4f8b8684f39aeaafea600fea6f43fb703",
        train_files="shard_*.parquet",
        val_files="shard_06542.parquet",
    ),
}
DEFAULT_DATASET = "climbmix"


def hub_files(name):
    """The corpus's parquet files on the Hub at the pinned commit, sorted."""
    spec = DATASETS[name]
    pattern = spec.files if isinstance(spec, CodeCorpus) else spec.train_files
    return sorted(f for f in HfApi().list_repo_files(spec.repo, repo_type="dataset", revision=spec.revision) if fnmatchcase(f, pattern))


def open_hub_file(name, file):
    """A parquet file of the corpus on the Hub, read by HTTP range requests at the storage URL it resolves to."""
    spec = DATASETS[name]
    meta = get_hf_file_metadata(hf_hub_url(spec.repo, file, repo_type="dataset", revision=spec.revision))
    return pq.ParquetFile(fsspec.filesystem("https").open(meta.location, cache_type="none", size=meta.size))


def raw_dir(name):
    return os.path.join(get_data_dir(), "raw", name)


def fetch_command(name, files="<N>"):
    return f"python -m nanochat.data.pretrain.fetch --dataset={name} --max-files={files}"


def read_manifest(name):
    path = os.path.join(raw_dir(name), "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset '{name}' is not available locally.\n\nFetch it first with:\n{fetch_command(name)}")
    with open(path) as f:
        return json.load(f)


def list_raw_files(name, split):
    """Local paths of the files of a split in manifest order, up to the first one not yet fetched."""
    root = raw_dir(name)
    return list(itertools.takewhile(os.path.exists, (os.path.join(root, f) for f in read_manifest(name)[split])))


def require_raw_files(name, split, count=None):
    """The first `count` files of a split, or all of them when count is None, failing with the fetch command when they
    are not fetched."""
    files = list_raw_files(name, split)
    if len(files) < (len(read_manifest(name)[split]) if count is None else count):
        raise FileNotFoundError(f"'{name}' {split}: {len(files)} files complete, not enough: run `{fetch_command(name, count or 1)}`")
    return files[:count]
