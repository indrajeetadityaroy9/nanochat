"""
Pretraining corpora: the registry and their local files.

A corpus's manifest.json lists the files of each split in order (fetch.py writes it): the pinned Hub parquet files of a
text corpus (TextCorpus), or the zstd parquet files fetch materializes for a code corpus (CodeCorpus, code/).
Everything here reads the local files under <data_dir>/raw/<name>/ in that order, up to the first missing one, and
fails with the fetch command when a corpus, or enough of its files, has not been fetched.
"""

import os
import json
import itertools
from dataclasses import dataclass

from nanochat.data.storage import get_data_dir


@dataclass
class CodeCorpus:
    repo: str         # Hub dataset repo the corpus is built from (metadata only for Stack-Edu)
    revision: str     # pinned commit


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
    ),
    # The Stack v3 (2026) train split: 173M GitHub repositories (August 2025), near-deduplicated and filtered by its
    # publisher, one repository per row with its PII-redacted text inline; ODC-BY
    "stack_v3": CodeCorpus(
        repo="HuggingFaceCode/stack-v3-train",
        revision="1f61b735bc0a5698345ce2196730f24bfa467f33",
    ),
    # Text corpora: the Hub parquet files as published, `text` column. Fetch lists train files sorted, directories taken
    # in turn, so the first N take every source alike; val is held out by file
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
    # HuggingFace Smol-Data (2026), ODC-BY: ~100B tokens of FinePDFs-Edu (~50B), DCLM (~30B) and FineWeb-Edu (~20B),
    # each source in its own directory of 100 files (source in `dataset`); val is the last file of each source
    "smol_pdfedu_dclm_fwedu": TextCorpus(
        repo="HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT",
        revision="c6ff5c68259ca98f23ecffe245d5f0da4598e704",
        train_files="*_100BT/*.parquet",
        val_files="*_100BT/000_00099.parquet",
    ),
    # TxT360-v2 (2026) web-high-medium, CC-BY-4.0: its Medium-High quality bucket, 828 shards of ~0.93M documents from
    # two source files (chunk0, chunk1). English web (Common Crawl, ClueWeb, HPLT) is 93% of the characters, curated
    # text (S2ORC, PubMed Central, arXiv, Wikipedia, ...) the rest, in the same shares in every sampled row group.
    # Updating a source file replaces its shards, so the revision pins them.
    "txt360_v2_web": TextCorpus(
        repo="IFM/TxT360-v2",
        revision="a86bdfb101ebaaf71b445f95fb0f9b9bc2a47511",
        train_files="web-high-medium/*.parquet",
        val_files="web-high-medium/Medium-High.chunk1-5d0785ab03-00413.parquet",
    ),
}
DEFAULT_DATASET = "climbmix"


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
