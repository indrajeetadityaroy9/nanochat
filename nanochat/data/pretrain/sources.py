"""
Raw pretraining corpora: HuggingFace parquet datasets pinned to a commit, one document per row in a "text" column.
A split is the sorted repo listing filtered by glob: val_files selects the held-out file(s), train_files the rest.
Files are downloaded on first read (storage.fetch_repo_file) by the tokenizer trainer and the compiler.
Gated repos need a HuggingFace token (HF_TOKEN or `hf auth login`) and accepted terms on the Hub.
"""

from fnmatch import fnmatchcase
from dataclasses import dataclass

import pyarrow.parquet as pq

from nanochat.data.storage import list_repo_files, fetch_repo_file


@dataclass
class DatasetSpec:
    repo: str         # HuggingFace dataset repo
    train_files: str  # glob over repo-relative paths selecting the train files (`*` also matches `/`)
    val_files: str    # glob selecting the held-out validation file(s), always excluded from train
    revision: str     # the commit the entry was verified against

# Every entry is pre-shuffled (or source-mixed within row groups) and holds out its last file.
DATASETS = {
    # NVIDIA ClimbMix (2025), repackaged and shuffled: 6543 shards of ~250M chars, 1024-doc row groups
    # (repo card says MIT; upstream nvidia/Nemotron-ClimbMix is CC-BY-NC-4.0)
    "climbmix": DatasetSpec(
        repo="karpathy/climbmix-400b-shuffle",
        train_files="shard_*.parquet",
        val_files="shard_06542.parquet",
        revision="915333b4f8b8684f39aeaafea600fea6f43fb703",
    ),
    # HuggingFace Smol-Data (2026), globally shuffled, ODC-BY, 100 shards of ~1B tokens each.
    # Closest ClimbMix analog: ~100BT mix of 50BT FinePDFs-Edu + 30BT DCLM + 20BT FineWeb-Edu
    "smol_pdfedu_dclm_fwedu": DatasetSpec(
        repo="HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled",
        train_files="data/train-*.parquet",
        val_files="data/train-00099-of-00100.parquet",
        revision="8904a95879538b9e7db6cf8636b2cd16b5e86a76",
    ),
    # same mix with unfiltered FinePDFs instead of FinePDFs-Edu
    "smol_pdf_dclm_fwedu": DatasetSpec(
        repo="HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled",
        train_files="data/train-*.parquet",
        val_files="data/train-00099-of-00100.parquet",
        revision="5688961c9af46fc0ea39ac4776f3c2d3b0783c6b",
    ),
    "dclm_100b": DatasetSpec(
        repo="HuggingFaceFW/dclm_100BT-shuffled",
        train_files="data/train-*.parquet",
        val_files="data/train-00099-of-00100.parquet",
        revision="2fa015e4044ec442a0734e89658cdcc538d10dd4",
    ),
    "fineweb_edu_100b": DatasetSpec(
        repo="HuggingFaceFW/fineweb_edu_100BT-shuffled",
        train_files="data/train-*.parquet",
        val_files="data/train-00099-of-00100.parquet",
        revision="be6b2a50d3a9c60d330c45384e80c7863cd3a25d",
    ),
    # PDF-extracted text: long documents, not web HTML
    "finepdfs_edu_100b": DatasetSpec(
        repo="HuggingFaceFW/finepdfs_edu_100BT-shuffled",
        train_files="data/train-*.parquet",
        val_files="data/train-00099-of-00100.parquet",
        revision="bf02dfc14b1eb344cf1511053b705f1298ef01fe",
    ),
    # TxT360-v2 (2026) organic web subset, CC-BY-4.0: 828 shards, sources (Common Crawl, HPLT,
    # ClueWeb, S2ORC, Wikipedia, PubMed) mixed within row groups. The repo is still being updated.
    "txt360_v2_web": DatasetSpec(
        repo="IFM/TxT360-v2",
        train_files="web-high-medium/*.parquet",
        val_files="web-high-medium/Medium-High.chunk1-5d0785ab03-00413.parquet",
        revision="a86bdfb101ebaaf71b445f95fb0f9b9bc2a47511",
    ),
}
DEFAULT_DATASET = "climbmix"


def list_raw_files(name, split):
    """Sorted repo-relative paths of the "train" or "val" split of a corpus."""
    spec = DATASETS[name]
    paths = list_repo_files(spec.repo, spec.revision)
    val = [p for p in paths if fnmatchcase(p, spec.val_files)]
    return sorted(val if split == "val" else [p for p in paths if fnmatchcase(p, spec.train_files) and p not in val])


def read_texts(pf, rg_idx):
    """The documents of one row group of an open parquet file, reading only the text column."""
    return pf.read_row_group(rg_idx, columns=["text"]).column("text").to_pylist()


def iter_text_batches(name, split):
    """Documents of a corpus split, one parquet row group (a list of texts) at a time."""
    spec = DATASETS[name]
    for filename in list_raw_files(name, split):
        pf = pq.ParquetFile(fetch_repo_file(spec.repo, spec.revision, filename))
        for rg_idx in range(pf.num_row_groups):
            yield read_texts(pf, rg_idx)
