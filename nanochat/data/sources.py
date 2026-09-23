"""
Raw pretraining corpora: HuggingFace parquet datasets with one document per row in a text column.

Every corpus is one DatasetSpec entry in DATASETS below (adding a corpus means adding an entry),
pinned to a commit so its file listing and contents are immutable. Splits come from the pinned
repo listing and filename globs; files are fetched on first use into $NANOCHAT_DATA_DIR/raw/<name>/
at their repo-relative paths. A local-only corpus (repo=None) is read from files placed there.

Raw text feeds the tokenizer trainer (nanochat.tokenizer) and the compiler (nanochat.data.compile),
which turns it into the pre-tokenized token shards that training streams. To pre-fetch raw files
(e.g. to compile on a machine without internet access later):
python -m nanochat.data.sources --dataset=climbmix -n 170

Gated repos need a HuggingFace token (HF_TOKEN or `hf auth login`) and accepted terms on the Hub.
"""

import os
import argparse
from fnmatch import fnmatchcase
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool

import pyarrow.parquet as pq

from nanochat.data.storage import get_data_dir, list_repo_files, fetch_repo_file

# -----------------------------------------------------------------------------
# Dataset registry

@dataclass(frozen=True)
class DatasetSpec:
    repo: str | None    # HuggingFace dataset repo to download from; None = local-only
    train_files: str    # glob over repo-relative paths selecting the train shards (`*` also matches `/`)
    val_files: str      # glob selecting the held-out validation shard(s), always excluded from train
    text_column: str = "text"
    revision: str = "main"

# Revisions are pinned to the commit each entry was verified against, so runs are reproducible.
# Every entry is pre-shuffled (or source-mixed within row groups) and holds out its last shard.
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
CURATED = "-curated" # '<dataset>-curated' is the NeMo Curator output of a registered corpus (nanochat.data.curate)

def get_dataset(name):
    """The spec of a registered corpus, or of '<registered>-curated': its curated copy, stored locally."""
    if name.endswith(CURATED) and name.removesuffix(CURATED) in DATASETS:
        source = DATASETS[name.removesuffix(CURATED)]
        return DatasetSpec(repo=None, train_files="train/*.parquet", val_files="val/*.parquet", text_column=source.text_column)
    assert name in DATASETS, f"Unknown dataset '{name}', registered datasets: {', '.join(DATASETS)} (or their {CURATED} copies)"
    return DATASETS[name]

def dataset_names():
    """Every valid corpus name: the registered corpora and their curated copies."""
    return [*DATASETS, *(name + CURATED for name in DATASETS)]

def get_raw_dir(name):
    return os.path.join(get_data_dir(), "raw", name)

def select_split(spec, paths, split):
    """Sorted subset of repo-relative paths in a split: val matches val_files, train matches train_files minus val."""
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    val = {p for p in paths if fnmatchcase(p, spec.val_files)}
    if split == "val":
        return sorted(val)
    return sorted(p for p in paths if fnmatchcase(p, spec.train_files) and p not in val)

# -----------------------------------------------------------------------------
# Reading

def list_raw_files(name, split):
    """Repo-relative paths of a corpus split in sorted order: the pinned repo listing, or local files if repo=None."""
    spec, raw_dir = get_dataset(name), get_raw_dir(name)
    if spec.repo is None:
        paths = [os.path.relpath(os.path.join(root, f), raw_dir) for root, _, files in os.walk(raw_dir) for f in files if f.endswith(".parquet")]
    else:
        paths = list_repo_files(spec.repo, spec.revision, raw_dir)
    files = select_split(spec, paths, split)
    assert files, f"No {split} files for dataset '{name}' (repo={spec.repo}, train_files={spec.train_files!r}, val_files={spec.val_files!r}, local dir {raw_dir})"
    return files

def fetch_raw_file(name, filename):
    """Local path of one raw file of a corpus, fetched once if missing."""
    return fetch_spec_file(get_dataset(name), get_raw_dir(name), filename)

def fetch_spec_file(spec, raw_dir, filename):
    if spec.repo is None:
        return os.path.join(raw_dir, filename)
    return fetch_repo_file(spec.repo, spec.revision, filename, raw_dir)

def read_texts(pf, rg_idx, text_column):
    """The documents of one row group of an open parquet file, reading only the text column."""
    return pf.read_row_group(rg_idx, columns=[text_column]).column(text_column).to_pylist()

def iter_text_batches(name, split):
    """Documents of a corpus split, one parquet row group (a list of texts) at a time, fetching files as needed."""
    text_column = get_dataset(name).text_column
    for filename in list_raw_files(name, split):
        pf = pq.ParquetFile(fetch_raw_file(name, filename))
        for rg_idx in range(pf.num_row_groups):
            yield read_texts(pf, rg_idx, text_column)

# -----------------------------------------------------------------------------
# Pre-fetching

def download(name, num_train_files=-1, num_workers=8):
    """Fetch the validation file(s) and the first num_train_files train files (-1 = all) of a corpus."""
    train = list_raw_files(name, "train")
    val = list_raw_files(name, "val")
    files = val + (train if num_train_files == -1 else train[:num_train_files])
    print(f"Fetching {len(files) - len(val)}/{len(train)} train + {len(val)} val files of '{name}' into {get_raw_dir(name)}")

    def fetch(filename):
        try:
            fetch_raw_file(name, filename)
        except Exception as e:
            print(f"Failed to fetch {filename}: {e}")
            return False
        return True

    with ThreadPool(num_workers) as pool:
        num_ok = sum(pool.map(fetch, files))
    print(f"Done! {num_ok}/{len(files)} files present in {get_raw_dir(name)}")
    if num_ok < len(files):
        raise SystemExit(f"{len(files) - num_ok} files failed to download")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-fetch raw pretraining corpus files from the HuggingFace Hub")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, choices=list(DATASETS), help=f"registered corpus (default: {DEFAULT_DATASET})")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="number of train files to fetch (-1 = all); the val file(s) are always fetched")
    parser.add_argument("-w", "--num-workers", type=int, default=8, help="number of parallel downloads (default: 8)")
    args = parser.parse_args()
    download(args.dataset, args.num_files, args.num_workers)
