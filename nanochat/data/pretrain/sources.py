"""
Pretraining corpora: the registry, their local files, and which documents training reads.

A corpus's manifest.json lists the files of each split in order (fetch.py writes it): the pinned Hub parquet files of a
general-domain corpus (TextCorpus), or the zstd parquet files fetch materializes for a code corpus (CodeCorpus, code/).
Everything here reads the local files under <data_dir>/raw/<name>/ in that order, up to the first missing one, and
fails with the fetch command when a corpus, or enough of its files, has not been fetched. The tokenizer and compile read
documents only through read_training_documents, so both see the same selection.
"""

import os
import json
import hashlib
import itertools
from dataclasses import dataclass

import pyarrow.parquet as pq

from nanochat.data.storage import get_data_dir
from nanochat.data.pretrain.code import SAMPLING


@dataclass
class CodeCorpus:
    repo: str         # Hub dataset repo the corpus is built from (metadata only for Software Heritage corpora)
    revision: str     # pinned commit


@dataclass
class TextCorpus:
    repo: str         # Hub dataset repo
    revision: str     # pinned commit
    train_files: str  # glob over repo paths of the train files (`*` also matches `/`)
    val_files: str    # glob of the held-out val file(s), excluded from train


DATASETS = {
    # Code corpora (code/swh.py, code/stack_v3.py)
    # Stack-Edu (2025): 167M educational files of The Stack v2 in 15 languages; text from Software Heritage by blob ID
    "stack_edu": CodeCorpus(
        repo="HuggingFaceTB/stack-edu",
        revision="eeec5caac5cc3758a18f1d3ba4416837a9ba814c",
    ),
    # RefineCode (OpenCoder, 2024), reconstructed: its metadata for the files it shares with The Stack v2 (about half
    # its raw code, none of its code-related web data), joined with bigcode/the-stack-v2 for blob IDs; text from
    # Software Heritage, without notebooks
    "refinecode_stackv2_reconstructed": CodeCorpus(
        repo="OpenCoder-LLM/RefineCode-code-corpus-meta",
        revision="017558900665343ca563733212623e37606167ab",
    ),
    # The Stack v3 (2026) train split: 173M GitHub repositories (August 2025), near-deduplicated and filtered by its
    # publisher, one repository per row with its PII-redacted text inline; ODC-BY
    "stack_v3": CodeCorpus(
        repo="HuggingFaceCode/stack-v3-train",
        revision="1f61b735bc0a5698345ce2196730f24bfa467f33",
    ),
    # General-domain corpora, pre-shuffled, each holding out its last file for val
    # NVIDIA ClimbMix (2025), repackaged and shuffled: 6543 shards of ~250M chars in 1024-document row groups
    # (card: MIT; upstream nvidia/Nemotron-ClimbMix: CC-BY-NC-4.0)
    "climbmix": TextCorpus(
        repo="karpathy/climbmix-400b-shuffle",
        revision="915333b4f8b8684f39aeaafea600fea6f43fb703",
        train_files="shard_*.parquet",
        val_files="shard_06542.parquet",
    ),
    # HuggingFace Smol-Data (2026), ODC-BY, the closest ClimbMix analog: 100 shuffled shards of ~1B tokens of 50BT
    # FinePDFs-Edu, 30BT DCLM and 20BT FineWeb-Edu (0.47 / 0.32 / 0.21 of the characters; source in `dataset`)
    "smol_pdfedu_dclm_fwedu": TextCorpus(
        repo="HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled",
        revision="8904a95879538b9e7db6cf8636b2cd16b5e86a76",
        train_files="data/train-*.parquet",
        val_files="data/train-00099-of-00100.parquet",
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


def read_training_documents(name, path, columns):
    """Per row group of a raw file: (table of `columns` holding only the documents training reads, documents in the row
    group). A corpus in SAMPLING keeps, per program_lang, the documents whose sha256(document_id) as an integer is below
    share * 2**256, independent of sharding and of how much is fetched (blob IDs themselves are not uniform: 0.4391 of
    RefineCode's 58.9M Java blobs fall below 0.4454 of their range)."""
    sampling = SAMPLING.get(name)
    pf = pq.ParquetFile(path)
    for i in range(pf.num_row_groups):
        table = pf.read_row_group(i, columns=columns + (["document_id", "program_lang"] if sampling else []))
        if sampling:
            table = table.filter([int(hashlib.sha256(d.encode()).hexdigest(), 16) < sampling.get(lang, 1.0) * 2**256
                                  for d, lang in zip(table["document_id"].to_pylist(), table["program_lang"].to_pylist())])
        yield table.select(columns), pf.metadata.row_group(i).num_rows
