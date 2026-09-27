"""
Pretraining corpora: the registry, their local files, and which documents training reads.

- Code corpora (CodeCorpus) are materialized by fetch into zstd parquet shards of one document schema
  (DOCUMENT_COLUMNS); a unit is complete once its sidecar <unit>.json lists its files.
- General-domain corpora (TextCorpus) are the pinned Hub parquet files as they are; a unit is one file.

Everything here reads the local files under <data_dir>/raw/<name>/, in the unit order of its manifest.json, and fails
with the fetch command when a corpus, or enough of its units, has not been fetched. The tokenizer and compile read
documents only through read_training_documents, so both see the same selection.
"""

import os
import json
import hashlib
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq

from nanochat.data.storage import get_data_dir


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
    # Code corpora (fetch_swh.py, fetch_stack_v3.py)
    # Stack-Edu (2025): 167M educational files of The Stack v2 in 15 languages; text from Software Heritage by blob ID
    "stack_edu": CodeCorpus(
        repo="HuggingFaceTB/stack-edu",
        revision="eeec5caac5cc3758a18f1d3ba4416837a9ba814c",
    ),
    # RefineCode (OpenCoder, 2024), reconstructed: its metadata for the files it shares with The Stack v2 (about half
    # its raw code, none of its code-related web data), joined with bigcode/the-stack-v2 for blob IDs; text from
    # Software Heritage, without notebooks. ~25% of its files are also in stack_edu.
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

# Code corpora: manifest order, document schema, overlaps and training-time sampling

# Seed of the code corpus manifests' hash order (fetch_swh.py, fetch_stack_v3.py)
SEED = 42

# Document schema of every code corpus; source-specific columns may follow
DOCUMENT_COLUMNS = [
    "text",            # the file's text (UTF-8)
    "source_dataset",  # the corpus name
    "language",        # Linguist language name
    "document_id",     # the source's own id: Software Heritage blob_id, Stack v3 content_id
    "content_hash",    # sha1 of the stored text: exact identity across sources
    "repository",      # owner/name
    "repository_id",   # GitHub repository id (stack_v3), else null
    "path",            # path within the repository
    "in_stack_edu",    # the blob is in Stack-Edu (null where unknown, as for stack_v3)
    "in_refinecode",   # the blob is in RefineCode (null where unknown)
    "license_type",
    "detected_licenses",
]

# In a mixture with the corpus it maps to, a corpus drops its documents flagged in_<that corpus>, so no blob is trained
# on twice
OVERLAPS = {"refinecode_stackv2_reconstructed": "stack_edu"}

# Share of each RefineCode program_lang that OpenCoder trained on (arXiv 2411.04905, 2.1.1: Java 449 -> 200 GB, HTML
# 474 -> 64 GB); other languages are kept whole
SAMPLING = {"refinecode_stackv2_reconstructed": {"java": 200 / 449, "html": 64 / 474}}


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


def completed_units(name, split):
    """Local file paths of each complete unit of a split, in manifest order up to the first incomplete unit: a text unit
    is its file; a code unit's sidecar <unit>.json lists its files."""
    root = raw_dir(name)
    text = isinstance(DATASETS[name], TextCorpus)
    done = []
    for unit in read_manifest(name)[split]:
        marker = os.path.join(root, unit if text else unit + ".json")
        if not os.path.exists(marker):
            break
        if text:
            done.append([marker])
        else:
            with open(marker) as f:
                done.append([os.path.join(root, p) for p in json.load(f)["files"]])
    return done


def list_raw_files(name, split):
    """Local paths of the completed files of a split, in manifest order."""
    return [path for paths in completed_units(name, split) for path in paths]


def require_raw_files(name, split, count=None):
    """The first `count` files of a split, or all of them when count is None, failing with the fetch command when they
    are not complete."""
    done = completed_units(name, split)
    files = [path for paths in done for path in paths]
    enough = len(files) >= count if count is not None else len(done) == len(read_manifest(name)[split])
    if not enough:
        raise FileNotFoundError(f"'{name}' {split}: {len(files)} files complete, not enough: run `{fetch_command(name, count or 1)}`")
    return files[:count]


def read_training_documents(name, path, columns, without=None):
    """Per row group of a raw file: (table of `columns` holding only the documents training reads, documents in the row
    group). A corpus in SAMPLING keeps, per program_lang, the documents whose sha256(document_id) as an integer is below
    share * 2**256, independent of sharding and of how much is fetched (blob IDs themselves are not uniform: 0.4391 of
    RefineCode's 58.9M Java blobs fall below 0.4454 of their range). without=<corpus> drops documents flagged
    in_<corpus>."""
    sampling = SAMPLING.get(name, {})
    extra = (["document_id", "program_lang"] if sampling else []) + ([f"in_{without}"] if without else [])
    pf = pq.ParquetFile(path)
    for i in range(pf.num_row_groups):
        table = pf.read_row_group(i, columns=columns + extra)
        keep = np.ones(table.num_rows, dtype=bool)
        if sampling:
            keep &= [int(hashlib.sha256(d.encode()).hexdigest(), 16) < sampling.get(lang, 1.0) * 2**256
                     for d, lang in zip(table["document_id"].to_pylist(), table["program_lang"].to_pylist())]
        if without:
            keep &= ~table[f"in_{without}"].to_numpy()
        yield table.filter(keep).select(columns), table.num_rows


def parse_mixture(spec):
    """'climbmix' or 'stack_edu:0.3,refinecode_stackv2_reconstructed:0.2,smol_pdfedu_dclm_fwedu:0.5' -> [(name, weight)],
    weights summing to 1."""
    parts = [p.split(":") for p in spec.split(",")]
    weights = [float(p[1]) if len(p) == 2 else 1.0 for p in parts]
    return [(p[0], w / sum(weights)) for p, w in zip(parts, weights)]


def mixture_without(name, mixture_names):
    """The corpus whose documents `name` drops in a mixture of `mixture_names` (OVERLAPS), or None."""
    other = OVERLAPS.get(name)
    return other if other in mixture_names else None


def compiled_name(name, without=None):
    """The name of a corpus's compiled data, without the documents flagged in_<without>."""
    return f"{name}-without-{without}" if without else name


def compiled_mixture(spec):
    """[(compiled data, weight)] that training and evaluation read for a mixture spec: each corpus without the documents
    it shares with another corpus of the mixture (OVERLAPS)."""
    mixture = parse_mixture(spec)
    names = [name for name, _ in mixture]
    return [(compiled_name(name, mixture_without(name, names)), weight) for name, weight in mixture]
