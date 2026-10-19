"""
Deduplication of the normalized sources (prepare.curate): a document is dropped when its cluster of exact and near
duplicates holds an earlier document, as StarCoder and StarCoder2 deduplicate their data (The Stack, Kocetkov et al.
arXiv 2211.15533 §3; The Stack v2, Lozhkov et al. arXiv 2402.19173 §3.1) and DeepSeek-Coder its data (arXiv
2401.14196 §2.3); repeated text wastes compute and capacity (Hernandez et al. arXiv 2205.10487).

The unit is StarCoder2's: a repository source's file, before files are grouped into repository samples (DeepSeek-Coder
deduplicates whole repository samples instead), and any other source's document. The repository files of all
repository sources form one universe (files are copied across corpora), the documents of all other sources another.
Documents are ordered by source (SOURCES), normalized file and row, and a cluster keeps its first document (StarCoder2
ranks copies by stars and forks, which these sources lack). A repository file GitHub Linguist marks vendored or
generated (go-enry is_vendor / is_generated, as The Stack v2: arXiv 2402.19173 §2.1) takes no part and is dropped.

Signatures and clusters are the crate's (data/bpe dedup.rs): StarCoder2's n-grams of words, rensa's R-MinHash in
SLOTS one-byte slots, exact duplicates by the text's id (its sha1), LSH candidates verified by estimated Jaccard
similarity at THRESHOLD; a document without an n-gram (fewer than NGRAM words) is deduplicated exactly only.
"""

import contextlib
import os
import shutil

import enry
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.integrate import quad

from nanochat.data.schema import REPOSITORY_FILE
from nanochat.data.sources import SOURCES
from nanochat.data.tokenizer import bpe
from nanochat.paths import data_dir

# StarCoder2's near-deduplication (bigcode-dataset near_deduplication/bigcode-v2: intra_dedup.py defaults ngram_size,
# num_perm; run.sh THRESHOLD): 5-grams of words, 250 permutations, Jaccard similarity 0.7
NGRAM, SLOTS, THRESHOLD = 5, 250, 0.7


def rows():
    """The rows per LSH band of StarCoder2's (bands, rows) for THRESHOLD and SLOTS: the pair minimizing the false
    positive area below the threshold plus the false negative area above it (intra_dedup.py optimal_param, after
    datasketch), the first minimum in its (bands, rows) order."""
    def error(b, r):
        def candidate(s):
            return 1 - (1 - s ** r) ** b
        return quad(candidate, 0, THRESHOLD)[0] + quad(lambda s: 1 - candidate(s), THRESHOLD, 1)[0]
    return min(((b, r) for b in range(1, SLOTS + 1) for r in range(1, SLOTS // b + 1)), key=lambda br: error(*br))[1]


def scan(task):
    """A normalized file's documents: their signatures, ids, whether each takes part (not a repository file go-enry
    marks vendored or generated) and whether each has an n-gram. Texts go to the crate and go-enry as UTF-8 bytes
    (Arrow strings cast to binary), not as Python str, whose wide forms take four bytes per character."""
    source, path = task
    repository = SOURCES[source].schema is REPOSITORY_FILE
    signatures, ids, takes_part, shingled = [], [], [], []
    pf = pq.ParquetFile(path)
    for g in range(pf.num_row_groups):
        table = pf.read_row_group(g, columns=["id", "content", "relative_path"] if repository else ["id", "content"])
        texts = table["content"].cast(pa.large_binary()).to_pylist()
        s, h = bpe().sketch(texts, n=NGRAM, slots=SLOTS)
        signatures.append(s)
        shingled.append(h)
        ids.append(bytes.fromhex("".join(table["id"].to_pylist())))
        takes_part.append(bytes(not (enry.is_vendor(p) or enry.is_generated(p, t)) for p, t in zip(table["relative_path"].to_pylist(), texts))
                          if repository else b"\x01" * len(texts))
    return [b"".join(parts) for parts in (signatures, ids, takes_part, shingled)]


def kept(files, pool):
    """{source: [the keep mask of each normalized file]} for files {source: normalized file paths in order}. pool
    (spawned: go-enry's Go runtime) scans the files into scratch files of the data store, which the crate memory-maps."""
    directory = os.path.join(data_dir(), "dedup")
    os.makedirs(directory)
    universes = {"repository_files": [source for source, spec in SOURCES.items() if spec.schema is REPOSITORY_FILE],
                 "documents": [source for source, spec in SOURCES.items() if spec.schema is not REPOSITORY_FILE]}
    r = rows()
    keep = {}
    for universe, sources in universes.items():
        paths = [os.path.join(directory, f"{universe}.{part}") for part in ("signatures", "ids", "takes_part", "shingled")]
        with contextlib.ExitStack() as stack:
            outputs = [stack.enter_context(open(path, "wb")) for path in paths]
            for parts in pool.map(scan, [(source, path) for source in sources for path in files[source]]):
                for output, part in zip(outputs, parts):
                    output.write(part)
        mask = np.frombuffer(bpe().duplicates(*paths, rows=r, threshold=THRESHOLD), np.bool_)
        counts = [pq.read_metadata(path).num_rows for source in sources for path in files[source]]
        masks = iter(np.split(mask, np.cumsum(counts)[:-1]))
        keep |= {source: [next(masks) for _ in files[source]] for source in sources}
        print(f"dedup {universe}: {len(mask):,}, kept {int(mask.sum()):,}")
    shutil.rmtree(directory)
    return keep
