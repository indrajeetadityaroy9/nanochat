"""
The fetch stage, the only step that downloads pretraining data. It is resumable and deterministic: run it in tmux, and
rerun it after an interruption.

Per corpus it writes <data_dir>/raw/<name>/manifest.json once, the files of each split in order: for a text corpus its
pinned Hub files, sorted, the train files of its directories taken in turn, so the first --max-files take every
directory alike (Smol-Data keeps each source in its own directory); for a code corpus the files its source's manifest
defines (code/swh.py, code/stack_v3.py). Then it writes, in parallel, every val file and the first
--max-files train files that are missing: a text file is downloaded as it is, a code file is materialized by its
source as zstd parquet. A file appears only once complete, so a rerun with a larger --max-files adds only the missing
files.

python -m nanochat.data.pretrain.fetch --dataset=stack_edu --max-files=8
"""

import os
import time
import argparse
import itertools
from fnmatch import fnmatchcase
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from nanochat.data.storage import write_json
from nanochat.data.pretrain.sources import DATASETS, raw_dir, read_manifest
from nanochat.data.pretrain.code import swh, stack_v3

ROW_GROUP = 1024  # documents per parquet row group, as in ClimbMix: compile's read batch
CODE_SOURCES = {"stack_edu": swh, "stack_v3": stack_v3}


def build_manifest(name):
    if name in CODE_SOURCES:
        return CODE_SOURCES[name].build_manifest(name)
    spec = DATASETS[name]
    files = sorted(HfApi().list_repo_files(spec.repo, repo_type="dataset", revision=spec.revision))
    val = [f for f in files if fnmatchcase(f, spec.val_files)]
    directories = defaultdict(list)
    for f in files:
        if fnmatchcase(f, spec.train_files) and f not in val:
            directories[os.path.dirname(f)].append(f)
    train = [f for turn in itertools.zip_longest(*directories.values()) for f in turn if f is not None]
    write_json(os.path.join(raw_dir(name), "manifest.json"), {"train": train, "val": val})


def materialize(name, file):
    """Write one file of a corpus into place; return its number of documents."""
    path = os.path.join(raw_dir(name), file)
    if name in CODE_SOURCES:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pq.write_table(CODE_SOURCES[name].table(name, file), path + ".tmp", compression="zstd", row_group_size=ROW_GROUP)
        os.replace(path + ".tmp", path)
    else:
        spec = DATASETS[name]
        hf_hub_download(spec.repo, file, repo_type="dataset", revision=spec.revision, local_dir=raw_dir(name))
    return pq.read_metadata(path).num_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch a pretraining corpus into local files (the only network step)")
    parser.add_argument("--dataset", type=str, required=True, help="corpus in nanochat/data/pretrain/sources.py")
    parser.add_argument("--max-files", type=int, required=True, help="write the first N train files (plus every val file)")
    args = parser.parse_args()
    root = raw_dir(args.dataset)
    if not os.path.exists(os.path.join(root, "manifest.json")):
        build_manifest(args.dataset)
    manifest = read_manifest(args.dataset)
    todo = [f for f in manifest["val"] + manifest["train"][:args.max_files] if not os.path.exists(os.path.join(root, f))]
    t0 = time.time()
    with ProcessPoolExecutor() as pool:
        for file, documents in zip(todo, pool.map(materialize, [args.dataset] * len(todo), todo)):
            print(f"{file}: {documents:,} documents ({time.time() - t0:.0f}s)", flush=True)
    print(f"{args.dataset}: {len(manifest['val'])} val and {min(args.max_files, len(manifest['train']))} train files in {root}")
