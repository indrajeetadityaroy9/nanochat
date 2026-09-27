"""
The fetch stage, the only step that downloads pretraining data. It is resumable and deterministic: run it in tmux, and
rerun it after an interruption.

Per corpus it writes <data_dir>/raw/<name>/manifest.json, which fixes the order of the train units and the val units,
then completes every val unit and the train units in manifest order until the first --max-files train files are local;
a rerun with a larger --max-files adds only the missing units.
- Code corpora: fetch_swh.py or fetch_stack_v3.py builds the manifest and yields each unit's shards. A shard is renamed
  into place once written; the unit's sidecar <unit>.json, written last, marks the unit complete.
- General-domain corpora: the pinned Hub files as they are; the manifest records each file's sha256 (its Hub LFS id),
  which verify.py checks.

python -m nanochat.data.pretrain.fetch --dataset=stack_edu --max-files=8
"""

import os
import math
import time
import argparse
from collections import Counter
from fnmatch import fnmatchcase
from concurrent.futures import ProcessPoolExecutor

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download

from nanochat.data.storage import write_json
from nanochat.data.pretrain.sources import DATASETS, TextCorpus, raw_dir, read_manifest, completed_units
from nanochat.data.pretrain import fetch_swh, fetch_stack_v3

ROW_GROUP = 1024  # documents per parquet row group, as in ClimbMix: compile's read batch
FETCHERS = {"stack_edu": fetch_swh, "refinecode_stackv2_reconstructed": fetch_swh, "stack_v3": fetch_stack_v3}


def fetch_text(name, max_files):
    spec = DATASETS[name]
    root = raw_dir(name)
    sha256 = {f.path: f.lfs.sha256 for f in HfApi().list_repo_tree(spec.repo, repo_type="dataset", revision=spec.revision, recursive=True)
              if fnmatchcase(f.path, spec.train_files) or fnmatchcase(f.path, spec.val_files)}
    val = sorted(p for p in sha256 if fnmatchcase(p, spec.val_files))
    train = sorted(p for p in sha256 if p not in val)
    write_json(os.path.join(root, "manifest.json"), {"repo": spec.repo, "revision": spec.revision, "train": train, "val": val, "sha256": sha256})
    t0 = time.time()
    snapshot_download(spec.repo, repo_type="dataset", revision=spec.revision, allow_patterns=val + train[:max_files], local_dir=root)
    print(f"{name}: {len(val + train[:max_files])} files ({len(val)} val) complete in {root} ({time.time() - t0:.0f}s)")


def materialize(name, unit):
    """Write a code unit's shards, then its sidecar, which lists them and marks the unit complete."""
    t0 = time.time()
    root = raw_dir(name)
    stats, files = Counter(), []
    for file, table, dropped in FETCHERS[name].shards(name, unit):
        path = os.path.join(root, file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pq.write_table(table, path + ".tmp", compression="zstd", row_group_size=ROW_GROUP)
        os.replace(path + ".tmp", path)
        files.append(file)
        stats["documents"] += table.num_rows
        stats["dropped"] += dropped
        stats["text_bytes"] += pc.sum(pc.binary_length(table["text"])).as_py()
    stats = {"files": files, **stats, "seconds": round(time.time() - t0, 1)}
    write_json(os.path.join(root, unit + ".json"), stats)
    return stats


def run_units(name, units):
    with ProcessPoolExecutor() as pool:
        for unit, stats in zip(units, pool.map(materialize, [name] * len(units), units)):
            print(f"{unit}: {len(stats['files'])} files, {stats['documents']:,} documents, {stats['text_bytes'] / 1e6:,.0f} MB "
                  f"of text, {stats['dropped']:,} dropped ({stats['seconds']:.0f}s)", flush=True)


def fetch_code(name, max_files):
    root = raw_dir(name)
    if not os.path.exists(os.path.join(root, "manifest.json")):
        FETCHERS[name].build_manifest(name)
    manifest = read_manifest(name)
    missing = lambda split: [u for u in manifest[split] if not os.path.exists(os.path.join(root, u + ".json"))]
    t0 = time.time()
    run_units(name, missing("val"))
    # train units in waves: the first is one unit, later ones sized by the files per unit so far, so a wave overshoots
    # --max-files by less than one wave and small requests stay small
    while True:
        done = completed_units(name, "train")
        have = sum(map(len, done))
        todo = missing("train")
        if have >= max_files or not todo:
            break
        run_units(name, todo[:min(os.cpu_count(), math.ceil((max_files - have) * len(done) / have)) if done else 1])
    print(f"{name}: {have} train files and {len(manifest['val'])} val units complete in {root} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch a pretraining corpus into local files (the only network step)")
    parser.add_argument("--dataset", type=str, required=True, help="corpus in nanochat/data/pretrain/sources.py")
    parser.add_argument("--max-files", type=int, required=True, help="materialize the first N train files (plus every val unit)")
    args = parser.parse_args()
    (fetch_text if isinstance(DATASETS[args.dataset], TextCorpus) else fetch_code)(args.dataset, args.max_files)
