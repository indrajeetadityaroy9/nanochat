"""
Publish the curated corpus (curate.py) to the Hub, and pull it where the tokenizer is trained. The curated shards are
text: normalized, checked, deduplicated, split and sharded, the repository files cut into import-connected samples;
nothing in them depends on a tokenizer. Tokenization, compile and pack run after the pull: packing places documents in
rows of a fixed number of tokens, and FIM encodes prefix, suffix and middle separately around sentinel ids
(nanochat/data/crate/src/pack.rs), so both need the tokenizer.

The dataset (config.DATASET, private) holds curated/<source>/<split>/<k>.parquet, manifest.json (prepare's sources,
the code's commit and the curated shards' rows) and its card. push replaces the dataset's whole tree and squashes its
history into one commit, whose sha it prints; pull downloads a revision into CORPUS.

python -m nanochat.corpus.publish push
python -m nanochat.corpus.publish pull --revision <sha>
"""

import argparse
import json
import subprocess

import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, dataset_info, snapshot_download

from nanochat.config import DATASET
from nanochat.corpus import mixture
from nanochat.corpus.schema import REPOSITORY_FILE, SAMPLE_FILE
from nanochat.corpus.sources import SOURCES, curated_dir
from nanochat.paths import CORPUS

PATTERNS = ["curated/**", "manifest.json", "README.md"]
# Bytes per preupload call: an upload URL the Hub's LFS batch call returns expires 15 minutes after the call
# (X-Amz-Expires=900), so every file of a call must start uploading within them
PREUPLOAD = 4 * 1024**3


def curated() -> dict[str, dict[str, dict[str, int]]]:
    """{source: {split: shards, rows, content bytes (Parquet footers)}} of every read source."""
    summary = {}
    for source in mixture.READ:
        summary[source] = {}
        for split in ("train", "val"):
            metadata = [pq.read_metadata(path) for path in curated_dir(source, split).iterdir()]
            column = metadata[0].schema.to_arrow_schema().get_field_index("content")
            summary[source][split] = {
                "shards": len(metadata), "rows": sum(md.num_rows for md in metadata),
                "bytes": sum(md.row_group(g).column(column).total_uncompressed_size
                             for md in metadata for g in range(md.num_row_groups))}
    return summary


def card(manifest: dict) -> str:
    """The dataset card: what the dataset holds, its layout, and per source its Hub origin, license, curated columns and
    rows. The Hub's viewer is off: its Parquet converter would copy every shard into one configuration of mixed
    schemas."""
    rows = []
    for source, splits in manifest["curated"].items():
        origin = manifest["sources"][source]
        license_ = (dataset_info(origin["repo"]).card_data or {}).get("license", "not declared")
        columns = (SAMPLE_FILE if SOURCES[source].schema is REPOSITORY_FILE else SOURCES[source].schema).names
        rows.append(f"| `{source}` | [{origin['repo']}](https://huggingface.co/datasets/{origin['repo']}) "
                    f"`{origin['revision'][:8]}` | {', '.join(license_) if isinstance(license_, list) else license_} | "
                    f"{', '.join(f'`{c}`' for c in columns)} | "
                    f"{splits['train']['shards']} / {splits['train']['rows']:,} | {splits['val']['rows']:,} |")
    return "\n".join([
        "---", "pretty_name: nanochat pretraining corpus, curated", "viewer: false", "---", "",
        "# nanochat pretraining corpus, curated", "",
        "The curated text of the nanochat pretraining corpus (`nanochat/corpus`, specified by",
        "`nanochat/corpus/normalization_doc.md`) at code commit "
        f"`{manifest['code']}`: every source normalized, Python files checked (and repaired) to parse as Python 3,",
        "exact copies removed across sources, 1% of groups held out as val, repository files cut into",
        "import-connected samples. Nothing depends on a tokenizer: tokenizer training, compile and packing (with FIM)",
        "run after `python -m nanochat.corpus.publish pull --revision <sha>`.", "",
        "Layout: `curated/<source>/<split>/<k>.parquet`. A repository source's shard k holds the repositories whose",
        "lowercased name has `md5_number(lower(repo_name)) % 100 = k`: `val/0`, `train/1` to `train/99`, each",
        "sorted by repository and path and cut into import-connected samples (`sample`). Any other source holds out",
        "1% of its groups (its lowercased repository for functions, the problem for Codeforces-CoTs, the document",
        "otherwise) in `val/0`, and its train rows in `id` order over `train/0` to `train/99`.",
        "`manifest.json` holds each source's Hub revision and files, and each split's shards, rows and text bytes.", "",
        "| Source | Origin (revision) | License | Columns | Train shards / rows | Val rows |",
        "|---|---|---|---|---|---|", *rows, ""])


def push() -> None:
    """Write the curated summary and the code's commit into the manifest and the card, then replace the dataset's tree:
    one commit deleting every file it holds, the files' upload in groups of at most PREUPLOAD bytes (the Hub skips an
    object it holds, so a rerun uploads only the rest), one commit adding them all, and a squash of the history into
    one commit."""
    path = CORPUS / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["code"] = subprocess.run(["git", "-c", f"safe.directory={CORPUS.parent.parent}", "rev-parse", "HEAD"],
                                      capture_output=True, text=True, check=True).stdout.strip()
    manifest["curated"] = curated()
    path.write_text(json.dumps(manifest, indent=1))
    (CORPUS / "README.md").write_text(card(manifest))
    api = HfApi()
    old = [f for f in api.list_repo_files(DATASET, repo_type="dataset") if f != ".gitattributes"]
    if old:
        api.create_commit(DATASET, repo_type="dataset", operations=[CommitOperationDelete(f) for f in old],
                          commit_message="Delete the previous corpus")
    files = [*sorted(CORPUS.glob("curated/*/*/*.parquet")), path, CORPUS / "README.md"]
    additions = [CommitOperationAdd(path_in_repo=f.relative_to(CORPUS).as_posix(), path_or_fileobj=f) for f in files]
    group, size = [], 0
    for addition in additions:
        group.append(addition)
        size += addition.upload_info.size
        if size >= PREUPLOAD or addition is additions[-1]:
            api.preupload_lfs_files(DATASET, additions=group, repo_type="dataset")
            group, size = [], 0
    api.create_commit(DATASET, repo_type="dataset", operations=additions, commit_message="nanochat corpus, curated")
    api.super_squash_history(DATASET, repo_type="dataset", commit_message="nanochat corpus, curated")
    print(f"{DATASET} revision {api.dataset_info(DATASET).sha}", flush=True)


def pull(revision: str) -> None:
    snapshot_download(DATASET, repo_type="dataset", revision=revision, local_dir=CORPUS, allow_patterns=PATTERNS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m nanochat.corpus.publish")
    parser.add_argument("step", choices=("push", "pull"))
    parser.add_argument("--revision")
    args = parser.parse_args()
    push() if args.step == "push" else pull(args.revision)
