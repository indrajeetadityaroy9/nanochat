"""
Publish the curated corpus (curate.py) to the Hub, and pull it where the tokenizer is trained. The curated shards are
text: normalized, checked, deduplicated, split and sharded, the repository files cut into import-connected samples;
nothing in them depends on a tokenizer. Tokenization, compile and pack run after the pull: packing places documents in
rows of a fixed number of tokens, and FIM encodes prefix, suffix and middle separately around sentinel ids
(nanochat/data/crate/src/pack.rs), so both need the tokenizer.

The dataset (config.DATASET, private) holds CORPUS/curated's tree at its root, <source>/<split>/<k>.parquet, beside
manifest.json (prepare's sources, the code's commit and the curated shards' rows) and its card. push replaces the
dataset's whole tree and squashes its history into one commit, whose sha it prints; pull downloads a revision's shards
into CORPUS/curated and its manifest into CORPUS.

python -m nanochat.corpus.publish push
python -m nanochat.corpus.publish pull --revision <sha>
"""

import argparse
import json
import subprocess

import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download, snapshot_download

from nanochat.config import DATASET
from nanochat.corpus import mixture
from nanochat.corpus.schema import REPOSITORY_FILE, SAMPLE_FILE
from nanochat.corpus.sources import SOURCES, curated_dir
from nanochat.paths import CORPUS

# Bytes per preupload call: an upload URL the Hub's LFS batch call returns expires 15 minutes after the call
# (X-Amz-Expires=900), so every file of a call must start uploading within them
PREUPLOAD = 4 * 1024**3
# The card's account of each folder and each column (normalization_doc.md §1-§3)
CONTENTS = {
    "stack_edu_python": "GitHub Python files filtered for educational value",
    "stack_edu_cpp": "GitHub C++ files filtered for educational value",
    "ultradata_stack_v3_l2_python": "GitHub Python files, quality-filtered",
    "ultradata_stack_v3_l2_cpp": "GitHub C++ files, quality-filtered",
    "ultradata_code_l2_python": "GitHub Python files, quality-filtered",
    "ultradata_code_l2_cpp": "GitHub C++ files, quality-filtered",
    "ultradata_code_l3_python": "synthetic Python exercises: a task, then its solution",
    "ultradata_code_l3_cpp": "synthetic C++ exercises: a task, then its solution",
    "nemotron_rewriting_python": "GitHub Python files rewritten by an LLM (Qwen3-32B)",
    "nemotron_transpilation_cpp": "GitHub Python files transpiled into C++ by an LLM (Qwen3-32B)",
    "codexglue_python": "Python functions with their docstrings (CodeSearchNet)",
    "ultradata_math_l3_qa": "synthetic mathematics: questions and answers",
    "ultradata_math_l3_conversation": "synthetic mathematics: conversations",
    "ultradata_math_l3_multi_style": "synthetic mathematics: varied styles",
    "ultradata_math_l3_textbook_exercise": "synthetic mathematics: textbook exercises",
    "codeforces_cots": "Codeforces problems with finished DeepSeek-R1 reasoning traces and Python or C++ solutions",
    "kaggle_notebooks": "Kaggle notebooks in Markdown: title, dataset description, the notebook as a Python script",
    "climbmix": "English web text",
}
COLUMNS = {
    "content": "the document's text",
    "repo_name": "its GitHub repository, `owner/name`",
    "relative_path": "the file's path in its repository",
    "sample": "within a shard, files sharing a number import one another; they are consecutive, in dependency order",
    "id": "the SHA-1 of `content`",
    "language": "the code's language, `Python` or `C++`",
    "problem_id": "the Codeforces problem",
}


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
    """The dataset card: what the dataset is, its layout and schema, and per folder its contents, columns, origin and
    rows. The Hub's viewer is off: its Parquet converter would copy every shard into one configuration of mixed
    schemas."""
    folders = []
    for source, splits in manifest["curated"].items():
        repo = manifest["sources"][source]["repo"]
        columns = (SAMPLE_FILE if SOURCES[source].schema is REPOSITORY_FILE else SOURCES[source].schema).names
        folders.append(f"| `{source}` | {CONTENTS[source]} | {', '.join(f'`{c}`' for c in columns)} | "
                       f"[{repo}](https://huggingface.co/datasets/{repo}) | {splits['train']['rows']:,} | "
                       f"{splits['val']['rows']:,} |")
    rows = sum(split["rows"] for splits in manifest["curated"].values() for split in splits.values())
    return "\n".join([
        "---", "pretty_name: nanochat pretraining corpus", "viewer: false", "---", "",
        "# nanochat pretraining corpus", "",
        f"Text for pretraining a Python and C++ code model from scratch: {rows:,} documents from "
        f"{len(manifest['curated'])} sources. Every text has LF line ends and no byte-order mark; every Python file "
        "parses as Python 3, repaired from Python 2 where it did not; exact copies are removed across all sources.", "",
        "## Layout", "",
        "One folder per source, each with `train/` and `val/` Parquet shards. Validation holds 1% of each source: "
        "whole repositories in the repository folders (`stack_edu_*`, `ultradata_stack_v3_l2_*`, "
        "`ultradata_code_l2_*`) and `codexglue_python`, whole problems in `codeforces_cots`, documents elsewhere. A "
        "repository folder's shard holds whole repositories, each cut into samples of files that import one another "
        "(`sample`); any other folder's train rows are in `id` order, so its leading shards are a uniform sample. "
        "`manifest.json` records each source's origin and each folder's rows.", "",
        "## Schema", "",
        "| Column | Meaning |", "|---|---|", *(f"| `{c}` | {m} |" for c, m in COLUMNS.items()), "",
        "## Folders", "",
        "| Folder | Contents | Columns | Source | Train rows | Val rows |", "|---|---|---|---|---|---|", *folders, ""])


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
    additions = [CommitOperationAdd(path_in_repo=f.relative_to(CORPUS / "curated").as_posix(), path_or_fileobj=f)
                 for f in sorted((CORPUS / "curated").glob("*/*/*.parquet"))]
    additions += [CommitOperationAdd(path_in_repo=f.name, path_or_fileobj=f) for f in (path, CORPUS / "README.md")]
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
    """The revision's shards into CORPUS/curated, its manifest into CORPUS."""
    snapshot_download(DATASET, repo_type="dataset", revision=revision, local_dir=CORPUS / "curated",
                      allow_patterns=[f"{source}/**" for source in mixture.READ])
    hf_hub_download(DATASET, "manifest.json", repo_type="dataset", revision=revision, local_dir=CORPUS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m nanochat.corpus.publish")
    parser.add_argument("step", choices=("push", "pull"))
    parser.add_argument("--revision")
    args = parser.parse_args()
    push() if args.step == "push" else pull(args.revision)
