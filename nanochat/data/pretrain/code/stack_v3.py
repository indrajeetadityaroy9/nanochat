"""
The Stack v3 train split (HuggingFaceCode/stack-v3-train): one GitHub repository per row, its files' text inline, in
parts of whole repositories.

The manifest orders the parts by md5 of their path: the first part's first row group is val, and every row group of the
others is a train file, read from the Hub by its byte range. A file is its row group's repositories flattened to one row
per source file, in repository order (documents): the file's fields (content as text, content_id, the sha1 of its
original bytes, as document_id, file_path as path), then its repository's (repo_path as repository).

Only source code is kept: a file whose language (go-enry's, i.e. Linguist's) has the Linguist type `programming`, or is
SQL, which Linguist types as data but code corpora keep (Stack-Edu, StarCoder2's the-stack-v2-smol); markup, data and
prose files (HTML, CSS, JSON, XML, YAML, Markdown, Gettext, ...) and vendored files are dropped. The manifest records the
kept languages, read once from the Linguist release that names all 713 of Stack v3's languages.
"""

import os
import hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import yaml
import pyarrow as pa
import pyarrow.compute as pc

from nanochat.data.storage import write_json
from nanochat.data.pretrain.sources import hub_files, open_hub_file, raw_dir, read_manifest

LINGUIST = "https://raw.githubusercontent.com/github-linguist/linguist/v9.2.0/lib/linguist/languages.yml"


def build_manifest(name):
    parts = sorted(hub_files(name), key=lambda p: hashlib.md5(p.encode()).hexdigest())
    with ThreadPoolExecutor() as pool:
        groups = list(pool.map(lambda p: open_hub_file(name, p).num_row_groups, parts[1:]))
    linguist = yaml.safe_load(urllib.request.urlopen(LINGUIST).read())
    languages = sorted({language for language, spec in linguist.items() if spec["type"] == "programming"} | {"SQL"})
    stems = [os.path.basename(p).removesuffix(".parquet") for p in parts]
    write_json(os.path.join(raw_dir(name), "manifest.json"), {
        "train": [f"train/{s}-{g}.parquet" for s, n in zip(stems[1:], groups) for g in range(n)],
        "val": [f"val/{stems[0]}-0.parquet"],
        "languages": languages,
    })
    print(f"{name}: manifest of {sum(groups)} train row groups of {len(parts) - 1} parts, 1 val row group, "
          f"{len(languages)} languages kept", flush=True)


def documents(repos):
    """A row group of repositories as documents: one row per source file, in repository order."""
    files = repos["files"].combine_chunks()
    flat = pa.Table.from_struct_array(pc.list_flatten(files))
    repo = repos.drop_columns(["files"]).take(pc.list_parent_indices(files))
    return pa.Table.from_arrays(flat.columns + repo.columns, flat.column_names + repo.column_names).rename_columns(
        {"content": "text", "content_id": "document_id", "file_path": "path", "repo_path": "repository"})


def table(name, file):
    """The source-code documents of a train or val file: kept languages, not vendored."""
    part, group = os.path.basename(file).removesuffix(".parquet").rsplit("-", 1)
    docs = documents(open_hub_file(name, f"data/{part}.parquet").read_row_group(int(group)))
    languages = pa.array(read_manifest(name)["languages"])
    return docs.filter(pc.and_(pc.is_in(docs["language"], value_set=languages), pc.invert(docs["is_vendor"])))
