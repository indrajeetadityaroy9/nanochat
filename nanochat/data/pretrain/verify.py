"""
Verify a fetched corpus; exits with status 1 when a check fails. Also reports how many units of each split are complete.

Code corpora, in one DuckDB scan: no document is empty, content_hash is the sha1 of the text, language and provenance
are present, source_dataset is the corpus, and a stack_v3 text is size_bytes long. For a corpus and the one it overlaps
(sources.OVERLAPS), over both whole manifests, every in_<other> flag equals whether the blob is in the other corpus, and
no fetched document with a blob of the other corpus is unflagged. For stack_v3, which has no blob IDs, reports its
exact copies (content_hash) of fetched documents of the other code corpora.

General-domain corpora: every fetched file has the sha256 fetch recorded for the pinned Hub file. Reports documents,
characters, empty documents, exact duplicates, val documents whose text is also in fetched train, and exact copies of
fetched documents of the other general-domain corpora.

python -m nanochat.data.pretrain.verify --dataset=stack_edu
"""

import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor

import duckdb
from huggingface_hub.utils.sha import sha_fileobj

from nanochat.data.pretrain.sources import DATASETS, OVERLAPS, CodeCorpus, raw_dir, read_manifest, completed_units, list_raw_files

failures = []


def check(ok, message):
    print(("ok      " if ok else "FAILED  ") + message)
    if not ok:
        failures.append(message)


def verify_units(name):
    manifest = read_manifest(name)
    for split in ["train", "val"]:
        print(f"{split}: {len(completed_units(name, split))} of {len(manifest[split])} units complete in manifest order")


def verify_documents(con, name, files):
    checks = {
        "empty": "text = ''",
        "hash mismatch": "sha1(text) != content_hash",
        "no language": "language IS NULL",
        "no provenance": "document_id IS NULL OR repository IS NULL OR path IS NULL",
        "foreign": f"source_dataset != '{name}'",
    } | ({"size mismatch": "strlen(text) != size_bytes"} if name == "stack_v3" else {})
    documents, *counts = con.execute(f"SELECT count(*), {', '.join(f'count(*) FILTER (WHERE {c})' for c in checks.values())} "
                                     f"FROM read_parquet({files})").fetchone()
    print(f"documents: {documents:,}")
    for label, n in zip(checks, counts):
        check(n == 0, f"{label}: {n:,}")


def verify_overlap(con, name, fetched):
    for corpus, other in OVERLAPS.items():
        if name in (corpus, other) and {corpus, other} <= fetched.keys():
            manifest = lambda n: f"read_parquet('{raw_dir(n)}/manifest.parquet')"
            a, b, both, wrong = con.execute(f"""SELECT (SELECT count(*) FROM {manifest(other)}), count(*), count(*) FILTER (WHERE shared),
                count(*) FILTER (WHERE shared != in_{other}) FROM (SELECT in_{other}, blob_id IN (SELECT blob_id FROM {manifest(other)})
                AS shared FROM {manifest(corpus)})""").fetchone()
            print(f"blobs: {other} {a:,}, {corpus} {b:,}, both {both:,} ({both / b:.1%} of {corpus}), union {a + b - both:,}")
            check(wrong == 0, f"{corpus} manifest rows whose in_{other} flag disagrees with the blob overlap: {wrong:,}")
            unflagged = con.execute(f"""SELECT count(*) FROM read_parquet({fetched[corpus]}) WHERE NOT in_{other}
                AND document_id IN (SELECT document_id FROM read_parquet({fetched[other]}))""").fetchone()[0]
            check(unflagged == 0, f"materialized {corpus} documents with a {other} blob but not flagged in_{other}: {unflagged:,}")
    if name == "stack_v3":
        for other in [n for n in fetched if n != name]:
            copies = con.execute(f"""SELECT count(*) FROM read_parquet({fetched[name]}) WHERE content_hash IN
                (SELECT content_hash FROM read_parquet({fetched[other]}))""").fetchone()[0]
            print(f"stack_v3 documents with the exact content of a materialized {other} document: {copies:,}")


def sha256(path):
    """The file's sha256, as the Hub computes an LFS object id."""
    with open(path, "rb") as f:
        return sha_fileobj(f).hex()


def verify_text(con, name, fetched):
    root, files, val = raw_dir(name), fetched[name], list_raw_files(name, "val")
    expected = read_manifest(name)["sha256"]
    with ThreadPoolExecutor() as pool:
        wrong = sum(h != expected[os.path.relpath(f, root)] for f, h in zip(files, pool.map(sha256, files)))
    check(wrong == 0, f"files whose sha256 differs from the pinned Hub file: {wrong} of {len(files)}")
    con.execute(f"""CREATE TABLE docs AS SELECT md5(text) AS hash, length(text) AS characters, list_contains({val}, filename) AS val
        FROM read_parquet({files}, filename = true)""")
    documents, characters, empty, distinct, leaked = con.execute("""SELECT count(*), sum(characters), count(*) FILTER (WHERE coalesce(characters, 0) = 0),
        count(DISTINCT hash), count(*) FILTER (WHERE val AND hash IN (SELECT hash FROM docs WHERE NOT val)) FROM docs""").fetchone()
    print(f"documents: {documents:,} ({characters:,} characters), empty: {empty:,}, exact duplicates: {documents - distinct:,}")
    print(f"val documents whose exact text is also in fetched train: {leaked:,}")
    for other in [n for n in fetched if n != name]:
        copies = con.execute(f"SELECT count(*) FROM docs WHERE hash IN (SELECT md5(text) FROM read_parquet({fetched[other]}))").fetchone()[0]
        print(f"{name} documents with the exact text of a fetched {other} document: {copies:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify a fetched pretraining corpus")
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    verify_units(args.dataset)
    kind = type(DATASETS[args.dataset])
    con = duckdb.connect()
    fetched = {n: files for n, spec in DATASETS.items() if isinstance(spec, kind) and os.path.exists(os.path.join(raw_dir(n), "manifest.json"))
               and (files := list_raw_files(n, "train") + list_raw_files(n, "val"))}
    if kind is CodeCorpus:
        verify_documents(con, args.dataset, fetched[args.dataset])
        verify_overlap(con, args.dataset, fetched)
    else:
        verify_text(con, args.dataset, fetched)
    print("verification FAILED" if failures else "verification passed")
    sys.exit(1 if failures else 0)
