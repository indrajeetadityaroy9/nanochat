"""
Code corpora whose text lives in Software Heritage: stack_edu (its metadata lists each file's blob ID) and
refinecode_stackv2_reconstructed (its metadata lists (repository, path), resolved to blob IDs through
bigcode/the-stack-v2).

build_manifest writes, with DuckDB, once per corpus:
- one row per blob, without Jupyter notebooks (RefineCode trained on them converted to a format its release does not
  specify; Stack-Edu has none);
- val: the whole repositories first in seeded hash order that together hold less than one shard of text, so val is one
  shard and shares no repository with train;
- train: the other files in seeded hash order, cut into shards of SHARD_BYTES, so every shard mixes all languages.
RefineCode rows also record whether the blob is in Stack-Edu (in_stack_edu).

A unit is one shard, its blobs fetched from Software Heritage's public S3 bucket through obstore, whose client retries
failed requests; a request that still fails stops the fetch, which a rerun resumes. A blob is kept if sha1(bytes) is
its ID and it decodes with its src_encoding to text that is non-empty once sanitized (sanitize.py); a blob that is
absent, corrupt or not decodable is dropped. document_id is the blob ID, content_hash the sha1 of the sanitized text.
"""

import os
import gzip
import time
import zlib
import shutil
import hashlib
import tempfile
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import duckdb
import obstore
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from obstore.store import S3Store
from huggingface_hub import snapshot_download

from nanochat.data.storage import get_data_dir, write_json
from nanochat.data.pretrain.sources import DATASETS, DOCUMENT_COLUMNS, SEED, raw_dir
from nanochat.data.pretrain.sanitize import sanitize

SHARD_BYTES = 250_000_000  # text per shard: a ClimbMix shard holds ~250M characters
STACK_V2 = ("bigcode/the-stack-v2", "e565caa3a78c2423bd374333a472b049eb090e47")  # gated: accept its terms on the Hub
EXTRA_COLUMNS = {"stack_edu": ["src_encoding", "score", "int_score"],
                 "refinecode_stackv2_reconstructed": ["src_encoding", "program_lang", "lang", "doc_type"]}


def download(repo, revision, pattern):
    local_dir = os.path.join(get_data_dir(), repo)
    snapshot_download(repo, repo_type="dataset", revision=revision, allow_patterns=pattern, local_dir=local_dir)
    return local_dir


def build_manifest(name):
    t0 = time.time()
    spec = DATASETS[name]
    root = raw_dir(name)
    os.makedirs(root, exist_ok=True)
    record = {"repo": spec.repo, "revision": spec.revision}
    with tempfile.TemporaryDirectory(dir=get_data_dir()) as spill, duckdb.connect(config={"temp_directory": spill}) as con:
        if name == "stack_edu":
            meta = download(spec.repo, spec.revision, "*/train-*.parquet")
            downloads = [meta]
            con.execute(f"""CREATE TABLE files AS SELECT blob_id, language, repo_name AS repository, ltrim(path, '/') AS path,
                length_bytes, license_type, detected_licenses, TRUE AS in_stack_edu, src_encoding, score, int_score
                FROM read_parquet('{meta}/*/train-*.parquet', union_by_name = true)""")
        else:
            if not os.path.exists(os.path.join(raw_dir("stack_edu"), "manifest.parquet")):
                build_manifest("stack_edu")
            meta = download(spec.repo, spec.revision, "data/*.parquet")
            stack_v2 = download(*STACK_V2, "data/*/*.parquet")
            downloads = [meta, stack_v2]
            con.execute(f"""CREATE TABLE files AS SELECT v.blob_id, v.language, m.repo_name AS repository, m.sub_path AS path,
                v.length_bytes, v.license_type, v.detected_licenses, v.blob_id IN (SELECT blob_id FROM
                read_parquet('{raw_dir('stack_edu')}/manifest.parquet')) AS in_stack_edu, v.src_encoding, m.program_lang, m.lang, m.doc_type
                FROM read_parquet('{meta}/data/*.parquet') m
                JOIN read_parquet('{stack_v2}/data/*/*.parquet') v ON v.repo_name = m.repo_name AND v.path = '/' || m.sub_path""")
            record |= {"joined_with": list(STACK_V2),
                       "metadata_rows": con.execute(f"SELECT count(*) FROM read_parquet('{meta}/data/*.parquet')").fetchone()[0],
                       "resolved_rows": con.execute("SELECT count(DISTINCT (repository, path)) FROM files").fetchone()[0]}
        # one row per blob: the same content in several repositories is one document. The whole row breaks ties, as
        # RefineCode lists some (repository, path) twice with different metadata
        con.execute("""CREATE TABLE unique_files AS SELECT * FROM (SELECT * FROM files
            QUALIFY row_number() OVER (PARTITION BY blob_id ORDER BY repository, path, files) = 1) WHERE language != 'Jupyter Notebook'""")
        con.execute(f"""CREATE TABLE val_repos AS SELECT repository FROM (
            SELECT repository, sum(bytes) OVER (ORDER BY md5(repository || '{SEED}') ROWS UNBOUNDED PRECEDING) AS cumulative
            FROM (SELECT repository, sum(length_bytes) AS bytes FROM unique_files GROUP BY repository))
            WHERE cumulative < {SHARD_BYTES}""")
        # the windowed sum is a HUGEINT, whose integer division DuckDB returns as a float: cast the shard index back
        con.execute(f"""COPY (SELECT split, CAST((sum(length_bytes) OVER (PARTITION BY split ORDER BY ordering ROWS UNBOUNDED PRECEDING)
                - length_bytes) // {SHARD_BYTES} AS BIGINT) AS shard, * EXCLUDE (split, ordering)
            FROM (SELECT CASE WHEN repository IN (SELECT repository FROM val_repos) THEN 'val' ELSE 'train' END AS split,
                  md5(blob_id || '{SEED}') AS ordering, * FROM unique_files)
            ORDER BY split, shard, ordering) TO '{root}/manifest.parquet.tmp' (FORMAT parquet, COMPRESSION zstd)""")
        shards = dict(con.execute(f"SELECT split, max(shard) + 1 FROM read_parquet('{root}/manifest.parquet.tmp') GROUP BY split").fetchall())
    os.replace(f"{root}/manifest.parquet.tmp", f"{root}/manifest.parquet")
    write_json(os.path.join(root, "manifest.json"), record | {
        "train": [f"train/shard_{i:05d}" for i in range(shards["train"])],
        "val": [f"val/shard_{i:05d}" for i in range(shards["val"])],
    })
    for path in downloads:  # the manifest holds everything the units need
        shutil.rmtree(path)
    print(f"{name}: manifest of {shards['train']} train and {shards['val']} val shards ({time.time() - t0:.0f}s)", flush=True)


def fetch(store, blob_id, encoding, path):
    """The blob's sanitized text, or None when it is absent, corrupt, does not match its ID or has no text."""
    try:
        data = gzip.decompress(obstore.get(store, "content/" + blob_id).bytes())
        if hashlib.sha1(data).hexdigest() != blob_id:
            return None
        return sanitize(data.decode(encoding), path) or None
    except (FileNotFoundError, gzip.BadGzipFile, EOFError, zlib.error, UnicodeDecodeError, LookupError):
        return None


def shards(name, unit):
    """The unit's one shard: (file, table of the document schema, number of dropped blobs)."""
    split, shard = unit.split("/")[0], int(unit.rsplit("_", 1)[1])
    rows = pq.read_table(os.path.join(raw_dir(name), "manifest.parquet"), filters=[("split", "=", split), ("shard", "=", shard)])
    store = S3Store("softwareheritage", region="us-east-1", skip_signature=True)  # Software Heritage's public bucket
    with ThreadPoolExecutor() as pool:
        texts = pa.array(list(pool.map(partial(fetch, store), rows["blob_id"].to_pylist(), rows["src_encoding"].to_pylist(),
                                       rows["path"].to_pylist())), pa.string())
    kept = rows.append_column("text", texts).filter(pc.is_valid(texts))
    n = kept.num_rows
    table = pa.table({
        "text": kept["text"],
        "source_dataset": pa.repeat(name, n),
        "language": kept["language"],
        "document_id": kept["blob_id"],
        "content_hash": pa.array([hashlib.sha1(t.encode()).hexdigest() for t in kept["text"].to_pylist()], pa.string()),
        "repository": kept["repository"],
        "repository_id": pa.nulls(n, pa.int64()),
        "path": kept["path"],
        "in_stack_edu": kept["in_stack_edu"],
        "in_refinecode": pa.repeat(True, n) if name == "refinecode_stackv2_reconstructed" else pa.nulls(n, pa.bool_()),
        "license_type": kept["license_type"],
        "detected_licenses": kept["detected_licenses"],
    } | {c: kept[c] for c in EXTRA_COLUMNS[name]})
    yield unit + ".parquet", table.select(DOCUMENT_COLUMNS + EXTRA_COLUMNS[name]), rows.num_rows - n
