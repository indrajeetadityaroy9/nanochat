"""
Code corpora whose text lives in Software Heritage: stack_edu (its metadata lists each file's blob ID) and
refinecode_stackv2_reconstructed (its metadata lists (repository, path), resolved to blob IDs through
bigcode/the-stack-v2).

build_manifest writes, with DuckDB, once per corpus, manifest.parquet (the columns of every file except its text,
document_id being the blob ID, plus its split and shard) and manifest.json (the shard files of each split):
- one row per blob, without Jupyter notebooks (RefineCode trained on them converted to a format its release does not
  specify; Stack-Edu has none);
- val: the whole repositories first in md5(repository) order that together hold less than one shard of text, so val is
  one shard and shares no repository with train;
- train: the other files in md5(document_id) order, cut into shards of SHARD_BYTES, so every shard mixes all languages.

A shard is its manifest rows with their text, fetched from Software Heritage's public S3 bucket through obstore, whose
client retries failed requests; a request that still fails stops the fetch, which a rerun resumes. A blob's text is
its original bytes decoded with its src_encoding; a blob the bucket does not hold is dropped.
"""

import os
import gzip
import time
import shutil
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
from nanochat.data.pretrain.sources import DATASETS, raw_dir

SHARD_BYTES = 250_000_000  # text per shard: a ClimbMix shard holds ~250M characters
STACK_V2 = ("bigcode/the-stack-v2", "e565caa3a78c2423bd374333a472b049eb090e47")  # gated: accept its terms on the Hub


def download(repo, revision, pattern):
    local_dir = os.path.join(get_data_dir(), repo)
    snapshot_download(repo, repo_type="dataset", revision=revision, allow_patterns=pattern, local_dir=local_dir)
    return local_dir


def build_manifest(name):
    t0 = time.time()
    spec = DATASETS[name]
    root = raw_dir(name)
    os.makedirs(root, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=get_data_dir()) as spill, duckdb.connect(config={"temp_directory": spill}) as con:
        if name == "stack_edu":
            meta = download(spec.repo, spec.revision, "*/train-*.parquet")
            downloads = [meta]
            con.execute(f"""CREATE TABLE files AS SELECT blob_id AS document_id, language, repo_name AS repository,
                ltrim(path, '/') AS path, length_bytes, license_type, detected_licenses, src_encoding, score, int_score
                FROM read_parquet('{meta}/*/train-*.parquet', union_by_name = true)""")
        else:
            meta = download(spec.repo, spec.revision, "data/*.parquet")
            stack_v2 = download(*STACK_V2, "data/*/*.parquet")
            downloads = [meta, stack_v2]
            con.execute(f"""CREATE TABLE files AS SELECT v.blob_id AS document_id, v.language, m.repo_name AS repository,
                m.sub_path AS path, v.length_bytes, v.license_type, v.detected_licenses, v.src_encoding, m.program_lang, m.lang, m.doc_type
                FROM read_parquet('{meta}/data/*.parquet') m
                JOIN read_parquet('{stack_v2}/data/*/*.parquet') v ON v.repo_name = m.repo_name AND v.path = '/' || m.sub_path""")
        # one row per blob: the same content in several repositories is one document. The whole row breaks ties, as
        # RefineCode lists some (repository, path) twice with different metadata
        con.execute("""CREATE TABLE unique_files AS SELECT * FROM (SELECT * FROM files
            QUALIFY row_number() OVER (PARTITION BY document_id ORDER BY repository, path, files) = 1)
            WHERE language != 'Jupyter Notebook'""")
        con.execute(f"""CREATE TABLE val_repos AS SELECT repository FROM (
            SELECT repository, sum(bytes) OVER (ORDER BY md5(repository) ROWS UNBOUNDED PRECEDING) AS cumulative
            FROM (SELECT repository, sum(length_bytes) AS bytes FROM unique_files GROUP BY repository))
            WHERE cumulative < {SHARD_BYTES}""")
        # the windowed sum is a HUGEINT, whose integer division DuckDB returns as a float: cast the shard index back
        con.execute(f"""COPY (SELECT split, CAST((sum(length_bytes) OVER (PARTITION BY split ORDER BY ordering ROWS UNBOUNDED PRECEDING)
                - length_bytes) // {SHARD_BYTES} AS BIGINT) AS shard, * EXCLUDE (split, ordering)
            FROM (SELECT CASE WHEN repository IN (SELECT repository FROM val_repos) THEN 'val' ELSE 'train' END AS split,
                  md5(document_id) AS ordering, * FROM unique_files)
            ORDER BY split, shard, ordering) TO '{root}/manifest.parquet.tmp' (FORMAT parquet, COMPRESSION zstd)""")
        shards = dict(con.execute(f"SELECT split, max(shard) + 1 FROM read_parquet('{root}/manifest.parquet.tmp') GROUP BY split").fetchall())
    os.replace(f"{root}/manifest.parquet.tmp", f"{root}/manifest.parquet")
    write_json(os.path.join(root, "manifest.json"), {split: [f"{split}/shard_{i:05d}.parquet" for i in range(n)] for split, n in shards.items()})
    for path in downloads:  # the manifest holds everything the shards need
        shutil.rmtree(path)
    print(f"{name}: manifest of {shards['train']} train and {shards['val']} val shards ({time.time() - t0:.0f}s)", flush=True)


def fetch(store, blob_id, encoding):
    """The blob's text, or None when the bucket does not hold it."""
    try:
        return gzip.decompress(obstore.get(store, "content/" + blob_id).bytes()).decode(encoding)
    except FileNotFoundError:
        return None


def table(name, file):
    """The documents of a shard file: its manifest rows with their text, without the blobs the bucket does not hold."""
    split, shard = os.path.dirname(file), int(os.path.basename(file).removeprefix("shard_").removesuffix(".parquet"))
    rows = pq.read_table(os.path.join(raw_dir(name), "manifest.parquet"), filters=[("split", "=", split), ("shard", "=", shard)])
    store = S3Store("softwareheritage", region="us-east-1", skip_signature=True)  # Software Heritage's public bucket
    # requests in flight: the rate is bound by the bucket's latency, not the link; 20 processes of 64 threads fetch
    # 11,128 files/s against 5,025 with Python's default pool of 24 (dev/LOG.md, 2026-09-27)
    with ThreadPoolExecutor(64) as pool:
        texts = pa.array(list(pool.map(partial(fetch, store), rows["document_id"].to_pylist(), rows["src_encoding"].to_pylist())), pa.string())
    return rows.drop_columns(["split", "shard"]).append_column("text", texts).filter(pc.is_valid(texts))
