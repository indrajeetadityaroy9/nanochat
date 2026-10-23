"""
Compile the curated corpus (curate.py) into tokens of the corpus's tokenizer (tokenizer.py): per corpus and split, in
CORPUS/compiled/<corpus>-<tokenizer data key>/, the tokens of every document in curated order, without padding or BOS
(<split>.bin, loader.TOKEN_DTYPE), the offset of each document's first token followed by the total (<split>.idx,
int64), and the number of documents of each curated shard or partition (<split>.json; pack.py packs each one's
documents together).

A corpus is a read source (mixture.READ), its documents its curated rows: a repository source's are its
import-connected samples, each file after a comment with its path (sources.texts). The 32K stage reads two more,
mixture.REPOSITORIES: per language, one document per repository across that language's repository sources
(normalization_doc.md §7.2), from the fewest trailing partitions whose repository documents hold the stage's draw of
repository code (stage_parts), and from partition 0 for val. Of a path that several sources hold in one repository,
the first source's file is kept (mixture.REPOSITORY_SOURCES order, deduplication's), and of a path one source holds
twice, its first row; a repository's files, in path order, are cut into import-connected components (the data crate's
repository_samples), each in dependency order, and the components follow one another in the order of their first file.

Row groups (a source's) and partitions (repositories') are the unit of parallel work; the parent writes their tokens
in task order, so the output does not depend on the number of workers.

python -m nanochat.corpus.compile
"""

import itertools
import json
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from nanochat.corpus import mixture
from nanochat.corpus.loader import TOKEN_DTYPE, split_path
from nanochat.corpus.schema import REPOSITORY_FILE
from nanochat.corpus.sources import SOURCES, curated_dir, texts
from nanochat.corpus.tokenizer import get_tokenizer
from nanochat.native import data_crate


def encode(documents: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(tokens, document lengths) of documents. A special token's text encodes as ordinary text (compiled data hold no
    special id); the tokenizer comes from get_tokenizer's per-process cache."""
    ids = [get_tokenizer().enc.encode_to_numpy(text, disallowed_special=()) for text in documents]
    lengths = np.array([len(doc) for doc in ids], dtype=np.int64)
    return np.concatenate([np.zeros(0, TOKEN_DTYPE), *ids]).astype(TOKEN_DTYPE), lengths


def shards(source: str, split: str) -> list[Path]:
    """The source's curated shards of a split in the order of their number k."""
    return sorted(curated_dir(source, split).iterdir(), key=lambda path: int(path.stem))


def row_group_documents(task: tuple[str, Path, int]) -> tuple[np.ndarray, np.ndarray]:
    """The documents of one curated row group: a repository sample's files are consecutive rows of one sample, which a
    row group holds whole, concatenated in the sample's order."""
    source, path, g = task
    table = pq.ParquetFile(path).read_row_group(g)
    documents = texts(source, table)
    if SOURCES[source].schema is REPOSITORY_FILE:
        starts = np.flatnonzero(np.diff(table["sample"].to_numpy(), prepend=-1)).tolist()
        documents = ["".join(documents[a:b]) for a, b in itertools.pairwise([*starts, len(documents)])]
    return encode(documents)


def repository_documents(task: tuple[str, str, int]) -> tuple[np.ndarray, np.ndarray]:
    """The repositories documents of one partition of one language's repository sources."""
    language, split, part = task
    sources = [s for s in mixture.REPOSITORY_SOURCES if SOURCES[s].language == language]
    schema = pa.schema([("repo_name", pa.string()), ("relative_path", pa.string()), ("content", pa.large_string())])
    tables = []
    for rank, source in enumerate(sources):
        table = pq.read_table(curated_dir(source, split) / f"{part}.parquet", columns=schema.names).cast(schema)
        tables.append(table.append_column("rank", pa.array(np.full(table.num_rows, rank))))
    # sorted so a repository is a zero-copy slice (large_string: 64-bit offsets hold a partition beyond 2 GiB) and the
    # first source's file of each (repository, path) comes first; the rest of each path's files are dropped
    table = pa.concat_tables(tables).sort_by([("repo_name", "ascending"), ("relative_path", "ascending"),
                                              ("rank", "ascending")]).combine_chunks()
    keys = pc.binary_join_element_wise(table["repo_name"], table["relative_path"], "\0").chunk(0)
    table = table.filter(np.concatenate([[True], pc.not_equal(keys[1:], keys[:-1]).to_numpy(zero_copy_only=False)]))
    table = table.combine_chunks()
    paths, contents, files = table["relative_path"], table["content"], texts(sources[0], table)
    documents = []
    for a, b in itertools.pairwise([0, *pc.run_end_encode(table["repo_name"].chunk(0)).run_ends.to_numpy()]):
        components = data_crate().repository_samples(paths.slice(a, b - a).to_pylist(),
                                                     contents.slice(a, b - a).to_pylist(), grammar=language)
        documents.append("".join(files[a + i] for component in components for i in component))
    return encode(documents)


def compile_split(corpus: str, split: str, fn: Callable, tasks: list[list]) -> None:
    """A corpus's split into <split>.bin, .idx and .json: fn of every task in order, tasks grouped by curated shard."""
    path = split_path(corpus, split, get_tokenizer())
    print(f"Compiling {len(tasks)} {split} shards ({sum(map(len, tasks)):,} tasks) of '{corpus}' into {path}.bin",
          flush=True)
    per_shard, offset = [], 0
    with (ProcessPoolExecutor() as pool, open(path.with_suffix(".bin"), "wb") as bin_file,
          open(path.with_suffix(".idx"), "wb") as idx_file):
        idx_file.write(np.zeros(1, np.int64))
        results = pool.map(fn, [task for shard in tasks for task in shard])  # in task order
        for shard in tasks:
            documents = 0
            for tokens, lengths in itertools.islice(results, len(shard)):
                bin_file.write(tokens)
                idx_file.write(offset + np.cumsum(lengths))
                offset += len(tokens)
                documents += len(lengths)
            per_shard.append({"documents": documents})
    path.with_suffix(".json").write_text(json.dumps({"files": per_shard}))


def shard_tokens(corpus: str, split: str) -> np.ndarray:
    """The tokens of each curated shard of a compiled split (its .idx at its .json's shard boundaries)."""
    path = split_path(corpus, split, get_tokenizer())
    offsets = np.memmap(path.with_suffix(".idx"), dtype=np.int64, mode="r")
    documents = np.cumsum([0] + [f["documents"] for f in json.loads(path.with_suffix(".json").read_text())["files"]])
    return np.diff(offsets[documents])


def train_parts() -> list[int]:
    """The repository sources' train partitions, in order."""
    return sorted(int(p.stem) for p in curated_dir(mixture.REPOSITORY_SOURCES[0], "train").iterdir())


def stage_parts() -> list[int]:
    """The fewest trailing train partitions whose repository documents (both languages, after the path rule) hold the
    32K stage's draw of repository code: the code share of its tokens less Codeforces-CoTs', which it reads whole. A
    partition at a time, from the last: each task holds a partition of its language's sources twice at peak."""
    need = mixture.CODE_SHARE * mixture.REPO_TOKENS - shard_tokens("codeforces_cots", "train").sum()
    parts = train_parts()
    with ProcessPoolExecutor(len(mixture.REPOSITORIES)) as pool:
        tokens = (sum(len(ids) for ids, _ in pool.map(repository_documents, [(language, "train", part)
                                                                             for language in mixture.REPOSITORIES]))
                  for part in reversed(parts))
        return parts[-next(k + 1 for k, total in enumerate(itertools.accumulate(tokens)) if total >= need):]


def repository_parts() -> list[int]:
    """The train partitions the compiled repositories corpora hold (stage_parts): the 32K stage's."""
    return train_parts()[-len(shard_tokens(mixture.REPOSITORIES["Python"], "train")):]


def main() -> None:
    tokenizer = get_tokenizer()
    for corpus in mixture.READ:
        # a new output directory per corpus and tokenizer: compile does not resume
        split_path(corpus, "val", tokenizer).parent.mkdir(parents=True)
        for split in ("val", "train"):
            tasks = [[(corpus, path, g) for g in range(pq.read_metadata(path).num_row_groups)]
                     for path in shards(corpus, split)]
            compile_split(corpus, split, row_group_documents, tasks)
    parts = stage_parts()
    for language, corpus in mixture.REPOSITORIES.items():
        split_path(corpus, "val", tokenizer).parent.mkdir(parents=True)
        for split, split_parts in (("val", [0]), ("train", parts)):
            compile_split(corpus, split, repository_documents, [[(language, split, part)] for part in split_parts])
        print(f"{corpus}: train partitions {parts[0]}..{parts[-1]}, {shard_tokens(corpus, 'train').sum():,} tokens",
              flush=True)


if __name__ == "__main__":
    main()
