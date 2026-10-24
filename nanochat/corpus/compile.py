"""
Compile the curated corpus (curate.py) into tokens of a vocabulary, named by its directory (tokenizer.get_tokenizer:
the corpus's tokenizer or a tokenizer ablation arm): per corpus and split, in <tokenizer.compiled>/<corpus>-<data key>/,
the tokens of every document in curated order, without padding or BOS (<split>.bin, loader.TOKEN_DTYPE), the offset of
each document's first token followed by the total (<split>.idx, int64), and the number of documents of each curated
shard or partition (<split>.json; pack.py packs each one's documents together).

A corpus is a source, its documents its curated rows: a repository source's are its import-connected samples, each
file after a comment with its path (sources.texts). The 32K stage reads two more, mixture.REPOSITORIES: per language,
one document per repository across that language's repository sources (normalization_doc.md §7.2), from the fewest
trailing partitions whose repository documents hold the stage's draw of repository code (stage_parts), and from
partition 0 for val. Of a path that several sources hold in one repository, the first source's file is kept
(mixture.REPOSITORY_SOURCES order, deduplication's), and of a path one source holds twice, its first row; a
repository's files, in path order, are cut into import-connected components (the data crate's repository_samples),
each in dependency order, and the components follow one another in the order of their first file.

Row groups (a source's) and partitions (repositories') are the unit of parallel work, in one process pool for every
split; the parent writes their tokens in task order, so the output does not depend on the number of workers.

python -m nanochat.corpus.compile
"""

import itertools
import json
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from nanochat.corpus import mixture
from nanochat.corpus.loader import TOKEN_DTYPE, shard_bounds, split_path
from nanochat.corpus.parallel import SPAWN
from nanochat.corpus.schema import REPOSITORY_FILE
from nanochat.corpus.sources import SOURCES, curated_dir, curated_shards, texts
from nanochat.corpus.tokenizer import TOKENIZER, get_tokenizer
from nanochat.native import data_crate


def encode(documents: list[str], vocabulary: Path) -> tuple[np.ndarray, np.ndarray]:
    """(tokens, document lengths) of documents under the vocabulary in directory `vocabulary`. A special token's text
    encodes as ordinary text (compiled data hold no special id); the tokenizer comes from get_tokenizer's per-process
    cache."""
    ids = [get_tokenizer(vocabulary).enc.encode_to_numpy(text, disallowed_special=()) for text in documents]
    lengths = np.array([len(doc) for doc in ids], dtype=np.int64)
    return np.concatenate([np.zeros(0, TOKEN_DTYPE), *ids]).astype(TOKEN_DTYPE), lengths


def row_group_documents(task: tuple[str, Path, int, Path]) -> tuple[np.ndarray, np.ndarray]:
    """The documents of one curated row group: a repository sample's files are consecutive rows of one sample, which a
    row group holds whole, concatenated in the sample's order."""
    source, path, g, vocabulary = task
    table = pq.ParquetFile(path).read_row_group(g)
    documents = texts(source, table)
    if SOURCES[source].category == REPOSITORY_FILE:
        starts = np.flatnonzero(np.diff(table["sample"].to_numpy(), prepend=-1)).tolist()
        documents = ["".join(documents[a:b]) for a, b in itertools.pairwise([*starts, len(documents)])]
    return encode(documents, vocabulary)


def repository_documents(task: tuple[str, str, int, Path]) -> tuple[np.ndarray, np.ndarray]:
    """The repository documents of one partition of one language's repository sources."""
    language, split, part, vocabulary = task
    sources = [s for s in mixture.REPOSITORY_SOURCES if SOURCES[s].language == language]
    schema = pa.schema([("repo_name", pa.string()), ("relative_path", pa.string()), ("content", pa.large_string())])
    tables = []
    for rank, source in enumerate(sources):
        table = pq.read_table(curated_dir(source, split) / f"{part}.parquet", columns=schema.names).cast(schema)
        tables.append(table.append_column("rank", pa.array(np.full(table.num_rows, rank))))
    # sorted so a repository (its lowercased name) is a zero-copy slice (large_string: 64-bit offsets hold a partition
    # beyond 2 GiB) and the first source's file of each (repository, path) comes first; the rest of each path's files
    # are dropped
    table = pa.concat_tables(tables)
    table = table.append_column("repository", pc.utf8_lower(table["repo_name"]))
    table = table.sort_by([("repository", "ascending"), ("relative_path", "ascending"),
                           ("rank", "ascending")]).combine_chunks()
    keys = pc.binary_join_element_wise(table["repository"], table["relative_path"], "\0").chunk(0)
    table = table.filter(np.concatenate([[True], pc.not_equal(keys[1:], keys[:-1]).to_numpy(zero_copy_only=False)]))
    table = table.combine_chunks()
    paths, contents, files = table["relative_path"], table["content"], texts(sources[0], table)
    documents = []
    for a, b in itertools.pairwise([0, *pc.run_end_encode(table["repository"].chunk(0)).run_ends.to_numpy()]):
        components = data_crate().repository_samples(paths.slice(a, b - a).to_pylist(),
                                                     contents.slice(a, b - a).to_pylist(), grammar=language)
        documents.append("".join(files[a + i] for component in components for i in component))
    return encode(documents, vocabulary)


def compile_split(corpus: str, split: str, vocabulary: Path, fn: Callable, tasks: list[list], pool: Executor) -> None:
    """A corpus's split into <split>.bin, .idx and .json: fn of every task in order, tasks grouped by curated shard."""
    path = split_path(corpus, split, get_tokenizer(vocabulary))
    print(f"Compiling {len(tasks)} {split} shards ({sum(map(len, tasks)):,} tasks) of '{corpus}' into {path}.bin",
          flush=True)
    per_shard, offset = [], 0
    with open(path.with_suffix(".bin"), "wb") as bin_file, open(path.with_suffix(".idx"), "wb") as idx_file:
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


def compile_source(source: str, split: str, vocabulary: Path, pool: Executor, leading: int | None = None) -> None:
    """A source's curated split, or its `leading` curated shards, a row group per task."""
    compile_split(source, split, vocabulary, row_group_documents,
                  [[(source, path, g, vocabulary) for g in range(pq.read_metadata(path).num_row_groups)]
                   for path in curated_shards(source, split)[:leading]], pool)


def shard_tokens(corpus: str, split: str, vocabulary: Path) -> np.ndarray:
    """The tokens of each curated shard of a compiled split (its .idx at its shards' document bounds)."""
    tokenizer = get_tokenizer(vocabulary)
    offsets = np.memmap(split_path(corpus, split, tokenizer).with_suffix(".idx"), dtype=np.int64, mode="r")
    return np.diff(offsets[shard_bounds(corpus, split, tokenizer)])


def train_parts() -> list[int]:
    """The repository sources' train partitions, in order."""
    return [int(path.stem) for path in curated_shards(mixture.REPOSITORY_SOURCES[0], "train")]


def stage_parts(vocabulary: Path, pool: Executor, need: float) -> list[int]:
    """The fewest trailing train partitions whose repository documents (both languages, after the path rule) hold
    `need` tokens. A partition at a time, from the last, its two languages at once: each task holds a partition of its
    language's sources twice at peak."""
    parts = train_parts()
    tokens = (sum(len(ids) for ids, _ in pool.map(repository_documents, [(language, "train", part, vocabulary)
                                                                         for language in mixture.REPOSITORIES]))
              for part in reversed(parts))
    return parts[-next(k + 1 for k, total in enumerate(itertools.accumulate(tokens)) if total >= need):]


def compile_repositories(vocabulary: Path, parts: list[int], pool: Executor) -> None:
    """The repositories corpora (mixture.REPOSITORIES) of train partitions `parts` and of val partition 0."""
    tokenizer = get_tokenizer(vocabulary)
    for language, corpus in mixture.REPOSITORIES.items():
        split_path(corpus, "val", tokenizer).parent.mkdir(parents=True)
        for split, split_parts in (("val", [0]), ("train", parts)):
            compile_split(corpus, split, vocabulary, repository_documents,
                          [[(language, split, part, vocabulary)] for part in split_parts], pool)
        print(f"{corpus}: train partitions {parts[0]}..{parts[-1]}, "
              f"{shard_tokens(corpus, 'train', vocabulary).sum():,} tokens", flush=True)


def repository_parts(vocabulary: Path) -> list[int]:
    """The train partitions the compiled repositories corpora hold (stage_parts): the 32K stage's."""
    return train_parts()[-len(shard_tokens(mixture.REPOSITORIES["Python"], "train", vocabulary)):]


def main() -> None:
    """The corpus's tokenizer's compiled corpus, one process pool for every split."""
    tokenizer = get_tokenizer(TOKENIZER)
    jobs = list(itertools.product(SOURCES, ("val", "train")))
    with ProcessPoolExecutor(mp_context=SPAWN) as pool, ThreadPoolExecutor(len(jobs)) as threads:
        for corpus in SOURCES:
            # a new output directory per corpus and tokenizer: compile does not resume
            split_path(corpus, "val", tokenizer).parent.mkdir(parents=True)
        # every split at once: the pool runs their row groups in submission order, so no core waits on a small split
        list(threads.map(lambda job: compile_source(job[0], job[1], TOKENIZER, pool), jobs))
        # the 32K stage's draw of repository code: the code share of its tokens less its supply sources'
        # (Codeforces-CoTs), which it reads whole
        need = mixture.CODE_SHARE * mixture.REPO_STAGE.tokens - sum(shard_tokens(s, "train", TOKENIZER).sum()
                                                                   for s in mixture.REPO_STAGE.supply)
        compile_repositories(TOKENIZER, stage_parts(TOKENIZER, pool, need), pool)


if __name__ == "__main__":
    main()
