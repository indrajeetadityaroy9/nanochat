"""
Tokenize the curated corpus into tokens of a vocabulary (tokenizer.get_tokenizer: the corpus's or an arm's): per corpus
and split, in <tokenizer.tokenized>/<corpus>-<data key>/, every document's tokens in curated order, no padding or BOS
(<split>.bin), each document's first offset then the total (<split>.idx, int64), and each curated shard's or
partition's document count (<split>.json; pack.py packs each one's documents together).

A source's documents are its curated rows (a repository source's: its import-connected samples, each file after a path
comment, sources.texts). The 32K stage adds mixture.REPOSITORIES: per language, one document per repository across that
language's repository sources, from the fewest trailing partitions holding the stage's draw
(tokenize_repo_level_stage), and partition 0 for val. A path several sources hold in one repository keeps the first
source's file (mixture.REPOSITORY_SOURCES order), a path one source holds twice its first row; a repository's files are
cut into import-connected components (the crate's topological_sort), each in dependency order, in order of their first
file.

Row groups and partitions are the units of parallel work; as many partitions are in flight as MEMORY holds. The parent
writes tokens in task order, so the output does not depend on the number of workers.

python -m corpus.tokenize
"""

import functools
import itertools
import json
import os
from collections.abc import Iterable, Iterator
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from corpus import mixture
from corpus.layout import TOKEN_DTYPE, shard_bounds, split_path
from corpus.parallel import MEMORY, SPAWN, ordered
from corpus.schema import REPOSITORY_FILE
from corpus.sources import (
    SOURCES,
    curated_dir,
    curated_shards,
    repository_samples,
    sample_starts,
    text_bytes,
    texts,
)
from corpus.tokenizer import TOKENIZER, get_tokenizer

Documents = tuple[np.ndarray, np.ndarray]  # tokens, document lengths
# A repository task's peak memory in its partition's text bytes, an estimate, not measured: the Arrow table and its
# sorted copy, the contents, files and documents as Python strings (up to 4 bytes a character) and their tokens
PEAK = 10


def encode(documents: list[str], tokenizer_path: Path, threads: ThreadPoolExecutor | None = None) -> Documents:
    """(tokens, document lengths) of documents under the vocabulary in `tokenizer_path`, on `threads` if given (tiktoken
    releases the GIL), longest first (LPT scheduling). A special token's text encodes as ordinary text."""
    enc = get_tokenizer(tokenizer_path).enc
    order = sorted(range(len(documents)), key=lambda i: -len(documents[i]))
    ids = [np.zeros(0, TOKEN_DTYPE)] * len(documents)
    for i, tokens in zip(order, (threads.map if threads else map)(
            lambda i: enc.encode_to_numpy(documents[i], disallowed_special=()).astype(TOKEN_DTYPE), order),
            strict=True):
        ids[i] = tokens
    lengths = np.array([len(doc) for doc in ids], dtype=np.int64)
    return np.concatenate([np.zeros(0, TOKEN_DTYPE), *ids]), lengths


def row_group_documents(task: tuple[str, Path, int, Path]) -> Documents:
    """The documents of one curated row group: a repository sample's files are consecutive rows of one sample, which a
    row group holds whole, concatenated in the sample's order."""
    source, path, g, tokenizer_path = task
    table = pq.ParquetFile(path).read_row_group(g)
    documents = texts(source, table)
    if SOURCES[source].category == REPOSITORY_FILE:
        starts = sample_starts(table["sample"].to_numpy())
        documents = ["".join(documents[a:b]) for a, b in itertools.pairwise([*starts, len(documents)])]
    return encode(documents, tokenizer_path)


def language_sources(language: str) -> list[str]:
    """A language's repository sources, in mixture.REPOSITORY_SOURCES order."""
    return [s for s in mixture.REPOSITORY_SOURCES if SOURCES[s].language == language]


def repository_documents(task: tuple[str, str, int, Path]) -> Documents:
    """The repository documents of one partition of one language's repository sources: its samples cut by one crate
    call, its documents encoded on a thread per core."""
    language, split, part, tokenizer_path = task
    sources = language_sources(language)
    schema = pa.schema([("repo_name", pa.string()), ("relative_path", pa.string()), ("content", pa.large_string())])
    table = pa.concat_tables(
        table.append_column("rank", pa.array(np.full(table.num_rows, rank)))
        for rank, table in enumerate(pq.read_table(curated_dir(source, split) / f"{part}.parquet",
                                                   columns=schema.names).cast(schema) for source in sources))
    # the first source's file of each (repository, path) sorts first and is kept
    table, ends, samples = repository_samples(table, ("relative_path", "rank"), language, one_per_path=True)
    files = texts(sources[0], table)
    del table
    files = [files[i] for sample in samples for i in sample]  # a repository's samples permute its rows
    documents = ["".join(files[a:b]) for a, b in itertools.pairwise([0, *ends])]
    del files
    with ThreadPoolExecutor(os.cpu_count()) as threads:
        return encode(documents, tokenizer_path, threads)


def tokenize_split(corpus: str, split: str, tokenizer_path: Path, shards: list[int], results: Iterable[Documents]
                   ) -> None:
    """A corpus's split into <split>.bin, .idx and .json: the documents of results in order, shards[k] of them curated
    shard k's."""
    path = split_path(corpus, split, get_tokenizer(tokenizer_path))
    print(f"Tokenizing {len(shards)} {split} shards ({sum(shards):,} tasks) of '{corpus}' into {path}.bin", flush=True)
    per_shard, offset, results = [], 0, iter(results)
    with open(path.with_suffix(".bin"), "wb") as bin_file, open(path.with_suffix(".idx"), "wb") as idx_file:
        idx_file.write(np.zeros(1, np.int64))
        for tasks in shards:
            documents = 0
            for tokens, lengths in itertools.islice(results, tasks):
                bin_file.write(tokens)
                idx_file.write(offset + np.cumsum(lengths))
                offset += len(tokens)
                documents += len(lengths)
            per_shard.append({"documents": documents})
    path.with_suffix(".json").write_text(json.dumps({"files": per_shard}))


def tokenize_source(source: str, split: str, tokenizer_path: Path, pool: Executor, leading: int | None = None) -> None:
    """A source's curated split, or its `leading` curated shards, a row group per task."""
    tasks = [[(source, path, g, tokenizer_path) for g in range(pq.read_metadata(path).num_row_groups)]
             for path in curated_shards(source, split)[:leading]]
    tokenize_split(source, split, tokenizer_path, [len(shard) for shard in tasks],
                  pool.map(row_group_documents, [task for shard in tasks for task in shard]))  # in task order


def shard_tokens(corpus: str, split: str, tokenizer_path: Path) -> np.ndarray:
    """The tokens of each curated shard of a tokenized split (its .idx at its shards' document bounds)."""
    tokenizer = get_tokenizer(tokenizer_path)
    offsets = np.memmap(split_path(corpus, split, tokenizer).with_suffix(".idx"), dtype=np.int64, mode="r")
    return np.diff(offsets[shard_bounds(corpus, split, tokenizer)])


def train_parts() -> list[int]:
    """The repository sources' train partitions, in order."""
    return [int(path.stem) for path in curated_shards(mixture.REPOSITORY_SOURCES[0], "train")]


def repository_tasks(parts: list[int]) -> list[tuple[str, str, int]]:
    """The repositories corpora's (language, split, partition) in write order: per language (mixture.REPOSITORIES
    order), val partition 0, then train partitions `parts`."""
    return [(language, split, part) for language in mixture.REPOSITORIES
            for split, split_parts in (("val", [0]), ("train", parts)) for part in split_parts]


@functools.cache
def repository_window() -> int:
    """Repository tasks in flight: as many as MEMORY holds at PEAK times the largest partition's text (one language's
    sources)."""
    largest = max(sum(text_bytes(pq.read_metadata(curated_dir(s, split) / f"{part}.parquet"), "content")
                      for s in language_sources(language))
                  for language, split, part in repository_tasks(train_parts()))
    return max(1, MEMORY // (PEAK * largest))


def write_repositories(tokenizer_path: Path, parts: list[int], results: Iterator[Documents]) -> None:
    """The repositories corpora's splits from results, in repository_tasks(parts) order."""
    tokenizer = get_tokenizer(tokenizer_path)
    for corpus in mixture.REPOSITORIES.values():
        split_path(corpus, "val", tokenizer).parent.mkdir(parents=True)
        tokenize_split(corpus, "val", tokenizer_path, [1], results)
        tokenize_split(corpus, "train", tokenizer_path, [1] * len(parts), results)
        print(f"{corpus}: train partitions {parts[0]}..{parts[-1]}, "
              f"{shard_tokens(corpus, 'train', tokenizer_path).sum():,} tokens", flush=True)


def tokenize_repositories(tokenizer_path: Path, parts: list[int], pool: Executor) -> None:
    """The repositories corpora (mixture.REPOSITORIES) of train partitions `parts` and of val partition 0, every
    (language, split, partition) a task, repository_window() in flight."""
    tasks = [(*task, tokenizer_path) for task in repository_tasks(parts)]
    write_repositories(tokenizer_path, parts, ordered(pool, repository_documents, tasks, repository_window()))


def tokenize_repo_level_stage(tokenizer_path: Path, pool: Executor, need: float) -> list[int]:
    """tokenize_repositories of the fewest trailing train partitions whose repository documents (both languages, after
    the path rule) hold `need` tokens, the 32K stage's; returns them. Val partition 0, then the train partitions from
    the last, both languages each, repository_window() in flight, are taken in order until the train partitions hold
    `need`, and written from the same tokens."""
    parts, languages = train_parts(), list(mixture.REPOSITORIES)
    tasks = [(language, "val", 0, tokenizer_path) for language in languages]
    tasks += [(language, "train", part, tokenizer_path) for part in reversed(parts) for language in languages]
    results = ordered(pool, repository_documents, tasks, repository_window())
    held, total = {}, 0
    for (language, split, part, _), documents in zip(tasks, results, strict=False):
        held[language, split, part] = documents
        total += len(documents[0]) if split == "train" else 0
        if split == "train" and language == languages[-1] and total >= need:
            break
    results.close()
    if total < need:
        raise RuntimeError(f"the repository partitions hold {total:,} of the {need:,.0f} tokens of the 32K stage")
    chosen = parts[-(len(held) // len(languages) - 1):]
    write_repositories(tokenizer_path, chosen, (held.pop(task) for task in repository_tasks(chosen)))
    return chosen


def repository_parts(tokenizer_path: Path) -> list[int]:
    """The train partitions the tokenized repositories corpora hold (tokenize_repo_level_stage): the 32K stage's."""
    return train_parts()[-len(shard_tokens(mixture.REPOSITORIES["Python"], "train", tokenizer_path)):]


def main() -> None:
    """The corpus's tokenizer's tokenized corpus: the sources' splits in a pool of a worker per core, then the 32K
    stage's repositories corpora in a pool of repository_window() workers, so idle workers hold no more than the tasks
    in flight."""
    tokenizer = get_tokenizer(TOKENIZER)
    jobs = list(itertools.product(SOURCES, ("val", "train")))
    with ProcessPoolExecutor(mp_context=SPAWN) as pool, ThreadPoolExecutor(len(jobs)) as threads:
        for corpus in SOURCES:
            # a new output directory per corpus and tokenizer: tokenize does not resume
            split_path(corpus, "val", tokenizer).parent.mkdir(parents=True)
        # every split at once: the pool runs their row groups in submission order, so no core waits on a small split
        list(threads.map(lambda job: tokenize_source(job[0], job[1], TOKENIZER, pool), jobs))
    # the 32K stage's draw of repository code: the code share of its tokens less its supply sources'
    # (Codeforces-CoTs), which it reads whole
    need = mixture.CODE_SHARE * mixture.REPO_LEVEL_STAGE.tokens - sum(shard_tokens(s, "train", TOKENIZER).sum()
                                                                     for s in mixture.REPO_LEVEL_STAGE.supply)
    with ProcessPoolExecutor(repository_window(), mp_context=SPAWN) as pool:
        tokenize_repo_level_stage(TOKENIZER, pool, need)


if __name__ == "__main__":
    main()
