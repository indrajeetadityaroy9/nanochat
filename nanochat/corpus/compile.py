"""
Compile the curated corpus (curate.py) into tokens of a vocabulary, named by its directory (tokenizer.get_tokenizer:
the corpus's tokenizer or a tokenizer ablation arm): per corpus and split, in <tokenizer.compiled>/<corpus>-<data key>/,
the tokens of every document in curated order, without padding or BOS (<split>.bin, loader.TOKEN_DTYPE), the offset of
each document's first token followed by the total (<split>.idx, int64), and the number of documents of each curated
shard or partition (<split>.json; pack.py packs each one's documents together).

A corpus is a source, its documents its curated rows: a repository source's are its import-connected samples, each
file after a comment with its path (sources.texts). The 32K stage reads two more, mixture.REPOSITORIES: per language,
one document per repository across that language's repository sources (normalization_doc.md §7.2), from the fewest
trailing partitions whose repository documents hold the stage's draw of repository code (compile_stage), and from
partition 0 for val. Of a path that several sources hold in one repository, the first source's file is kept
(mixture.REPOSITORY_SOURCES order, deduplication's), and of a path one source holds twice, its first row; a
repository's files, in path order, are cut into import-connected components (the data crate's repository_samples),
each in dependency order, and the components follow one another in the order of their first file.

Row groups (a source's) and partitions (repositories', per language) are the unit of parallel work in a process pool;
a partition's samples are cut and its documents encoded on every core (the crate's rayon, tiktoken in threads), and
as many partitions are in flight as MEMORY holds. The parent writes the tokens in task order, so the output does not
depend on the number of workers.

python -m nanochat.corpus.compile
"""

import itertools
import json
import os
from collections.abc import Iterable, Iterator
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from nanochat.corpus import mixture
from nanochat.corpus.loader import TOKEN_DTYPE, shard_bounds, split_path
from nanochat.corpus.parallel import MEMORY, SPAWN, ordered
from nanochat.corpus.schema import REPOSITORY_FILE
from nanochat.corpus.sources import SOURCES, curated_dir, curated_shards, text_bytes, texts
from nanochat.corpus.tokenizer import TOKENIZER, get_tokenizer
from nanochat.native import data_crate

Documents = tuple[np.ndarray, np.ndarray]  # tokens, document lengths
# A repository task's peak memory in its partition's text bytes: the Arrow table and its sorted copy, the contents,
# files and documents as Python strings (up to 4 bytes a character) and their tokens
PEAK = 10


def encode(documents: list[str], vocabulary: Path, threads: ThreadPoolExecutor | None = None) -> Documents:
    """(tokens, document lengths) of documents under the vocabulary in directory `vocabulary`, on `threads` if given
    (tiktoken releases the GIL as it encodes), longest first so the longest does not start last (LPT scheduling). A
    special token's text encodes as ordinary text (compiled data hold no special id); the tokenizer comes from
    get_tokenizer's per-process cache."""
    enc = get_tokenizer(vocabulary).enc
    order = sorted(range(len(documents)), key=lambda i: -len(documents[i]))
    ids = [np.zeros(0, TOKEN_DTYPE)] * len(documents)
    for i, tokens in zip(order, (threads.map if threads else map)(
            lambda i: enc.encode_to_numpy(documents[i], disallowed_special=()).astype(TOKEN_DTYPE), order), strict=True):
        ids[i] = tokens
    lengths = np.array([len(doc) for doc in ids], dtype=np.int64)
    return np.concatenate([np.zeros(0, TOKEN_DTYPE), *ids]), lengths


def row_group_documents(task: tuple[str, Path, int, Path]) -> Documents:
    """The documents of one curated row group: a repository sample's files are consecutive rows of one sample, which a
    row group holds whole, concatenated in the sample's order."""
    source, path, g, vocabulary = task
    table = pq.ParquetFile(path).read_row_group(g)
    documents = texts(source, table)
    if SOURCES[source].category == REPOSITORY_FILE:
        starts = np.flatnonzero(np.diff(table["sample"].to_numpy(), prepend=-1)).tolist()
        documents = ["".join(documents[a:b]) for a, b in itertools.pairwise([*starts, len(documents)])]
    return encode(documents, vocabulary)


def language_sources(language: str) -> list[str]:
    """A language's repository sources, in mixture.REPOSITORY_SOURCES order."""
    return [s for s in mixture.REPOSITORY_SOURCES if SOURCES[s].language == language]


def repository_documents(task: tuple[str, str, int, Path]) -> Documents:
    """The repository documents of one partition of one language's repository sources: its samples cut by one crate
    call, its documents encoded on a thread per core."""
    language, split, part, vocabulary = task
    sources = language_sources(language)
    schema = pa.schema([("repo_name", pa.string()), ("relative_path", pa.string()), ("content", pa.large_string())])
    table = pa.concat_tables(
        table.append_column("rank", pa.array(np.full(table.num_rows, rank)))
        for rank, table in enumerate(pq.read_table(curated_dir(source, split) / f"{part}.parquet",
                                                   columns=schema.names).cast(schema) for source in sources))
    # sorted so a repository (its lowercased name) is a run of rows (large_string: 64-bit offsets hold a partition
    # beyond 2 GiB) and the first source's file of each (repository, path) comes first; the rest of each path's files
    # are dropped
    table = table.append_column("repository", pc.utf8_lower(table["repo_name"]))
    table = table.sort_by([("repository", "ascending"), ("relative_path", "ascending"),
                           ("rank", "ascending")]).combine_chunks()
    keys = pc.binary_join_element_wise(table["repository"], table["relative_path"], "\0").chunk(0)
    table = table.filter(np.concatenate([[True], pc.not_equal(keys[1:], keys[:-1]).to_numpy(zero_copy_only=False)]))
    table = table.combine_chunks()
    ends = pc.run_end_encode(table["repository"].chunk(0)).run_ends.to_numpy().tolist()
    samples = data_crate().repository_samples(table["relative_path"].to_pylist(), table["content"].to_pylist(), ends,
                                              grammar=language)
    files = texts(sources[0], table)
    del table
    files = [files[i] for sample in samples for i in sample]  # a repository's samples permute its rows
    documents = ["".join(files[a:b]) for a, b in itertools.pairwise([0, *ends])]
    del files
    with ThreadPoolExecutor(os.cpu_count()) as threads:
        return encode(documents, vocabulary, threads)


def compile_split(corpus: str, split: str, vocabulary: Path, shards: list[int], results: Iterable[Documents]) -> None:
    """A corpus's split into <split>.bin, .idx and .json: the documents of results in order, shards[k] of them curated
    shard k's."""
    path = split_path(corpus, split, get_tokenizer(vocabulary))
    print(f"Compiling {len(shards)} {split} shards ({sum(shards):,} tasks) of '{corpus}' into {path}.bin", flush=True)
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


def compile_source(source: str, split: str, vocabulary: Path, pool: Executor, leading: int | None = None) -> None:
    """A source's curated split, or its `leading` curated shards, a row group per task."""
    tasks = [[(source, path, g, vocabulary) for g in range(pq.read_metadata(path).num_row_groups)]
             for path in curated_shards(source, split)[:leading]]
    compile_split(source, split, vocabulary, [len(shard) for shard in tasks],
                  pool.map(row_group_documents, [task for shard in tasks for task in shard]))  # in task order


def shard_tokens(corpus: str, split: str, vocabulary: Path) -> np.ndarray:
    """The tokens of each curated shard of a compiled split (its .idx at its shards' document bounds)."""
    tokenizer = get_tokenizer(vocabulary)
    offsets = np.memmap(split_path(corpus, split, tokenizer).with_suffix(".idx"), dtype=np.int64, mode="r")
    return np.diff(offsets[shard_bounds(corpus, split, tokenizer)])


def train_parts() -> list[int]:
    """The repository sources' train partitions, in order."""
    return [int(path.stem) for path in curated_shards(mixture.REPOSITORY_SOURCES[0], "train")]


def repository_window() -> int:
    """Repository tasks in flight: as many as MEMORY holds at PEAK times the largest partition's text (one language's
    sources)."""
    largest = max(sum(text_bytes(pq.read_metadata(curated_dir(s, split) / f"{part}.parquet"), "content")
                      for s in language_sources(language))
                  for language in mixture.REPOSITORIES for split, parts in (("val", [0]), ("train", train_parts()))
                  for part in parts)
    return max(1, MEMORY // (PEAK * largest))


def write_repositories(vocabulary: Path, parts: list[int], results: Iterator[Documents]) -> None:
    """The repositories corpora's splits from results, per language (mixture.REPOSITORIES order) the documents of val
    partition 0, then of train partitions `parts`."""
    tokenizer = get_tokenizer(vocabulary)
    for corpus in mixture.REPOSITORIES.values():
        split_path(corpus, "val", tokenizer).parent.mkdir(parents=True)
        compile_split(corpus, "val", vocabulary, [1], results)
        compile_split(corpus, "train", vocabulary, [1] * len(parts), results)
        print(f"{corpus}: train partitions {parts[0]}..{parts[-1]}, "
              f"{shard_tokens(corpus, 'train', vocabulary).sum():,} tokens", flush=True)


def compile_repositories(vocabulary: Path, parts: list[int], pool: Executor) -> None:
    """The repositories corpora (mixture.REPOSITORIES) of train partitions `parts` and of val partition 0, every
    (language, split, partition) a task, repository_window() in flight."""
    tasks = [(language, split, part, vocabulary) for language in mixture.REPOSITORIES
             for split, split_parts in (("val", [0]), ("train", parts)) for part in split_parts]
    write_repositories(vocabulary, parts, ordered(pool, repository_documents, tasks, repository_window()))


def compile_stage(vocabulary: Path, pool: Executor, need: float) -> list[int]:
    """compile_repositories of the fewest trailing train partitions whose repository documents (both languages, after
    the path rule) hold `need` tokens, the 32K stage's; returns them. Val partition 0, then the train partitions from
    the last, both languages each, repository_window() in flight, are taken in order until the train partitions hold
    `need`, and written from the same tokens."""
    parts, languages = train_parts(), list(mixture.REPOSITORIES)
    tasks = [(language, "val", 0, vocabulary) for language in languages]
    tasks += [(language, "train", part, vocabulary) for part in reversed(parts) for language in languages]
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
    write_repositories(vocabulary, chosen, (held.pop((language, split, part)) for language in languages
                                            for split, split_parts in (("val", [0]), ("train", chosen))
                                            for part in split_parts))
    return chosen


def repository_parts(vocabulary: Path) -> list[int]:
    """The train partitions the compiled repositories corpora hold (compile_stage): the 32K stage's."""
    return train_parts()[-len(shard_tokens(mixture.REPOSITORIES["Python"], "train", vocabulary)):]


def main() -> None:
    """The corpus's tokenizer's compiled corpus: the sources' splits in a pool of a worker per core, then the 32K
    stage's repositories corpora in a pool of repository_window() workers, so idle workers hold no more than the tasks
    in flight."""
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
    with ProcessPoolExecutor(repository_window(), mp_context=SPAWN) as pool:
        compile_stage(TOKENIZER, pool, need)


if __name__ == "__main__":
    main()
