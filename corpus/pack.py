"""
Pack the tokenized corpus into training rows of seq_len + 1 tokens, stage by stage, one curated shard per crate call,
with FIM examples in repository code's train splits; each document is packed at most once.

python -m corpus.pack
"""

import hashlib
import os
from collections.abc import Generator, Iterable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from corpus import mixture
from corpus.layout import (
    PIECE_COLUMNS,
    TOKEN_DTYPE,
    fim_packed,
    grammar,
    packed_path,
    shard_bounds,
    split_path,
)
from corpus.parallel import ordered
from corpus.sources import curated_shards
from corpus.tokenize import repository_parts, shard_tokens
from corpus.tokenizer import FIM_TOKENS, get_tokenizer
from native import corpus_crate

# FIM rate 0.5 (arXiv 2401.14196 §3.1.2); Rand-FIM 0.1, Single-Node 0.5 (arXiv 2506.00204 §5.1, §3.3); line-start
# 0.5 (basis: docs/corpus.md §8); SPM 0.5 (arXiv 2207.14255 §3.1, App. D variant 2)
FIM_RATE, RANDOM_SPAN_RATE, SINGLE_NODE_RATE, LINE_START_RATE, SPM_RATE = 0.5, 0.1, 0.5, 0.5, 0.5
# One pack call per core (the crate releases the GIL)
WINDOW = os.cpu_count()

Shard = tuple[np.ndarray, np.ndarray, np.ndarray]


def pack_shard(task: tuple[str, str, int, int]) -> Shard:
    """Curated shard i of a tokenized split packed into rows of seq_len + 1 tokens: (pieces, pieces per row, FIM
    tokens), the FIM range within the shard's FIM tokens."""
    corpus, split, i, seq_len = task
    tokenizer = get_tokenizer()
    tokenized = split_path(corpus, split, tokenizer)
    bounds = shard_bounds(corpus, split, tokenizer)
    # the corpus in the FIM key: its name's hash, so equal (document, piece) of two corpora draw independently
    fim = ({"rate": FIM_RATE, "random": RANDOM_SPAN_RATE, "single": SINGLE_NODE_RATE, "line_start": LINE_START_RATE,
            "spm": SPM_RATE, "sentinels": [tokenizer.special_tokens[name] for name in FIM_TOKENS],
            "corpus": int.from_bytes(hashlib.sha256(corpus.encode()).digest()[:8], "little")}
           if fim_packed(corpus, split) else None)
    pieces, counts, tokens = corpus_crate().pack(str(tokenized.with_suffix(".bin")),
                                                 str(tokenized.with_suffix(".idx")),
                                                 (int(bounds[i]), int(bounds[i + 1])), row_len=seq_len + 1,
                                                 token_bytes=tokenizer.mergeable,
                                                 pattern=tokenizer.pattern, grammar=grammar(corpus), fim=fim)
    return (np.frombuffer(pieces, np.int64).reshape(-1, PIECE_COLUMNS), np.frombuffer(counts, np.int64),
            np.frombuffer(tokens, TOKEN_DTYPE))


def packed_shards(corpus: str, split: str, shards: Iterable[int], seq_len: int) -> Generator[Shard]:
    """pack_shard of each shard, in order, WINDOW in flight."""
    with ThreadPoolExecutor(WINDOW) as pool:
        yield from ordered(pool, pack_shard, [(corpus, split, i, seq_len) for i in shards], WINDOW)


def write(corpus: str, split: str, shards: Iterable[Shard], seq_len: int) -> int:
    """Packed shards, in order, as <split>-T<seq_len>.pieces, .rows and .fim; returns the rows written."""
    path = packed_path(corpus, split, get_tokenizer(), seq_len)
    placed, fim_tokens, rows = 0, 0, 0
    with (open(path.with_suffix(".pieces"), "wb") as pieces_file, open(path.with_suffix(".rows"), "wb") as rows_file,
          open(path.with_suffix(".fim"), "wb") as fim_file):
        for pieces, counts, tokens in shards:
            located = pieces.copy()
            located[:, 2:] += fim_tokens  # the FIM range within the split's FIM tokens
            pieces_file.write(located)
            rows_file.write(placed + np.cumsum(counts) - counts)  # each row's first piece
            fim_file.write(tokens)
            placed, fim_tokens, rows = placed + len(pieces), fim_tokens + len(tokens), rows + len(counts)
        rows_file.write(np.array([placed], dtype=np.int64))
    return rows


def shard_count(corpus: str, split: str) -> int:
    return len(shard_bounds(corpus, split, get_tokenizer())) - 1


def fit(stage: mixture.Stage, rows: dict[str, int], more: dict[str, Generator[Shard]],
        held: dict[str, list[Shard]]) -> dict[str, float]:
    """The stage's weights once every corpus read in part (more) has packed shards into held to cover its draw."""
    while True:
        shares, _, over = mixture.weights(stage, rows, stage.rows)
        if not over:
            return shares
        for name in over:
            shard = next(more[name])
            held[name].append(shard)
            rows[name] += len(shard[1])


def pack_stage(stage: mixture.Stage, whole: dict[str, list[int]], part: dict[str, list[int]]) -> dict[str, int]:
    """Pack a stage's train splits: whole corpora's listed shards, then corpora read in part until they hold their
    draw (fit). Returns the shards each corpus read in part took."""
    rows = {corpus: write(corpus, "train", packed_shards(corpus, "train", shards, stage.seq_len), stage.seq_len)
            for corpus, shards in whole.items()}
    more = {corpus: packed_shards(corpus, "train", shards, stage.seq_len) for corpus, shards in part.items()}
    held = {corpus: [next(more[corpus])] for corpus in part}
    rows |= {corpus: len(shards[0][1]) for corpus, shards in held.items()}
    shares = fit(stage, rows, more, held)
    for corpus, shards in held.items():
        more[corpus].close()
        write(corpus, "train", shards, stage.seq_len)
    for corpus, count in rows.items():
        print(f"T{stage.seq_len} {corpus}: {count:,} rows, weight {shares[corpus]:.4f}", flush=True)
    return {corpus: len(shards) for corpus, shards in held.items()}


def main() -> None:
    """The corpus's packs, sized by the MAX_STEPS plan (mixture.Stage.rows); val splits in threads beside the stages."""
    corpus_crate()  # built once, before the threads call it
    stages = (mixture.FILE_LEVEL_STAGE, mixture.REPO_LEVEL_STAGE)
    with ThreadPoolExecutor() as threads:
        vals = [threads.submit(write, corpus, "val",
                               packed_shards(corpus, "val", range(shard_count(corpus, "val")), stage.seq_len),
                               stage.seq_len)
                for stage in stages for corpus in stage.sources]
        trailing = set(repository_parts())
        file_whole = {s: [i for i, path in enumerate(curated_shards(s, "train")) if int(path.stem) not in trailing]
                      for s in mixture.REPOSITORY_SOURCES}
        file_whole |= {s: list(range(shard_count(s, "train"))) for s in mixture.FILE_LEVEL_STAGE.supply}
        taken = pack_stage(mixture.FILE_LEVEL_STAGE, file_whole,
                           {s: list(range(shard_count(s, "train"))) for s in mixture.PART})
        repo_whole = {c: list(range(shard_count(c, "train")))
                      for c in [*mixture.REPOSITORIES.values(), *mixture.REPO_LEVEL_STAGE.supply]}
        pack_stage(mixture.REPO_LEVEL_STAGE, repo_whole,
                   {s: list(range(taken[s], shard_count(s, "train")))
                    for s in mixture.PART if s in mixture.REPO_LEVEL_STAGE.sources})
        for val in vals:
            val.result()
    held = sum(shard_tokens(c, "train").sum() for c in mixture.REPOSITORIES.values())
    print(f"repository partitions of the 32K stage: {sorted(trailing)}; its repositories hold {held:,} tokens",
          flush=True)


if __name__ == "__main__":
    main()
