"""
Pack the tokenized corpus into training rows of seq_len + 1 tokens, stage by stage: the crate's pack, one curated shard
per call (Best-Fit-Decreasing, each piece its own attention segment, long documents cut between statements), with
FIM examples in repository code's train splits. Output per corpus and stage:
<split>-T<seq_len>.pieces, .rows and .fim beside the tokenized split (layout.py).

A stage's train packs hold at least the rows the base model's run reads (mixture.MAX_STEPS), each document at most once:
- file stage: the repository partitions before the 32K stage's, and the supply sources, whole; a source read in part
  (mixture.IN_PART), its leading shards;
- repository stage: the repositories corpora and Codeforces-CoTs, whole; ClimbMix and UltraData-Math L3, the shards
  after the file stage's.
A source read in part takes shards until the stage's weights (mixture.weights) draw no more than it holds. Val splits
are packed whole at each reading stage's seq_len.

python -m corpus.pack
"""

import os
from collections.abc import Generator, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
from corpus.mixture import weights
from corpus.parallel import ordered
from corpus.sources import curated_shards
from corpus.tokenize import repository_parts, shard_tokens
from corpus.tokenizer import FIM_TOKENS, TOKENIZER, get_tokenizer
from native import corpus_crate

# FIM (the crate's Fim): a piece becomes an example at rate 0.5 (DeepSeek-Coder, arXiv 2401.14196 §3.1.2; arXiv
# 2207.14255 §3); its middle a random character span at 0.1, else AST-FIM's (arXiv 2506.00204 §5.1), Single-Node or
# Aligned-Span masking at 0.5 each (§3.3); an AST middle whose node starts its line takes the line's indentation at
# 0.5, so the prefix ends in the line break as in line-level infilling (arXiv 2207.14255 §2.2); SPM at 0.5, else PSM
# (arXiv 2207.14255 §3.1, App. D variant 2; arXiv 2506.00204 §5.1). The crate draws each piece's FIM from ChaCha8 keyed
# by document and piece, so packs do not depend on the order of work
FIM_RATE, RANDOM_SPAN_RATE, SINGLE_NODE_RATE, LINE_START_RATE, SPM_RATE = 0.5, 0.1, 0.5, 0.5, 0.5
# Pack calls in flight: one per core (the crate releases the GIL)
WINDOW = os.cpu_count()

Shard = tuple[np.ndarray, np.ndarray, np.ndarray]


def pack_shard(task: tuple[str, str, int, int, Path]) -> Shard:
    """Curated shard i of a tokenized split of the vocabulary in directory `tokenizer_path` packed into rows of
    seq_len + 1 tokens: its pieces (PIECE_COLUMNS int64 each, the FIM range within the shard's FIM tokens), the pieces
    of each row, the shard's FIM tokens."""
    corpus, split, i, seq_len, tokenizer_path = task
    tokenizer = get_tokenizer(tokenizer_path)
    tokenized = split_path(corpus, split, tokenizer)
    bounds = shard_bounds(corpus, split, tokenizer)
    fim = ({"rate": FIM_RATE, "random": RANDOM_SPAN_RATE, "single": SINGLE_NODE_RATE, "line_start": LINE_START_RATE,
            "spm": SPM_RATE,
            "sentinels": [tokenizer.special_tokens[name] for name in FIM_TOKENS]}
           if fim_packed(corpus, split) else None)
    pieces, counts, tokens = corpus_crate().pack(str(tokenized.with_suffix(".bin")),
                                                 str(tokenized.with_suffix(".idx")),
                                                 (int(bounds[i]), int(bounds[i + 1])), row_len=seq_len + 1,
                                                 token_bytes=tokenizer.token_bytes[:tokenizer.bos_token_id],
                                                 pattern=tokenizer.pattern, grammar=grammar(corpus), fim=fim)
    return (np.frombuffer(pieces, np.int64).reshape(-1, PIECE_COLUMNS), np.frombuffer(counts, np.int64),
            np.frombuffer(tokens, TOKEN_DTYPE))


def packed_shards(corpus: str, split: str, tokenizer_path: Path, shards: Iterable[int], seq_len: int
                  ) -> Generator[Shard]:
    """pack_shard of each shard, in order, WINDOW in flight; a shard is packed only as the window reaches it."""
    with ThreadPoolExecutor(WINDOW) as pool:
        yield from ordered(pool, pack_shard, [(corpus, split, i, seq_len, tokenizer_path) for i in shards], WINDOW)


def write(corpus: str, split: str, tokenizer_path: Path, shards: Iterable[Shard], seq_len: int) -> int:
    """Packed shards, in order, as <split>-T<seq_len>.pieces, .rows and .fim; returns the rows written."""
    path = packed_path(corpus, split, get_tokenizer(tokenizer_path), seq_len)
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


def shard_count(corpus: str, split: str, tokenizer_path: Path) -> int:
    return len(shard_bounds(corpus, split, get_tokenizer(tokenizer_path))) - 1


def take(corpus: str, more: Iterator[Shard]) -> Shard:
    """The next packed shard of a corpus read in part; it has one while its rows fall short of its draw."""
    shard = next(more, None)
    if shard is None:
        raise RuntimeError(f"{corpus} holds fewer rows than its draw: its extent is too small (prepare.extent; raise "
                           f"mixture.SPREAD)")
    return shard


def fit(stage: mixture.Stage, rows: dict[str, int], more: dict[str, Generator[Shard]],
        held: dict[str, list[Shard]]) -> dict[str, float]:
    """The stage's weights once every corpus holds its draw over the stage's rows (mixture.weights): a corpus read in
    part (more) packs its next shard into held while it draws more rows than it holds."""
    while True:
        shares, _, over = weights(stage, rows, stage.rows)
        if not over:
            return shares
        for name, excess in over.items():
            if name not in more:
                raise RuntimeError(f"{name}, read whole, holds {rows[name]:,} rows of the {rows[name] + excess:,} "
                                   f"its stage draws")
            shard = take(name, more[name])
            held[name].append(shard)
            rows[name] += len(shard[1])


def pack_stage(stage: mixture.Stage, tokenizer_path: Path, whole: dict[str, list[int]], part: dict[str, list[int]]
               ) -> tuple[dict[str, float], dict[str, int]]:
    """Pack a stage's train splits: the whole corpora's listed shards, written as packed; the corpora read in part,
    their listed shards in order until they hold their draw (fit), written then. Returns the weights and, of each
    corpus read in part, the shards it took."""
    rows = {corpus: write(corpus, "train", tokenizer_path,
                          packed_shards(corpus, "train", tokenizer_path, shards, stage.seq_len), stage.seq_len)
            for corpus, shards in whole.items()}
    more = {corpus: packed_shards(corpus, "train", tokenizer_path, shards, stage.seq_len)
            for corpus, shards in part.items()}
    held = {corpus: [take(corpus, more[corpus])] for corpus in part}  # one shard each; fit takes more as draws need
    rows |= {corpus: len(shards[0][1]) for corpus, shards in held.items()}
    shares = fit(stage, rows, more, held)
    for corpus, shards in held.items():
        more[corpus].close()
        write(corpus, "train", tokenizer_path, shards, stage.seq_len)
    for corpus, count in rows.items():
        print(f"T{stage.seq_len} {corpus}: {count:,} rows, weight {shares[corpus]:.4f}", flush=True)
    return shares, {corpus: len(shards) for corpus, shards in held.items()}


def main() -> None:
    """The corpus tokenizer's packs for the base model's run; val splits in threads beside the stages, so a val
    shard's longest document does not hold up the cores."""
    corpus_crate()  # built once, before the threads call it
    stages = (mixture.FILE_LEVEL_STAGE, mixture.REPO_LEVEL_STAGE)
    with ThreadPoolExecutor() as threads:
        vals = [threads.submit(write, corpus, "val", TOKENIZER,
                               packed_shards(corpus, "val", TOKENIZER, range(shard_count(corpus, "val", TOKENIZER)),
                                             stage.seq_len), stage.seq_len)
                for stage in stages for corpus in stage.sources]
        trailing = set(repository_parts(TOKENIZER))
        file_whole = {s: [i for i, path in enumerate(curated_shards(s, "train")) if int(path.stem) not in trailing]
                      for s in mixture.REPOSITORY_SOURCES}
        file_whole |= {s: list(range(shard_count(s, "train", TOKENIZER))) for s in mixture.FILE_LEVEL_STAGE.supply}
        _, taken = pack_stage(mixture.FILE_LEVEL_STAGE, TOKENIZER, file_whole,
                              {s: list(range(shard_count(s, "train", TOKENIZER))) for s in mixture.PART})
        repo_whole = {c: list(range(shard_count(c, "train", TOKENIZER)))
                      for c in [*mixture.REPOSITORIES.values(), *mixture.REPO_LEVEL_STAGE.supply]}
        pack_stage(mixture.REPO_LEVEL_STAGE, TOKENIZER, repo_whole,
                   {s: list(range(taken[s], shard_count(s, "train", TOKENIZER)))
                    for s in mixture.PART if s in mixture.REPO_LEVEL_STAGE.sources})
        for val in vals:
            val.result()
    held = sum(shard_tokens(c, "train", TOKENIZER).sum() for c in mixture.REPOSITORIES.values())
    print(f"repository partitions of the 32K stage: {sorted(trailing)}; its repositories hold {held:,} tokens",
          flush=True)


if __name__ == "__main__":
    main()
