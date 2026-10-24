"""
Pack the compiled corpus (compile.py) into training rows of seq_len + 1 tokens, stage by stage (mixture.py): the data
crate's pack, one curated shard per call (Best-Fit-Decreasing, every piece its own attention segment, long documents
cut between whole statements; normalization_doc.md §7.1), with fill-in-the-middle examples in repository code's train
splits (§6). Per corpus and stage, <split>-T<seq_len>.pieces, .rows and .fim beside its compiled split (loader.py).

A stage's train packs hold exactly the rows the base model's pre-training run reads (mixture.py STEPS), so no document
is read twice:
- file stage: the repository sources' partitions before the repository stage's (compile.repository_parts) and the
  supply sources, whole; a source read in part (mixture.IN_PART), its leading shards;
- repository stage: the repositories corpora and Codeforces-CoTs, whole; ClimbMix and UltraData-Math L3, the shards
  after the file stage's.
A source read in part takes its next shard while the run's stage weights (loader.weights over the stage's rows) draw
more of it than it holds. Training fits the same weights to any run's size (train/stages.py). Val splits are packed
whole at the seq_len of each stage that reads them.

python -m nanochat.corpus.pack
"""

import os
from collections.abc import Generator, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from nanochat.corpus import mixture
from nanochat.corpus.compile import repository_parts, shard_tokens
from nanochat.corpus.loader import (
    PIECE_COLUMNS,
    TOKEN_DTYPE,
    fim_packed,
    grammar,
    packed_path,
    shard_bounds,
    split_path,
    weights,
)
from nanochat.corpus.parallel import ordered
from nanochat.corpus.sources import curated_shards
from nanochat.corpus.tokenizer import FIM_TOKENS, TOKENIZER, get_tokenizer
from nanochat.native import data_crate

# Fill-in-the-middle, without line widening (normalization_doc.md §6): FIM rate 0.5 (DeepSeek-Coder, arXiv 2401.14196
# §3.1.2: "an FIM rate of 0.5"; arXiv 2207.14255 §3: p = 0.5); of FIM, 90% AST middles and 10% random character spans,
# 50% PSM and 50% SPM (AST-FIM, arXiv 2506.00204v1 §5.1); AST middles "an equal mix" of Single-Node and Aligned-Span
# masking (§3.3). The crate draws each piece's FIM from ChaCha8 keyed by its document and piece, so the packs do not
# depend on the order of work
FIM_RATE, RANDOM_SHARE, SINGLE_NODE_SHARE, SPM_SHARE = 0.5, 0.1, 0.5, 0.5
# Pack calls in flight: one per core (the crate releases the GIL)
WINDOW = os.cpu_count()

Shard = tuple[np.ndarray, np.ndarray, np.ndarray]


def pack_shard(task: tuple[str, str, int, int, Path]) -> Shard:
    """Curated shard i of a compiled split of the vocabulary in directory `vocabulary` packed into rows of seq_len + 1
    tokens: its pieces (PIECE_COLUMNS int64 each, the FIM range within the shard's FIM tokens), the pieces of each row,
    the shard's FIM tokens."""
    corpus, split, i, seq_len, vocabulary = task
    tokenizer = get_tokenizer(vocabulary)
    compiled = split_path(corpus, split, tokenizer)
    bounds = shard_bounds(corpus, split, tokenizer)
    fim = (dict(rate=FIM_RATE, random=RANDOM_SHARE, single=SINGLE_NODE_SHARE, line=0.0, spm=SPM_SHARE,
                sentinels=[tokenizer.special_tokens[name] for name in FIM_TOKENS])
           if fim_packed(corpus, split) else None)
    pieces, counts, tokens = data_crate().pack(str(compiled.with_suffix(".bin")), str(compiled.with_suffix(".idx")),
                                               (int(bounds[i]), int(bounds[i + 1])), row_len=seq_len + 1,
                                               token_bytes=tokenizer.token_bytes[:tokenizer.bos_token_id],
                                               pattern=tokenizer.pattern, grammar=grammar(corpus), fim=fim)
    return (np.frombuffer(pieces, np.int64).reshape(-1, PIECE_COLUMNS), np.frombuffer(counts, np.int64),
            np.frombuffer(tokens, TOKEN_DTYPE))


def packed_shards(corpus: str, split: str, vocabulary: Path, shards: Iterable[int], seq_len: int) -> Generator[Shard]:
    """pack_shard of each shard, in order, WINDOW in flight; a shard is packed only as the window reaches it."""
    with ThreadPoolExecutor(WINDOW) as pool:
        yield from ordered(pool, pack_shard, [(corpus, split, i, seq_len, vocabulary) for i in shards], WINDOW)


def write(corpus: str, split: str, vocabulary: Path, shards: Iterable[Shard], seq_len: int) -> int:
    """Packed shards, in order, as <split>-T<seq_len>.pieces, .rows and .fim; returns the rows written."""
    path = packed_path(corpus, split, get_tokenizer(vocabulary), seq_len)
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


def shard_count(corpus: str, split: str, vocabulary: Path) -> int:
    return len(shard_bounds(corpus, split, get_tokenizer(vocabulary))) - 1


def take(corpus: str, more: Iterator[Shard]) -> Shard:
    """The next packed shard of a corpus read in part; it has one while its rows fall short of its draw."""
    shard = next(more, None)
    if shard is None:
        raise RuntimeError(f"{corpus} holds fewer rows than its draw: its extent is too small (prepare.extent; raise "
                           f"mixture.SPREAD)")
    return shard


def fit(stage: mixture.Stage, rows: dict[str, int], more: dict[str, Generator[Shard]],
        held: dict[str, list[Shard]]) -> dict[str, float]:
    """The stage's weights once every corpus holds its draw over the stage's rows (loader.weights): a corpus read in
    part (more) packs its next shard into held while it draws more rows than it holds."""
    while True:
        shares, over = weights(stage, rows, stage.rows)
        if not over:
            return shares
        for name, excess in over.items():
            if name not in more:
                raise RuntimeError(f"{name}, read whole, holds {rows[name]:,} rows of the {rows[name] + excess:,} "
                                   f"its stage draws")
            shard = take(name, more[name])
            held[name].append(shard)
            rows[name] += len(shard[1])


def pack_stage(stage: mixture.Stage, vocabulary: Path, whole: dict[str, list[int]], part: dict[str, list[int]]
               ) -> tuple[dict[str, float], dict[str, int]]:
    """Pack a stage's train splits: the whole corpora's listed shards, written as packed; the corpora read in part,
    their listed shards in order until they hold their draw (fit), written then. Returns the weights and, of each
    corpus read in part, the shards it took."""
    rows = {corpus: write(corpus, "train", vocabulary,
                          packed_shards(corpus, "train", vocabulary, shards, stage.seq_len), stage.seq_len)
            for corpus, shards in whole.items()}
    more = {corpus: packed_shards(corpus, "train", vocabulary, shards, stage.seq_len)
            for corpus, shards in part.items()}
    held = {corpus: [take(corpus, more[corpus])] for corpus in part}  # one shard each; fit takes more as draws need
    rows |= {corpus: len(shards[0][1]) for corpus, shards in held.items()}
    shares = fit(stage, rows, more, held)
    for corpus, shards in held.items():
        more[corpus].close()
        write(corpus, "train", vocabulary, shards, stage.seq_len)
    for corpus, count in rows.items():
        print(f"T{stage.seq_len} {corpus}: {count:,} rows, weight {shares[corpus]:.4f}", flush=True)
    return shares, {corpus: len(shards) for corpus, shards in held.items()}


def main() -> None:
    """The corpus's tokenizer's packs for the base model's pre-training run."""
    data_crate()  # built once, before the threads call it
    stages = (mixture.FILE_STAGE, mixture.REPO_STAGE)
    for stage in stages:
        for corpus in stage.sources:
            write(corpus, "val", TOKENIZER,
                  packed_shards(corpus, "val", TOKENIZER, range(shard_count(corpus, "val", TOKENIZER)), stage.seq_len),
                  stage.seq_len)
    trailing = set(repository_parts(TOKENIZER))
    file_whole = {s: [i for i, path in enumerate(curated_shards(s, "train")) if int(path.stem) not in trailing]
                  for s in mixture.REPOSITORY_SOURCES}
    file_whole |= {s: list(range(shard_count(s, "train", TOKENIZER))) for s in mixture.FILE_STAGE.supply}
    _, taken = pack_stage(mixture.FILE_STAGE, TOKENIZER, file_whole,
                          {s: list(range(shard_count(s, "train", TOKENIZER))) for s in mixture.PART})
    repo_whole = {c: list(range(shard_count(c, "train", TOKENIZER)))
                  for c in [*mixture.REPOSITORIES.values(), *mixture.REPO_STAGE.supply]}
    pack_stage(mixture.REPO_STAGE, TOKENIZER, repo_whole,
               {s: list(range(taken[s], shard_count(s, "train", TOKENIZER)))
                for s in mixture.PART if s in mixture.REPO_STAGE.sources})
    held = sum(shard_tokens(c, "train", TOKENIZER).sum() for c in mixture.REPOSITORIES.values())
    print(f"repository partitions of the 32K stage: {sorted(trailing)}; its repositories hold {held:,} tokens",
          flush=True)


if __name__ == "__main__":
    main()
