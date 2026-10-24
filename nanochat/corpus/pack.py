"""
Pack the compiled corpus (compile.py) into training rows of seq_len + 1 tokens, stage by stage (mixture.py), as
nanochat/data/pack.py packs production's: the data crate's pack, one curated shard per call (Best-Fit-Decreasing,
every piece its own attention segment, long documents cut between whole statements; normalization_doc.md §7.1), with
fill-in-the-middle examples in repository code's train splits (§6). Per corpus and stage, <split>-T<seq_len>.pieces,
.rows and .fim beside its compiled split.

A stage's train packs hold exactly the rows it may read, so no document is read twice:
- file stage: the repository sources' partitions before the repository stage's (compile.repository_parts) and the
  supply sources, whole; a source read in part (mixture.IN_PART), its leading shards;
- repository stage: the repositories corpora and Codeforces-CoTs, whole; ClimbMix and UltraData-Math L3, the shards
  after the file stage's.
A stage's weights are its shares (Stage.shares) over its packed rows, replayed through the loader's mixture_rows over
the stage's rows: a source read in part takes its next shard while its draw exceeds its rows; a "supply" source whose
draw exceeds its rows by the replay's discrepancy weighs that many rows less, the "rest" group taking them. The
weights go to loader.WEIGHTS. Val splits are packed whole at the seq_len of each stage that reads them.

python -m nanochat.corpus.pack
"""

import json
import os
from collections.abc import Generator, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from nanochat.corpus import mixture
from nanochat.corpus.compile import repository_parts, shard_tokens
from nanochat.corpus.loader import (
    PIECE_COLUMNS,
    TOKEN_DTYPE,
    WEIGHTS,
    fim_packed,
    grammar,
    mixture_rows,
    packed_path,
    shard_bounds,
    split_path,
)
from nanochat.corpus.parallel import ordered
from nanochat.corpus.sources import curated_shards
from nanochat.corpus.tokenizer import get_tokenizer
from nanochat.data.tokenizer import FIM_TOKENS, Tokenizer
from nanochat.native import data_crate

# Fill-in-the-middle as production's (nanochat/data/pack.py), without its line widening (normalization_doc.md §6):
# DeepSeek-Coder's rate (arXiv 2401.14196 §3.1.2: "an FIM rate of 0.5") and arXiv 2207.14255's 50% FIM; AST-FIM's
# middles (arXiv 2506.00204v1 §5.1: of FIM, 90% AST-FIM and 10% random character spans, 50% PSM and 50% SPM; §3.3: "an
# equal mix" of Single-Node and Aligned-Span masking). The crate draws each piece's FIM from ChaCha8 keyed by its
# document and piece, so the packs do not depend on the order of work
FIM_RATE, RANDOM_SHARE, SINGLE_NODE_SHARE, SPM_SHARE = 0.5, 0.1, 0.5, 0.5
# Pack calls in flight: one per core, so a core finishing its share of one shard's documents takes up another's (the
# crate releases the GIL)
WINDOW = os.cpu_count()

Shard = tuple[np.ndarray, np.ndarray, np.ndarray]


def pack_shard(task: tuple[str, str, int, int, str]) -> Shard:
    """Curated shard i of a compiled split of tokenizer `name` packed into rows of seq_len + 1 tokens: its pieces
    (PIECE_COLUMNS int64 each, the FIM range within the shard's FIM tokens), the pieces of each row, the shard's FIM
    tokens."""
    corpus, split, i, seq_len, name = task
    tokenizer = get_tokenizer(name)
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


def packed_shards(corpus: str, split: str, tokenizer: Tokenizer, shards: Iterable[int], seq_len: int
                  ) -> Generator[Shard]:
    """pack_shard of each shard, in order, WINDOW in flight; a shard is packed only as the window reaches it."""
    with ThreadPoolExecutor(WINDOW) as pool:
        yield from ordered(pool, pack_shard, [(corpus, split, i, seq_len, tokenizer.name) for i in shards], WINDOW)


def write(corpus: str, split: str, tokenizer: Tokenizer, shards: Iterable[Shard], seq_len: int) -> int:
    """Packed shards, in order, as <split>-T<seq_len>.pieces, .rows and .fim; returns the rows written."""
    path = packed_path(corpus, split, tokenizer, seq_len)
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


def shard_count(corpus: str, split: str, tokenizer: Tokenizer) -> int:
    return len(shard_bounds(corpus, split, tokenizer)) - 1


def take(corpus: str, more: Iterator[Shard]) -> Shard:
    """The next packed shard of a corpus read in part; it has one while its rows fall short of its draw."""
    shard = next(more, None)
    if shard is None:
        raise RuntimeError(f"{corpus} holds fewer rows than its draw: its extent is too small (prepare.extent; raise "
                           f"mixture.SPREAD)")
    return shard


def fit(stage: mixture.Stage, rows: dict[str, int], more: dict[str, Generator[Shard]],
        held: dict[str, list[Shard]]) -> dict[str, float]:
    """The stage's weights once every corpus holds its draw: replay mixture_rows over the stage's rows; a corpus read
    in part (more) packs its next shard into held while it draws more rows than it holds; a "supply" corpus that draws
    more weighs as many rows less. Shares are of the stage's rows here: each corpus's packed rows over the stage's."""
    supply = set(stage.supply)
    weighed: dict[str, float] = dict(rows)
    while True:
        shares = stage.shares(weighed, stage.rows)
        names = list(shares)
        draws = np.bincount(mixture_rows(np.array([shares[n] for n in names]), start=0, count=stage.rows),
                            minlength=len(names))
        over = {name: int(draw) - rows[name] for name, draw in zip(names, draws, strict=True) if draw > rows[name]}
        if not over:
            return shares
        for name, excess in over.items():
            if name in more:
                shard = take(name, more[name])
                held[name].append(shard)
                rows[name] += len(shard[1])
                weighed[name] = rows[name]
            elif name in supply:
                weighed[name] -= excess
            else:
                raise RuntimeError(f"{name}, read whole, holds {rows[name]:,} rows of the {rows[name] + excess:,} "
                                   f"its stage draws")


def pack_stage(stage: mixture.Stage, tokenizer: Tokenizer, whole: dict[str, list[int]], part: dict[str, list[int]]
               ) -> tuple[dict[str, float], dict[str, int]]:
    """Pack a stage's train splits: the whole corpora's listed shards, written as packed; the corpora read in part,
    their listed shards in order until they hold their draw (fit), written then. Returns the weights and, of each
    corpus read in part, the shards it took."""
    rows = {corpus: write(corpus, "train", tokenizer, packed_shards(corpus, "train", tokenizer, shards, stage.seq_len),
                          stage.seq_len)
            for corpus, shards in whole.items()}
    more = {corpus: packed_shards(corpus, "train", tokenizer, shards, stage.seq_len) for corpus, shards in part.items()}
    held = {corpus: [take(corpus, more[corpus])] for corpus in part}  # one shard each; fit takes more as draws need
    rows |= {corpus: len(shards[0][1]) for corpus, shards in held.items()}
    weights = fit(stage, rows, more, held)
    for corpus, shards in held.items():
        more[corpus].close()
        write(corpus, "train", tokenizer, shards, stage.seq_len)
    for corpus, count in rows.items():
        print(f"T{stage.seq_len} {corpus}: {count:,} rows, weight {weights[corpus]:.4f}", flush=True)
    return weights, {corpus: len(shards) for corpus, shards in held.items()}


def main() -> None:
    data_crate()  # built once, before the threads call it
    tokenizer = get_tokenizer()
    stages = (mixture.FILE_STAGE, mixture.REPO_STAGE)
    for stage in stages:
        for corpus in stage.sources:
            write(corpus, "val", tokenizer,
                  packed_shards(corpus, "val", tokenizer, range(shard_count(corpus, "val", tokenizer)), stage.seq_len),
                  stage.seq_len)
    trailing = set(repository_parts(tokenizer))
    file_whole = {s: [i for i, path in enumerate(curated_shards(s, "train")) if int(path.stem) not in trailing]
                  for s in mixture.REPOSITORY_SOURCES}
    file_whole |= {s: list(range(shard_count(s, "train", tokenizer))) for s in mixture.FILE_STAGE.supply}
    file_weights, taken = pack_stage(mixture.FILE_STAGE, tokenizer, file_whole,
                                     {s: list(range(shard_count(s, "train", tokenizer))) for s in mixture.PART})
    repo_whole = {c: list(range(shard_count(c, "train", tokenizer)))
                  for c in [*mixture.REPOSITORIES.values(), *mixture.REPO_STAGE.supply]}
    repo_weights, _ = pack_stage(mixture.REPO_STAGE, tokenizer, repo_whole,
                                 {s: list(range(taken[s], shard_count(s, "train", tokenizer)))
                                  for s in mixture.PART if s in mixture.REPO_STAGE.sources})
    WEIGHTS.write_text(json.dumps({str(stage.seq_len): {"rows": stage.rows, "weights": weights}
                                   for stage, weights in zip(stages, (file_weights, repo_weights), strict=True)},
                                  indent=1))
    held = sum(shard_tokens(c, "train", tokenizer).sum() for c in mixture.REPOSITORIES.values())
    print(f"repository partitions of the 32K stage: {sorted(trailing)}; its repositories hold {held:,} tokens",
          flush=True)


if __name__ == "__main__":
    main()
