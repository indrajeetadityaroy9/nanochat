"""
Pack the compiled corpus (compile.py) into training rows of seq_len + 1 tokens, stage by stage (mixture.py), as
nanochat/data/pack.py packs production's: the data crate's pack, one curated shard per call (Best-Fit-Decreasing,
every piece its own attention segment, long documents cut between whole statements; normalization_doc.md §7.1), with
fill-in-the-middle examples in repository code's train splits (§6). Per corpus and stage, <split>-T<seq_len>.pieces,
.rows and .fim beside its compiled split.

A stage's train packs hold exactly the rows it may read, so no document is read twice:
- file stage: the repository sources' partitions before the repository stage's (compile.repository_parts) and the
  notebooks, whole; a source read in part (mixture.IN_PART), its leading shards;
- repository stage: the repositories corpora and Codeforces-CoTs, whole; ClimbMix and UltraData-Math L3, the shards
  after the file stage's.
A stage's weights are its mixture's shares over its packed rows (mixture.source_shares), replayed through the loader's
mixture_rows over the stage's rows: a source read in part takes its next shard while its draw exceeds its rows; a
"supply" source whose draw exceeds its rows by the replay's discrepancy weighs that many rows less, the "rest" group
taking them. The weights go to loader.WEIGHTS. Val splits are packed whole at the seq_len of each stage that reads
them.

python -m nanochat.corpus.pack
"""

import json
import os
from collections import deque
from collections.abc import Generator, Iterable
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
    shard_documents,
    split_path,
)
from nanochat.corpus.sources import curated_dir
from nanochat.corpus.tokenizer import get_tokenizer
from nanochat.data.tokenizer import FIM_TOKENS
from nanochat.native import data_crate

# Fill-in-the-middle as production's (nanochat/data/pack.py), without its line widening (normalization_doc.md §6):
# DeepSeek-Coder's rate (arXiv 2401.14196 §3.1.2: "an FIM rate of 0.5") and arXiv 2207.14255's 50% FIM; AST-FIM's
# middles (arXiv 2506.00204v1 §5.1: of FIM, 90% AST-FIM and 10% random character spans, 50% PSM and 50% SPM; §3.3: "an
# equal mix" of Single-Node and Aligned-Span masking)
FIM_RATE, RANDOM_SHARE, SINGLE_NODE_SHARE, SPM_SHARE = 0.5, 0.1, 0.5, 0.5
# Pack calls in flight: one per core, so a core finishing its share of one shard's documents takes up another's
WINDOW = os.cpu_count()

Shard = tuple[np.ndarray, np.ndarray, np.ndarray]


def pack_shard(corpus: str, split: str, i: int, *, seq_len: int) -> Shard:
    """Curated shard i of a compiled split packed into rows of seq_len + 1 tokens: its pieces (PIECE_COLUMNS int64
    each, the FIM range within the shard's FIM tokens), the pieces of each row, the shard's FIM tokens."""
    tokenizer = get_tokenizer()
    compiled = split_path(corpus, split, tokenizer)
    bounds = shard_documents(corpus, split, tokenizer)
    fim = (dict(rate=FIM_RATE, random=RANDOM_SHARE, single=SINGLE_NODE_SHARE, line=0.0, spm=SPM_SHARE,
                sentinels=[tokenizer.special_tokens[name] for name in FIM_TOKENS])
           if fim_packed(corpus, split) else None)
    pieces, counts, tokens = data_crate().pack(str(compiled.with_suffix(".bin")), str(compiled.with_suffix(".idx")),
                                               (int(bounds[i]), int(bounds[i + 1])), row_len=seq_len + 1,
                                               token_bytes=tokenizer.token_bytes[:tokenizer.bos_token_id],
                                               pattern=tokenizer.pattern, grammar=grammar(corpus), fim=fim)
    return (np.frombuffer(pieces, np.int64).reshape(-1, PIECE_COLUMNS), np.frombuffer(counts, np.int64),
            np.frombuffer(tokens, TOKEN_DTYPE))


def in_order(corpus: str, split: str, shards: Iterable[int], *, seq_len: int) -> Generator[Shard]:
    """pack_shard of each shard, WINDOW in flight, in order (the crate releases the GIL)."""
    with ThreadPoolExecutor(WINDOW) as pool:
        pending = deque()
        for i in shards:
            pending.append(pool.submit(pack_shard, corpus, split, i, seq_len=seq_len))
            if len(pending) == WINDOW:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def write(corpus: str, split: str, shards: Iterable[Shard], *, seq_len: int) -> int:
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
    return len(shard_documents(corpus, split, get_tokenizer())) - 1


def take(corpus: str, more: Generator[Shard]) -> Shard:
    """The next packed shard of a corpus read in part; it has one while its rows fall short of its draw."""
    shard = next(more, None)
    assert shard is not None, f"{corpus} holds fewer rows than its draw: its extent is too small (prepare.extent)"
    return shard


def fit(stage: dict[str, tuple[mixture.Share, list[str]]], n_rows: int, rows: dict[str, int],
        more: dict[str, Generator[Shard]], held: dict[str, list[Shard]]) -> dict[str, float]:
    """The stage's weights once every corpus holds its draw: replay mixture_rows over the stage's n_rows; a corpus read
    in part (more) packs its next shard into held while it draws more rows than it holds; a "supply" corpus that draws
    more weighs as many rows less."""
    supply = {s for share, sources in stage.values() if share == "supply" for s in sources}
    weighed: dict[str, float] = dict(rows)
    while True:
        shares = mixture.source_shares(stage, n_rows, weighed)
        names = list(shares)
        draws = np.bincount(mixture_rows(np.array([shares[n] for n in names]), start=0, count=n_rows),
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
            else:
                assert name in supply, f"{name} holds fewer rows than its draw"
                weighed[name] -= excess


def stage(stage_groups: dict[str, tuple[mixture.Share, list[str]]], *, seq_len: int, n_rows: int,
          whole: dict[str, list[int]], part: dict[str, list[int]]) -> tuple[dict[str, float], dict[str, int]]:
    """Pack a stage's train splits: the whole corpora's listed shards, written as packed; the corpora read in part,
    their listed shards in order until they hold their draw (fit), written then. Returns the weights and, of each
    corpus read in part, the shards it took."""
    rows = {corpus: write(corpus, "train", in_order(corpus, "train", shards, seq_len=seq_len), seq_len=seq_len)
            for corpus, shards in whole.items()}
    more = {corpus: in_order(corpus, "train", shards, seq_len=seq_len) for corpus, shards in part.items()}
    held = {corpus: [] for corpus in part}
    for corpus in part:  # one shard each, then fit takes more as their draws need
        held[corpus].append(take(corpus, more[corpus]))
        rows[corpus] = len(held[corpus][0][1])
    weights = fit(stage_groups, n_rows, rows, more, held)
    for corpus, shards in held.items():
        more[corpus].close()
        write(corpus, "train", shards, seq_len=seq_len)
    for corpus, count in rows.items():
        print(f"T{seq_len} {corpus}: {count:,} rows, weight {weights[corpus]:.4f}", flush=True)
    return weights, {corpus: len(shards) for corpus, shards in held.items()}


def main() -> None:
    data_crate()  # built once, before the threads call it
    file_corpora = [s for _, sources in mixture.FILE_STAGE.values() for s in sources]
    repo_corpora = [s for _, sources in mixture.REPO_STAGE.values() for s in sources]
    for corpus in dict.fromkeys(file_corpora + repo_corpora):
        for seq_len in [s for s, corpora in ((mixture.FILE_SEQ, file_corpora), (mixture.REPO_SEQ, repo_corpora))
                        if corpus in corpora]:
            write(corpus, "val", in_order(corpus, "val", range(shard_count(corpus, "val")), seq_len=seq_len),
                  seq_len=seq_len)
    trailing = set(repository_parts())
    part_sources = list(mixture.PART)
    file_whole = {s: [i for i, p in enumerate(sorted(int(f.stem) for f in curated_dir(s, "train").iterdir()))
                      if p not in trailing] for s in mixture.REPOSITORY_SOURCES}
    file_whole["kaggle_notebooks"] = list(range(shard_count("kaggle_notebooks", "train")))
    file_weights, taken = stage(mixture.FILE_STAGE, seq_len=mixture.FILE_SEQ, n_rows=mixture.FILE_ROWS,
                                whole=file_whole,
                                part={s: list(range(shard_count(s, "train"))) for s in part_sources})
    repo_whole = {c: list(range(shard_count(c, "train"))) for c in [*mixture.REPOSITORIES.values(), "codeforces_cots"]}
    repo_weights, _ = stage(mixture.REPO_STAGE, seq_len=mixture.REPO_SEQ, n_rows=mixture.REPO_ROWS,
                            whole=repo_whole,
                            part={s: list(range(taken[s], shard_count(s, "train")))
                                  for s in part_sources if s in repo_corpora})
    WEIGHTS.write_text(json.dumps({
        str(mixture.FILE_SEQ): {"rows": mixture.FILE_ROWS, "weights": file_weights},
        str(mixture.REPO_SEQ): {"rows": mixture.REPO_ROWS, "weights": repo_weights}}, indent=1))
    print(f"repository partitions of the 32K stage: {sorted(trailing)}; "
          f"its repositories hold {sum(shard_tokens(c, 'train').sum() for c in mixture.REPOSITORIES.values()):,} "
          f"tokens", flush=True)


if __name__ == "__main__":
    main()
