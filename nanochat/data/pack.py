"""
Pack a compiled split (compile.py) into training rows of seq_len + 1 tokens (loader.py), seq_len a training stage's
(config.Stage): an index over its tokens, which are not copied, and the tokens of its fill-in-the-middle examples; the
loader builds each row on the fly ("Fewer Truncations Improve Language Modeling", arXiv 2404.10830, App. C.1: chunk
indices offline, sequences online).

Each document of the split is
- one piece, or, when it does not fit a row, pieces cut between two whole statements or declarations (repository
  samples, by their tree-sitter grammar, Source.grammar; the cut follows the statement before, so comments stay with
  the code after them), else after a line end, else between two characters; every piece enters its row after a BOS;
- in the train split of a repository corpus, made a fill-in-the-middle example per piece after cutting (context-level
  FIM, arXiv 2207.14255 §3.2) as AST-FIM trains it (arXiv 2506.00204v1 §5.1): with
  probability FIM_RATE; its middle a random character span with probability RANDOM_SHARE (§7: without random spans
  the model fails at random-span infilling), else a whole syntactic unit by Single-Node masking with probability
  SINGLE_NODE_SHARE, else by Aligned-Span masking (§3.3; a piece that does not parse takes a random span, §5.1), an
  AST middle widened to whole lines with probability LINE_SHARE (from its line's indentation through its last line
  break where only blanks lie beyond it: the middles of line-level infilling, which AST node spans rarely are); in SPM
  order with probability SPM_SHARE else PSM; prefix, suffix and middle encoded each on its own (arXiv 2207.14255 §3, as
  inference.decoder.fim_prompt encodes a prompt). Each piece draws from a generator keyed by its document and its
  index, so the packs are one deterministic function of the compiled data. Val splits stay left to right;
- packed by Best-Fit-Decreasing with the other pieces of its curated shard: no piece is cut and no token discarded.
  A piece takes the slots of the longer of its two readings, as a FIM example (loader reading "fim") and left to right
  ("left_to_right"), so every row fits both. A piece that ends its document or is a FIM example has the BOS after it as
  its target; a row of only such pieces keeps one slot for the BOS after its last piece. Within a row those pieces come
  first and the pieces cut from their document left to right last, so a cut piece is followed by the row's end, its
  padding or another cut piece; the loader writes the BOS after a piece read left to right that its document continues
  after as an input only (loader.PretrainingBatches.assemble). The end of a row that no piece fits is padding, neither
  context nor target.

What is packed is what some run can read, no document being read twice (train/stages.py):
- file stage: every val split, and the train split of every corpus of the stage's first group (code), whole; their
  rows bound every run's draw of that group (bound). A later group's runs draw at most bound · its share / the first
  group's share rows (group_rows), so its corpora keep, in proportion to their compiled train tokens, the fewest
  leading curated shards that hold that many: the rest of their compiled train split is cut (cut), as the corpora are
  read in file order (ClimbMix a shuffled crawl). A group's corpora keep about the same fraction, so its mixture
  weights (loader.mixture_weights: packed rows) stay about its natural composition and each corpus fits when the group
  does.
- repo stage: every val split whole; the train split of each of the stage's corpora at its last k curated shards,
  the repository-level stage's (stages.readable_rows), k the fewest such that they hold the most rows a repository
  stage draws (repo_rows), the same k for every corpus so that each fits when the stage does.

The data crate (data/crate pack.rs) packs one curated shard per call, its documents in parallel; WINDOW calls run at
once (it releases the GIL), so the cores stay busy through each call's serial best-fit-decreasing (Executor.map's
buffersize, Python 3.14, would bound the window as in_order does; the image's Python is 3.12). In
compiled/<source>-<tokenizer data key>/: <split>-T<seq_len>.pieces holds each piece's loader.PIECE_COLUMNS (int64:
token range [start, end) of <split>.bin, then the range of its FIM example in <split>-T<seq_len>.fim, empty for a
piece read left to right) in row order; <split>-T<seq_len>.rows the index of each row's first piece, then the number of
pieces (int64); <split>-T<seq_len>.fim the FIM examples' tokens (uint16; empty unless the split is packed with FIM,
loader.fim_packed). Shards are written in file order, so the output does not depend on the number of threads.

python -m nanochat.data.pack   # the file-level stage (every corpus's val and train splits), then the repo-level stage
"""

import json
import math
import os
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from nanochat.config import FILE_STAGE, REPO_SHARE, REPO_STAGE
from nanochat.data.loader import (
    PIECE_COLUMNS,
    TOKEN_DTYPE,
    fim_packed,
    mixture_corpora,
    packed_path,
    packed_rows,
    shard_documents,
    split_path,
)
from nanochat.data.sources import SOURCES
from nanochat.data.tokenizer import FIM_TOKENS, Tokenizer, get_tokenizer
from nanochat.native import data_crate

# The FIM rate of DeepSeek-Coder (arXiv 2401.14196 §3.1.2: "an FIM rate of 0.5"; its Fig. 3 ablation: a higher rate
# costs left-to-right completion) and arXiv 2207.14255 (50% FIM "for free"); AST-FIM's middles,
# arXiv 2506.00204v1 §5.1: of FIM, 90% AST-FIM and 10% random character spans, 50% PSM and 50% SPM; §3.3: "an equal mix"
# of Single-Node and Aligned-Span masking. LINE_SHARE: half the AST middles keep AST-FIM's node spans, half take the line
# boundaries of line-level infilling (HumanEval single- and multi-line infilling, arXiv 2204.05999 and
# arXiv 2207.14255: the prefix ends with a line break, the middle runs from the indentation through a line break)
FIM_RATE, RANDOM_SHARE, SINGLE_NODE_SHARE, LINE_SHARE, SPM_SHARE = 0.5, 0.1, 0.5, 0.5, 0.5
# Pack calls in flight: one per core, so a core finishing its share of one shard's documents takes up another's
WINDOW = os.cpu_count()


def pack_shard(source: str, split: str, tokenizer: Tokenizer, i: int, *, seq_len: int
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Curated shard i of a compiled split packed into rows of seq_len + 1 tokens: its pieces (PIECE_COLUMNS int64
    each, the FIM range within the shard's FIM tokens), the pieces of each row, the shard's FIM tokens."""
    compiled = split_path(source, split, tokenizer)
    bounds = shard_documents(source, split, tokenizer)
    fim = (dict(rate=FIM_RATE, random=RANDOM_SHARE, single=SINGLE_NODE_SHARE, line=LINE_SHARE, spm=SPM_SHARE,
                sentinels=[tokenizer.special_tokens[name] for name in FIM_TOKENS])
           if fim_packed(source, split) else None)
    pieces, counts, tokens = data_crate().pack(str(compiled.with_suffix(".bin")), str(compiled.with_suffix(".idx")),
                                               (int(bounds[i]), int(bounds[i + 1])), row_len=seq_len + 1,
                                               token_bytes=tokenizer.token_bytes[:tokenizer.bos_token_id],
                                               pattern=tokenizer.pattern, grammar=SOURCES[source].grammar, fim=fim)
    return (np.frombuffer(pieces, np.int64).reshape(-1, PIECE_COLUMNS), np.frombuffer(counts, np.int64),
            np.frombuffer(tokens, TOKEN_DTYPE))


def in_order(source: str, split: str, tokenizer: Tokenizer, shards: Iterable[int], *, seq_len: int
             ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """pack_shard of each shard, WINDOW in flight, in order."""
    with ThreadPoolExecutor(WINDOW) as pool:
        pending = deque()
        for i in shards:
            pending.append(pool.submit(pack_shard, source, split, tokenizer, i, seq_len=seq_len))
            if len(pending) == WINDOW:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def write(source: str, split: str, tokenizer: Tokenizer, shards: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
          *, seq_len: int) -> int:
    """Packed shards, in order, as <split>-T<seq_len>.pieces, .rows and .fim; returns the rows written."""
    path = packed_path(source, split, tokenizer, seq_len)
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


def pack_split(source: str, split: str, tokenizer: Tokenizer, *, seq_len: int) -> None:
    """Pack a compiled split whole."""
    shards = len(shard_documents(source, split, tokenizer)) - 1
    rows = write(source, split, tokenizer, in_order(source, split, tokenizer, range(shards), seq_len=seq_len),
                 seq_len=seq_len)
    print(f"{source} {split} T{seq_len}: {shards} files, {rows:,} rows")


def bound(tokenizer: Tokenizer) -> int:
    """The file-level stage's first group's train rows: the most rows of that group a run reads."""
    _, corpora = next(iter(FILE_STAGE.mixture.values()))
    return sum(packed_rows(corpus, "train", tokenizer, FILE_STAGE.seq_len) for corpus in corpora)


def group_rows(tokenizer: Tokenizer) -> dict[str, int]:
    """{group: the most file-level rows a run draws of it} for the stage's groups after the first (mixture_rows draws
    each group in proportion to its share)."""
    (_, (share, _)), *later = FILE_STAGE.mixture.items()
    return {group: math.ceil(bound(tokenizer) * group_share / share) for group, (group_share, _) in later}


def repo_rows(tokenizer: Tokenizer) -> int:
    """The most rows a repository-level stage draws: REPO_SHARE of a run's steps (schedule.plan) at REPO_STAGE's rows
    per micro-batch, against a file-level stage whose first group draws at most bound rows."""
    share, _ = next(iter(FILE_STAGE.mixture.values()))
    return math.ceil(bound(tokenizer) / share * REPO_SHARE / (1 - REPO_SHARE) * REPO_STAGE.batch_rows
                     / FILE_STAGE.batch_rows)


def tokens(source: str, split: str, tokenizer: Tokenizer) -> int:
    """The tokens of a compiled split (its .idx's total)."""
    return int(np.memmap(split_path(source, split, tokenizer).with_suffix(".idx"), dtype=np.int64, mode="r")[-1])


def cut(source: str, split: str, tokenizer: Tokenizer, *, files: int) -> None:
    """A compiled split cut to its first `files` curated files: .bin and .idx end with their last document, .json
    lists those files."""
    path = split_path(source, split, tokenizer)
    documents = int(shard_documents(source, split, tokenizer)[files])
    end = int(np.memmap(path.with_suffix(".idx"), dtype=np.int64, mode="r")[documents])
    os.truncate(path.with_suffix(".bin"), end * np.dtype(TOKEN_DTYPE).itemsize)
    os.truncate(path.with_suffix(".idx"), (documents + 1) * np.dtype(np.int64).itemsize)
    listing = json.loads(path.with_suffix(".json").read_text())
    path.with_suffix(".json").write_text(json.dumps({"files": listing["files"][:files]}))


def file_stage(tokenizer: Tokenizer) -> None:
    seq_len = FILE_STAGE.seq_len
    for source in mixture_corpora(FILE_STAGE.mixture):
        pack_split(source, "val", tokenizer, seq_len=seq_len)
    _, first = next(iter(FILE_STAGE.mixture.values()))
    for source in first:
        pack_split(source, "train", tokenizer, seq_len=seq_len)
    for group, need in group_rows(tokenizer).items():
        corpora = FILE_STAGE.mixture[group][1]
        total = sum(tokens(source, "train", tokenizer) for source in corpora)
        for source in corpora:
            part = math.ceil(need * tokens(source, "train", tokenizer) / total)
            files = len(shard_documents(source, "train", tokenizer)) - 1
            kept, held = [], 0
            shards = in_order(source, "train", tokenizer, range(files), seq_len=seq_len)
            for shard in shards:
                kept.append(shard)
                held += len(shard[1])
                if held >= part:
                    break
            shards.close()
            write(source, "train", tokenizer, kept, seq_len=seq_len)
            print(f"{source} train T{seq_len}: {len(kept)} of {files} files, {held:,} rows (its part of {group}'s "
                  f"{need:,})")
            cut(source, "train", tokenizer, files=len(kept))


def repo_stage(tokenizer: Tokenizer) -> None:
    seq_len = REPO_STAGE.seq_len
    corpora = mixture_corpora(REPO_STAGE.mixture)
    for source in corpora:
        pack_split(source, "val", tokenizer, seq_len=seq_len)
    files = {source: len(shard_documents(source, "train", tokenizer)) - 1 for source in corpora}
    need, held, rows = repo_rows(tokenizer), {source: [] for source in corpora}, 0
    with ThreadPoolExecutor(len(corpora)) as pool:
        while rows < need:
            k = len(held[corpora[0]]) + 1
            shards = pool.map(lambda s, k=k: pack_shard(s, "train", tokenizer, files[s] - k, seq_len=seq_len), corpora)
            for source, shard in zip(corpora, shards, strict=True):
                held[source].insert(0, shard)
                rows += len(shard[1])
    for source in corpora:
        write(source, "train", tokenizer, held[source], seq_len=seq_len)
    print(f"repo train T{seq_len}: the last {len(held[corpora[0]])} files of each corpus, {rows:,} rows "
          f"(a repository stage draws at most {need:,})")


if __name__ == "__main__":
    tokenizer = get_tokenizer()
    data_crate()  # built once, before the threads call it
    file_stage(tokenizer)  # the repo-level stage's draw is bounded by the file-level stage's code rows (repo_rows)
    repo_stage(tokenizer)
