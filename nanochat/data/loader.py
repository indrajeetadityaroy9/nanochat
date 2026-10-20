"""
Micro-batches of packed token rows (compile.py, pack.py) of one corpus or a mixture of corpora, in given row orders.

A packed split is an index of rows of seq_len + 1 tokens (inputs row[:-1], targets row[1:]; seq_len a training stage's,
config.Stage) over a corpus's
compiled tokens, both memory-mapped in place: each row is its pieces, every piece after a BOS, then padding to the
row's end, which enters as BOS and is never a target (-1, ignored by the loss and by bpb). A piece is a document or a
document's cut part, read left to right or, in the train packs of the repository corpora, possibly a fill-in-the-middle
example (pack.py; PretrainingBatches.assemble). Each corpus's rows are read once, in the order the caller gives. A run
interleaves the corpora by their weights (mixture_rows) and resumes from the number of rows it consumed. The mixture
is config.MIXTURE, {group: (share, corpora)}: each group's share is split among its corpora by their packed rows
(mixture_weights).
"""

import itertools
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import torch

from nanochat.data.sources import SOURCES
from nanochat.data.tokenizer import Tokenizer
from nanochat.paths import DATA

# the Weyl sequence frac(n * GOLDEN) spreads every prefix of itself most evenly over [0, 1) (Knuth, TAOCP Vol. 3 §6.4)
GOLDEN = (math.sqrt(5) - 1) / 2
TOKEN_DTYPE = np.uint16  # compiled tokens, as the crate reads them (data/bpe pack: &[u16])
# a pack index entry of the crate (data/bpe/pack.rs pack_shard): token range [start, end) of the compiled split, then
# the range of the piece's FIM example in the split's FIM tokens (empty for a piece read left to right)
PIECE_COLUMNS = 4


def split_path(dataset: str, split: str, tokenizer: Tokenizer) -> Path:
    """The stem of a compiled split's files (compile.py: .bin tokens, .idx document offsets, .json documents per file),
    in compiled/<corpus>-<tokenizer data key>/."""
    return DATA / "compiled" / f"{dataset}-{tokenizer.data_key}" / split


def packed_path(dataset: str, split: str, tokenizer: Tokenizer, seq_len: int) -> Path:
    """The stem of a packed split's files (pack.py: .pieces, .rows, .fim), the compiled split's at the row length."""
    return split_path(dataset, split, tokenizer).with_name(f"{split}-T{seq_len}")


def fim_packed(dataset: str, split: str) -> bool:
    """Whether a split is packed with fill-in-the-middle examples: the train split of a repository corpus (pack.py)."""
    return SOURCES[dataset].grammar is not None and split == "train"


class PackedSplit(NamedTuple):
    """A packed split, memory-mapped: the compiled tokens; each document's first token, then the number of tokens; each
    piece's PIECE_COLUMNS in row order; each row's first piece, then the number of pieces (row r is
    pieces[rows[r]:rows[r + 1]]); the FIM examples' tokens of a split packed with them (fim_packed), else None (the file
    is empty, which cannot be memory-mapped)."""
    tokens: np.memmap
    offsets: np.memmap
    pieces: np.memmap
    rows: np.memmap
    fim: np.memmap | None


def packed(dataset: str, split: str, tokenizer: Tokenizer, seq_len: int) -> PackedSplit:
    """The packed split of a corpus at rows of seq_len + 1 tokens (pack.py), memory-mapped in place."""
    path, packing = split_path(dataset, split, tokenizer), packed_path(dataset, split, tokenizer, seq_len)
    return PackedSplit(np.memmap(path.with_suffix(".bin"), dtype=TOKEN_DTYPE, mode="r"),
                       np.memmap(path.with_suffix(".idx"), dtype=np.int64, mode="r"),
                       np.memmap(packing.with_suffix(".pieces"), dtype=np.int64, mode="r").reshape(-1, PIECE_COLUMNS),
                       np.memmap(packing.with_suffix(".rows"), dtype=np.int64, mode="r"),
                       np.memmap(packing.with_suffix(".fim"), dtype=TOKEN_DTYPE, mode="r")
                       if fim_packed(dataset, split) else None)


def packed_rows(dataset: str, split: str, tokenizer: Tokenizer, seq_len: int) -> int:
    """The number of rows of a packed split."""
    return len(packed(dataset, split, tokenizer, seq_len).rows) - 1


def shard_documents(dataset: str, split: str, tokenizer: Tokenizer) -> np.ndarray:
    """Each curated shard's first document in a compiled split, then the number of documents (compiled <split>.json:
    each file's documents, in order)."""
    files = json.loads(split_path(dataset, split, tokenizer).with_suffix(".json").read_text())["files"]
    return np.cumsum([0] + [file["documents"] for file in files])


def row_shards(dataset: str, split: str, tokenizer: Tokenizer, seq_len: int) -> np.ndarray:
    """Each row's curated shard of a packed split: the packer packs one shard at a time (pack.py), so a row's pieces
    share the shard of its first piece's first token."""
    _, offsets, pieces, rows, _ = packed(dataset, split, tokenizer, seq_len)
    documents = shard_documents(dataset, split, tokenizer)
    return np.searchsorted(offsets[documents[:-1]], pieces[rows[:-1], 0], side="right") - 1


def mixture_corpora(groups: dict[str, tuple[float, list[str]]]) -> list[str]:
    """The corpora of a mixture {group: (share, corpora)}, in its order."""
    return [corpus for _, corpora in groups.values() for corpus in corpora]


def mixture_weights(groups: dict[str, tuple[float, list[str]]], tokenizer: Tokenizer, seq_len: int) -> dict[str, float]:
    """{corpus: weight} of a mixture {group: (share, corpora)}: each group's share split among its corpora in proportion
    to their packed train rows (at rows of seq_len + 1 tokens), so every row of a group is equally likely (its natural
    composition: no weight within a group is chosen)."""
    weights = {}
    for share, corpora in groups.values():
        rows = {corpus: packed_rows(corpus, "train", tokenizer, seq_len) for corpus in corpora}
        weights.update({corpus: share * count / sum(rows.values()) for corpus, count in rows.items()})
    return weights


def mixture_rows(weights: np.ndarray, *, start: int, count: int) -> np.ndarray:
    """Corpus index (into weights) of each run row start..start+count-1: row n goes to the corpus whose share of the
    cumulative weights holds the Weyl point frac((n + 1) * GOLDEN), so every prefix of the run holds each corpus nearly
    in proportion to its weight (the sequence's low discrepancy), and any stretch of the run is computed without the
    rows before it."""
    bounds = np.cumsum(weights)
    bounds /= bounds[-1]
    points = np.modf(np.arange(start + 1, start + count + 1) * GOLDEN)[0]
    return np.searchsorted(bounds, points, side="right")


class PretrainingBatches:
    """Re-iterable micro-batches (inputs, targets, cu_seqlens) of a mixture {corpus: weight} of packed corpora.
    orders: {corpus: row ids in read order}; start: the run rows already consumed (resume), whose rows of each corpus
    (mixture_rows) are skipped; iteration stops when a corpus's order is exhausted, without the partial micro-batch.
    reading: "fim" reads the packed fill-in-the-middle examples in place of their pieces' tokens, "left_to_right" every
    piece left to right (assemble).
    - inputs (int32), targets (int64): on the device
    - cu_seqlens: the attention segments of the micro-batch (document masking), int32 offsets into its flattened
      batch_rows x seq_len inputs (FA4's varlen layout) on the device: each piece (its BOS and its tokens: a document,
      a document's cut part, or a fill-in-the-middle example) and each row's padding is a segment. Their number is
      fixed per loader, so a compiled model sees one shape: num_segments, the unused entries repeating the last offset
      (empty segments)."""

    PAD = -1  # an assembled row's padding: the loss's ignore index as a target, BOS as an input

    def __init__(self, mixture: dict[str, float], split: str, tokenizer: Tokenizer, orders: dict[str, np.ndarray], *,
                 seq_len: int, batch_rows: int, device: torch.device | str,
                 reading: Literal["fim", "left_to_right"], start: int) -> None:
        self.names, self.weights = list(mixture), np.array(list(mixture.values()))
        self.packed = {name: packed(name, split, tokenizer, seq_len) for name in self.names}
        self.orders = [orders[name] for name in self.names]
        self.seq_len, self.batch_rows, self.device = seq_len, batch_rows, device
        self.bos = tokenizer.bos_token_id
        self.fim = reading == "fim"
        self.start = start
        # segments per micro-batch: batch_rows x (the most pieces in a row + 1 for the padding)
        self.num_segments = batch_rows * (max(int(np.diff(packing.rows).max()) for packing in self.packed.values()) + 1)

    def assemble(self, out: np.ndarray, name: str, row: int) -> list[int]:
        """Write a corpus's row into out (its padding already in place): every piece after a BOS, and a BOS after the
        last piece when that BOS is its target. A piece is its tokens or, reading "fim", the tokens of its
        fill-in-the-middle example if it is packed as one (pack.py: <fim_prefix> P <fim_suffix> S <fim_middle> M or
        <fim_prefix> <fim_suffix> S <fim_middle> P M). A piece that ends its document, and a fill-in-the-middle example,
        has the BOS after it as its target (end of document, end of the middle); a piece read left to right that its
        document continues after has none: the BOS after it is written as PAD, an input BOS and an ignored target (or
        nothing, at the row's end). Returns where each piece starts in the row, then where the pieces end."""
        tokens, offsets, pieces, rows, fim = self.packed[name]
        row_pieces = pieces[rows[row]:rows[row + 1]]
        examples = (row_pieces[:, 3] > row_pieces[:, 2]) & self.fim
        # a piece ending where a document ends (offsets) has its BOS as target, as has every FIM example
        targeted = (examples | (offsets[np.searchsorted(offsets, row_pieces[:, 1])] == row_pieces[:, 1])).tolist()
        examples = examples.tolist()
        at, starts = 0, []
        for k, (start, end, f0, f1) in enumerate(row_pieces.tolist()):
            starts.append(at)
            out[at] = self.bos if k == 0 or targeted[k - 1] else self.PAD
            part = fim[f0:f1] if examples[k] else tokens[start:end]
            out[at + 1:at + 1 + len(part)] = part
            at += 1 + len(part)
        if targeted[-1]:  # write the last piece's target BOS
            out[at] = self.bos
        return [*starts, at]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        consumed = np.bincount(mixture_rows(self.weights, start=0, count=self.start), minlength=len(self.names))
        orders = [iter(order[skip:]) for order, skip in zip(self.orders, consumed, strict=True)]
        for position in itertools.count(self.start, self.batch_rows):
            corpora = mixture_rows(self.weights, start=position, count=self.batch_rows)
            batch = np.full((self.batch_rows, self.seq_len + 1), self.PAD, dtype=np.int32)
            bounds = []
            for r, c in enumerate(corpora):
                row = next(orders[c], None)
                if row is None:  # a corpus's order read whole
                    return
                starts = self.assemble(batch[r], self.names[c], row)
                # the pieces, then the padding if the inputs hold any
                bounds += [r * self.seq_len + start for start in starts if start < self.seq_len]
            # page-locked, so the copy is asynchronous
            batch = torch.from_numpy(batch).pin_memory().to(self.device, non_blocking=True)
            cu = np.full(self.num_segments + 1, self.batch_rows * self.seq_len, dtype=np.int32)
            cu[:len(bounds)] = bounds
            cu_seqlens = torch.from_numpy(cu).pin_memory().to(self.device, non_blocking=True)
            inputs = batch[:, :-1]
            yield inputs.masked_fill(inputs == self.PAD, self.bos), batch[:, 1:].to(torch.int64), cu_seqlens
