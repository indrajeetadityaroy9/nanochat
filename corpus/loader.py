"""
Training's reading of the packed data (layout.py): a run's mixture of packed corpora (mixture.weights, row by row
mixture.mixture_rows) assembled into micro-batches (PretrainingBatches).
"""

import itertools
from collections.abc import Iterator
from typing import Literal, NamedTuple

import numpy as np
import torch

from corpus.layout import PIECE_COLUMNS, TOKEN_DTYPE, fim_packed, packed_path, split_path
from corpus.mixture import mixture_rows
from corpus.tokenizer import Tokenizer

# The label of an ignored position (padding, a BOS that is no target); as an input it becomes BOS
IGNORE_INDEX = -1


class PackedSplit(NamedTuple):
    """A packed split, memory-mapped: the split's tokens; each document's first token, then the number of tokens; each
    piece's PIECE_COLUMNS in row order; each row's first piece, then the number of pieces (row r is
    pieces[rows[r]:rows[r + 1]]); the FIM examples' tokens of a split packed with them (fim_packed), else None (the file
    is empty, which cannot be memory-mapped)."""
    tokens: np.memmap
    offsets: np.memmap
    pieces: np.memmap
    rows: np.memmap
    fim: np.memmap | None


def packed(corpus: str, split: str, tokenizer: Tokenizer, seq_len: int) -> PackedSplit:
    """The packed split of a corpus at rows of seq_len + 1 tokens (pack.py), memory-mapped in place."""
    path, packing = split_path(corpus, split, tokenizer), packed_path(corpus, split, tokenizer, seq_len)
    return PackedSplit(np.memmap(path.with_suffix(".bin"), dtype=TOKEN_DTYPE, mode="r"),
                       np.memmap(path.with_suffix(".idx"), dtype=np.int64, mode="r"),
                       np.memmap(packing.with_suffix(".pieces"), dtype=np.int64, mode="r").reshape(-1, PIECE_COLUMNS),
                       np.memmap(packing.with_suffix(".rows"), dtype=np.int64, mode="r"),
                       np.memmap(packing.with_suffix(".fim"), dtype=TOKEN_DTYPE, mode="r")
                       if fim_packed(corpus, split) else None)


class PretrainingBatches:
    """Re-iterable micro-batches (input_ids, labels, cu_seqlens, segments, lengths) of a mixture {corpus: weight} of
    packed corpora.
    orders: {corpus: row ids in read order, -1 an empty row (all padding)}; start: the run rows already consumed
    (resume), whose rows of each corpus (mixture_rows) are skipped; iteration stops when a
    corpus's order is exhausted, without the partial micro-batch. reading: "fim" reads the packed fill-in-the-middle
    examples in place of their pieces' tokens, "left_to_right" every piece left to right (assemble).
    - input_ids (int32), labels (int64): on the device
    - cu_seqlens: the attention segments of the micro-batch (document masking), int32 offsets into its flattened
      micro_batch_size x seq_len input_ids (FA4's varlen layout) on the device: each piece (its BOS and its tokens: a
      document, a document's cut part, or a fill-in-the-middle example) and each row's padding is a segment. Their
      number is fixed per loader, so a compiled model sees one shape: num_segments, the unused entries repeating the
      last offset (empty segments).
    - segments: the micro-batch's segments, a 0-dim int32 tensor on the host: the attention reads cu_seqlens'
      first segments + 1 entries only (model/ops/flash_attn.py), as FA4 spends time on every segment it is given,
      empty or not.
    - lengths: the segments' lengths (cu_seqlens' differences), int64 on the host: the micro-batch's attention work
      (metrics.executed_model_flops), read without a device synchronization."""

    def __init__(self, mixture: dict[str, float], split: str, tokenizer: Tokenizer, orders: dict[str, np.ndarray], *,
                 seq_len: int, micro_batch_size: int, device: torch.device | str,
                 reading: Literal["fim", "left_to_right"], start: int) -> None:
        self.names, self.weights = list(mixture), np.array(list(mixture.values()))
        self.packed = {name: packed(name, split, tokenizer, seq_len) for name in self.names}
        self.orders = [orders[name] for name in self.names]
        self.seq_len, self.micro_batch_size, self.device = seq_len, micro_batch_size, device
        self.bos = tokenizer.bos_token_id
        self.fim = reading == "fim"
        self.start = start
        # segments per micro-batch: micro_batch_size x (the most pieces in a row + 1 for the padding)
        self.num_segments = micro_batch_size * (
            max(int(np.diff(packing.rows).max()) for packing in self.packed.values()) + 1)

    def assemble(self, out: np.ndarray, name: str, row: int) -> list[int]:
        """Write a corpus's row into out (its padding already in place): every piece after a BOS, and a BOS after the
        last piece when that BOS is its target. A piece is its tokens or, reading "fim", the tokens of its
        fill-in-the-middle example if it is packed as one (pack.py: FIM_BEGIN P FIM_HOLE S FIM_END M or
        FIM_BEGIN FIM_HOLE S FIM_END P M). A piece that ends its document, and a fill-in-the-middle
        example, has the BOS after it as its label (end of document, end of the middle); a piece read left to right that
        its document continues after has none: the BOS after it is written as IGNORE_INDEX, an input BOS and an ignored
        label (or nothing, at the row's end). Returns where each piece starts in the row, then where the pieces end."""
        tokens, offsets, pieces, rows, fim = self.packed[name]
        row_pieces = pieces[rows[row]:rows[row + 1]]
        examples = (row_pieces[:, 3] > row_pieces[:, 2]) & self.fim
        # a piece ending where a document ends (offsets) has its BOS as target, as has every FIM example
        targeted = (examples | (offsets[np.searchsorted(offsets, row_pieces[:, 1])] == row_pieces[:, 1])).tolist()
        examples = examples.tolist()
        at, starts = 0, []
        for k, (start, end, f0, f1) in enumerate(row_pieces.tolist()):
            starts.append(at)
            out[at] = self.bos if k == 0 or targeted[k - 1] else IGNORE_INDEX
            part = fim[f0:f1] if examples[k] else tokens[start:end]
            out[at + 1:at + 1 + len(part)] = part
            at += 1 + len(part)
        if targeted[-1]:  # write the last piece's target BOS
            out[at] = self.bos
        return [*starts, at]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]]:
        consumed = np.bincount(mixture_rows(self.weights, start=0, count=self.start), minlength=len(self.names))
        orders = [iter(order[skip:]) for order, skip in zip(self.orders, consumed, strict=True)]
        for position in itertools.count(self.start, self.micro_batch_size):
            corpora = mixture_rows(self.weights, start=position, count=self.micro_batch_size)
            batch = np.full((self.micro_batch_size, self.seq_len + 1), IGNORE_INDEX, dtype=np.int32)
            bounds = []
            for r, c in enumerate(corpora):
                row = next(orders[c], None)
                if row is None:  # a corpus's order read whole
                    return
                starts = self.assemble(batch[r], self.names[c], row) if row >= 0 else [0]
                # the pieces, then the padding if the input_ids hold any
                bounds += [r * self.seq_len + start for start in starts if start < self.seq_len]
            # page-locked, so the copy is asynchronous
            batch = torch.from_numpy(batch).pin_memory().to(self.device, non_blocking=True)
            cu = np.full(self.num_segments + 1, self.micro_batch_size * self.seq_len, dtype=np.int32)
            cu[:len(bounds)] = bounds
            cu_seqlens = torch.from_numpy(cu).pin_memory().to(self.device, non_blocking=True)
            input_ids, labels = batch[:, :-1], batch[:, 1:].to(torch.int64)
            yield (input_ids.masked_fill(input_ids == IGNORE_INDEX, self.bos), labels, cu_seqlens,
                   torch.tensor(len(bounds), dtype=torch.int32), np.diff(cu[:len(bounds) + 1]).astype(np.int64))
