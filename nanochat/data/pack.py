"""
Pack a compiled split (compile.py) into training rows of SEQ_LEN + 1 tokens (loader.py): an index over its tokens,
which are not copied, and the tokens of its fill-in-the-middle examples; the loader builds each row on the fly (Ding
et al. 2024, "Fewer Truncations Improve Language Modeling", arXiv 2404.10830, App. C.1: chunk indices offline,
sequences online).

Each document of the split is
- one piece, or, when it does not fit a row, pieces cut between two whole statements or declarations (repository
  samples, by their tree-sitter grammar, Source.grammar; the cut follows the statement before, so comments stay with
  the code after them), else after a line end, else between two characters; every piece enters its row after a BOS;
- in the train split of a repository corpus, made a fill-in-the-middle example per piece after cutting (context-level
  FIM, Bavarian et al. arXiv 2207.14255 §3.2) as AST-FIM trains it (Gong et al. arXiv 2506.00204v1 §5.1): with
  probability FIM_RATE; its middle a random character span with probability RANDOM_SHARE (§7: without random spans
  the model fails at random-span infilling), else a whole syntactic unit by Single-Node masking with probability
  SINGLE_NODE_SHARE, else by Aligned-Span masking (§3.3; a piece that does not parse takes a random span, §5.1); in SPM
  order with probability SPM_SHARE else PSM; prefix, suffix and middle encoded each on its own (Bavarian et al. §3, as
  inference.decoder.fim_prompt encodes a prompt). Each piece draws from a generator keyed by its document and its
  index, so the packs are one deterministic function of the compiled data. Val splits stay left to right;
- packed by Best-Fit-Decreasing with the other pieces of its curated shard: no piece is cut and no token discarded.
  A piece takes the slots of the longer of its two readings, as a FIM example (--objective spec) and left to right
  (ntp), so every row fits both. A piece that ends its document or is a FIM example has the BOS after it as its
  target; a row of only such pieces keeps one slot for the BOS after its last piece. Within a row those pieces come
  first and the pieces cut from their document left to right last, so a cut piece is followed by the row's end, its
  padding or another cut piece; the loader writes the BOS after a piece read left to right that its document continues
  after as an input only (loader.PretrainingBatches.assemble). The end of a row that no piece fits is padding, neither
  context nor target.

The crate (tokenizer.bpe: pack) packs one curated shard at a time, its documents in parallel. In
compiled/<source>-<tokenizer data key>/: <split>-T<seq_len>.pieces holds each piece's loader.PIECE_COLUMNS (int64:
token range [start, end) of <split>.bin, then the range of its FIM example in <split>-T<seq_len>.fim, empty for a
piece read left to right) in row order; <split>-T<seq_len>.rows the index of each row's first piece, then the number of
pieces (int64); <split>-T<seq_len>.fim the FIM examples' tokens (uint16; empty unless the split is packed with FIM,
loader.fim_packed). Shards are packed and written in file order, so the output does not depend on the number of
threads.

python -m nanochat.data.pack
"""

import itertools
import json

import numpy as np

from nanochat.config import SEQ_LEN
from nanochat.data.loader import TOKEN_DTYPE, fim_packed, packed_path, split_path
from nanochat.data.sources import SOURCES
from nanochat.data.tokenizer import FIM_TOKENS, bpe, get_tokenizer

# AST-FIM's training mixture, arXiv 2506.00204v1 §5.1 "FIM": 70% FIM and 30% left to right; of FIM, 90% AST-FIM and
# 10% random character spans; 50% PSM and 50% SPM; §3.3: "an equal mix" of Single-Node and Aligned-Span masking
FIM_RATE, RANDOM_SHARE, SINGLE_NODE_SHARE, SPM_SHARE = 0.7, 0.1, 0.5, 0.5


def pack_split(source, split, tokenizer):
    """Pack a compiled split into <split>-T<SEQ_LEN>.pieces, .rows and .fim."""
    compiled = split_path(source, split, tokenizer)
    with open(compiled + ".json") as f:
        files = json.load(f)["files"]
    bounds = np.cumsum([0] + [file["documents"] for file in files]).tolist()
    path = packed_path(source, split, tokenizer)
    fim = (FIM_RATE, RANDOM_SHARE, SINGLE_NODE_SHARE, SPM_SHARE, [tokenizer.special_tokens[name] for name in FIM_TOKENS]) \
        if fim_packed(source, split) else None
    print(f"Packing {len(files)} {split} files of '{source}' into rows of {SEQ_LEN + 1} tokens{' with FIM' if fim else ''}: {path}")
    mergeable = tokenizer.token_bytes[:tokenizer.bos_token_id]
    placed, fim_tokens = 0, 0
    with open(path + ".pieces", "wb") as pieces_file, open(path + ".rows", "wb") as rows_file, open(path + ".fim", "wb") as fim_file:
        for shard in itertools.pairwise(bounds):
            pieces, counts, tokens = bpe().pack(compiled + ".bin", compiled + ".idx", shard, row_len=SEQ_LEN + 1,
                                                token_bytes=mergeable, pattern=tokenizer.pattern,
                                                grammar=SOURCES[source].grammar, fim=fim, fim_base=fim_tokens)
            counts = np.frombuffer(counts, dtype=np.int64)
            pieces_file.write(pieces)
            rows_file.write(placed + np.cumsum(counts) - counts)  # each row's first piece
            fim_file.write(tokens)
            placed += int(counts.sum())
            fim_tokens += len(tokens) // np.dtype(TOKEN_DTYPE).itemsize
        rows_file.write(np.array([placed], dtype=np.int64))


if __name__ == "__main__":
    tokenizer = get_tokenizer()
    for source in SOURCES:
        for split in ("val", "train"):
            pack_split(source, split, tokenizer)
