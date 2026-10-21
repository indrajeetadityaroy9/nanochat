"""
TokEval's intrinsic metrics (arXiv 2608.18062 §3; github.com/cimeister/tokenizer-intrinsic-evals at 7d03a61:
tokenizer_analysis/metrics/code_ast.py and information_theoretic.py), replicated for the tokenizer ablation's arms
(ablations/tokenizer.py describe) on their compiled val documents. A document's tokens are its compiled ids and their
spans its bytes, so positions are bytes where TokEval's are characters (the same on ASCII; a byte-level token can end
inside a character, which a character map cannot hold), and a token's leading space before anything but whitespace is
covered by no token (TokEval's _trim_word_start_space_offsets: " foo" starts at foo).
- AST boundary alignment, per leaf category (TokEval's five; data/bpe leaves.rs): a leaf is start-aligned when a token
  starts at its first covered byte, end-aligned when one ends at its last, fully aligned when both;
- identifier fragmentation: the identifier leaves spanning more than one token, and tokens per identifier;
- indentation consistency (Python): the Spearman correlation, over the indented lines of a corpus, between a line's
  depth (its leading whitespace's width, tabs to 8 columns, over the GCD of its snippet's widths) and the
  whitespace-only tokens covering that whitespace, None below 3 depths;
- Rényi efficiency of order 2 (TokEval's Table 2): H_2 of the documents' token distribution over log2 of the
  vocabulary size, descriptive only: two tokenizers built to raise it lower translation quality ("Two Counterexamples
  to Tokenization and the Noiseless Channel", LREC-COLING 2024).
The code of a corpus is each document of a repository corpus, parsed by its source's grammar, and each fenced Python or
C++ block of the other code corpora (kaggle_notebooks' notebooks, codeforces_cots' solutions).
"""
import itertools
import math
import re

import numpy as np
from scipy.stats import spearmanr

from nanochat.data.loader import TOKEN_DTYPE, split_path
from nanochat.data.sources import SOURCES
from nanochat.data.tokenizer import Tokenizer, bpe

CATEGORIES = ("identifier", "keyword", "operator", "literal", "delimiter")
FENCE = re.compile(rb"```(python|py|cpp|c\+\+)[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
FENCE_GRAMMARS = {b"python": "Python", b"py": "Python", b"cpp": "C++", b"c++": "C++"}


def renyi_efficiency(counts: np.ndarray, vocab_size: int) -> float:
    """H_2 of the token counts' distribution over log2 of the vocabulary size."""
    p = counts[counts > 0] / counts.sum()
    return float(-np.log2(p @ p) / np.log2(vocab_size))


def code(corpus: str, tokenizers: list[Tokenizer]) -> dict[str, dict[str, dict]]:
    """{arm: {grammar: metrics}} of a code corpus's val documents, the same in every arm's compiled split."""
    splits = {}
    for tokenizer in tokenizers:
        path = split_path(corpus, "val", tokenizer)
        tokens = np.fromfile(path.with_suffix(".bin"), dtype=TOKEN_DTYPE).astype(np.int64)
        offsets = np.fromfile(path.with_suffix(".idx"), dtype=np.int64)
        splits[tokenizer.name] = tokens, [tokenizer.enc.decode_bytes(tokens[a:b].tolist())
                                          for a, b in itertools.pairwise(offsets)]
    docs = splits[tokenizers[0].name][1]
    assert all(other == docs for _, other in splits.values())
    starts = np.cumsum([0, *map(len, docs)])
    # (grammar, byte offset in the split, snippet)
    snippets = ([(SOURCES[corpus].grammar, start, doc) for start, doc in zip(starts, docs, strict=False)]
                if SOURCES[corpus].grammar else
                [(FENCE_GRAMMARS[m.group(1).lower()], start + m.start(2), m.group(2))
                 for start, doc in zip(starts, docs, strict=False) for m in FENCE.finditer(doc)])
    result = {tokenizer.name: {} for tokenizer in tokenizers}
    for grammar in sorted({g for g, _, _ in snippets}):
        mine = [(start, snippet) for g, start, snippet in snippets if g == grammar]
        parsed = bpe().leaves([snippet.decode() for _, snippet in mine], grammar=grammar)
        leaf_starts = np.concatenate([np.frombuffer(s, dtype=np.uint32) + start
                                      for (start, _), (s, _, _, _) in zip(mine, parsed, strict=True)])
        leaf_ends = np.concatenate([np.frombuffer(e, dtype=np.uint32) + start
                                    for (start, _), (_, e, _, _) in zip(mine, parsed, strict=True)])
        categories = np.concatenate([np.frombuffer(c, dtype=np.uint8) for _, _, c, _ in parsed])
        lines = indented_lines(mine) if grammar == "Python" else None
        for tokenizer in tokenizers:
            result[tokenizer.name][grammar] = {
                "snippets": len(mine), "parse_error_spans": sum(errors for *_, errors in parsed),
                **alignment(tokenizer, splits[tokenizer.name][0], leaf_starts.astype(np.int64),
                            leaf_ends.astype(np.int64), categories, lines)}
    return result


def indented_lines(snippets: list[tuple[int, bytes]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The indented non-blank lines of Python snippets (byte offset in the split, snippet): each one's leading
    whitespace's start and end in the split and its depth."""
    begins, ends, depths = [], [], []
    for start, snippet in snippets:
        found = []
        offset = start
        for line in snippet.split(b"\n"):
            width = len(line) - len(line.lstrip())
            if line.strip() and width:
                found.append((offset, offset + width, len(line[:width].expandtabs())))
            offset += len(line) + 1
        unit = math.gcd(*(w for *_, w in found)) or 1
        for begin, end, width in found:
            begins.append(begin)
            ends.append(end)
            depths.append(width // unit)
    return np.array(begins, dtype=np.int64), np.array(ends, dtype=np.int64), np.array(depths, dtype=np.int64)


def alignment(tokenizer: Tokenizer, tokens: np.ndarray, leaf_starts: np.ndarray, leaf_ends: np.ndarray,
              categories: np.ndarray, lines: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> dict:
    """AST boundary alignment per category and overall, identifier fragmentation and (lines given) indentation
    consistency of a split's tokens against its leaves (byte spans in the split)."""
    lengths = np.array([len(b) for b in tokenizer.token_bytes], dtype=np.int64)
    # a token's leading space before non-whitespace is covered by no token; a whitespace-only token covers indentation
    trim = np.array([b[:1] == b" " and not b.isspace() for b in tokenizer.token_bytes])[tokens]
    blank = np.array([b.isspace() for b in tokenizer.token_bytes])[tokens]
    ends = np.cumsum(lengths[tokens])
    begins = ends - lengths[tokens] + trim
    first = np.searchsorted(ends, leaf_starts, side="right")
    covered_first = leaf_starts + (leaf_starts < begins[first])
    last_byte = leaf_ends - 1
    last = np.searchsorted(ends, last_byte, side="right")
    uncovered_last = last_byte < begins[last]
    last_byte, last = last_byte - uncovered_last, last - uncovered_last
    mappable = covered_first <= last_byte
    start_aligned = begins[first] == covered_first
    end_aligned = ends[last] == last_byte + 1
    full = start_aligned & end_aligned
    rates = {name: {"count": int(taken.sum()), "start": float(start_aligned[taken].mean()),
                    "end": float(end_aligned[taken].mean()), "full": float(full[taken].mean())}
             for name, taken in ((name, mappable & (categories == c)) for c, name in enumerate(CATEGORIES))}
    identifiers = mappable & (categories == 0)
    pieces = (last - first + 1)[identifiers]
    record = {"unmappable": int((~mappable).sum()),
              "ast_boundary_alignment": rates | {"overall": {"count": int(mappable.sum()),
                                                             "full": float(full[mappable].mean())}},
              "identifier_fragmentation": {"count": int(identifiers.sum()), "rate": float((pieces > 1).mean()),
                                           "tokens_per_identifier": float(pieces.mean())}}
    if lines is not None:
        line_begins, line_ends, depths = lines
        blanks = np.concatenate([[0], np.cumsum(blank)])
        whitespace = (blanks[np.searchsorted(ends, line_ends - 1, side="right") + 1]
                      - blanks[np.searchsorted(ends, line_begins, side="right")])
        rho = spearmanr(depths, whitespace).statistic if len(np.unique(depths)) >= 3 else math.nan
        record["indentation_consistency"] = {"lines": len(depths), "depths": len(np.unique(depths)),
                                             "spearman": None if math.isnan(rho) else float(rho)}
    return record
