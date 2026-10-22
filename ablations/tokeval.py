"""
TokEval's intrinsic metrics (arXiv 2608.18062 §3; github.com/cimeister/tokenizer-intrinsic-evals at 7d03a61:
tokenizer_analysis/metrics/code_ast.py, math.py and information_theoretic.py), replicated for the tokenizer ablation's
arms (ablations/tokenizer.py describe) on their compiled val documents. A document's tokens are its compiled ids and
their spans its bytes, so positions are bytes where TokEval's are characters (the same on ASCII; a byte-level token can
end inside a character, which a character map cannot hold).
- AST boundary alignment, per leaf category (TokEval's five; data/crate leaves.rs): a leaf is start-aligned when a token
  starts at its first covered byte, end-aligned when one ends at its last, fully aligned when both; one leading space of
  a token holding non-whitespace is covered by no token (_trim_word_start_space_offsets: " foo" starts at foo);
- identifier fragmentation: the identifier leaves spanning more than one token, and tokens per identifier;
- indentation consistency (Python): the Spearman correlation, over the indented lines of a corpus, between a line's
  depth (its leading whitespace's width, tabs to 8 columns, over the GCD of its snippet's widths) and the
  whitespace-only tokens covering that whitespace, None below 3 depths;
- operator isolation (math.py _iter_operator_scores, _build_operator_results; reported for math, §3.5, and correlated
  with MBPP in §F.3): each of 24 operators in 6 categories found in the raw text is isolated when no token covering it
  holds a non-whitespace character outside it (" +" isolates "+"; a compound operator cut in two still counts), and a
  compound operator is preserved when one token holds it; micro-averaged over occurrences;
- Rényi efficiency of order 2 (TokEval's Table 2, which reports order 2 of its 1, 2, 2.5 and 3): H_2 of the documents'
  token distribution over log2 of the vocabulary size, descriptive only: two inference-time BPE variants raise it (at
  order 3) while lowering BLEU ("Two Counterexamples to Tokenization and the Noiseless Channel", LREC-COLING 2024,
  Tables 3 and 4).
The code of a corpus is each document of a repository corpus, parsed by its source's grammar, and each fenced Python or
C++ block of the other code corpora (kaggle_notebooks' notebooks, codeforces_cots' solutions); operators are counted in
whole documents, as TokEval counts them in its code and math corpora.
"""
import itertools
import math
import re

import numpy as np
from scipy.stats import spearmanr

from nanochat.data.loader import TOKEN_DTYPE, split_path
from nanochat.data.sources import SOURCES
from nanochat.data.tokenizer import Tokenizer
from nanochat.native import data_crate

CATEGORIES = ("identifier", "keyword", "operator", "literal", "delimiter")
FENCE = re.compile(rb"```(python|py|cpp|c\+\+)[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
FENCE_GRAMMARS = {b"python": "Python", b"py": "Python", b"cpp": "C++", b"c++": "C++"}
# math.py _OPERATOR_SPAN and _OPERATOR_CATEGORIES (a match outside the categories, a lone "!", is skipped)
OPERATOR = re.compile(rb"(?:\*\*|//|<<|>>|<=|>=|=>|==|!=|&&|\|\||\?:|[+\-*/=<>!&|^~%])")
OPERATOR_CATEGORIES = {"arithmetic": ("+", "-", "*", "/", "//", "%", "**"),
                       "comparison": ("<", ">", "<=", ">=", "==", "!="), "assignment": ("=", "=>"),
                       "logical_bitwise": ("&", "|", "^", "~", "&&", "||"), "shift": ("<<", ">>"), "ternary": ("?:",)}
# str.isspace of each byte; a byte of a multi-byte character counts as non-whitespace
WHITESPACE = np.array([b < 128 and chr(b).isspace() for b in range(256)])


def renyi_efficiency(counts: np.ndarray, vocab_size: int) -> float:
    """H_2 of the token counts' distribution over log2 of the vocabulary size."""
    p = counts[counts > 0] / counts.sum()
    return float(-np.log2(p @ p) / np.log2(vocab_size))


def val_split(corpus: str, tokenizers: list[Tokenizer]) -> tuple[dict[str, np.ndarray], list[bytes]]:
    """A corpus's compiled val split in every arm: {arm: its tokens} and each document's bytes, asserted the same in
    every arm."""
    tokens, docs = {}, None
    for tokenizer in tokenizers:
        path = split_path(corpus, "val", tokenizer)
        tokens[tokenizer.name] = np.fromfile(path.with_suffix(".bin"), dtype=TOKEN_DTYPE).astype(np.int64)
        offsets = np.fromfile(path.with_suffix(".idx"), dtype=np.int64)
        mine = [tokenizer.enc.decode_bytes(tokens[tokenizer.name][a:b].tolist())
                for a, b in itertools.pairwise(offsets)]
        assert docs is None or mine == docs, corpus
        docs = mine
    return tokens, docs


def operator_isolation(tokens: dict[str, np.ndarray], docs: list[bytes], tokenizers: list[Tokenizer]
                       ) -> dict[str, dict[str, dict[str, int]]]:
    """{arm: {category: {"isolated", "total"}, "compound": {"preserved", "total"}}}: the operator counts of documents
    (val_split), each document matched on its own."""
    category = {op.encode(): name for name, ops in OPERATOR_CATEGORIES.items() for op in ops}
    starts, stops, kinds, at = [], [], [], 0
    for doc in docs:
        for m in OPERATOR.finditer(doc):
            if m.group() in category:
                starts.append(at + m.start())
                stops.append(at + m.end())
                kinds.append(category[m.group()])
        at += len(doc)
    starts, stops, kinds = np.array(starts, dtype=np.int64), np.array(stops, dtype=np.int64), np.array(kinds)
    # non-whitespace bytes before each position of the documents' bytes
    marks = np.concatenate([[0], np.cumsum(~WHITESPACE[np.frombuffer(b"".join(docs), dtype=np.uint8)])])
    result = {}
    for tokenizer in tokenizers:
        ends = np.cumsum(tokenizer.token_lengths[tokens[tokenizer.name]])
        begins = ends - tokenizer.token_lengths[tokens[tokenizer.name]]
        first = np.searchsorted(ends, starts, side="right")
        last = np.searchsorted(ends, stops - 1, side="right")
        isolated = (marks[starts] == marks[begins[first]]) & (marks[ends[last]] == marks[stops])
        compound = stops - starts > 1
        result[tokenizer.name] = {
            **{name: {"isolated": int(isolated[kinds == name].sum()), "total": int((kinds == name).sum())}
               for name in OPERATOR_CATEGORIES},
            "compound": {"preserved": int((compound & (first == last)).sum()), "total": int(compound.sum())}}
    return result


def code(corpus: str, tokens: dict[str, np.ndarray], docs: list[bytes], tokenizers: list[Tokenizer]
         ) -> dict[str, dict[str, dict]]:
    """{arm: {grammar: metrics}} of a code corpus's val documents (val_split)."""
    starts = np.cumsum([0, *map(len, docs)])
    # (grammar, byte offset in the split, snippet)
    snippets = ([(SOURCES[corpus].grammar, start, doc) for start, doc in zip(starts, docs, strict=False)]
                if SOURCES[corpus].grammar else
                [(FENCE_GRAMMARS[m.group(1).lower()], start + m.start(2), m.group(2))
                 for start, doc in zip(starts, docs, strict=False) for m in FENCE.finditer(doc)])
    result = {tokenizer.name: {} for tokenizer in tokenizers}
    for grammar in sorted({g for g, _, _ in snippets}):
        mine = [(start, snippet) for g, start, snippet in snippets if g == grammar]
        parsed = data_crate().leaves([snippet.decode() for _, snippet in mine], grammar=grammar)
        leaf_starts = np.concatenate([np.frombuffer(s, dtype=np.uint32) + start
                                      for (start, _), (s, _, _, _) in zip(mine, parsed, strict=True)])
        leaf_ends = np.concatenate([np.frombuffer(e, dtype=np.uint32) + start
                                    for (start, _), (_, e, _, _) in zip(mine, parsed, strict=True)])
        categories = np.concatenate([np.frombuffer(c, dtype=np.uint8) for _, _, c, _ in parsed])
        lines = indented_lines(mine) if grammar == "Python" else None
        for tokenizer in tokenizers:
            result[tokenizer.name][grammar] = {
                "snippets": len(mine), "parse_error_spans": sum(errors for *_, errors in parsed),
                **alignment(tokenizer, tokens[tokenizer.name], leaf_starts.astype(np.int64),
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
    lengths = tokenizer.token_lengths
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
