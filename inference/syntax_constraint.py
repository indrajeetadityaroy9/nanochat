"""Complete Python 3.12 syntax constraint for decoding (decoder.Constraint), over the Rust crate syntax/.

Allows exactly the tokens that keep the text a prefix of a module CPython 3.12 accepts (ast.parse), up to
over-approximations (a rejected prefix allowed, never the reverse), and BOS iff ast.parse accepts the text.
"""

import ast
import functools
import keyword
import sys
import token as pytoken
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from lark import Lark

from corpus.tokenizer import Tokenizer
from native import crate_module

if TYPE_CHECKING:
    from inference.decoder import Reemit

CRATE = Path(__file__).resolve().parent / "syntax"
OPERATORS = (*pytoken.EXACT_TOKEN_TYPES, "<>")  # python/cpython Parser/token.c _PyToken_TwoChars: "<>" is NOTEQUAL
SOFT_KEYWORDS = {"match": "MATCH_SOFT", "case": "CASE_SOFT", "type": "TYPE_SOFT"}  # "_" is a name in every pattern
BASIC = ("NAME", "NUMBER", "STRING", "FSTRING_START", "FSTRING_END")
SPECIAL = ("$END", "NEWLINE", "INDENT", "DEDENT")
NO_ACTION = np.iinfo(np.int32).min  # parser.rs NO_ACTION


def identifier_ranges(test: Callable[[str], bool]) -> list[tuple[int, int]]:
    """The inclusive ranges of the non-ASCII code points c where test(chr(c)) holds."""
    chars = range(0x80, sys.maxunicode + 1)
    hits = np.flatnonzero(np.fromiter(map(test, map(chr, chars)), dtype=bool, count=len(chars))) + 0x80
    starts = np.flatnonzero(np.diff(hits, prepend=-2) != 1)
    ends = np.append(starts[1:], len(hits)) - 1
    return list(zip(hits[starts].tolist(), hits[ends].tolist(), strict=True))


@functools.cache
def python_syntax(tokenizer: Tokenizer):
    """The crate's Syntax of a vocabulary (Lark's LALR tables of python.lark, lexemes, identifier classes, byte trie).
    Cached per process and tokenizer; shared by every constraint and thread."""
    lark = Lark((CRATE / "python.lark").read_text(), parser="lalr", lexer="basic", start="file_input", strict=True)
    conf = lark.parse_interactive().parser_state.parse_conf
    rows = conf.states
    rule_index = {rule: i for i, rule in enumerate(lark.rules)}
    terminal = {t.pattern.value: t.name for t in lark.terminals}
    # every terminal and nonterminal of the grammar has an action or goto in some state
    terminals = sorted({t for row in rows.values() for t in row if t.isupper()})
    nonterminals = sorted({t for row in rows.values() for t in row if not t.isupper()})
    tid, nid = {t: i for i, t in enumerate(terminals)}, {t: i for i, t in enumerate(nonterminals)}
    actions = np.full((len(rows), len(terminals)), NO_ACTION, dtype=np.int32)
    goto = np.full((len(rows), len(nonterminals)), -1, dtype=np.int32)
    for s, row in rows.items():
        for t, (action, arg) in row.items():
            if t.isupper():
                actions[s, tid[t]] = arg if action.name == "Shift" else ~rule_index[arg]
            else:
                goto[s, nid[t]] = arg
    return crate_module(CRATE).Syntax(
        keywords=[(w, [tid[terminal[w]]]) for w in keyword.kwlist],
        soft=[(w, [tid["NAME"], tid[t]]) for w, t in SOFT_KEYWORDS.items()],
        operators=[(op, [tid[terminal["!=" if op == "<>" else op]]]) for op in OPERATORS],
        basic=[tid[t] for t in BASIC], special=[tid[t] for t in SPECIAL],
        terminals=len(terminals), nonterminals=len(nonterminals),
        actions=actions.ravel().tolist(), goto=goto.ravel().tolist(),
        rules=[(len(rule.expansion), nid[rule.origin.name]) for rule in lark.rules],
        start=conf.start_state, end=conf.end_state,
        xid_start=identifier_ranges(str.isidentifier),
        xid_continue=identifier_ranges(lambda c: ("a" + c).isidentifier()),
        token_bytes=tokenizer.mergeable)


def syntax_error(source: str) -> list[str]:
    """source's syntax error as "line:col: SyntaxError: message", [] if it parses. Parser overflow (MemoryError,
    RecursionError) and CPython 3.12.3's ValueError for f"{x:{y=}}" (python/cpython#121130, fixed in 3.12.5) count as
    failures to parse."""
    try:
        ast.parse(source)
    except SyntaxError as e:
        return [f"{e.lineno}:{e.offset}: SyntaxError: {e.msg}"]
    except (MemoryError, RecursionError, ValueError) as e:
        return [f"{type(e).__name__}: {e}"]
    return []


class SyntaxConstraint:
    """decoder.Constraint for the tokens generated after prefix (UTF-8 bytes) of a healed prompt; masks by masks(),
    intersected with the healing decoder.Reemit's while it owes bytes, BOS by ends(). Raises ValueError when prefix
    is not a prefix of any Python 3.12 module."""

    def __init__(self, tokenizer: Tokenizer, prefix: bytes, *, healing: "Reemit"):
        self.bos, self.healing = tokenizer.bos_token_id, healing
        self.native = python_syntax(tokenizer).constraint(prefix)

    def push(self, token: int) -> None:
        self.native.push(token)
        self.healing.push(token)


def ends(constraints: list[SyntaxConstraint]) -> list[bool]:
    """Whether each BOS candidate's text is a module: the grammar ends one, then ast.parse accepts it."""
    grammar = crate_module(CRATE).grammar_ends([c.native for c in constraints])
    return [ok and not syntax_error(c.native.text.decode()) for c, ok in zip(constraints, grammar, strict=True)]


def masks(constraints: list[SyntaxConstraint], out: np.ndarray, rows: list[int]) -> list[bool]:
    """Writes the next masks into distinct rows of out (C-contiguous uint32 words; bit t % 32 of word t // 32 allows
    token t). Returns whether each row allows some token."""
    allowed = crate_module(CRATE).masks([c.native for c in constraints], out.ctypes.data, rows, out.shape[1])
    for i, (c, row) in enumerate(zip(constraints, rows, strict=True)):
        if (healing := c.healing.mask()) is not None:
            out[row] &= healing
            allowed[i] = bool(out[row].any())
    return allowed
