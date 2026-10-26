"""
Complete Python 3.12 syntax constraint for decoding (decoder.Constraint): after each token, exactly the tokens whose
bytes keep the text a prefix of some module CPython 3.12's parser accepts (ast.parse), up to the over-approximations
listed below, and BOS iff ast.parse accepts the text (followed by the FIM suffix). Never rejects a valid prefix.

Layers, in the Rust crate syntax/ (built on first use, native.crate_module):
- lexer (lexer.rs): CPython 3.12.3's tokenizer (Parser/tokenizer.c tok_get_normal_mode, tok_get_fstring_mode,
  verify_end_of_number) restated as a character-level machine whose states are values, so a prefix may end anywhere
  (mid-token, mid-indent, mid-UTF-8 character). Its input is the text as compile() sees it: "\\r\\n" and "\\r" read as
  "\\n" (tokenizer.c translate_newlines), a final "\\n" implied at the end.
- indentation (parser.rs): tokenizer.c's INDENT/DEDENT (tab size 8, the tab-consistency check at tab size 1, a
  backslash line's first column) and its bracket level (newlines inside brackets are not NEWLINE tokens).
- parser (parser.rs): syntax/python.lark, an LALR(1) grammar accepting every token sequence of CPython's PEG grammar
  (Grammar/python.gram, 3.12; Lark builds the tables here, strict mode proves them conflict-free), read as a
  viable-prefix recognizer; the soft keywords match/case/type are read both ways (a name and a keyword: two parser
  stacks until one dies).
Over-approximations (a prefix allowed that CPython rejects; never the reverse): assignment, deletion and with targets
are expressions; parameter and argument orders are free; "(a as b)" parses anywhere (the with statement's parenthesized
items); patterns accept f-strings and any dotted name as a mapping key; escapes, the f-string conversion character, a
replacement field's unparenthesized lambda, and the limits on nesting are not checked; "<>" is "!=" (PEP 401's
barry_as_FLUFL); a partial token at the end is judged by the lexemes it can begin with. BOS is exact: the crate
proposes it when the lexer and parser end a module (always, before a suffix), and a chosen BOS stands only if the
grammar ends a module after the suffix too and ast.parse accepts the text and suffix (ends; the decoder takes the
step's best other token otherwise, NativeDecoder.generate).

Masks (vocab.rs): for each lexer state, every vocabulary token's events (tokens emitted, newlines, line-start
whitespace) and end state are computed once over a byte trie of the vocabulary, the subtrie after each first byte
shared by the lexer states that byte takes to one state; a step then runs the parser once per distinct event sequence
and maps the verdicts back to tokens. masks() computes a batch of constraints' masks in parallel, without the GIL,
into the decoder's mask rows. A healed prompt's constraint holds its decoder.Reemit (decoder.heal_tokens): until the
generated tokens re-emit the dropped bytes, masks() intersects each mask with the Reemit's (BOS excluded).
"""

import ast
import functools
import keyword
import sys
import token as pytoken
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from lark import Lark

from corpus.tokenizer import Tokenizer
from native import crate_module

if TYPE_CHECKING:
    from inference.decoder import Reemit

CRATE = Path(__file__).resolve().parent / "syntax"
OPERATORS = (*pytoken.EXACT_TOKEN_TYPES, "<>")  # tokenizer.c _PyToken_TwoChars: "<>" is NOTEQUAL (PEP 401)
SOFT_KEYWORDS = {"match": "MATCH_SOFT", "case": "CASE_SOFT", "type": "TYPE_SOFT"}  # "_" is a name in every pattern
BASIC = ("NAME", "NUMBER", "STRING", "FSTRING_START", "FSTRING_END")
SPECIAL = ("$END", "NEWLINE", "INDENT", "DEDENT")
NO_ACTION = np.iinfo(np.int32).min  # parser.rs NO_ACTION


def identifier_ranges(test) -> list[tuple[int, int]]:
    """The inclusive ranges of the non-ASCII code points c where test(chr(c)) holds."""
    chars = range(0x80, sys.maxunicode + 1)
    hits = np.flatnonzero(np.fromiter(map(test, map(chr, chars)), dtype=bool, count=len(chars))) + 0x80
    starts = np.flatnonzero(np.diff(hits, prepend=-2) != 1)
    ends = np.append(starts[1:], len(hits)) - 1
    return list(zip(hits[starts].tolist(), hits[ends].tolist(), strict=True))


@functools.cache
def python_syntax(tokenizer: Tokenizer):
    """The crate's Syntax of a vocabulary, built once per process and tokenizer and shared by every constraint and
    thread: Lark's LALR tables of python.lark (dense, terminals and nonterminals by sorted name), the lexemes'
    terminals, Python's identifier classes beyond ASCII (str.isidentifier of c, of "a" + c), the vocabulary's byte trie,
    and the per-lexer-state token groupings, filled on first use."""
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
        token_bytes=tokenizer.token_bytes[:tokenizer.bos_token_id])


def syntax_error(source: str) -> list[str]:
    """source's syntax error as a diagnostic ("line:col: SyntaxError: message"), [] if it parses. Nesting too deep for
    CPython's parser (its MemoryError "Parser stack overflowed", or RecursionError) is a failure to parse too."""
    try:
        ast.parse(source)
    except SyntaxError as e:
        return [f"{e.lineno}:{e.offset}: SyntaxError: {e.msg}"]
    except (MemoryError, RecursionError) as e:
        return [f"{type(e).__name__}: {e}"]
    return []


class SyntaxConstraint:
    """decoder.Constraint for the tokens generated after prefix (the program's UTF-8 bytes before them), its masks by
    masks(). suffix: the bytes after the generation (a FIM middle's suffix; b"": none); healing: a healed prompt's
    decoder.Reemit (decoder.heal_tokens; None: an unhealed prompt), whose mask each mask is intersected with. A BOS
    candidate stands iff ends() holds for it. Raises ValueError when prefix is not a prefix of any Python 3.12
    module."""

    def __init__(self, tokenizer: Tokenizer, prefix: bytes, *, suffix: bytes, healing: "Reemit | None" = None):
        self.suffix, self.bos, self.healing = suffix, tokenizer.bos_token_id, healing
        self.native = python_syntax(tokenizer).constraint(prefix, suffix=suffix)

    def push(self, token: int) -> None:
        self.native.push(token)
        if self.healing is not None:
            self.healing.push(token)


def ends(constraints: list[SyntaxConstraint]) -> list[bool]:
    """Whether each BOS candidate's text and suffix are a module (masks() proposes no BOS while a healing Reemit owes
    bytes): the grammar ends one (the crate, in parallel without the GIL), then ast.parse accepts them (the grammar's
    end implies valid UTF-8: the lexer rejects any other)."""
    grammar = crate_module(CRATE).grammar_ends([c.native for c in constraints])
    return [ok and not syntax_error((c.native.text + c.suffix).decode())
            for c, ok in zip(constraints, grammar, strict=True)]


def masks(constraints: list[SyntaxConstraint], out: np.ndarray, rows: list[int]) -> list[bool]:
    """Writes the next masks of constraints into rows of out (C-contiguous uint32 words, a row per mask; bit t % 32 of
    word t // 32 set iff token t is allowed), computed by the crate in parallel without the GIL, BOS's bit set for a
    candidate end (ends decides it), then each intersected with its healing Reemit's mask while that owes bytes;
    whether each row allows some token. The assertion is the crate's row writes' precondition: every row within out."""
    assert out.dtype == np.uint32 and out.flags.c_contiguous and out.ndim == 2 and max(rows) < len(out)
    allowed = crate_module(CRATE).masks([c.native for c in constraints], out.ctypes.data, rows, out.shape[1])
    for i, (c, row) in enumerate(zip(constraints, rows, strict=True)):
        if c.healing is not None and (healing := c.healing.mask()) is not None:
            out[row] &= healing
            allowed[i] = bool(out[row].any())
    return allowed
