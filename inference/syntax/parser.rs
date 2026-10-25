//! The parser: Lark's LALR(1) tables of the grammar (inference/syntax/python.lark) read as a viable-prefix
//! recognizer, with tokenizer.c's INDENT/DEDENT (tab size 8, the tab-consistency check at tab size 1, a backslash
//! line's first column) and its bracket level (newlines inside brackets are not NEWLINE tokens). A soft keyword's
//! lexeme has two terminals (a name and the keyword): every stack takes each, so the stacks fork until one dies.

use crate::lexer::{Ev, Lexicon, Terms};

const TAB_SIZE: u32 = 8; // tokenizer.c TABSIZE; ALTTABSIZE is 1: a tab is one alt column, as a space
/// an action table entry with no action
const NO_ACTION: i32 = i32::MIN;

/// Lark's parse table: actions[state * terminals + terminal] a state to shift to (>= 0) or !rule to reduce by (or
/// NO_ACTION); goto[state * nonterminals + nonterminal]; each rule's (length, left-hand side).
pub struct Tables {
    pub terminals: usize,
    pub nonterminals: usize,
    pub actions: Vec<i32>,
    pub goto: Vec<i32>,
    pub rules: Vec<(u32, u32)>,
    pub start: u32,
    pub end: u32,
    pub t_end: u16,
    pub t_newline: u16,
    pub t_indent: u16,
    pub t_dedent: u16,
}

/// A parser state: the LR stacks (bottom first), bracket depth, indentation stack of (column, alt column), whether the
/// line has tokens, the line's column and alt column so far, the column of the line's first backslash (0: none).
#[derive(Clone)]
pub struct Parse {
    pub stacks: Vec<Vec<u32>>,
    pub depth: i32,
    pub indents: Vec<(u32, u32)>,
    pub line: bool,
    pub col: u32,
    pub alt: u32,
    pub cont: u32,
}

impl Tables {
    fn action(&self, state: u32, terminal: u16) -> i32 {
        self.actions[state as usize * self.terminals + terminal as usize]
    }

    fn goto(&self, state: u32, lhs: u32) -> u32 {
        self.goto[state as usize * self.nonterminals + lhs as usize] as u32
    }

    pub fn start(&self) -> Parse {
        Parse { stacks: vec![vec![self.start]], depth: 0, indents: vec![(0, 0)], line: false, col: 0, alt: 0, cont: 0 }
    }

    /// The LR stack after terminal (reductions, then its shift), None on a syntax error. The stack is
    /// stack[..base] + tail while reducing, so only the result is allocated.
    pub fn shift(&self, stack: &[u32], terminal: u16) -> Option<Vec<u32>> {
        let mut base = stack.len();
        let mut tail: Vec<u32> = Vec::new();
        loop {
            let top = *tail.last().unwrap_or_else(|| &stack[base - 1]);
            let action = self.action(top, terminal);
            if action == NO_ACTION {
                return None;
            }
            if action >= 0 {
                let mut out = Vec::with_capacity(base + tail.len() + 1);
                out.extend_from_slice(&stack[..base]);
                out.extend_from_slice(&tail);
                out.push(action as u32);
                return Some(out);
            }
            let (size, lhs) = self.rules[!action as usize];
            let from_tail = (size as usize).min(tail.len());
            tail.truncate(tail.len() - from_tail);
            base -= size as usize - from_tail;
            let top = *tail.last().unwrap_or_else(|| &stack[base - 1]);
            tail.push(self.goto(top, lhs));
        }
    }

    /// Whether the LR stack is a whole module (lalr_parser_state.ParserState.feed_token at $END).
    pub fn accepts_end(&self, stack: &[u32]) -> bool {
        let mut stack = stack.to_vec();
        loop {
            let action = self.action(*stack.last().unwrap(), self.t_end);
            if action == NO_ACTION || action >= 0 {
                return false;
            }
            let (size, lhs) = self.rules[!action as usize];
            stack.truncate(stack.len() - size as usize);
            let g = self.goto(*stack.last().unwrap(), lhs);
            stack.push(g);
            if g == self.end {
                return true;
            }
        }
    }

    pub fn shift_all(&self, stacks: &[Vec<u32>], terminal: u16) -> Vec<Vec<u32>> {
        stacks.iter().filter_map(|s| self.shift(s, terminal)).collect()
    }

    /// The parser state after a lexer event, None on a syntax error.
    pub fn feed(&self, lexicon: &Lexicon, state: &Parse, event: Ev) -> Option<Parse> {
        match event {
            Ev::Lex(lexeme) => {
                let mut indents_changed = None;
                let mut stacks_changed: Option<Vec<Vec<u32>>> = None;
                if !state.line {
                    // the line's first token (outside brackets: a bracket's lexeme sets line): tokenizer.c's
                    // indentation check
                    let (col, alt) = if state.cont != 0 { (state.cont, state.cont) } else { (state.col, state.alt) };
                    let (top, top_alt) = *state.indents.last().unwrap();
                    if col == top {
                        if alt != top_alt {
                            return None;
                        }
                    } else if col > top {
                        if alt <= top_alt {
                            return None;
                        }
                        let mut indents = state.indents.clone();
                        indents.push((col, alt));
                        indents_changed = Some(indents);
                        stacks_changed = Some(self.shift_all(&state.stacks, self.t_indent));
                    } else {
                        let mut indents = state.indents.clone();
                        let mut stacks = state.stacks.clone();
                        while col < indents.last().unwrap().0 {
                            indents.pop();
                            stacks = self.shift_all(&stacks, self.t_dedent);
                        }
                        if *indents.last().unwrap() != (col, alt) {
                            return None;
                        }
                        indents_changed = Some(indents);
                        stacks_changed = Some(stacks);
                    }
                }
                let stacks_in = stacks_changed.as_deref().unwrap_or(&state.stacks);
                let terminals = &lexicon.lexemes[lexeme as usize];
                let mut stacks = Vec::with_capacity(stacks_in.len() * terminals.len());
                for s in stacks_in {
                    for &t in terminals {
                        if let Some(n) = self.shift(s, t) {
                            stacks.push(n);
                        }
                    }
                }
                // never negative: the grammar pairs every closer with an opener before it
                let depth = state.depth + lexicon.depth_change[lexeme as usize] as i32;
                if stacks.is_empty() {
                    return None;
                }
                let indents = indents_changed.unwrap_or_else(|| state.indents.clone());
                Some(Parse { stacks, depth, indents, line: true, col: 0, alt: 0, cont: 0 })
            }
            Ev::Nl => {
                if state.depth != 0 {
                    return Some(state.clone());
                }
                let stacks = if state.line {
                    let stacks = self.shift_all(&state.stacks, self.t_newline);
                    if stacks.is_empty() {
                        return None;
                    }
                    stacks
                } else {
                    state.stacks.clone()
                };
                Some(Parse { stacks, depth: 0, indents: state.indents.clone(), line: false, col: 0, alt: 0, cont: 0 })
            }
            _ if state.line => Some(state.clone()),
            Ev::Cont => Some(Parse { cont: if state.cont != 0 { state.cont } else { state.col }, ..state.clone() }),
            Ev::Space => Some(Parse { col: state.col + 1, alt: state.alt + 1, ..state.clone() }),
            Ev::Tab => Some(Parse { col: (state.col / TAB_SIZE + 1) * TAB_SIZE, alt: state.alt + 1, ..state.clone() }),
            Ev::FormFeed => Some(Parse { col: 0, alt: 0, ..state.clone() }),
        }
    }

    /// Whether the parser state takes one of the partial token's lexeme sequences next (any state if none).
    pub fn viable(&self, lexicon: &Lexicon, state: &Parse, terms: Terms) -> bool {
        let set = &lexicon.terms[terms as usize];
        set.is_empty() || set.iter().any(|sequence| {
            let mut s = state.clone();
            for &lexeme in sequence {
                match self.feed(lexicon, &s, Ev::Lex(lexeme)) {
                    Some(n) => s = n,
                    None => return false,
                }
            }
            true
        })
    }
}
