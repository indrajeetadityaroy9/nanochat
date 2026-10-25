//! Masks: for each lexer state, every vocabulary token's events (lexemes emitted, newlines, line-start whitespace) and
//! end state are computed once over a byte trie of the vocabulary, the subtrie after each first byte shared by the
//! lexer states that byte takes to one state (Syntax::grouping, Syntax::subtree); a step then runs the parser once per
//! distinct event sequence and maps the verdicts back to tokens.

use std::sync::{Arc, OnceLock, RwLock};

use rustc_hash::FxHashMap;

use crate::lexer::{Ev, Lex, Lexicon, Sub, Terms};
use crate::parser::{Parse, Tables};

/// A key of a token's verdict: its events, and the beginnings of the partial token it ends with. A token after which
/// no character completes the pending UTF-8 sequence (no module continues) has no key.
type Key = (Vec<Ev>, Terms);

struct Node {
    children: Vec<(u8, u32)>,
    id: Option<u32>,
}

/// The tokens starting with one byte, read from the lexer state after it: their ids, the distinct keys, each id's key.
struct Subtree {
    ids: Vec<u32>,
    keys: Vec<Key>,
    index: Vec<u32>,
}

/// A lexer state's vocabulary: each token's group (NO_GROUP: lexically invalid), the event-prefix tree of the groups'
/// event sequences (node 0 the empty sequence; a node's parent comes first), each group's node and partial-token terms.
struct Grouping {
    group_of: Vec<u16>,
    parent: Vec<u32>,
    event: Vec<Ev>,
    group_node: Vec<u32>,
    group_terms: Vec<Terms>,
}

const NO_GROUP: u16 = u16::MAX;

type Memo<K, V> = RwLock<FxHashMap<K, Arc<OnceLock<Arc<V>>>>>;

/// The value of key in memo, computed once by make (other threads asking for it meanwhile wait for it).
fn memoized<K: std::hash::Hash + Eq + Clone, V>(memo: &Memo<K, V>, key: &K, make: impl FnOnce() -> V) -> Arc<V> {
    let cell = memo.read().unwrap().get(key).cloned();
    let cell = cell.unwrap_or_else(|| memo.write().unwrap().entry(key.clone()).or_default().clone());
    cell.get_or_init(|| Arc::new(make())).clone()
}

/// The keys of an interning map (key -> index), in index order.
fn by_index(keys: FxHashMap<Key, u32>) -> Vec<Key> {
    let mut out = vec![(Vec::new(), 0); keys.len()];
    for (key, k) in keys {
        out[k as usize] = key;
    }
    out
}

/// The Python 3.12 syntax of a tokenizer's vocabulary: the lexicon, the LALR tables, the vocabulary's byte trie, and
/// the per-lexer-state token groupings, filled on first use and shared by every constraint (and thread).
pub struct Syntax {
    lexicon: Lexicon,
    tables: Tables,
    pub vocab_size: usize,
    pub bos: u32,
    pub token_bytes: Vec<Vec<u8>>,
    trie: Vec<Node>,
    groupings: Memo<Lex, Grouping>,
    /// keyed by the trie node of the first byte
    subtrees: Memo<(u32, Lex), Subtree>,
    /// the tokens holding a "\n": (id, the offsets of their "\n" bytes)
    newline_tokens: Vec<(u32, Vec<usize>)>,
}

/// A constraint's state: its text's lexer and parser states.
#[derive(Clone)]
pub struct State {
    pub lex: Lex,
    pub parse: Parse,
}

impl Syntax {
    /// token_bytes: the mergeable tokens' bytes (ids below bos, each distinct); vocab_size counts the special tokens
    /// too.
    pub fn new(lexicon: Lexicon, tables: Tables, token_bytes: Vec<Vec<u8>>, vocab_size: usize, bos: u32) -> Syntax {
        let mut trie = vec![Node { children: Vec::new(), id: None }];
        for (id, bytes) in token_bytes.iter().enumerate() {
            let mut node = 0usize;
            for &b in bytes {
                node = match trie[node].children.iter().find(|&&(c, _)| c == b) {
                    Some(&(_, child)) => child as usize,
                    None => {
                        let child = trie.len();
                        trie.push(Node { children: Vec::new(), id: None });
                        trie[node].children.push((b, child as u32));
                        child
                    }
                };
            }
            trie[node].id = Some(id as u32);
        }
        let newline_tokens = token_bytes.iter().enumerate()
            .map(|(id, b)| (id as u32, b.iter().enumerate().filter(|&(_, &c)| c == b'\n').map(|(i, _)| i).collect()))
            .filter(|(_, at): &(u32, Vec<usize>)| !at.is_empty()).collect();
        Syntax {
            lexicon, tables, vocab_size, bos, token_bytes, trie,
            groupings: RwLock::default(), subtrees: RwLock::default(), newline_tokens,
        }
    }

    /// The bytes of token up to its lines-th "\n" (lines > 0), None if it holds fewer (or lines is 0).
    pub fn cut(&self, token: u32, lines: u32) -> Option<&[u8]> {
        let bytes = self.token_bytes.get(token as usize)?;
        let n = (lines as usize).checked_sub(1)?;
        bytes.iter().enumerate().filter(|&(_, &c)| c == b'\n').nth(n).map(|(i, _)| &bytes[..=i])
    }

    /// For each token holding a lines-th "\n" (lines > 0): sets its bit (t % 8 of bits[t / 8]) iff its bytes up to
    /// that line end keep the text viable and, if check, the grammar ends a module after them and suffix; else clears
    /// it. Whether some bit was set. The cuts that leave one state (line ends after whitespace) share a verdict.
    pub fn line_ends(&self, state: &State, lines: u32, suffix: &[u8], bits: &mut [u8], check: bool) -> bool {
        let mut verdicts: FxHashMap<(Lex, Parse), bool> = FxHashMap::default();
        let mut any = false;
        for (id, at) in &self.newline_tokens {
            let Some(&end) = at.get(lines as usize - 1) else { continue };
            let ok = self.extend(state, &self.token_bytes[*id as usize][..=end]).is_ok_and(|s| if check {
                let key = (s.lex.clone(), s.parse.clone());
                *verdicts.entry(key).or_insert_with(|| self.complete_with(&s, suffix, true))
            } else {
                self.viable_end(&s)
            });
            let (byte, bit) = (*id as usize / 8, 1u8 << (id % 8));
            if ok {
                bits[byte] |= bit;
                any = true;
            } else {
                bits[byte] &= !bit;
            }
        }
        any
    }

    pub fn start(&self) -> State {
        State { lex: Lex::start(), parse: self.tables.start() }
    }

    /// The parser state after events, None on a syntax error.
    fn feed_all(&self, parse: &Parse, events: &[Ev]) -> Option<Parse> {
        let mut parse = parse.clone();
        for &e in events {
            parse = self.tables.feed(&self.lexicon, &parse, e)?;
        }
        Some(parse)
    }

    /// The state after data, Err(i) if the text stops being a prefix of a module at data[i].
    pub fn extend(&self, state: &State, data: &[u8]) -> Result<State, usize> {
        let (mut lex, mut parse) = (state.lex.clone(), state.parse.clone());
        let mut ev = Vec::new();
        for (i, &b) in data.iter().enumerate() {
            ev.clear();
            lex = self.lexicon.byte(&lex, b, &mut ev).ok_or(i)?;
            parse = self.feed_all(&parse, &ev).ok_or(i)?;
        }
        Ok(State { lex, parse })
    }

    /// Whether a state's partial token (and pending UTF-8 character) can still continue the text.
    pub fn viable_end(&self, state: &State) -> bool {
        let mut ev = Vec::new();
        let Some(terms) = self.lexicon.partial(&state.lex, &mut ev) else { return false };
        self.feed_all(&state.parse, &ev).is_some_and(|parse| self.tables.viable(&self.lexicon, &parse, terms))
    }

    /// Whether the state (ends_line: its text ends with a line end) ends a module: the implied final line end, then
    /// EOF (tokenizer.c: DEDENTs, ENDMARKER).
    pub fn complete(&self, state: &State, ends_line: bool) -> bool {
        let (lex, parse) = if ends_line {
            (state.lex.clone(), state.parse.clone())
        } else {
            let mut ev = Vec::new();
            let Some(lex) = self.lexicon.byte(&state.lex, b'\n', &mut ev) else { return false };
            let Some(parse) = self.feed_all(&state.parse, &ev) else { return false };
            (lex, parse)
        };
        if lex.npend != 0 || !lex.frames.is_empty() || lex.sub != Sub::Bol(false) || parse.depth != 0 {
            return false;
        }
        let mut stacks = parse.stacks;
        for _ in 1..parse.indents.len() {
            stacks = self.tables.shift_all(&stacks, self.tables.t_dedent);
        }
        stacks.iter().any(|s| self.tables.accepts_end(s))
    }

    /// The tokens below trie node root (a first byte's), read from the lexer state after that byte. Shared by every
    /// lexer state that the byte takes to state: from most states, the first byte ends the partial token and the
    /// lexers converge.
    fn subtree(&self, root: u32, state: &Lex) -> Arc<Subtree> {
        memoized(&self.subtrees, &(root, state.clone()), || {
            let mut sub = Subtree { ids: Vec::new(), keys: Vec::new(), index: Vec::new() };
            let mut keys: FxHashMap<Key, u32> = FxHashMap::default();
            let mut events = Vec::new();
            self.visit(root, state, &mut events, &mut sub, &mut keys);
            sub.keys = by_index(keys);
            sub
        })
    }

    fn visit(&self, node: u32, state: &Lex, events: &mut Vec<Ev>, sub: &mut Subtree, keys: &mut FxHashMap<Key, u32>) {
        let node = &self.trie[node as usize];
        if let Some(id) = node.id {
            let mut key_events = events.clone();
            if let Some(terms) = self.lexicon.partial(state, &mut key_events) {
                let n = keys.len() as u32;
                sub.ids.push(id);
                sub.index.push(*keys.entry((key_events, terms)).or_insert(n));
            }
        }
        let len = events.len();
        for &(b, child) in &node.children {
            if let Some(next) = self.lexicon.byte(state, b, events) {
                self.visit(child, &next, events, sub, keys);
            }
            events.truncate(len);
        }
    }

    /// Every vocabulary token's events and end state from lexer state lex (computed once per state).
    fn grouping(&self, lex: &Lex) -> Arc<Grouping> {
        memoized(&self.groupings, lex, || {
            let mut group_of = vec![NO_GROUP; self.token_bytes.len()];
            let mut groups: FxHashMap<Key, u32> = FxHashMap::default();
            let mut ev = Vec::new();
            for &(b, child) in &self.trie[0].children {
                ev.clear();
                let Some(next) = self.lexicon.byte(lex, b, &mut ev) else { continue };
                let sub = self.subtree(child, &next);
                let remap: Vec<u32> = sub.keys.iter().map(|(events, terms)| {
                    let mut key_events = ev.clone();
                    key_events.extend_from_slice(events);
                    let n = groups.len() as u32;
                    *groups.entry((key_events, *terms)).or_insert(n)
                }).collect();
                for (&id, &k) in sub.ids.iter().zip(&sub.index) {
                    group_of[id as usize] = remap[k as usize] as u16;
                }
            }
            assert!(groups.len() < NO_GROUP as usize);
            let order = by_index(groups);
            // the event-prefix tree: a node per distinct prefix of the groups' event sequences
            let (mut parent, mut event) = (vec![u32::MAX], vec![Ev::Nl]);
            let mut nodes: FxHashMap<(u32, Ev), u32> = FxHashMap::default();
            let group_node = order.iter().map(|(events, _)| {
                events.iter().fold(0u32, |at, &e| {
                    *nodes.entry((at, e)).or_insert_with(|| {
                        parent.push(at);
                        event.push(e);
                        (parent.len() - 1) as u32
                    })
                })
            }).collect();
            let group_terms = order.iter().map(|&(_, terms)| terms).collect();
            Grouping { group_of, parent, event, group_node, group_terms }
        })
    }

    /// Sets bit t % 8 of bits[t / 8] for each mergeable token t that keeps the text viable from the state.
    pub fn allowed(&self, state: &State, bits: &mut [u8]) {
        let g = self.grouping(&state.lex);
        let mut states: Vec<Option<Parse>> = Vec::with_capacity(g.parent.len());
        states.push(Some(state.parse.clone()));
        for i in 1..g.parent.len() {
            let p = states[g.parent[i] as usize].as_ref().and_then(|p| self.tables.feed(&self.lexicon, p, g.event[i]));
            states.push(p);
        }
        let ok: Vec<bool> = g.group_node.iter().zip(&g.group_terms).map(|(&node, &terms)| {
            states[node as usize].as_ref().is_some_and(|parse| self.tables.viable(&self.lexicon, parse, terms))
        }).collect();
        for (t, &k) in g.group_of.iter().enumerate() {
            if k != NO_GROUP && ok[k as usize] {
                bits[t / 8] |= 1 << (t % 8);
            }
        }
    }

    /// Whether the state followed by suffix ends a module (complete over the text and suffix; text_ends_line: the
    /// text before the suffix ends with a line end).
    pub fn complete_with(&self, state: &State, suffix: &[u8], text_ends_line: bool) -> bool {
        if suffix.is_empty() {
            return self.complete(state, text_ends_line);
        }
        match self.extend(state, suffix) {
            Ok(end) => self.complete(&end, matches!(suffix.last(), Some(b'\n' | b'\r'))),
            Err(_) => false,
        }
    }

    /// Clears bit t % 8 of bits[t / 8] for each mergeable token t that neither is a non-empty proper prefix of owed
    /// nor starts with owed (decoder.Reemit's tokens: owed's bytes re-emitted over one or more tokens).
    pub fn reemit(&self, owed: &[u8], bits: &mut [u8]) {
        let mut keep = vec![0u8; bits.len()];
        let mut set = |id: u32| keep[id as usize / 8] |= 1 << (id % 8);
        let mut node = 0usize;
        let mut whole = true;
        for (i, &b) in owed.iter().enumerate() {
            if i > 0 && let Some(id) = self.trie[node].id {
                set(id);
            }
            match self.trie[node].children.iter().find(|&&(c, _)| c == b) {
                Some(&(_, child)) => node = child as usize,
                None => {
                    whole = false;
                    break;
                }
            }
        }
        if whole {
            let mut stack = vec![node];
            while let Some(n) = stack.pop() {
                if let Some(id) = self.trie[n].id {
                    set(id);
                }
                stack.extend(self.trie[n].children.iter().map(|&(_, c)| c as usize));
            }
        }
        for (b, k) in bits.iter_mut().zip(&keep) {
            *b &= k;
        }
    }
}
