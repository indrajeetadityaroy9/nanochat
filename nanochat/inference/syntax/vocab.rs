//! Masks: for each lexer state, every vocabulary token's events (lexemes emitted, newlines, line-start whitespace) and
//! end state are computed once over a byte trie of the vocabulary, the subtrie after each first byte shared by the
//! lexer states that byte takes to one state (Syntax::grouping, Syntax::subtree); a step then runs the parser once per
//! distinct event sequence and maps the verdicts back to tokens.

use std::sync::{Arc, OnceLock, RwLock};

use rustc_hash::FxHashMap;

use crate::lexer::{Ev, Lex, Lexicon, Sub, Terms};
use crate::parser::{Parse, Tables};

/// A key of a token's verdict: its events, and the beginnings of the partial token it ends with (None: no character
/// completes its pending UTF-8 sequence, so no module continues).
type Key = (Vec<Ev>, Option<Terms>);

struct Node {
    children: Vec<(u8, u32)>,
    ids: Vec<u32>,
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
    group_terms: Vec<Option<Terms>>,
}

const NO_GROUP: u16 = u16::MAX;

type Memo<K, V> = RwLock<FxHashMap<K, Arc<OnceLock<Arc<V>>>>>;

/// The value of key in memo, computed once by make (other threads asking for it meanwhile wait for it).
fn memoized<K: std::hash::Hash + Eq + Clone, V>(memo: &Memo<K, V>, key: &K, make: impl FnOnce() -> V) -> Arc<V> {
    let cell = memo.read().unwrap().get(key).cloned();
    let cell = cell.unwrap_or_else(|| memo.write().unwrap().entry(key.clone()).or_default().clone());
    cell.get_or_init(|| Arc::new(make())).clone()
}

/// The Python 3.12 syntax of a tokenizer's vocabulary: the lexicon, the LALR tables, the vocabulary's byte trie, and
/// the per-lexer-state token groupings, filled on first use and shared by every constraint (and thread).
pub struct Syntax {
    pub lexicon: Lexicon,
    pub tables: Tables,
    pub vocab_size: usize,
    pub bos: u32,
    pub token_bytes: Vec<Vec<u8>>,
    trie: Vec<Node>,
    groupings: Memo<Lex, Grouping>,
    subtrees: Memo<(u8, Lex), Subtree>,
}

/// A constraint's state: its text's lexer and parser states.
#[derive(Clone)]
pub struct State {
    pub lex: Lex,
    pub parse: Parse,
}

impl Syntax {
    /// token_bytes: the mergeable tokens' bytes (ids below bos); vocab_size counts the special tokens too.
    pub fn new(lexicon: Lexicon, tables: Tables, token_bytes: Vec<Vec<u8>>, vocab_size: usize, bos: u32) -> Syntax {
        let mut trie = vec![Node { children: Vec::new(), ids: Vec::new() }];
        for (id, bytes) in token_bytes.iter().enumerate() {
            let mut node = 0usize;
            for &b in bytes {
                node = match trie[node].children.iter().find(|&&(c, _)| c == b) {
                    Some(&(_, child)) => child as usize,
                    None => {
                        let child = trie.len();
                        trie.push(Node { children: Vec::new(), ids: Vec::new() });
                        trie[node].children.push((b, child as u32));
                        child
                    }
                };
            }
            trie[node].ids.push(id as u32);
        }
        Syntax {
            lexicon, tables, vocab_size, bos, token_bytes, trie,
            groupings: RwLock::default(), subtrees: RwLock::default(),
        }
    }

    pub fn start(&self) -> State {
        State { lex: Lex::start(), parse: self.tables.start() }
    }

    /// The parser state after events, None on a syntax error.
    pub fn feed_all(&self, parse: &Parse, events: &[Ev]) -> Option<Parse> {
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
        match self.feed_all(&state.parse, &ev) {
            Some(parse) => self.tables.viable(&self.lexicon, &parse, terms),
            None => false, // the events before the partial token do not parse: nothing can continue (as `allowed`)
        }
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
            stacks = stacks.iter().filter_map(|s| self.tables.shift(s, self.tables.t_dedent)).collect();
        }
        stacks.iter().any(|s| self.tables.accepts_end(s))
    }

    /// The tokens starting with byte b, read from the lexer state after b. Shared by every lexer state that b takes
    /// to state: from most states, the first byte ends the partial token and the lexers converge.
    fn subtree(&self, b: u8, state: &Lex) -> Arc<Subtree> {
        memoized(&self.subtrees, &(b, state.clone()), || {
            let mut sub = Subtree { ids: Vec::new(), keys: Vec::new(), index: Vec::new() };
            let mut keys: FxHashMap<Key, u32> = FxHashMap::default();
            let root = self.trie[0].children.iter().find(|&&(c, _)| c == b).unwrap().1;
            let mut events = Vec::new();
            self.visit(root, state, &mut events, &mut sub, &mut keys);
            sub.keys.resize(keys.len(), (Vec::new(), None));
            for (key, k) in keys {
                sub.keys[k as usize] = key;
            }
            sub
        })
    }

    fn visit(&self, node: u32, state: &Lex, events: &mut Vec<Ev>, sub: &mut Subtree, keys: &mut FxHashMap<Key, u32>) {
        let node = &self.trie[node as usize];
        if !node.ids.is_empty() {
            let mut key_events = events.clone();
            let terms = self.lexicon.partial(state, &mut key_events);
            if terms.is_none() {
                key_events.truncate(events.len());
            }
            let n = keys.len() as u32;
            let k = *keys.entry((key_events, terms)).or_insert(n);
            sub.ids.extend_from_slice(&node.ids);
            sub.index.extend(std::iter::repeat_n(k, node.ids.len()));
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
            let mut group_of = vec![NO_GROUP; self.vocab_size];
            let mut groups: FxHashMap<Key, u32> = FxHashMap::default();
            let mut order: Vec<Key> = Vec::new();
            let mut ev = Vec::new();
            for &(b, _) in &self.trie[0].children {
                ev.clear();
                let Some(next) = self.lexicon.byte(lex, b, &mut ev) else { continue };
                let sub = self.subtree(b, &next);
                let remap: Vec<u32> = sub.keys.iter().map(|(events, terms)| {
                    let mut key_events = ev.clone();
                    key_events.extend_from_slice(events);
                    let key = (key_events, *terms);
                    *groups.entry(key.clone()).or_insert_with(|| {
                        order.push(key);
                        (order.len() - 1) as u32
                    })
                }).collect();
                for (&id, &k) in sub.ids.iter().zip(&sub.index) {
                    group_of[id as usize] = remap[k as usize] as u16;
                }
            }
            assert!(order.len() < NO_GROUP as usize);
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

    /// The mask words (bit t % 32 of word t / 32: token t) of the mergeable tokens that keep the text viable from the
    /// state; special tokens unset.
    pub fn allowed(&self, state: &State) -> Vec<u32> {
        let g = self.grouping(&state.lex);
        let mut states: Vec<Option<Parse>> = Vec::with_capacity(g.parent.len());
        states.push(Some(state.parse.clone()));
        for i in 1..g.parent.len() {
            let p = states[g.parent[i] as usize].as_ref().and_then(|p| self.tables.feed(&self.lexicon, p, g.event[i]));
            states.push(p);
        }
        let ok: Vec<bool> = g.group_node.iter().zip(&g.group_terms).map(|(&node, &terms)| {
            match (terms, &states[node as usize]) {
                (Some(terms), Some(parse)) => self.tables.viable(&self.lexicon, parse, terms),
                _ => false,
            }
        }).collect();
        let mut words = vec![0u32; self.vocab_size / 32];
        for (t, &k) in g.group_of.iter().enumerate() {
            if k != NO_GROUP && ok[k as usize] {
                words[t / 32] |= 1 << (t % 32);
            }
        }
        words
    }
}
