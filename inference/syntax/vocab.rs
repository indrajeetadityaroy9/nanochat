//! Masks: each token's lexer events and end state computed once per lexer state over the vocabulary's byte trie; a
//! step runs the parser once per distinct event sequence and maps the verdicts back to tokens.

use std::sync::{Arc, OnceLock, RwLock};

use rustc_hash::FxHashMap;

use crate::lexer::{Ev, Lex, Lexicon, Sub, Terms};
use crate::parser::{Parse, Tables};

/// A key of a token's verdict: its events and its trailing partial token's beginnings. A token after which no
/// character completes the pending UTF-8 sequence (no module continues) has no key.
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

/// A lexer state's vocabulary: each token's group (NO_GROUP: lexically invalid), the groups' event-prefix tree (node 0
/// empty; parents first), each group's node and partial-token terms.
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

/// The Python 3.12 syntax of a tokenizer's vocabulary, its per-lexer-state groupings filled on first use; shared.
pub struct Syntax {
    lexicon: Lexicon,
    tables: Tables,
    /// BOS's id: the number of mergeable tokens (corpus/tokenizer.py)
    pub bos: u32,
    pub token_bytes: Vec<Vec<u8>>,
    trie: Vec<Node>,
    groupings: Memo<Lex, Grouping>,
    /// keyed by the trie node of the first byte
    subtrees: Memo<(u32, Lex), Subtree>,
}

/// A constraint's state: its text's lexer and parser states.
#[derive(Clone)]
pub struct State {
    pub lex: Lex,
    pub parse: Parse,
}

impl Syntax {
    /// token_bytes: the mergeable tokens' bytes (the ids below BOS's, each distinct).
    pub fn new(lexicon: Lexicon, tables: Tables, token_bytes: Vec<Vec<u8>>) -> Syntax {
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
        Syntax {
            lexicon, tables, bos: token_bytes.len() as u32, token_bytes, trie,
            groupings: RwLock::default(), subtrees: RwLock::default(),
        }
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

    /// Whether the state ends a module: the implied final line end, then EOF (tokenizer.c: DEDENTs, ENDMARKER).
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

    /// The tokens below a first byte's trie node, from the lexer state after it; shared by the lexer states the byte
    /// takes to one state.
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
}
