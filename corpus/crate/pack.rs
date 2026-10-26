//! The packer: one curated shard of a tokenized split cut into row-sized pieces, repository train pieces drawn as
//! context-level FIM examples (arXiv 2207.14255 §3.2) with AST-FIM middles (arXiv 2506.00204 §3.3, §5.1), placed in
//! rows by Best-Fit-Decreasing (arXiv 2404.10830 §3.1).

use std::cell::OnceCell;
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fs::File;

use memmap2::Mmap;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rand::distr::weighted::WeightedIndex;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use tiktoken_rs::CoreBPE;
use tree_sitter::{Language, Node, Parser, Tree};

use crate::{Lang, PerWorker};

/// A language's tree-sitter grammar and its containers: node kinds whose named children are whole statements or
/// declarations.
struct Grammar {
    language: Language,
    containers: FxHashSet<u16>,
    comment: u16,
    /// fields of a container's header children, which are not items of its sequence
    headers: Vec<u16>,
}

impl Grammar {
    fn new(lang: Lang) -> Self {
        let containers: &[&str] = match lang {
            Lang::Python => &["module", "block"],
            Lang::Cpp => &["translation_unit", "compound_statement", "declaration_list", "field_declaration_list", "preproc_if", "preproc_ifdef",
                           "preproc_else", "preproc_elif", "preproc_elifdef", "case_statement"],
        };
        let language = lang.language();
        let kind = |name: &str| language.id_for_node_kind(name, true);
        let containers = containers.iter().map(|&name| kind(name)).collect();
        let comment = kind("comment");
        let headers = ["condition", "name", "value", "alternative"].iter().map(|&name| language.field_id_for_name(name).unwrap().get()).collect();
        Grammar { language, containers, comment, headers }
    }

    fn parse(&self, text: &[u8]) -> Tree {
        let mut parser = Parser::new();
        parser.set_language(&self.language).unwrap();
        parser.parse(text, None).unwrap()
    }
}

/// The byte ranges between consecutive items of every container (and the root's leading and trailing ranges); no range
/// borders an ERROR node.
fn statement_gaps(tree: &Tree, grammar: &Grammar, text: &[u8]) -> Vec<(usize, usize)> {
    let root = tree.root_node();
    let end = |node: &Node| node.end_byte() - (node.end_byte() > 0 && text[node.end_byte() - 1] == b'\n') as usize;
    let (mut gaps, mut stack, mut cursor) = (Vec::new(), vec![root], root.walk());
    while let Some(node) = stack.pop() {
        let children: Vec<Node> = node.named_children(&mut cursor).collect();
        if grammar.containers.contains(&node.kind_id()) {
            let headers: Vec<Node> = grammar.headers.iter().filter_map(|&field| node.child_by_field_id(field)).collect();
            let items: Vec<Node> = children.iter().copied().filter(|child| child.kind_id() != grammar.comment && !headers.contains(child)).collect();
            gaps.extend(items.windows(2).filter(|pair| !pair[0].is_error() && !pair[1].is_error()).map(|pair| (end(&pair[0]), pair[1].start_byte())));
            if node == root {
                gaps.push((0, items.first().map_or(text.len(), |item| item.start_byte())));
                gaps.push((items.last().map_or(0, end), text.len()));
            }
        }
        stack.extend(children);
    }
    gaps
}

/// Positions q (between ids[q - 1] and ids[q]) after the first line-ending token whose byte offset lies in each
/// statement gap (start < offset <= end).
fn statement_cuts(ids: &[u16], token_bytes: &[Vec<u8>], ends_line: &[bool], grammar: &Grammar) -> Vec<usize> {
    let mut text = Vec::new();
    let mut ends = Vec::with_capacity(ids.len()); // byte offset after each token
    for &id in ids {
        text.extend_from_slice(&token_bytes[id as usize]);
        ends.push(text.len());
    }
    let tree = grammar.parse(&text);
    let mut gaps = statement_gaps(&tree, grammar, &text);
    gaps.sort_unstable(); // disjoint, so in order
    gaps.dedup(); // an empty root's leading and trailing ranges are one
    let mut g = 0;
    let mut cuts = Vec::new();
    for q in 1..ids.len() {
        if ends_line[ids[q - 1] as usize] {
            let at = ends[q - 1];
            while g < gaps.len() && gaps[g].1 < at {
                g += 1;
            }
            if g < gaps.len() && gaps[g].0 < at {
                cuts.push(q);
                g += 1;
            }
        }
    }
    cuts
}

/// Fill-in-the-middle shares (arXiv 2506.00204 §5.1, §3.3), from pack.py: FIM rate, Rand-FIM middles, Single-Node vs
/// Aligned-Span, line-start extension, SPM vs PSM; ids of <｜fim▁begin｜>, <｜fim▁hole｜>, <｜fim▁end｜>.
#[derive(FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct Fim {
    rate: f64,
    random: f64,
    single: f64,
    line_start: f64,
    spm: f64,
    sentinels: [u32; 3],
}

/// ChaCha8 keyed by (document, piece), the rest of the key zero: Some(the example's generator) with probability rate.
fn fim_draw(fim: &Fim, document: usize, piece: usize) -> Option<ChaCha8Rng> {
    let mut key = [0; 32];
    key[..8].copy_from_slice(&(document as u64).to_le_bytes());
    key[8..16].copy_from_slice(&(piece as u64).to_le_bytes());
    let mut rng = ChaCha8Rng::from_seed(key);
    rng.random_bool(fim.rate).then_some(rng)
}

/// Two uniform character boundaries of text, in order (arXiv 2207.14255 §3).
fn random_span(text: &str, rng: &mut ChaCha8Rng) -> (usize, usize) {
    let bounds: Vec<usize> = text.char_indices().map(|(at, _)| at).chain([text.len()]).collect();
    let (i, j) = (rng.random_range(0..bounds.len()), rng.random_range(0..bounds.len()));
    (bounds[i.min(j)], bounds[i.max(j)])
}

/// The AST-FIM middle [a, b) of text (arXiv 2506.00204 §3.3): Single-Node with probability single (byte-weighted node),
/// else Aligned-Span (max-IoU run of children); None when text has errors or no named child (§5.1: Rand-FIM).
fn ast_middle(text: &str, grammar: &Grammar, fim: &Fim, rng: &mut ChaCha8Rng) -> Option<(usize, usize)> {
    let tree = grammar.parse(text.as_bytes());
    let root = tree.root_node();
    if root.has_error() || root.named_child_count() == 0 {
        return None;
    }
    Some(if rng.random_bool(fim.single) {
        let mut nodes = Vec::new();
        let mut stack: Vec<Node> = root.named_children(&mut root.walk()).collect();
        while let Some(node) = stack.pop() {
            stack.extend(node.named_children(&mut node.walk()));
            nodes.push(node);
        }
        let node = nodes[rng.sample(WeightedIndex::new(nodes.iter().map(|node| node.byte_range().len())).unwrap())];
        (node.start_byte(), node.end_byte())
    } else {
        let (s, e) = random_span(text, rng);
        let node = root.named_descendant_for_byte_range(s, e).unwrap();
        let children: Vec<Node> = node.named_children(&mut node.walk()).collect();
        let (starts, ends): (Vec<usize>, Vec<usize>) = children.iter().map(|child| (child.start_byte(), child.end_byte())).unzip();
        // For a fixed end, IoU peaks at the last start <= s or the first start >= s; likewise the end: four candidates.
        let first = [starts.partition_point(|&a| a <= s).checked_sub(1), Some(starts.partition_point(|&a| a < s)).filter(|&i| i < children.len())];
        let last = [ends.partition_point(|&b| b <= e).checked_sub(1), Some(ends.partition_point(|&b| b < e)).filter(|&j| j < children.len())];
        let (mut best, mut best_iou) = ((node.start_byte(), node.end_byte()), (0, 1));
        for i in first.into_iter().flatten() {
            for j in last.into_iter().flatten().filter(|&j| j >= i) {
                let (a, b) = (starts[i], ends[j]);
                let iou = (b.min(e).saturating_sub(a.max(s)), b.max(e) - a.min(s));
                if iou.0 * best_iou.1 > best_iou.0 * iou.1 {
                    (best, best_iou) = ((a, b), iou);
                }
            }
        }
        best
    })
}

/// The FIM example of a piece's text, sections encoded separately (arXiv 2207.14255 §3; SPM is App. D, variant 2):
///   PSM: <｜fim▁begin｜> prefix <｜fim▁hole｜> suffix <｜fim▁end｜> middle
///   SPM: <｜fim▁begin｜> <｜fim▁hole｜> suffix <｜fim▁end｜> prefix middle
fn fim_example(text: &str, fim: &Fim, grammar: &Grammar, bpe: &CoreBPE, mut rng: ChaCha8Rng) -> Vec<u16> {
    let spm = rng.random_bool(fim.spm);
    let random = rng.random_bool(fim.random);
    let ast = (!random).then(|| ast_middle(text, grammar, fim, &mut rng)).flatten();
    let (mut a, b) = ast.unwrap_or_else(|| random_span(text, &mut rng));
    if ast.is_some() && rng.random_bool(fim.line_start) {
        let line = text[..a].rfind('\n').map_or(0, |i| i + 1);
        if text[line..a].bytes().all(|c| c == b' ' || c == b'\t') {
            a = line;
        }
    }
    let (prefix, middle, suffix) = (bpe.encode_ordinary(&text[..a]), bpe.encode_ordinary(&text[a..b]), bpe.encode_ordinary(&text[b..]));
    let [pre, suf, mid] = fim.sentinels;
    let sections: &[&[u32]] = if spm { &[&[pre, suf], &suffix, &[mid], &prefix, &middle] } else { &[&[pre], &prefix, &[suf], &suffix, &[mid], &middle] };
    sections.iter().flat_map(|section| section.iter().map(|&id| id as u16)).collect() // ids fit uint16 (a tokenized split's tokens)
}

type Piece = ([usize; 2], Option<Vec<u16>>);

/// What a document's pieces are cut from.
struct Split<'a> {
    tokens: &'a [u16],
    token_bytes: &'a [Vec<u8>],
    ends_line: &'a [bool],
    starts_char: &'a [bool],
    row_len: usize,
    grammar: Option<&'a Grammar>,
    fim: Option<(&'a Fim, &'a PerWorker<'a, CoreBPE>)>,
}

impl Split<'_> {
    /// The text of tokens [start, end) (context-level FIM decodes a piece, arXiv 2207.14255 App. C).
    fn text(&self, start: usize, end: usize) -> String {
        String::from_utf8(self.tokens[start..end].iter().flat_map(|&id| self.token_bytes[id as usize].iter().copied()).collect()).unwrap()
    }

    /// The pieces covering document `document`, tokens [start, end), each cut at the window's last statement boundary,
    /// else line end, else character start; a FIM piece's window shrinks until BOS, example and target BOS fit.
    fn pieces(&self, document: usize, start: usize, end: usize) -> Vec<Piece> {
        let statements = OnceCell::new();
        let cut = |s: usize, window: usize| {
            let window = window.min(end - s - 1); // a cut leaves at least one token to the rest
            let statements: &Vec<usize> = statements.get_or_init(|| match self.grammar {
                Some(grammar) => statement_cuts(&self.tokens[start..end], self.token_bytes, self.ends_line, grammar).into_iter().map(|q| start + q).collect(),
                None => Vec::new(),
            });
            let i = statements.partition_point(|&q| q <= s + window);
            if i > 0 && statements[i - 1] > s {
                statements[i - 1]
            } else if let Some(last) = self.tokens[s..s + window].iter().rposition(|&id| self.ends_line[id as usize]) {
                s + last + 1
            } else {
                (s + 1..=s + window).rev().find(|&q| self.starts_char[self.tokens[q] as usize]).unwrap()
            }
        };
        let (mut pieces, mut s) = (Vec::new(), start);
        while s < end {
            let piece = match self.fim.and_then(|(fim, encoders)| fim_draw(fim, document, pieces.len()).map(|rng| (fim, encoders, rng))) {
                None => {
                    let q = if end - s <= self.row_len - 2 { end } else { cut(s, self.row_len - 1) };
                    ([s, q], None)
                }
                Some((fim, encoders, rng)) => {
                    let mut window = self.row_len - 2 - fim.sentinels.len();
                    loop {
                        let q = if end - s <= window { end } else { cut(s, window) };
                        let example = fim_example(&self.text(s, q), fim, self.grammar.unwrap(), encoders.get(), rng.clone());
                        let excess = (1 + example.len() + 1).saturating_sub(self.row_len);
                        if excess == 0 {
                            break ([s, q], Some(example));
                        }
                        window -= excess;
                    }
                }
            };
            s = piece.0[1];
            pieces.push(piece);
        }
        pieces
    }
}

/// Rows of pieces of at most row_len tokens by Best-Fit-Decreasing (arXiv 2404.10830 §3.1); a row of targeted pieces
/// only keeps one slot for its final BOS. Each row as its piece indices in placement order.
fn best_fit_decreasing(lengths: &[usize], targeted: &[bool], row_len: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lengths.len()).collect();
    order.sort_by_key(|&p| Reverse(lengths[p]));
    // waiting rows by room: [targeted pieces only (net of the final BOS slot), with a cut piece]
    let mut rows: Vec<Vec<usize>> = Vec::new();
    let mut waiting: [BTreeMap<usize, Vec<usize>>; 2] = [BTreeMap::new(), BTreeMap::new()];
    let least = |map: &BTreeMap<usize, Vec<usize>>, need: usize| map.range(need..).next().map(|(&room, _)| room);
    for piece in order {
        let (length, cut) = (lengths[piece], !targeted[piece]);
        // a cut piece may also take a targeted row's reserved slot
        let offers = [least(&waiting[0], length - cut as usize).map(|room| (room + cut as usize, 0)), least(&waiting[1], length).map(|room| (room, 1))];
        let (free, row) = match offers.into_iter().flatten().min_by_key(|&(free, _)| free) {
            Some((free, kind)) => {
                let room = free - (kind == 0 && cut) as usize;
                let stack = waiting[kind].get_mut(&room).unwrap();
                let row = stack.pop().unwrap();
                if stack.is_empty() {
                    waiting[kind].remove(&room);
                }
                (free, row)
            }
            None => {
                rows.push(Vec::new());
                (row_len - !cut as usize, rows.len() - 1)
            }
        };
        let cut_row = cut || rows[row].iter().any(|&p| !targeted[p]);
        rows[row].push(piece);
        if free > length {
            waiting[cut_row as usize].entry(free - length).or_default().push(row);
        }
    }
    rows
}

/// One curated shard's documents first..last packed: per piece in row order [start, end, fim_start, fim_end]
/// (layout.PIECE_COLUMNS), pieces per row, FIM tokens. Targeted pieces precede cut pieces in each row.
fn pack_shard(split: &Split, offsets: &[i64], (first, last): (usize, usize)) -> (Vec<[i64; 4]>, Vec<i64>, Vec<u16>) {
    let mut longest: Vec<usize> = (first..last).collect();
    longest.sort_by_key(|&d| Reverse(offsets[d + 1] - offsets[d]));
    let mut documents: Vec<Vec<Piece>> = (first..last).map(|_| Vec::new()).collect();
    let cut: Vec<Vec<Piece>> = longest.par_iter().with_max_len(1).map(|&d| split.pieces(d, offsets[d] as usize, offsets[d + 1] as usize)).collect();
    for (d, pieces) in longest.into_iter().zip(cut) {
        documents[d - first] = pieces;
    }
    let (mut pieces, mut targeted) = (Vec::new(), Vec::new());
    for document in documents {
        let n = document.len();
        for (p, piece) in document.into_iter().enumerate() {
            targeted.push(p + 1 == n || piece.1.is_some());
            pieces.push(piece);
        }
    }
    let lengths: Vec<usize> = pieces.iter().map(|([s, e], example)| 1 + example.as_ref().map_or(0, Vec::len).max(e - s)).collect();
    let rows = best_fit_decreasing(&lengths, &targeted, split.row_len);
    let (mut index, mut fim_tokens) = (Vec::with_capacity(pieces.len()), Vec::new());
    for &p in rows.iter().flat_map(|row| row.iter().filter(|&&p| targeted[p]).chain(row.iter().filter(|&&p| !targeted[p]))) {
        let ([s, e], example) = &pieces[p];
        let at = fim_tokens.len();
        fim_tokens.extend_from_slice(example.as_deref().unwrap_or_default());
        index.push([*s as i64, *e as i64, at as i64, fim_tokens.len() as i64]);
    }
    (index, rows.iter().map(|row| row.len() as i64).collect(), fim_tokens)
}

/// Packs one curated shard (documents first..last) of a tokenized split into rows of row_len tokens.
/// Returns native-endian bytes: pack index (int64, 4 per piece), pieces per row (int64), FIM tokens (uint16).
#[pyfunction]
#[pyo3(signature = (tokens_path, offsets_path, shard, *, row_len, token_bytes, pattern, grammar, fim))]
#[allow(clippy::too_many_arguments)] // the keyword arguments of Python's call
pub(crate) fn pack<'py>(py: Python<'py>, tokens_path: &str, offsets_path: &str, shard: (usize, usize), row_len: usize, token_bytes: Vec<Vec<u8>>,
                        pattern: &str, grammar: Option<Lang>, fim: Option<Fim>)
                        -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>) {
    // safe: nothing writes a tokenized split's .bin or .idx while pack runs
    let tokens = unsafe { Mmap::map(&File::open(tokens_path).unwrap()).unwrap() };
    let offsets = unsafe { Mmap::map(&File::open(offsets_path).unwrap()).unwrap() };
    let ends_line: Vec<bool> = token_bytes.iter().map(|token| token.last() == Some(&b'\n')).collect();
    // a token starts a character unless its first byte is a UTF-8 continuation byte 10xxxxxx (RFC 3629 §3)
    let starts_char: Vec<bool> = token_bytes.iter().map(|token| token[0] & 0xC0 != 0x80).collect();
    let grammar = grammar.map(Grammar::new);
    let encoders = PerWorker::new(|| CoreBPE::new(token_bytes.iter().enumerate().map(|(rank, token)| (token.clone(), rank as u32)).collect(), Default::default(), pattern).unwrap());
    let (index, counts, fim_tokens) = py.detach(|| {
        let split = Split { tokens: bytemuck::cast_slice(&tokens), token_bytes: &token_bytes, ends_line: &ends_line, starts_char: &starts_char, row_len,
                            grammar: grammar.as_ref(), fim: fim.as_ref().map(|fim| (fim, &encoders)) };
        pack_shard(&split, bytemuck::cast_slice(&offsets), shard)
    });
    (PyBytes::new(py, bytemuck::cast_slice(&index)), PyBytes::new(py, bytemuck::cast_slice(&counts)), PyBytes::new(py, bytemuck::cast_slice(&fim_tokens)))
}
