//! The packer (corpus/pack.py): one curated shard of a tokenized split at a time, its documents cut into pieces
//! that fit a training row, the pieces of a train split of a repository corpus drawn fill-in-the-middle examples
//! (context-level FIM: per piece, after cutting; arXiv 2207.14255 §3.2) whose middles AST-FIM chooses
//! (arXiv 2506.00204v1 §3.3, §5.1), and the pieces placed in rows by Best-Fit-Decreasing.

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

/// A source language's tree-sitter grammar and its containers: the node kinds whose named children are a sequence of
/// whole statements or declarations, so that a position between two of them cuts no statement. Python
/// (tree-sitter-python): a module, a block. C++ (tree-sitter-cpp): a translation unit, a compound statement, a
/// namespace's or linkage specification's declaration list, a class's member list, the bodies of conditional
/// preprocessor directives and the statements of a case label. Everything else is one item: a compound statement with
/// its header and its elif / else / except / finally / catch clauses, a decorated or template declaration, a
/// preprocessor directive.
struct Grammar {
    language: Language,
    containers: FxHashSet<u16>,
    comment: u16,
    /// the fields of a container's header children, not items of its sequence: the condition or name of a conditional
    /// preprocessor directive, its #else / #elif branch (a container of its own), a case label's value; the first child
    /// of a container in each field is a header (so is a Python match block's first case clause, field alternative)
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

/// The byte ranges between consecutive items of every container of a parsed document, where comments and blank lines
/// lie, and the root container's leading and trailing ranges. An item ends before its line break when its span holds
/// it (a preprocessor directive), as a Python statement does. A syntax error (an ERROR node) breaks only its own
/// sequence: no range between two items borders it, the rest of the document keeps its ranges. Comment lines that end
/// a block are part of it (tree-sitter attaches them to the block). Walked with a stack, as nesting depth is unbounded.
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
/// statement gap (start < offset <= end) of the document's text: the cut follows the item before the gap (in the root's
/// leading range, the document's first line), so the gap's other comments and blank lines stay with the item after it.
fn statement_cuts(ids: &[u16], token_bytes: &[Vec<u8>], ends_line: &[bool], grammar: &Grammar) -> Vec<usize> {
    let mut text = Vec::new();
    let mut ends = Vec::with_capacity(ids.len()); // byte offset after each token
    for &id in ids {
        text.extend_from_slice(&token_bytes[id as usize]);
        ends.push(text.len());
    }
    let tree = grammar.parse(&text);
    let mut gaps = statement_gaps(&tree, grammar, &text);
    gaps.sort_unstable(); // disjoint (a container's gaps lie inside an item of the container holding it), so in order
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

/// Fill-in-the-middle as AST-FIM trains it (arXiv 2506.00204v1 §5.1 "FIM", §3.3 "Masking Algorithm"), extracted from
/// pack.py's mapping: the share of pieces made FIM examples (rate); of those the share whose middle is a random
/// character span (random: Rand-FIM) rather than AST-aligned; of AST-aligned middles the share by Single-Node rather
/// than Aligned-Span masking (single); of AST-aligned middles whose node starts its line, the share that take the
/// line's indentation into the middle (line_start), so the prefix ends with the line break as in line-level infilling
/// (HumanEval single-line and multi-line infilling, arXiv 2207.14255 §5.1); the share in SPM rather than PSM order
/// (spm); the ids of <｜fim▁begin｜>, <｜fim▁hole｜>, <｜fim▁end｜> (sentinels).
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

/// The generator of piece `piece` of document `document`: ChaCha8 keyed by the pair (the rest of its 256-bit key
/// zero), so that every piece's draws are a stream of their own whatever the order of work; after its first draw, a FIM
/// example with probability rate: Some(the generator for the example's draws), else None (the piece is read left to
/// right).
fn fim_draw(fim: &Fim, document: usize, piece: usize) -> Option<ChaCha8Rng> {
    let mut key = [0; 32];
    key[..8].copy_from_slice(&(document as u64).to_le_bytes());
    key[8..16].copy_from_slice(&(piece as u64).to_le_bytes());
    let mut rng = ChaCha8Rng::from_seed(key);
    rng.random_bool(fim.rate).then_some(rng)
}

/// Two character boundaries of text drawn uniformly and independently, in order (arXiv 2207.14255 §3: a character-level
/// split at uniform positions).
fn random_span(text: &str, rng: &mut ChaCha8Rng) -> (usize, usize) {
    let bounds: Vec<usize> = text.char_indices().map(|(at, _)| at).chain([text.len()]).collect();
    let (i, j) = (rng.random_range(0..bounds.len()), rng.random_range(0..bounds.len()));
    (bounds[i.min(j)], bounds[i.max(j)])
}

/// The middle [a, b) of text by AST-FIM's masking (arXiv 2506.00204v1 §3.3) on its tree-sitter parse, whose named
/// nodes are the syntax tree (keywords and punctuation are tree-sitter's anonymous nodes); None when text does not
/// parse (§5.1: Rand-FIM for code that fails to parse) or has no subtree below its root (named nodes of a tree without
/// errors span at least one byte). With probability single, Single-Node masking: a named node below the root drawn with
/// probability proportional to its bytes; else Aligned-Span masking: a random character span (random_span), the lowest
/// named node holding it, and the run of that node's named children of the largest intersection over union with the
/// span, measured in bytes (the node itself when it has no named child or none of them meets the span).
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
        // For a fixed last child, IoU is largest at the last start <= s or the first start >= s (moving the start
        // further out grows the union only, further in shrinks the intersection only); likewise the end at the last
        // end <= e or the first end >= e. So the best run is one of these four.
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

/// The FIM example of a piece's text: its middle a random character span (random_span) with probability random, else
/// AST-FIM's (ast_middle; a random span where it has none), an AST middle whose node starts its line extended to the
/// line's start with probability line_start; prefix, suffix and middle each encoded on its own by tiktoken's encoder
/// (arXiv 2207.14255 §3), in SPM order with probability spm (arXiv 2207.14255 App. D, variant 2), else PSM:
///   PSM: <｜fim▁begin｜> prefix <｜fim▁hole｜> suffix <｜fim▁end｜> middle
///   SPM: <｜fim▁begin｜> <｜fim▁hole｜> suffix <｜fim▁end｜> prefix middle
/// The BOS before the example and its target BOS after the middle are the loader's (corpus/loader.py).
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

/// A piece of a document: its token range, and as a FIM example its tokens (fim_example).
type Piece = ([usize; 2], Option<Vec<u16>>);

/// What a document's pieces are cut from: the split's tokens, each token's bytes and whether it ends a line or starts a
/// character, the row length, the grammar of a repository corpus and, in its train split, fill-in-the-middle with the
/// tokenizer's encoder of each worker.
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
    /// The text of tokens [start, end): context-level FIM decodes a piece back to characters (arXiv 2207.14255 App. C).
    fn text(&self, start: usize, end: usize) -> String {
        String::from_utf8(self.tokens[start..end].iter().flat_map(|&id| self.token_bytes[id as usize].iter().copied()).collect()).unwrap()
    }

    /// The pieces covering document `document`, tokens [start, end), left to right. A piece holds the rest of the
    /// document when it fits its row, else ends at a cut in its window: the last statement cut of the window (the
    /// document's statement_cuts, computed on first need), else after its last line-ending token, else before the last
    /// token it reaches that starts a character, so that every piece is whole characters. A piece read left to right
    /// fits its row with its BOS, and when it ends the document the BOS after it, its target (a cut piece's next BOS
    /// is no target: corpus/loader.py). A piece drawn a FIM example (fim_draw) fits with its BOS, the example and the
    /// target BOS after its middle: its window first leaves room for the three sentinels; as the sections are encoded
    /// each on its own, the example's length is known once built, and an example longer than its row shrinks the
    /// window by the excess.
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

/// Rows of pieces of at most row_len tokens by Best-Fit-Decreasing ("Fewer Truncations Improve Language Modeling",
/// arXiv 2404.10830 §3.1): longest first (a stable sort: equal lengths keep document order), each piece into the row
/// with the fewest free slots that holds it, else a new row, so no piece is cut; rows of one kind with equal room wait
/// last-in first-out, and at equal free slots a row of targeted pieces only is taken before a row with a cut piece. A
/// row whose pieces all have the BOS after them as their target (targeted) keeps one slot for the BOS after its last
/// piece; a row holding a cut piece ends with it (pack_shard) and needs none. Each row as the indices of its pieces in
/// the order placed.
fn best_fit_decreasing(lengths: &[usize], targeted: &[bool], row_len: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lengths.len()).collect();
    order.sort_by_key(|&p| Reverse(lengths[p]));
    // rows waiting by their room: [rows of targeted pieces only (room net of the final BOS slot), rows with a cut
    // piece]
    let mut rows: Vec<Vec<usize>> = Vec::new();
    let mut waiting: [BTreeMap<usize, Vec<usize>>; 2] = [BTreeMap::new(), BTreeMap::new()];
    let least = |map: &BTreeMap<usize, Vec<usize>>, need: usize| map.range(need..).next().map(|(&room, _)| room);
    for piece in order {
        let (length, cut) = (lengths[piece], !targeted[piece]);
        // free slots each kind of row offers this piece: a cut piece may also take a targeted row's reserved slot
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

/// One curated shard's documents first..last packed: the pack index entry of every piece in row order (token range
/// [start, end) of the split, then the range of its FIM example in the shard's FIM tokens, empty for a piece read
/// left to right; corpus/layout.py PIECE_COLUMNS, pack.py offsets the FIM range into the split's), the pieces per row,
/// and the FIM examples' tokens in row order. A piece's BOS is followed by its tokens: a FIM example's when the loader
/// reads "fim", else (reading "left_to_right", and every piece read left to right) its range; it takes the longer of
/// the two in its row. A piece has the BOS after it as its target when it ends its document or is a FIM example (the
/// end of its middle); such pieces come first in their row and the pieces cut from their document left to right last,
/// so the BOS after a piece is its target. The documents are cut in parallel, longest first and one task each, so that
/// no long document starts last or queues behind others on one worker.
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

/// Packs one curated shard (documents first..last) of a tokenized split (its uint16 tokens and int64 document offsets
/// at the given paths) into rows of row_len tokens (pack_shard), its documents in parallel. token_bytes: every
/// mergeable token's bytes by id (= rank), with the split pattern the tokenizer's encoder (tiktoken-rs's CoreBPE:
/// OpenAI tiktoken's core, fancy-regex as in tiktoken) for FIM sections; grammar: a repository corpus's language (None
/// for other documents); fim: AST-FIM's shares and sentinel ids for its train split (Fim). Returns the pack index
/// entries (native-endian int64, 4 per piece), the pieces per row (int64) and the FIM tokens (uint16).
#[pyfunction]
#[pyo3(signature = (tokens_path, offsets_path, shard, *, row_len, token_bytes, pattern, grammar, fim))]
#[allow(clippy::too_many_arguments)] // the keyword arguments of Python's call
pub(crate) fn pack<'py>(py: Python<'py>, tokens_path: &str, offsets_path: &str, shard: (usize, usize), row_len: usize, token_bytes: Vec<Vec<u8>>,
                        pattern: &str, grammar: Option<Lang>, fim: Option<Fim>)
                        -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>) {
    // pack.py cuts a tokenized split only after its last pack call returns, so the files are not modified while mapped
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
