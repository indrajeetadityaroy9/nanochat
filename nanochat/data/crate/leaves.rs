//! The leaves of a code snippet's parse tree in TokEval's five categories (arXiv 2608.18062 §3.6; its library
//! github.com/cimeister/tokenizer-intrinsic-evals at 7d03a61, tokenizer_analysis/metrics/_treesitter_worker.py
//! classify_node and extract_leaf_spans), for the tokenizer ablation's code metrics (ablations/tokeval.py).
//! Parsed with this crate's grammars rather than tree-sitter-language-pack's; Python's str predicates are read as
//! Rust's char predicates (isalpha as is_alphabetic, isalnum as is_alphanumeric, strip as trim), equal on the texts
//! they test, a grammar's anonymous tokens and keywords, which are ASCII.

use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use tree_sitter::{Node, Parser};

use crate::Lang;

const IDENTIFIER_TYPES: [&str; 10] = ["identifier", "type_identifier", "field_identifier", "property_identifier",
    "shorthand_property_identifier", "shorthand_property_identifier_pattern", "variable_name", "name", "attribute",
    "word"];

const LITERAL_TYPES: [&str; 30] = ["string", "string_literal", "string_content", "string_fragment",
    "raw_string_literal", "char_literal", "character_literal", "template_string", "heredoc_body", "integer",
    "integer_literal", "decimal_integer_literal", "hex_integer_literal", "octal_integer_literal",
    "binary_integer_literal", "float", "float_literal", "decimal_floating_point_literal", "number", "number_literal",
    "true", "false", "none", "null", "nil", "boolean", "boolean_literal", "null_literal", "regex", "regex_pattern"];

const DELIMITERS: [&str; 8] = ["(", ")", "{", "}", "[", "]", ";", ","];

/// Punctuation that is syntactically not an operator in most languages.
const NON_OPERATOR_PUNCTUATION: [&str; 7] = [".", ":", "::", "@", "#", "...", "\\"];

/// Keywords that some grammars expose as named nodes.
const KNOWN_KEYWORDS: [&str; 90] = ["if", "else", "elif", "for", "while", "do", "switch", "case", "break", "continue",
    "return", "yield", "class", "struct", "enum", "interface", "trait", "impl", "def", "func", "fn", "function", "var",
    "let", "const", "static", "public", "private", "protected", "import", "from", "export", "module", "package", "try",
    "catch", "except", "finally", "throw", "throws", "raise", "async", "await", "match", "where", "in", "is", "as",
    "not", "and", "or", "with", "lambda", "new", "delete", "typeof", "instanceof", "void", "self", "this", "super",
    "then", "end", "local", "elsif", "unless", "until", "begin", "rescue", "ensure", "when", "val", "object", "extends",
    "override", "abstract", "final", "sealed", "lazy", "mut", "ref", "pub", "crate", "mod", "use", "extern", "type",
    "foreach", "namespace", "using"];

/// A leaf's category, in TokEval's order: 0 identifier, 1 keyword, 2 operator, 3 literal, 4 delimiter; None for the
/// rest (comments, structural punctuation, a named leaf of another kind).
fn category(node: &Node, text: &str) -> Option<u8> {
    let kind = node.kind();
    if text.is_empty() {
        return None;
    }
    if DELIMITERS.contains(&text) {
        return Some(4);
    }
    // identifiers before keywords: a kind the grammar calls an identifier stays one (Python's self)
    if IDENTIFIER_TYPES.contains(&kind) {
        return Some(0);
    }
    if LITERAL_TYPES.contains(&kind) {
        return Some(3);
    }
    let alpha = text.chars().all(char::is_alphabetic);
    // a keyword: an anonymous node whose kind is its alphabetic text, or a named one among the known keywords
    if alpha && (!node.is_named() && kind == text || node.is_named() && KNOWN_KEYWORDS.contains(&text)) {
        return Some(1);
    }
    let stripped = text.trim();
    if !node.is_named() && !text.chars().all(char::is_alphanumeric) && !stripped.is_empty()
        && !NON_OPERATOR_PUNCTUATION.contains(&stripped) {
        return Some(2);
    }
    None
}

/// A snippet's categorized leaves, (byte starts, byte ends, categories), and its number of ERROR nodes: regions the
/// grammar could not parse, whose leaves are not taken. Walked with a stack, as nesting depth is unbounded.
fn snippet_leaves(parser: &mut Parser, text: &str) -> (Vec<u32>, Vec<u32>, Vec<u8>, usize) {
    let tree = parser.parse(text, None).unwrap();
    let (mut starts, mut ends, mut categories, mut errors) = (Vec::new(), Vec::new(), Vec::new(), 0);
    let mut stack = vec![tree.root_node()];
    while let Some(node) = stack.pop() {
        if node.kind() == "ERROR" {
            errors += 1;
        } else if node.child_count() == 0 {
            let (start, end) = (node.start_byte(), node.end_byte());
            if let Some(c) = category(&node, &text[start..end]) {
                starts.push(start as u32);
                ends.push(end as u32);
                categories.push(c);
            }
        } else {
            stack.extend(node.children(&mut node.walk()));
        }
    }
    (starts, ends, categories, errors)
}

/// The categorized leaves of each snippet of the grammar's language: per snippet, its leaves' byte starts and ends
/// (native-endian uint32) and categories (uint8, category above) in tree order reversed, and its ERROR node count.
#[pyfunction]
#[pyo3(signature = (texts, *, grammar))]
pub(crate) fn leaves<'py>(py: Python<'py>, texts: Vec<PyBackedStr>, grammar: Lang)
                          -> Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, usize)> {
    let language = grammar.language();
    let parsed: Vec<_> = py.detach(|| texts.par_iter().map_init(|| {
        let mut parser = Parser::new();
        parser.set_language(&language).unwrap();
        parser
    }, |parser, text| snippet_leaves(parser, text)).collect());
    parsed.into_iter().map(|(starts, ends, categories, errors)| (PyBytes::new(py, bytemuck::cast_slice(&starts)),
        PyBytes::new(py, bytemuck::cast_slice(&ends)), PyBytes::new(py, &categories), errors)).collect()
}
