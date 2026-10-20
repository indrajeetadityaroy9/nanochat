//! The Python 3.12 syntax constraint, Python module nanochat_syntax (built and wrapped by
//! nanochat/inference/syntax_constraint.py): after each token, exactly the tokens whose bytes keep the text a prefix of
//! some module of the grammar python.lark (whose LALR(1) tables Lark builds), lexed as CPython 3.12.3's tokenizer does
//! (lexer.rs), indented and parsed as tokenizer.c and the tables say (parser.rs), over the vocabulary's byte trie
//! (vocab.rs). A batch of constraints' masks is computed in parallel (rayon) without the GIL (masks).

mod lexer;
mod parser;
mod vocab;

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;
use rayon::prelude::*;

/// The Python 3.12 syntax of a tokenizer's vocabulary (vocab::Syntax), shared by the constraints it builds.
#[pyclass(frozen)]
struct Syntax {
    inner: Arc<vocab::Syntax>,
}

#[pymethods]
impl Syntax {
    /// keywords, soft, operators: (word, its terminals); basic: the terminals of NAME, NUMBER, STRING, FSTRING_START,
    /// FSTRING_END; special: of $END, NEWLINE, INDENT, DEDENT; actions, goto: the dense parse table (rows of
    /// terminals, nonterminals entries); rules: (length, left-hand side); start, end: the LR start and end states;
    /// xid_start, xid_continue: the non-ASCII code point ranges of str.isidentifier() on c and on "a" + c;
    /// token_bytes: the mergeable tokens' bytes.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (*, keywords, soft, operators, basic, special, terminals, nonterminals, actions, goto, rules,
                        start, end, xid_start, xid_continue, token_bytes, vocab_size, bos))]
    fn new(keywords: Vec<(String, Vec<u16>)>, soft: Vec<(String, Vec<u16>)>, operators: Vec<(String, Vec<u16>)>,
           basic: [u16; 5], special: [u16; 4], terminals: usize, nonterminals: usize, actions: Vec<i32>,
           goto: Vec<i32>, rules: Vec<(u32, u32)>, start: u32, end: u32, xid_start: Vec<(u32, u32)>,
           xid_continue: Vec<(u32, u32)>, token_bytes: Vec<PyBackedBytes>, vocab_size: usize, bos: u32) -> Syntax {
        let lexicon = lexer::Lexicon::new(keywords, soft, operators, basic, xid_start, xid_continue);
        let [t_end, t_newline, t_indent, t_dedent] = special;
        let tables = parser::Tables {
            terminals, nonterminals, actions, goto, rules, start, end, t_end, t_newline, t_indent, t_dedent,
        };
        let token_bytes = token_bytes.iter().map(|b| b.to_vec()).collect();
        Syntax { inner: Arc::new(vocab::Syntax::new(lexicon, tables, token_bytes, vocab_size, bos)) }
    }

    /// The constraint of the tokens generated after prefix (UTF-8); suffix: whether a suffix follows the generation
    /// (then BOS is a candidate at every step). Raises ValueError when prefix is not a prefix of any module.
    fn constraint(&self, prefix: &[u8], suffix: bool) -> PyResult<Constraint> {
        let syntax = &self.inner;
        let state = syntax.extend(&syntax.start(), prefix).map_err(|i| PyValueError::new_err(
            format!("the text stops being a prefix of a Python 3.12 module at byte {i}")))?;
        if !syntax.viable_end(&state) {
            return Err(PyValueError::new_err(format!(
                "the prefix is not a prefix of a Python 3.12 module (at byte {})", prefix.len())));
        }
        Ok(Constraint { syntax: syntax.clone(), state: Arc::new(state), text: prefix.to_vec(), suffix })
    }
}

/// A constraint's text (bytes), its lexer and parser states.
#[pyclass]
struct Constraint {
    syntax: Arc<vocab::Syntax>,
    state: Arc<vocab::State>,
    text: Vec<u8>,
    suffix: bool,
}

#[pymethods]
impl Constraint {
    /// Advance by a token; ValueError if its bytes end the text's viability (the state is then unchanged).
    fn push(&mut self, token: u32) -> PyResult<()> {
        let bytes = &self.syntax.token_bytes[token as usize];
        let state = self.syntax.extend(&self.state, bytes).map_err(|i| PyValueError::new_err(format!(
            "the text stops being a prefix of a Python 3.12 module at byte {}", self.text.len() + i)))?;
        self.state = Arc::new(state);
        self.text.extend_from_slice(bytes);
        Ok(())
    }

    /// The text so far (prefix and generated tokens), UTF-8 bytes possibly ending mid-character.
    #[getter]
    fn text<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.text)
    }
}

/// The next masks of constraints, rows in order, computed in parallel without the GIL: the rows' uint32 words as
/// bytes, bit t % 32 of word t / 32 set iff token t keeps the text viable, BOS's set iff it is a candidate (a suffix
/// follows, or the lexer and parser end a module here), which the caller's ast.parse decides.
#[pyfunction]
fn masks<'py>(py: Python<'py>, constraints: Vec<Bound<'py, Constraint>>) -> Bound<'py, PyBytes> {
    let jobs: Vec<_> = constraints.iter().map(|c| {
        let c = c.borrow();
        let ends_line = matches!(c.text.last(), Some(b'\n' | b'\r'));
        (c.syntax.clone(), c.state.clone(), c.suffix, ends_line)
    }).collect();
    let rows: Vec<Vec<u32>> = py.detach(|| jobs.par_iter().map(|(syntax, state, suffix, ends_line)| {
        let mut words = syntax.allowed(state);
        if *suffix || syntax.complete(state, *ends_line) {
            words[syntax.bos as usize / 32] |= 1 << (syntax.bos % 32);
        }
        words
    }).collect());
    PyBytes::new(py, &rows.iter().flatten().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>())
}

#[pymodule]
fn nanochat_syntax(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Syntax>()?;
    m.add_class::<Constraint>()?;
    m.add_function(wrap_pyfunction!(masks, m)?)
}
