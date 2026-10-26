//! The Python 3.12 syntax constraint, Python module nanochat_syntax (built and wrapped by
//! inference/syntax_constraint.py): after each token, exactly the tokens whose bytes keep the text a prefix of
//! some module of the grammar python.lark (whose LALR(1) tables Lark builds), lexed as CPython 3.12.3's tokenizer does
//! (lexer.rs), indented and parsed as tokenizer.c and the tables say (parser.rs), over the vocabulary's byte trie
//! (vocab.rs). A batch of constraints' masks is computed in parallel (rayon) without the GIL, straight into the
//! caller's mask rows (masks).

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
    /// token_bytes: the mergeable tokens' bytes (the ids below BOS's: corpus/tokenizer.py).
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (*, keywords, soft, operators, basic, special, terminals, nonterminals, actions, goto, rules,
                        start, end, xid_start, xid_continue, token_bytes))]
    fn new(keywords: Vec<(String, Vec<u16>)>, soft: Vec<(String, Vec<u16>)>, operators: Vec<(String, Vec<u16>)>,
           basic: [u16; 5], special: [u16; 4], terminals: usize, nonterminals: usize, actions: Vec<i32>,
           goto: Vec<i32>, rules: Vec<(u32, u32)>, start: u32, end: u32, xid_start: Vec<(u32, u32)>,
           xid_continue: Vec<(u32, u32)>, token_bytes: Vec<PyBackedBytes>) -> Syntax {
        let lexicon = lexer::Lexicon::new(keywords, soft, operators, basic, xid_start, xid_continue);
        let [t_end, t_newline, t_indent, t_dedent] = special;
        let tables = parser::Tables {
            terminals, nonterminals, actions, goto, rules, start, end, t_end, t_newline, t_indent, t_dedent,
        };
        let token_bytes = token_bytes.iter().map(|b| b.to_vec()).collect();
        Syntax { inner: Arc::new(vocab::Syntax::new(lexicon, tables, token_bytes)) }
    }

    /// The constraint of the tokens generated after prefix (UTF-8 bytes, possibly ending mid-character), suffix the
    /// text that follows the generation (b"": none; a candidate end at every step). Raises ValueError when prefix is
    /// not a prefix of any module.
    #[pyo3(signature = (prefix, *, suffix))]
    fn constraint(&self, prefix: &[u8], suffix: &[u8]) -> PyResult<Constraint> {
        let syntax = &self.inner;
        let state = syntax.extend(&syntax.start(), prefix).map_err(stops_at)?;
        if !syntax.viable_end(&state) {
            return Err(stops_at(prefix.len()));
        }
        Ok(Constraint {
            syntax: syntax.clone(), state: Arc::new(state), text: prefix.to_vec(), settled: prefix.len(),
            suffix: suffix.to_vec(),
        })
    }
}

/// The ValueError of a text that stops being a prefix of a Python 3.12 module at byte.
fn stops_at(byte: usize) -> PyErr {
    PyValueError::new_err(format!("the text stops being a prefix of a Python 3.12 module at byte {byte}"))
}

/// A constraint's text (bytes; the prefix its first bytes), its lexer and parser states after the text's first settled
/// bytes (the pushed bytes after them are settled by the next masks or grammar_ends: settle), and the suffix after the
/// generation.
#[pyclass]
struct Constraint {
    syntax: Arc<vocab::Syntax>,
    state: Arc<vocab::State>,
    text: Vec<u8>,
    settled: usize,
    suffix: Vec<u8>,
}

impl Constraint {
    fn ends_line(&self) -> bool {
        matches!(self.text.last(), Some(b'\n' | b'\r'))
    }
}

/// The states of constraints after their pushed bytes, computed in parallel without the GIL; ValueError if some
/// constraint's bytes end its text's viability.
fn settle(py: Python<'_>, constraints: &[Bound<'_, Constraint>]) -> PyResult<()> {
    let jobs: Vec<_> = constraints.iter().enumerate().filter_map(|(i, c)| {
        let c = c.borrow();
        let start = c.settled;
        (start < c.text.len()).then(|| (i, c.syntax.clone(), c.state.clone(), start, c.text[start..].to_vec()))
    }).collect();
    let states: Vec<_> = py.detach(|| jobs.par_iter().map(|(_, syntax, state, _, bytes)| {
        syntax.extend(state, bytes)
    }).collect());
    for ((i, _, _, start, _), state) in jobs.into_iter().zip(states) {
        let mut c = constraints[i].borrow_mut();
        c.state = Arc::new(state.map_err(|k| stops_at(start + k))?);
        c.settled = c.text.len();
    }
    Ok(())
}

#[pymethods]
impl Constraint {
    /// Advance by a mergeable token. Its lexer and parser states are computed by the next masks or grammar_ends (in
    /// parallel, without the GIL), which raise ValueError if its bytes end the text's viability.
    fn push(&mut self, token: u32) {
        self.text.extend_from_slice(&self.syntax.token_bytes[token as usize]);
    }

    /// The text so far (prefix and generated tokens), UTF-8 bytes possibly ending mid-character.
    #[getter]
    fn text<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.text)
    }
}

/// Whether the grammar ends a module after each constraint's text and suffix: necessary for ast.parse to accept them
/// (the grammar accepts every module CPython's parser does), computed in parallel without the GIL.
#[pyfunction]
fn grammar_ends(py: Python<'_>, constraints: Vec<Bound<'_, Constraint>>) -> PyResult<Vec<bool>> {
    settle(py, &constraints)?;
    let jobs: Vec<_> = constraints.iter().map(|c| {
        let c = c.borrow();
        let ends_line = matches!(c.suffix.last().or(c.text.last()), Some(b'\n' | b'\r'));
        (c.syntax.clone(), c.state.clone(), c.suffix.clone(), ends_line)
    }).collect();
    Ok(py.detach(|| jobs.par_iter().map(|(syntax, state, suffix, ends_line)| {
        syntax.extend(state, suffix).is_ok_and(|end| syntax.complete(&end, *ends_line))
    }).collect()))
}

/// The next masks of constraints, computed in parallel without the GIL into rows of the caller's buffer of uint32
/// words at address (row r at byte address + 4 words rows[r]; bit t % 32 of word t / 32 set iff token t keeps the text
/// viable), each row's other bits cleared. BOS's bit is set iff it is a candidate end: a suffix follows or the grammar
/// ends a module here; the caller decides a chosen candidate (grammar_ends, then ast.parse). Returns whether each row
/// allows some token. The caller keeps the buffer alive, each row within it, and the rows distinct.
#[pyfunction]
fn masks(py: Python<'_>, constraints: Vec<Bound<'_, Constraint>>, address: usize, rows: Vec<usize>, words: usize)
         -> PyResult<Vec<bool>> {
    if rows.len() != constraints.len() {
        return Err(PyValueError::new_err("one row per constraint"));
    }
    let mut sorted = rows.clone();
    sorted.sort_unstable();
    if sorted.windows(2).any(|w| w[0] == w[1]) {
        return Err(PyValueError::new_err("rows must be distinct"));
    }
    settle(py, &constraints)?;
    let jobs: Vec<_> = constraints.iter().zip(&rows).map(|(c, &row)| {
        let c = c.borrow();
        (c.syntax.clone(), c.state.clone(), !c.suffix.is_empty(), c.ends_line(), row)
    }).collect();
    let bytes = words * 4;
    Ok(py.detach(|| jobs.par_iter().map(|(syntax, state, suffixed, ends_line, row)| {
        // SAFETY: the caller's buffer holds every row, and the rows are distinct (checked above): each job writes its
        // own row only
        let bits = unsafe { std::slice::from_raw_parts_mut((address + row * bytes) as *mut u8, bytes) };
        bits.fill(0);
        syntax.allowed(state, bits);
        if *suffixed || syntax.complete(state, *ends_line) {
            bits[syntax.bos as usize / 8] |= 1 << (syntax.bos % 8);
        }
        bits.iter().any(|&b| b != 0)
    }).collect()))
}

#[pymodule]
fn nanochat_syntax(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Syntax>()?;
    m.add_class::<Constraint>()?;
    m.add_function(wrap_pyfunction!(masks, m)?)?;
    m.add_function(wrap_pyfunction!(grammar_ends, m)?)
}
