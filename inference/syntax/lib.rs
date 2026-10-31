/*! The Python 3.12 syntax constraint, Python module nanochat_syntax (wrapped by inference/syntax_constraint.py):
    lexer.rs, parser.rs and vocab.rs; masks of a batch in parallel (rayon) without the GIL, into the caller's rows. */

mod lexer;
mod parser;
mod vocab;

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;
use rayon::prelude::*;

/** The Python 3.12 syntax of a tokenizer's vocabulary (vocab::Syntax), shared by the constraints it builds. */
#[pyclass(frozen)]
struct Syntax {
    inner: Arc<vocab::Syntax>,
}

#[pymethods]
impl Syntax {
    /** The syntax from inference/syntax_constraint.python_syntax's lexicon, LALR tables and token bytes. */
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

    /** The constraint after prefix (UTF-8); ValueError if prefix is not viable. */
    fn constraint(&self, prefix: &[u8]) -> PyResult<Constraint> {
        let syntax = &self.inner;
        let state = syntax.extend(&syntax.start(), prefix).map_err(stops_at)?;
        if !syntax.viable_end(&state) {
            return Err(stops_at(prefix.len()));
        }
        Ok(Constraint { syntax: syntax.clone(), state: Arc::new(state), text: prefix.to_vec(), settled: prefix.len() })
    }
}

/** The ValueError of a text that stops being a prefix of a Python 3.12 module at byte. */
fn stops_at(byte: usize) -> PyErr {
    PyValueError::new_err(format!("the text stops being a prefix of a Python 3.12 module at byte {byte}"))
}

/** A constraint's text (the prefix, then pushed bytes) and its states after the settled bytes. */
#[pyclass]
struct Constraint {
    syntax: Arc<vocab::Syntax>,
    state: Arc<vocab::State>,
    text: Vec<u8>,
    settled: usize,
}

impl Constraint {
    fn ends_line(&self) -> bool {
        matches!(self.text.last(), Some(b'\n' | b'\r'))
    }
}

/** The states of constraints after their pushed bytes, in parallel without the GIL; ValueError if some text dies. */
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
    /** Advance by a mergeable token; its states are settled by the next masks or grammar_ends. */
    fn push(&mut self, token: u32) {
        self.text.extend_from_slice(&self.syntax.token_bytes[token as usize]);
    }

    /** The text so far (prefix and generated tokens), UTF-8 bytes possibly ending mid-character. */
    #[getter]
    fn text<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.text)
    }
}

/** Whether the grammar ends a module after each text (necessary for ast.parse). */
#[pyfunction]
fn grammar_ends(py: Python<'_>, constraints: Vec<Bound<'_, Constraint>>) -> PyResult<Vec<bool>> {
    settle(py, &constraints)?;
    let jobs: Vec<_> = constraints.iter().map(|c| {
        let c = c.borrow();
        (c.syntax.clone(), c.state.clone(), c.ends_line())
    }).collect();
    Ok(py.detach(|| jobs.par_iter().map(|(syntax, state, ends_line)| syntax.complete(state, *ends_line)).collect()))
}

/** Writes each next mask into row rows[r] (words uint32 per row) at address; whether each row allows a token. */
#[pyfunction]
fn masks(py: Python<'_>, constraints: Vec<Bound<'_, Constraint>>, address: usize, rows: Vec<usize>, words: usize)
         -> PyResult<Vec<bool>> {
    settle(py, &constraints)?;
    let jobs: Vec<_> = constraints.iter().zip(&rows).map(|(c, &row)| {
        let c = c.borrow();
        (c.syntax.clone(), c.state.clone(), c.ends_line(), row)
    }).collect();
    let bytes = words * 4;
    Ok(py.detach(|| jobs.par_iter().map(|(syntax, state, ends_line, row)| {
        /* SAFETY: the caller's buffer holds every row, and the rows are distinct (decoder.step_masks: one per slot) */
        let bits = unsafe { std::slice::from_raw_parts_mut((address + row * bytes) as *mut u8, bytes) };
        bits.fill(0);
        syntax.allowed(state, bits);
        if syntax.complete(state, *ends_line) {
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
