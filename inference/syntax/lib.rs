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
    /// token_bytes: the mergeable tokens' bytes.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (*, keywords, soft, operators, basic, special, terminals, nonterminals, actions, goto, rules,
                        start, end, xid_start, xid_continue, token_bytes, vocab_size, bos_token_id))]
    fn new(keywords: Vec<(String, Vec<u16>)>, soft: Vec<(String, Vec<u16>)>, operators: Vec<(String, Vec<u16>)>,
           basic: [u16; 5], special: [u16; 4], terminals: usize, nonterminals: usize, actions: Vec<i32>,
           goto: Vec<i32>, rules: Vec<(u32, u32)>, start: u32, end: u32, xid_start: Vec<(u32, u32)>,
           xid_continue: Vec<(u32, u32)>, token_bytes: Vec<PyBackedBytes>, vocab_size: usize,
           bos_token_id: u32) -> Syntax {
        let lexicon = lexer::Lexicon::new(keywords, soft, operators, basic, xid_start, xid_continue);
        let [t_end, t_newline, t_indent, t_dedent] = special;
        let tables = parser::Tables {
            terminals, nonterminals, actions, goto, rules, start, end, t_end, t_newline, t_indent, t_dedent,
        };
        let token_bytes = token_bytes.iter().map(|b| b.to_vec()).collect();
        Syntax { inner: Arc::new(vocab::Syntax::new(lexicon, tables, token_bytes, vocab_size, bos_token_id)) }
    }

    /// The constraint of the tokens generated after prefix (UTF-8 bytes, possibly ending mid-character): suffix, the
    /// text that follows the generation (b"": none; a candidate end at every step), owed: bytes the first tokens
    /// re-emit (decoder.heal_tokens: each token a non-empty proper prefix of the bytes still owed or starting with
    /// them, BOS excluded, until they are re-emitted), lines: the generation is cut after its lines-th "\n" (None:
    /// uncut; eval/code_eval.py fim_lines), so a token holding that line end ends it there, and a generation ended
    /// earlier (BOS) gets a final "\n" if it has none. Raises ValueError when prefix is not a prefix of any module.
    #[pyo3(signature = (prefix, *, suffix, owed, lines=None))]
    fn constraint(&self, prefix: &[u8], suffix: &[u8], owed: &[u8], lines: Option<u32>) -> PyResult<Constraint> {
        let syntax = &self.inner;
        let state = syntax.extend(&syntax.start(), prefix).map_err(stops_at)?;
        if !syntax.viable_end(&state) {
            return Err(PyValueError::new_err(format!(
                "the prefix is not a prefix of a Python 3.12 module (at byte {})", prefix.len())));
        }
        if lines == Some(0) {
            return Err(PyValueError::new_err("lines must be positive"));
        }
        Ok(Constraint {
            syntax: syntax.clone(), state: Arc::new(state), text: prefix.to_vec(), pending: Vec::new(),
            prefix: prefix.len(), suffix: suffix.to_vec(), owed: owed.to_vec(), lines: lines.unwrap_or(0),
            cut: lines.is_some(),
        })
    }
}

/// The ValueError of a text that stops being a prefix of a module at byte.
fn stops_at(byte: usize) -> PyErr {
    PyValueError::new_err(format!("the text stops being a prefix of a Python 3.12 module at byte {byte}"))
}

/// A constraint's text (bytes; the prefix its first bytes), its lexer and parser states after all but its pending
/// bytes (pushed, their states computed by the next masks or grammar_ends: settle), the suffix after the generation,
/// the bytes still owed, the line ends left before the cut (0: none, or uncut) and whether it is cut.
#[pyclass]
struct Constraint {
    syntax: Arc<vocab::Syntax>,
    state: Arc<vocab::State>,
    text: Vec<u8>,
    pending: Vec<u8>,
    prefix: usize,
    suffix: Vec<u8>,
    owed: Vec<u8>,
    lines: u32,
    cut: bool,
}

impl Constraint {
    fn ends_line(&self) -> bool {
        matches!(self.text.last(), Some(b'\n' | b'\r'))
    }

    /// The bytes before a BOS end's suffix: a cut generation's final "\n" if it has none (fim_lines).
    fn final_newline(&self) -> &'static [u8] {
        if self.cut && self.text.len() > self.prefix && self.text.last() != Some(&b'\n') { b"\n" } else { b"" }
    }
}

/// The states of constraints after their pending bytes, computed in parallel without the GIL; ValueError if some
/// constraint's bytes end its text's viability.
fn settle(py: Python<'_>, constraints: &[Bound<'_, Constraint>]) -> PyResult<()> {
    let jobs: Vec<_> = constraints.iter().enumerate().filter_map(|(i, c)| {
        let mut c = c.borrow_mut();
        (!c.pending.is_empty()).then(|| {
            let pending = std::mem::take(&mut c.pending);
            (i, c.syntax.clone(), c.state.clone(), c.text.len() - pending.len(), pending)
        })
    }).collect();
    let states: Vec<_> = py.detach(|| jobs.par_iter().map(|(_, syntax, state, _, bytes)| {
        syntax.extend(state, bytes)
    }).collect());
    for ((i, _, _, start, _), state) in jobs.into_iter().zip(states) {
        constraints[i].borrow_mut().state = Arc::new(state.map_err(|k| stops_at(start + k))?);
    }
    Ok(())
}

#[pymethods]
impl Constraint {
    /// Advance by a token (a line end's token by its bytes up to the cut, where the generation ends). Its lexer and
    /// parser states are computed by the next masks or grammar_ends (in parallel, without the GIL), which raise
    /// ValueError if its bytes end the text's viability.
    fn push(&mut self, token: u32) -> PyResult<()> {
        let syntax = self.syntax.clone();
        let full = syntax.token_bytes.get(token as usize)
            .ok_or_else(|| PyValueError::new_err(format!("token {token} is not a mergeable token")))?;
        let bytes = syntax.cut(token, self.lines).unwrap_or(full);
        self.text.extend_from_slice(bytes);
        self.pending.extend_from_slice(bytes);
        if bytes.starts_with(&self.owed) {
            self.owed.clear();
        } else {
            self.owed.drain(..bytes.len().min(self.owed.len()));
        }
        if self.lines > 0 {
            self.lines = self.lines.saturating_sub(bytes.iter().filter(|&&b| b == b'\n').count() as u32);
        }
        Ok(())
    }

    /// The text so far (prefix and generated tokens), UTF-8 bytes possibly ending mid-character.
    #[getter]
    fn text<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.text)
    }

    /// The text a BOS end puts before the suffix: the text, and a cut generation's final "\n" if it has none.
    #[getter]
    fn end_text<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &[&self.text[..], self.final_newline()].concat())
    }

    /// The suffix after the generation (b"": none).
    #[getter]
    fn suffix<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.suffix)
    }

    /// The text that token ends the generation with (the text and its bytes up to the line end of the cut), None if
    /// it holds no such line end.
    fn line_end<'py>(&self, py: Python<'py>, token: u32) -> Option<Bound<'py, PyBytes>> {
        self.syntax.cut(token, self.lines).map(|cut| PyBytes::new(py, &[&self.text[..], cut].concat()))
    }
}

/// Whether the grammar ends a module after each constraint's end text (Constraint.end_text) and suffix (no bytes
/// owed): necessary for ast.parse to accept them (the grammar accepts every module CPython's parser does), computed
/// in parallel without the GIL.
#[pyfunction]
fn grammar_ends(py: Python<'_>, constraints: Vec<Bound<'_, Constraint>>) -> PyResult<Vec<bool>> {
    settle(py, &constraints)?;
    let jobs: Vec<_> = constraints.iter().map(|c| {
        let c = c.borrow();
        let after = [c.final_newline(), &c.suffix[..]].concat();
        (c.syntax.clone(), c.state.clone(), after, c.ends_line(), c.owed.is_empty())
    }).collect();
    Ok(py.detach(|| jobs.par_iter().map(|(syntax, state, after, ends_line, paid)| {
        *paid && syntax.complete_with(state, after, *ends_line)
    }).collect()))
}

/// The next masks of constraints, computed in parallel without the GIL into rows of the caller's buffer of uint32
/// words at address (row r at address + 4 rows[r] words; bit t % 32 of word t / 32 set iff token t keeps the text
/// viable and, while bytes are owed, re-emits them; for a token holding the cut's line end, iff the grammar ends a
/// module after the text up to it and the suffix), each row's other bits cleared. BOS's bit is set iff it is a
/// candidate end: no bytes owed, and a suffix follows or the grammar ends a module here. The caller decides a chosen
/// candidate (grammar_ends, then ast.parse) or line end (ast.parse). A cut text that only a line end continues (after
/// a backslash, say) and that no line end ends validly is a dead end: its line ends are allowed by viability alone and
/// stand unchecked. Returns whether each row allows some token and whether it is a dead end. The caller keeps the
/// buffer alive and the rows distinct.
#[pyfunction]
fn masks(py: Python<'_>, constraints: Vec<Bound<'_, Constraint>>, address: usize, rows: Vec<usize>, words: usize)
         -> PyResult<Vec<(bool, bool)>> {
    if rows.len() != constraints.len() {
        return Err(PyValueError::new_err("one row per constraint"));
    }
    let mut sorted = rows.clone();
    sorted.sort_unstable();
    if sorted.windows(2).any(|w| w[0] == w[1]) {
        return Err(PyValueError::new_err("rows must be distinct"));
    }
    // each constraint's pending bytes, taken: their states are computed with the mask
    let jobs: Vec<_> = constraints.iter().zip(&rows).map(|(c, &row)| {
        let mut c = c.borrow_mut();
        let pending = std::mem::take(&mut c.pending);
        let start = c.text.len() - pending.len();
        (c.syntax.clone(), c.state.clone(), pending, start, c.suffix.clone(), c.ends_line(), c.owed.clone(), c.lines,
         row)
    }).collect();
    if jobs.iter().any(|j| words * 32 < j.0.vocab_size) {
        return Err(PyValueError::new_err("a row holds fewer words than the vocabulary needs"));
    }
    // each job writes its own row only
    let base = address;
    type Verdict = Result<(Arc<vocab::State>, bool, bool), usize>;
    let out: Vec<Verdict> = py.detach(|| jobs.par_iter().map(|(syntax, state, pending, _, suffix, ends_line, owed,
                                                                lines, row)| {
        let state = if pending.is_empty() { state.clone() } else { Arc::new(syntax.extend(state, pending)?) };
        let bytes = words * 4;
        // SAFETY: the caller's buffer holds every row, and the rows are distinct (checked above)
        let bits = unsafe { std::slice::from_raw_parts_mut((base + row * bytes) as *mut u8, bytes) };
        bits.fill(0);
        syntax.allowed(&state, bits);
        let mut dead = false;
        if *lines > 0 && !syntax.line_ends(&state, *lines, suffix, bits, true) && bits.iter().all(|&b| b == 0) {
            syntax.line_ends(&state, *lines, suffix, bits, false);
            dead = true;
        }
        if owed.is_empty() {
            if !suffix.is_empty() || syntax.complete(&state, *ends_line) {
                bits[syntax.bos as usize / 8] |= 1 << (syntax.bos % 8);
            }
        } else {
            syntax.reemit(owed, bits);
        }
        Ok((state, bits.iter().any(|&b| b != 0), dead))
    }).collect());
    let mut verdicts = Vec::with_capacity(out.len());
    for ((c, job), result) in constraints.iter().zip(&jobs).zip(out) {
        let (state, nonempty, dead) = result.map_err(|k: usize| stops_at(job.3 + k))?;
        c.borrow_mut().state = state;
        verdicts.push((nonempty, dead));
    }
    Ok(verdicts)
}

#[pymodule]
fn nanochat_syntax(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Syntax>()?;
    m.add_class::<Constraint>()?;
    m.add_function(wrap_pyfunction!(masks, m)?)?;
    m.add_function(wrap_pyfunction!(grammar_ends, m)?)
}
