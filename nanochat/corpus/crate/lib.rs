//! The corpus crate, Python module nanochat_crate (built by nanochat/native.py): SA-BPE (arXiv 2604.14053)
//! tokenizer training (train.rs; ablations/tokenizer.py), repository samples (repository.rs; curate.py, compile.py)
//! and the packer of training rows (pack.rs; pack.py).

mod pack;
mod repository;
mod train;

use std::sync::OnceLock;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use tree_sitter::Language;

/// A value per worker of rayon's global pool, made on the worker's first get (only rayon workers call it). Clones of a
/// fancy-regex Regex (also inside tiktoken-rs's CoreBPE) share its pool of matching scratch space, which workers
/// sharing one regex contend for; a regex made per worker has its own.
pub(crate) struct PerWorker<'a, T> {
    make: Box<dyn Fn() -> T + Sync + 'a>,
    values: Vec<OnceLock<T>>,
}

impl<'a, T> PerWorker<'a, T> {
    pub(crate) fn new(make: impl Fn() -> T + Sync + 'a) -> Self {
        PerWorker { make: Box::new(make), values: (0..rayon::current_num_threads()).map(|_| OnceLock::new()).collect() }
    }

    pub(crate) fn get(&self) -> &T {
        self.values[rayon::current_thread_index().unwrap()].get_or_init(&self.make)
    }
}

/// A repository source's language, extracted from its name (sources.Source.language: "Python" or "C++").
#[derive(Clone, Copy)]
pub(crate) enum Lang {
    Python,
    Cpp,
}

impl Lang {
    /// Its tree-sitter grammar.
    pub(crate) fn language(self) -> Language {
        match self {
            Lang::Python => tree_sitter_python::LANGUAGE.into(),
            Lang::Cpp => tree_sitter_cpp::LANGUAGE.into(),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Lang {
    type Error = PyErr;

    fn extract(name: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        match &*name.extract::<PyBackedStr>()? {
            "Python" => Ok(Lang::Python),
            "C++" => Ok(Lang::Cpp),
            other => Err(PyValueError::new_err(format!("grammar {other:?}: \"Python\" or \"C++\""))),
        }
    }
}

#[pymodule]
fn nanochat_crate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train::train, m)?)?;
    m.add_function(wrap_pyfunction!(repository::repository_samples, m)?)?;
    m.add_function(wrap_pyfunction!(pack::pack, m)?)
}
