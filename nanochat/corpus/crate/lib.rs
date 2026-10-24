//! The data crate, Python module nanochat_crate (built by nanochat/native.py data_crate()): SA-BPE (arXiv 2604.14053)
//! tokenizer training (train.rs; ablations/tokenizer.py), repository samples (repository.rs; nanochat/corpus/curate.py
//! and compile.py) and the packer of training rows (pack.rs; nanochat/corpus/pack.py).

mod pack;
mod repository;
mod train;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use tree_sitter::Language;

/// A repository source's language, extracted from its name (sources.Source.grammar: "Python" or "C++").
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
