//! The data crate, Python module nanochat_bpe (built by nanochat/data/tokenizer.py bpe()): SA-BPE (arXiv 2604.14053)
//! tokenizer training (train.rs), deduplication (dedup.rs; nanochat/data/dedup.py) and repository samples
//! (repository.rs; nanochat/data/prepare.py) of curation, and the packer of training rows (pack.rs;
//! nanochat/data/pack.py).

mod dedup;
mod pack;
mod repository;
mod train;

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
        Ok(match &*name.extract::<PyBackedStr>()? {
            "Python" => Lang::Python,
            "C++" => Lang::Cpp,
            _ => unreachable!(),
        })
    }
}

#[pymodule]
fn nanochat_bpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train::train, m)?)?;
    m.add_function(wrap_pyfunction!(dedup::sketch, m)?)?;
    m.add_function(wrap_pyfunction!(dedup::duplicates, m)?)?;
    m.add_function(wrap_pyfunction!(repository::repository_samples, m)?)?;
    m.add_function(wrap_pyfunction!(pack::pack, m)?)
}
