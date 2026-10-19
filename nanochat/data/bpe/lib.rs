//! The data crate, Python module nanochat_bpe (built by nanochat/data/tokenizer.py bpe()): SA-BPE tokenizer training
//! (train.rs), deduplication (dedup.rs; nanochat/data/dedup.py) and repository samples (repository.rs;
//! nanochat/data/prepare.py) of curation, and the packer of training rows (pack.rs; nanochat/data/pack.py).

mod dedup;
mod pack;
mod repository;
mod train;

use pyo3::prelude::*;
use tree_sitter::Language;

/// The tree-sitter grammar of a repository source's language (sources.Source.grammar): "Python" or "C++".
pub(crate) fn language(grammar: &str) -> Language {
    match grammar {
        "Python" => tree_sitter_python::LANGUAGE.into(),
        "C++" => tree_sitter_cpp::LANGUAGE.into(),
        _ => unreachable!(),
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
