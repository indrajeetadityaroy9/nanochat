//! The corpus crate (Python module nanochat_crate): SA-BPE tokenizer training (arXiv 2604.14053), repository samples,
//! benchmark decontamination and the packer of training rows.

mod decontaminate;
mod pack;
mod repository;
mod train;

use std::sync::OnceLock;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use tree_sitter::Language;

/// A value per worker of rayon's global pool, made on first get (rayon workers only): a fancy-regex Regex's clones
/// share one scratch pool that workers would contend for.
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

/// A repository source's language ("Python" or "C++").
#[derive(Clone, Copy)]
pub(crate) enum Lang {
    Python,
    Cpp,
}

impl Lang {
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

/// The module; rayon's global pool is built first with the RLIMIT_STACK soft limit as its workers' stack (64 MiB under
/// scripts/container.sh's --ulimit stack=67108864:-1), not std's 2 MiB: workers run stolen jobs on their own stack
/// while waiting in a join, and nesting grows with pack calls.
#[pymodule]
fn nanochat_crate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let mut stack = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
    // SAFETY: getrlimit writes the limit into the struct it is given
    unsafe { libc::getrlimit(libc::RLIMIT_STACK, &mut stack) };
    rayon::ThreadPoolBuilder::new().stack_size(stack.rlim_cur as usize).build_global()
        .map_err(|e| PyValueError::new_err(format!("rayon's global pool: {e}")))?;
    m.add_function(wrap_pyfunction!(train::train, m)?)?;
    m.add_function(wrap_pyfunction!(repository::topological_sort, m)?)?;
    m.add_class::<decontaminate::Grams>()?;
    m.add_function(wrap_pyfunction!(pack::pack, m)?)
}
