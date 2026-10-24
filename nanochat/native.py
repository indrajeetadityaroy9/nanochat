"""Native libraries (CUDA kernels, the Rust crates' Python modules), built on first use, cached under NATIVE; pointer
and stream, the ctypes arguments of csrc's entry points."""

import ctypes
import functools
import hashlib
import importlib.util
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import torch

from nanochat.config import HEAD_DIM
from nanochat.paths import NATIVE

CSRC = Path(__file__).resolve().parent / "csrc"
DATA_CRATE = Path(__file__).resolve().parent / "corpus" / "crate"


def native_library(name: str, sources: list[Path], build: Callable[[Path], None], *, settings: bytes) -> Path:
    """NATIVE/<name>-<sha256(settings, sources)>.so, built once per version of its sources and settings by build(tmp)
    and renamed atomically, so concurrent first builds never share a file."""
    digest = hashlib.sha256(settings)
    for source in sources:
        digest.update(source.read_bytes())
    path = NATIVE / f"{name}-{digest.hexdigest()}.so"
    if not path.exists():
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        build(tmp)
        tmp.replace(path)
    return path


def pointer(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def stream() -> ctypes.c_void_p:
    """The current CUDA stream, which the kernels launch on."""
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


@functools.cache
def cuda_library(name: str) -> ctypes.CDLL:
    """csrc/<name>.cu and qk_row.cuh built by nvcc for the current device's architecture with config.HEAD_DIM, keyed by
    the nvcc command and version, loaded by ctypes once per process."""
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    source = CSRC / f"{name}.cu"
    command = ["nvcc", f"-arch={arch}", f"-DHEAD_DIM={HEAD_DIM}", "-Xcompiler", "-fPIC", "-shared", "-o"]
    version = subprocess.run(["nvcc", "--version"], capture_output=True, check=True).stdout

    def build(path: Path) -> None:
        subprocess.run([*command, path, source], check=True)

    return ctypes.CDLL(native_library(f"{name}-{arch}", [source, CSRC / "qk_row.cuh"], build,
                                      settings=" ".join(command).encode() + version))


@functools.cache
def crate_module(crate: Path) -> ModuleType:
    """The Python module nanochat_<crate.name> of the Rust crate in directory crate (pyo3), built by cargo in its target
    directory NATIVE/<crate.name>-target, keyed by the cargo command and the rustc version, imported once per
    process."""
    name = f"nanochat_{crate.name}"
    target = NATIVE / f"{crate.name}-target"
    command = ["cargo", "build", "--release", "--locked", "--manifest-path", str(crate / "Cargo.toml"),
               "--target-dir", str(target)]
    version = subprocess.run(["rustc", "-vV"], capture_output=True, check=True).stdout

    def build(path: Path) -> None:
        subprocess.run(command, check=True)
        shutil.copyfile(target / "release" / f"lib{name}.so", path)

    sources = [*sorted(crate.glob("*.rs")), crate / "Cargo.toml", crate / "Cargo.lock"]
    path = native_library(crate.name, sources, build, settings=" ".join(command).encode() + version)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def data_crate() -> ModuleType:
    """The corpus's Rust crate (corpus/crate): train (ablations/tokenizer.py), repository_samples (corpus/curate.py,
    compile.py), pack (corpus/pack.py)."""
    return crate_module(DATA_CRATE)
