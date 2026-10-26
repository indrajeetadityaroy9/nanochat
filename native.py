"""Native libraries (CUDA kernels, the Rust crates' Python modules), built on first use, cached under NATIVE; pointer
and stream, the ctypes arguments of model/kernels' entry points. model/kernels follows ThunderKittens' layout
(header-only shared code, common.cuh and qk_row.cuh; one self-contained file per kernel, its C entry points beside its
kernels); this module is its one build, with one nvcc command for every kernel. torch is imported by the CUDA paths
alone, so the corpus's spawned workers, which load only the crates, import none."""

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
from typing import TYPE_CHECKING

from model.config import HEAD_DIM
from paths import NATIVE

if TYPE_CHECKING:
    import torch

KERNELS = Path(__file__).resolve().parent / "model" / "kernels"
CORPUS_CRATE = Path(__file__).resolve().parent / "corpus" / "crate"


def native_library(name: str, command: list[str], sources: list[Path], build: Callable[[Path], None]) -> Path:
    """NATIVE/<name>-<sha256(command, sources)>.so, built once per version of its build command and sources by
    build(tmp) and renamed atomically, so concurrent first builds never share a file. The toolchain is the image's."""
    digest = hashlib.sha256(" ".join(command).encode())
    for source in sources:
        digest.update(source.read_bytes())
    path = NATIVE / f"{name}-{digest.hexdigest()}.so"
    if not path.exists():
        NATIVE.mkdir(parents=True, exist_ok=True)  # a fresh checkout has none, as the other kernel caches create theirs
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        build(tmp)
        tmp.replace(path)
    return path


def pointer(t: "torch.Tensor") -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def stream() -> ctypes.c_void_p:
    """The current CUDA stream, which the kernels launch on."""
    import torch

    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


@functools.cache
def cuda_library(name: str) -> ctypes.CDLL:
    """model/kernels/<name>.cu built by nvcc for the current device's architecture-specific target (sm_90a: wgmma,
    setmaxnreg; CUDA C++ Programming Guide, "Feature Availability") with HEAD_DIM (model/config.py), keyed by the
    command, the file and every model/kernels header, loaded by ctypes once per process."""
    import torch

    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}a"
    source = KERNELS / f"{name}.cu"
    command = ["nvcc", f"-gencode=arch=compute_{major}{minor}a,code={arch}", f"-DHEAD_DIM={HEAD_DIM}", "-Xcompiler",
               "-fPIC", "-shared", "-o"]

    def build(path: Path) -> None:
        subprocess.run([*command, path, source], check=True)

    return ctypes.CDLL(native_library(f"{name}-{arch}", command, [source, *sorted(KERNELS.glob("*.cuh"))], build))


@functools.cache
def crate_module(crate: Path) -> ModuleType:
    """The Python module nanochat_<crate.name> of the Rust crate in directory crate (pyo3), built by cargo in its target
    directory NATIVE/<crate.name>-target, keyed by the cargo command and the crate's files, imported once per
    process."""
    name = f"nanochat_{crate.name}"
    target = NATIVE / f"{crate.name}-target"
    command = ["cargo", "build", "--release", "--locked", "--manifest-path", str(crate / "Cargo.toml"),
               "--target-dir", str(target)]

    def build(path: Path) -> None:
        subprocess.run(command, check=True)
        shutil.copyfile(target / "release" / f"lib{name}.so", path)

    sources = [*sorted(crate.glob("*.rs")), crate / "Cargo.toml", crate / "Cargo.lock"]
    path = native_library(crate.name, command, sources, build)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corpus_crate() -> ModuleType:
    """The corpus's Rust crate (corpus/crate): train (corpus/train_tokenizer.py), topological_sort
    (corpus/sources.py), pack (corpus/pack.py)."""
    return crate_module(CORPUS_CRATE)
