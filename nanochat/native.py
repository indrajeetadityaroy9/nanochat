"""Native libraries (CUDA kernels, the Rust crates' Python modules), built on first use, cached under NATIVE."""

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

    settings = " ".join(command).encode() + version
    return ctypes.CDLL(native_library(f"{name}-{arch}", [source, CSRC / "qk_row.cuh"], build, settings=settings))


@functools.cache
def crate_module(crate: Path) -> ModuleType:
    """The Python module nanochat_<crate.name> of the Rust crate in directory crate (pyo3), built by cargo in its target
    directory NATIVE/<crate.name>-target, keyed by the cargo command, imported once per process."""
    name = f"nanochat_{crate.name}"
    target = NATIVE / f"{crate.name}-target"
    command = ["cargo", "build", "--release", "--locked", "--manifest-path", str(crate / "Cargo.toml"),
               "--target-dir", str(target)]

    def build(path: Path) -> None:
        subprocess.run(command, check=True)
        shutil.copyfile(target / "release" / f"lib{name}.so", path)

    sources = [*sorted(crate.glob("*.rs")), crate / "Cargo.toml", crate / "Cargo.lock"]
    path = native_library(crate.name, sources, build, settings=" ".join(command).encode())
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
