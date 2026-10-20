"""Native libraries (CUDA kernels, the Rust tokenizer crate), built on first use, cached under CACHE/native."""

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import torch

from nanochat.paths import CACHE

CSRC = Path(__file__).resolve().parent / "csrc"


def native_library(name, sources, build, settings=b""):
    """CACHE/native/<name>-<sha256(settings, sources)>.so, built by build(tmp) and renamed atomically, so
    concurrent first builds never share a file."""
    digest = hashlib.sha256(settings)
    for source in sources:
        digest.update(source.read_bytes())
    path = CACHE / "native" / f"{name}-{digest.hexdigest()}.so"
    if not path.exists():
        # a cache shared by every build and process: its directory may already exist (resuming is its design)
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        build(tmp)
        tmp.replace(path)
    return path


def cuda_library(source: Path, *, headers: tuple[Path, ...], defines: tuple[str, ...]):
    """`source` (a .cu including `headers`) built by nvcc with -D`defines` for the current device's architecture, keyed
    by the nvcc command and version, loaded by ctypes."""
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    command = ["nvcc", f"-arch={arch}", *(f"-D{d}" for d in defines), "-Xcompiler", "-fPIC", "-shared", "-o"]
    version = subprocess.run(["nvcc", "--version"], capture_output=True, check=True).stdout

    def build(path):
        subprocess.run([*command, path, source], check=True)

    settings = " ".join(command).encode() + version
    return ctypes.CDLL(native_library(f"{source.stem}-{arch}", [source, *headers], build, settings))
