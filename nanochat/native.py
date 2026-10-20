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


def cuda_library(name, *, head_dim):
    """csrc/<name>.cu and qk_row.cuh built by nvcc with -DHEAD_DIM=head_dim for the current device's architecture, keyed
    by the nvcc command and version (one library per head_dim), loaded by ctypes."""
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    source = CSRC / f"{name}.cu"
    command = ["nvcc", f"-arch={arch}", f"-DHEAD_DIM={head_dim}", "-Xcompiler", "-fPIC", "-shared", "-o"]
    version = subprocess.run(["nvcc", "--version"], capture_output=True, check=True).stdout

    def build(path):
        subprocess.run([*command, path, source], check=True)

    settings = " ".join(command).encode() + version
    return ctypes.CDLL(native_library(f"{name}-{arch}-d{head_dim}", [source, CSRC / "qk_row.cuh"], build, settings))
