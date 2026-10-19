"""
Native libraries (CUDA kernels, the Rust tokenizer crate) built on first use and cached per source revision under
<base_dir>/cache/native.
"""

import ctypes
import hashlib
import os
import subprocess

import torch

from nanochat.paths import base_dir

# The device header every CUDA library includes (ops/qk_prologue.cu, inference/decode.cu), hashed with each source.
CUDA_HEADER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops", "qk_row.cuh")


def native_library(name, sources, build):
    """Path of the native library <base_dir>/cache/native/<name>-<sha256 of the sources' bytes>.so; when it is missing,
    build(tmp) writes it to tmp = <path>.tmp, which then replaces <path>, so an interrupted build leaves no library."""
    digest = hashlib.sha256()
    for source in sources:
        with open(source, "rb") as f:
            digest.update(f.read())
    path = os.path.join(base_dir(), "cache", "native", f"{name}-{digest.hexdigest()}.so")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        build(path + ".tmp")
        os.replace(path + ".tmp", path)
    return path


def cuda_library(name, source):
    """The CUDA source file (with CUDA_HEADER) built by nvcc into a shared library for the current device's
    architecture (native library <name>-sm_<cc>), loaded by ctypes. Building it initializes CUDA."""
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"

    def build(path):
        subprocess.run(["nvcc", "-O3", f"-arch={arch}", "-Xcompiler", "-fPIC", "-shared", "-o", path, source],
                       check=True)

    return ctypes.CDLL(native_library(f"{name}-{arch}", [source, CUDA_HEADER], build))
