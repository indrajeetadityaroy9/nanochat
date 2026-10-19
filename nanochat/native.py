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

# The CUDA sources (csrc/<name>.cu) and the device header every one of them includes, hashed with each source.
CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
CUDA_HEADER = os.path.join(CSRC, "qk_row.cuh")


def native_library(name, sources, build, settings=b""):
    """Path of the native library <base_dir>/cache/native/<name>-<sha256 of the sources' bytes and the build's
    settings>.so; when it is missing, build(tmp) writes it to tmp = <path>.<pid>.tmp, which then replaces <path>, so an
    interrupted build leaves no library and concurrent first builds do not share a file."""
    digest = hashlib.sha256(settings)
    for source in sources:
        with open(source, "rb") as f:
            digest.update(f.read())
    path = os.path.join(base_dir(), "cache", "native", f"{name}-{digest.hexdigest()}.so")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        build(tmp)
        os.replace(tmp, path)
    return path


def cuda_library(name):
    """csrc/<name>.cu (with CUDA_HEADER) built by nvcc into a shared library for the current device's architecture
    (native library <name>-sm_<cc>, keyed also by the nvcc command and version: the cache under <base_dir> outlives
    the image), loaded by ctypes. Building it initializes CUDA."""
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    source = os.path.join(CSRC, f"{name}.cu")
    command = ["nvcc", "-O3", f"-arch={arch}", "-Xcompiler", "-fPIC", "-shared", "-o"]
    version = subprocess.run(["nvcc", "--version"], capture_output=True, check=True).stdout

    def build(path):
        subprocess.run([*command, path, source], check=True)

    settings = " ".join(command).encode() + version
    return ctypes.CDLL(native_library(f"{name}-{arch}", [source, CUDA_HEADER], build, settings))
