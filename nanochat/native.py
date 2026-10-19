"""Native libraries (CUDA kernels, the Rust tokenizer crate), built on first use, cached under <base_dir>/cache/native."""

import ctypes
import hashlib
import os
import subprocess

import torch

from nanochat.paths import base_dir

CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")


def native_library(name, sources, build, settings=b""):
    """<base_dir>/cache/native/<name>-<sha256(settings, sources)>.so, built by build(tmp) and renamed atomically, so
    concurrent first builds never share a file."""
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
    """csrc/<name>.cu and qk_row.cuh built by nvcc for the current device's architecture, keyed by the nvcc command and
    version, loaded by ctypes."""
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    source = os.path.join(CSRC, f"{name}.cu")
    command = ["nvcc", f"-arch={arch}", "-Xcompiler", "-fPIC", "-shared", "-o"]
    version = subprocess.run(["nvcc", "--version"], capture_output=True, check=True).stdout

    def build(path):
        subprocess.run([*command, path, source], check=True)

    settings = " ".join(command).encode() + version
    return ctypes.CDLL(native_library(f"{name}-{arch}", [source, os.path.join(CSRC, "qk_row.cuh")], build, settings))
