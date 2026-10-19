"""
The token and value embeddings of model/gpt.py as one torch.library op, nanochat::embedding (embedding.cu): fp32 table
rows looked up as bf16, and the table's fp32 gradient summed without atomics in a fixed order, so a training step's
gradients are bitwise reproducible (Inductor's own embedding backward adds rows with atomics). The run of each token's
positions comes from a stable sort; its partials, slots and first partial per row are computed on the device (no host
synchronization). The library is compiled from embedding.cu once per source revision and GPU architecture
(native.cuda_library) on first use.
"""

import ctypes
import functools
import os

import torch
from torch import Tensor

from nanochat.native import cuda_library

SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embedding.cu")
# Positions per partial sum: a frequent token's run is summed as ceil(run / PARTIAL_ROWS) partials in parallel, then in
# order, so no block's serial sum exceeds PARTIAL_ROWS rows or one run's partials. A fixture (any fixed value is
# reproducible; it sets the summation order), 32.
PARTIAL_ROWS = 32


@functools.cache
def _lib():
    lib = cuda_library("embedding", SOURCE)
    lib.emb_forward.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_longlong, ctypes.c_int, ctypes.c_void_p]
    lib.emb_backward.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_longlong] + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    return lib


def _stream():
    return torch.cuda.current_stream().cuda_stream


@torch.library.custom_op("nanochat::embedding", mutates_args=(), device_types="cuda")
def embedding(table: Tensor, idx: Tensor) -> Tensor:
    """Rows idx (B, T) int32 of table (V, C) fp32 contiguous as bf16 (B, T, C)."""
    out = table.new_empty((*idx.shape, table.size(1)), dtype=torch.bfloat16)
    _lib().emb_forward(table.data_ptr(), idx.data_ptr(), out.data_ptr(), idx.numel(), table.size(1), _stream())
    return out


@embedding.register_fake
def _(table, idx):
    return table.new_empty((*idx.shape, table.size(1)), dtype=torch.bfloat16)


@torch.library.custom_op("nanochat::embedding_backward", mutates_args=(), device_types="cuda")
def embedding_backward(grad: Tensor, idx: Tensor, rows: int) -> Tensor:
    """The fp32 gradient (rows, C) of the table given grad (B, T, C) bf16 wrt embedding(table, idx)."""
    grad = grad.contiguous()
    flat = idx.flatten()
    n, C = flat.numel(), grad.size(-1)
    tokens, order = torch.sort(flat, stable=True)
    pos = torch.arange(n, device=flat.device, dtype=torch.int32)
    run_first = torch.searchsorted(tokens, tokens, out_int32=True)  # each position's first position of its token's run
    start = (pos - run_first) % PARTIAL_ROWS == 0
    before = torch.cat([start.new_zeros(1, dtype=torch.int32), start.cumsum(0, dtype=torch.int32)])  # partials before j
    first = before[torch.searchsorted(tokens, torch.arange(rows + 1, device=flat.device, dtype=tokens.dtype))]
    partial = grad.new_empty((min(n, -(-n // PARTIAL_ROWS) + min(n, rows)), C), dtype=torch.float32)  # >= partials
    out = grad.new_empty((rows, C), dtype=torch.float32)
    _lib().emb_backward(grad.data_ptr(), order.data_ptr(), tokens.data_ptr(), start.data_ptr(), before.data_ptr(),
                        first.data_ptr(), partial.data_ptr(), out.data_ptr(), n, rows, C, PARTIAL_ROWS, _stream())
    return out


@embedding_backward.register_fake
def _(grad, idx, rows):
    return grad.new_empty((rows, grad.size(-1)), dtype=torch.float32)


def _setup(ctx, inputs, output):
    table, idx = inputs
    ctx.rows = table.size(0)
    ctx.save_for_backward(idx)


def _backward(ctx, grad):
    (idx,) = ctx.saved_tensors
    return embedding_backward(grad, idx, ctx.rows), None


embedding.register_autograd(_backward, setup_context=_setup)
