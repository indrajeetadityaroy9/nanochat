"""
Block Attention Residuals' depth attention (Chen et al., arXiv 2603.15031 Eqs. 2-6) as torch.library ops on
ablations/depth_attention.cu: h = sum_k softmax_k(q . RMSNorm(v_k)) v_k over a list of sources v_k, the RMSNorm
gain-free with gpt.norm's epsilon; the sources read in place through a pointer table, one pass over memory per
direction, the query gradient reduced in a fixed order.
"""
import ctypes
import functools
from pathlib import Path

import torch
from torch import Tensor

from nanochat.native import CSRC, cuda_library

# Blocks of Block Attention Residuals (arXiv 2603.15031 §3.2)
ATTN_RES_BLOCKS = 8


@functools.cache
def library():
    # the most sources an attention reads: the embedding, the completed blocks and the current block's partial sum
    lib = cuda_library(Path(__file__).with_suffix(".cu"), headers=(CSRC / "qk_row.cuh",),
                       defines=(f"MAX_SOURCES={ATTN_RES_BLOCKS + 2}",))
    lib.da_backward_warps.restype = ctypes.c_longlong
    return lib


def pointers(tensors):
    return (ctypes.c_void_p * len(tensors))(*(t.data_ptr() for t in tensors))


def address(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


@torch.library.custom_op("ablations::depth_attention", mutates_args=(), device_types="cuda")
def depth_attention(sources: list[Tensor], query: Tensor) -> tuple[Tensor, Tensor]:
    """sources: (B, T, C) bf16 contiguous; query (C,) fp32 -> (h like sources[0], lse (B, T) fp32)."""
    h = torch.empty_like(sources[0])
    lse = torch.empty(h.shape[:-1], dtype=torch.float32, device=h.device)
    C = h.size(-1)
    library().da_forward(pointers(sources), len(sources), address(query), address(h), address(lse),
                         ctypes.c_longlong(h.numel() // C), C, stream())
    return h, lse


@depth_attention.register_fake
def depth_attention_fake(sources, query):
    return torch.empty_like(sources[0]), sources[0].new_empty(sources[0].shape[:-1], dtype=torch.float32)


@torch.library.custom_op("ablations::depth_attention_backward", mutates_args=(), device_types="cuda")
def depth_attention_backward(sources: list[Tensor], query: Tensor, h: Tensor, lse: Tensor,
                             dh: Tensor) -> list[Tensor]:
    """[dv_k like v_k for each source, then dq (C,) fp32]."""
    grads = [torch.empty_like(s) for s in sources]
    dq = torch.empty_like(query)
    C = h.size(-1)
    part = torch.empty((library().da_backward_warps(C), C), dtype=torch.float32, device=h.device)
    library().da_backward(pointers(sources), pointers(grads), len(sources), address(query), address(h), address(lse),
                          address(dh), address(part), address(dq), ctypes.c_longlong(h.numel() // C), C, stream())
    return [*grads, dq]


@depth_attention_backward.register_fake
def depth_attention_backward_fake(sources, query, h, lse, dh):
    return [*(torch.empty_like(s) for s in sources), torch.empty_like(query)]


def depth_attention_save(ctx, inputs, output):
    sources, query = inputs
    ctx.save_for_backward(*sources, query, *output)


def depth_attention_grad(ctx, dh, dlse):
    *sources, query, h, lse = ctx.saved_tensors
    *grads, dq = depth_attention_backward(sources, query, h, lse, dh.contiguous())
    return grads, dq


depth_attention.register_autograd(depth_attention_grad, setup_context=depth_attention_save)
