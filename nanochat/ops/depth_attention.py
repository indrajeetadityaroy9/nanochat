"""
Block Attention Residuals' depth attention (model/gpt.py GPT.hidden; Chen et al., arXiv 2603.15031 §3.1-3.2, Eqs. 2-6)
in the paper's two-phase computation (§4.2, Algorithm 1), one fused op per phase and direction (csrc/depth_attention.cu):
depth_inter once per block, its queries against the completed blocks (each block read once per block, not once per
sublayer), depth_merge per later sublayer of the block, its partial sum merged into depth_inter's output by online
softmax. Each returns its sublayer's input norm(h), the norm fused (F.rms_norm's fp32 epsilon); fp32 math, h rounded
once to bf16. The library is compiled once per source revision and GPU architecture (native.cuda_library) on first use.
"""

import ctypes
import functools

import torch
from torch import Tensor

from nanochat.native import cuda_library

MERGE_STATS = 5  # depth_merge's statistics per row: the merge weights a and b, the partial sum's 1 / RMS and logit, rstd


@functools.cache
def library():
    """csrc/depth_attention.cu, its entry points typed."""
    lib = cuda_library("depth_attention")
    ptr, ptrs, rows = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), [ctypes.c_longlong, ctypes.c_int, ctypes.c_void_p]
    lib.dai_forward.argtypes = [ptrs, ctypes.c_int, ptr, ctypes.c_int, ptr, ptrs, ptr, ptr, *rows]
    lib.dai_backward.argtypes = [ptrs, ctypes.c_int, ptr, ctypes.c_int, ptr, ptr, ptrs, ptr, ptr, ptrs, ptr, ptr, *rows]
    lib.dai_backward_blocks.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.dai_backward_blocks.restype = ctypes.c_uint
    lib.dam_forward.argtypes = [*[ptr] * 6, *rows]
    lib.dam_backward.argtypes = [*[ptr] * 10, *rows]
    lib.dam_backward_blocks.argtypes = [ctypes.c_longlong, ctypes.c_int]
    lib.dam_backward_blocks.restype = ctypes.c_uint
    return lib


def pointers(tensors):
    """The tensors' device pointers as a C array."""
    return (ctypes.c_void_p * len(tensors))(*(t.data_ptr() for t in tensors))


def launch_args(x):
    """The kernels' trailing arguments: rows and width of x, current stream."""
    return x.numel() // x.size(-1), x.size(-1), torch.cuda.current_stream().cuda_stream


def inter_outputs(sources, queries):
    """depth_inter's outputs, empty; its statistics per row: the logits (S, K), each source's 1 / RMS, y0's rstd."""
    x, S, K = sources[0], queries.size(0), len(sources)
    return (torch.empty_like(x), [x.new_empty(x.shape, dtype=torch.float32) for _ in range(S - 1)],
            queries.new_empty((S - 1, *x.shape[:-1])), queries.new_empty((*x.shape[:-1], S * K + K + 1)))


@torch.library.custom_op("nanochat::depth_inter", mutates_args=(), device_types="cuda")
def depth_inter(sources: list[Tensor], queries: Tensor) -> tuple[Tensor, list[Tensor], Tensor, Tensor]:
    """Phase 1 of a block: sources K contiguous (B, T, C) bf16 (the embedding and the completed blocks), queries (S, C)
    fp32 contiguous (the block's sublayers' pseudo-queries). Returns the first sublayer's input norm(h_0) (bf16), the
    later sublayers' inter-block softmax averages (S - 1 tensors (B, T, C) fp32), their log-sum-exps (S - 1, B, T) fp32
    and the backward's statistics (B, T, S K + K + 1) fp32."""
    y0, inter, lse, stats = inter_outputs(sources, queries)
    library().dai_forward(pointers(sources), len(sources), queries.data_ptr(), queries.size(0), y0.data_ptr(),
                          pointers(inter), lse.data_ptr(), stats.data_ptr(), *launch_args(y0))
    return y0, inter, lse, stats


@depth_inter.register_fake
def depth_inter_fake(sources, queries):
    return inter_outputs(sources, queries)


@torch.library.custom_op("nanochat::depth_inter_backward", mutates_args=(), device_types="cuda")
def depth_inter_backward(sources: list[Tensor], queries: Tensor, y0: Tensor, stats: Tensor, dy0: Tensor,
                         dinter: list[Tensor], dlse: Tensor) -> tuple[list[Tensor], Tensor]:
    """Gradients wrt the sources (bf16) and the queries (fp32, summed over the rows in a fixed order: bitwise
    reproducible) of depth_inter, given those wrt its first input, inter-block averages and log-sum-exps."""
    K, S, (rows, C, stream) = len(sources), queries.size(0), launch_args(y0)
    dsources, dq = [torch.empty_like(s) for s in sources], torch.empty_like(queries)
    part = queries.new_empty((library().dai_backward_blocks(rows, K, S, C), S, C))  # the blocks' partial dq
    library().dai_backward(pointers(sources), K, queries.data_ptr(), S, y0.data_ptr(), dy0.data_ptr(), pointers(dinter),
                           dlse.data_ptr(), stats.data_ptr(), pointers(dsources), part.data_ptr(), dq.data_ptr(), rows,
                           C, stream)
    return dsources, dq


@depth_inter_backward.register_fake
def depth_inter_backward_fake(sources, queries, y0, stats, dy0, dinter, dlse):
    return [torch.empty_like(s) for s in sources], torch.empty_like(queries)


def depth_inter_setup(ctx, inputs, output):
    sources, queries = inputs
    y0, _, _, stats = output
    ctx.save_for_backward(*sources, queries, y0, stats)


def depth_inter_grad(ctx, dy0, dinter, dlse, dstats):
    """depth_inter's backward; its statistics take no gradient (dstats unused)."""
    *sources, queries, y0, stats = ctx.saved_tensors
    return depth_inter_backward(sources, queries, y0, stats, dy0, dinter, dlse)


depth_inter.register_autograd(depth_inter_grad, setup_context=depth_inter_setup)


@torch.library.custom_op("nanochat::depth_merge", mutates_args=(), device_types="cuda")
def depth_merge(inter: Tensor, lse: Tensor, partial: Tensor, query: Tensor) -> tuple[Tensor, Tensor]:
    """Phase 2 of a later sublayer: its input norm(h) (bf16), h the online-softmax merge of depth_inter's inter-block
    average inter (B, T, C) fp32 with log-sum-exp lse (B, T) and the block's partial sum (B, T, C) bf16 under the
    sublayer's query (C,) fp32; and the backward's statistics (B, T, MERGE_STATS) fp32. All contiguous."""
    y, stats = torch.empty_like(partial), query.new_empty((*partial.shape[:-1], MERGE_STATS))
    library().dam_forward(inter.data_ptr(), lse.data_ptr(), partial.data_ptr(), query.data_ptr(), y.data_ptr(),
                          stats.data_ptr(), *launch_args(y))
    return y, stats


@depth_merge.register_fake
def depth_merge_fake(inter, lse, partial, query):
    return torch.empty_like(partial), query.new_empty((*partial.shape[:-1], MERGE_STATS))


@torch.library.custom_op("nanochat::depth_merge_backward", mutates_args=(), device_types="cuda")
def depth_merge_backward(partial: Tensor, query: Tensor, y: Tensor, stats: Tensor,
                         dy: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Gradients wrt depth_merge's inter (fp32), lse, partial (bf16) and query (fp32, summed over the rows in a fixed
    order), given that wrt its output y."""
    rows, C, stream = launch_args(y)
    dinter, dlse = partial.new_empty(partial.shape, dtype=torch.float32), stats.new_empty(partial.shape[:-1])
    dpartial, dq = torch.empty_like(partial), torch.empty_like(query)
    part = query.new_empty((library().dam_backward_blocks(rows, C), C))  # the blocks' partial dq
    library().dam_backward(dy.data_ptr(), y.data_ptr(), stats.data_ptr(), partial.data_ptr(), query.data_ptr(),
                           dinter.data_ptr(), dlse.data_ptr(), dpartial.data_ptr(), part.data_ptr(), dq.data_ptr(),
                           rows, C, stream)
    return dinter, dlse, dpartial, dq


@depth_merge_backward.register_fake
def depth_merge_backward_fake(partial, query, y, stats, dy):
    return (partial.new_empty(partial.shape, dtype=torch.float32), stats.new_empty(partial.shape[:-1]),
            torch.empty_like(partial), torch.empty_like(query))


def depth_merge_setup(ctx, inputs, output):
    _, _, partial, query = inputs
    y, stats = output
    ctx.save_for_backward(partial, query, y, stats)


def depth_merge_grad(ctx, dy, dstats):
    """depth_merge's backward; its statistics take no gradient (dstats unused)."""
    return depth_merge_backward(*ctx.saved_tensors, dy)


depth_merge.register_autograd(depth_merge_grad, setup_context=depth_merge_setup)
