"""
The language-model head's loss straight from the hidden states x, over the softcapped logits z' = s * tanh(z / s),
z = x W^T, as model/gpt.py's head defines it, against K weighted targets per row (column 0 the next token; further
columns e.g. multi-token prediction or the next token's prefix token, data/tokenizer.py prefix_targets):

    loss = sum_k w_k * (sum over the rows whose target t_k is not ignore_index of (logsumexp(z') - z'[t_k])) / n_0,

n_0 the valid count of column 0 (the next token): every valid target weighs w_k under one normalizer, modded-nanogpt's
per-target weighting (track_1_short perf/kernels/cross_entropy.py:190-207). One GEMM writes the bf16 logits z; one pass
of a cross-entropy kernel (QuACK's CrossEntropy, Tri Dao, with the softcap applied in registers) reads each row and its
K target logits once, takes one log-sum-exp, writes the K losses and, in place of z, the logit gradient relative to the
scale a_0 = w_0 / n_0 (a_k = w_k / n_0):

    (sum_k c_k [t_k valid] * softmax(z') - sum_k c_k [t_k valid] onehot(t_k)) * (1 - tanh^2(z / s)),  c_k = w_k / w_0;

two GEMMs take dx and dW from it with alpha = a_0. With K = 1 and w = (1,) every coefficient is 1 and the arithmetic is
the single-target kernel's, bit for bit. The logits are bf16 and read once, where the compiled head keeps them in fp32
and passes over them several times.

z' lies in [-s, s], so the log-sum-exp is taken against the bound s instead of the row's maximum: exp(z' - s) lies in
[exp(-2s), 1], one reduction and no second pass over the row.

The compiled kernel is cached by QuACK's jit_cache under QUACK_CACHE_DIR, keyed on this file's contents (SOURCE), so an
edit recompiles it.
"""

import hashlib
import math
from functools import partial
from pathlib import Path
from typing import Optional

import cutlass
import cutlass.cute as cute
import quack.copy_utils as copy_utils
import torch
from cutlass import Float32, Int32, const_expr
from quack.cache import jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cross_entropy import CrossEntropy
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.reduce import row_reduce
from quack.reduction_base import ReductionBase
from torch import Tensor

from nanochat.ops.quack_gemm import gemm

SOURCE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class SoftcapCrossEntropy(CrossEntropy):
    """QuACK's CrossEntropy kernel over the softcapped logits s * tanh(z / s), the log-sum-exp against the bound s, K
    targets per row, over the first `vocab` of the row's N columns (the rest: the lm head's padding rows)."""

    def __init__(self, dtype, N, K, softcap, vocab):
        ReductionBase.__init__(self, dtype, N, stage=1, reduction_dtype=Float32)
        self.K = K
        self.softcap = softcap
        self.vocab = vocab

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,  # (M, N) logits z
        mTarget: cute.Tensor,  # (M, K)
        mTargetLogit: cute.Tensor,  # unused: the target logits are read from mX
        mLoss: cute.Tensor,  # (M, K)
        mLSE: Optional[cute.Tensor],  # unused
        mdX: Optional[cute.Tensor],  # (M, N) logit gradient, if computed (may alias mX)
        mWeight: Optional[cute.Tensor],  # (K,) fp32 gradient coefficients c_k, with mdX
        ignore_index: Int32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled

        shape = mX.shape
        # a cluster tile wider than the row (N not a multiple of it, e.g. the padded vocabulary): loads and stores
        # predicated on the row's length, as QuACK's CrossEntropy.kernel (quack/cross_entropy.py:152-156)
        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        idX = cute.make_identity_tensor(shape)
        gX, cX = [cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) for mT in (mX, idX)]

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16)
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        thr_copy = tiled_copy.get_slice(tidx)
        tXgX = thr_copy.partition_S(gX)
        tXsX = thr_copy.partition_D(sX)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]
        tXcFull = thr_copy.partition_S(cX)
        tXrX = cute.make_rmem_tensor_like(tXgX)
        tXpX = None if is_even_N else copy_utils.predicate_k(tXcFull, limit=shape[1])
        copy = partial(copy_utils.copy, pred=tXpX)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        row = tXcX[0][0]
        target = cute.make_rmem_tensor(self.K, Int32)
        for k in cutlass.range_constexpr(self.K):
            target[k] = Int32.zero
            if row < shape[0]:
                target[k] = Int32(mTarget[row, k])
        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(tXsX, tXrX)
        inv_cap = 1.0 / self.softcap
        t = cute.math.tanh(tXrX.load().to(Float32) * inv_cap, fastmath=False)
        x = t * self.softcap

        valid = [target[k] != ignore_index for k in range(self.K)]
        target_logit = cute.make_rmem_tensor(self.K, Float32)
        for k in cutlass.range_constexpr(self.K):
            target_logit[k] = Float32.zero
            if row < shape[0] and tXcX[0][1] == 0 and valid[k]:
                target_logit[k] = self.softcap * cute.math.tanh(Float32(mX[row, target[k]]) * inv_cap, fastmath=False)

        log2_e = math.log2(math.e)
        # the columns of the vocabulary only: padding columns (>= vocab) and the cluster tile's columns past the row
        # contribute exp 0 to the log-sum-exp and get gradient 0, whatever the registers hold there
        tXrE = cute.make_rmem_tensor_like(tXrX, Float32)
        tXrE.store(cute.math.exp2((x - self.softcap) * log2_e, fastmath=False))
        for i in cutlass.range(cute.size(tXrX), unroll_full=True):
            tXrE[i] = tXrE[i] if tXcFull[i][1] < self.vocab else Float32.zero
        exp_x = tXrE.load()
        denom = row_reduce(
            exp_x,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr + 0 if const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
        )

        if tXcX[0][1] == 0 and row < shape[0] and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0):
            lse = self.softcap + cute.math.log(denom, fastmath=True)
            for k in cutlass.range_constexpr(self.K):
                mLoss[row, k] = mLoss.element_type(lse - target_logit[k] if valid[k] else Float32.zero)

        if const_expr(mdX is not None):
            coef = [Float32(mWeight[k]) for k in range(self.K)]
            softmax_coef = Float32.zero  # sum of the valid targets' coefficients
            for k in cutlass.range_constexpr(self.K):
                softmax_coef = softmax_coef + (coef[k] if valid[k] else Float32.zero)
            denom_inv = cute.arch.rcp_approx(denom)
            gdX = cute.local_tile(mdX, tiler_mn, (bidx, cluster_y))
            tXgdX = thr_copy.partition_D(gdX)
            tXrdX_f32 = cute.make_rmem_tensor_like(tXrX, Float32)
            tXrdX_f32.store(exp_x * (denom_inv * softmax_coef))
            for k in cutlass.range_constexpr(self.K):
                if valid[k]:
                    for i in cutlass.range(cute.size(tXrX), unroll_full=True):
                        tXrdX_f32[i] = tXrdX_f32[i] if tXcFull[i][1] != target[k] else tXrdX_f32[i] - coef[k]
            tXrdX_f32.store(tXrdX_f32.load() * (1.0 - t * t))
            for i in cutlass.range(cute.size(tXrX), unroll_full=True):
                tXrdX_f32[i] = tXrdX_f32[i] if tXcFull[i][1] < self.vocab else Float32.zero
            tXrdX = tXrdX_f32.to(tXgdX.element_type)
            if row < shape[0]:
                copy(tXrdX, tXgdX)

    @staticmethod
    @jit_cache
    def compile(dtype, target_dtype, N, K, has_dx, softcap, vocab, source):
        batch_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, N)
        return cute.compile(
            SoftcapCrossEntropy(dtype, N, K, softcap, vocab),
            fake_tensor(dtype, (batch_sym, N), div),
            fake_tensor(target_dtype, (batch_sym, K)),
            None,
            fake_tensor(Float32, (batch_sym, K)),
            None,
            fake_tensor(dtype, (batch_sym, N), div) if has_dx else None,
            fake_tensor(Float32, (K,)) if has_dx else None,
            Int32(0),  # ignore_index, a runtime argument
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


def _losses(logits, target, softcap, ignore_index, vocab, dlogits, coef):
    """Each row's loss (fp32) against each target column of the logits (M, N) and targets (M, K) over the first vocab
    columns, and, if dlogits is given (it may be the logits), their gradient with coefficients coef (K,) fp32 into it
    (0 in the columns from vocab on)."""
    loss = torch.empty(target.shape, device=logits.device, dtype=torch.float32)
    SoftcapCrossEntropy.compile(torch2cute_dtype_map[logits.dtype], torch2cute_dtype_map[target.dtype], logits.size(1),
                                target.size(1), dlogits is not None, softcap, vocab, SOURCE)(
        logits, target, None, loss, None, dlogits, coef, Int32(ignore_index))
    return loss


# Opaque to torch.compile, as ops/flash_attn.py's FA4 ops are: the logit gradient overwrites the logits in place, an
# aliasing a traced graph must not see (functionalized, it corrupted the gradient of x).
@torch.library.custom_op("nanochat::softcap_lce", mutates_args=(), device_types="cuda")
def softcap_lce(x: Tensor, weight: Tensor, target: Tensor, weights: Tensor, softcap: float,
                ignore_index: int, vocab: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """x (M, C) bf16, weight (N, C) the master weight (fp32; its rows from vocab on the padding of the vocabulary to the
    GEMMs' alignment, outside the softmax and with gradient 0), rounded to x's dtype for the GEMMs, target (M, K) int64
    (every id < vocab), weights (K,) fp32 on x's device, weights[0] nonzero (the gradient's scale): (sum_k weights[k] *
    the summed loss of column k's targets other than ignore_index / column 0's valid count, its gradient wrt x, its
    gradient wrt weight in weight's dtype straight from the GEMM's fp32 accumulators, each column's mean loss over its
    own valid targets (K,) fp32 without gradient: monitoring). A tensor, so per-step weights do not recompile a compiled
    caller."""
    w = weight.to(x.dtype)
    logits = x @ w.t()
    count = (target != ignore_index).sum(0).float()
    scale = weights / count[0]  # a_k = w_k / n_0
    loss = _losses(logits, target, softcap, ignore_index, vocab, logits, weights / weights[0])
    column_sums = loss.sum(0)
    C = x.shape[1]
    return ((column_sums * scale).sum(), gemm(logits, w, "lm_dx", C, alpha=scale[0]),
            gemm(logits.t(), x, "lm_dw", C, out_dtype=weight.dtype, alpha=scale[0]), column_sums / count)


@softcap_lce.register_fake
def _(x, weight, target, weights, softcap, ignore_index, vocab):
    return (x.new_empty((), dtype=torch.float32), x.new_empty(x.shape), weight.new_empty(weight.shape),
            x.new_empty(target.shape[1:], dtype=torch.float32))


def _softcap_lce_setup(ctx, inputs, output):
    ctx.save_for_backward(output[1], output[2])


def _softcap_lce_backward(ctx, dloss, _ddx, _ddw, _dcolumns):
    dx, dw = ctx.saved_tensors
    return dx * dloss.to(dx.dtype), dw * dloss, None, None, None, None, None


softcap_lce.register_autograd(_softcap_lce_backward, setup_context=_softcap_lce_setup)


@torch.library.custom_op("nanochat::softcap_ce_losses", mutates_args=(), device_types="cuda")
def softcap_ce_losses(x: Tensor, weight: Tensor, target: Tensor, softcap: float, ignore_index: int, vocab: int) -> Tensor:
    """Each position's loss against each target column (M, K) fp32 over the first vocab columns, 0 where the target is
    ignore_index; no gradient. weight: the master weight (N, C), rounded to x's dtype for the GEMM."""
    return _losses(x @ weight.to(x.dtype).t(), target, softcap, ignore_index, vocab, None, None)


@softcap_ce_losses.register_fake
def _(x, weight, target, softcap, ignore_index, vocab):
    return x.new_empty(target.shape, dtype=torch.float32)
