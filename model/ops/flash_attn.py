"""
Attention of model/modeling.py as one torch.library op over the fused q/k/v projection (model/ops/linear.py
qkv_proj): the q/k prologue (model/ops/qk_prologue.py), the value embeddings' gate and mix (model/ops/ve_gate.py) and
FlashAttention-4 (Dao-AILab flash_attn.cute), causal with `window_left` keys before each query; FA4's own
autograd.Function cannot be traced by torch.compile. One op, so the backward writes every gradient of the projection
into one (B, T, 3C) buffer: FA4's dv into the v block (the value embeddings' backward reads it there; the mix passes
dv through to v), the prologue's dq, dk into the q and k blocks, with no concatenation and no autograd add.

Document masking (ProLong, arXiv 2410.02660): FA4's varlen kernels restrict attention to the segments in
cu_seqlens, the int32 offsets of the flattened (B*T) tokens (corpus/loader.py PretrainingBatches). cu_seqlens has a
fixed length per loader (one compiled shape), its unused entries empty segments; `segments`, a 0-dim int32 tensor on
the host (no device synchronization), is how many are not, and the op passes FA4 only those: FA4 schedules work for
every segment it is given, so padding segments cost time without changing outputs (McCormick, "Optimizing Training
with FlashAttention varlen", 2026).

Local layers' forward runs persistent. FA4's SM90 forward launches one CTA per (segment m-block, head)
(flash_fwd_sm90.py FlashAttentionForwardSm90.__call__: SingleTileVarlenScheduler, "TODO: dispatch to
DynamicPersistentVarlenScheduler"), and a window-512 m-block of 128 queries covers ~5 key blocks, so each CTA's launch,
Q load and output store are exposed. PersistentVarlenScheduler runs one CTA per SM over those tiles, and
PersistentForwardSm90 stages the output tile in its own shared buffer (FlashAttentionForwardSm90.kernel puts it in sQ,
which the producer refills with the next tile's Q once released, FlashAttentionForwardSm90.mma): d24 8K +0.32%
(docs/findings.md §5.11). Full causal layers keep FA4's launch: their m-blocks' work differs up to T / 128-fold, which
a static stride balances worse than the hardware's CTA scheduling (measured slower, §5.11).

Local layers' backward accumulates dQ in bf16 (BackwardSm90Bf16dQ): FA4's SM90 backward reduce-adds fp32 dQ partial
tiles into a zeroed fp32 dq_accum and converts it in a postprocess kernel (flash_bwd_postprocess.py); here the
consumers scale dQ by the softmax scale and stage it as bf16 rows, and the dQ warp reduce-adds each row straight into
dq with cp.reduce.async.bulk .add.noftz.bf16 (PTX ISA, cp.reduce.async.bulk): no fp32 zero fill, no postprocess,
interior tiles skip the mask; with the persistent forward d24 8K +0.94%, 32K +0.72%. A dq element there sums <= 5
bf16-rounded partials (one per 128-key block of the 512-key window; norm rel 2.45e-3 against fp32 accumulation);
full layers would sum up to T / 128 (64 at 8K, 256 at 32K), so they keep FA4's fp32 accumulator. Tile shapes and
stages are FA4's own (_tile_size_fwd_sm90, _tile_size_bwd_sm90): every other forward and backward configuration
measured on the d24 shapes was slower (§5.11).

The op ends with o_proj (model/ops/linear.py residual_rms: h = residual + out W_o^T and its rows' rstd), so its
backward takes dO from o_proj's dX GEMM together with FA4's dpsum = rowsum(dO O) per (row, head), summed in that
GEMM's epilogue (linear.py residual_rms_bwd), where dO is in registers and O one more tile load: FA4's backward
preprocess (flash_bwd_preprocess.py), which reads O and dO again for dpsum, gives way to fa4_rows
(model/kernels/fa4_rows.cu), which places dpsum and lse log2(e) at FA4's padded rows from (rows, H) and lse alone.
"""

import contextlib
import ctypes
import hashlib
import math
from pathlib import Path

import cutlass
import cutlass.cute as cute
import flash_attn.cute.flash_fwd_sm90 as flash_fwd_sm90
import flash_attn.cute.interface as fa4
import torch
from cutlass import Int32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import warpgroup
from cutlass.utils import HardwareInfo
from flash_attn.cute import utils
from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
from flash_attn.cute.flash_bwd_sm90 import FlashAttentionBackwardSm90
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
from flash_attn.cute.named_barrier import NamedBarrierBwd
from flash_attn.cute.tile_scheduler import SingleTileVarlenScheduler, WorkTileInfo
from flash_attn.cute.utils import AuxData
from quack import copy_utils, layout_utils
from torch import Tensor
from torch.autograd.function import FunctionCtx

from model.config import HEAD_DIM
from model.ops.linear import residual_rms, residual_rms_bwd
from model.ops.qk_prologue import qk_backward, qk_forward
from model.ops.ve_gate import ve_backward, ve_forward
from native import cuda_library, pointer, stream


class PersistentVarlenScheduler(SingleTileVarlenScheduler):
    """SingleTileVarlenScheduler's tile decode (VarlenDecoder.decode) with one CTA per SM, CTA i taking the tiles
    i, i + grid, i + 2 grid, ..."""

    @staticmethod
    def get_grid_shape(params, *, loc=None, ip=None):
        d = params.decoder
        tiles = (params.total_q + d.num_batch * (d.tile_shape_mn[0] - 1)) // d.tile_shape_mn[0] * d.num_head
        return cutlass.min(HardwareInfo().get_device_multiprocessor_count(), tiles), Int32(1), Int32(1)

    @staticmethod
    @cute.jit
    def create(params, ctx=None, *, loc=None, ip=None):
        return PersistentVarlenScheduler(params, cute.arch.block_idx()[0], Int32(0), loc=loc, ip=ip)

    @cute.jit
    def _decode_work_tile(self) -> WorkTileInfo:
        block, head, batch, _, _, _, is_valid = self.params.decoder.decode(self._tile_idx, Int32(0), Int32(0))
        return WorkTileInfo((Int32(block), Int32(head), Int32(batch), Int32(0)), is_valid)

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._tile_idx += cute.arch.grid_dim()[0]
        return self.get_current_work()

    def __new_from_mlir_values__(self, values):
        objs = []
        for obj, n in zip((self.params, self._tile_idx, self._split_idx), self._values_pos, strict=True):
            objs.append(cutlass.new_from_mlir_values(obj, values[:n]))
            values = values[n:]
        return PersistentVarlenScheduler(*objs, loc=self._loc)


class PersistentForwardSm90(flash_fwd_sm90.FlashAttentionForwardSm90):
    """FlashAttentionForwardSm90 with the output tile in its own shared buffer after sQ (sQ's allocation doubled;
    the rest of FlashAttentionForwardSm90._get_shared_storage_cls unchanged)."""

    def _get_shared_storage_cls(self):
        q_t, k_t, v_t = (cute.struct.Align[cute.struct.MemRange[self.dtype, n], self.buffer_align_bytes]
                         for n in (2 * self.tile_m * self.tile_hdim, cute.cosize(self.sK_layout),
                                   cute.cosize(self.sV_layout)))
        p_t = cute.struct.Align[cute.struct.MemRange[
            self.dtype, cute.cosize(self.sP_layout) if self.sP_layout is not None else 0], 1024]
        mbar_q_t = cute.struct.MemRange[cutlass.Int64, 2]
        mbar_kv_t = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: mbar_q_t
            mbar_ptr_K: mbar_kv_t
            mbar_ptr_V: mbar_kv_t
            sV: v_t
            sQ: q_t
            sK: k_t
            sP: p_t

        return SharedStorageQKV

    @cute.jit
    def mma(self, tiled_mma_qk, tiled_mma_pv, mO, mLSE, sQ, sK, sVt, sP, sO, *args, **kwargs):
        sO = cute.make_tensor(sO.iterator + self.tile_m * self.tile_hdim, sO.layout)
        super().mma(tiled_mma_qk, tiled_mma_pv, mO, mLSE, sQ, sK, sVt, sP, sO, *args, **kwargs)


# FA4's persistent JIT cache keys on FA4's sources (cache_utils.get_jit_cache): this file's hash keys the subclasses
SOURCE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
PERSISTENT_FWD_CACHE = get_jit_cache(f"fwd_persistent_{SOURCE_HASH}")
BF16DQ_CACHE = get_jit_cache(f"bwd_bf16dq_{SOURCE_HASH}")


@contextlib.contextmanager
def persistent_forward():
    """_flash_attn_fwd compiling PersistentForwardSm90 with PersistentVarlenScheduler, cached apart from FA4's own
    kernels (interface.py and flash_fwd_sm90.py look both classes up by module name)."""
    saved = fa4.FlashAttentionForwardSm90, flash_fwd_sm90.SingleTileVarlenScheduler, _flash_attn_fwd.compile_cache
    fa4.FlashAttentionForwardSm90 = PersistentForwardSm90
    flash_fwd_sm90.SingleTileVarlenScheduler = PersistentVarlenScheduler
    _flash_attn_fwd.compile_cache = PERSISTENT_FWD_CACHE
    try:
        yield
    finally:
        fa4.FlashAttentionForwardSm90, flash_fwd_sm90.SingleTileVarlenScheduler, _flash_attn_fwd.compile_cache = saved


def fa4_rows(rows: Tensor, lse: Tensor, cu: Tensor, m: int, dpsum: Tensor, lse_log2: Tensor) -> None:
    """FA4's backward row statistics (H, padded) fp32 at its padded rows (m: its backward m-block) into dpsum and
    lse_log2, from rows (total, H) fp32, the o_proj epilogue's dpsum, and lse (H, total) fp32
    (model/kernels/fa4_rows.cu); cu int32 (segments + 1)."""
    H, padded = dpsum.shape
    cuda_library("fa4_rows").fa4_rows(pointer(rows), pointer(lse), ctypes.c_longlong(lse.stride(0)), pointer(cu),
                                      cu.shape[0] - 1, H, m,
                                      ctypes.c_longlong(padded), pointer(dpsum), pointer(lse_log2), stream())


@contextlib.contextmanager
def precomputed_dpsum(rows: Tensor):
    """_flash_attn_bwd with its preprocess (interface.py _bwd_preprocess, looked up by module name), which reads O and
    dO for dpsum, replaced by fa4_rows from rows (total, H), and the preprocess's zero fill of dq_accum."""
    saved = fa4._bwd_preprocess

    def preprocess(out, dout, dpsum, lse, lse_log2, dq_accum, cu_seqlens_q, seqused_q, dlse, dtype, head_dim,
                   head_dim_v, m_block_size, *, cu_total_m_blocks, fake_mode, hdim_multiple_of):
        fa4_rows(rows, lse, cu_seqlens_q, m_block_size, dpsum, lse_log2)
        dq_accum.zero_()

    fa4._bwd_preprocess = preprocess
    try:
        yield
    finally:
        fa4._bwd_preprocess = saved


def reduce_add_bf16(smem_ptr: cute.Pointer, gmem_ptr: cute.Pointer, nbytes: int):
    """gmem[0:nbytes] += smem[0:nbytes] as bf16, asynchronously (PTX ISA, cp.reduce.async.bulk; a bulk group)."""
    llvm.inline_asm(
        None,
        [gmem_ptr.llvm_ptr, smem_ptr.toint().ir_value(), Int32(nbytes).ir_value()],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class BackwardSm90Bf16dQ(FlashAttentionBackwardSm90):
    def __init__(self, *args, num_heads: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        # The staging and store below are written for FA4's hdim-128 causal/local config (_tile_size_bwd_sm90)
        assert self.tile_m == 64 and self.tile_hdim == 128 and not self.dQ_swapAB and self.num_wg_dQ == 2
        assert self.AtomLayoutMdQ == 1 and self.mma_dkv_is_rs
        # Staged dQ rows tile_hdim + one 16-byte vector apart: 16-byte aligned, and the MMA C layout's rows fall in
        # different banks
        self.pitch = self.tile_hdim + 128 // self.dtype.width
        self.row_bytes = self.tile_hdim * self.dtype.width // 8

    @cute.jit
    def _needs_mask(self, mask_fn, m_block):
        """Whether the (m_block, n_block) tile crosses the causal diagonal, the window's left edge or a segment end
        (FlashAttentionBackwardSm90.mma masks every tile)."""
        mask, n_block = mask_fn.func.__self__, mask_fn.keywords["n_block"]
        m0, n0 = m_block * self.tile_m, n_block * self.tile_n
        need = (m0 + self.tile_m > mask.seqlen_q) or (n0 + self.tile_n > mask.seqlen_k)
        need = need or (n0 + self.tile_n - 1 > m0 + mask.seqlen_k - mask.seqlen_q)
        if const_expr(self.is_local and mask.window_size_left is not None):
            need = need or (n0 < m0 + self.tile_m - 1 + mask.seqlen_k - mask.seqlen_q - mask.window_size_left)
        return need

    def _check_type(self, *types):
        pass  # mdQaccum is dq (bf16) viewed as fp32

    def _sdQ(self, sdQaccum):
        return cute.make_tensor(cute.recast_ptr(sdQaccum.iterator, dtype=self.dtype),
                                cute.make_layout((self.tile_m, self.tile_hdim), stride=(self.pitch, 1)))

    @cute.jit
    def mma(self, tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ, mdK, mdV, mdK_semaphore, mdV_semaphore,
            mdQaccum, sQ, sK, sV, sdO, sP, sdS, sLSE, sdPsum, sdQaccum, pipeline_Q, pipeline_dO, tidx, tma_atom_dK,
            tma_atom_dV, r2s_tiled_copy_dQaccum, softmax_scale_log2, softmax_scale, *args, **kwargs):
        self._tdQsdQ = tiled_mma_dQ.get_slice(tidx).partition_C(self._sdQ(sdQaccum))
        self._softmax_scale = softmax_scale
        super().mma(tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ, mdK, mdV, mdK_semaphore, mdV_semaphore,
                    mdQaccum, sQ, sK, sV, sdO, sP, sdS, sLSE, sdPsum, sdQaccum, pipeline_Q, pipeline_dO, tidx,
                    tma_atom_dK, tma_atom_dV, r2s_tiled_copy_dQaccum, softmax_scale_log2, softmax_scale, *args,
                    **kwargs)

    @cute.jit
    def mma_one_m_block(self, m_block, consumer_state_Q, consumer_state_dO, warp_group_idx, mma_qk_fn, mma_dov_fn,
                        mma_pdo_fn, mma_dsq_fn, mma_dsk_fn, copy_P_r2s, copy_dS_r2s, pipeline_Q, pipeline_dO,
                        tLSEsLSE, tLSEsdPsum, tdQsdQaccum, softmax_scale_log2, PdS_barrier,
                        is_dQ_wg: cutlass.Constexpr[bool] = True, mask_fn=None, score_mod_fn=None,
                        score_mod_bwd_fn=None, dKV_accumulate=True):
        # FlashAttentionBackwardSm90.mma_one_m_block (mma_dkv_is_rs, Q_stage == dO_stage == PdS_stage, so one
        # pipeline state and stage index serve Q, dO, P and dS; no score_mod) with dQ staged as scaled bf16 rows
        # instead of fp32 accumulator fragments, and the mask applied only to tiles that cross a boundary
        smem_idx = consumer_state_Q.index
        pipeline_Q.consumer_wait(consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q))
        acc_S = mma_qk_fn(A_idx=smem_idx, wg_wait=-1)
        tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, smem_idx])
        pipeline_dO.consumer_wait(consumer_state_Q, pipeline_dO.consumer_try_wait(consumer_state_Q))
        acc_dP = mma_dov_fn(A_idx=smem_idx, wg_wait=1)
        if self._needs_mask(mask_fn, m_block):
            mask_fn(acc_S, m_block=m_block)
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.SdP_swapAB)
        lane_idx = cute.arch.lane_idx()
        for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
            lse_val = self._get_stat(tLSErLSE, r, lane_idx, shuffle=self.shuffle_LSE)
            for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                acc_S_mn[r, c] = cute.math.exp2(acc_S_mn[r, c] * softmax_scale_log2 - lse_val, fastmath=True)
        tLSErdPsum = copy_utils.load_s2r(tLSEsdPsum[None, smem_idx])
        tdVrP = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), self.dtype)
        warpgroup.wait_group(0)
        acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=self.SdP_swapAB)
        for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
            dpsum_val = self._get_stat(tLSErdPsum, r, lane_idx, shuffle=self.shuffle_dPsum)
            for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
                acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - dpsum_val)
        tdKrdS = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_dP), self.dtype)
        copy_dS_r2s(tdKrdS, dst_idx=smem_idx)
        mma_pdo_fn(tCrA=tdVrP, B_idx=smem_idx, zero_init=not dKV_accumulate, wg_wait=-1)
        cute.arch.fence_view_async_shared()
        PdS_barrier.arrive_and_wait()
        acc_dQ = mma_dsk_fn(A_idx=smem_idx, wg_wait=1)
        pipeline_dO.consumer_release(consumer_state_Q)
        mma_dsq_fn(tCrA=tdKrdS, B_idx=smem_idx, zero_init=not dKV_accumulate, wg_wait=1)
        rdQ = cute.make_fragment_like(acc_dQ, self.dtype)
        rdQ.store((acc_dQ.load() * self._softmax_scale).to(self.dtype))
        cute.arch.barrier(barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                          number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE)
        cute.autovec_copy(rdQ, self._tdQsdQ)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier_arrive(barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                                 number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE)
        warpgroup.wait_group(0)
        pipeline_Q.consumer_release(consumer_state_Q)
        consumer_state_Q.advance()
        consumer_state_dO.advance()
        return consumer_state_Q, consumer_state_dO

    @cute.jit
    def dQaccum_store(self, mdQaccum, sdQaccum, block_info, TileSchedulerCls, SeqlenInfoCls,
                      blocksparse_tensors=None, mdQ_semaphore=None):
        """The dQ warp: per m-block, once the previous m-block's bulk group has been read releases the dQ warpgroups'
        staging buffers, waits for their staged rows and reduce-adds every valid row (token, head) into dq."""
        lane = cute.arch.lane_idx()
        sdQ = self._sdQ(sdQaccum)
        dq = cute.recast_ptr(mdQaccum.iterator, dtype=self.dtype)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                rows = seqlen.seqlen_q - m_block * self.tile_m
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                for wg in cutlass.range_constexpr(self.num_wg_dQ):
                    cute.arch.barrier_arrive(barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + wg,
                                             number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE)
                for wg in cutlass.range_constexpr(self.num_wg_dQ):
                    cute.arch.barrier(barrier_id=int(NamedBarrierBwd.dQFullWG0) + wg,
                                      number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE)
                for i in cutlass.range_constexpr(self.tile_m // cute.arch.WARP_SIZE):
                    row = lane + i * cute.arch.WARP_SIZE
                    if row < rows:
                        token = seqlen.offset_q + m_block * self.tile_m + row
                        gptr = dq + (token * self.num_heads + head_idx) * self.tile_hdim
                        reduce_add_bf16(sdQ.iterator + row * self.pitch, gptr, self.row_bytes)
                cute.arch.cp_async_bulk_commit_group()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        cute.arch.cp_async_bulk_wait_group(0, read=True)


def bwd_bf16dq(q: Tensor, k: Tensor, v: Tensor, dout: Tensor, lse: Tensor, rows: Tensor, cu: Tensor, window: int,
               dv: Tensor) -> tuple[Tensor, Tensor]:
    """q, k, dout (total, H, D) bf16, contiguous; v and dv (total, H, D) with heads contiguous, any token stride;
    lse (H, total) fp32; rows (total, H) fp32 the dpsum of o_proj's epilogue; cu int32 (segments + 1); window: the
    left window -> (dq, dk); dv written in place."""
    total, H, D = q.shape
    causal, local, wl, wr = fa4._resolve_causal_local_window(True, window, None)
    cfg = fa4._tile_size_bwd_sm90(D, D, causal, local)
    m = cfg.m_block_size
    padded = (total + cu.shape[0] * m - 1) // m * m
    dpsum = torch.empty(H, padded, dtype=torch.float32, device=q.device)
    lse_log2 = torch.empty(H, padded, dtype=torch.float32, device=q.device)
    dq, dk = torch.zeros_like(q), torch.empty_like(k)
    dtype = fa4.torch2cute_dtype_map[q.dtype]
    fa4_rows(rows, lse, cu, m, dpsum, lse_log2)
    dq_as_f32 = dq.view(torch.float32).view(total, H * D // 2)
    key = H
    softmax_scale = 1.0 / math.sqrt(D)
    if key not in BF16DQ_CACHE:
        obj = BackwardSm90Bf16dQ(
            dtype, D, D, 1, causal, is_local=local, deterministic=False, tile_m=m, tile_n=cfg.n_block_size,
            Q_stage=cfg.num_stages_Q, dO_stage=cfg.num_stages_dO, PdS_stage=cfg.num_stages_PdS,
            SdP_swapAB=cfg.SdP_swapAB, dKV_swapAB=cfg.dKV_swapAB, dQ_swapAB=cfg.dQ_swapAB,
            AtomLayoutMSdP=cfg.AtomLayoutMSdP, AtomLayoutNdKV=cfg.AtomLayoutNdKV, AtomLayoutMdQ=cfg.AtomLayoutMdQ,
            num_threads=(cfg.num_wg + 1) * 128, num_heads=H)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        ts = [to_cute_tensor(t) for t in (q, k, v, dout, lse_log2, dpsum, dq_as_f32, dk, dv)]
        cu_t = to_cute_tensor(cu, assumed_align=4)
        BF16DQ_CACHE[key] = cute.compile(obj, *ts, softmax_scale, cu_t, cu_t, None, None, wl, wr, None, None, None,
                                         AuxData(None, None), None, None, stream, options="--enable-tvm-ffi")
    BF16DQ_CACHE[key](q, k, v, dout, lse_log2, dpsum, dq_as_f32, dk, dv, softmax_scale, cu, cu, None, None, wl, wr,
                      None, None, None, AuxData(None, None), None, None)
    return dq, dk


def fa4_window_left(window_left: int, T: int) -> int | None:
    """None (FA4's causal kernel) when the window spans T, else the local window: a window spanning T is plain causal
    attention, so the cheaper causal kernel computes it; FA4 keeps the local kernel (interface.py
    _resolve_causal_local_window)."""
    return None if window_left >= T else window_left


def segmented(cu_seqlens: Tensor, segments: Tensor) -> Tensor:
    """cu_seqlens' first segments + 1 offsets: the segments that are not padding."""
    return cu_seqlens[:int(segments) + 1]


def heads(qkv: Tensor, i: int) -> Tensor:
    """Block i (q 0, k 1, v 2) of the fused (B, T, 3C) projection as a (B, T, H, HEAD_DIM) view, token stride 3C."""
    C = qkv.size(-1) // 3
    return qkv[..., i * C:(i + 1) * C].unflatten(-1, (-1, HEAD_DIM))


@torch.library.custom_op("nanochat::attention", mutates_args=(), device_types="cuda")
def attention(qkv: Tensor, q_norm_weight: Tensor, k_norm_weight: Tensor, cos: Tensor, sin: Tensor, window_left: int,
              cu_seqlens: Tensor, segments: Tensor, o_weight: Tensor, o_grad: Tensor | None, residual: Tensor,
              n: Tensor | None, input_ids: Tensor | None, table: Tensor | None, w_gate: Tensor | None,
              ids: Tensor | None,
              order: Tensor | None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """qkv (B, T, 3C) bf16; norm weights (HEAD_DIM,) fp32; cos/sin (T_table >= T, HEAD_DIM / 2) fp32; o_weight (C, C)
    o_proj's bf16 copy, o_grad its fp32 .grad or None (forward only), residual (B, T, C) the stream; value embeddings
    when table is given (model/ops/ve_gate.py: n the normed stream, input_ids, table, w_gate; ids, order torch.sort
    of input_ids, for the backward) -> (h = residual + o_proj(out) (B, T, C), its rows' rstd (B*T,) fp32
    (non-differentiable: relu2_mlp, its one consumer, differentiates through it), out (B, T, H, HEAD_DIM), lse fp32
    (H, B*T), normed q, k, mixed v, gate), the last six for the backward (mixed v and gate empty without value
    embeddings)."""
    T, cu = qkv.size(1), segmented(cu_seqlens, segments)
    window = fa4_window_left(window_left, T)
    qn, kn = qk_forward(heads(qkv, 0), heads(qkv, 1), q_norm_weight, k_norm_weight, cos, sin)
    if table is None:
        v, vmix, gate = heads(qkv, 2), qkv.new_empty(0), qkv.new_empty(0, dtype=torch.float32)
    else:
        vmix, gate = ve_forward(n, heads(qkv, 2), input_ids, table, w_gate)
        v = vmix
    with contextlib.nullcontext() if window is None else persistent_forward():
        out, lse = _flash_attn_fwd(qn.flatten(0, 1), kn.flatten(0, 1), v.flatten(0, 1), cu_seqlens_q=cu,
                                   cu_seqlens_k=cu, max_seqlen_q=T, max_seqlen_k=T, causal=True,
                                   window_size_left=window, return_lse=True)[:2]
    h, rstd = residual_rms(out.flatten(1), o_weight, residual.flatten(0, 1))
    return h.view(residual.shape), rstd, out.view(qn.shape), lse, qn, kn, vmix, gate


@attention.register_fake
def attention_fake(qkv: Tensor, q_norm_weight: Tensor, k_norm_weight: Tensor, cos: Tensor, sin: Tensor,
                   window_left: int, cu_seqlens: Tensor, segments: Tensor, o_weight: Tensor, o_grad: Tensor | None,
                   residual: Tensor, n: Tensor | None, input_ids: Tensor | None, table: Tensor | None,
                   w_gate: Tensor | None, ids: Tensor | None,
                   order: Tensor | None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    q = heads(qkv, 0)
    B, T, H = q.shape[:3]
    vmix = qkv.new_empty(0) if table is None else q.new_empty(q.shape)
    gate = qkv.new_empty(0 if table is None else (B * T, H), dtype=torch.float32)
    return (residual.new_empty(residual.shape), q.new_empty(B * T, dtype=torch.float32), q.new_empty(q.shape),
            q.new_empty((H, B * T), dtype=torch.float32), q.new_empty(q.shape), q.new_empty(q.shape), vmix, gate)


@torch.library.custom_op("nanochat::attention_bwd", mutates_args=("o_grad",), device_types="cuda")
def attention_bwd(dh: Tensor, qkv: Tensor, q_norm_weight: Tensor, k_norm_weight: Tensor, cos: Tensor, sin: Tensor,
                  out: Tensor, lse: Tensor, qn: Tensor, kn: Tensor, vmix: Tensor, gate: Tensor, window_left: int,
                  cu_seqlens: Tensor, segments: Tensor, o_weight: Tensor, o_grad: Tensor, n: Tensor | None,
                  table: Tensor | None, w_gate: Tensor | None, ids: Tensor | None,
                  order: Tensor | None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """(d qkv (B, T, 3C), d q_norm_weight, d k_norm_weight, dn, dw_gate; the last two empty without value
    embeddings, whose table's gradient is added into table.grad; o_proj's dW accumulated into o_grad); FA4's
    backward on dO and dpsum from o_proj's dX GEMM, local layers' dq accumulated in bf16 (bwd_bf16dq), atomically
    (not bitwise reproducible)."""
    T, cu = qkv.size(1), segmented(cu_seqlens, segments)
    window = fa4_window_left(window_left, T)
    dout, rows = residual_rms_bwd(dh.flatten(0, 1), out.flatten(2).flatten(0, 1), o_weight, o_grad)
    dqkv = torch.empty_like(qkv)
    dv = heads(dqkv, 2)
    v = heads(qkv, 2) if table is None else vmix
    q, k, dout = qn.flatten(0, 1), kn.flatten(0, 1), dout.view(out.shape).flatten(0, 1)
    if window is None:
        with precomputed_dpsum(rows):
            dqn, dkn = _flash_attn_bwd(q, k, v.flatten(0, 1), out.flatten(0, 1), dout, lse, causal=True,
                                       cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=T, max_seqlen_k=T,
                                       dv=dv.flatten(0, 1))[:2]
    else:
        dqn, dkn = bwd_bf16dq(q, k, v.flatten(0, 1), dout, lse, rows, cu, window, dv.flatten(0, 1))
    dq_norm_weight, dk_norm_weight = qk_backward(heads(qkv, 0), heads(qkv, 1), q_norm_weight, k_norm_weight,
                                                 dqn.view(qn.shape), dkn.view(kn.shape), cos, sin, heads(dqkv, 0),
                                                 heads(dqkv, 1))
    if table is None:
        return dqkv, dq_norm_weight, dk_norm_weight, qkv.new_empty(0), qkv.new_empty(0)
    return dqkv, dq_norm_weight, dk_norm_weight, *ve_backward(dv, n, ids, order, table, w_gate, gate)


@attention_bwd.register_fake
def attention_bwd_fake(dh: Tensor, qkv: Tensor, q_norm_weight: Tensor, k_norm_weight: Tensor, cos: Tensor,
                       sin: Tensor, out: Tensor, lse: Tensor, qn: Tensor, kn: Tensor, vmix: Tensor, gate: Tensor,
                       window_left: int, cu_seqlens: Tensor, segments: Tensor, o_weight: Tensor, o_grad: Tensor,
                       n: Tensor | None, table: Tensor | None, w_gate: Tensor | None, ids: Tensor | None,
                       order: Tensor | None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    ve = table is not None
    return (torch.empty_like(qkv), torch.empty_like(q_norm_weight), torch.empty_like(k_norm_weight),
            torch.empty_like(n) if ve else qkv.new_empty(0), torch.empty_like(w_gate) if ve else qkv.new_empty(0))


def attention_save(ctx: FunctionCtx, inputs: tuple, output: tuple[Tensor, ...]) -> None:
    (qkv, q_norm_weight, k_norm_weight, cos, sin, ctx.window_left, cu_seqlens, segments, o_weight, o_grad, _, n, _,
     table, w, ids, order) = inputs
    ctx.mark_non_differentiable(output[1])
    ctx.save_for_backward(qkv, q_norm_weight, k_norm_weight, cos, sin, *output[2:], cu_seqlens, segments, o_weight,
                          o_grad, n, table, w, ids, order)


def attention_grad(ctx: FunctionCtx, dh: Tensor, *_: Tensor) -> tuple:
    """Gradients of qkv, the norm weights, the residual (dh), the normed stream and the gate weight (the table's goes
    to table.grad, o_proj's to its .grad); the outputs other than h and rstd are for the backward only."""
    (qkv, qnw, knw, cos, sin, out, lse, qn, kn, vmix, gate, cu_seqlens, segments, o_weight, o_grad, n, table, w, ids,
     order) = ctx.saved_tensors
    dqkv, dqnw, dknw, dn, dw = attention_bwd(dh, qkv, qnw, knw, cos, sin, out, lse, qn, kn, vmix, gate,
                                             ctx.window_left, cu_seqlens, segments, o_weight, o_grad, n, table, w,
                                             ids, order)
    ve = table is not None
    return (dqkv, dqnw, dknw, None, None, None, None, None, None, None, dh, dn if ve else None, None, None,
            dw if ve else None, None, None)


attention.register_autograd(attention_grad, setup_context=attention_save)
