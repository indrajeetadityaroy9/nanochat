"""
FlashAttention-4 (Dao-AILab flash_attn.cute) as torch.library ops, causal with `window_left` keys before each query;
FA4's own autograd.Function cannot be traced by torch.compile.

Document masking (ProLong, arXiv 2410.02660): FA4's varlen kernels restrict attention to the segments in
cu_seqlens, the int32 offsets of the flattened (B*T) tokens (corpus/loader.py PretrainingBatches). cu_seqlens has a
fixed length per loader (one compiled shape), its unused entries empty segments; `segments`, a 0-dim int32 tensor on
the host (no device synchronization), is how many are not, and the ops pass FA4 only those: FA4 schedules work for
every segment it is given, so padding segments cost time without changing outputs (McCormick, "Optimizing Training
with FlashAttention varlen", 2026).

Local layers' forward runs persistent. FA4's SM90 forward launches one CTA per (segment m-block, head)
(flash_fwd_sm90.py FlashAttentionForwardSm90.__call__: SingleTileVarlenScheduler, "TODO: dispatch to
DynamicPersistentVarlenScheduler"), and a window-512 m-block of 128 queries covers ~5 key blocks, so each CTA's launch,
Q load and output store are exposed. PersistentVarlenScheduler runs one CTA per SM over those tiles, and
PersistentForwardSm90 stages the output tile in its own shared buffer (FlashAttentionForwardSm90.kernel puts it in sQ,
which the producer refills with the next tile's Q once released, FlashAttentionForwardSm90.mma). Full causal layers
keep FA4's launch: their m-blocks' work differs up to T / 128-fold, which a static stride balances worse than the
hardware's CTA scheduling (measured slower).

The backward accumulates dQ in bf16 (BackwardSm90Bf16dQ): FA4's SM90 backward reduce-adds fp32 dQ partial tiles into
a zeroed fp32 dq_accum and converts it in a postprocess kernel (flash_bwd_postprocess.py); here the consumers scale
dQ by the softmax scale and stage it as bf16 rows, and the dQ warp reduce-adds each 256-byte row straight into dq with
cp.reduce.async.bulk .add.noftz.bf16 (PTX ISA, cp.reduce.async.bulk): no fp32 zero fill, no postprocess, interior
tiles skip the mask. Local layers only: a dq element there sums <= 5 bf16-rounded partials (one per 128-key block of
the 512-key window), about one extra bf16 rounding; full layers would sum up to T / 128 (64 at 8K, 256 at 32K), so they
keep FA4's fp32 accumulator.
"""

import contextlib
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



BF16DQ_CACHE = get_jit_cache(f"bwd_bf16dq_{SOURCE_HASH}")


PITCH = 136  # bf16 elements per staged dQ row: 128 + 8, rows 272 B apart (16 B aligned, conflict-free for the MMA C layout)


def reduce_add_bf16(smem_ptr: cute.Pointer, gmem_ptr: cute.Pointer, nbytes: int):
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
        assert self.tile_m == 64 and self.tile_hdim == 128 and not self.dQ_swapAB and self.num_wg_dQ == 2
        assert self.AtomLayoutMdQ == 1 and self.mma_dkv_is_rs

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
                                cute.make_layout((self.tile_m, self.tile_hdim), stride=(PITCH, 1)))

    @cute.jit
    def mma(self, tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ, mdK, mdV, mdK_semaphore, mdV_semaphore,
            mdQaccum, sQ, sK, sV, sdO, sP, sdS, sLSE, sdPsum, sdQaccum, pipeline_Q, pipeline_dO, tidx, tma_atom_dK,
            tma_atom_dV, r2s_tiled_copy_dQaccum, softmax_scale_log2, softmax_scale, *args, **kwargs):
        self._tdQsdQ = tiled_mma_dQ.get_slice(tidx).partition_C(self._sdQ(sdQaccum))
        self._tdQsdQ_base = self._sdQ(sdQaccum).iterator
        self._dq = cute.recast_ptr(mdQaccum.iterator, dtype=self.dtype)
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
        # FlashAttentionBackwardSm90.mma_one_m_block (mma_dkv_is_rs, Q_stage == dO_stage == PdS_stage, no score_mod)
        # with dQ staged as scaled bf16 rows instead of fp32 accumulator fragments
        consumer_state_dO_cur = consumer_state_Q
        smem_idx_Q = consumer_state_Q.index
        smem_idx_dO = consumer_state_dO_cur.index
        smem_idx_PdS = smem_idx_Q
        pipeline_Q.consumer_wait(consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q))
        acc_S = mma_qk_fn(A_idx=smem_idx_Q, wg_wait=-1)
        tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, smem_idx_Q])
        pipeline_dO.consumer_wait(consumer_state_dO_cur, pipeline_dO.consumer_try_wait(consumer_state_dO_cur))
        acc_dP = mma_dov_fn(A_idx=smem_idx_Q, wg_wait=1)
        if cutlass.const_expr(mask_fn is not None):  # noqa: SIM102 (compile-time outer, runtime inner condition)
            if self._needs_mask(mask_fn, m_block):
                mask_fn(acc_S, m_block=m_block)
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.SdP_swapAB)
        lane_idx = cute.arch.lane_idx()
        for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
            lse_val = self._get_stat(tLSErLSE, r, lane_idx, shuffle=self.shuffle_LSE)
            for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                acc_S_mn[r, c] = cute.math.exp2(acc_S_mn[r, c] * softmax_scale_log2 - lse_val, fastmath=True)
        tLSErdPsum = copy_utils.load_s2r(tLSEsdPsum[None, smem_idx_dO])
        tdVrP = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), self.dtype)
        warpgroup.wait_group(0)
        acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=self.SdP_swapAB)
        for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
            dpsum_val = self._get_stat(tLSErdPsum, r, lane_idx, shuffle=self.shuffle_dPsum)
            for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
                acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - dpsum_val)
        tdKrdS = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_dP), self.dtype)
        copy_dS_r2s(tdKrdS, dst_idx=smem_idx_PdS)
        mma_pdo_fn(tCrA=tdVrP, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1)
        cute.arch.fence_view_async_shared()
        PdS_barrier.arrive_and_wait()
        acc_dQ = mma_dsk_fn(A_idx=smem_idx_PdS, wg_wait=1)
        pipeline_dO.consumer_release(consumer_state_dO_cur)
        mma_dsq_fn(tCrA=tdKrdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1)
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
                for wg in cutlass.range_constexpr(2):
                    cute.arch.cp_async_bulk_wait_group(1 - wg, read=True)
                    cute.arch.barrier_arrive(barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + wg,
                                             number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE)
                for wg in cutlass.range_constexpr(2):
                    cute.arch.barrier(barrier_id=int(NamedBarrierBwd.dQFullWG0) + wg,
                                      number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE)
                for i in cutlass.range_constexpr(self.tile_m // cute.arch.WARP_SIZE):
                    row = lane + i * cute.arch.WARP_SIZE
                    if row < rows:
                        token = seqlen.offset_q + m_block * self.tile_m + row
                        gptr = dq + (token * self.num_heads + head_idx) * self.tile_hdim
                        reduce_add_bf16(sdQ.iterator + row * PITCH, gptr, 256)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_commit_group()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        cute.arch.cp_async_bulk_wait_group(0, read=True)


def bwd_bf16dq(q, k, v, out, dout, lse, cu, T, window):
    """q, k, v, out, dout (total, H, D) bf16; lse (H, total) fp32; cu int32 (batch + 1); window: left window or None.
    -> (dq, dk, dv)."""
    total, H, D = q.shape
    causal, local, wl, wr = fa4._resolve_causal_local_window(True, window, None)
    cfg = fa4._tile_size_bwd_sm90(D, D, causal, local)
    m = cfg.m_block_size
    padded = (total + cu.shape[0] * m - 1) // m * m
    dpsum = torch.empty(H, padded, dtype=torch.float32, device=q.device)
    lse_log2 = torch.empty(H, padded, dtype=torch.float32, device=q.device)
    dq, dk, dv = torch.zeros_like(q), torch.empty_like(k), torch.empty_like(v)
    dtype = fa4.torch2cute_dtype_map[q.dtype]
    fa4._bwd_preprocess(out, dout, dpsum, lse, lse_log2, None, cu, None, None, dtype, D, D, m, fake_mode=False,
                       hdim_multiple_of=32)
    dq_as_f32 = dq.view(torch.float32).view(total, H * D // 2)
    key = (causal, local, H)
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
    return dq, dk, dv


def fa4_window_left(window_left: int, T: int) -> int | None:
    """None (FA4's causal kernel) when the window spans T, else the local window: a window spanning T is plain causal
    attention, so the cheaper causal kernel computes it; FA4 keeps the local kernel (interface.py
    _resolve_causal_local_window)."""
    return None if window_left >= T else window_left


def segmented(cu_seqlens: Tensor, segments: Tensor) -> Tensor:
    """cu_seqlens' first segments + 1 offsets: the segments that are not padding."""
    return cu_seqlens[:int(segments) + 1]


@torch.library.custom_op("nanochat::flash_attn", mutates_args=(), device_types="cuda")
def flash_attn(q: Tensor, k: Tensor, v: Tensor, window_left: int, cu_seqlens: Tensor,
               segments: Tensor) -> tuple[Tensor, Tensor]:
    """q, k, v (B, T, H, D) bf16 -> (out like q, lse fp32 (H, B*T))."""
    T, cu = q.size(1), segmented(cu_seqlens, segments)
    window = fa4_window_left(window_left, T)
    with contextlib.nullcontext() if window is None else persistent_forward():
        out, lse = _flash_attn_fwd(q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), cu_seqlens_q=cu,
                                   cu_seqlens_k=cu, max_seqlen_q=T, max_seqlen_k=T, causal=True,
                                   window_size_left=window, return_lse=True)[:2]
    return out.view(q.shape), lse


@flash_attn.register_fake
def flash_attn_fake(q: Tensor, k: Tensor, v: Tensor, window_left: int, cu_seqlens: Tensor,
                    segments: Tensor) -> tuple[Tensor, Tensor]:
    B, T, H = q.shape[:3]
    return torch.empty_like(q), q.new_empty((H, B * T), dtype=torch.float32)


@torch.library.custom_op("nanochat::flash_attn_bwd", mutates_args=(), device_types="cuda")
def flash_attn_bwd(dout: Tensor, q: Tensor, k: Tensor, v: Tensor, out: Tensor, lse: Tensor, window_left: int,
                   cu_seqlens: Tensor, segments: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(dq, dk, dv); FA4's backward, local layers' dq accumulated in bf16 (bwd_bf16dq); not bitwise
    reproducible."""
    T, cu = q.size(1), segmented(cu_seqlens, segments)
    window = fa4_window_left(window_left, T)
    args = q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), out.flatten(0, 1), dout.flatten(0, 1), lse
    grads = (_flash_attn_bwd(*args, causal=True, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=T, max_seqlen_k=T)
             if window is None else bwd_bf16dq(*args, cu, T, window))
    return tuple(g.view(q.shape) for g in grads)  # MHA: q, k, v share one shape


@flash_attn_bwd.register_fake
def flash_attn_bwd_fake(dout: Tensor, q: Tensor, k: Tensor, v: Tensor, out: Tensor, lse: Tensor, window_left: int,
                        cu_seqlens: Tensor, segments: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def flash_attn_save(ctx: FunctionCtx, inputs: tuple[Tensor, Tensor, Tensor, int, Tensor, Tensor],
                    output: tuple[Tensor, Tensor]) -> None:
    q, k, v, ctx.window_left, cu_seqlens, segments = inputs
    ctx.save_for_backward(q, k, v, *output, cu_seqlens, segments)


def flash_attn_grad(ctx: FunctionCtx, dout: Tensor,
                    dlse: Tensor) -> tuple[Tensor, Tensor, Tensor, None, None, None]:
    q, k, v, out, lse, cu_seqlens, segments = ctx.saved_tensors
    return *flash_attn_bwd(dout, q, k, v, out, lse, ctx.window_left, cu_seqlens, segments), None, None, None


flash_attn.register_autograd(flash_attn_grad, setup_context=flash_attn_save)
