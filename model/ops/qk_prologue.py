"""
q/k prologue of model/modeling.py's attention: per-head RMS norm with F.rms_norm's fp32 epsilon, fp32 per-dim norm
weight (Qwen3, arXiv 2505.09388 §2), rotary by -theta over the full head (karpathy/nanochat 92d63d4 gpt.py:57-65); fp32
math on the bf16 projection, one bf16 rounding per output.

Forward: in the epilogue of the fused q/k/v projection (model/ops/linear.py qkv_proj; TileKernels' statistics and norms
on the GEMM output, deepseek-ai/TileKernels mhc/norm_fn_cuda.py), a QuACK fn epilogue (quack.epilogue.frontend) on
QuACK's SM90 GEMM. The GEMM stores the bf16 projection q | k | v, which the backward reads, and, for the q and k column
tiles, the normed and rotated q | k, which FlashAttention reads: the separate prologue pass that read q, k back and
wrote them again (2 of the 6 (M, C) bf16 tensors of traffic per call) is gone. Everything is register-local in the
WGMMA accumulator fragment, whose tiles are head-aligned (tile N a multiple of HEAD_DIM): the per-head square sum is
QuACK's HeadRstd prepass statistic (each thread's columns of the head, then the lane quad's butterfly), and RoPE's
partner of column c, c + HEAD_DIM / 2 in a head's first half and c - HEAD_DIM / 2 in its second, sits in the same thread
(HeadHalfValues). Per element, as model/kernels/qk_row.cuh: x the bf16 projection, n = (x rstd) g, and y = fma(n, cos,
(x' rstd) g' sin') with x', g' the partner's value and gain and sin' = sin in the first half, -sin in the second
(y1 = n1 cos + n2 sin, y2 = n2 cos - n1 sin; the second half's sign rides g'). Rows are positions: the GEMM is batched
over the batch dimension (M = T per batch), so row m reads the rotary tables' row m.

Backward (model/kernels/qk_prologue.cu k_backward), recomputing n from the projection: dm = R^T dy, dg = sum over rows
of dm * n, dn = dm * g, dx = rstd * (dn - n * mean(dn * n)). q, k are the (B, T, H, HEAD_DIM) bf16 q and k blocks of the
fused (B, T, 3C) projection (heads contiguous, token stride 3C), dq, dk the same blocks of its gradient; norm weights
(HEAD_DIM,) fp32; cos/sin (T_table >= T, HEAD_DIM / 2) fp32.
"""

import ctypes

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, const_expr
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.head_rmsnorm import HeadRstd
from quack.epilogue.ops import EpiOp, RowVecLoad, TileStore
from quack.epilogue.rotary import rotary_cos_sin_load
from quack.gemm_config import GemmConfig
from torch import Tensor

from model.config import HEAD_DIM
from native import cuda_library, pointer, stream

# F.rms_norm's epsilon for fp32 (model/modeling.py norm)
EPS = torch.finfo(torch.float32).eps


class HeadHalfValues(EpiOp):
    """The prepass fn's value of each accumulator element, served to the main fn from registers: at the element's own
    column, or (partner=True) at its rotary partner's, HEAD_DIM / 2 columns further in a head's first half and back in
    its second. gemm_base.epilogue unrolls its subtile loops (range_constexpr), so every subtile's values are registers
    at static indices (as GroupedColStatsBase keeps its statistics), and a thread holds the same rows and in-subtile
    columns in every subtile (GemmSm90.epi_retile_acc): the partner of fragment element i is element i of the subtile
    HEAD_DIM / 2 columns away, whose subtile index along N is computed at trace time. Host argument: HEAD_DIM / 2, a
    constexpr; the tile N must hold whole heads."""

    fn_port = "value"

    def __init__(self, name: str, partner: bool):
        super().__init__(name)
        self.partner = partner

    def config_key(self):
        return (self.partner,)

    def host_arg_key(self, value):
        return ("width", value)

    def host_fake_arg(self, key, fctx):
        return key[1]

    def host_call_arg(self, value, key):
        return None

    def arg_spec_type(self, const=False):
        return cutlass.Constexpr[int]

    def host_arg_form(self, value):
        return "Const"

    def host_validate(self, value, *, n, tile_N, **_):
        if n % (2 * value) or tile_N % (2 * value):
            raise ValueError(f"'{self.name}': head width {2 * value} must divide N={n} and tile_N={tile_N}")

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        return {self.name: getattr(args, self.name)}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        tiles = cute.zipped_divide(cute.make_layout((ctx.tile_M, ctx.tile_N)), ctx.epi_tile).shape
        epi_n = const_expr(cute.size(tiles[0][1]))
        assert param % epi_n == 0, "HEAD_DIM / 2 must be a multiple of the epilogue subtile's N"
        values = tuple(
            tuple(cute.make_rmem_tensor(ctx.tRS_rD_layout.shape, Float32) for _ in range(cute.size(tiles[1][1])))
            for _ in range(cute.size(tiles[1][0])))
        return values, param // epi_n

    def begin_loop(self, gemm, state, epi_coord):
        """(this subtile's slot, filled by the prepass; the slot the main fn reads)."""
        values, shift = state
        m, n = epi_coord
        read = n + shift if (n // shift) % 2 == 0 else n - shift
        return values[m][n], values[m][read if self.partner else n]

    @cute.jit
    def fn_sink_flush(self, gemm, state, frag):
        state[0].store(frag.load())

    def fn_prepare(self, gemm, state, paired):
        return state[1]


class QKStore(TileStore):
    """The q and k column blocks' output (l, m, 2C) of the (l, m, 3C) GEMM. The frontend checks an aux output's shape
    against the GEMM's (l, m, n), so the host passes a (l, m, 3C) view with the buffer's row stride 2C (qk_forward);
    the kernel's TMA tensor has the buffer's real extent 2C (the first two of the three column blocks), and the v tiles
    skip the store."""

    def to_params(self, gemm, args):
        view = getattr(args, self.name)  # kernel order (m, n, l)
        qk = cute.make_tensor(view.iterator, cute.make_layout(
            (view.shape[0], view.shape[1] // 3 * 2, view.shape[2]), stride=view.stride))
        return super().to_params(gemm, args._replace(**{self.name: qk}))

    def store_setup(self, gemm, params, smem_tensor, tiled_copy_r2s, tiled_copy_t2r, tile_coord_mnkl, varlen_manager,
                    tidx):
        copy_r2s, smem, copy_out, _ = super().store_setup(gemm, params, smem_tensor, tiled_copy_r2s, tiled_copy_t2r,
                                                          tile_coord_mnkl, varlen_manager, tidx)
        qk_cols = cute.size(getattr(params, self.name), mode=[1])
        return copy_r2s, smem, copy_out, tile_coord_mnkl[1] * gemm.cta_tile_shape_mnk[1] < qk_cols


def qk_prologue_prepass(acc):
    """The bf16 projection (what the GEMM stores and the backward reads): its squares for the per-head rstd, its values
    for the main fn."""
    x = acc.to(BFloat16).to(Float32)
    return {"rstd": x * x, "x": x, "x_partner": x}


# The rotary tables load gmem -> registers per subtile (rotary_cos_sin_load tma=False): two TMA-staged table loads in
# one epilogue deadlock the GEMM (measured: one TMA table runs, two hang in the first launch; two LDG tables run)
@gemm_epilogue(
    outputs=(QKStore("qk"),),
    ops={"rstd": HeadRstd("rstd", eps=EPS), "x": HeadHalfValues("x", partner=False),
         "x_partner": HeadHalfValues("x_partner", partner=True), "gain": RowVecLoad("gain"),
         "gain_partner": RowVecLoad("gain_partner"), "cos": rotary_cos_sin_load("cos", tma=False),
         "sin": rotary_cos_sin_load("sin", tma=False)},
    prepass=qk_prologue_prepass,
    prepass_outs=("rstd", "x", "x_partner"),
)
def qk_prologue_epi(acc, rstd, x, x_partner, gain, gain_partner, cos, sin):
    """D = the bf16 projection; qk = its normed and rotated q | k (module docstring); the fma and product order are
    k_forward's (model/kernels/qk_row.cuh qk_rotate as nvcc compiles it: y1 = fma(n1, cos, n2 sin))."""
    return {"D": x, "qk": cute.math.fma(x * rstd * gain, cos, x_partner * rstd * gain_partner * sin)}


def qk_gains(q_norm_weight: Tensor, k_norm_weight: Tensor, heads: int) -> Tensor:
    """(2, 3C) fp32 rows of qk_prologue_epi: the norm weight at each q and k column, and its rotary partner's with the
    second half's sign (module docstring); zeros at the v columns, whose qk output is not stored."""
    half = HEAD_DIM // 2
    gains = torch.stack((q_norm_weight, k_norm_weight))  # (2, HEAD_DIM)
    partner = torch.cat((gains[:, half:], -gains[:, :half]), dim=1)
    rows = torch.stack((gains, partner), dim=0).repeat(1, 1, heads)  # (gain | partner, q | k, C)
    return torch.cat((rows.flatten(1), rows.new_zeros(2, heads * HEAD_DIM)), dim=1)


def qk_forward(x: Tensor, weight: Tensor, gains: Tensor, cos: Tensor, sin: Tensor,
               config: GemmConfig) -> tuple[Tensor, Tensor]:
    """x (B, T, C) bf16, weight (3C, C) bf16, gains qk_gains' (2, 3C), cos/sin (T_table >= T, HEAD_DIM / 2) fp32 ->
    (q | k | v (B, T, 3C), normed and rotated q | k (B, T, 2C)), both bf16 and contiguous."""
    B, T, _ = x.shape
    N = weight.shape[0]
    QK = N // 3 * 2
    qkv = x.new_empty(B, T, N)
    # QKStore's (B, T, N) view of qk keeps qk's row stride; the view's last row ends N - QK elements past qk
    buffer = x.new_empty(B * T * QK + N - QK)
    qk_prologue_epi(x, weight.t().expand(B, -1, -1), out={"D": qkv, "qk": buffer.as_strided((B, T, N), (T * QK, QK, 1))},
                    rstd=HEAD_DIM, x=HEAD_DIM // 2, x_partner=HEAD_DIM // 2, gain=gains[0], gain_partner=gains[1],
                    cos=cos, sin=sin, config=config)
    return qkv, buffer[:B * T * QK].view(B, T, QK)


def qk_backward(q: Tensor, k: Tensor, q_norm_weight: Tensor, k_norm_weight: Tensor, dq_out: Tensor, dk_out: Tensor,
                cos: Tensor, sin: Tensor, dq: Tensor, dk: Tensor) -> tuple[Tensor, Tensor]:
    """Writes q's and k's gradients into dq, dk from the outputs' gradients dq_out, dk_out (contiguous); returns the
    norm weights' gradients, summed in a fixed order (bitwise reproducible)."""
    B, T, H, D = q.shape
    dq_norm_weight, dk_norm_weight = torch.empty_like(q_norm_weight), torch.empty_like(k_norm_weight)
    lib = cuda_library("qk_prologue")
    part = q_norm_weight.new_empty((2 * lib.qkp_backward_blocks(), D))  # per-block partial weight gradients
    lib.qkp_backward(
        pointer(q), pointer(k), pointer(q_norm_weight), pointer(k_norm_weight), pointer(dq_out), pointer(dk_out),
        pointer(dq), pointer(dk), pointer(dq_norm_weight), pointer(dk_norm_weight), pointer(part), pointer(cos),
        pointer(sin), ctypes.c_longlong(B * T * H), H, T, stream())
    return dq_norm_weight, dk_norm_weight
