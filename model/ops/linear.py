"""The decoder layer's GEMMs as torch.library ops (traceable by fullgraph torch.compile): forward and dX on QuACK's
(Dao-AILab/quack) GEMMs (quack.gemm_interface, quack.epilogue fused epilogues) with W the bf16 copy of the fp32 weight
(model/modeling.py GemmLinear), at fixed tile configurations (CONFIGS), never autotuned at run time; dW accumulated in
fp32 by cuBLAS into the weight's .grad (aten addmm.dtype_out, beta = 1; NVIDIA/Megatron-LM's gradient-accumulation
fusion, deepseek-ai/DeepGEMM's D = C + A B).

Weight gradients: each op takes its weights' .grad (fp32, allocated and zeroed by train/loop.py) and its backward adds
dW into them in place, so micro-batches accumulate in the GEMM; autograd sees no gradient for those weights.
Forward-only callers pass None.

nanochat::qkv_proj (q_proj, k_proj, v_proj): one GEMM y = x [Wq; Wk; Wv]^T (..., 3C) against the stacked bf16 copy
(model/modeling.py Attention.qkv_bf16), q, k, v its column blocks; backward dX = dY [Wq; Wk; Wv] (one GEMM, K = 3C),
each dW = dY_block^T X on the block's strided view of dY.

The post-attention half of a layer runs on the un-normed stream: the norm is gain-free, so
norm(h) W^T = diag(rstd) (h W^T) exactly, and rstd is a per-row (colvec) scale in the GEMM epilogues; the normed
tensor is never stored.
- residual_rms (o_proj, inside model/ops/flash_attn.py's attention op): h = residual + x W^T and
  rstd = (mean(h^2) + eps)^-1/2 from the same epilogue (quack.gemm_interface.gemm_rms: per-tile squared sums,
  finalized by quack.rms_final_reduce). Backward (residual_rms_bwd): dO = dh W with FA4's dpsum = rowsum(dO O) per
  head in the epilogue (dx_dpsum, x = O); the column tile is one head (DPSUM_CONFIG), so the per-tile reduce partials
  are dpsum per (row, head).
- nanochat::relu2_mlp: h + down_proj(relu(rstd up_proj h)^2), the MLP of karpathy/nanochat 92d63d4 gpt.py:131-141 on
  the normed stream plus its residual: rstd and relu^2 in the up-projection's epilogue
  (quack.gemm_interface.gemm_norm_act, storing the scaled pre-activation and the activation for the backward; dW_down's
  operand is the saved activation), the residual as the down-projection's C (quack.gemm_interface.gemm_add). Its
  backward owns the norm's backward (residual_rms's rstd gets no gradient of its own): with P = h W_up^T,
  preact = rstd P and dpreact from relu^2's derivative, dP = rstd dpreact, d rstd = sum(dpreact preact) / rstd and
  d rstd/dh = -rstd^3 h / C, so dh = dy + dP W_up - (rstd^2 sum(dpreact preact) / C) h; the row sum is a reduce of the
  dact epilogue, the rest the dX GEMM's epilogue (C = dy, h a tile load)."""
import torch
from cutlass import BFloat16, Float32
from quack.activation import drelu_sq
from quack.cute_dsl_utils import get_device_capacity
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.library import sq_reduce_mod
from quack.epilogue.ops import ColVecLoad, ColVecReduce
from quack.gemm_config import GemmConfig
from quack.gemm_interface import gemm_norm_act, gemm_tuned
from quack.rms_final_reduce import rms_final_reduce
from torch import Tensor
from torch.autograd.function import FunctionCtx

from model.config import HEAD_DIM

# model/modeling.py norm's epsilon
EPS = torch.finfo(torch.float32).eps


def sm90(tile_m: int, tile_n: int, cluster_m: int, cluster_n: int, *, swap_ab: bool = False,
         pingpong: bool = True) -> GemmConfig:
    """An SM90 config of QuACK's autotuning space (quack/gemm_config.py _get_sm90_configs)."""
    return GemmConfig(tile_m=tile_m, tile_n=tile_n, pingpong=pingpong, is_dynamic_persistent=False,
                      cluster_m=cluster_m, cluster_n=cluster_n, swap_ab=swap_ab, device_capacity=9)


# Per call, the hidden width C -> config: the d8, d16 and d24 widths (model/modeling.py model_config), M = 98,304
# tokens (corpus/mixture.py MICRO_BATCH_TOKENS). Each is the lowest mean over three runs of QuACK's autotuning
# benchmark (quack/autotuner.py: L2-cold CUDA-graph medians of every candidate, GH200 at 700 W, exclusive). Other
# widths and GPUs: QuACK's default_config.
CONFIGS: dict[str, dict[int, GemmConfig]] = {
    "qkv": {256: sm90(128, 192, 2, 1, swap_ab=True), 512: sm90(128, 192, 2, 1, swap_ab=True),
            768: sm90(128, 192, 1, 2)},
    "qkv_dx": {256: sm90(128, 192, 2, 1, swap_ab=True), 512: sm90(192, 128, 1, 2),
               768: sm90(128, 160, 1, 2, swap_ab=True)},
    "o": {256: sm90(128, 128, 1, 2), 512: sm90(192, 128, 1, 2), 768: sm90(128, 192, 1, 2)},
    "up": {256: sm90(128, 192, 2, 1, swap_ab=True), 512: sm90(128, 192, 2, 1, swap_ab=True),
           768: sm90(128, 192, 1, 2)},
    "down": {256: sm90(128, 192, 2, 1, swap_ab=True), 512: sm90(192, 128, 1, 2), 768: sm90(128, 192, 1, 2)},
    "down_dx": {256: sm90(128, 256, 2, 1, pingpong=False), 512: sm90(192, 128, 1, 2), 768: sm90(128, 192, 2, 1)},
    "up_dx": {256: sm90(128, 128, 1, 2), 512: sm90(192, 128, 1, 2), 768: sm90(128, 192, 1, 2)},
}


def config(call: str, C: int, x: Tensor) -> GemmConfig | None:
    """CONFIGS' entry for call at width C on x's GPU; None (QuACK's default_config, with tuned=False) where there is
    none."""
    return CONFIGS[call].get(C) if get_device_capacity(x)[0] == 9 else None


def gemm(A: Tensor, B: Tensor, cfg: GemmConfig | None, C: Tensor | None = None) -> Tensor:
    """A B (+ C) in A's dtype at configuration cfg: QuACK's gemm_tuned body, which gemm and gemm_add run with
    config=None for tuned=False (quack/gemm_interface.py); neither takes a config."""
    out = A.new_empty(A.shape[0], B.shape[1])
    gemm_tuned.fn(A, B, out, C=C, config=cfg)
    return out


@torch.library.custom_op("nanochat::qkv_proj", mutates_args=(), device_types="cuda")
def qkv_proj(x: Tensor, weight: Tensor, gq: Tensor | None, gk: Tensor | None, gv: Tensor | None) -> Tensor:
    """x (..., C) bf16, weight (3C, C) the stacked bf16 [Wq; Wk; Wv], gq, gk, gv their fp32 .grad or None (forward
    only) -> (..., 3C) bf16: q, k, v the column blocks [0, C), [C, 2C), [2C, 3C)."""
    x2 = x.reshape(-1, x.shape[-1])
    y = gemm(x2, weight.t(), config("qkv", x.shape[-1], x))
    return y.view(*x.shape[:-1], weight.shape[0])


@qkv_proj.register_fake
def qkv_proj_fake(x: Tensor, weight: Tensor, gq: Tensor | None, gk: Tensor | None, gv: Tensor | None) -> Tensor:
    return x.new_empty(*x.shape[:-1], weight.shape[0])


@torch.library.custom_op("nanochat::qkv_proj_bwd", mutates_args=("gq", "gk", "gv"), device_types="cuda")
def qkv_proj_bwd(dy: Tensor, x: Tensor, weight: Tensor, gq: Tensor, gk: Tensor, gv: Tensor) -> Tensor:
    """dX like x; each block's dW accumulated into its grad."""
    dy2, x2 = dy.reshape(-1, dy.shape[-1]), x.reshape(-1, x.shape[-1])
    dx = gemm(dy2, weight, config("qkv_dx", x.shape[-1], x)).view(x.shape)
    C = gq.shape[0]
    for i, grad in enumerate((gq, gk, gv)):
        torch.addmm(grad, dy2[:, i * C:(i + 1) * C].t(), x2, out_dtype=grad.dtype, out=grad)
    return dx


@qkv_proj_bwd.register_fake
def qkv_proj_bwd_fake(dy: Tensor, x: Tensor, weight: Tensor, gq: Tensor, gk: Tensor, gv: Tensor) -> Tensor:
    return x.new_empty(x.shape)


def qkv_proj_save(ctx: FunctionCtx, inputs: tuple[Tensor, ...], output: Tensor) -> None:
    ctx.save_for_backward(*inputs)


def qkv_proj_grad(ctx: FunctionCtx, dy: Tensor) -> tuple[Tensor, None, None, None, None]:
    x, weight, gq, gk, gv = ctx.saved_tensors
    return qkv_proj_bwd(dy, x, weight, gq, gk, gv), None, None, None, None


qkv_proj.register_autograd(qkv_proj_grad, setup_context=qkv_proj_save)


def residual_rms(x: Tensor, weight: Tensor, residual: Tensor) -> tuple[Tensor, Tensor]:
    """x (M, K), weight (N, K), residual (M, N) -> (h = residual + x W^T in x's dtype, rstd (M,) fp32 of h's rows
    before rounding)."""
    N = weight.shape[0]
    r = sq_reduce_mod(has_c=True, has_rowvec=False, has_aux=False)(
        x, weight.t(), residual, config=config("o", N, x), tuned=False)
    # gemm_rms's epilogue and finalize (quack/gemm_interface.py _gemm_rms_call, _rms_finalize)
    rstd = rms_final_reduce(r["mColVecReduce"], scale=1 / N, eps=EPS)
    return r["D"], rstd


# o_proj's dX tiling: tile_n = HEAD_DIM, so dx_dpsum's per-tile row sums are FA4's dpsum; fixed, as QuACK's tuner does
# not pin tile_n: the fastest QuACK SM90 tile_n = 128 config on the d24 shape (98,304 x 768 x 768)
DPSUM_CONFIG = sm90(128, HEAD_DIM, 1, 2)


@gemm_epilogue(reduces={"dpsum": ColVecReduce("dpsum", scaled=True)})
def dx_dpsum(acc, c):
    """acc = dh W (dO), c = O, FA4's output: D = dO, dpsum = the row sums per column tile of bf16(dO) O (bf16(dO):
    the dO FA4's backward reads)."""
    d = acc.to(BFloat16).to(Float32)
    return {"D": d, "dpsum": (d, c)}


def residual_rms_bwd(dh: Tensor, x: Tensor, weight: Tensor, grad: Tensor) -> tuple[Tensor, Tensor]:
    """dh (M, N); x (M, K) FA4's output (heads of HEAD_DIM); weight (N, K); grad its fp32 .grad -> (dx = dh W bf16,
    dpsum (M, K / HEAD_DIM) fp32: the sum over each head's dims of dx x, FA4's backward dpsum); dW accumulated into
    grad."""
    dpsum = dh.new_empty((dh.shape[0], x.shape[1] // HEAD_DIM), dtype=torch.float32)
    dx = dx_dpsum(dh, weight, x, dpsum=dpsum, config=DPSUM_CONFIG)["D"]
    torch.addmm(grad, dh.t(), x, out_dtype=grad.dtype, out=grad)
    return dx, dpsum


@gemm_epilogue(ops={"rstd": ColVecLoad("rstd")}, reduces={"dsum": ColVecReduce("dsum", scaled=True)})
def relu2_dact_rstd(acc, c, rstd):
    """acc = dY W_down (d postact), c = the stored preact: D = dP = rstd dpreact, dsum = sum over the row of
    dpreact preact."""
    dpreact, _ = drelu_sq(c, acc)
    return {"D": dpreact * rstd, "dsum": (dpreact, c)}


@gemm_epilogue(ops={"corr": ColVecLoad("corr")})
def norm_bwd_apply(acc, c, h, corr):
    """acc = dP W_up, c = dy (the residual's gradient), h the stream, corr = -rstd^2 dsum / C: D = dh."""
    return {"D": acc + c + h * corr}


@torch.library.custom_op("nanochat::relu2_mlp", mutates_args=(), device_types="cuda")
def relu2_mlp(h: Tensor, rstd: Tensor, w_up: Tensor, w_down: Tensor, g_up: Tensor | None,
              g_down: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
    """h (..., C) the un-normed stream, rstd (M,) fp32 its rows' (residual_rms), w_up (4C, C), w_down (C, 4C), g_up,
    g_down their fp32 .grad or None -> (h + MLP(norm(h)) like h, scaled pre-activation and activation (M, 4C) saved
    for the backward)."""
    h2, C = h.reshape(-1, h.shape[-1]), h.shape[-1]
    preact, postact = gemm_norm_act(h2, w_up.t(), rstd=rstd, activation="relu_sq", store_preact=True,
                                    config=config("up", C, h), tuned=False)
    y = gemm(postact, w_down.t(), config("down", C, h), C=h2)
    return y.view(h.shape), preact, postact


@relu2_mlp.register_fake
def relu2_mlp_fake(h: Tensor, rstd: Tensor, w_up: Tensor, w_down: Tensor, g_up: Tensor | None,
                   g_down: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
    act = h.new_empty(h.numel() // h.shape[-1], w_up.shape[0])
    return h.new_empty(h.shape), act, torch.empty_like(act)


@torch.library.custom_op("nanochat::relu2_mlp_bwd", mutates_args=("g_up", "g_down"), device_types="cuda")
def relu2_mlp_bwd(dy: Tensor, h: Tensor, rstd: Tensor, preact: Tensor, postact: Tensor, w_up: Tensor, w_down: Tensor,
                  g_up: Tensor, g_down: Tensor) -> Tensor:
    """dh like h, through the residual, the MLP and the norm; dW_up, dW_down accumulated into g_up, g_down."""
    dy2, h2, C = dy.reshape(-1, dy.shape[-1]), h.reshape(-1, h.shape[-1]), h.shape[-1]
    r = relu2_dact_rstd(dy2, w_down, preact, rstd=rstd, dynamic_scheduler=True, config=config("down_dx", C, h),
                        tuned=False)
    dp = r["D"]
    torch.addmm(g_down, dy2.t(), postact, out_dtype=g_down.dtype, out=g_down)
    corr = rstd.square().mul_(r["dsum"]).mul_(-1 / C)
    dh = norm_bwd_apply(dp, w_up, dy2, h=h2, corr=corr, config=config("up_dx", C, h), tuned=False)["D"].view(h.shape)
    torch.addmm(g_up, dp.t(), h2, out_dtype=g_up.dtype, out=g_up)
    return dh


@relu2_mlp_bwd.register_fake
def relu2_mlp_bwd_fake(dy: Tensor, h: Tensor, rstd: Tensor, preact: Tensor, postact: Tensor, w_up: Tensor,
                       w_down: Tensor, g_up: Tensor, g_down: Tensor) -> Tensor:
    return h.new_empty(h.shape)


def relu2_mlp_save(ctx: FunctionCtx, inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None],
                   output: tuple[Tensor, Tensor, Tensor]) -> None:
    ctx.save_for_backward(*inputs, *output[1:])


def relu2_mlp_grad(ctx: FunctionCtx, dy: Tensor, dpreact: Tensor,
                   dpostact: Tensor) -> tuple[Tensor, None, None, None, None, None]:
    h, rstd, w_up, w_down, g_up, g_down, preact, postact = ctx.saved_tensors
    return relu2_mlp_bwd(dy, h, rstd, preact, postact, w_up, w_down, g_up, g_down), None, None, None, None, None


relu2_mlp.register_autograd(relu2_mlp_grad, setup_context=relu2_mlp_save)
