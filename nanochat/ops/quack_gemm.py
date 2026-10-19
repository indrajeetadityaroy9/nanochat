"""
The model's QuACK GEMMs (Tri Dao, CuTe DSL; Dao-AILab/quack quack.gemm_interface) at one kernel config per (GEMM, model
width) on the H100, fixed in TABLE; QuACK's run-time autotuner is not used (it keys on M and re-benchmarks every new
shape). QuACK compiles kernels for symbolic shapes and caches them under QUACK_CACHE_DIR, so one width's kernels serve
every M.

GEMMs (C the width, h = 4C the MLP hidden size, V the vocabulary, M tokens; ops/linear.py, ops/cross_entropy.py):
  proj_fwd    x W^T                    (M, C) x (C, C)     attention projections c_q, c_k, c_v, c_o
  proj_dx     dY W                     (M, C) x (C, C)
  proj_dw     dY^T X -> fp32           (C, M) x (M, C)
  mlp_up      x W_fc^T                 (M, C) x (C, h)     relu^2 epilogue (quack gemm_act)
  mlp_down    u W_proj^T               (M, h) x (h, C)
  mlp_dact    dY W_proj                (M, C) x (C, h)     relu^2-derivative epilogue (quack gemm_dact)
  mlp_dw_proj dY^T u -> fp32           (C, M) x (M, h)
  mlp_dx      dpre W_fc                (M, h) x (h, C)
  mlp_dw_fc   dpre^T x -> fp32         (h, M) x (M, C)
  lm_dx       dz W_lm                  (M, V) x (V, C)
  lm_dw       dz^T x -> fp32           (V, M) x (M, C)

TABLE holds the widths of the d8 smoke model (256) and the d24 production model (768). It is a measurement record on the
H100 80 GB SXM5 (sm_90, at its 700 W power cap): median times of interleaved CUDA-graph rounds at
M = config.MICRO_BATCH_TOKENS (98,304) of QuACK's SM90 candidates (all 44: tiles x pingpong x cluster 1x2 / 2x1 x
swap_ab), the weight-gradient roles also at serial split-K 1..16 and cuBLAS (torch.mm, fp32 out; "cublas"):
benchmarks/gemmsweep.py (gemmsweep_sm90.jsonl; benchmarks/gemmtable.py -> gemmtable_sm90.txt).
Forward and dX roles: the fastest. Weight-gradient roles: among the configs within 1% of the fastest, the most accurate
(rel. error vs fp64 depends only on the split-K factor s: 1.14e-4 / 5.67e-5 / 2.82e-5 / 1.38e-5 / 6.76e-6 at
s = 1 / 2 / 4 / 8 / 16; cuBLAS 1.38e-5 to 5.68e-5), then the fastest.

Each config: the tile (tile_m x tile_n), pingpong or cooperative, swap_ab, the thread-block cluster, serial split-K
(deterministic; 1 when not given); not dynamic-persistent, as QuACK's SM90 candidates (quack.gemm_config
._get_sm90_configs). One GemmConfig object per entry: QuACK's launch cache keys on the config object.
"""
from functools import partial

import torch
from quack.gemm_config import GemmConfig
from quack.gemm_interface import gemm_tuned
from torch import Tensor

SM90 = partial(GemmConfig, is_dynamic_persistent=False, device_capacity=9)
TABLE = {
    256: dict(
        proj_fwd=SM90(tile_m=128, tile_n=128, pingpong=True, swap_ab=False, cluster_m=1, cluster_n=2),
        proj_dx=SM90(tile_m=192, tile_n=128, pingpong=True, swap_ab=False, cluster_m=1, cluster_n=2),
        proj_dw="cublas",
        mlp_up=SM90(tile_m=128, tile_n=256, pingpong=False, swap_ab=True, cluster_m=1, cluster_n=2),
        mlp_down=SM90(tile_m=128, tile_n=192, pingpong=True, swap_ab=True, cluster_m=2, cluster_n=1),
        mlp_dact=SM90(tile_m=128, tile_n=256, pingpong=False, swap_ab=True, cluster_m=1, cluster_n=2),
        mlp_dw_proj="cublas",
        mlp_dx=SM90(tile_m=192, tile_n=128, pingpong=True, swap_ab=False, cluster_m=1, cluster_n=2),
        mlp_dw_fc="cublas",
        lm_dx=SM90(tile_m=128, tile_n=192, pingpong=True, swap_ab=True, cluster_m=2, cluster_n=1),
        lm_dw=SM90(tile_m=256, tile_n=128, pingpong=False, swap_ab=False, cluster_m=1, cluster_n=2)),
    768: dict(
        proj_fwd=SM90(tile_m=128, tile_n=208, pingpong=True, swap_ab=True, cluster_m=2, cluster_n=1),
        proj_dx=SM90(tile_m=128, tile_n=208, pingpong=True, swap_ab=True, cluster_m=2, cluster_n=1),
        proj_dw="cublas",
        mlp_up=SM90(tile_m=192, tile_n=128, pingpong=True, swap_ab=True, cluster_m=2, cluster_n=1),
        mlp_down=SM90(tile_m=256, tile_n=160, pingpong=False, swap_ab=True, cluster_m=1, cluster_n=2),
        mlp_dact=SM90(tile_m=192, tile_n=128, pingpong=True, swap_ab=True, cluster_m=2, cluster_n=1),
        mlp_dw_proj="cublas",
        mlp_dx=SM90(tile_m=128, tile_n=208, pingpong=True, swap_ab=True, cluster_m=1, cluster_n=2),
        mlp_dw_fc="cublas",
        lm_dx=SM90(tile_m=256, tile_n=160, pingpong=False, swap_ab=True, cluster_m=1, cluster_n=2),
        lm_dw=SM90(tile_m=256, tile_n=192, pingpong=False, swap_ab=False, cluster_m=1, cluster_n=2, split_k=2)),
}


def gemm(A: Tensor, B: Tensor, role: str, C: int, out_dtype: torch.dtype, alpha) -> Tensor:
    """alpha A (M, K) @ B (K, N) in out_dtype with role's config at width C: QuACK's gemm_tuned.fn with the config given
    (the function its autotuner wraps), split-K from the config; "cublas": torch.mm, alpha unused, as every role
    mapped to it is a weight gradient of ops/linear.py called with alpha 1 (proj_dw, mlp_dw_proj, mlp_dw_fc)."""
    cfg = TABLE[C][role]
    if cfg == "cublas":
        return torch.mm(A, B, out_dtype=out_dtype)
    out = torch.empty(A.shape[0], B.shape[1], dtype=out_dtype, device=A.device)
    gemm_tuned.fn(A, B, out, alpha=alpha, config=cfg, split_k=cfg.split_k)
    return out
