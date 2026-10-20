"""
The model's QuACK GEMMs (Dao-AILab/quack quack.gemm_interface), one config per (role, width), fixed in TABLE. QuACK's
autotuner (quack/autotuner.py) times each candidate once per new shape, so its pick varies with GPU load, and it offers
neither cuBLAS nor an accuracy criterion for weight gradients. Kernels compile for symbolic M (QUACK_CACHE_DIR).

Roles (C width, h = 4C, V vocabulary, M tokens; ops/linear.py, ops/cross_entropy.py):
  proj_fwd    x W^T                    (M, C) x (C, C)     c_q, c_k, c_v, c_o
  proj_dx     dY W                     (M, C) x (C, C)
  proj_dw     dY^T X -> fp32           (C, M) x (M, C)
  mlp_up      x W_fc^T                 (M, C) x (C, h)     relu^2 epilogue (gemm_act)
  mlp_down    u W_proj^T               (M, h) x (h, C)
  mlp_dact    dY W_proj                (M, C) x (C, h)     relu^2-derivative epilogue (gemm_dact)
  mlp_dw_proj dY^T u -> fp32           (C, M) x (M, h)
  mlp_dx      dpre W_fc                (M, h) x (h, C)
  mlp_dw_fc   dpre^T x -> fp32         (h, M) x (M, C)
  lm_dx       dz W_lm                  (M, V) x (V, C)
  lm_dw       dz^T x -> fp32           (V, M) x (M, C)

TABLE: d8 (256) and d24 (768) widths, measured on H100 SXM5 (sm_90, 700 W cap) at M = config.MICRO_BATCH_TOKENS over
QuACK's 44 SM90 candidates, plus serial split-K 1..16 and cuBLAS ("cublas"; torch.mm, fp32 out) for weight-gradient
roles: benchmarks/gemmsweep.py -> gemmsweep_sm90.jsonl -> benchmarks/gemmtable.py -> gemmtable_sm90.txt. Forward and dX
roles: fastest. Weight-gradient roles: most accurate within the fastest's median absolute deviation (rel. error vs fp64
1.14e-4 / 5.67e-5 / 2.82e-5 / 1.38e-5 / 6.76e-6 at split-K 1 / 2 / 4 / 8 / 16; cuBLAS 1.38e-5 to 5.68e-5), ties broken
by speed. Not dynamic-persistent, as quack.gemm_config._get_sm90_configs. One GemmConfig object per entry because
QuACK's launch cache keys on the object.
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


def gemm(A: Tensor, B: Tensor, *, role: str, width: int, out_dtype: torch.dtype, alpha: float | Tensor) -> Tensor:
    """alpha A (M, K) @ B (K, N) in out_dtype via gemm_tuned.fn at TABLE[width][role]; "cublas" roles are only called
    with alpha 1 (benchmarks/gemmsweep.py)."""
    cfg = TABLE[width][role]
    if cfg == "cublas":
        return torch.mm(A, B, out_dtype=out_dtype)
    out = torch.empty(A.shape[0], B.shape[1], dtype=out_dtype, device=A.device)
    gemm_tuned.fn(A, B, out, alpha=alpha, config=cfg, split_k=cfg.split_k)
    return out
