"""
The model's QuACK GEMMs (Tri Dao, CuTe DSL; Dao-AILab/quack quack.gemm_interface) at one kernel config per (GEMM, model
width), fixed in TABLE; QuACK's run-time autotuner is not used (it keys on M and re-benchmarks every new shape). QuACK
compiles kernels for symbolic shapes and caches them under QUACK_CACHE_DIR, so one width's kernels serve every M.

GEMMs (C the width, h the MLP hidden size, V the padded vocabulary (model/config.py padded_vocab), M tokens; ops/linear.py,
ops/softcap_ce.py):
  proj_fwd    x W^T                    (M, C) x (C, C)     attention projections c_q, c_k, c_v, c_proj
  proj_dx     dY W                     (M, C) x (C, C)
  proj_dw     dY^T X -> fp32           (C, M) x (M, C)     "cublas": torch.mm
  mlp_up      gated: x [gate; up]^T    (M, C) x (C, 2h)    swiglu epilogue (quack gemm_gated)
  mlp_down    u W_proj^T               (M, h) x (h, C)
  mlp_dgated  dY W_proj                (M, C) x (C, h)     dswiglu epilogue (quack gemm_dgated)
  mlp_dw_proj dY^T u -> fp32           (C, M) x (M, h)
  mlp_dx      dpre W_fc                (M, 2h) x (2h, C)
  mlp_dw_fc   dpre^T x -> fp32         (2h, M) x (M, C)
  lm_dx       dz W_lm                  (M, V) x (V, C)
  lm_dw       dz^T x -> fp32           (V, M) x (M, C)

TABLE is a measurement record on GB10 (sm_121): QuACK's SM120 candidates (plus cuBLAS for proj_dw), median times of
interleaved CUDA-graph rounds at M = 65,536 for C = 256..2048 (dev-ignore/autotune/sweep.py, sweep*.jsonl, table.txt).
Forward and dX roles: the fastest. Weight-gradient roles: the most accurate split-K class (rel. error vs fp64 depends
only on the serial split-K factor s: 7.59e-5 / 3.77e-5 / 1.86e-5 / 9.11e-6 / 4.40e-6 at s = 1 / 2 / 4 / 8 / 16) among
the configs no slower than the fastest (dev-ignore/PackageUpgrade/dwsweep.py, dwsweep.jsonl, dwdom.txt).

Code: <tile_m>x<tile_n>, then p (pingpong) or c (cooperative), s (swap_ab), k<n> (serial split-K n: deterministic).
Every SM120 config is cluster 1x1 and dynamic-persistent, as in quack.gemm_config._get_sm120_configs.
"""
import re

import torch
from quack.gemm_config import GemmConfig
from quack.gemm_interface import gemm_tuned
from torch import Tensor

TABLE = {
    256: dict(proj_fwd="64x128p", proj_dx="64x128p", proj_dw="128x64pk16", mlp_up="128x160c", mlp_down="128x64c",
              mlp_dgated="64x128p", mlp_dw_proj="128x64ps", mlp_dx="128x64cs", mlp_dw_fc="64x128cs", lm_dx="128x160c",
              lm_dw="128x192cs"),
    384: dict(proj_fwd="64x128c", proj_dx="64x128p", proj_dw="128x64ck2", mlp_up="128x128p", mlp_down="64x128p",
              mlp_dgated="64x128p", mlp_dw_proj="64x128c", mlp_dx="64x128p", mlp_dw_fc="128x128cs", lm_dx="128x192c",
              lm_dw="128x192cs"),
    512: dict(proj_fwd="128x192c", proj_dx="128x128p", proj_dw="cublas", mlp_up="128x128c", mlp_down="128x192cs",
              mlp_dgated="128x192c", mlp_dw_proj="128x128p", mlp_dx="128x192cs", mlp_dw_fc="128x128ps", lm_dx="128x192cs",
              lm_dw="128x192cs"),
    640: dict(proj_fwd="128x192cs", proj_dx="128x192cs", proj_dw="128x64pk16", mlp_up="128x160p", mlp_down="128x192cs",
              mlp_dgated="64x128p", mlp_dw_proj="128x192c", mlp_dx="128x192cs", mlp_dw_fc="128x192cs", lm_dx="128x192cs",
              lm_dw="128x192cs"),
    768: dict(proj_fwd="128x192cs", proj_dx="128x192cs", proj_dw="128x192csk4", mlp_up="128x128p", mlp_down="128x192cs",
              mlp_dgated="64x128p", mlp_dw_proj="128x128ps", mlp_dx="128x192cs", mlp_dw_fc="128x192c", lm_dx="128x192cs",
              lm_dw="128x192cs"),
    896: dict(proj_fwd="128x192cs", proj_dx="128x192cs", proj_dw="cublas", mlp_up="128x128p", mlp_down="128x192cs",
              mlp_dgated="64x128c", mlp_dw_proj="128x192cs", mlp_dx="128x192cs", mlp_dw_fc="128x192c", lm_dx="128x192cs",
              lm_dw="128x192cs"),
    1024: dict(proj_fwd="128x192cs", proj_dx="128x192cs", proj_dw="cublas", mlp_up="128x128p", mlp_down="128x192cs",
               mlp_dgated="64x128p", mlp_dw_proj="128x160p", mlp_dx="128x192cs", mlp_dw_fc="128x192cs", lm_dx="128x192cs",
               lm_dw="128x192cs"),
    1152: dict(proj_fwd="128x192c", proj_dx="128x192c", proj_dw="cublas", mlp_up="128x128p", mlp_down="128x192c",
               mlp_dgated="64x128p", mlp_dw_proj="128x192cs", mlp_dx="128x192c", mlp_dw_fc="128x192c", lm_dx="128x192cs",
               lm_dw="128x192c"),
    1280: dict(proj_fwd="128x192c", proj_dx="128x192c", proj_dw="128x192csk4", mlp_up="128x128p", mlp_down="128x160c",
               mlp_dgated="64x128p", mlp_dw_proj="128x192cs", mlp_dx="128x192c", mlp_dw_fc="128x192c", lm_dx="128x192cs",
               lm_dw="128x192ck2"),
    1408: dict(proj_fwd="128x192cs", proj_dx="128x192cs", proj_dw="128x192cs", mlp_up="128x128p", mlp_down="128x192c",
               mlp_dgated="64x128p", mlp_dw_proj="128x192cs", mlp_dx="128x192c", mlp_dw_fc="128x192c", lm_dx="128x192cs",
               lm_dw="128x192c"),
    1536: dict(proj_fwd="128x192c", proj_dx="128x192cs", proj_dw="128x192cs", mlp_up="128x128p", mlp_down="128x192c",
               mlp_dgated="64x128p", mlp_dw_proj="128x192cs", mlp_dx="128x192c", mlp_dw_fc="128x192c", lm_dx="128x192cs",
               lm_dw="128x192c"),
    1664: dict(proj_fwd="128x192cs", proj_dx="128x192cs", proj_dw="cublas", mlp_up="128x128p", mlp_down="128x128p",
               mlp_dgated="64x128p", mlp_dw_proj="128x192c", mlp_dx="128x192cs", mlp_dw_fc="128x192cs", lm_dx="128x192cs",
               lm_dw="128x192cs"),
    1792: dict(proj_fwd="128x192cs", proj_dx="128x192cs", proj_dw="128x192cs", mlp_up="128x128p", mlp_down="128x192cs",
               mlp_dgated="64x128p", mlp_dw_proj="128x192c", mlp_dx="128x192cs", mlp_dw_fc="128x192cs", lm_dx="128x192cs",
               lm_dw="128x192cs"),
    1920: dict(proj_fwd="128x192cs", proj_dx="128x192cs", proj_dw="128x160cs", mlp_up="128x128p", mlp_down="128x128p",
               mlp_dgated="64x128p", mlp_dw_proj="128x192cs", mlp_dx="128x192cs", mlp_dw_fc="128x192cs", lm_dx="128x192cs",
               lm_dw="128x192cs"),
    2048: dict(proj_fwd="128x192cs", proj_dx="128x192cs", proj_dw="128x192c", mlp_up="128x128p", mlp_down="128x128p",
               mlp_dgated="64x128p", mlp_dw_proj="128x192c", mlp_dx="128x192cs", mlp_dw_fc="128x192c", lm_dx="128x192cs",
               lm_dw="128x192cs"),
}


def _config(code):
    tile_m, tile_n, kind, swap, split_k = re.fullmatch(r"(\d+)x(\d+)([pc])(s?)(?:k(\d+))?", code).groups()
    return GemmConfig(tile_m=int(tile_m), tile_n=int(tile_n), pingpong=kind == "p", swap_ab=swap == "s",
                      split_k=int(split_k or 1), cluster_m=1, cluster_n=1, is_dynamic_persistent=True,
                      device_capacity=12, use_tma_gather=False)


# One GemmConfig object per entry: QuACK's gated/dgated launch cache keys on the config's id
CONFIGS = {C: {role: code if code == "cublas" else _config(code) for role, code in roles.items()}
           for C, roles in TABLE.items()}


def gemm(A: Tensor, B: Tensor, role: str, C: int, out_dtype=None, alpha=1.0, concat_layout=None) -> Tensor:
    """A (M, K) @ B (K, N) with role's config at width C: quack.gemm_interface.gemm's eager call with the config given
    (the function its autotuner wraps), split-K from the config."""
    cfg = CONFIGS[C][role]
    out = torch.empty(A.shape[0], B.shape[1], dtype=out_dtype or A.dtype, device=A.device)
    gemm_tuned.fn(A, B, out, alpha=alpha, concat_layout=concat_layout, config=cfg, split_k=cfg.split_k)
    return out
