"""SwiGLU MLP (Shazeer, arXiv 2002.05202): c_proj(silu(c_gate x) * c_fc x), for the MLP ablation against ReLU^2.

Kernels mirror ops/linear.py relu2_mlp on quack.mlp.mlp_func(activation="swiglu")'s (QuACK's autotuned SM90 GEMMs):
silu(gate) * up in the up projection's epilogue (gemm_gated, which stores the pre-activation), its derivative and the
recomputed post-activation in the backward's gemm_dgated. QuACK's gated epilogue reads gate and up from interleaved
columns (gemm_gated_ref: gate = preact[..., ::2], up = preact[..., 1::2]), so each call interleaves the separate c_gate
and c_fc rows; Muon orthogonalizes each matrix on its own."""
import torch
from quack.gemm_interface import gemm, gemm_dgated, gemm_gated
from torch import Tensor, nn


def hidden(n_embd: int) -> int:
    """Width h parameter-matched to the ReLU^2 MLP's two C x 4C matrices: 3 h C = 8 C^2 (Shazeer §2), rounded to the
    nearest multiple of the TMA's 16-byte global stride alignment in bf16 elements (CUDA Driver API
    cuTensorMapEncodeTiled), so the (M, h) post-activation rows stay TMA-addressable."""
    align = 16 // torch.bfloat16.itemsize
    return round(2 * 4 * n_embd / 3 / align) * align


def interleave(w_gate: Tensor, w_fc: Tensor, *, dtype: torch.dtype) -> Tensor:
    """(h, C), (h, C) -> (2h, C) in dtype, rows gate_0, up_0, gate_1, up_1, ..."""
    return torch.stack((w_gate.to(dtype), w_fc.to(dtype)), dim=1).flatten(0, 1)


@torch.library.custom_op("nanochat::swiglu_mlp", mutates_args=(), device_types="cuda")
def swiglu_mlp(x: Tensor, w_gate: Tensor, w_fc: Tensor, w_proj: Tensor) -> tuple[Tensor, Tensor]:
    """x (..., C), w_gate and w_fc (h, C), w_proj (C, h) -> (y (..., C), interleaved pre-activation (M, 2h) saved for
    the backward)."""
    x2 = x.reshape(-1, x.shape[-1])
    preact, postact = gemm_gated(x2, interleave(w_gate, w_fc, dtype=x.dtype).t(), activation="swiglu")
    y = gemm(postact, w_proj.to(x.dtype).t(), out_dtype=x.dtype)
    return y.view(*x.shape[:-1], w_proj.shape[0]), preact


@swiglu_mlp.register_fake
def swiglu_mlp_fake(x, w_gate, w_fc, w_proj):
    return x.new_empty(*x.shape[:-1], w_proj.shape[0]), x.new_empty(x.numel() // x.shape[-1], 2 * w_fc.shape[0])


@torch.library.custom_op("nanochat::swiglu_mlp_bwd", mutates_args=(), device_types="cuda")
def swiglu_mlp_bwd(dy: Tensor, x: Tensor, preact: Tensor, w_gate: Tensor, w_fc: Tensor,
                   w_proj: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(dX like x, dW_gate, dW_fc, dW_proj in the weights' dtype)."""
    C, h = x.shape[-1], w_fc.shape[0]
    dy2, x2 = dy.reshape(-1, C), x.reshape(-1, C)
    dpreact, postact = gemm_dgated(dy2, w_proj.to(x.dtype), preact, activation="swiglu")
    dw_proj = gemm(dy2.t(), postact, out_dtype=w_proj.dtype)
    del postact
    dx = gemm(dpreact, interleave(w_gate, w_fc, dtype=x.dtype), out_dtype=x.dtype).view(x.shape)
    dw_gate, dw_fc = gemm(dpreact.t(), x2, out_dtype=w_fc.dtype).view(h, 2, C).unbind(1)
    return dx, dw_gate.contiguous(), dw_fc.contiguous(), dw_proj


@swiglu_mlp_bwd.register_fake
def swiglu_mlp_bwd_fake(dy, x, preact, w_gate, w_fc, w_proj):
    return x.new_empty(x.shape), w_gate.new_empty(w_gate.shape), w_fc.new_empty(w_fc.shape), w_proj.new_empty(
        w_proj.shape)


def swiglu_mlp_save(ctx, inputs, output):
    ctx.save_for_backward(*inputs, output[1])


def swiglu_mlp_grad(ctx, dy, dpreact):
    x, w_gate, w_fc, w_proj, preact = ctx.saved_tensors
    return swiglu_mlp_bwd(dy, x, preact, w_gate, w_fc, w_proj)


swiglu_mlp.register_autograd(swiglu_mlp_grad, setup_context=swiglu_mlp_save)


class SwiGLU(nn.Module):
    """Drop-in for gpt.MLP: c_proj(silu(c_gate x) * c_fc x), h = hidden(C)."""

    def __init__(self, config):
        super().__init__()
        C = config.n_embd
        h = hidden(C)
        self.c_gate = nn.Linear(C, h, bias=False)
        self.c_fc = nn.Linear(C, h, bias=False)
        self.c_proj = nn.Linear(h, C, bias=False)

    def forward(self, x):
        return swiglu_mlp(x, self.c_gate.weight, self.c_fc.weight, self.c_proj.weight)[0]
