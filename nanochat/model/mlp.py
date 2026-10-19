"""
The block's MLP: SwiGLU on QuACK's gated GEMMs (ops/linear.py swiglu_mlp) and its hidden width.
"""

import torch.nn as nn

from nanochat.model.layers import Linear, norm
from nanochat.ops.linear import swiglu_mlp


def mlp_hidden(config):
    """The SwiGLU hidden width: the parameters and FLOPs of a two-matrix MLP of width 4·n_embd (3·n_embd·hidden =
    8·n_embd²), rounded to whole heads as the width is (dev/D20_SPEC.md "MLP")."""
    head_dim = config.n_embd // config.n_head
    return head_dim * round(8 * config.n_embd / (3 * head_dim))


class MLP(nn.Module):
    """SwiGLU (Shazeer 2020): c_proj(silu(gate) * up) with [gate; up] = c_fc(x), as nanochat::swiglu_mlp on QuACK's gated
    GEMMs (Tri Dao, CuTe DSL; at least as fast as cuBLAS on sm_121): the gated activation in the up-projection GEMM's
    epilogue (from its fp32 accumulators) and its derivative in the backward GEMM's, so it costs no pass over memory;
    the weight gradients in fp32 straight from the accumulators. c_fc holds the gate rows, then the up rows (QuACK's
    concat layout)."""
    def __init__(self, config):
        super().__init__()
        hidden = mlp_hidden(config)
        self.c_fc = Linear(config.n_embd, 2 * hidden, bias=False)
        self.c_proj = Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        """x: the residual stream, normed here."""
        return swiglu_mlp(norm(x), self.c_fc.weight, self.c_proj.weight)[0]
