"""
The model's elementary layers: the parameter-free RMS norm, the fp32-master embedding and linear layers whose forward
runs in the bf16 activation dtype, and the attention projections on QuACK's GEMMs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.ops.linear import linear


def norm(x):
    """RMS norm over the last dim: F.rms_norm on bf16 computes its statistics in fp32 and rounds the output once. The
    one epsilon convention of every RMS norm (here and ops/qk_prologue.cu): F.rms_norm's default, the fp32 machine
    epsilon FLT_EPSILON = 2^-23 also for bf16 inputs (torch.finfo of the fp32 opmath type; modded-nanogpt
    track_1_short perf/kernels/qkv_rope.py:8)."""
    return F.rms_norm(x, (x.size(-1),))


class Embedding(nn.Embedding):
    """nn.Embedding with an fp32 table (the optimizer's master) whose lookups are rounded once to bf16, the dtype of every
    activation."""
    def forward(self, idx):
        return super().forward(idx).to(torch.bfloat16)


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (bf16 from the embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


class ProjLinear(Linear):
    """The attention projections through nanochat::linear (ops/linear.py): QuACK GEMMs, dW in fp32 straight from the
    accumulators."""
    def forward(self, x):
        return linear(x, self.weight)
