"""
MUDD: multiway dynamic dense connections ([MUDD]: Xiao et al., ICML 2025, arXiv 2502.12170), at modded-nanogpt's two sites
(track_1_short model/gpt.py:368-420 init_mudd / forward_mudd, :676-681 last layer, :758-772 post-loop; record #81).

A site mixes earlier residual states (its sources) into the current state X with per-token coefficients from a small
MLP on X ([MUDD] Eq. 6; hidden width MUDD_DIM, modded model/gpt.py:388):
    c = (gelu(rms_norm(X) W1^T) W2^T + b) * MUDD_SCALE
W1 (MUDD_DIM, C) kaiming-uniform (a = sqrt 5, modded :393), W2 (K, MUDD_DIM) zero, so c = b * MUDD_SCALE at init: each stream's
static prior. The output scale and the bias pre-divided by it are modded's parametrization (:387, :397-410), the one
its Adam recipe (lr_mul, betas, decays: model/training.py:73-76) belongs to; the optimizer groups are train/optim.py's.
model/gpt.py's sites ([MUDD]'s coefficient MLP at modded's two sites; dev/D20_SPEC.md "Modded-nanogpt arms (d10)", A1):
- site 1, the input X of the last block, K = 2|S1|: a value stream added to the last attention's values (modded's
  v_mudd, the paper's V stream) and a residual stream, x = X + sum c_s h_s (the paper's R stream; the identity on X is
  fixed as modded's 1 + mu5), both priors 0;
- site 2, before the final norm, K = |S2|: x = X + sum c_s h_s with prior -0.2 on the output of block L // 2 (nanochat's
  backout at its init, 92d63d4 gpt.py:188, 243-246, 497).
The mixes are fp32 sums of the bf16 sources (source order ascending), rounded once to bf16 by the caller.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.model.layers import norm
from nanochat.ops.linear import narrow_linear

MUDD_SCALE = 0.1   # modded model/gpt.py:387 _mudd_scale: coefficients = (MLP output + bias) * MUDD_SCALE
MUDD_DIM = 64      # modded model/gpt.py:388 mudd_dim: the coefficient MLP's hidden width


def site_sources(n_layer, ve_layers):
    """The sources of site 1 and site 2, as keys: "x0" (the smeared normed embedding of the lambda mix), ("h", i) the
    residual after block i, ("ve", l) the value embedding of layer l. Modded's source set (record #81; model/gpt.py:69
    CACHE_LAYERS = (3, 7) and :676-681, 762-766 in its 11-layer model) at its depth fractions: the states after 4/11 and
    8/11 of the blocks (cache[3], cache[7]) and the last block's input (cache[9], after block L - 2). Site 1: x0, the
    8/11 state, the last block's input (modded's mu0..mu5 sources). Site 2: x0, the 4/11, 8/11 and L - 2 states, the
    state after block L // 2 (the backout's, for site 2's prior) and every value-embedding plane (modded mixes all of
    its planes post-loop, plus the last block's MLP and attention inputs and per-channel-group deltas (mudd_w2g), which
    this model omits)."""
    after = lambda frac: round(frac * n_layer) - 1  # the block after which frac of the n_layer blocks are done
    site1 = ["x0"] + [("h", i) for i in sorted({after(8 / 11), n_layer - 2})]
    site2 = ["x0"] + [("h", i) for i in sorted({after(4 / 11), after(8 / 11), n_layer - 2, n_layer // 2})]
    return site1, site2 + [("ve", l) for l in ve_layers]


class MuddSite(nn.Module):
    """The coefficient MLP of one site: n_sources sources, n_streams streams (outputs n_streams * n_sources: stream j's
    coefficients are columns j*n_sources .. (j+1)*n_sources - 1)."""

    def __init__(self, n_sources, n_streams, n_embd):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(MUDD_DIM, n_embd))
        self.w2 = nn.Parameter(torch.empty(n_streams * n_sources, MUDD_DIM))
        self.b = nn.Parameter(torch.empty(n_streams * n_sources))

    @torch.no_grad()
    def init_weights(self, prior):
        """prior: (n_streams * n_sources,) the coefficients at init."""
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.zeros_(self.w2)
        self.b.copy_(prior / MUDD_SCALE)

    def forward(self, x):
        """Coefficients (B, T, n_streams * n_sources) fp32 of the state x (B, T, C) bf16: the hidden layer from the
        rms-normed bf16 state with fp32 output (nanochat::narrow_linear), the rest in fp32."""
        h = F.gelu(narrow_linear(norm(x), self.w1))
        return F.linear(h, self.w2, self.b) * MUDD_SCALE


def mix(coefs, sources, x=None):
    """fp32 (x +) sum_s coefs[..., s] * sources[s], s ascending; coefs (B, T, len(sources)) fp32, sources and x (B, T, C)
    bf16."""
    out = None if x is None else x.float()
    for s, src in enumerate(sources):
        term = coefs[..., s:s + 1] * src.float()
        out = term if out is None else out + term
    return out
