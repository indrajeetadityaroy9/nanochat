"""
Causal self-attention (dev/D20_SPEC.md "Model"): the layers' windows and key offsets, the rotary tables, the document
mask's segment starts, and CausalSelfAttention (value embeddings, fused q/k prologue, FlashAttention-4, XSA and the
head-wise output gate).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.model.config import ROPE_BASE, SHORT_WINDOW, has_ve
from nanochat.model.layers import Linear, ProjLinear, norm
from nanochat.ops.flash_attn import flash_attn
from nanochat.ops.linear import narrow_linear
from nanochat.ops.qk_prologue import qk_norm_rope
from nanochat.ops.xsa import xsa


def layer_windows(config):
    """(window_sizes, key_offsets) per layer. The left window: the pattern tiled across layers, the last layer full
    context; S = SHORT_WINDOW keys at every context length (karpathy/nanochat 92d63d4 gpt.py:287-314: S = T/4 at its
    T = 2048, i.e. 512; fixed in tokens as Gemma 2/3's local layers, arXiv 2408.00118 §2, 2503.19786 §2), at most the
    context. The key offset on the full-context layers' keys (modded-nanogpt track_1_short applies it on its long-window
    layers, model/gpt.py:38 LONG_WINDOW_LAYERS): the SSSL 'L' layers and the last."""
    window = {"L": config.sequence_len, "S": min(SHORT_WINDOW, config.sequence_len)}
    window_sizes = [window[config.window_pattern[i % len(config.window_pattern)]] for i in range(config.n_layer)]
    window_sizes[-1] = config.sequence_len
    return window_sizes, [w == config.sequence_len for w in window_sizes]


def rotary_tables(config, device):
    """cos/sin (fp32, contiguous, shape (n_layer, sequence_len, R/2): layer l's table in QuACK's rotary layout) of the
    rotary embeddings over the first R = head_dim/2 dims of each head (modded-nanogpt's half-truncated RoPE,
    track_1_short model/attention.py:53-55; the other half stationary, the key offset's dims). Each layer's R/2
    frequencies are geometrically spaced from 1 rad/token, omega_i = base ** (-i / (R/2 - 1)), i = 0 .. R/2 - 1, with
    the base from the layer's window w: a full-context layer nanochat's ROPE_BASE (92d63d4 gpt.py:270-285; slowest
    wavelength 2 pi 1e5 beyond the context); a short-window layer base = w / (2 pi), so the slowest pair turns once
    (2 pi rad) over the window. Angles and their cos/sin in fp64, rounded once to fp32 (an fp32 angle t * f would
    carry an absolute error up to t * 2^-24)."""
    pairs = config.n_embd // config.n_head // 4  # R/2 rotated pairs per head
    exponent = torch.arange(pairs, dtype=torch.float64, device=device) / (pairs - 1)
    t = torch.arange(config.sequence_len, dtype=torch.float64, device=device)
    bases = [ROPE_BASE if w == config.sequence_len else w / (2 * math.pi) for w in layer_windows(config)[0]]
    freqs = torch.stack([torch.outer(t, base ** -exponent) for base in bases])  # angle at (layer, position, pair)
    return freqs.cos().float(), freqs.sin().float()


def segment_starts(cu_seqlens, B, T):
    """(B, T) bool: the first token of each attention segment of the flattened (B*T) tokens (document masking). The final
    offset B*T (and the loader's padding entries) land in the extra slot."""
    starts = torch.zeros(B * T + 1, dtype=torch.bool, device=cu_seqlens.device)
    return starts.index_fill_(0, cu_seqlens.long(), True)[:-1].view(B, T)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = ProjLinear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = ProjLinear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = ProjLinear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = ProjLinear(self.n_embd, self.n_embd, bias=False)
        # Value-embedding gate of modded record #82 (modded-nanogpt @4ea6b93 README.md:220, "Learnable XSA"; log
        # R82 = records/track_1_short/2026-04-29_XSAGatedLayers/this_pr_v1-s1410/06563169-6435-48ba-a1ad-f3e61bfcc573.txt)
        # on its non-paired layers (R82:1101-1104, 1179: 2 * sigmoid(W cat(x[..., :6], ve[..., :6])),
        # W zero-init, range (0, 2))
        self.ve_gate_channels = 12  # the gate's inputs: half from the normed x, half from the value embedding
        self.ve_gate_scale = 2
        # Attention logit scale: modded track_1_short's softmax scale on unit-RMS q, k (attention.py:220
        # softmax_scale=yarn.attn_scale): WIDE_ATTN_SCALE at head_dim 128 (model/gpt.py:53, 214) at its starting long
        # window ws_long 3 blocks of BLOCK_SIZE 128 tokens (config.py:62, 76), grown by 0.2 ln(new / old) + 1 per
        # window change (attention.py:93); here one change, to this layer's window. FA4 applies 1/sqrt(head_dim), so
        # the prologue scales q and k each by sqrt(attn_scale * sqrt(head_dim)).
        window = layer_windows(config)[0][layer_idx]
        attn_scale = 0.085 * (0.2 * math.log(window / (3 * 128)) + 1)
        self.qk_scale = math.sqrt(attn_scale * math.sqrt(self.head_dim))
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        # Head-wise attention-output gate (Qiu et al., arXiv 2505.06708, G1 head-wise; modded record #28's zero init:
        # sigmoid(0) = 1/2 per head)
        self.attn_gate = Linear(self.n_embd, self.n_head, bias=False)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, key_offset, starts, cu_seqlens, v_mix=None):
        """x: the residual stream, normed here. key_offset: this layer's keys take the previous position's stationary
        dims (none at row and segment starts: starts (B, T) bool or None). v_mix: MUDD's value stream (B, T, C) fp32,
        added to the values or None."""
        B, T, C = x.size()

        # Project the normed input to get queries, keys, and values
        x = norm(x)
        q, k, v = self.c_q(x), self.c_k(x), self.c_v(x)
        # Shape: (B, T, H, D) - FlashAttention's native layout, no transpose needed!
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)
        v = self._mix_values(x, v, ve, v_mix)

        # Rotary by -theta over the first half of each head (the tables' dims), QK norm and qk_scale fused, the key
        # offset if set, fp32 inside and one bf16 rounding per output (ops/qk_prologue.cu)
        cos, sin = cos_sin
        q_in, k_in = q, k
        q, k = qk_norm_rope(q, k, cos, sin, self.qk_scale, key_offset, starts)
        if kv_cache is not None:
            self._fill_cache(kv_cache, q_in, k_in, k, v, cos, sin, key_offset)

        # FlashAttention-4, causal: each query sees itself and `window_size` keys before it
        y = flash_attn(q, k, v, window_size, cu_seqlens)[0]

        # XSA at strength 1 (Zhai, arXiv 2603.09078 Eq. 2; modded-nanogpt track_1_short attention.py:224-228):
        # y -= (y.v / max(||v||^2, 1e-8)) v per head, v the attention's value input (value embedding included), then the
        # output gate sigmoid(norm(x) W_g^T) per head (its logits fp32, nanochat::narrow_linear), both in one fused op
        # (ops/xsa.py)
        y = xsa(y, v, narrow_linear(x, self.attn_gate.weight))

        # Re-assemble the heads and project back to residual stream
        y = y.view(B, T, -1)
        y = self.c_proj(y)
        return y

    def _mix_values(self, x, v, ve, v_mix):
        """Value residual (ResFormer): mix in the value embedding ve with an input-dependent gate per head, in fp32 (the
        gate's weight gradient reduced in fp32 straight to its fp32 .grad), rounded once to bf16; MUDD's value stream
        v_mix joins the same fp32 sum. x: the normed input; v (B, T, H, D)."""
        if ve is None and v_mix is None:
            return v
        B, T = x.shape[:2]
        v_sum = v.float()
        if ve is not None:
            half = self.ve_gate_channels // 2
            gate_in = torch.cat([x[..., :half], ve[..., :half]], dim=-1).float()
            gate = self.ve_gate_scale * torch.sigmoid(F.linear(gate_in, self.ve_gate.weight))  # (B, T, H), range (0, 2)
            v_sum = v_sum + gate.unsqueeze(-1) * ve.view(B, T, self.n_kv_head, self.head_dim).float()
        if v_mix is not None:
            v_sum = v_sum + v_mix.view(B, T, self.n_kv_head, self.head_dim)
        return v_sum.to(v.dtype)

    def _fill_cache(self, kv_cache, q_in, k_in, k, v, cos, sin, key_offset):
        """Inference prefill (from position 0): the decoder continues from this layer's keys/values, and with the key
        offset from the last position's own (unshifted) stationary key dims. q_in, k_in: the prologue's inputs."""
        T = k.size(1)
        kv_cache.k_cache[self.layer_idx, :, :T], kv_cache.v_cache[self.layer_idx, :, :T] = k, v
        if key_offset:
            last = lambda t: t[:, T - 1:]
            own = qk_norm_rope(last(q_in), last(k_in), cos[T - 1:T], sin[T - 1:T], self.qk_scale, False, None)[1]
            kv_cache.key_tail[self.layer_idx] = own[:, 0, :, 2 * cos.size(1):]
