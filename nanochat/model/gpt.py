"""
The GPT: karpathy/nanochat 92d63d4 nanochat/gpt.py with the wins of its experiment log (dev/LOG.md, "LOG") and Qwen3's
QK norm with learnable gains (arXiv 2505.09388 §2).
- Normed token embedding, no positional embedding; RoPE over the full head (92d63d4 gpt.py:57-65).
- Per layer x = resid_lambda x + x0_lambda x0, x0 the normed embedding (LOG 2026-01-11).
- Attention: MHA, head_dim 128 (LOG 2026-01-17); per-head RMS norm, a per-dim gain shared across heads, then RoPE
  (transformers modeling_qwen3.py Qwen3Attention.forward); SSSL windows; value embeddings on alternating layers behind a
  gate (ResFormer, arXiv 2410.17897; LOG 2026-01-17); FlashAttention-4 with document masking.
- MLP ReLU^2, hidden 4C (LOG 2026-02-05: SwiGLU worse at d12/d24).
- Gain-free RMS norm, bias-free linears, untied lm_head (modded-nanogpt record #8), plain cross entropy.
fp32 parameters, bf16 activations; the ops round weights to bf16 inside. Gradient reductions use the fastest kernels,
not a fixed order (FA4's and the embedding backward's atomics: ~1% faster per step, benchmarks/h100.txt).
"""

import zlib
from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.config import ROPE_BASE, VOCAB_SIZE
from nanochat.ops.cross_entropy import lm_loss, lm_losses
from nanochat.ops.depth_attention import depth_inter, depth_merge
from nanochat.ops.flash_attn import flash_attn
from nanochat.ops.linear import linear, relu2_mlp
from nanochat.ops.qk_prologue import qk_norm_rope

# Width per layer: MobileLLM-350M's deep-and-thin 32 x 960 (arXiv 2402.14905 §2.2.2, Fig. 4)
ASPECT_RATIO = 960 // 32
# Block Attention Residuals' blocks: N ~ 8 recovers most of Full AttnRes's gain (arXiv 2603.15031 §3.2)
ATTN_RES_BLOCKS = 8


@dataclass(kw_only=True)
class Mechanism:
    """The components the appendix studies switch (experiments/studies.py), baseline first in each Literal.
    value_embeds: value embeddings and gates on alternating layers, or none; x0: the learned x0 skip, or none; resid:
    learned residual scalars, or 1. Candidate additions: attn_res, Block Attention Residuals (arXiv 2603.15031;
    replaces resid and x0); key_offset, p-RoPE with the partial key offset on full-context layers (modded-nanogpt
    record #49)."""
    value_embeds: Literal["alternating", "none"] = "alternating"
    x0: Literal["learned", "none"] = "learned"
    resid: Literal["learned", "one"] = "learned"
    attn_res: Literal["none", "block"] = "none"
    key_offset: Literal["none", "partial"] = "none"


@dataclass(kw_only=True)
class GPTConfig(Mechanism):
    sequence_len: int
    vocab_size: int  # rows of wte, value_embeds and lm_head
    rope_base: float
    n_layer: int
    n_head: int
    n_embd: int


def gpt_config(*, depth, stage):
    """nanochat's shape rule (92d63d4 scripts/base_train.py:51-52, 131-134): width ASPECT_RATIO x depth rounded up to
    whole heads of 128 (LOG 2026-01-17; csrc/qk_row.cuh HEAD_DIM); the stage's context; config.ROPE_BASE."""
    head_dim = 128
    n_embd = -(-depth * ASPECT_RATIO // head_dim) * head_dim
    return GPTConfig(sequence_len=stage.seq_len, vocab_size=VOCAB_SIZE, rope_base=ROPE_BASE, n_layer=depth,
                     n_head=n_embd // head_dim, n_embd=n_embd)


def has_ve(layer_idx, *, n_layer):
    """Alternating layers, the last included (92d63d4 gpt.py:53-55)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def layer_windows(config):
    """Keys before each query, per layer: SSSL tiled, the last layer full (LOG 2026-01-11; 92d63d4 gpt.py:287-314);
    S = 512, the smallest local window Gemma 3 finds near-lossless (arXiv 2503.19786 §5.2, Fig. 4)."""
    pattern = "SSSL"
    window = {"S": min(512, config.sequence_len), "L": config.sequence_len}
    sizes = [window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
    sizes[-1] = config.sequence_len
    return sizes


def rotary_tables(config, *, device):
    """cos/sin (sequence_len, head_dim/2) fp32 of angle(t, i) = t rope_base^(-2i/head_dim) (92d63d4 gpt.py:270-285),
    computed in fp64 and rounded once (an fp32 angle would carry an error up to t 2^-24)."""
    head_dim = config.n_embd // config.n_head
    exponent = torch.arange(0, head_dim, 2, dtype=torch.float64, device=device) / head_dim
    t = torch.arange(config.sequence_len, dtype=torch.float64, device=device)
    freqs = torch.outer(t, config.rope_base ** -exponent)
    return freqs.cos().float(), freqs.sin().float()


def norm(x):
    """RMS norm, statistics in fp32, eps FLT_EPSILON (the convention of csrc/qk_prologue.cu and csrc/decode.cu)."""
    return F.rms_norm(x, (x.size(-1),), eps=torch.finfo(torch.float32).eps)


class Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        C, H = config.n_embd, config.n_head
        self.layer_idx = layer_idx
        self.n_head = H
        self.c_q = nn.Linear(C, C, bias=False)
        self.c_k = nn.Linear(C, C, bias=False)
        self.c_v = nn.Linear(C, C, bias=False)
        self.c_o = nn.Linear(C, C, bias=False)
        self.q_gain = nn.Parameter(torch.empty(C // H))
        self.k_gain = nn.Parameter(torch.empty(C // H))
        self.ve_gate = (nn.Linear(C, H, bias=False)
                        if config.value_embeds == "alternating" and has_ve(layer_idx, n_layer=config.n_layer) else None)
        self.depth_query = nn.Parameter(torch.empty(C)) if config.attn_res == "block" else None

    def forward(self, n, ve, *, cos_sin, window_size, kv_cache, cu_seqlens, starts):
        """n: the normed stream; ve: this layer's value embeddings or None; starts: segment starts on a key-offset
        layer, else None; kv_cache (decoder prefill) receives the keys and values."""
        B, T, C = n.size()
        q, k, v = (linear(n, w).view(B, T, self.n_head, -1) for w in (self.c_q.weight, self.c_k.weight, self.c_v.weight))
        # Value residual with a per-head gate 2 sigmoid(.) in (0, 2), 1 at the zero-init gate; fp32, rounded once. The
        # gate as a broadcast-sum, fused by Inductor: 0.88 vs 1.20 ms per layer and micro-batch for fp32 GEMMs
        # (benchmarks/h100.txt)
        if ve is not None:
            gate = 2 * torch.sigmoid((n.float().unsqueeze(-2) * self.ve_gate.weight).sum(-1))
            v = (v.float() + gate.unsqueeze(-1) * ve.view_as(v).float()).to(v.dtype)
        q, k = qk_norm_rope(q, k, self.q_gain, self.k_gain, *cos_sin)
        if starts is not None:
            # Partial key offset (modded-nanogpt track_1_short perf/kernels/qkv_rope.py KEY_OFFSET): a key's unrotated
            # dims (D/4..D/2-1, 3D/4..D-1) are the previous token's, except at a segment start
            D = k.size(-1)
            stationary = torch.arange(D, device=k.device) % (D // 2) >= D // 4
            previous = torch.roll(k.flatten(0, 1), 1, 0).view_as(k)
            k = torch.where(stationary & ~starts[..., None, None], previous, k)
        if kv_cache is not None:
            kv_cache.k_cache[self.layer_idx, :, :T], kv_cache.v_cache[self.layer_idx, :, :T] = k, v
        return linear(flash_attn(q, k, v, window_size, cu_seqlens)[0].view(B, T, C), self.c_o.weight)


class MLP(nn.Module):
    """c_proj(relu(c_fc(x))^2), the activation and its derivative in QuACK GEMM epilogues (ops/linear.py)."""

    def __init__(self, config):
        super().__init__()
        C = config.n_embd
        self.c_fc = nn.Linear(C, 4 * C, bias=False)
        self.c_proj = nn.Linear(4 * C, C, bias=False)
        self.depth_query = nn.Parameter(torch.empty(C)) if config.attn_res == "block" else None

    def forward(self, x):
        return relu2_mlp(x, self.c_fc.weight, self.c_proj.weight)[0]


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = Attention(config, layer_idx)
        self.mlp = MLP(config)


class GPT(nn.Module):
    def __init__(self, config):
        """Shapes only (built on the meta device); init_weights sets the values."""
        super().__init__()
        self.config = config
        self.window_sizes = layer_windows(config)
        C, V = config.n_embd, config.vocab_size
        self.wte = nn.Embedding(V, C)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.lm_head = nn.Linear(C, V, bias=False)
        self.resid_lambdas = nn.Parameter(torch.empty(config.n_layer)) if config.resid == "learned" else None
        self.x0_lambdas = nn.Parameter(torch.empty(config.n_layer)) if config.x0 == "learned" else None
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(V, C) for i in range(config.n_layer)
                                           if config.value_embeds == "alternating" and has_ve(i, n_layer=config.n_layer)})
        self.depth_query = nn.Parameter(torch.empty(C)) if config.attn_res == "block" else None
        # init_buffers: rotary tables, and with key_offset the p-RoPE tables
        for name in ("cos", "sin", "cos_p", "sin_p"):
            self.register_buffer(name, None, persistent=False)

    @torch.no_grad()
    def init_weights(self, *, seed):
        """Matrices and tables that read the stream: uniform, std C^-0.5 (LeCun et al., "Efficient BackProp" §4.6).
        Zero: c_o, c_proj, lm_head (each block the identity at init, modded-nanogpt record #5), the VE gates (gate 1),
        the depth queries (arXiv 2603.15031 §5), x0_lambdas. One: resid_lambdas, gains (LOG 2026-01-11). Each random
        tensor has its own generator seeded by `seed` and its name, so a model without a component starts every other
        tensor as the full model does (common random numbers across a study's runs)."""
        bound = (3 / self.config.n_embd) ** 0.5
        zero = ("attn.c_o.weight", "mlp.c_proj.weight", "lm_head.weight", "attn.ve_gate.weight", "x0_lambdas",
                "depth_query")
        one = ("q_gain", "k_gain", "resid_lambdas")
        for name, t in self.named_parameters():
            if name.endswith(zero):
                t.zero_()
            elif name.endswith(one):
                t.fill_(1)
            else:
                t.uniform_(-bound, bound,
                           generator=torch.Generator(t.device).manual_seed(seed * 2**32 + zlib.crc32(name.encode())))
        self.init_buffers()

    def init_buffers(self):
        """The non-checkpointed buffers: rotary tables; with key_offset the p-RoPE tables, the lower-frequency half of
        the pairs unrotated (Barbero et al., arXiv 2410.06205 §6; modded-nanogpt's half-truncated RoPE)."""
        device = self.wte.weight.device
        self.cos, self.sin = rotary_tables(self.config, device=device)
        if self.config.key_offset == "partial":
            half = self.cos.size(-1)
            stationary = torch.arange(half, device=device) >= half // 2
            self.cos_p, self.sin_p = self.cos.masked_fill(stationary, 1), self.sin.masked_fill(stationary, 0)

    def set_stage(self, stage):
        """Continue at a stage's context: windows and rotary tables recomputed, weights unchanged."""
        self.config.sequence_len = stage.seq_len
        self.window_sizes = layer_windows(self.config)
        self.init_buffers()

    def hidden(self, idx, cu_seqlens, kv_cache):
        """Final normed hidden states (B*T, C) of idx (B, T); cu_seqlens: int32 document segments of the flattened idx.
        attn_res (arXiv 2603.15031 §3.2 Eq. 6, two-phase computation §4.2 Algorithm 1): the sources are the embedding,
        the completed blocks (sums of 2 n_layer / ATTN_RES_BLOCKS consecutive sublayer outputs) and the current block's
        sum; depth_inter scores every query of a block against the completed blocks once, depth_merge merges the
        block's sum into each later sublayer's input by online softmax; both return the normed input."""
        cfg = self.config
        B, T = idx.shape
        x = norm(F.embedding(idx, self.wte.weight).to(torch.bfloat16))
        x0 = x
        # segment starts: the positions in cu_seqlens (its trailing repeats of B*T are out of range)
        starts = (torch.isin(torch.arange(B * T, dtype=cu_seqlens.dtype, device=idx.device), cu_seqlens).view(B, T)
                  if cfg.key_offset == "partial" else None)
        blocks, block_sum, modules = [x0], None, [m for block in self.blocks for m in (block.attn, block.mlp)]
        per_block = 2 * cfg.n_layer // ATTN_RES_BLOCKS
        for i, block in enumerate(self.blocks):
            # 0-dim fp32 lambdas x bf16: Inductor reduces their gradients in fp32 without fp32 copies of the mix
            if cfg.resid == "learned":
                x = self.resid_lambdas[i] * x
            if cfg.x0 == "learned":
                x = x + self.x0_lambdas[i] * x0
            ve = (F.embedding(idx, self.value_embeds[str(i)].weight).to(torch.bfloat16) if str(i) in self.value_embeds
                  else None)
            offset = cfg.key_offset == "partial" and self.window_sizes[i] == cfg.sequence_len
            tables = (self.cos_p, self.sin_p) if offset else (self.cos, self.sin)
            sublayers = ((block.attn, partial(block.attn, ve=ve, cos_sin=tables, window_size=self.window_sizes[i],
                                              kv_cache=kv_cache, cu_seqlens=cu_seqlens,
                                              starts=starts if offset else None)),
                         (block.mlp, block.mlp))
            for j, (module, sublayer) in enumerate(sublayers):
                if cfg.attn_res == "block":
                    k = (2 * i + j) % per_block  # the sublayer's place in its block
                    if k == 0:
                        queries = torch.stack([m.depth_query for m in modules[2 * i + j:2 * i + j + per_block]])
                        n, inter, lse, _ = depth_inter(blocks, queries)
                    else:
                        n = depth_merge(inter[k - 1], lse[k - 1], block_sum, module.depth_query)[0]
                    out = sublayer(n)
                    block_sum = out if k == 0 else block_sum + out
                    if k == per_block - 1:
                        blocks = [*blocks, block_sum]
                else:
                    x = x + sublayer(norm(x))
        if cfg.attn_res == "block":
            return depth_inter(blocks, self.depth_query[None])[0].flatten(0, 1)
        return norm(x).flatten(0, 1)

    def forward(self, idx, targets, cu_seqlens):
        """Mean next-token loss over the non-padding targets, fp32."""
        return lm_loss(self.hidden(idx, cu_seqlens, None), self.lm_head.weight, targets.flatten())[0]

    def losses(self, idx, targets, cu_seqlens):
        """Per-position next-token loss (B*T,) fp32, 0 at padding targets."""
        return lm_losses(self.hidden(idx, cu_seqlens, None), self.lm_head.weight, targets.flatten())

    def prefill(self, idx, kv_cache):
        """Fill kv_cache with the keys and values of one prompt idx (1, T)."""
        self.hidden(idx, torch.tensor([0, idx.size(1)], dtype=torch.int32, device=idx.device), kv_cache)


def scaling_params(model):
    """The blocks' parameters + lm_head, the parameters the token budget scales with (92d63d4
    scripts/base_train.py:263-267)."""
    return sum(p.numel() for p in model.blocks.parameters()) + model.lm_head.weight.numel()


def non_embedding_flops_per_token(model):
    """DeepSeek LLM's M (arXiv 2401.02954 §3.1 Eq. 2): 6 per block matrix weight plus 12 C l_seq per layer, l_seq the
    layer's window (92d63d4 gpt.py:319-339)."""
    matrices = sum(p.numel() for p in model.blocks.parameters() if p.ndim == 2)
    return 6 * matrices + sum(12 * model.config.n_embd * w for w in model.window_sizes)
