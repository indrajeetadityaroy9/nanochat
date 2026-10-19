"""
The GPT: karpathy/nanochat 92d63d4 nanochat/gpt.py with the measured wins of its experiment log (dev/LOG.md, "LOG" below)
and Qwen3's QK norm with learnable gains (arXiv 2505.09388 §2).
- Token embedding normed, no positional embedding; rotary embeddings over the full head (92d63d4 gpt.py:57-65).
- Per layer x = resid_lambdas[i] x + x0_lambdas[i] x0, x0 the normed embedding (LOG 2026-01-11).
- Attention: MHA, head_dim 128 (LOG 2026-01-17); per head RMS norm, a per-dim gain shared across heads, then RoPE
  (Qwen3; transformers modeling_qwen3.py Qwen3Attention.forward: q_norm/k_norm, then apply_rotary_pos_emb); windows
  SSSL (layer_windows); value embeddings on alternating layers with a gate (ResFormer arXiv 2410.17897; LOG
  2026-01-17); FlashAttention-4 with document masking (ops/flash_attn.py).
- MLP ReLU^2, hidden 4C (LOG 2026-02-05: SwiGLU worse at d12/d24).
- RMS norm without gains, linear layers without bias, untied lm_head (modded-nanogpt record #8), plain cross entropy.
Parameters are fp32 masters; activations bf16: the tables are looked up and rounded once, every matmul runs through the
nanochat ops, which round the weights inside. The optimizer's parameter groups are train/optim.py's.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.ops.cross_entropy import lm_loss, lm_losses
from nanochat.ops.flash_attn import flash_attn
from nanochat.ops.linear import linear, relu2_mlp
from nanochat.ops.qk_prologue import qk_norm_rope

# Width per layer of depth: MobileLLM-350M's deep-and-thin shape, 32 layers x 960 (Liu et al. arXiv 2402.14905 §2.2.2
# Fig. 4: below 1B, deeper and thinner beats wider and shallower at comparable size)
ASPECT_RATIO = 960 // 32


@dataclass(kw_only=True)
class GPTConfig:
    sequence_len: int
    vocab_size: int  # the tokenizer's ids, specials included: the rows of wte, value_embeds and lm_head
    rope_base: int   # the rotary frequencies rope_base^(-i/64), i = 0..63, over the full head
    n_layer: int
    n_head: int      # query heads = key/value heads (MHA)
    n_embd: int


def gpt_config(depth, vocab_size, stage, *, aspect_ratio):
    """The architecture at a depth: nanochat's shape rule (92d63d4 scripts/base_train.py:51-52, 131-134: width =
    aspect_ratio x depth rounded up to whole heads of head_dim 128, LOG 2026-01-17; inference/decode.cu HEAD_DIM); the
    context and rotary base of a training stage (config.Stage)."""
    head_dim = 128
    n_embd = -(-depth * aspect_ratio // head_dim) * head_dim
    return GPTConfig(sequence_len=stage.seq_len, vocab_size=vocab_size, rope_base=stage.rope_base, n_layer=depth,
                     n_head=n_embd // head_dim, n_embd=n_embd)


def has_ve(layer_idx, n_layer):
    """Whether layer layer_idx has a value embedding: alternating layers, the last one included (92d63d4 gpt.py:53-55;
    LOG 2026-01-17: alternating layers worked best)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def layer_windows(config):
    """The left window (keys before each query) per layer: the pattern SSSL tiled across layers, the last layer L (LOG
    2026-01-11; 92d63d4 gpt.py:287-314); L the context, S = 512 keys at most the context (Gemma 3 arXiv 2503.19786
    §5.2 Fig. 4: local windows of 512..4096 change perplexity minimally)."""
    pattern = "SSSL"
    window = {"S": min(512, config.sequence_len), "L": config.sequence_len}
    sizes = [window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
    sizes[-1] = config.sequence_len
    return sizes


def rotary_tables(config, device):
    """cos/sin (fp32, contiguous, shape (sequence_len, head_dim/2)) of the rotary embeddings over the full head, one table
    for every layer: angle(t, i) = t * rope_base^(-i / (head_dim/2)), i = 0 .. head_dim/2 - 1 (92d63d4 gpt.py:270-285:
    inv_freq = base^(-(2i)/head_dim)). Angles and their cos/sin in fp64, rounded once to fp32 (an fp32 angle t * f would
    carry an absolute error up to t * 2^-24)."""
    head_dim = config.n_embd // config.n_head
    exponent = torch.arange(0, head_dim, 2, dtype=torch.float64, device=device) / head_dim
    t = torch.arange(config.sequence_len, dtype=torch.float64, device=device)
    freqs = torch.outer(t, config.rope_base ** -exponent)
    return freqs.cos().float(), freqs.sin().float()


def norm(x):
    """RMS norm over the last dim: F.rms_norm on bf16 computes its statistics in fp32 and rounds the output once. The
    one epsilon convention of every RMS norm (here, ops/qk_prologue.cu and inference/decode.cu): F.rms_norm's default,
    the fp32 machine epsilon FLT_EPSILON = 2^-23 also for bf16 inputs (torch.finfo of the fp32 opmath type;
    modded-nanogpt track_1_short perf/kernels/qkv_rope.py:8)."""
    return F.rms_norm(x, (x.size(-1),))


class Block(nn.Module):
    """One transformer layer: attention (the projections, the value-embedding mix, the q/k prologue and FlashAttention-4,
    which computes only softmax(q k^T / sqrt(D)) v) and the MLP, each pre-normed and added to the residual stream."""

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
        self.ve_gate = nn.Linear(C, H, bias=False) if has_ve(layer_idx, config.n_layer) else None
        self.c_fc = nn.Linear(C, 4 * C, bias=False)
        self.c_proj = nn.Linear(4 * C, C, bias=False)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, cu_seqlens):
        """x: the residual stream; ve: this layer's value embeddings (bf16) or None. kv_cache (decoder prefill from
        position 0): receives this layer's keys (after the prologue) and values (after the value-embedding mix)."""
        B, T, C = x.size()
        n = norm(x)
        q, k, v = (linear(n, w).view(B, T, self.n_head, -1) for w in (self.c_q.weight, self.c_k.weight, self.c_v.weight))
        # Value residual: the value embedding mixed in with an input-dependent gate per head, 2 sigmoid(.) in (0, 2),
        # 1 at the zero-init gate weight (unit weight at init); in fp32 (the gate's weight gradient reduced in fp32
        # straight to its fp32 .grad), rounded once to bf16
        if ve is not None:
            gate = 2 * torch.sigmoid(F.linear(n.float(), self.ve_gate.weight))  # (B, T, H)
            v = (v.float() + gate.unsqueeze(-1) * ve.view_as(v).float()).to(v.dtype)
        # per head RMS norm, the q / k gain, RoPE (ops/qk_prologue.cu, one pass over memory per direction: its forward
        # and backward take 228 ms of a d24 step where Inductor's fusion of the same ops takes 336 ms, bench/h100.txt)
        q, k = qk_norm_rope(q, k, self.q_gain, self.k_gain, *cos_sin)
        if kv_cache is not None:
            kv_cache.k_cache[self.layer_idx, :, :T], kv_cache.v_cache[self.layer_idx, :, :T] = k, v
        # FlashAttention-4, causal: each query sees itself and `window_size` keys before it
        x = x + linear(flash_attn(q, k, v, window_size, cu_seqlens)[0].view(B, T, C), self.c_o.weight)
        # the MLP c_proj(relu(c_fc(norm(x)))^2) on QuACK's GEMMs (ops/linear.py relu2_mlp: the activation in the
        # up-projection GEMM's epilogue, its derivative in the backward GEMM's)
        return x + relu2_mlp(norm(x), self.c_fc.weight, self.c_proj.weight)[0]


class GPT(nn.Module):
    def __init__(self, config):
        """Shapes only (the trainer and the checkpoint loader build it on the meta device); init_weights sets the
        values."""
        super().__init__()
        self.config = config
        self.window_sizes = layer_windows(config)
        C, V = config.n_embd, config.vocab_size
        self.wte = nn.Embedding(V, C)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.lm_head = nn.Linear(C, V, bias=False)
        self.resid_lambdas = nn.Parameter(torch.empty(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.empty(config.n_layer))
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(V, C) for i in range(config.n_layer)
                                           if has_ve(i, config.n_layer)})
        # the rotary tables of the context (rotary_tables), computed by init_weights and set_stage
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """Every matrix and table that reads the stream uniform with std C^-0.5 (variance preserving, LeCun et al.
        "Efficient BackProp" §4.6; bound sqrt(3) std, the std of U(-a, a) being a / sqrt(3)): c_q, c_k, c_v, c_fc,
        wte, value_embeds. Zero: c_o, c_proj, lm_head (each block the identity at init, modded-nanogpt record #5) and
        ve_gate (gate 1). resid_lambdas 1, x0_lambdas 0 (LOG 2026-01-11), q_gain = k_gain = 1."""
        bound = (3 / self.config.n_embd) ** 0.5
        for block in self.blocks:
            for w in (block.c_q.weight, block.c_k.weight, block.c_v.weight, block.c_fc.weight):
                nn.init.uniform_(w, -bound, bound)
            nn.init.zeros_(block.c_o.weight)
            nn.init.zeros_(block.c_proj.weight)
            nn.init.ones_(block.q_gain)
            nn.init.ones_(block.k_gain)
            if block.ve_gate is not None:
                nn.init.zeros_(block.ve_gate.weight)
        for table in (self.wte, *self.value_embeds.values()):
            nn.init.uniform_(table.weight, -bound, bound)
        nn.init.zeros_(self.lm_head.weight)
        nn.init.ones_(self.resid_lambdas)
        nn.init.zeros_(self.x0_lambdas)
        self.cos, self.sin = rotary_tables(self.config, self.wte.weight.device)

    def set_stage(self, stage):
        """Continue at a training stage's context and rotary base (config.Stage; the weights are unchanged): the layers'
        windows and the rotary tables of the new context, on the parameters' device."""
        self.config.sequence_len, self.config.rope_base = stage.seq_len, stage.rope_base
        self.window_sizes = layer_windows(self.config)
        self.cos, self.sin = rotary_tables(self.config, self.wte.weight.device)

    def hidden(self, idx, cu_seqlens, kv_cache):
        """The final normed hidden states (B*T, C) of idx (B, T) token ids; cu_seqlens: the int32 attention segments of
        the flattened idx (document masking; data/loader.py PretrainingBatches). kv_cache (decoder prefill) receives
        every layer's keys and values."""
        x = norm(self.wte(idx).to(torch.bfloat16))
        x0 = x
        for i, block in enumerate(self.blocks):
            # 0-dim fp32 lambdas x bf16: Inductor reduces their gradients in fp32 (no fp32 copies of the mix saved for
            # the backward, as an explicit fp32 mix would)
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(torch.bfloat16) if str(i) in self.value_embeds else None
            x = block(x, ve, (self.cos, self.sin), self.window_sizes[i], kv_cache, cu_seqlens)
        return norm(x).flatten(0, 1)

    def forward(self, idx, targets, cu_seqlens):
        """The training loss: the mean next-token loss over the targets (B, T) that are not the loader's padding,
        scalar fp32."""
        return lm_loss(self.hidden(idx, cu_seqlens, None), self.lm_head.weight, targets.flatten())[0]

    def losses(self, idx, targets, cu_seqlens):
        """Each position's next-token loss (B*T,) fp32, 0 at padding targets (evaluation)."""
        return lm_losses(self.hidden(idx, cu_seqlens, None), self.lm_head.weight, targets.flatten())

    def prefill(self, idx, kv_cache):
        """Fill kv_cache with the keys and values of one prompt idx (1, T) read from position 0 (one segment)."""
        self.hidden(idx, torch.tensor([0, idx.size(1)], dtype=torch.int32, device=idx.device), kv_cache)


def scaling_params(model):
    """The parameters the token budget scales with: the blocks' parameters + lm_head (nanochat's transformer_matrices +
    lm_head, the cleanest scaling laws: 92d63d4 scripts/base_train.py:263-267, gpt.py:403-406)."""
    return sum(p.numel() for p in model.blocks.parameters()) + model.lm_head.weight.numel()


def non_embedding_flops_per_token(model):
    """DeepSeek LLM's non-embedding FLOPs per token M (arXiv 2401.02954 §3.1 Eq. 2): 6 per weight of the blocks'
    matrices (forward 2, backward 4) plus the attention's 12 C l_seq per layer, l_seq the layer's window (at most the
    context; 92d63d4 gpt.py:319-339)."""
    matrices = sum(p.numel() for p in model.blocks.parameters() if p.ndim == 2)
    return 6 * matrices + sum(12 * model.config.n_embd * w for w in model.window_sizes)


def flops_per_token(model):
    """Training FLOPs per token: non_embedding_flops_per_token plus lm_head's 6 V C."""
    return non_embedding_flops_per_token(model) + 6 * model.lm_head.weight.numel()
