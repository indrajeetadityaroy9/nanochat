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
Parameters are fp32 masters; activations bf16: the tables are looked up and rounded once (F.embedding, the cast fused
by Inductor), every matmul runs through the nanochat ops, which round the weights inside. The gradients are reduced by
the fastest kernels, not in a fixed order (FA4's and Inductor's embedding backward add with atomics:
benchmarks/h100.txt, ~1% faster per step than fixed-order reductions). The optimizer's parameter groups are
train/optim.py's.
"""

import zlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.config import VOCAB_SIZE
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
    # The ablation study's switches (experiments/): each component of the mechanism under test, on in the canonical
    # model. Off: value_embeds no value embeddings or gates; x0 no x0 path; resid no residual scalars (1); qk_gain gains
    # fixed at 1; windows every layer full context.
    value_embeds: bool = True
    x0: bool = True
    resid: bool = True
    qk_gain: bool = True
    windows: bool = True


def gpt_config(depth, stage):
    """The architecture at a depth: nanochat's shape rule (92d63d4 scripts/base_train.py:51-52, 131-134: width =
    ASPECT_RATIO x depth rounded up to whole heads of head_dim 128, LOG 2026-01-17; csrc/qk_row.cuh HEAD_DIM), the
    vocabulary config.VOCAB_SIZE (the tokenizer is trained to it); the context and rotary base of a training stage
    (config.Stage)."""
    head_dim = 128
    n_embd = -(-depth * ASPECT_RATIO // head_dim) * head_dim
    return GPTConfig(sequence_len=stage.seq_len, vocab_size=VOCAB_SIZE, rope_base=stage.rope_base, n_layer=depth,
                     n_head=n_embd // head_dim, n_embd=n_embd)


def has_ve(layer_idx, n_layer):
    """Whether layer layer_idx has a value embedding: alternating layers, the last one included (92d63d4 gpt.py:53-55;
    LOG 2026-01-17: alternating layers worked best)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def layer_windows(config):
    """The left window (keys before each query) per layer: the pattern SSSL tiled across layers, the last layer L (LOG
    2026-01-11; 92d63d4 gpt.py:287-314); L the context, S = 512 keys at most the context (Gemma 3 arXiv 2503.19786
    §5.2 Fig. 4: local windows of 512..4096 change perplexity minimally). Every layer L without config.windows."""
    pattern = "SSSL" if config.windows else "L"
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
    one epsilon convention of every RMS norm (here, csrc/qk_prologue.cu and csrc/decode.cu): the fp32 machine epsilon
    FLT_EPSILON = 2^-23 also for bf16 inputs, passed explicitly (F.rms_norm's default is the epsilon of its fp32 opmath
    type; modded-nanogpt track_1_short perf/kernels/qkv_rope.py:8)."""
    return F.rms_norm(x, (x.size(-1),), eps=torch.finfo(torch.float32).eps)


class Attention(nn.Module):
    """Causal self-attention, MHA with heads of 128: the q, k, v projections of the normed stream; on value-embedding
    layers the value residual; per head RMS norm, the q / k gain and RoPE; FlashAttention-4 over each query's window
    within its document; the output projection c_o."""

    def __init__(self, config, layer_idx):
        super().__init__()
        C, H = config.n_embd, config.n_head
        self.layer_idx = layer_idx
        self.n_head = H
        self.c_q = nn.Linear(C, C, bias=False)
        self.c_k = nn.Linear(C, C, bias=False)
        self.c_v = nn.Linear(C, C, bias=False)
        self.c_o = nn.Linear(C, C, bias=False)
        if config.qk_gain:
            self.q_gain = nn.Parameter(torch.empty(C // H))
            self.k_gain = nn.Parameter(torch.empty(C // H))
        else:  # fixed at 1, not trained, not in checkpoints (GPT.init_buffers)
            self.register_buffer("q_gain", None, persistent=False)
            self.register_buffer("k_gain", None, persistent=False)
        self.ve_gate = (nn.Linear(C, H, bias=False) if config.value_embeds and has_ve(layer_idx, config.n_layer)
                        else None)

    def forward(self, n, ve, cos_sin, window_size, kv_cache, cu_seqlens):
        """n: the normed residual stream; ve: this layer's value embeddings (bf16) or None. kv_cache (decoder prefill
        from position 0): receives this layer's keys (after the prologue) and values (after the value-embedding mix)."""
        B, T, C = n.size()
        q, k, v = (linear(n, w).view(B, T, self.n_head, -1) for w in (self.c_q.weight, self.c_k.weight, self.c_v.weight))
        # Value residual: the value embedding mixed in with an input-dependent gate per head, 2 sigmoid(.) in (0, 2),
        # 1 at the zero-init gate weight (unit weight at init); in fp32, rounded once to bf16. The gate's H dot products
        # as a broadcast-sum, which Inductor fuses into reductions over n (fp32 throughout, the weight gradient
        # included) where F.linear ran fp32 GEMMs on CUDA cores: 0.88 vs 1.20 ms per layer and d24 micro-batch,
        # fwd+bwd of gate and mix (benchmarks/h100.txt)
        if ve is not None:
            gate = 2 * torch.sigmoid((n.float().unsqueeze(-2) * self.ve_gate.weight).sum(-1))  # (B, T, H)
            v = (v.float() + gate.unsqueeze(-1) * ve.view_as(v).float()).to(v.dtype)
        # per head RMS norm, the q / k gain, RoPE (csrc/qk_prologue.cu, one pass over memory per direction: its forward
        # and backward take 168 ms of a d24 step where Inductor's fusion of the same ops takes 336 ms,
        # benchmarks/h100.txt)
        q, k = qk_norm_rope(q, k, self.q_gain, self.k_gain, *cos_sin)
        if kv_cache is not None:
            kv_cache.k_cache[self.layer_idx, :, :T], kv_cache.v_cache[self.layer_idx, :, :T] = k, v
        # FlashAttention-4, causal: each query sees itself and `window_size` keys before it
        return linear(flash_attn(q, k, v, window_size, cu_seqlens)[0].view(B, T, C), self.c_o.weight)


class MLP(nn.Module):
    """The ReLU^2 MLP with hidden width 4C, c_proj(relu(c_fc(x))^2), on QuACK's GEMMs (ops/linear.py relu2_mlp: the
    activation in the up-projection GEMM's epilogue, its derivative in the backward GEMM's)."""

    def __init__(self, config):
        super().__init__()
        C = config.n_embd
        self.c_fc = nn.Linear(C, 4 * C, bias=False)
        self.c_proj = nn.Linear(4 * C, C, bias=False)

    def forward(self, x):
        return relu2_mlp(x, self.c_fc.weight, self.c_proj.weight)[0]


class Block(nn.Module):
    """One transformer layer: attention and the MLP, each on the RMS-normed stream and added to it."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = Attention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, cu_seqlens):
        """x: the residual stream; the rest as Attention.forward."""
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache, cu_seqlens)
        return x + self.mlp(norm(x))


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
        self.resid_lambdas = nn.Parameter(torch.empty(config.n_layer)) if config.resid else None
        self.x0_lambdas = nn.Parameter(torch.empty(config.n_layer)) if config.x0 else None
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(V, C) for i in range(config.n_layer)
                                           if config.value_embeds and has_ve(i, config.n_layer)})
        # not in checkpoints (init_buffers): the rotary tables of the context (rotary_tables)
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)

    @torch.no_grad()
    def init_weights(self, seed):
        """Every matrix and table that reads the stream uniform with std C^-0.5 (variance preserving, LeCun et al.
        "Efficient BackProp" §4.6; bound sqrt(3) std, the std of U(-a, a) being a / sqrt(3)): attn.c_q, attn.c_k,
        attn.c_v, mlp.c_fc, wte, value_embeds. Zero: attn.c_o, mlp.c_proj, lm_head (each block the identity at init,
        modded-nanogpt record #5) and attn.ve_gate (gate 1). resid_lambdas 1, x0_lambdas 0 (LOG 2026-01-11), q_gain =
        k_gain = 1. Each random tensor draws from its own generator, seeded by `seed` and the tensor's name, so a model
        without some component starts every other tensor exactly as the full model does (common random numbers across
        an ablation's arms). The non-persistent buffers: init_buffers."""
        bound = (3 / self.config.n_embd) ** 0.5
        zero = ("attn.c_o.weight", "mlp.c_proj.weight", "lm_head.weight", "attn.ve_gate.weight", "x0_lambdas")
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
        """The buffers no checkpoint holds, on the parameters' device: the rotary tables of the context and, without
        config.qk_gain, the q / k gains fixed at 1 (init_weights, set_stage, checkpoint.load_model)."""
        device = self.wte.weight.device
        self.cos, self.sin = rotary_tables(self.config, device)
        if not self.config.qk_gain:
            for block in self.blocks:
                block.attn.q_gain = torch.ones(self.config.n_embd // self.config.n_head, device=device)
                block.attn.k_gain = torch.ones(self.config.n_embd // self.config.n_head, device=device)

    def set_stage(self, stage):
        """Continue at a training stage's context and rotary base (config.Stage; the weights are unchanged): the layers'
        windows and the rotary tables of the new context, on the parameters' device."""
        self.config.sequence_len, self.config.rope_base = stage.seq_len, stage.rope_base
        self.window_sizes = layer_windows(self.config)
        self.init_buffers()

    def hidden(self, idx, cu_seqlens, kv_cache):
        """The final normed hidden states (B*T, C) of idx (B, T) token ids; cu_seqlens: the int32 attention segments of
        the flattened idx (document masking; data/loader.py PretrainingBatches). kv_cache (decoder prefill) receives
        every layer's keys and values."""
        x = norm(F.embedding(idx, self.wte.weight).to(torch.bfloat16))
        x0 = x
        for i, block in enumerate(self.blocks):
            # 0-dim fp32 lambdas x bf16: Inductor reduces their gradients in fp32 (no fp32 copies of the mix saved for
            # the backward, as an explicit fp32 mix would)
            if self.config.resid:
                x = self.resid_lambdas[i] * x
            if self.config.x0:
                x = x + self.x0_lambdas[i] * x0
            ve = (F.embedding(idx, self.value_embeds[str(i)].weight).to(torch.bfloat16) if str(i) in self.value_embeds
                  else None)
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

