"""
The GPT: karpathy/nanochat 92d63d4 nanochat/gpt.py (a Block applies pre-norm attention and MLP, each added back; GPT's
layer loop mixes in x0 and looks up value embeddings) with the wins of its experiment log (dev/LOG.md, "LOG") and
learnable per-dimension QK-norm gains (transformers modeling_qwen3.py; per-head QK norm with learnable parameters,
arXiv 2309.14322 §3.2.5; a fixed scale costs 0.11% of loss in nGPT, arXiv 2410.01131 App. A.9 Table 5).
- Normed token embedding, no positional embedding; RoPE over the full head (92d63d4 gpt.py:57-65).
- Per layer x = resid_lambda x + x0_lambda x0, x0 the normed embedding (LOG 2026-01-11).
- Attention: MHA, head_dim config.HEAD_DIM; per-head RMS norm, a per-dim gain shared across heads, then RoPE
  (transformers modeling_qwen3.py Qwen3Attention.forward); SSSL windows; value embeddings on alternating layers behind a
  per-head gate 2 sigmoid(.) of the normed stream, zero init (modded-nanogpt record #14's per-layer value embeddings,
  MoVE's LaVE, arXiv 2601.22887 §4.1 Eq. 6; nanochat e85db6b, whose gate read the stream's first 32 channels);
  FlashAttention-4 with document masking.
- MLP ReLU^2, hidden 4C (LOG 2026-02-05: SwiGLU worse at d12/d24).
- Gain-free RMS norm, bias-free linears, untied lm_head (modded-nanogpt record #8), plain cross entropy.
Not taken from 92d63d4: its smear, backout and logit softcap, its QK order (RoPE, norm, x 1.2) and its init
(init_weights states this model's).
fp32 parameters, bf16 activations; the ops round weights to bf16. Gradient reductions use the fastest kernels, not a
fixed order (FA4's and the embedding backward's atomics: faster per step).
"""

import zlib
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanochat.config import HEAD_DIM, ROPE_BASE, VOCAB_SIZE
from nanochat.corpus.mixture import Stage
from nanochat.ops.cross_entropy import lm_loss, lm_losses
from nanochat.ops.flash_attn import flash_attn
from nanochat.ops.linear import linear, relu2_mlp
from nanochat.ops.qk_prologue import qk_norm_rope

# Width per layer: MobileLLM-350M's deep-and-thin 32 x 960 (arXiv 2402.14905 §2.2.2, Fig. 4)
ASPECT_RATIO = 960 // 32


@dataclass(frozen=True, kw_only=True)
class GPTConfig:
    sequence_len: int
    n_layer: int
    n_embd: int


def gpt_config(*, depth: int, stage: Stage) -> GPTConfig:
    """nanochat's shape rule (92d63d4 scripts/base_train.py:51-52, 131-134): width ASPECT_RATIO x depth, rounded up to
    whole heads."""
    return GPTConfig(sequence_len=stage.seq_len, n_layer=depth,
                     n_embd=-(-depth * ASPECT_RATIO // HEAD_DIM) * HEAD_DIM)


def has_ve(layer_idx: int, *, n_layer: int) -> bool:
    """Alternating layers, the last included (92d63d4 gpt.py:53-55)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def layer_windows(config: GPTConfig) -> list[int]:
    """Keys before each query, per layer: SSSL tiled (every fourth layer L), the last layer L too (LOG 2026-01-11;
    92d63d4 gpt.py:287-314); L the full context, S = 512, the smallest local window Gemma 3 finds near-lossless
    (arXiv 2503.19786 §5.2, Fig. 4)."""
    return [config.sequence_len if i % 4 == 3 or i == config.n_layer - 1 else 512 for i in range(config.n_layer)]


def norm(x: Tensor) -> Tensor:
    """RMS norm, statistics in fp32, eps FLT_EPSILON (the convention of csrc/qk_prologue.cu and csrc/decode.cu)."""
    return F.rms_norm(x, (x.size(-1),), eps=torch.finfo(torch.float32).eps)


class Attention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        C = config.n_embd
        self.c_q = nn.Linear(C, C, bias=False)
        self.c_k = nn.Linear(C, C, bias=False)
        self.c_v = nn.Linear(C, C, bias=False)
        self.c_o = nn.Linear(C, C, bias=False)
        self.q_gain = nn.Parameter(torch.empty(HEAD_DIM))
        self.k_gain = nn.Parameter(torch.empty(HEAD_DIM))
        self.ve_gate = nn.Linear(C, C // HEAD_DIM, bias=False) if has_ve(layer_idx, n_layer=config.n_layer) else None

    def forward(self, n: Tensor, ve: Tensor | None, *, cos_sin: tuple[Tensor, Tensor], window_size: int,
                cu_seqlens: Tensor, segments: Tensor) -> Tensor:
        """n: normed stream; ve: this layer's value embeddings or None."""
        B, T, C = n.size()
        q, k, v = (linear(n, w).view(B, T, -1, HEAD_DIM) for w in (self.c_q.weight, self.c_k.weight, self.c_v.weight))
        # Value residual, per-head gate 2 sigmoid(.) in (0, 2), 1 at zero init; fp32, rounded once. Broadcast-sum gate
        # fused by Inductor: faster than separate fp32 GEMMs
        if ve is not None:
            gate = 2 * torch.sigmoid((n.float().unsqueeze(-2) * self.ve_gate.weight).sum(-1))
            v = (v.float() + gate.unsqueeze(-1) * ve.view_as(v).float()).to(v.dtype)
        q, k = qk_norm_rope(q, k, self.q_gain, self.k_gain, *cos_sin)
        return linear(flash_attn(q, k, v, window_size, cu_seqlens, segments)[0].view(B, T, C), self.c_o.weight)


class MLP(nn.Module):
    """c_proj(relu(c_fc(x))^2); activation and derivative run in QuACK GEMM epilogues (ops/linear.py)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        C = config.n_embd
        self.c_fc = nn.Linear(C, 4 * C, bias=False)
        self.c_proj = nn.Linear(4 * C, C, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return relu2_mlp(x, self.c_fc.weight, self.c_proj.weight)[0]


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = Attention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, ve: Tensor | None, *, cos_sin: tuple[Tensor, Tensor], window_size: int,
                cu_seqlens: Tensor, segments: Tensor) -> Tensor:
        x = x + self.attn(norm(x), ve, cos_sin=cos_sin, window_size=window_size, cu_seqlens=cu_seqlens,
                          segments=segments)
        return x + self.mlp(norm(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        """Shapes only (meta device); init_weights sets values."""
        super().__init__()
        self.config = config
        self.window_sizes = layer_windows(config)
        C = config.n_embd
        self.wte = nn.Embedding(VOCAB_SIZE, C)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.lm_head = nn.Linear(C, VOCAB_SIZE, bias=False)
        self.resid_lambdas = nn.Parameter(torch.empty(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.empty(config.n_layer))
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(VOCAB_SIZE, C) for i in range(config.n_layer)
                                           if has_ve(i, n_layer=config.n_layer)})
        # Rotary tables, set by init_buffers
        for name in ("cos", "sin"):
            self.register_buffer(name, None, persistent=False)

    @torch.no_grad()
    def init_weights(self, *, seed: int) -> None:
        """Stream-reading matrices and tables: uniform, std C^-0.5 ("Efficient BackProp" §4.6).
        Zero: c_o, c_proj (blocks start as identity, modded-nanogpt record #5), lm_head (record #8), VE gates (gate 1),
        x0_lambdas. One: resid_lambdas (LOG 2026-01-11), gains (transformers Qwen3RMSNorm). Each random tensor draws
        from a generator seeded by `seed` and its name."""
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

    def init_buffers(self) -> None:
        """Non-checkpointed buffers: the fp32 rotary tables cos/sin (sequence_len, HEAD_DIM/2) of
        t ROPE_BASE^(-2i/HEAD_DIM) (92d63d4 gpt.py:270-285), computed in fp64 and rounded once (an fp32 angle errs by
        up to t 2^-24)."""
        device = self.wte.weight.device
        exponent = torch.arange(0, HEAD_DIM, 2, dtype=torch.float64, device=device) / HEAD_DIM
        t = torch.arange(self.config.sequence_len, dtype=torch.float64, device=device)
        freqs = torch.outer(t, ROPE_BASE ** -exponent)
        self.cos, self.sin = freqs.cos().float(), freqs.sin().float()

    def set_stage(self, stage: Stage) -> None:
        """Switch to a stage's context: recompute windows and rotary tables; weights unchanged."""
        self.config = replace(self.config, sequence_len=stage.seq_len)
        self.window_sizes = layer_windows(self.config)
        self.init_buffers()

    def hidden(self, idx: Tensor, cu_seqlens: Tensor, segments: Tensor) -> Tensor:
        """Final normed hidden states (B*T, C) of idx (B, T); cu_seqlens: int32 document bounds in flattened idx, the
        first segments + 1 of them in use (segments: 0-dim int32 on the host; corpus/loader.py PretrainingBatches)."""
        x = x0 = norm(F.embedding(idx, self.wte.weight).to(torch.bfloat16))
        for i, block in enumerate(self.blocks):
            # 0-dim fp32 lambdas x bf16: Inductor reduces their gradients in fp32 without fp32 copies of the mix
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = (F.embedding(idx, self.value_embeds[str(i)].weight).to(torch.bfloat16) if str(i) in self.value_embeds
                  else None)
            x = block(x, ve, cos_sin=(self.cos, self.sin), window_size=self.window_sizes[i], cu_seqlens=cu_seqlens,
                      segments=segments)
        return norm(x).flatten(0, 1)

    def forward(self, idx: Tensor, targets: Tensor, cu_seqlens: Tensor, segments: Tensor) -> Tensor:
        """Mean next-token loss over the non-padding targets, fp32."""
        return lm_loss(self.hidden(idx, cu_seqlens, segments), self.lm_head.weight, targets.flatten())[0]

    def losses(self, idx: Tensor, targets: Tensor, cu_seqlens: Tensor, segments: Tensor) -> Tensor:
        """Per-position next-token loss (B*T,) fp32, 0 at padding targets."""
        return lm_losses(self.hidden(idx, cu_seqlens, segments), self.lm_head.weight, targets.flatten())


def scaling_params(model: GPT) -> int:
    """Block parameters + lm_head: what the token budget scales with (92d63d4 scripts/base_train.py:263-267)."""
    return sum(p.numel() for p in model.blocks.parameters()) + model.lm_head.weight.numel()


def non_embedding_flops_per_token(model: GPT) -> int:
    """DeepSeek LLM's M (arXiv 2401.02954 §3.2 Eq. 2), 72 n_layer d_model² + 12 n_layer d_model l_seq: full attention
    over the model's context l_seq, the count the batch and lr laws' constants were fit with (§3.1-3.2; every Table 3
    model at l_seq 4,096), for windowed layers too, as a near-lossless windowed model takes its full-attention twin's
    hyperparameters [inference]."""
    n, d = model.config.n_layer, model.config.n_embd
    return 72 * n * d * d + 12 * n * d * model.config.sequence_len
