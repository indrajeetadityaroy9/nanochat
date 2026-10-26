"""
The CausalLM: karpathy/nanochat 92d63d4 nanochat/gpt.py (a DecoderLayer applies pre-norm attention and MLP, each added
back; CausalLM's layer loop mixes in x0 and looks up value embeddings) with the wins of its experiment log (dev/LOG.md,
"LOG") and learnable per-dimension QK-norm weights (transformers modeling_qwen3.py; per-head QK norm with learnable
parameters, arXiv 2309.14322 §3.2.5; a fixed scale costs 0.11% of loss in nGPT, arXiv 2410.01131 App. A.9 Table 5).
- Normed token embedding, no positional embedding; RoPE over the full head (92d63d4 gpt.py:57-65).
- Per layer x = resid_lambda x + x0_lambda x0, x0 the normed embedding (LOG 2026-01-11); after layer 0 the mix and the
  next layer's norm run with the previous layer's MLP (model/ops/linear.py relu2_mlp_mix).
- Attention: MHA, head_dim HEAD_DIM (model/config.py); per-head RMS norm, a per-dim weight shared across heads, then
  RoPE (transformers modeling_qwen3.py Qwen3Attention.forward); SSSL windows; value embeddings on alternating layers
  behind a per-head gate 2 sigmoid(.) of the normed stream, zero init (modded-nanogpt record #14's per-layer value
  embeddings, MoVE's LaVE, arXiv 2601.22887 §4.1 Eq. 6; nanochat e85db6b, whose gate read the stream's first 32
  channels); FlashAttention-4 with document masking.
- MLP ReLU^2, hidden 4C (LOG 2026-02-05: SwiGLU worse at d12/d24).
- Gain-free RMS norm, bias-free linears, untied lm_head (modded-nanogpt record #8), plain cross entropy; the
  output-logit cap c tanh(z / c) (ModelConfig.final_logit_softcapping; run.py's c = 15 is 92d63d4's).
Not taken from 92d63d4: its smear and backout, its QK order (RoPE, norm, x 1.2) and its init (init_weights states this
model's).
fp32 parameters, bf16 activations; the GEMM ops multiply by bf16 copies of the weights (GemmLinear), the other ops
round theirs to bf16. Gradient reductions run in no fixed order (FA4's and the embedding backward's atomic
accumulation), so gradients are not bitwise reproducible.
"""

import zlib
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from corpus.mixture import Stage
from model.config import HEAD_DIM, ROPE_THETA, VOCAB_SIZE
from model.ops.cross_entropy import lm_loss, lm_losses
from model.ops.flash_attn import attention
from model.ops.linear import materialize, qkv_proj, relu2_mlp, relu2_mlp_mix

# Width per layer: MobileLLM-350M's deep-and-thin 32 x 960 (arXiv 2402.14905 Table 9)
ASPECT_RATIO = 960 // 32


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    max_position_embeddings: int
    num_hidden_layers: int
    hidden_size: int
    # Output logits capped as c tanh(z / c), c = final_logit_softcapping, in every loss and sample
    # (model/ops/cross_entropy.py, model/kernels/decode.cu); None: uncapped
    final_logit_softcapping: float | None = None


def model_config(*, num_hidden_layers: int, stage: Stage) -> ModelConfig:
    """nanochat's shape rule (92d63d4 scripts/base_train.py:51-52, 131-134): width ASPECT_RATIO x num_hidden_layers,
    rounded up to whole heads."""
    return ModelConfig(max_position_embeddings=stage.seq_len, num_hidden_layers=num_hidden_layers,
                       hidden_size=-(-num_hidden_layers * ASPECT_RATIO // HEAD_DIM) * HEAD_DIM)


def has_ve(layer_idx: int, *, num_hidden_layers: int) -> bool:
    """Alternating layers, the last included (92d63d4 gpt.py:53-55)."""
    return layer_idx % 2 == (num_hidden_layers - 1) % 2


def layer_windows(config: ModelConfig) -> list[int]:
    """Keys before each query, per layer: SSSL tiled (every fourth layer L), the last layer L too (LOG 2026-01-11;
    92d63d4 gpt.py:287-314); L the full context, S = 512, the smallest local window Gemma 3 finds near-lossless
    (arXiv 2503.19786 §5.2, Fig. 4)."""
    n = config.num_hidden_layers
    return [config.max_position_embeddings if i % 4 == 3 or i == n - 1 else 512 for i in range(n)]


def norm(x: Tensor) -> Tensor:
    """RMS norm, statistics in fp32, eps FLT_EPSILON (the convention of model/kernels/qk_prologue.cu and decode.cu)."""
    return F.rms_norm(x, (x.size(-1),), eps=torch.finfo(torch.float32).eps)


class GemmLinear(nn.Linear):
    """A bias-free linear of the decoder layers' GEMM ops (model/ops/linear.py): the ops multiply by weight_bf16, a
    bf16 copy of the fp32 master weight refreshed once per weight change (CausalLM.refresh_gemm_weights), not by a
    cast per micro-batch, and accumulate dW into the master's .grad."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=False)
        self.register_buffer("weight_bf16", torch.empty_like(self.weight, dtype=torch.bfloat16), persistent=False)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        C = config.hidden_size
        self.q_proj = nn.Linear(C, C, bias=False)
        self.k_proj = nn.Linear(C, C, bias=False)
        self.v_proj = nn.Linear(C, C, bias=False)
        # q_proj, k_proj, v_proj as one GEMM (model/ops/linear.py qkv_proj) by their stacked bf16 copies, refreshed as
        # the GemmLinears' (CausalLM.refresh_gemm_weights); dW accumulates into each weight's .grad
        self.register_buffer("qkv_bf16", self.q_proj.weight.new_empty((3 * C, C), dtype=torch.bfloat16),
                             persistent=False)
        self.o_proj = GemmLinear(C, C)
        self.q_norm_weight = nn.Parameter(torch.empty(HEAD_DIM))
        self.k_norm_weight = nn.Parameter(torch.empty(HEAD_DIM))
        self.ve_gate = (nn.Linear(C, C // HEAD_DIM, bias=False)
                        if has_ve(layer_idx, num_hidden_layers=config.num_hidden_layers) else None)

    def qkv_weights(self) -> tuple[nn.Parameter, nn.Parameter, nn.Parameter]:
        return self.q_proj.weight, self.k_proj.weight, self.v_proj.weight

    def forward(self, hidden_states: Tensor, residual: Tensor, input_ids: Tensor, ve_table: Tensor | None, *,
                cos_sin: tuple[Tensor, Tensor], window_size: int, cu_seqlens: Tensor,
                segments: Tensor) -> tuple[Tensor, Tensor]:
        """hidden_states: normed stream; residual: the stream; input_ids: the token ids; ve_table: this layer's value
        embeddings or None -> (residual + o_proj(attention output), its rows' rstd; model/ops/flash_attn.py
        attention, o_proj inside it: its backward takes FA4's dpsum from o_proj's dX epilogue)."""
        qkv = qkv_proj(hidden_states, self.qkv_bf16, *(w.grad for w in self.qkv_weights()))
        # Value residual, per-head gate 2 sigmoid(.) in (0, 2), 1 at zero init; fp32, rounded once
        # (model/ops/ve_gate.py, inside the attention op; its backward visits the tokens in sorted order)
        ve = ((hidden_states, input_ids, ve_table, self.ve_gate.weight, *torch.sort(input_ids.flatten()))
              if ve_table is not None else (None,) * 6)
        return attention(qkv, self.q_norm_weight, self.k_norm_weight, *cos_sin, window_size, cu_seqlens, segments,
                         self.o_proj.weight_bf16, self.o_proj.weight.grad, residual, *ve)[:2]


class MLP(nn.Module):
    """h + down_proj(relu(up_proj(norm(h)))^2) from the stream h and its rows' rstd; norm, activation, residual and
    their derivatives run in QuACK GEMM epilogues (model/ops/linear.py relu2_mlp), and so do the next layer's mix and
    norm when there is one (relu2_mlp_mix)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        C = config.hidden_size
        self.up_proj = GemmLinear(C, 4 * C)
        self.down_proj = GemmLinear(4 * C, C)

    def forward(self, h: Tensor, rstd: Tensor, mix: tuple[Tensor, Tensor, Tensor] | None) -> tuple[Tensor, Tensor]:
        """mix: the next layer's (x0, resid_lambda, x0_lambda), None after the last layer -> (the stream, normed)."""
        weights = (self.up_proj.weight_bf16, self.down_proj.weight_bf16, self.up_proj.weight.grad,
                   self.down_proj.weight.grad)
        if mix is None:
            y = relu2_mlp(h, rstd, *weights)[0]
            return y, norm(y)
        x, n = relu2_mlp_mix(h, rstd, *weights, *mix)[:2]
        return x, n


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, hidden_states: Tensor, normed: Tensor, input_ids: Tensor, ve_table: Tensor | None, *,
                cos_sin: tuple[Tensor, Tensor], window_size: int, cu_seqlens: Tensor, segments: Tensor,
                mix: tuple[Tensor, Tensor, Tensor] | None) -> tuple[Tensor, Tensor]:
        """hidden_states: the stream, normed: norm(hidden_states); mix: MLP.forward's -> (the stream, normed)."""
        h, rstd = self.self_attn(normed, hidden_states, input_ids, ve_table, cos_sin=cos_sin,
                                 window_size=window_size, cu_seqlens=cu_seqlens, segments=segments)
        return self.mlp(h, rstd, mix)


class CausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        """Shapes only (meta device); init_weights sets values."""
        super().__init__()
        self.config = config
        self.window_sizes = layer_windows(config)
        C, n = config.hidden_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, C)
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(n)])
        self.lm_head = nn.Linear(C, VOCAB_SIZE, bias=False)
        self.resid_lambdas = nn.Parameter(torch.empty(n))
        self.x0_lambdas = nn.Parameter(torch.empty(n))
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(VOCAB_SIZE, C) for i in range(n)
                                           if has_ve(i, num_hidden_layers=n)})
        # Rotary tables, set by init_buffers
        for name in ("cos", "sin"):
            self.register_buffer(name, None, persistent=False)
        self.register_load_state_dict_post_hook(lambda module, _: module.refresh_gemm_weights())

    @torch.no_grad()
    def init_weights(self, *, seed: int) -> None:
        """Stream-reading matrices and tables: uniform, std C^-0.5 ("Efficient BackProp" §4.6).
        Zero: o_proj, down_proj (layers start as identity, modded-nanogpt record #5), lm_head (record #8), VE gates
        (gate 1), x0_lambdas. One: resid_lambdas (LOG 2026-01-11), QK-norm weights (transformers Qwen3RMSNorm). Each
        random tensor draws from a generator seeded by `seed` and its name."""
        bound = (3 / self.config.hidden_size) ** 0.5
        zero = ("self_attn.o_proj.weight", "mlp.down_proj.weight", "lm_head.weight", "self_attn.ve_gate.weight",
                "x0_lambdas")
        one = ("q_norm_weight", "k_norm_weight", "resid_lambdas")
        for name, t in self.named_parameters():
            if name.endswith(zero):
                t.zero_()
            elif name.endswith(one):
                t.fill_(1)
            else:
                t.uniform_(-bound, bound,
                           generator=torch.Generator(t.device).manual_seed(seed * 2**32 + zlib.crc32(name.encode())))
        self.init_buffers()
        self.refresh_gemm_weights()

    @torch.no_grad()
    def refresh_gemm_weights(self) -> None:
        """The bf16 copies the GEMM ops multiply by (GemmLinear.weight_bf16, Attention.qkv_bf16) from their fp32
        weights: after init, load and every optimizer step; in place (a compiled graph holds the copies), reallocated
        only where a load assigned weights on another device (train/checkpoint.py load_model's meta model)."""
        linears = [m for m in self.modules() if isinstance(m, GemmLinear)]
        attentions = [m for m in self.modules() if isinstance(m, Attention)]
        for m in linears:
            if m.weight_bf16.device != m.weight.device:
                m.weight_bf16 = torch.empty_like(m.weight, dtype=torch.bfloat16)
        for m in attentions:
            if m.qkv_bf16.device != m.q_proj.weight.device:
                m.qkv_bf16 = torch.empty_like(m.qkv_bf16, device=m.q_proj.weight.device)
        torch._foreach_copy_([m.weight_bf16 for m in linears] + [b for m in attentions for b in m.qkv_bf16.chunk(3)],
                             [m.weight for m in linears] + [w for m in attentions for w in m.qkv_weights()])

    def init_buffers(self) -> None:
        """Non-checkpointed buffers: the fp32 rotary tables cos/sin (max_position_embeddings, HEAD_DIM/2) of
        t ROPE_THETA^(-2i/HEAD_DIM) (92d63d4 gpt.py:270-285), computed in fp64 and rounded once (an fp32 angle errs by
        up to t 2^-24)."""
        device = self.embed_tokens.weight.device
        exponent = torch.arange(0, HEAD_DIM, 2, dtype=torch.float64, device=device) / HEAD_DIM
        t = torch.arange(self.config.max_position_embeddings, dtype=torch.float64, device=device)
        freqs = torch.outer(t, ROPE_THETA ** -exponent)
        self.cos, self.sin = freqs.cos().float(), freqs.sin().float()

    def set_stage(self, stage: Stage) -> None:
        """Switch to a stage's context: recompute windows and rotary tables; weights unchanged."""
        self.config = replace(self.config, max_position_embeddings=stage.seq_len)
        self.window_sizes = layer_windows(self.config)
        self.init_buffers()

    def hidden(self, input_ids: Tensor, cu_seqlens: Tensor, segments: Tensor) -> Tensor:
        """Final normed hidden states (B*T, C) of input_ids (B, T) int32; cu_seqlens: int32 document bounds in
        flattened input_ids, the first segments + 1 of them in use (segments: 0-dim int32 on the host;
        corpus/loader.py PretrainingBatches)."""
        emb = F.embedding(input_ids, self.embed_tokens.weight).to(torch.bfloat16)
        hidden_states = x0 = norm(emb)
        # 0-dim fp32 lambdas x bf16: Inductor reduces their gradients in fp32 without fp32 copies of the mix; layer 0's
        # mix and norm fused by Inductor, every later layer's in the previous layer's down-projection epilogue
        # (MLP.forward), which reads x0 in fp32, as Inductor's fused mix computes it (never rounded to bf16), from one
        # buffer (model/ops/linear.py materialize)
        hidden_states = self.resid_lambdas[0] * hidden_states + self.x0_lambdas[0] * x0
        normed = norm(hidden_states)
        x0 = materialize(norm(emb.float()))
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            ve_table = self.value_embeds[str(i)].weight if str(i) in self.value_embeds else None
            mix = (x0, self.resid_lambdas[i + 1], self.x0_lambdas[i + 1]) if i + 1 < n else None
            hidden_states, normed = layer(hidden_states, normed, input_ids, ve_table, cos_sin=(self.cos, self.sin),
                                          window_size=self.window_sizes[i], cu_seqlens=cu_seqlens, segments=segments,
                                          mix=mix)
        return normed.flatten(0, 1)

    def forward(self, input_ids: Tensor, labels: Tensor, cu_seqlens: Tensor,
                segments: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(mean next-token loss over the non-padding labels fp32; per-position loss (B*T,) fp32, 0 at padding labels;
        per-position logsumexp of the logits (B*T,) fp32), the last two without gradient (model/ops/cross_entropy.py
        lm_loss)."""
        loss, _, _, losses, lse = lm_loss(self.hidden(input_ids, cu_seqlens, segments), self.lm_head.weight,
                                          labels.flatten(), self.config.final_logit_softcapping)
        return loss, losses, lse

    def losses(self, input_ids: Tensor, labels: Tensor, cu_seqlens: Tensor, segments: Tensor) -> Tensor:
        """Per-position next-token loss (B*T,) fp32, 0 at padding labels."""
        return lm_losses(self.hidden(input_ids, cu_seqlens, segments), self.lm_head.weight, labels.flatten(),
                         self.config.final_logit_softcapping)


def num_non_embedding_params(model: CausalLM) -> int:
    """Decoder-layer parameters + lm_head: what the token budget scales with (92d63d4 scripts/base_train.py:263-267)."""
    return sum(p.numel() for p in model.layers.parameters()) + model.lm_head.weight.numel()


def accumulated_weights(model: CausalLM) -> list[nn.Parameter]:
    """The weights whose fp32 .grad the ops of model/ops/linear.py accumulate into in place (q/k/v and the
    GemmLinears o, up, down of every layer): a training backward needs their .grad allocated."""
    return [w for m in model.modules()
            for w in (m.qkv_weights() if isinstance(m, Attention) else (m.weight,) if isinstance(m, GemmLinear) else ())]


def num_matmul_params(model: CausalLM) -> int:
    """The weights the forward multiplies by, the N of PaLM's model FLOPs (metrics.py): every matrix but the embedding
    tables (lookups); norm gains and scalars are elementwise."""
    return sum(p.numel() for m in model.modules() if not isinstance(m, nn.Embedding)
               for p in m.parameters(recurse=False) if p.ndim == 2)


def non_embedding_flops_per_token(model: CausalLM) -> int:
    """DeepSeek LLM's M (arXiv 2401.02954 §3.2 Eq. 2), 72 n_layer d_model² + 12 n_layer d_model l_seq: full attention
    over the model's context l_seq, the count the batch and lr laws' constants were fit with (§3.1-3.2; every Table 3
    model at l_seq 4,096). Windowed layers are counted as full attention too, on the assumption that a model whose
    local windows are near-lossless (arXiv 2503.19786 §5.2) shares its full-attention counterpart's hyperparameters."""
    n, d = model.config.num_hidden_layers, model.config.hidden_size
    return 72 * n * d * d + 12 * n * d * model.config.max_position_embeddings
