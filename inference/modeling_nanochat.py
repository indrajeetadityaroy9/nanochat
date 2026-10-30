"""model/modeling.py's CausalLM as a transformers model, for the directory inference/export_hf.py writes (auto_map).

Qwen3's layout (transformers modeling_qwen3.py); q/k rotate by -theta (arXiv 2104.09864 §3.2.2,
model/kernels/qk_row.cuh), so no library rotary is used; fp32 where the training forward computes in fp32.
"""

from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PreTrainedConfig, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class NanochatConfig(PreTrainedConfig):
    """inference/export_hf.py's config.json; every field is explicit there."""
    model_type = "nanochat"


def norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """model.modeling.norm: parameter-free RMS norm, statistics in fp32."""
    return F.rms_norm(x, (x.size(-1),), eps=eps)


def rotary(positions: torch.Tensor, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin (*positions.shape, head_dim / 2) fp32 of the angles t theta^(-2i/D), computed in fp64
    (CausalLM.init_buffers)."""
    exponent = torch.arange(0, head_dim, 2, dtype=torch.float64, device=positions.device) / head_dim
    freqs = positions.double()[..., None] * theta**-exponent
    return freqs.cos().float(), freqs.sin().float()


def qk_norm_rope(x: torch.Tensor, gain: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                 eps: float) -> torch.Tensor:
    """x (..., H, D) -> like x: norm, gain and -theta rotation in fp32, rounded once (qk_row.cuh qk_rotate);
    cos, sin (..., D / 2)."""
    m = x.float()
    m = m * torch.rsqrt(m.square().mean(-1, keepdim=True) + eps) * gain
    m1, m2 = m.chunk(2, dim=-1)
    c, s = cos[..., None, :], sin[..., None, :]
    return torch.cat([m1 * c + m2 * s, m2 * c - m1 * s], dim=-1).to(x.dtype)


def eager_attention_forward(module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                            attention_mask: torch.Tensor | None, scaling: float,
                            **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """transformers' attention-interface signature; the mask carries the causal and sliding-window structure."""
    weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask
    weights = weights.float().softmax(-1).to(query.dtype)
    return torch.matmul(weights, value).transpose(1, 2).contiguous(), weights


class Gate(nn.Module):
    """The value-embedding gate's weight W (H, C), outside nn.Linear so that vLLM's Transformers backend keeps it fp32:
    its linear layers' weights carry a weight_loader, which _keep_in_fp32_modules_strict skips."""

    def __init__(self, C: int, H: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(H, C))


class NanochatAttention(nn.Module):
    def __init__(self, config: NanochatConfig, layer_idx: int) -> None:
        super().__init__()
        C, D, H = config.hidden_size, config.head_dim, config.num_attention_heads
        self.config, self.layer_idx, self.head_dim = config, layer_idx, D
        self.scaling, self.is_causal = D**-0.5, True
        self.q_proj = nn.Linear(C, H * D, bias=False)
        self.k_proj = nn.Linear(C, H * D, bias=False)
        self.v_proj = nn.Linear(C, H * D, bias=False)
        self.o_proj = nn.Linear(H * D, C, bias=False)
        self.q_norm_weight = nn.Parameter(torch.ones(D))
        self.k_norm_weight = nn.Parameter(torch.ones(D))
        self.ve_gate = Gate(C, H) if layer_idx in config.value_embedding_layers else None
        self.sliding_window = (config.sliding_window if config.layer_types[layer_idx] == "sliding_attention"
                               else None)

    def forward(self, n, value_embeds, position_embeddings, attention_mask, past_key_values, **kwargs):
        """n (B, T, C) the normed stream; value_embeds (B, T, C) this layer's or None."""
        shape = (*n.shape[:-1], -1, self.head_dim)
        q, k, v = (proj(n).view(shape) for proj in (self.q_proj, self.k_proj, self.v_proj))
        if value_embeds is not None:  # model/kernels/ve_gate.cu: v + 2 sigmoid(W n) ve per head, in fp32
            gate = 2 * torch.sigmoid(F.linear(n.float(), self.ve_gate.weight))
            v = (v.float() + gate[..., None] * value_embeds.view(v.shape).float()).to(v.dtype)
        eps, (cos, sin) = self.config.rms_norm_eps, position_embeddings
        q = qk_norm_rope(q, self.q_norm_weight, cos, sin, eps).transpose(1, 2)
        k = qk_norm_rope(k, self.k_norm_weight, cos, sin, eps).transpose(1, 2)
        v = v.transpose(1, 2)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        attention = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
        out, _ = attention(self, q, k, v, attention_mask, scaling=self.scaling, sliding_window=self.sliding_window,
                           **kwargs)
        return self.o_proj(out.reshape(*shape[:-2], -1).contiguous())


class NanochatMLP(nn.Module):
    def __init__(self, config: NanochatConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.relu(self.up_proj(x)).square())


class NanochatDecoderLayer(nn.Module):
    def __init__(self, config: NanochatConfig, layer_idx: int) -> None:
        super().__init__()
        self.eps = config.rms_norm_eps
        self.self_attn = NanochatAttention(config, layer_idx)
        self.mlp = NanochatMLP(config)

    def forward(self, h, value_embeds, position_embeddings, attention_mask, past_key_values, **kwargs):
        h = h + self.self_attn(norm(h, self.eps), value_embeds, position_embeddings, attention_mask, past_key_values,
                               **kwargs)
        return h + self.mlp(norm(h, self.eps))


class NanochatPreTrainedModel(PreTrainedModel):
    config_class = NanochatConfig
    base_model_prefix = "model"
    _supports_sdpa = True
    _supports_attention_backend = True  # vLLM's Transformers backend runs models that declare it
    # the parameters the training forward reads in fp32; export_hf.py writes these fp32, the rest bf16
    _keep_in_fp32_modules_strict: ClassVar[list[str]] = ["resid_lambdas", "x0_lambdas", "q_norm_weight",
                                                         "k_norm_weight", "ve_gate"]


class NanochatModel(NanochatPreTrainedModel):
    def __init__(self, config: NanochatConfig) -> None:
        super().__init__(config)
        n, V, C = config.num_hidden_layers, config.vocab_size, config.hidden_size
        self.embed_tokens = nn.Embedding(V, C)
        self.layers = nn.ModuleList([NanochatDecoderLayer(config, i) for i in range(n)])
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(V, C) for i in config.value_embedding_layers})
        self.resid_lambdas = nn.Parameter(torch.ones(n))
        self.x0_lambdas = nn.Parameter(torch.zeros(n))
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, **kwargs) -> BaseModelOutputWithPast:
        """input_ids always (the value embeddings read them); inputs_embeds, when given, their embeddings."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = (torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + seen)[None]
        if not isinstance(masks := attention_mask, dict):
            mask_kwargs = {"config": self.config, "inputs_embeds": inputs_embeds, "attention_mask": attention_mask,
                           "past_key_values": past_key_values, "position_ids": position_ids}
            masks = {"full_attention": create_causal_mask(**mask_kwargs),
                     "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs)}
        # per call: a buffer would take the model's dtype
        position_embeddings = rotary(position_ids, self.config.head_dim, self.config.rope_parameters["rope_theta"])
        eps = self.config.rms_norm_eps
        h = x0 = norm(inputs_embeds, eps)
        for i, layer in enumerate(self.layers):
            # CausalLM.hidden's mix, which Inductor computes in fp32 and rounds once
            h = (self.resid_lambdas[i] * h.float() + self.x0_lambdas[i] * x0.float()).to(h.dtype)
            ve = self.value_embeds[str(i)](input_ids) if str(i) in self.value_embeds else None
            h = layer(h, ve, position_embeddings, masks[self.config.layer_types[i]], past_key_values, **kwargs)
        return BaseModelOutputWithPast(last_hidden_state=norm(h, eps),
                                       past_key_values=past_key_values if use_cache else None)


class NanochatForCausalLM(NanochatPreTrainedModel, GenerationMixin):
    def __init__(self, config: NanochatConfig) -> None:
        super().__init__(config)
        self.model = NanochatModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, logits_to_keep=0, **kwargs) -> CausalLMOutputWithPast:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                         past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        keep = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden = out.last_hidden_state[:, keep]
        # the evaluation's head (model/ops/cross_entropy.py lm_losses): the bf16 GEMM's fp32 output (a bf16 product is
        # exact in fp32), the cap in fp32
        logits = F.linear(hidden.float(), self.lm_head.weight.float())
        c = self.config.final_logit_softcapping
        return CausalLMOutputWithPast(logits=c * torch.tanh(logits / c), past_key_values=out.past_key_values)
