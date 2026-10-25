"""
model/modeling.py's CausalLM as a transformers model, for inference from the directory inference/export_hf.py writes
(this file is copied there and named in config.json's auto_map, Hugging Face's custom-code convention; torch and
transformers only). vLLM runs it through its Transformers backend (vLLM docs, "Transformers backend"), which honors
_keep_in_fp32_modules_strict, the per-layer sliding windows (layer_types) and final_logit_softcapping with its own
attention, KV cache, torch.compile and CUDA graphs.

Layout as transformers' Qwen3 (modeling_qwen3.py): attention through ALL_ATTENTION_FUNCTIONS with each layer's window
(config.layer_types "sliding_attention" / "full_attention" and sliding_window, as Qwen3 and Gemma 3), so the same code
runs in transformers (eager, sdpa, generate with a cache) and in vLLM. The training forward's precision, where its
kernels keep fp32:
- q/k (model/kernels/qk_row.cuh): per-head RMS norm (eps FLT_EPSILON), fp32 gain, rotation by -theta over the full
  head ((m1, m2) -> (m1 cos + m2 sin, m2 cos - m1 sin), karpathy/nanochat 92d63d4 gpt.py:57-65), angles t theta^(-2i/D)
  in fp64 rounded once to fp32 (CausalLM.init_buffers), one bf16 rounding. transformers' and vLLM's rotary embeddings
  rotate by +theta, so none is used.
- value embeddings (model/kernels/ve_gate.cu): v + 2 sigmoid(n W^T) per head x table[input_ids], the gate's logits in
  fp32, the mix rounded once.
- per layer h = resid_lambda h + x0_lambda x0 in fp32, rounded once; the per-dim and per-layer vectors and the gate
  stay fp32 (_keep_in_fp32_modules_strict).
"""

from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PreTrainedConfig, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class NanochatConfig(PreTrainedConfig):
    """inference/export_hf.py's config.json, every field explicit there: vocab_size, hidden_size, intermediate_size,
    num_hidden_layers, num_attention_heads (= num_key_value_heads), head_dim, max_position_embeddings,
    rope_parameters {"rope_theta"}, sliding_window (keys attended including the query's own), layer_types,
    value_embedding_layers, final_logit_softcapping, rms_norm_eps."""
    model_type = "nanochat"


def norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """model.modeling.norm: parameter-free RMS norm, statistics in fp32."""
    return F.rms_norm(x, (x.size(-1),), eps=eps)


def rotary(positions: torch.Tensor, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin (*positions.shape, head_dim / 2) fp32 of the angles t theta^(-2i/D), computed in fp64."""
    exponent = torch.arange(0, head_dim, 2, dtype=torch.float64, device=positions.device) / head_dim
    freqs = positions.double()[..., None] * theta**-exponent
    return freqs.cos().float(), freqs.sin().float()


def qk_norm_rope(x: torch.Tensor, gain: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                 eps: float) -> torch.Tensor:
    """x (..., H, D) -> like x: norm, gain and -theta rotation in fp32, rounded once; cos, sin (..., D / 2)."""
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
        self.ve_gate = nn.Linear(C, H, bias=False) if layer_idx in config.value_embedding_layers else None
        self.sliding_window = (config.sliding_window if config.layer_types[layer_idx] == "sliding_attention"
                               else None)

    def forward(self, n, value_embeds, position_embeddings, attention_mask, past_key_values, **kwargs):
        """n (B, T, C) the normed stream; value_embeds (B, T, C) this layer's or None."""
        shape = (*n.shape[:-1], -1, self.head_dim)
        q, k, v = (proj(n).view(shape) for proj in (self.q_proj, self.k_proj, self.v_proj))
        if value_embeds is not None:
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
    # export_hf.py writes these fp32 (FP32 there is this list)
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
        logits = self.lm_head(out.last_hidden_state[:, keep])
        if (c := self.config.final_logit_softcapping) is not None:
            logits = c * torch.tanh(logits / c)
        return CausalLMOutputWithPast(logits=logits, past_key_values=out.past_key_values)
