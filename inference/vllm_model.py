"""
model/modeling.py's CausalLM as an out-of-tree vLLM model (vLLM docs, "Registering a model": ModelRegistry and the
vllm.general_plugins entry point; the structure of nanosweb/nanoswe-vllm, which serves the same family), reading the
directory inference/export_hf.py writes; the config and the forward's math are inference/modeling_nanochat.py's.
vLLM's layers carry the work: fused QKV (QKVParallelLinear), paged attention with a per-layer sliding window (Attention
per_layer_sliding_window, as Gemma 3; vLLM's hybrid KV-cache manager keeps a sliding layer's last window), relu^2
(ReLUSquaredActivation, as Nemotron-H), the optional logit cap (LogitsProcessor soft_cap, as Gemma 2), torch.compile
and CUDA graphs (support_torch_compile). The q/k prologue, value-embedding mix and residual mix are PyTorch expressions
at the training kernels' precision, fused by vLLM's torch.compile. One GPU (tensor parallel size 1): the value
embeddings and their gates are not sharded.

LinesStop: a logits processor (vLLM's model runner V2 interface) ending a request once its output holds
extra_args["lines"] line breaks, eval/code_eval.py LinesStoppingCriteria's rule for FIM middles cut to their
reference's lines: from then on only BOS (the end of a middle) is allowed.
"""

import torch
from tokenizers import Tokenizer
from torch import nn
from transformers import AutoConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import ReLUSquaredActivation
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (ColumnParallelLinear, QKVParallelLinear, ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.sample.logits_processor import interface

from inference.modeling_nanochat import NanochatConfig, mix, norm, qk_norm_rope, rotary, ve_mix


class NanochatAttention(nn.Module):
    def __init__(self, config: NanochatConfig, layer_idx: int, *, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        C, D, H = config.hidden_size, config.head_dim, config.num_attention_heads
        self.head_dim, self.eps = D, config.rms_norm_eps
        self.qkv_proj = QKVParallelLinear(C, D, H, H, bias=False, prefix=f"{prefix}.qkv_proj")
        self.o_proj = RowParallelLinear(H * D, C, bias=False, prefix=f"{prefix}.o_proj")
        self.q_norm_weight = nn.Parameter(torch.empty(D, dtype=torch.float32), requires_grad=False)
        self.k_norm_weight = nn.Parameter(torch.empty(D, dtype=torch.float32), requires_grad=False)
        self.ve_gate = (ReplicatedLinear(C, H, bias=False, params_dtype=torch.float32, prefix=f"{prefix}.ve_gate")
                        if layer_idx in config.value_embedding_layers else None)
        sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.attn = Attention(H, D, D**-0.5, num_kv_heads=H, cache_config=vllm_config.cache_config,
                              per_layer_sliding_window=config.sliding_window if sliding else None,
                              prefix=f"{prefix}.attn")

    def forward(self, n: torch.Tensor, ve: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """n (N, C) the normed stream; ve (N, C) this layer's value embeddings of the tokens or None."""
        q, k, v = (t.unflatten(-1, (-1, self.head_dim)) for t in self.qkv_proj(n)[0].chunk(3, dim=-1))
        if ve is not None:
            v = ve_mix(n, v, ve, self.ve_gate.weight)
        q = qk_norm_rope(q, self.q_norm_weight, cos, sin, self.eps)
        k = qk_norm_rope(k, self.k_norm_weight, cos, sin, self.eps)
        return self.o_proj(self.attn(q.flatten(1), k.flatten(1), v.flatten(1)))[0]


class NanochatMLP(nn.Module):
    def __init__(self, config: NanochatConfig, *, prefix: str) -> None:
        super().__init__()
        self.up_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False,
                                            prefix=f"{prefix}.up_proj")
        self.act = ReLUSquaredActivation()
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False,
                                           prefix=f"{prefix}.down_proj")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)[0]))[0]


class NanochatDecoderLayer(nn.Module):
    def __init__(self, config: NanochatConfig, layer_idx: int, *, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        self.eps = config.rms_norm_eps
        self.self_attn = NanochatAttention(config, layer_idx, vllm_config=vllm_config, prefix=f"{prefix}.self_attn")
        self.mlp = NanochatMLP(config, prefix=f"{prefix}.mlp")

    def forward(self, h: torch.Tensor, ve: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        h = h + self.self_attn(norm(h, self.eps), ve, cos, sin)
        return h + self.mlp(norm(h, self.eps))


@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": 0})
class NanochatModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.eps = config.rms_norm_eps
        n, V, C = config.num_hidden_layers, config.vocab_size, config.hidden_size
        self.embed_tokens = VocabParallelEmbedding(V, C, prefix=f"{prefix}.embed_tokens")
        self.layers = nn.ModuleList([NanochatDecoderLayer(config, i, vllm_config=vllm_config,
                                                          prefix=f"{prefix}.layers.{i}") for i in range(n)])
        self.value_embeds = nn.ModuleDict({str(i): VocabParallelEmbedding(V, C, prefix=f"{prefix}.value_embeds.{i}")
                                           for i in config.value_embedding_layers})
        self.resid_lambdas = nn.Parameter(torch.empty(n, dtype=torch.float32), requires_grad=False)
        self.x0_lambdas = nn.Parameter(torch.empty(n, dtype=torch.float32), requires_grad=False)
        # the rotary tables of every position, fp32
        self.cos: torch.Tensor
        self.sin: torch.Tensor
        cos, sin = rotary(torch.arange(config.max_position_embeddings), config.head_dim,
                          config.rope_parameters["rope_theta"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        h = x0 = norm(self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds, self.eps)
        cos, sin = self.cos[positions], self.sin[positions]
        for i, layer in enumerate(self.layers):
            h = mix(h, x0, self.resid_lambdas[i], self.x0_lambdas[i])
            ve = self.value_embeds[str(i)](input_ids) if str(i) in self.value_embeds else None
            h = layer(h, ve, cos, sin)
        return norm(h, self.eps)


class NanochatForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("NanochatForCausalLM runs on one GPU (tensor_parallel_size 1)")
        config = vllm_config.model_config.hf_config
        self.model = NanochatModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix=maybe_prefix(prefix, "lm_head"))
        self.logits_processor = LogitsProcessor(config.vocab_size, soft_cap=config.final_logit_softcapping)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights) -> set[str]:
        """export_hf.py's tensors by name; q/k/v_proj into qkv_proj's shards."""
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            for shard in ("q", "k", "v"):
                if f".{shard}_proj." in name:
                    name = name.replace(f".{shard}_proj.", ".qkv_proj.")
                    getattr(params[name], "weight_loader")(params[name], tensor, shard)
                    break
            else:
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, tensor)
            loaded.add(name)
        return loaded


class LinesStop(interface.LogitsProcessor):
    """Requests with extra_args {"lines": k}: once their output tokens hold k line breaks, only BOS is allowed."""

    def __init__(self, vllm_config: VllmConfig, req_states: interface.LogitsProcRequestState) -> None:
        self.req_states = req_states
        model = vllm_config.model_config
        tokenizer = Tokenizer.from_file(f"{model.tokenizer}/tokenizer.json")
        self.bos = model.hf_config.bos_token_id
        # line breaks per token; a ByteLevel token string spells byte 0x0A "\u010a"; special tokens (BOS on) none
        breaks = [tokenizer.id_to_token(i).count("\u010a") if i < self.bos else 0 for i in range(req_states.vocab_size)]
        self.breaks = torch.tensor(breaks, dtype=torch.int32, device=req_states.device)
        self.lines = torch.zeros(req_states.max_num_reqs, dtype=torch.int32, device=req_states.device)

    def add_request(self, req_idx: int, sampling_params) -> bool:
        lines = (sampling_params.extra_args or {}).get("lines")
        self.lines[req_idx] = lines or 0
        return lines is not None

    def apply(self, logits: torch.Tensor, ctx: interface.LogitsContext) -> torch.Tensor:
        if not ctx.idx_mapping_np.size:
            return logits
        slots, s = ctx.expanded_idx_mapping, self.req_states
        # each row's output tokens: all_token_ids[slot, prompt_len:total_len]
        width = int(ctx.seq_lens_upper_bound_np.max()) + 1
        col = torch.arange(width, device=logits.device)
        out = (col >= s.prompt_len.gpu[slots, None]) & (col < s.total_len.gpu[slots, None])
        ids = s.all_token_ids.gpu[slots, :width].clamp(0, self.breaks.numel() - 1)
        count = (self.breaks[ids] * out).sum(1)
        lines = self.lines[slots]
        done = (lines > 0) & (count >= lines)
        bos = logits[:, self.bos].clone()
        logits.masked_fill_(done[:, None], float("-inf"))
        logits[:, self.bos] = torch.where(done, 0.0, bos)
        return logits


def register() -> None:
    """NanochatConfig under model_type "nanochat" and NanochatForCausalLM in vLLM's registry (lazily, by name); the
    vllm.general_plugins entry point (docker/Dockerfile.vllm) and in-process users call it. Idempotent."""
    AutoConfig.register("nanochat", NanochatConfig, exist_ok=True)
    ModelRegistry.register_model("NanochatForCausalLM", "inference.vllm_model:NanochatForCausalLM")
