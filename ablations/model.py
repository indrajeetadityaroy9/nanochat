"""
Production's GPT (nanochat/model/gpt.py) with the architecture experiments' alternatives (harness.Run), each off in
production; with all off every method is GPT's:
- mlp "swiglu": SwiGLU, hidden 8C/3 (ablations/swiglu.py) in place of ReLU^2.
- value_embeds "none": no value-embedding tables and gates.
- residual "identity": resid_lambdas and x0_lambdas frozen at their init, 1 and 0 (a plain residual stream);
  "attn_res": Block Attention Residuals (Chen et al., arXiv 2603.15031 §3.1-3.2, Eqs. 2-6) in place of the scalars: the
  sources are the embedding, the completed blocks (each the sum of 2 n_layer / ATTN_RES_BLOCKS consecutive sublayers'
  outputs) and the current block's partial sum; a sublayer reads depth_attention over them, the output over the blocks.
- qk_gains "fixed": the q and k gains frozen at their init, 1 (QK norm without Qwen3's learned gains).
- windows "full": every layer attends over the whole context (production: SSSL, gpt.layer_windows).
- rope "staged": RoPE base 10^4 in the file-level stage, config.ROPE_BASE (10^6) in the repository stage, as Code
  Llama's long-context stage raises it (arXiv 2308.12950 §2.4) (production: ROPE_BASE in both).
- key_offset "on": on the full-context layers, p-RoPE (Barbero et al., arXiv 2410.06205 §6: the lower half of the
  rotation frequencies left unrotated, modded-nanogpt's half-truncation) and the partial key offset (modded-nanogpt
  record #49): a key's unrotated dims are the previous token's, so one layer can match a token and read what followed
  it; a segment's first token keeps its own.
"""
import torch
import torch.nn.functional as F
from torch import nn

from ablations.depth_attention import ATTN_RES_BLOCKS, depth_attention
from ablations.swiglu import SwiGLU
from nanochat.config import REPO_STAGE, ROPE_BASE
from nanochat.model.gpt import GPT, norm
from nanochat.ops.flash_attn import flash_attn
from nanochat.ops.linear import linear


class Candidate(GPT):
    def __init__(self, config, *, mlp: str, value_embeds: str, residual: str, qk_gains: str, windows: str, rope: str,
                 key_offset: str):
        """Shapes only (meta device)."""
        super().__init__(config)
        self.residual, self.windows, self.rope, self.key_offset = residual, windows, rope, key_offset
        C = config.n_embd
        if mlp == "swiglu":
            for block in self.blocks:
                block.mlp = SwiGLU(config)
        if value_embeds == "none":
            self.value_embeds = nn.ModuleDict()
            for block in self.blocks:
                block.attn.ve_gate = None
        if residual == "identity":
            self.resid_lambdas.requires_grad_(False)
            self.x0_lambdas.requires_grad_(False)
        if residual == "attn_res":
            del self.resid_lambdas, self.x0_lambdas
            for block in self.blocks:
                block.attn.depth_query = nn.Parameter(torch.empty(C))
                block.mlp.depth_query = nn.Parameter(torch.empty(C))
            self.depth_query = nn.Parameter(torch.empty(C))
        if qk_gains == "fixed":
            for block in self.blocks:
                block.attn.q_gain.requires_grad_(False)
                block.attn.k_gain.requires_grad_(False)
        if windows == "full":
            self.window_sizes = [config.sequence_len] * config.n_layer
        for name in ("cos_p", "sin_p"):
            self.register_buffer(name, None, persistent=False)

    @torch.no_grad()
    def init_weights(self, *, seed):
        """GPT's, then the depth queries zero: uniform weights over the sources at init (arXiv 2603.15031 §3.2)."""
        super().init_weights(seed=seed)
        for name, t in self.named_parameters():
            if name.endswith("depth_query"):
                t.zero_()

    def init_buffers(self):
        """GPT's rotary tables; with the key offset, the p-RoPE tables."""
        super().init_buffers()
        if self.key_offset == "on":
            half = self.cos.size(-1)
            stationary = torch.arange(half, device=self.cos.device) >= half // 2
            self.cos_p, self.sin_p = self.cos.masked_fill(stationary, 1), self.sin.masked_fill(stationary, 0)

    def set_stage(self, stage):
        if self.rope == "staged":
            self.config.rope_base = ROPE_BASE if stage is REPO_STAGE else 10_000  # Llama 2's base, Code Llama's start
        super().set_stage(stage)
        if self.windows == "full":
            self.window_sizes = [self.config.sequence_len] * self.config.n_layer

    def hidden(self, idx, cu_seqlens, kv_cache):
        if self.residual != "attn_res" and self.key_offset == "off":
            return super().hidden(idx, cu_seqlens, kv_cache)
        B, T = idx.shape
        C = self.config.n_embd
        x = norm(F.embedding(idx, self.wte.weight).to(torch.bfloat16))
        x0 = x
        # the segments' first tokens (cu_seqlens' offsets; its trailing repeats of B*T are none)
        starts = (torch.isin(torch.arange(B * T, dtype=cu_seqlens.dtype, device=idx.device), cu_seqlens).view(B, T)
                  if self.key_offset == "on" else None)
        blocks, partial = [x0], None
        per_block = 2 * self.config.n_layer // ATTN_RES_BLOCKS
        for i, block in enumerate(self.blocks):
            if self.residual != "attn_res":
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = (F.embedding(idx, self.value_embeds[str(i)].weight).to(torch.bfloat16) if str(i) in self.value_embeds
                  else None)

            def attention(n, i=i, block=block, ve=ve):
                if self.key_offset == "on" and self.window_sizes[i] == self.config.sequence_len:
                    q, k, v = block.attn.qkv(n, ve, cos_sin=(self.cos_p, self.sin_p))
                    D = k.size(-1)
                    stationary = torch.arange(D, device=k.device) % (D // 2) >= D // 4  # rotation pairs (d, d + D/2)
                    previous = torch.roll(k.flatten(0, 1), 1, 0).view_as(k)
                    k = torch.where(stationary & ~starts[..., None, None], previous, k)
                    return linear(flash_attn(q, k, v, self.window_sizes[i], cu_seqlens)[0].view(B, T, C),
                                  block.attn.c_o.weight)
                return block.attn(n, ve, cos_sin=(self.cos, self.sin), window_size=self.window_sizes[i],
                                  kv_cache=kv_cache, cu_seqlens=cu_seqlens)

            for j, (module, sublayer) in enumerate(((block.attn, attention), (block.mlp, block.mlp))):
                if self.residual == "attn_res":
                    out = sublayer(norm(depth_attention(blocks if partial is None else [*blocks, partial],
                                                        module.depth_query)[0]))
                    partial = out if partial is None else partial + out
                    if (2 * i + j + 1) % per_block == 0:
                        blocks, partial = [*blocks, partial], None
                else:
                    x = x + sublayer(norm(x))
        if self.residual == "attn_res":
            x = depth_attention(blocks, self.depth_query)[0]
        return norm(x).flatten(0, 1)
