"""
GPT model (nanochat, karpathy/nanochat 92d63d4 nanochat/gpt.py, adapted to dev/D20_SPEC.md)
Notable features:
- rotary embeddings on the first half of each head (and no positional embeddings), the key offset on the full-context
  layers
- QK norm
- untied weights for token embedding and lm_head
- SwiGLU MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- one key/value head per query head (XSA pairs them)
- FlashAttention-4 (sm_12x patched) with document masking, XSA at strength 1, a head-wise attention-output gate, MUDD
  dense connections (model/mudd.py), Engram (model/engram.py; dev/D20_SPEC.md "Model")
The optimizer's parameter groups are train/optim.py's.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.model.attention import (
    CausalSelfAttention,
    layer_windows,
    rotary_tables,
    segment_starts,
)
from nanochat.model.config import SOFTCAP, has_ve, padded_vocab
from nanochat.model.engram import Engram
from nanochat.model.layers import Embedding, Linear, norm
from nanochat.model.mlp import MLP
from nanochat.model.mudd import MuddSite, mix, site_sources
from nanochat.ops.softcap_ce import softcap_ce_losses, softcap_lce


def scaling_params(model):
    """The parameters the token budget scales with (nanochat's choice, the cleanest scaling laws): the blocks' matrices
    (the attention-output gates excluded, as MUDD and Engram are) + lm_head."""
    blocks = model.transformer.h
    attn_gates = sum(b.attn.attn_gate.weight.numel() for b in blocks)
    return sum(p.numel() for p in blocks.parameters()) - attn_gates + model.lm_head.weight.numel()


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, key_offset, starts, cu_seqlens, v_mix=None):
        x = x + self.attn(x, ve, cos_sin, window_size, kv_cache, key_offset, starts, cu_seqlens, v_mix)
        return x + self.mlp(x)


class GPT(nn.Module):
    def __init__(self, config):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        self.window_sizes, self.key_offsets = layer_windows(config)
        head_dim = config.n_embd // config.n_head
        # Token-indexed tables have padded_vocab rows; ids >= config.vocab_size never occur as inputs or targets
        V = padded_vocab(config.vocab_size)
        self.transformer = nn.ModuleDict({
            "wte": Embedding(V, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, V, bias=False)
        # Per-layer learnable scalars (nanochat, 92d63d4 gpt.py:236-241, 500; from modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer
        # x0_lambdas: blends initial embedding back in at each layer
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.empty(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.empty(config.n_layer))
        # Smear: mix previous token's embedding into current token (cheap bigram-like info); gate on the first 24 channels
        # (92d63d4 gpt.py:185, 479)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.empty(1))
        # MUDD site 2's prior on block backout_layer's output holds nanochat's backout
        self.backout_layer = config.n_layer // 2
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): Embedding(V, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # XSA (Zhai, arXiv 2603.09078 Eq. 2; modded-nanogpt track_1_short attention.py:224-228) in every layer: each
        # head's output loses its component along the head's value; y and v pair up head by head
        # Engram layers, before each block of engram.layers, each with its own table and hash of idx's n-grams
        self.engram = nn.ModuleDict({str(i): Engram(config.engram, config.n_embd, i) for i in config.engram.layers})
        # MUDD (model/mudd.py): the coefficient MLPs of site 1 (the last block's input: value and residual streams)
        # and site 2 (before the final norm: residual stream)
        self.mudd_sources = site_sources(config.n_layer, [int(i) for i in self.value_embeds])
        self.mudd_in = MuddSite(len(self.mudd_sources[0]), 2, config.n_embd)
        self.mudd_out = MuddSite(len(self.mudd_sources[1]), 1, config.n_embd)
        # rotary embeddings of every position of the context, computed by init_rotary
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity: the ANVIL matrices as modded-nanogpt
        track_1_short (the optimizer's recipe; model/gpt.py:254-261 attention banks, :291-295 MLP bank), the rest as
        nanochat (karpathy/nanochat 92d63d4 gpt.py:205-255; their AdamW recipe), plus the Engram layers (their own init).

        wte (embedding):     normal, std=0.8
        lm_head:             normal, std=0.001
        for each block (uniform, std=0.5/sqrt(n_embd)):
            attn.c_q, c_k, c_v, c_proj (modded's vo_bank draws O as V)
            mlp.c_fc (gate and up rows)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding (92d63d4 gpt.py:220-221)
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        torch.nn.init.zeros_(self.lm_head.weight[self.config.vocab_size:])  # padded rows: never a logit (softcap_ce)

        # Block matrices: modded's uniform with bound sqrt(3) * std, std = 0.5 / sqrt(n_embd) (model/gpt.py:255-256, 291-292)
        n_embd = self.config.n_embd
        s = 3**0.5 * 0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_proj.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)  # model/gpt.py:295

        # Per-layer scalars (92d63d4 gpt.py:236-241): resid init stronger at early layers, weaker at deep layers; x0 init
        # decaying, earlier layers get more input embedding blending
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / (n_layer - 1))
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / (n_layer - 1))

        # Smear scalar and gate (92d63d4 gpt.py:243-246)
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (92d63d4 gpt.py:248-250: uniform like nanochat's c_v, std = 1/sqrt(n_embd))
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -3**0.5 * n_embd**-0.5, 3**0.5 * n_embd**-0.5)

        # Value-embedding gates zero (R82:1179, torch.zeros): 2 * sigmoid(0) = 1 at step 0; attention-output gates zero
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            torch.nn.init.zeros_(block.attn.attn_gate.weight)

        # Engram projections and tables last, so the dense weights are those of the model without them (same seed)
        for m in self.engram.values():
            m.init_weights()
        # MUDD after them (its W1 draws from the generator): site 1 priors 0, site 2 the backout's -0.2 on block
        # backout_layer's output
        self.mudd_in.init_weights(torch.zeros_like(self.mudd_in.b))
        prior = torch.zeros_like(self.mudd_out.b)
        # nanochat's backout at its init (karpathy/nanochat 92d63d4 gpt.py:188, 243-246, 497: x - 0.2 h_{L/2} before the
        # final norm)
        prior[self.mudd_sources[1].index(("h", self.backout_layer))] = -0.2
        self.mudd_out.init_weights(prior)
        self.init_rotary()

    def init_rotary(self):
        """The rotary tables (attention.rotary_tables) on the model's device: non-persistent buffers, so a model built
        from a checkpoint computes them here."""
        self.cos, self.sin = rotary_tables(self.config, self.get_device())

    def get_device(self):
        return self.transformer.wte.weight.device

    def set_loss_targets(self, mtp_depth, bos_token_id, prefix_table):
        """The training loss's target columns (base_train, before training: modded-nanogpt's multi-token prediction and
        prefix-token loss): the next mtp_depth tokens (multi-token prediction), then, if prefix_table
        ((padded_vocab,) of data/tokenizer.py prefix_targets, on the model's device) is given, the next token's prefix
        token. bos_token_id: the first special token; the tokenizer's special tokens are it and every id after it (BOS,
        then the FIM sentinels: data/tokenizer.py)."""
        self.mtp_depth, self.bos_token_id, self.prefix_table = mtp_depth, bos_token_id, prefix_table

    def loss_targets(self, targets):
        """(B*T, K) target columns of the (B, T) next-token targets: column j < mtp_depth the token j positions after the
        next one, -1 past the row's end or after a boundary among the j tokens before it: a special token (BOS or a FIM
        sentinel) or a -1 target (padding, or the loader's masked BOS of the next piece): a target in the next,
        unrelated document or across a FIM section boundary; a special target itself is kept, as column 0 keeps it.
        Then the next token's prefix token, -1 where it has none and for padding (target -1 indexes the table's last
        entry, a special or padding id's -1)."""
        columns = [targets]
        special = (targets >= self.bos_token_id) | (targets < 0)
        crossed = special  # a special token among the j tokens before column j's target
        for j in range(1, self.mtp_depth):
            if j > 1:
                crossed = crossed | F.pad(special[:, j - 1:], (0, j - 1), value=False)
            ahead = F.pad(targets[:, j:], (0, j), value=-1)
            columns.append(ahead.masked_fill(crossed, -1))
        if self.prefix_table is not None:
            columns.append(self.prefix_table[targets])
        return torch.stack(columns, dim=-1).view(-1, len(columns))

    def forward(self, idx, targets=None, loss_weights=None, cu_seqlens=None, kv_cache=None, loss_reduction='mean'):
        """cu_seqlens: the attention segments of the flattened idx (document masking; data/loader.py PretrainingBatches);
        None, every row one segment (a single document, e.g. a prompt: the same attention as plain causal rows); the
        Engram convolutions restart at every segment's first token, as attention (their n-gram rows stop at BOS:
        engram.engram_ids). kv_cache (inference prefill): receives every layer's keys/values and each Engram layer's
        convolution inputs (kv_cache.engram_taps[j], layer j of engram.layers); the head is skipped (returns None: the
        decoder continues from the caches).
        Training (targets, loss_reduction 'mean') returns (loss, column_losses, engram_gate), column_losses (K,) fp32
        detached: each target column's mean loss over its own valid targets (unweighted); engram_gate
        (len(config.engram.layers),) fp32 detached: each Engram layer's mean gate alpha (Eq. 4) over the micro-batch's
        tokens. Evaluation (targets, loss_reduction 'none'): each position's next-token loss (B*T,) fp32."""
        x, starts = self._embed(idx, kv_cache, cu_seqlens)
        x, states = self._trunk(idx, x, kv_cache, starts, cu_seqlens)
        if kv_cache is not None:
            return None
        # MUDD site 2 (its prior: the backout's -0.2 on block backout_layer's output)
        x = mix(self.mudd_out(x), [states[key] for key in self.mudd_sources[1]], x).to(x.dtype)
        x = norm(x)
        # the loss straight from the hidden states, the softcap inside the cross-entropy pass (ops/softcap_ce.py; the
        # fp32 master weight is rounded to bf16 inside, its gradient fp32; the softmax over the first config.vocab_size
        # columns, the padded ones excluded); targets -1 (the loader's padding) are ignored
        x = x.reshape(-1, x.size(-1))
        if loss_reduction == 'mean':
            return *self._training_loss(x, targets, loss_weights), torch.stack(states["engram_gate"])
        return softcap_ce_losses(x, self.lm_head.weight, targets.reshape(-1, 1), SOFTCAP, -1, self.config.vocab_size)[:, 0]

    def _embed(self, idx, kv_cache, cu_seqlens):
        """(x, starts): the normed token embeddings of idx (B, T) with the smear, and the attention segments' first tokens
        ((B, T) bool; None without cu_seqlens), where the smear stops."""
        x = self.transformer.wte(idx)
        x = norm(x)
        # Smear: mix previous token's embedding into current position (cheap bigram info); none for the first token, and
        # with attention segments none for each segment's first token (its predecessor is another document's)
        if kv_cache is not None:
            kv_cache.prev_embedding = x[:, -1:, :]  # the decoder smears its first token with this
        # The smear lambda and gate weight meet the bf16 activations in fp32 (their gradients' rel err vs fp64 5e-7 /
        # 4e-7; as bf16 operands 3e-3 / 5e-3, measured on GB10); the mix rounds once to bf16
        gate = self.smear_lambda * torch.sigmoid(F.linear(x[:, 1:, :self.smear_gate.in_features].float(), self.smear_gate.weight))
        starts = None
        if cu_seqlens is not None:
            starts = segment_starts(cu_seqlens, *idx.shape)
            gate = gate.masked_fill(starts[:, 1:, None], 0)
        x = torch.cat([x[:, :1], (x[:, 1:].float() + gate * x[:, :-1].float()).to(x.dtype)], dim=1)
        return x, starts

    def _trunk(self, idx, x, kv_cache, starts, cu_seqlens):
        """The blocks over the embedded x: (x, states). states: MUDD's sources by mudd.site_sources key (x0, the blocks'
        outputs, the value embeddings) and "engram_gate", the Engram layers' mean gates (0-dim fp32 each)."""
        x0 = x  # the initial normalized embedding for the x0 residual
        states = {"x0": x0, "engram_gate": []}
        last = self.config.n_layer - 1
        for i, block in enumerate(self.transformer.h):
            # 0-dim fp32 lambdas x bf16: Inductor already reduces their gradients in fp32 (rel err vs fp64 5e-8 / 9e-8,
            # as with an explicit fp32 mix, which costs fp32 copies saved for the backward: +880 MiB at d12)
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            if str(i) in self.engram:
                if kv_cache is not None:
                    kv_cache.engram_taps[self.config.engram.layers.index(i)] = self.engram[str(i)].taps(x, idx)
                x, alpha = self.engram[str(i)](x, idx, starts)
                states["engram_gate"].append(alpha.mean())
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            if ve is not None:
                states[("ve", i)] = ve
            v_mix = None
            if i == last:
                # MUDD site 1: the value stream joins this block's values, the residual stream replaces its input
                sources = [states[key] for key in self.mudd_sources[0]]
                coefs = self.mudd_in(x)
                S = len(sources)
                v_mix = mix(coefs[..., :S], sources)
                x = mix(coefs[..., S:], sources, x).to(x.dtype)
            key_offset = self.key_offsets[i]
            x = block(x, ve, (self.cos[i], self.sin[i]), self.window_sizes[i], kv_cache, key_offset,
                      starts if key_offset else None, cu_seqlens, v_mix)
            states[("h", i)] = x
        return x, states

    def _training_loss(self, x, targets, loss_weights):
        """(loss, column_losses) of the normed hidden states x (B*T, C): the target columns of loss_targets weighted by
        loss_weights ((K,) fp32 on the device, a tensor so that a per-step schedule does not recompile)."""
        columns = self.loss_targets(targets)
        loss, _, _, column_losses = softcap_lce(x, self.lm_head.weight, columns, loss_weights, SOFTCAP, -1,
                                                self.config.vocab_size)
        return loss, column_losses
