"""
Engram: DeepSeek's conditional memory of hashed n-grams (arXiv 2601.07372 v2, "Conditional Memory via Scalable Lookup";
engram_demo_v1.py) beside the dense model: its configuration (derived from the model's, engram_config), its table size
rule (engram_rows), the n-gram row ids (engram_ids) and the layer that owns a table and fuses a token's rows into the
residual stream (Engram). Every model has it.

Placement: the tables are ordinary GPU parameters (gathered, differentiated, optimized and checkpointed as every other
parameter). On GB10 CPU and GPU share one LPDDR5x pool, so the paper's host-DRAM offload (§6.4) would only add copies;
the addresses depend on token ids alone (engram_ids), which keeps that offload possible on discrete-GPU machines.

Every Engram layer (config.layers, 0-indexed blocks) has its own table and its own hash. Per token, S =
len(orders) * heads sub-tables (order major, then head), sub-table s of config.moduli[s] rows (a prime: §2.2) holding
rows [offset_s, offset_s + moduli[s]) of the layer's table (offset_s = sum(moduli[:s]), the official demo's
MultiHeadEmbedding, engram_demo_v1.py:305-322) of `dim` fp32 values each; the token reads, in every sub-table, the row
of the n-gram of token ids ending at it (the exact ids, the special tokens included: the paper's tokenizer compression,
§2.2, would merge distinctions code needs, such as indentation depth, line breaks, case and literal contents; this
rests on our own measurement on code, while the paper's §6.2 ablation on general data finds compression among the three
largest contributors). No row (id -1, zeros) where the n-gram reaches before the token row or holds a BOS after its
leftmost token: the leftmost token may be the document's BOS, so no n-gram reaches into a previous document. The model
reads the layer's concatenation x of its S rows and adds Y to the residual stream H before the block (Engram.forward;
paper §2.3, Eq. 3-5):

    [k; v] = W_kv x                          (one (2C, S*dim) projection: W_K, W_V)
    alpha = sigmoid(RMSNorm_h(h) . RMSNorm_k(k) / sqrt(C)),   V~ = alpha v           (Eq. 4)
    Y = SiLU(Conv1D(RMSNorm_v(V~))) + V~,   H <- H + Y                              (Eq. 5)

Conv1D is depthwise and causal over the sequence: kernel ENGRAM_CONV_KERNEL = 4, dilation max(orders), zero-initialized
(Y = V~ at the start); its taps before the row's first token or the token's attention segment's first token (document
masking) read zeros. The projection and the fusion are the custom ops of nanochat.ops.engram.
"""

from dataclasses import dataclass

import numpy as np
import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.ops.engram import ENGRAM_CONV_KERNEL, engram_fuse, engram_kv, engram_taps

# The configuration's constants, each from the paper (arXiv 2601.07372 v2, Table 5 in Appendix A: Engram-27B and -40B)
ENGRAM_ORDERS = (2, 3)        # "Engram N-gram [2,3]"
ENGRAM_HEADS = 8              # "Engram Num Head 8": hash heads per order
# Rows per sub-table: one per ENGRAM_TOKENS_PER_ROW training tokens, the endpoint of DeepSeek's infinite-memory sweep
# (§3.2: loss improves log-linearly, unsaturated, up to 1e7 slots at 100B training tokens). A slot = one d_mem-wide
# entry = one row of every sub-table [inferred: §3.2 13e9 params / 1e7 slots ~ 1,300 ~ d_mem 1280; Table 5 vocab size
# x 1280 x 2 layers = 5.7B / 18.5B]; §3.2 does not state its number of Engram layers (with 2 layers the endpoint would
# be 2e4 tokens per row per layer)
ENGRAM_TOKENS_PER_ROW = 1e4
# The hash multipliers' generator seed: the official demo's EngramConfig.seed (engram_demo_v1.py:47), offset per layer
# by its PRIME_1 x layer id (engram_demo_v1.py:217-222)
ENGRAM_HASH_SEED, ENGRAM_HASH_LAYER_PRIME = 0, 10007


@dataclass(frozen=True)
class EngramConfig:
    """The Engram layers of a model (engram_config)."""
    layers: tuple[int, ...]   # the Engram layers: block indices, each with its own table
    orders: tuple[int, ...]   # n-gram lengths
    heads: int                # hash heads per order
    dim: int                  # d_e: fp32 values per row
    moduli: tuple[int, ...]   # S distinct primes, sub-table s's rows (and its hash modulus), order major then head
    bos: int                  # BOS id: n-grams do not reach past it
    # multipliers[l][i][k][j]: odd multiplier of the n-gram's token j back (j = 0 the newest) in Engram layer layers[l]'s
    # sub-table (orders[i], k)
    multipliers: tuple[tuple[tuple[tuple[int, ...], ...], ...], ...]

    @property
    def slots(self):
        """S: sub-tables per layer, rows read per token."""
        return len(self.orders) * self.heads


def engram_config(n_layer, n_embd, vocab_size, bos, rows):
    """The Engram of a model of n_layer blocks of width n_embd over vocab_size token ids (BOS = bos) with sub-tables of
    at most rows rows. Paper Table 5 (30 layers, hidden 2560):
    - layers: before the 1-indexed layers 2 and L/2 ([2, 15]), i.e. blocks 1 and L//2 - 1 (one layer where they
      coincide: L <= 4);
    - orders (2, 3), 8 heads per order;
    - d_mem = S * d_e = C/2 (1280 of 2560), so d_e = C/32 (C a multiple of 128: gpt_config).
    Sub-table sizes: distinct primes, as Eq. 1's "prime size M_{n,k}" and the demo's per-head primes
    (engram_demo_v1.py:235-260, sympy): here the S largest primes <= rows, order major then head (the same for every
    layer: each layer has its own table and its own multipliers).
    Hash multipliers: the demo's draw (engram_demo_v1.py:214-230: odd values below int64 max // vocab size, so every
    product token x multiplier fits int64 and the xor of products stays non-negative) from its per-layer seed (seed +
    PRIME_1 x block index), one per (order, head, token offset): every layer hashes independently."""
    layers = tuple(sorted({1, n_layer // 2 - 1}))
    dim = n_embd // 2 // (len(ENGRAM_ORDERS) * ENGRAM_HEADS)
    half_bound = np.iinfo(np.int64).max // vocab_size // 2
    multipliers = []
    for layer in layers:
        draws = np.random.default_rng(ENGRAM_HASH_SEED + ENGRAM_HASH_LAYER_PRIME * layer).integers(
            0, half_bound, size=(len(ENGRAM_ORDERS), ENGRAM_HEADS, max(ENGRAM_ORDERS)), dtype=np.int64)
        multipliers.append(tuple(tuple(tuple(int(2 * r + 1) for r in head[:n]) for head in order)
                                 for n, order in zip(ENGRAM_ORDERS, draws, strict=True)))
    moduli = [rows + 1]
    for _ in range(len(ENGRAM_ORDERS) * ENGRAM_HEADS):
        moduli.append(sympy.prevprime(moduli[-1]))
    return EngramConfig(layers=layers, orders=ENGRAM_ORDERS, heads=ENGRAM_HEADS, dim=dim, moduli=tuple(moduli[1:]),
                        bos=bos, multipliers=tuple(multipliers))


def engram_rows(train_tokens):
    """The rows bound of every (layer, order, head) sub-table of a run of train_tokens training tokens: one row per
    ENGRAM_TOKENS_PER_ROW tokens (every token reads one row of every sub-table)."""
    return round(train_tokens / ENGRAM_TOKENS_PER_ROW)


def engram_ids(tokens, config: EngramConfig, layer):
    """Row ids (..., T, S) int64 in the table of Engram layer `layer` (a block index of config.layers) of token ids
    (..., T), each last-dim row hashed on its own: position i's id in sub-table s = (order n, head k) is offset_s +
    (xor_j t[i - j] m[j]) mod moduli[s] over j = 0..n-1 (the xor-of-products hash of the official demo,
    engram_demo_v1.py:282-294, and modded-nanogpt record #360, track_1_short/ngram_table.py:112-117), -1 where
    i < n - 1 or a BOS is among t[i - n + 2..i]."""
    t = tokens.long()
    longest = max(config.orders)
    back = [t] + [F.pad(t[..., :-j], (j, 0)) for j in range(1, longest)]  # back[j][..., i] = t[i - j] (i >= j)
    pos = torch.arange(t.size(-1), device=t.device)
    clean = [t != config.bos]  # clean[j]: no BOS among t[i - j..i]
    for j in range(1, longest - 1):
        clean.append(clean[-1] & (back[j] != config.bos))
    ids = []
    multipliers = config.multipliers[config.layers.index(layer)]
    for n, heads in zip(config.orders, multipliers, strict=True):
        valid = (pos >= n - 1) & clean[n - 2]
        for m in heads:
            h = back[0] * m[0]
            for j in range(1, n):
                h = h ^ (back[j] * m[j])
            s = len(ids)  # the sub-table
            ids.append(torch.where(valid, h % config.moduli[s] + sum(config.moduli[:s]), -1))
    return torch.stack(ids, dim=-1)


class Engram(nn.Module):
    """One Engram layer: its table and the fusion of its rows of tokens (B, T) into the residual stream h (B, T, C)
    (Eq. 3-5)."""

    def __init__(self, config: EngramConfig, n_embd, layer):
        """layer: the block index this Engram layer precedes (its hash: engram_ids)."""
        super().__init__()
        self.config, self.layer = config, layer
        self.table = nn.Parameter(torch.empty(sum(config.moduli), config.dim))
        self.w_kv = nn.Linear(config.slots * config.dim, 2 * n_embd, bias=False)  # [W_K; W_V]
        self.norm_h = nn.Parameter(torch.empty(n_embd))  # RMSNorm weights of h, k and V~ (DeepSeek's nn.RMSNorm)
        self.norm_k = nn.Parameter(torch.empty(n_embd))
        self.norm_v = nn.Parameter(torch.empty(n_embd))
        self.conv_w = nn.Parameter(torch.empty(n_embd, ENGRAM_CONV_KERNEL))  # depthwise, no bias; PyTorch Conv1d's tap
        self.dilation = max(config.orders)  # order: conv_w[:, k] reads the token (K - 1 - k) * dilation back

    @torch.no_grad()
    def init_weights(self):
        """W_kv (an ANVIL matrix) as modded-nanogpt track_1_short's ANVIL matrices (model/gpt.py:255-256, 291-292:
        uniform, std 0.5 / sqrt(fan_in)); the table 0 (modded-nanogpt record #360's hashed n-gram table,
        track_1_short/ngram_table.py:173-174); the norm weights 1 (nn.RMSNorm's init); the convolution 0 (Table 5 "Engram
        Conv Zero Init"). Y = 0 at the start: the layer is the identity."""
        s = 3**0.5 * 0.5 * self.w_kv.in_features**-0.5
        nn.init.uniform_(self.w_kv.weight, -s, s)
        nn.init.zeros_(self.table)
        for w in (self.norm_h, self.norm_k, self.norm_v):
            nn.init.ones_(w)
        nn.init.zeros_(self.conv_w)

    def rows(self, ids):
        """x (..., S*dim) bf16 of row ids (..., S) (engram_ids): the fp32 rows rounded once to bf16, zeros at id -1."""
        x = F.embedding(ids.clamp(min=0), self.table).masked_fill((ids < 0).unsqueeze(-1), 0)
        return x.to(torch.bfloat16).flatten(-2)

    def forward(self, h, tokens, starts):
        """(H + Y, alpha): alpha the gate (Eq. 4) per token, fp32."""
        kv = engram_kv(self.rows(engram_ids(tokens, self.config, self.layer)), self.w_kv.weight)
        return engram_fuse(h, kv, self.norm_h, self.norm_k, self.norm_v, self.conv_w, starts, self.dilation)

    def taps(self, h, tokens):
        """The convolution's inputs u (B, T, C) of forward(h, tokens, None)."""
        kv = engram_kv(self.rows(engram_ids(tokens, self.config, self.layer)), self.w_kv.weight)
        return engram_taps(h, kv, self.norm_h, self.norm_k, self.norm_v)
