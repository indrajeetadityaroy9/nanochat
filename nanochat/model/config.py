"""
The model's architecture: GPTConfig and its derivation from the depth (gpt_config), the token-indexed tables' padded row
count (padded_vocab), and the architecture constants every model module shares (dev/D20_SPEC.md "Model").
"""

from dataclasses import dataclass

import torch

from nanochat.data.tokenizer import SPECIAL_TOKENS
from nanochat.model.engram import EngramConfig, engram_config

# The head's logit soft cap: logits = SOFTCAP tanh(z / SOFTCAP) (karpathy/nanochat 92d63d4 gpt.py:511-515)
SOFTCAP = 15
# Keys before each query on the short-window ('S') layers (attention.layer_windows)
SHORT_WINDOW = 512
# The rotary frequencies' range: 1 rad/token down to 1/ROPE_BASE (karpathy/nanochat 92d63d4 gpt.py:270-285: base 1e5)
ROPE_BASE = 100000


def padded_vocab(vocab_size):
    """Rows of the token-indexed tables (wte, value embeddings, lm_head) for a vocabulary of vocab_size ids: the next
    multiple of 8, so that a bf16 row of the (tokens, rows) logits spans a multiple of 16 bytes, the global-stride
    alignment of the TMA descriptors QuACK's lm-head GEMMs read the logits through (CUDA Driver API
    cuTensorMapEncodeTiled: globalStrides multiple of 16). The cross-entropy excludes the padded columns (softcap_ce)."""
    align = 16 // torch.bfloat16.itemsize
    return -(-vocab_size // align) * align


@dataclass
class GPTConfig:
    sequence_len: int
    vocab_size: int  # the tokenizer's ids (specials included); the tables have padded_vocab(vocab_size) rows
    n_layer: int
    n_head: int  # number of query heads
    n_kv_head: int  # number of key/value heads (= n_head: XSA pairs each query head with its own value head)
    n_embd: int
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (SHORT_WINDOW keys)
    window_pattern: str
    engram_rows: int  # rows per Engram sub-table (engram.engram_rows of the run's training tokens)
    # Engram (arXiv 2601.07372 v2; model/engram.py): a learned memory of hashed n-grams of token ids added to the
    # residual stream before each block of engram.layers, each layer with its own table; derived from the architecture
    # (engram_config; BOS = the first special token, data/tokenizer.py SPECIAL_TOKENS; a checkpoint's stored value is
    # replaced by the derivation)
    engram: EngramConfig | None = None

    def __post_init__(self):
        self.engram = engram_config(self.n_layer, self.n_embd, self.vocab_size,
                                    self.vocab_size - len(SPECIAL_TOKENS), self.engram_rows)


def gpt_config(depth, vocab_size, sequence_len, engram_rows):
    """The architecture of dev/D20_SPEC.md at a depth: nanochat's shape rule (karpathy/nanochat 92d63d4
    scripts/base_train.py:51-52, 131-134: width = aspect ratio 64 x depth rounded up to whole heads of head_dim 128),
    one key/value head per query head, windows SSSL (92d63d4 base_train.py:54)."""
    head_dim = 128
    n_embd = -(-depth * 64 // head_dim) * head_dim
    n_head = n_embd // head_dim
    return GPTConfig(sequence_len=sequence_len, vocab_size=vocab_size, n_layer=depth, n_head=n_head, n_kv_head=n_head,
                     n_embd=n_embd, window_pattern="SSSL", engram_rows=engram_rows)


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included; 92d63d4 gpt.py:53)."""
    return layer_idx % 2 == (n_layer - 1) % 2
