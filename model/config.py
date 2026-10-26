"""The model's fixed dimensions, shared by the tokenizer, the model and its CUDA kernels (torch-free)."""

# Ids including special tokens: nanochat's 2^15 (karpathy/nanochat 92d63d4 scripts/tok_train.py:19)
VOCAB_SIZE = 2**15
# Attention head width (nanochat 92d63d4); a compile-time constant of model/kernels (native.py)
HEAD_DIM = 128
# RoPE base, both stages: above the 32K context's lower bound 6.4e5 (arXiv 2405.14591 Table 2)
ROPE_THETA = 1_000_000
