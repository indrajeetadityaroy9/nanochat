"""The model's fixed dimensions, shared by the tokenizer, the model and its CUDA kernels (torch-free)."""

# Ids including special tokens: nanochat's 2^15 (karpathy/nanochat 92d63d4 scripts/tok_train.py:19)
VOCAB_SIZE = 2**15
# Attention head width: nanochat's 128; compiled into model/kernels (native.py), whose register arrays need it at
# compile time
HEAD_DIM = 128
# One RoPE base for both stages: 10^6, above the 32K context's lower bound of arXiv 2405.14591 (Table 2: 6.4e5), large
# from the start as Llama 3 (500,000, arXiv 2407.21783 §3.2); Qwen2.5-Coder instead raises 10^4 to 10^6 at its
# repo-level stage (arXiv 2409.12186 §3.2.2)
ROPE_THETA = 1_000_000
