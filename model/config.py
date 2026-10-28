"""The model's fixed dimensions, shared by the tokenizer, the model and its CUDA kernels (torch-free)."""

# Ids including special tokens: 2^15 (arXiv 2402.01035 §3.2; arXiv 2402.19173 §6.2)
VOCAB_SIZE = 2**15
# Attention head width (head size independent of the head count: arXiv 2002.07028 Theorem 1); a compile-time constant
# of model/kernels (native.py)
HEAD_DIM = 128
# RoPE base, both stages: above the 32K context's lower bound 6.4e5 (arXiv 2405.14591 Table 2)
ROPE_THETA = 1_000_000
