"""
Attention windows. Production: SSSL, three of four layers a 512-token window and the last layer full (Gemma 3, arXiv
2503.19786 §5.2: near-lossless; nanochat LOG 2026-01-11), adopted without a study. Alternative: full attention in every
layer, as DeepSeek-Coder, Qwen2.5-Coder and Code Llama attend; it adds about 4.4% to the d24 step (ablations.md §2).

    python -m ablations.windows    (ablations/harness.py; ablations.md)
"""
from ablations.harness import ARCHITECTURE, SEEDS, main

if __name__ == "__main__":
    main(alternatives={"full attention": {"windows": "full"}}, setting=ARCHITECTURE, seeds=SEEDS)
