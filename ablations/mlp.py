"""
MLP. Production: ReLU^2 with hidden 4C (Primer, arXiv 2109.08668; nanochat LOG 2026-02-05: SwiGLU worse at d12 and
d24). Alternative: SwiGLU with hidden 8C/3, parameter-matched (Shazeer, arXiv 2002.05202; ablations/swiglu.py), the MLP
of DeepSeek-Coder, Qwen2.5-Coder and Code Llama. Adopting it also needs csrc/decode.cu's MLP gated.

    python -m ablations.mlp    (ablations/harness.py; ablations.md)
"""
from ablations.harness import ARCHITECTURE, main

if __name__ == "__main__":
    main(alternatives={"swiglu": {"mlp": "swiglu"}}, setting=ARCHITECTURE)
