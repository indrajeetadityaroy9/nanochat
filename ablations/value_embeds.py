"""
Value embeddings. Production: value embeddings on alternating layers behind a gate (ResFormer, arXiv 2410.17897;
nanochat LOG 2026-01-17; MoVE, arXiv 2601.22887: -0.016 bits per byte at d12), kept without a study for 4.7% of the d24
step and 12 vocabulary tables. Alternative: none, as every code model cited in ablations.md (ablations/model.py).

    python -m ablations.value_embeds    (ablations/harness.py; ablations.md)
"""
from ablations.harness import ARCHITECTURE, main
from ablations.lr import ADOPTED

if __name__ == "__main__":
    main(alternatives={"no value embeddings": {"value_embeds": "none"}}, settings=ARCHITECTURE, lr_scale=ADOPTED)
