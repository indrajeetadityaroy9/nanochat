"""
Value embeddings. Production: value embeddings on alternating layers behind a gate (ResFormer, arXiv 2410.17897;
nanochat LOG 2026-01-17; MoVE, arXiv 2601.22887: -0.016 bits per byte at d12, -0.010 at d20), kept without a study for
4.7% of the d24 step and 12 vocabulary tables (302M parameters, 1.55x the scaling parameters). Alternative: none, as
every reference code model (ablations/model.py).

d16 (67.1M scaling parameters, 1/2.9 of d24) at Chinchilla's 20 tokens per parameter (arXiv 2203.15556 §3), 4 seeds
(ablations.md, Power): the gain shrinks with scale, and d8's tables are 2.3x its scaling parameters. Trains at the lr
ablations/lr.py adopts.

    python -m ablations.value_embeds    (ablations/harness.py; ablations.md)
"""
from ablations.harness import main
from ablations.lr import ADOPTED

if __name__ == "__main__":
    main(alternatives={"no value embeddings": {"value_embeds": "none"}}, settings=((16, 20, range(4)),),
         lr_scale=ADOPTED)
