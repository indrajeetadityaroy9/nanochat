"""
Residual scalars. Production: per layer x = resid_lambda x + x0_lambda x0, learned (nanochat LOG 2026-01-11: the x0
scalars -0.004 to -0.010 bits per byte at d8-d20), adopted without a study. Alternative: resid_lambda 1 and x0_lambda
0, frozen: a plain residual stream, as every code model cited in ablations.md (ablations/model.py).

    python -m ablations.residual_scalars    (ablations/harness.py; ablations.md)
"""
from ablations.harness import ARCHITECTURE, SEEDS, main

if __name__ == "__main__":
    main(alternatives={"identity residual": {"residual": "identity"}}, setting=ARCHITECTURE, seeds=SEEDS)
