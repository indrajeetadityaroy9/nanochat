"""
Block Attention Residuals. Production: not adopted (docs/h100-pretraining.md: +12% d24 step time for a published 1.1%
loss gain at 194M parameters, arXiv 2603.15031). Alternative: the residual stream as Block Attention Residuals in place
of the resid and x0 scalars (ablations/model.py). Adopting it also needs a native decode path.

    python -m ablations.attn_res    (ablations/harness.py; ablations.md)
"""
from ablations.harness import ARCHITECTURE, SEEDS, main

if __name__ == "__main__":
    main(alternatives={"block attention residuals": {"residual": "attn_res"}}, setting=ARCHITECTURE, seeds=SEEDS)
