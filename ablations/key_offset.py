"""
Partial key offset. Production: not adopted (docs/h100-pretraining.md: a speedrun result, modded-nanogpt record #49,
at 124M and short training). Alternative: p-RoPE and the partial key offset on the full-context layers
(ablations/model.py). Adopting it also needs a native decode path.

    python -m ablations.key_offset    (ablations/harness.py; ablations.md)
"""
from ablations.harness import ARCHITECTURE, SEEDS, main

if __name__ == "__main__":
    main(alternatives={"key offset": {"key_offset": "on"}}, setting=ARCHITECTURE, seeds=SEEDS)
