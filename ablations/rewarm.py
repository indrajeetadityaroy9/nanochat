"""
Repository-stage schedule. Production: linear decay to zero continuing through the repository stage, as
Qwen2.5-Coder and DeepSeek-Coder continue it (arXiv 2401.14196 §3.6), adopted without a study. Alternative: re-warmed to
the peak lr at the repository stage and decayed to zero over it (harness.rewarm; Gupta et al., arXiv 2308.04014), the
file-level stage unchanged.

    python -m ablations.rewarm    (ablations/harness.py; ablations.md)
"""
from ablations.harness import REGIME, SEEDS, main

if __name__ == "__main__":
    main(alternatives={"rewarm": {"schedule": "rewarm"}}, setting=REGIME, seeds=SEEDS)
