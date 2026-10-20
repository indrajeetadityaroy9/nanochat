"""
Weight decay: production's tau law, lambda = 1 / (eta tau* T) with tau* = 1.084 TPP^-0.527 (Power Lines, arXiv
2505.13738; train/optim.py tau_weight_decay), 0.317 at d24, against DeepSeek LLM's 0.1 (arXiv 2401.02954 §2.3), whose
EMA timescale is 3.2x the tau law's at d24 (13% against 4.1% of the run) and 3.1x at the experiments' setting.

    python -m ablations.weight_decay    (ablations/harness.py; ablations.md)
"""
from ablations.harness import SEEDS, SETTING, main

if __name__ == "__main__":
    main(alternatives={"weight decay 0.1": {"weight_decay": 0.1}}, setting=SETTING, seeds=SEEDS)
