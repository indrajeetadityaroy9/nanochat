"""
QK-norm gains. Production: QK norm with learned per-dim gains shared across heads (Qwen3, arXiv 2505.09388 §2),
adopted without a study. Alternative: the gains frozen at 1, QK norm alone (ablations/model.py).

    python -m ablations.qk_gains    (ablations/harness.py; ablations.md)
"""
from ablations.harness import ARCHITECTURE, SEEDS, main

if __name__ == "__main__":
    main(alternatives={"fixed QK gains": {"qk_gains": "fixed"}}, setting=ARCHITECTURE, seeds=SEEDS)
