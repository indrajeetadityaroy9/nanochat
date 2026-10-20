"""
Gradient clipping. Production: none, the regime choice docs/h100-pretraining.md (Regime) names the least settled.
Alternative: the global norm clipped to 1.0 (DeepSeek LLM, arXiv 2401.02954 §2.3). A single-seed d8 A/B in the previous
regime had clipping lower on all 11 corpora by 0.003 bits per byte (ablations.md §3).

    python -m ablations.clip    (ablations/harness.py; ablations.md)
"""
from ablations.harness import REGIME, main
from ablations.lr import ADOPTED

if __name__ == "__main__":
    main(alternatives={"clip 1.0": {"max_norm": 1.0}}, settings=REGIME, lr_scale=ADOPTED)
