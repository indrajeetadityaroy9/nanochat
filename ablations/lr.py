"""
Peak lr. Production: DeepSeek LLM's law eta = 0.3118 C^-0.125 (arXiv 2401.02954 §3.1; train/schedule.py plan) as the
peak lr, adopted without a study (docs/h100-pretraining.md). Alternatives: it halved and doubled. The law was fit for
AdamW under a multi-step schedule holding the peak for 80% of tokens; production applies it to Muon (matched to AdamW's
update RMS, train/optim.py) as the peak of linear decay to zero, whose mean lr is 59% of that schedule's.

    python -m ablations.lr    (ablations/harness.py; ablations.md)
"""
from ablations.harness import REGIME, SEEDS, main

if __name__ == "__main__":
    main(alternatives={"lr x0.5": {"lr_scale": 0.5}, "lr x2": {"lr_scale": 2.0}}, setting=REGIME, seeds=SEEDS)
