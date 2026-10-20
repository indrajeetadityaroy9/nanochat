"""
Peak lr: production's, DeepSeek LLM's law eta = 0.3118 C^-0.125 (arXiv 2401.02954 §3.1; train/schedule.py plan),
against it halved and doubled. The law was fit for AdamW under a multi-step schedule holding the peak for 80% of tokens;
production applies it to Muon (matched to AdamW's update RMS, train/optim.py) as the peak of linear decay to zero, whose
mean lr is 59% of that schedule's. A wrong lr is the regime's largest plausible loss and costs nothing to correct.

    python -m ablations.lr    (ablations/harness.py; ablations.md)
"""
from ablations.harness import SEEDS, SETTING, main

if __name__ == "__main__":
    main(alternatives={"lr x0.5": {"lr_scale": 0.5}, "lr x2": {"lr_scale": 2.0}}, setting=SETTING, seeds=SEEDS)
