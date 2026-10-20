"""
Peak lr. Production: DeepSeek LLM's law eta = 0.3118 C^-0.125 (arXiv 2401.02954 §3.1; train/schedule.py plan) as the
peak lr, adopted without a study (docs/h100-pretraining.md). The law was fit for AdamW over C = 1e17 to 2e19 FLOPs under
a multi-step schedule; production applies it to Muon (matched to AdamW's update RMS, train/optim.py) as the peak of
linear decay to zero at C = 1.5e20, and at a fixed model it falls with tokens as D^-0.125 where the measured optimum
falls as D^-0.32 (Bjorck et al., arXiv 2409.19913), so at production's 490.8 tokens per parameter it may lie above the
optimum. Alternatives: the law's lr times 0.25 (at production's ratio only, where that gap points), 0.5 and 2.

Settings (ablations.md, Settings): d8 at 50 tokens per parameter with 4 seeds (the seed and paired-arm noise) and at
production's ratio, one seed (the horizon slope at one width); d24 at 5, one seed (production's width). Runs first:
ablations/value_embeds.py trains at ADOPTED.

    python -m ablations.lr    (ablations/harness.py; ablations.md)
"""
from ablations.harness import main
from nanochat.data.tokenizer import get_tokenizer
from nanochat.train.stages import max_ratio

# Peak lr the other experiments train at, in units of the law's: the law's own, until results/lr.json sets the
# prediction ablations.md (Analysis) fixes
ADOPTED = 1.0

if __name__ == "__main__":
    grid = {"lr x0.5": {"lr_scale": 0.5}, "lr x2": {"lr_scale": 2.0}}
    main(settings=((8, 50, range(4), grid),
                   (8, max_ratio(get_tokenizer(), depth=24), range(1), {"lr x0.25": {"lr_scale": 0.25}} | grid),
                   (24, 5, range(1), grid)),
         lr_scale=1.0)
