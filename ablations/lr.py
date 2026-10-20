"""
Peak lr. Production: DeepSeek LLM's law eta = 0.3118 C^-0.125 (arXiv 2401.02954 §3.1; train/schedule.py plan) as the
peak lr, adopted without a study (docs/h100-pretraining.md). Alternatives: it halved and doubled. The law was fit for
AdamW over C = 1e17 to 2e19 FLOPs under a multi-step schedule holding the peak for 80% of tokens; production applies it
to Muon (matched to AdamW's update RMS, train/optim.py) as the peak of linear decay to zero, whose mean lr is 59% of
that schedule's, at C = 1.5e20.

Two settings bracket production (d24 at the most tokens per scaling parameter its data holds, runs/h100.sh), one seed
each, the three lrs of a setting read together as a sweep (ablations.md): its horizon at d8 (7.2B tokens, C = 7.0e17)
and its width at d24 over 5 tokens per parameter (1.0B tokens, C = 1.5e18), the longest d24 run three arms afford. Runs
first: ablations/value_embeds.py trains at ADOPTED.

    python -m ablations.lr    (ablations/harness.py; ablations.md)
"""
from ablations.harness import main
from nanochat.data.tokenizer import get_tokenizer
from nanochat.train.stages import max_ratio

# Peak lr the other experiments train at, in units of the law's: the law's own, until results/lr.json adopts an
# alternative's (ablations.md, Analysis)
ADOPTED = 1.0

if __name__ == "__main__":
    production_ratio = max_ratio(get_tokenizer(), depth=24)
    main(alternatives={"lr x0.5": {"lr_scale": 0.5}, "lr x2": {"lr_scale": 2.0}},
         settings=((8, production_ratio, range(1)), (24, 5, range(1))), lr_scale=1.0)
