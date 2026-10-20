"""
Value embeddings. Production: value embeddings on alternating layers behind a gate (ResFormer, arXiv 2410.17897;
nanochat LOG 2026-01-17; MoVE, arXiv 2601.22887: -0.016 bits per byte at d12, -0.010 at d20), kept without a study for
4.7% of the d24 step and 12 vocabulary tables (302M parameters, 1.55x the scaling parameters). Alternative: none, as
every reference code model (ablations/model.py).

Settings (ablations.md, Settings): d16 at Chinchilla's 20 tokens per parameter (arXiv 2203.15556 §3), 4 seeds, the
decision's; d8 at 20, 4 seeds, and at production's ratio, one seed: the gain's trends in width and in training length.
Trains at schedule.plan's peak lr, which carries ablations/lr.py's adopted result.

    python -m ablations.value_embeds    (ablations/harness.py; ablations.md)
"""
from ablations.harness import main
from nanochat.data.tokenizer import get_tokenizer
from nanochat.train.stages import max_ratio

if __name__ == "__main__":
    none = {"no value embeddings": {"value_embeds": "none"}}
    main(settings=((8, 20, range(4), none), (8, max_ratio(get_tokenizer(), depth=24), range(1), none),
                   (16, 20, range(4), none)))
