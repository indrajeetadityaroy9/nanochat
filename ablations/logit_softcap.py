"""
The logit_softcap ablation (conf/experiment/logit_softcap.yaml): the output logits capped at c = 15
(final_logit_softcapping; karpathy/nanochat 92d63d4 gpt.py:511-515; modded-nanogpt record #18 "lowers that to 15")
against none, d16 at Chinchilla's 20 tokens per parameter (arXiv 2203.15556 Table 3) on the sa-bpe arm's data, 4 paired
seeds, at the base model's learning_rate_scale (conf/run.yaml's default, ablations/peak_lr.py analysis), as the cap's
published effect is a high-lr one. Its pre-registered analysis: adopt c = 15 iff the contrast adopts it
(ablations/contrast.py); else no cap ("equal" keeps: the cap costs the loss, eval and decoder paths).
"""
from ablations.contrast import adopted, contrast, evaluated


def analysis(lines: list[dict]) -> dict:
    """The softcap decision (module docstring) over RESULTS/logit_softcap.jsonl's records."""
    c = contrast(evaluated(lines, "final_logit_softcapping"), 15.0, None)
    return {"contrast": c, "decision": {"final_logit_softcapping": 15.0 if adopted(c) else None}}
