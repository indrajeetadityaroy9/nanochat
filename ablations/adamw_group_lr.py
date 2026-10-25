"""
The adamw_group_lr ablation (conf/experiment/adamw_group_lr.yaml): the AdamW groups' lr x 1/4 and x 4 at their decay
timescale (their weight decay / scale), Muon's untouched, d16 at the sa-bpe arm's max_tokens_per_param, 2 seeds; the
baseline is tokenizer_split_merge's sa-bpe runs (the same run names). Its pre-registered analysis: the scale sweep over
{1/4, 1, 4} (ablations/contrast.py scaled), adamw_learning_rate_scale the adopted scale, else one lr for all groups.
"""
from ablations.contrast import scaled


def analysis(lines: list[dict]) -> dict:
    """The AdamW group lr decision: "scale" the adopted adamw_learning_rate_scale."""
    return scaled(lines, "adamw_learning_rate_scale")
