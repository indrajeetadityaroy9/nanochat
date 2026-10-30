"""
The SFT corpus's name, its one training stage and §11.1's markers (docs/post-training.md), torch-free and light for
every process that plans or trains an SFT run; the corpus itself: posttrain/sft_data.py.
"""
from corpus.mixture import REPO_LEVEL_STAGE, Group, Stage

# The corpus (corpus/layout.py paths) and its one stage: 3 rows of 32,768 tokens per micro-batch, as the base's 32K
# stage; tokens 0, since no plan sizes its packs (mixture.Stage.rows)
NAME = "sft"
SFT_STAGE = Stage(REPO_LEVEL_STAGE.seq_len, 0, {NAME: Group(1.0, (NAME,))})
# §11.1's markers: the context ends with THINK; END_THINK separates the reasoning from the answer
THINK, END_THINK = "\n\n<think>\n", "\n</think>\n\n"
