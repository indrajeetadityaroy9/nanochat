"""
One arm of a study (experiments/studies.py): its training run, scripts/base_train.train of its Variant at its depth,
ratio and seeds (a branch arm first takes its parent's kept checkpoint as its own, hard-linked, and continues it), then
the loss evaluation of its final model (scripts/base_eval.loss_evaluation), written with the code's commit to
<base_dir>/evals/<tag>.json. A trained arm is only evaluated (train returns its final checkpoint's model).

    python -m experiments.arm STUDY NAME COMMIT
"""
import os
import sys

import torch

from experiments.studies import STUDIES
from nanochat.data.tokenizer import get_tokenizer
from nanochat.eval.code_eval import tasks
from nanochat.paths import base_dir, write_json
from nanochat.train.checkpoint import checkpoint_path
from nanochat.train.schedule import plan
from scripts.base_eval import loss_evaluation
from scripts.base_train import Variant, train

study, name, commit = sys.argv[1:4]
arm = next(a for a in STUDIES[study]() if a.name == name)
path = checkpoint_path(arm.tag)
if arm.parent and not os.path.exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.link(checkpoint_path(*arm.parent), path)
variant = Variant(**arm.variant)
steps = plan(arm.depth, arm.ratio, variant.tokens).num_iterations
model = train(depth=arm.depth, ratio=arm.ratio, seed=arm.seed, data_seed=arm.data_seed, save_every=steps,
              keep=set(arm.keep), tag=arm.tag, variant=variant)
results = loss_evaluation(model, get_tokenizer(), tasks(), torch.device("cuda"))
os.makedirs(os.path.join(base_dir(), "evals"), exist_ok=True)
write_json(os.path.join(base_dir(), "evals", f"{arm.tag}.json"),
           {"tag": arm.tag, "commit": commit, "step": steps, "eval": results})
