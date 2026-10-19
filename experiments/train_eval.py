"""
Train one run of a study (experiments/studies.py) with scripts/base_train.train, checkpointed at its last step, then its
loss evaluation (scripts/base_eval.loss_evaluation), written with the code's commit to <base_dir>/evals/<tag>.json. A
run whose evaluation exists is done: its tag (config.run_tag) names its configuration and code.

    python -m experiments.train_eval STUDY NAME COMMIT
"""
import os
import sys

import torch

from experiments.studies import STUDIES
from nanochat.data.tokenizer import get_tokenizer
from nanochat.eval.code_eval import tasks
from nanochat.paths import base_dir, write_json
from nanochat.train.schedule import plan
from scripts.base_eval import loss_evaluation
from scripts.base_train import Variant, train


def main():
    study, name, commit = sys.argv[1:]
    run = next(r for r in STUDIES[study]() if r.name == name)
    path = os.path.join(base_dir(), "evals", f"{run.tag}.json")
    if os.path.exists(path):
        return
    variant = Variant(**run.config)
    steps = plan(depth=run.depth, ratio=run.ratio, tokens=variant.tokens).num_iterations
    model = train(depth=run.depth, ratio=run.ratio, seed=run.seed, data_seed=run.data_seed, save_every=steps,
                  tag=run.tag, variant=variant)
    results = loss_evaluation(model, get_tokenizer(), tasks(), device=torch.device("cuda"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_json(path, {"tag": run.tag, "commit": commit, "step": steps, "eval": results})


if __name__ == "__main__":
    main()
