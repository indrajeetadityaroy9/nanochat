"""
The production run end to end: d24 at the most tokens per scaling parameter its data holds without rereading a
document (train/stages.py max_ratio), seed 0, the peak lr LR_SCALE times the plan's, both stages, a resume checkpoint
every 1% of steps and a model-only snapshot every 10% and at the repository stage's start (scripts/base_train.py);
then the final model's and every snapshot's evaluation (scripts/base_eval.py: loss, greedy and filtered decoding) and
the run's tracked metrics (train/metrics.py export: results/<tag>.jsonl, <tag>-curves.jsonl). Training and each
evaluation run in a fresh process of their own: torch.compile's recompile limit and the CUDA caching allocator's
reservations are per process, so no phase inherits another's. Rerunning resumes training from its checkpoint and
evaluates every model again.

    runs/container.sh python -m scripts.production
"""
import multiprocessing
from dataclasses import replace

import torch

from nanochat.config import FILE_STAGE, run_tag
from nanochat.data.tokenizer import get_tokenizer
from nanochat.model.gpt import GPT, gpt_config
from nanochat.train.metrics import export
from nanochat.train.schedule import plan
from nanochat.train.stages import max_ratio
from scripts.base_eval import evaluate
from scripts.base_train import train

DEPTH, SEED = 24, 0
# Production's multiplier of the plan's (DeepSeek LLM law's) peak lr: ablations/lr.py's pre-registered estimator,
# results/lr.json "analysis" "production_multiplier" (d24 at 5's optimum x2.0235, carried over production's horizon by
# the d8 ratio 0.9105 measured over 5 -> 200 tokens per parameter and the square-root batch rule; ablations.md)
LR_SCALE = 2.546731698195362
# Checkpoints per run: a resume checkpoint every 1% of steps (a failure costs at most 1% of the run), a model-only
# snapshot every 10% (capability curves; ~2.1 GB each)
SAVES, SNAPSHOTS = 100, 10


def main() -> None:
    ratio = max_ratio(get_tokenizer(), depth=DEPTH)
    with torch.device("meta"):
        model = GPT(gpt_config(depth=DEPTH, stage=FILE_STAGE))
    law = plan(model=model, ratio=ratio)
    run = replace(law, lr=LR_SCALE * law.lr)
    variant = {"lr_scale": LR_SCALE}
    tag = run_tag(depth=DEPTH, ratio=ratio, seed=SEED, variant=variant)
    every = -(-run.num_iterations // SNAPSHOTS)
    snapshots = sorted({*range(every, run.num_iterations, every), run.repo_start})
    phases = [(train, {"model": model, "run": run, "variant": variant, "seed": SEED,
                       "save_every": -(-run.num_iterations // SAVES), "snapshots": frozenset(snapshots), "tag": tag})]
    phases += [(evaluate, {"tag": tag, "step": step}) for step in (None, *snapshots)]
    spawn = multiprocessing.get_context("spawn")
    for target, kwargs in phases:
        process = spawn.Process(target=target, kwargs=kwargs)
        process.start()
        process.join()
        if process.exitcode != 0:
            raise ChildProcessError(f"{target.__name__} of {tag} exited with {process.exitcode}")
    export(tag)


if __name__ == "__main__":
    main()
