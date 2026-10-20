"""
The production run end to end: d24 at the most tokens per scaling parameter its data holds without rereading a
document (train/stages.py max_ratio), seed 0, both stages, a resume checkpoint every 1% of steps and a model-only
snapshot every 10% (scripts/base_train.py); then the final model's and every snapshot's evaluation
(scripts/base_eval.py: loss, greedy and filtered decoding), and the run's tracked metrics (train/metrics.py export:
results/<tag>.jsonl, <tag>-curves.jsonl). Training and each evaluation run in their own process (torch.compile's caches
and limits are per process), their output appended to LOGS/production-<tag>.log. Rerunning resumes training from its
checkpoint and evaluates every model again.

    runs/h100.sh    (runs/container.sh python -m scripts.production, in tmux)
"""
import subprocess
import sys

import torch

from nanochat.config import FILE_STAGE, run_tag
from nanochat.data.tokenizer import get_tokenizer
from nanochat.model.gpt import GPT, gpt_config
from nanochat.paths import CHECKPOINTS, LOGS
from nanochat.train.metrics import export
from nanochat.train.schedule import plan
from nanochat.train.stages import max_ratio

DEPTH, SEED = 24, 0
# Checkpoints per run: a resume checkpoint every 1% of steps (a failure costs at most 1% of the run), a model-only
# snapshot every 10% (capability curves; ~2.1 GB each)
SAVES, SNAPSHOTS = 100, 10


def main() -> None:
    ratio = max_ratio(get_tokenizer(), depth=DEPTH)
    with torch.device("meta"):
        steps = plan(model=GPT(gpt_config(depth=DEPTH, stage=FILE_STAGE)), ratio=ratio).num_iterations
    tag = run_tag(depth=DEPTH, ratio=ratio, seed=SEED, data_seed=SEED, variant={})
    with (LOGS / f"production-{tag}.log").open("a", encoding="utf-8") as log:
        def run(module: str, *args: object) -> None:
            subprocess.run([sys.executable, "-m", module, *map(str, args)], stdout=log, stderr=subprocess.STDOUT,
                           check=True)

        run("scripts.base_train", "--depth", DEPTH, "--target-param-data-ratio", repr(ratio), "--seed", SEED,
            "--save-every", -(-steps // SAVES), "--snapshot-every", -(-steps // SNAPSHOTS))
        # the model-only snapshots (checkpoint.py), ascending
        snapshots = sorted(int(p.stem.removeprefix(f"{tag}-step")) for p in CHECKPOINTS.glob(f"{tag}-step*.pt"))
        for step in (None, *snapshots):
            run("scripts.base_eval", "--model-tag", tag, *(() if step is None else ("--step", step)))
    export(tag)


if __name__ == "__main__":
    main()
