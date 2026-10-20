"""
The production run end to end: d24 at the most tokens per scaling parameter its data holds without rereading a
document (train/stages.py max_ratio), seed 0, both stages, a resume checkpoint every 1% of steps and a model-only
snapshot every 10% (scripts/base_train.py); then the final model's and every snapshot's evaluation
(scripts/base_eval.py: loss, greedy and system decoding) and probes (eval/probes.py), and the run's tracked metrics
(scripts/report.py: results/<tag>.jsonl, <tag>-curves.jsonl). Each step runs in its own process (torch.compile's
caches and limits are per process), its output appended to LOGS/production-<tag>.log. Rerunning resumes training from
its checkpoint and skips evaluations already written.

    runs/h100.sh    (runs/container.sh python -m scripts.production, in tmux)
"""
import subprocess
import sys

import torch

from nanochat.config import FILE_STAGE, run_tag
from nanochat.data.tokenizer import get_tokenizer
from nanochat.model.gpt import GPT, gpt_config
from nanochat.paths import BASE_EVAL, LOGS, PROBES
from nanochat.train.checkpoint import checkpoint_name, snapshot_steps
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
        for step in (None, *snapshot_steps(tag)):
            name, at = checkpoint_name(tag, step=step), () if step is None else ("--step", step)
            if not (BASE_EVAL / f"{name}.json").exists():
                run("scripts.base_eval", "--model-tag", tag, *at)
            if not (PROBES / f"{name}.json").exists():
                run("nanochat.eval.probes", "--model-tag", tag, *at)
        run("scripts.report", tag)


if __name__ == "__main__":
    main()
