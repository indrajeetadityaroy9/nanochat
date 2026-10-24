"""
The production run end to end: d24 at the most tokens per scaling parameter its data holds without rereading a
document (train/stages.py max_ratio), seed 0, the peak lr ablations/lr.py's multiplier times the plan's, both stages, a
checkpoint at each tenth of the run and at the repository stage's start (train/loop.py); then every
checkpoint's evaluation (evaluate) and the run's tracked metrics (eval/suite.py export: results/<tag>.jsonl,
<tag>-curves.jsonl). Training and each evaluation run in a fresh process of their own. Rerunning resumes training from
its newest checkpoint and evaluates every checkpoint again.

    runs/container.sh python -m scripts.production
"""
import json
import multiprocessing
import warnings
from collections.abc import Callable
from dataclasses import replace

import torch

from ablations.lr import analysis
from nanochat.config import FILE_STAGE, run_tag
from nanochat.data.tokenizer import get_tokenizer
from nanochat.eval import suite
from nanochat.model.gpt import GPT, gpt_config
from nanochat.paths import RESULTS
from nanochat.train.checkpoint import checkpoint_name, load_model
from nanochat.train.loop import train
from nanochat.train.schedule import plan
from nanochat.train.stages import max_ratio

DEPTH, SEED = 24, 0
# Checkpoints at each tenth of the run (the capability curve over training; the tenth tenth is the final model) and at
# the repository stage's start (the model before it); each is a resume point and an evaluated model
TENTHS = 10


def evaluate(tag: str, step: int) -> None:
    """eval/suite.py evaluate of run `tag`'s checkpoint after `step` steps, with every benchmark."""
    # SyntaxWarnings of ast.parse on generated code (eval/verify.py, inference/)
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    suite.evaluate(load_model(tag, step, device=torch.device("cuda")), get_tokenizer(),
                   name=checkpoint_name(tag, step), step=step, generate="benchmarks")


def isolated(target: Callable[..., object], **kwargs: object) -> None:
    """target(**kwargs) in a fresh process, raising if it fails: torch.compile's recompile limit and the CUDA caching
    allocator's reservations are per process, so no phase inherits another's."""
    process = multiprocessing.get_context("spawn").Process(target=target, kwargs=kwargs)
    process.start()
    process.join()
    if process.exitcode != 0:
        raise ChildProcessError(f"{target.__name__} exited with {process.exitcode}")


def main() -> None:
    ratio = max_ratio(get_tokenizer(), depth=DEPTH)
    with torch.device("meta"):
        model = GPT(gpt_config(depth=DEPTH, stage=FILE_STAGE))
    law = plan(model=model, ratio=ratio)
    # the multiplier of the law's (DeepSeek LLM, arXiv 2401.02954 §3.1) peak lr: lr.py's estimator (amended 2026-10-22:
    # d8's horizon trend net of its batch growth) at this data's horizon, the optima it measured at dataset revision
    # 1e711aa carried, not measured again (d24 at 5's optimum, the d8 ratio over 5 -> 200 tokens per parameter, the
    # square-root batch rule, arXiv 2205.10287)
    lr_scale = analysis([json.loads(line) for line in (RESULTS / "lr.jsonl").read_text(encoding="utf-8").splitlines()]
                        )["production_multiplier"]
    run = replace(law, lr=lr_scale * law.lr)
    variant = {"lr_scale": lr_scale}
    tag = run_tag(depth=DEPTH, ratio=ratio, seed=SEED, variant=variant)
    checkpoints = {round(run.num_iterations * k / TENTHS) for k in range(1, TENTHS + 1)} | {run.repo_start}
    isolated(train, model=model, run=run, tokenizer_name="production", variant=variant, seed=SEED,
             checkpoints=frozenset(checkpoints), tag=tag)
    for step in sorted(checkpoints):
        isolated(evaluate, tag=tag, step=step)
    suite.export(tag)


if __name__ == "__main__":
    main()
