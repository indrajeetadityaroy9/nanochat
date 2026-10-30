"""
Run app (Hydra, conf/run.yaml): each job trains a RunConfig (train/loop.py) and evaluates its checkpoints
(eval/suite.py), each phase in a fresh process; a checkpoint's evaluation starts once the previous one has released
the GPU. The top level imports no torch, model, training or evaluation code: eval/code_eval.py's forkserver workers
import this module.
"""
import multiprocessing
import os
import threading
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

import metrics
from ablations.peak_lr import lr_multiplier
from ablations.sft_lr import sft_lr_scale
from corpus.tokenizer import get_tokenizer
from paths import ROOT, RUNS

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event

    from train.schedule import RunPlan

# A run with more than one evaluated checkpoint also keeps analysis checkpoints (docs/training.md §6): full state
# every FULL_STATE_EVERY steps (Pythia's cadence, arXiv 2304.01373 §2.4), fp32 weights only at steps 1, 2,
# 4, ..., EARLY_WEIGHTS_LAST (Pythia's log-spaced early checkpoints) and at repo_level_start + AFTER_STAGE_WEIGHTS
FULL_STATE_EVERY = 1000
EARLY_WEIGHTS_LAST = 512
AFTER_STAGE_WEIGHTS = (100, 200, 400, 800, 1600)


@dataclass
class RunConfig:
    """A run of the base model (d24): its recipe (the defaults, learning_rate_scale's conf/run.yaml's) except the
    fields set; with `base`, an SFT run of it (docs/post-training.md §11.2)."""
    tokens_per_param: float | None = None  # None: the data's max_tokens_per_param (train/stages.py)
    # times the law's peak lr (conf/run.yaml sets the measured multiplier); an SFT run's: times its base run's peak lr
    learning_rate_scale: float = 1.0
    num_checkpoints: int = 1
    generate: str = "parse"  # eval/suite.py evaluate's mode; "none": no evaluation, the run trained only
    base: str | None = None  # SFT: the base run's name; its checkpoint after base_step steps is the start
    base_step: int = 69_874
    sft_rows: float = 1.0  # SFT: the share of the SFT corpus's packed train rows trained, one epoch


def run_name(config: RunConfig) -> str:
    """The fields that differ from RunConfig()'s, sorted, k=v joined by ","; "defaults" if none."""
    default = RunConfig()
    changed = sorted((f.name, getattr(config, f.name)) for f in fields(config)
                     if getattr(config, f.name) != getattr(default, f.name))
    return ",".join(f"{name}={value}" for name, value in changed) or "defaults"


def plan_of(config: RunConfig) -> "RunPlan":
    """The base model's plan (schedule.plan) at the run's tokens_per_param (default max_tokens_per_param), lr
    scaled; an SFT run's (schedule.sft_plan) over its share of the SFT rows, its base run's peak lr scaled."""
    from model.modeling import meta_model
    from train.schedule import plan, sft_plan
    from train.stages import max_tokens_per_param

    if config.base is not None:
        from corpus.layout import packed_rows
        from posttrain.sft import NAME, SFT_STAGE

        started = next(r for r in metrics.records(config.base) if r["kind"] == "run")
        rows = packed_rows(NAME, "train", get_tokenizer(), SFT_STAGE.seq_len)
        return sft_plan(model=meta_model(), rows=round(config.sft_rows * rows),
                        learning_rate=config.learning_rate_scale * started["learning_rate"]["muon"][0])
    tokens_per_param = (max_tokens_per_param(get_tokenizer()) if config.tokens_per_param is None
                        else config.tokens_per_param)
    base = plan(model=meta_model(), tokens_per_param=tokens_per_param)
    return replace(base, learning_rate=config.learning_rate_scale * base.learning_rate)


def checkpoint_steps(config: RunConfig, run: "RunPlan") -> frozenset[int]:
    """num_checkpoints evenly spaced step counts, the final one last; if more than one, plus a base run's
    repo_level_start."""
    n = config.num_checkpoints
    steps = {round(run.max_steps * k / n) for k in range(1, n + 1)}
    return frozenset(steps | {run.repo_level_start} if n > 1 and config.base is None else steps)


def analysis_steps(config: RunConfig, run: "RunPlan") -> tuple[frozenset[int], frozenset[int]]:
    """The analysis checkpoints' (full-state, weights-only) step counts below run.max_steps, the weights-only ones
    disjoint from every full state; none if num_checkpoints is 1. An SFT run's: full states only, every
    FULL_STATE_EVERY steps (resume points)."""
    full = frozenset(range(FULL_STATE_EVERY, run.max_steps, FULL_STATE_EVERY))
    if config.base is not None:
        return full, frozenset()
    if config.num_checkpoints == 1:
        return frozenset(), frozenset()
    weights = ({2**k for k in range(EARLY_WEIGHTS_LAST.bit_length())}
               | {run.repo_level_start + s for s in AFTER_STAGE_WEIGHTS})
    return full, frozenset(s for s in weights if s < run.max_steps) - full - checkpoint_steps(config, run)


def validation_steps(config: RunConfig, run: "RunPlan") -> frozenset[int]:
    """An SFT run's held-out loss: every FULL_STATE_EVERY steps and after its last, and before its first if it trains
    every row (a calibration arm, on a share of them, starts from the same base); none for a base run."""
    if config.base is None:
        return frozenset()
    first = {0} if config.sft_rows == 1 else set()
    return frozenset({*first, *range(FULL_STATE_EVERY, run.max_steps, FULL_STATE_EVERY), run.max_steps})


def training(config: RunConfig, run: "RunPlan") -> None:
    from model.modeling import meta_model
    from train.checkpoint import checkpoint_path
    from train.loop import train

    full, weights = analysis_steps(config, run)
    train(model=meta_model(), run=run, checkpoints=checkpoint_steps(config, run) | full, weights=weights,
          run_name=run_name(config), config=asdict(config),
          base=None if config.base is None else checkpoint_path(config.base, config.base_step),
          validations=validation_steps(config, run))


def evaluation(config: RunConfig, step: int, previous: "Event | None", released: "Event") -> None:
    from eval import suite

    # ast.parse's SyntaxWarnings on generated code
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    suite.evaluate(get_tokenizer(), run_name=run_name(config), step=step, generate=config.generate,
                   previous=previous, released=released)


def logged(log: Path, target: Callable[..., None], kwargs: dict) -> None:
    """target(**kwargs), this process's stdout and stderr appended to `log`."""
    with log.open("a", encoding="utf-8") as f:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
    target(**kwargs)


def evaluations(log: Path, checkpoints: list[tuple[RunConfig, int]]) -> None:
    """Each (config, step)'s evaluation in a fresh process, output appended to `log`. Checkpoint k + 1's starts once
    k's has released the GPU (suite.evaluate's `released`) or ended, with previous = k's end; none starts after a
    failure. Raises ChildProcessError, once every started evaluation has ended, if any failed."""
    spawn = multiprocessing.get_context("spawn")
    started, previous = [], None
    for config, step in checkpoints:
        released, ended = spawn.Event(), spawn.Event()
        process = spawn.Process(target=logged, args=(log, evaluation, {"config": config, "step": step,
                                                                         "previous": previous, "released": released}))
        process.start()
        watcher = threading.Thread(target=lambda p=process, e=ended, r=released: (p.join(), e.set(), r.set()))
        watcher.start()
        started.append((config, step, process, watcher))
        released.wait()  # set by the evaluation, or by its watcher once it has ended
        previous = ended
        if any(p.exitcode for _, _, p, _ in started):  # a failed evaluation's exit code is set before its `released`
            break
    failed = []
    for config, step, process, watcher in started:
        watcher.join()
        if process.exitcode:
            failed.append(f"{run_name(config)} step {step}")
    if failed:
        raise ChildProcessError(f"evaluation of {', '.join(failed)} failed ({log})")


def job(config: RunConfig) -> None:
    """A run's job: skipped if every checkpoint is evaluated (with generate "none": saved) or the run diverged; else
    training (resuming) in a fresh process, then its unevaluated checkpoints' evaluations (evaluations). Raises
    ChildProcessError on a failed phase other than a divergence."""
    from eval.suite import evaluation_path
    from train.checkpoint import checkpoint_path

    name, run = run_name(config), plan_of(config)
    done = checkpoint_path if config.generate == "none" else evaluation_path
    unevaluated = [step for step in sorted(checkpoint_steps(config, run)) if not done(name, step).exists()]
    if not unevaluated or metrics.diverged(metrics.records(name)) is not None:
        return
    output = RUNS / name / "run.log"
    process = multiprocessing.get_context("spawn").Process(
        target=logged, args=(output, training, {"config": config, "run": run}))
    process.start()
    process.join()
    if process.exitcode:
        if metrics.diverged(metrics.records(name)) is not None:
            return
        raise ChildProcessError(f"{name}: training failed ({output})")
    if config.generate != "none":
        evaluations(output, [(config, step) for step in unevaluated])


@hydra.main(version_base="1.3", config_path=str(ROOT / "conf"), config_name="run")
def main(cfg: DictConfig) -> None:
    job(OmegaConf.to_object(cfg))


if __name__ == "__main__":
    # the resolvers and schema, in the app's process only: modules importing this file as `run` would register again
    # and fail
    OmegaConf.register_new_resolver("runs", lambda: str(RUNS))
    OmegaConf.register_new_resolver(
        "run_name", lambda *, _root_: run_name(RunConfig(**{f.name: _root_[f.name] for f in fields(RunConfig)})))
    OmegaConf.register_new_resolver("lr_multiplier", lr_multiplier, use_cache=True)
    OmegaConf.register_new_resolver("sft_lr_scale", sft_lr_scale, use_cache=True)
    ConfigStore.instance().store(name="run_schema", node=RunConfig)
    main()
