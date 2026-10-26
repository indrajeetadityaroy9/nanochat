"""
Run app (Hydra, conf/run.yaml): each job trains a RunConfig (train/loop.py) and evaluates its checkpoints
(eval/suite.py), each phase in a fresh process. The top level imports no torch, model, training or evaluation code:
eval/code_eval.py's forkserver workers import this module.
"""
import multiprocessing
import os
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

import metrics
from corpus.tokenizer import ABLATION, TOKENIZER, get_tokenizer
from paths import ROOT, RUNS

if TYPE_CHECKING:
    from train.schedule import RunPlan


@dataclass
class RunConfig:
    """A run: the base model's recipe (the defaults, learning_rate_scale's conf/run.yaml's) except the fields set."""
    num_hidden_layers: int = 24
    tokens_per_param: float | None = None  # None: the data's max_tokens_per_param (train/stages.py)
    seed: int = 0
    tokenizer: str = "corpus"
    learning_rate_scale: float = 1.0  # times the law's peak lr; conf/run.yaml sets the measured multiplier
    num_checkpoints: int = 1
    generate: str = "parse"


def run_name(config: RunConfig) -> str:
    """The fields that differ from RunConfig()'s, sorted, k=v joined by ","; "defaults" if none."""
    default = RunConfig()
    changed = sorted((f.name, getattr(config, f.name)) for f in fields(config)
                     if getattr(config, f.name) != getattr(default, f.name))
    return ",".join(f"{name}={value}" for name, value in changed) or "defaults"


def tokenizer_path(config: RunConfig) -> Path:
    return TOKENIZER if config.tokenizer == "corpus" else ABLATION / config.tokenizer


def base_plan(*, num_hidden_layers: int, tokens_per_param: float) -> "RunPlan":
    """The base model's plan at num_hidden_layers and tokens per non-embedding parameter (schedule.plan)."""
    from model.modeling import meta_model
    from train.schedule import plan

    return plan(model=meta_model(num_hidden_layers=num_hidden_layers), tokens_per_param=tokens_per_param)


def plan_of(config: RunConfig) -> "RunPlan":
    """The base plan at the run's depth and tokens_per_param (default max_tokens_per_param), lr scaled."""
    from train.stages import max_tokens_per_param

    tokens_per_param = (max_tokens_per_param(get_tokenizer(tokenizer_path(config)),
                                             num_hidden_layers=config.num_hidden_layers)
                        if config.tokens_per_param is None else config.tokens_per_param)
    base = base_plan(num_hidden_layers=config.num_hidden_layers, tokens_per_param=tokens_per_param)
    return replace(base, learning_rate=config.learning_rate_scale * base.learning_rate)


def checkpoint_steps(config: RunConfig, run: "RunPlan") -> frozenset[int]:
    """num_checkpoints evenly spaced step counts, the final one last; if more than one, plus repo_level_start."""
    n = config.num_checkpoints
    steps = {round(run.max_steps * k / n) for k in range(1, n + 1)}
    return frozenset(steps | {run.repo_level_start} if n > 1 else steps)


def training(config: RunConfig, run: "RunPlan") -> None:
    from model.modeling import meta_model
    from train.loop import train

    model = meta_model(num_hidden_layers=config.num_hidden_layers)
    train(model=model, run=run, tokenizer_path=tokenizer_path(config), seed=config.seed,
          checkpoints=checkpoint_steps(config, run), run_name=run_name(config), config=asdict(config))


def evaluation(config: RunConfig, step: int) -> None:
    import torch

    from eval import suite
    from train.checkpoint import load_model

    # ast.parse's SyntaxWarnings on generated code
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    name = run_name(config)
    suite.evaluate(load_model(name, step, device=torch.device("cuda")), get_tokenizer(tokenizer_path(config)),
                   run_name=name, step=step, generate=config.generate)


def logged(log: Path, target: Callable[..., None], kwargs: dict) -> None:
    """target(**kwargs), this process's stdout and stderr appended to `log`."""
    with log.open("a", encoding="utf-8") as f:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
    target(**kwargs)


def phase(log: Path, target: Callable[..., None], **kwargs: object) -> int:
    """target(**kwargs) in a fresh process, its output appended to `log`; its exit code."""
    process = multiprocessing.get_context("spawn").Process(target=logged, args=(log, target, kwargs))
    process.start()
    process.join()
    return process.exitcode


def job(config: RunConfig) -> None:
    """A run's job: stale guard, skip if evaluated or diverged, then training and each checkpoint's evaluation.
    Raises RuntimeError on a changed recipe, ChildProcessError on a failed phase."""
    from eval.suite import evaluation_path

    name, run = run_name(config), plan_of(config)
    data_key = get_tokenizer(tokenizer_path(config)).data_key
    log = metrics.records(name)
    if any(r["plan"] != asdict(run) or r["data_key"] != data_key or r["config"] != asdict(config)
           for r in log if r["kind"] == "run"):
        raise RuntimeError(f"recipe changed since {name} ran: delete {RUNS / name}")
    if evaluation_path(name, run.max_steps).exists() or metrics.diverged(log) is not None:
        return
    output = RUNS / name / "run.log"
    if phase(output, training, config=config, run=run):
        if metrics.diverged(metrics.records(name)) is not None:
            return
        raise ChildProcessError(f"{name}: training failed ({output})")
    for step in sorted(checkpoint_steps(config, run)):
        if not evaluation_path(name, step).exists() and phase(output, evaluation, config=config, step=step):
            raise ChildProcessError(f"{name}: evaluation of step {step} failed ({output})")


def lr_multiplier() -> float:
    """The base multiplier of the law's peak lr: ablations/peak_lr.py analysis over the peak_lr_d24 records."""
    from ablations.peak_lr import analysis, records

    return analysis(records())["base_multiplier"]


def register() -> None:
    """Register the resolvers and schema, in the app's process only: modules importing this file as `run` would
    register again and fail."""
    OmegaConf.register_new_resolver("runs", lambda: str(RUNS))
    OmegaConf.register_new_resolver(
        "run_name", lambda *, _root_: run_name(RunConfig(**{f.name: _root_[f.name] for f in fields(RunConfig)})))
    OmegaConf.register_new_resolver("lr_multiplier", lr_multiplier, use_cache=True)
    ConfigStore.instance().store(name="run_schema", node=RunConfig)


@hydra.main(version_base="1.3", config_path=str(ROOT / "conf"), config_name="run")
def main(cfg: DictConfig) -> None:
    job(OmegaConf.to_object(cfg))


if __name__ == "__main__":
    register()
    main()
