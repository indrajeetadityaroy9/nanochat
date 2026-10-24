"""
The run app (Hydra, conf/run.yaml): every run of pre-training, the base model's and each ablation's, is a RunConfig
trained by train/loop.py train and evaluated by eval/suite.py evaluate, the same code paths. Its plan (batch, steps,
peak lr) is the base model's at its depth and ratio (base_plan, schedule.plan) times lr_scale, so runs differing in
other fields match in hyperparameters and tokens. Every invocation is a sweep (hydra.mode MULTIRUN): one job without
sweep parameters, or an experiment's arms x seeds (conf/experiment/<name>.yaml), one job at a time, whose records
ablations/records.py writes once every job has ended; the first failed job stops the sweep.

A run's key (key) names its directory RUNS/<key> (nanochat/paths.py), so a run two experiments share is one run. Its
job (job): the stale guard, which raises when a recorded run record's plan or vocabulary data key is not the current
one; then the run is skipped if its final checkpoint's evaluation exists or it diverged (train/metrics.py failure: a
non-finite loss or gradient norm, an outcome of a lr sweep's high end, arXiv 2309.14322 §3.1); else training, then
each checkpoint's evaluation, each in a fresh process (torch.compile's recompile limit and the CUDA caching
allocator's reservations are per process), their output appended to RUNS/<key>/run.log. A training process that fails
without a failure record raises; one that diverged ends the job, recorded as diverged. A stopped run resumes from its
newest checkpoint. This process plans on the meta device and never initializes CUDA. The module's top level imports no
torch, model, training or evaluation code (the functions import what they run): every worker of an evaluation's
sanitize pool (eval/code_eval.py sanitizers, a forkserver) imports this module, the phases' main module.

    python -m nanochat.run experiment=<name>               (conf/experiment/<name>.yaml)
    python -m nanochat.run depth=8 ratio=0.5 tokenizer=sa-bpe   (one run)
"""
import json
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

from nanochat.corpus.tokenizer import ABLATION, TOKENIZER, get_tokenizer
from nanochat.paths import RESULTS, ROOT, RUNS
from nanochat.train import metrics

if TYPE_CHECKING:
    from nanochat.train.schedule import RunPlan


@dataclass
class RunConfig:
    """A run: the base model's recipe (the defaults) except the fields set."""
    depth: int = 24
    ratio: float | None = None      # tokens per scaling parameter; None: train/stages.py max_ratio of the data
    seed: int = 0                   # initialization and data order
    tokenizer: str = "corpus"       # "corpus": CORPUS/tokenizer; else an arm: CORPUS/tokenizer-ablation/<arm>
    lr_scale: float = 1.0           # times the law's peak lr (schedule.plan)
    muon_tau_scale: float = 1.0     # D7: Muon matrices' tau x scale (their weight decay / scale; train/optim.py)
    adamw_lr_scale: float = 1.0     # D6: AdamW groups' lr x scale, their weight decay / scale (decay timescale held)
    softcap: float | None = None    # D5: output logits cap * tanh(logits / cap); None: no cap (model/gpt.py)
    checkpoints: int = 1            # evenly spaced checkpoints, the final among them; > 1 adds repo_start
    generate: str = "parse"         # suite.evaluate's generate: "parse" (ablations) or "benchmarks" (base)


def key(config: RunConfig) -> str:
    """The fields that differ from RunConfig()'s, sorted, k=v joined by "," (floats by repr); "defaults" if none. A
    field added with a default renames no run."""
    default = RunConfig()
    changed = sorted((f.name, getattr(config, f.name)) for f in fields(config)
                     if getattr(config, f.name) != getattr(default, f.name))
    return ",".join(f"{name}={value!r}" if isinstance(value, float) else f"{name}={value}"
                    for name, value in changed) or "defaults"


def vocabulary(config: RunConfig) -> Path:
    return TOKENIZER if config.tokenizer == "corpus" else ABLATION / config.tokenizer


def base_plan(*, depth: int, ratio: float) -> "RunPlan":
    """The base model's plan at a depth and tokens per scaling parameter (schedule.plan)."""
    import torch

    from nanochat.corpus.mixture import FILE_STAGE
    from nanochat.model.gpt import GPT, gpt_config
    from nanochat.train.schedule import plan

    with torch.device("meta"):
        return plan(model=GPT(gpt_config(depth=depth, stage=FILE_STAGE)), ratio=ratio)


def plan_of(config: RunConfig) -> "RunPlan":
    """The base model's plan at the run's depth and ratio (its data's max_ratio if None), at the run's peak lr."""
    from nanochat.train.stages import max_ratio

    ratio = max_ratio(get_tokenizer(vocabulary(config)), depth=config.depth) if config.ratio is None else config.ratio
    base = base_plan(depth=config.depth, ratio=ratio)
    return replace(base, lr=config.lr_scale * base.lr)


def checkpoint_steps(config: RunConfig, run: "RunPlan") -> frozenset[int]:
    """config.checkpoints evenly spaced step counts, the last the final model's; with more than one, also repo_start
    (the model before the repository stage)."""
    n = config.checkpoints
    steps = {round(run.num_iterations * k / n) for k in range(1, n + 1)}
    return frozenset(steps | {run.repo_start} if n > 1 else steps)


def training(config: RunConfig, run: "RunPlan") -> None:
    import torch

    from nanochat.corpus.mixture import FILE_STAGE
    from nanochat.model.gpt import GPT, gpt_config
    from nanochat.train.loop import train

    with torch.device("meta"):
        model = GPT(replace(gpt_config(depth=config.depth, stage=FILE_STAGE), softcap=config.softcap))
    train(model=model, run=run, vocabulary=vocabulary(config), seed=config.seed,
          muon_tau_scale=config.muon_tau_scale, adamw_lr_scale=config.adamw_lr_scale,
          checkpoints=checkpoint_steps(config, run), tag=key(config), config=asdict(config))


def evaluation(config: RunConfig, step: int) -> None:
    import torch

    from nanochat.eval import suite
    from nanochat.train.checkpoint import load_model

    # SyntaxWarnings of ast.parse on generated code (eval/verify.py, inference/)
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    tag = key(config)
    suite.evaluate(load_model(tag, step, device=torch.device("cuda")), get_tokenizer(vocabulary(config)), tag=tag,
                   step=step, generate=config.generate)


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
    """A run's job (module docstring)."""
    from nanochat.eval.suite import evaluation_path

    tag, run = key(config), plan_of(config)
    data_key = get_tokenizer(vocabulary(config)).data_key
    if any(r["plan"] != asdict(run) or r["data_key"] != data_key for r in metrics.records(tag) if r["kind"] == "run"):
        raise RuntimeError(f"recipe changed since {tag} ran: delete {RUNS / tag}")
    if evaluation_path(tag, run.num_iterations).exists() or metrics.diverged(tag) is not None:
        return
    log = RUNS / tag / "run.log"
    if phase(log, training, config=config, run=run):
        if metrics.diverged(tag) is not None:
            return
        raise ChildProcessError(f"{tag}: training failed ({log})")
    for step in sorted(checkpoint_steps(config, run)):
        if not evaluation_path(tag, step).exists() and phase(log, evaluation, config=config, step=step):
            raise ChildProcessError(f"{tag}: evaluation of step {step} failed ({log})")


def lr_multiplier() -> float:
    """The base model's multiplier of the law's peak lr: ablations/lr.py's estimator over its recorded runs."""
    from ablations.lr import analysis

    lines = [json.loads(line) for line in (RESULTS / "lr.jsonl").read_text(encoding="utf-8").splitlines()]
    return analysis(lines)["base_multiplier"]


OmegaConf.register_new_resolver("runs", lambda: str(RUNS))
OmegaConf.register_new_resolver(
    "run_key", lambda *, _root_: key(RunConfig(**{f.name: _root_[f.name] for f in fields(RunConfig)})))
OmegaConf.register_new_resolver("lr_multiplier", lr_multiplier, use_cache=True)
ConfigStore.instance().store(name="run_schema", node=RunConfig)


@hydra.main(version_base="1.3", config_path=str(ROOT / "conf"), config_name="run")
def main(cfg: DictConfig) -> None:
    job(OmegaConf.to_object(cfg))


if __name__ == "__main__":
    main()
