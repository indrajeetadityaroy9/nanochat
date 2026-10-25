"""
The run app (Hydra, conf/run.yaml): every run of pre-training, the base model's and each ablation's, is a RunConfig
trained by train/loop.py train and evaluated by eval/suite.py evaluate, the same code paths. Its plan (batch, steps,
peak learning rate) is the base model's at its num_hidden_layers and tokens_per_param (base_plan, schedule.plan) times
learning_rate_scale, so runs differing in other fields match in hyperparameters and tokens. Every invocation is a sweep
(hydra.mode MULTIRUN): one job without sweep parameters, or an experiment's arms x seeds
(conf/experiment/<name>.yaml), one job at a time, whose records ablations/records.py writes once every job has ended;
the first failed job stops the sweep.

A run's name (run_name) names its directory RUNS/<run_name> (paths.py), so a run two experiments share is one run. Its
job (job): the stale guard, which raises when a recorded run record's plan or tokenizer data key is not the current
one; then the run is skipped if its final checkpoint's evaluation exists or it diverged (metrics.py failure: a
non-finite loss or gradient norm, an outcome of a learning-rate sweep's high end, arXiv 2309.14322 §3.1); else
training, then each checkpoint's evaluation, each in a fresh process (torch.compile's recompile limit and the CUDA
caching allocator's reservations are per process), their output appended to RUNS/<run_name>/run.log. A training
process that fails without a failure record raises; one that diverged ends the job, recorded as diverged. A stopped run
resumes from its newest checkpoint. This process plans on the meta device and never initializes CUDA. The module's top
level imports no torch, model, training or evaluation code (the functions import what they run): every worker of an
evaluation's sanitize pool (eval/code_eval.py sanitizers, a forkserver) imports this module, the phases' main module.

    python run.py experiment=<name>               (conf/experiment/<name>.yaml)
    python run.py num_hidden_layers=8 tokens_per_param=0.5 tokenizer=sa-bpe   (one run)
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

import metrics
from corpus.tokenizer import ABLATION, TOKENIZER, get_tokenizer
from paths import RESULTS, ROOT, RUNS

if TYPE_CHECKING:
    from train.schedule import RunPlan


@dataclass
class RunConfig:
    """A run: the base model's recipe (the defaults) except the fields set."""
    num_hidden_layers: int = 24
    # tokens per non-embedding parameter; None: train/stages.py max_tokens_per_param of the data
    tokens_per_param: float | None = None
    seed: int = 0                   # initialization and data order
    tokenizer: str = "corpus"       # "corpus": CORPUS/tokenizer; else an arm: CORPUS/tokenizer-ablation/<arm>
    learning_rate_scale: float = 1.0  # times the law's peak learning rate (schedule.plan)
    muon_tau_scale: float = 1.0     # Muon matrices' tau x scale (their weight decay / scale; train/optim.py)
    # AdamW groups' learning rate x scale, their weight decay / scale (decay timescale held)
    adamw_learning_rate_scale: float = 1.0
    # output logits cap * tanh(logits / cap); None: no cap (model/modeling.py)
    final_logit_softcapping: float | None = None
    num_checkpoints: int = 1        # evenly spaced checkpoints, the final among them; > 1 adds repo_level_start
    generate: str = "parse"         # suite.evaluate's generate: "parse" (ablations) or "benchmarks" (base)


def run_name(config: RunConfig) -> str:
    """The fields that differ from RunConfig()'s, sorted, k=v joined by "," (floats by repr); "defaults" if none. A
    field added with a default renames no run."""
    default = RunConfig()
    changed = sorted((f.name, getattr(config, f.name)) for f in fields(config)
                     if getattr(config, f.name) != getattr(default, f.name))
    return ",".join(f"{name}={value!r}" if isinstance(value, float) else f"{name}={value}"
                    for name, value in changed) or "defaults"


def tokenizer_path(config: RunConfig) -> Path:
    return TOKENIZER if config.tokenizer == "corpus" else ABLATION / config.tokenizer


def base_plan(*, num_hidden_layers: int, tokens_per_param: float) -> "RunPlan":
    """The base model's plan at num_hidden_layers and tokens per non-embedding parameter (schedule.plan)."""
    import torch

    from corpus.mixture import FILE_LEVEL_STAGE
    from model.modeling import CausalLM, model_config
    from train.schedule import plan

    with torch.device("meta"):
        return plan(model=CausalLM(model_config(num_hidden_layers=num_hidden_layers, stage=FILE_LEVEL_STAGE)),
                    tokens_per_param=tokens_per_param)


def plan_of(config: RunConfig) -> "RunPlan":
    """The base model's plan at the run's num_hidden_layers and tokens_per_param (its data's max_tokens_per_param if
    None), at the run's peak learning rate."""
    from train.stages import max_tokens_per_param

    tokens_per_param = (max_tokens_per_param(get_tokenizer(tokenizer_path(config)),
                                             num_hidden_layers=config.num_hidden_layers)
                        if config.tokens_per_param is None else config.tokens_per_param)
    base = base_plan(num_hidden_layers=config.num_hidden_layers, tokens_per_param=tokens_per_param)
    return replace(base, learning_rate=config.learning_rate_scale * base.learning_rate)


def checkpoint_steps(config: RunConfig, run: "RunPlan") -> frozenset[int]:
    """config.num_checkpoints evenly spaced step counts, the last the final model's; with more than one, also
    repo_level_start (the model before the repository stage)."""
    n = config.num_checkpoints
    steps = {round(run.max_steps * k / n) for k in range(1, n + 1)}
    return frozenset(steps | {run.repo_level_start} if n > 1 else steps)


def training(config: RunConfig, run: "RunPlan") -> None:
    import torch

    from corpus.mixture import FILE_LEVEL_STAGE
    from model.modeling import CausalLM, model_config
    from train.loop import train

    with torch.device("meta"):
        model = CausalLM(replace(model_config(num_hidden_layers=config.num_hidden_layers, stage=FILE_LEVEL_STAGE),
                                 final_logit_softcapping=config.final_logit_softcapping))
    train(model=model, run=run, tokenizer_path=tokenizer_path(config), seed=config.seed,
          muon_tau_scale=config.muon_tau_scale, adamw_learning_rate_scale=config.adamw_learning_rate_scale,
          checkpoints=checkpoint_steps(config, run), run_name=run_name(config), config=asdict(config))


def evaluation(config: RunConfig, step: int) -> None:
    import torch

    from eval import suite
    from train.checkpoint import load_model

    # SyntaxWarnings of ast.parse on generated code (eval/verify.py, inference/)
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
    """A run's job (module docstring)."""
    from eval.suite import evaluation_path

    name, run = run_name(config), plan_of(config)
    data_key = get_tokenizer(tokenizer_path(config)).data_key
    if any(r["plan"] != asdict(run) or r["data_key"] != data_key for r in metrics.records(name) if r["kind"] == "run"):
        raise RuntimeError(f"recipe changed since {name} ran: delete {RUNS / name}")
    if evaluation_path(name, run.max_steps).exists() or metrics.diverged(name) is not None:
        return
    log = RUNS / name / "run.log"
    if phase(log, training, config=config, run=run):
        if metrics.diverged(name) is not None:
            return
        raise ChildProcessError(f"{name}: training failed ({log})")
    for step in sorted(checkpoint_steps(config, run)):
        if not evaluation_path(name, step).exists() and phase(log, evaluation, config=config, step=step):
            raise ChildProcessError(f"{name}: evaluation of step {step} failed ({log})")


def lr_multiplier() -> float:
    """The base model's multiplier of the law's peak lr: ablations/peak_lr.py's estimator over its recorded runs."""
    from ablations.peak_lr import analysis

    lines = [json.loads(line) for line in (RESULTS / "peak_lr.jsonl").read_text(encoding="utf-8").splitlines()]
    return analysis(lines)["base_multiplier"]


OmegaConf.register_new_resolver("runs", lambda: str(RUNS))
OmegaConf.register_new_resolver(
    "run_name", lambda *, _root_: run_name(RunConfig(**{f.name: _root_[f.name] for f in fields(RunConfig)})))
OmegaConf.register_new_resolver("lr_multiplier", lr_multiplier, use_cache=True)
ConfigStore.instance().store(name="run_schema", node=RunConfig)


@hydra.main(version_base="1.3", config_path=str(ROOT / "conf"), config_name="run")
def main(cfg: DictConfig) -> None:
    job(OmegaConf.to_object(cfg))


if __name__ == "__main__":
    main()
