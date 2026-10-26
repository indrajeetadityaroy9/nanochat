"""
The run app (Hydra, conf/run.yaml). Every run, the base model's and each ablation's, is a RunConfig trained by
train/loop.py train and evaluated by eval/suite.py evaluate. Its plan (batch, steps, peak learning rate) is the base
model's at its num_hidden_layers and tokens_per_param (base_plan, schedule.plan), the lr times learning_rate_scale.
Every invocation is a sweep (hydra.mode MULTIRUN) run one job at a time: one job without sweep parameters, or an
experiment's arms x seeds (conf/experiment/<name>.yaml); the first failed job stops it; ablations/records.py writes the
records once every job has ended.

A run's directory is RUNS/<run_name> (paths.py), so a run two experiments share is one run. Its job (job): the stale
guard raises when a recorded run record's plan, tokenizer data key or config (same_config) differs from the current
one; the run is skipped if its final checkpoint's evaluation exists or it diverged (metrics.diverged: a non-finite loss
or gradient norm); else training, then each checkpoint's evaluation, each in a fresh process (torch.compile's
recompile limit and the CUDA caching allocator's reservations are per process), their output appended to
RUNS/<run_name>/run.log. A training process that fails without a failure record raises; one that diverged ends the
job. A stopped run resumes from its newest checkpoint. This process plans on the meta device and never initializes
CUDA. The module's top level imports no torch, model, training or evaluation code: the workers of eval/code_eval.py's
sanitize pool (a forkserver) import this module, the phases' main module.

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
    from model.modeling import CausalLM
    from train.schedule import RunPlan


@dataclass
class RunConfig:
    """A run: the base model's recipe (the defaults, learning_rate_scale's conf/run.yaml's) except the fields set."""
    num_hidden_layers: int = 24
    # tokens per non-embedding parameter; None: train/stages.py max_tokens_per_param of the data
    tokens_per_param: float | None = None
    seed: int = 0                   # initialization and data order
    tokenizer: str = "corpus"       # "corpus": CORPUS/tokenizer; else an arm: CORPUS/tokenizer-ablation/<arm>
    # times the law's peak learning rate (schedule.plan), 1.0 the law's own; conf/run.yaml's default is the base
    # model's multiplier (lr_multiplier), so a run's name carries it
    learning_rate_scale: float = 1.0
    # output logits cap * tanh(logits / cap) (model/modeling.py), cap 15 as karpathy/nanochat 92d63d4
    # nanochat/gpt.py:511-515; None: no cap
    final_logit_softcapping: float | None = 15.0
    num_checkpoints: int = 1        # evenly spaced checkpoints, the final among them; > 1 adds repo_level_start
    generate: str = "parse"         # suite.evaluate's generate: "parse" (ablations) or "benchmarks" (base)


def run_name(config: RunConfig) -> str:
    """The fields that differ from RunConfig()'s, sorted, k=v joined by ","; "defaults" if none. A field added with a
    default renames no run."""
    default = RunConfig()
    changed = sorted((f.name, getattr(config, f.name)) for f in fields(config)
                     if getattr(config, f.name) != getattr(default, f.name))
    return ",".join(f"{name}={value}" for name, value in changed) or "defaults"


def same_config(recorded: dict, config: RunConfig) -> bool:
    """Whether a run record's config (train/loop.py: the RunConfig it ran, as a dict) equals `config` field by field.
    run_name omits default-valued fields, so a changed default would keep a run's name under a new meaning. A field
    the record lacks matches iff `config` holds its RunConfig() default."""
    default = RunConfig()
    return all(recorded.get(f.name, getattr(default, f.name)) == getattr(config, f.name) for f in fields(config))


def tokenizer_path(config: RunConfig) -> Path:
    return TOKENIZER if config.tokenizer == "corpus" else ABLATION / config.tokenizer


def meta_model(num_hidden_layers: int, final_logit_softcapping: float | None) -> "CausalLM":
    """The model on the meta device (no memory) at the file-level stage's context: what plans and training build."""
    import torch

    from corpus.mixture import FILE_LEVEL_STAGE
    from model.modeling import CausalLM, model_config

    config = model_config(num_hidden_layers=num_hidden_layers, stage=FILE_LEVEL_STAGE)
    with torch.device("meta"):
        return CausalLM(replace(config, final_logit_softcapping=final_logit_softcapping))


def base_plan(*, num_hidden_layers: int, tokens_per_param: float) -> "RunPlan":
    """The base model's plan at num_hidden_layers and tokens per non-embedding parameter (schedule.plan)."""
    from train.schedule import plan

    return plan(model=meta_model(num_hidden_layers, None), tokens_per_param=tokens_per_param)


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
    from train.loop import train

    model = meta_model(config.num_hidden_layers, config.final_logit_softcapping)
    train(model=model, run=run, tokenizer_path=tokenizer_path(config), seed=config.seed,
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
    log = metrics.records(name)
    if any(r["plan"] != asdict(run) or r["data_key"] != data_key or not same_config(r["config"], config)
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
    """The base model's multiplier of the law's peak lr, every run's default learning_rate_scale (conf/run.yaml):
    ablations/peak_lr.py's estimator over its recorded runs."""
    from ablations.peak_lr import analysis

    lines = [json.loads(line) for line in (RESULTS / "peak_lr.jsonl").read_text(encoding="utf-8").splitlines()]
    return analysis(lines)["base_multiplier"]


def register() -> None:
    """The run app's resolvers and schema (conf/run.yaml), once, in the app's process only: modules the app imports
    (ablations/peak_lr.py, ablations/records.py) import this file again as `run`, whose registrations would fail."""
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
