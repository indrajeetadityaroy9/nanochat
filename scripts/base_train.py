"""
Pretrain a base model:

    python -m scripts.base_train --depth=<depth> --target-param-data-ratio=<tokens per scaling parameter> --seed=<seed>
        --save-every=<steps> --snapshot-every=<steps>

Everything else is derived (config.py, train/schedule.py, train/optim.py). File-level stage, then the repo-level stage
from repo_start; next-token loss with the packs' FIM pieces read as FIM examples (DeepSeek-Coder, arXiv 2401.14196 §3.1;
data/pack.py). Each start resumes from the tag's newest checkpoint (weights, optimizer state, data position).
Metrics: train/metrics.py.
"""

import argparse
import gc
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict
from functools import partial

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from nanochat.config import CODE, FILE_STAGE, REPO_STAGE, run_tag
from nanochat.data.loader import PretrainingBatches, mixture_weights
from nanochat.data.tokenizer import get_tokenizer
from nanochat.model.gpt import GPT, gpt_config
from nanochat.train import metrics
from nanochat.train.checkpoint import checkpoint_path, save_checkpoint
from nanochat.train.optim import optimizers, tau_weight_decay
from nanochat.train.schedule import RunPlan, lr_multiplier, plan
from nanochat.train.stages import readable_rows


def optimizer_step(model: torch.nn.Module, batches: list[tuple[torch.Tensor, ...]], params: list[torch.Tensor],
                   optimizers: tuple[torch.optim.Optimizer, ...], reads: Iterator | None, *,
                   max_norm: float) -> tuple[torch.Tensor, torch.Tensor, list]:
    """One optimizer step over micro-batches (inputs, targets, cu_seqlens). Each micro-batch's mean loss is weighted by
    its share of the step's valid targets, so the step loss is the mean over all. The next step's micro-batches are read
    from `reads` during backward (None on the last step); gradients clipped to global norm `max_norm` (inf: none).
    Returns (step loss fp64, gradient norm before clipping, next micro-batches) on device, without host
    synchronization."""
    valid = torch.stack([(y != PretrainingBatches.PAD).sum() for _, y, _ in batches]).double()
    shares = (valid / valid.sum()).float()
    losses, upcoming = [], []
    for micro_step, (x, y, cu_seqlens) in enumerate(batches):
        loss = model(x, y, cu_seqlens) * shares[micro_step]
        losses.append(loss.detach())
        loss.backward()
        if reads is not None:
            upcoming.append(next(reads))
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    for opt in optimizers:
        opt.step()
    model.zero_grad()
    return torch.stack(losses).double().sum(), grad_norm, upcoming


def train(*, model: GPT, run: RunPlan, weight_decay: float, lr_schedule: Callable[[int], float], max_norm: float,
          variant: dict, seed: int, data_seed: int, save_every: int, snapshot_every: int, tag: str) -> GPT:
    """Train (or resume) run `tag` of `model` (shapes on the meta device, at FILE_STAGE's context) by the plan `run`
    (schedule.plan) and return it: init by `seed`, data order by `data_seed`, decoupled `weight_decay`, each step's lr
    the plan's times `lr_schedule(step)`, gradients clipped to global norm `max_norm` (inf: none), FILE_STAGE then
    REPO_STAGE from run.repo_start; `variant` (config.run_tag's) recorded with the run. Resume checkpoint every
    `save_every` steps; a model-only snapshot every `snapshot_every` steps and at repo_start (the model before the
    repo-level stage), the final model being the last checkpoint."""
    device = torch.device("cuda")
    tokenizer = get_tokenizer()
    model.to_empty(device=device)
    model.init_weights(seed=seed)
    opts = optimizers(model, lr=run.lr, weight_decay=weight_decay)
    path = checkpoint_path(tag, step=None)
    start_step = stage_rows = 0
    if path.exists():  # resume: weights into the optimizers' parameters, then optimizer state
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        for opt, state in zip(opts, checkpoint["optimizers"], strict=True):
            opt.load_state_dict(state)
        start_step, stage_rows = checkpoint["step"], checkpoint["stage_rows"]
        del checkpoint
    # a resumed scheduler restarts at start_step
    schedulers = [LambdaLR(opt, lr_schedule, last_epoch=start_step - 1) for opt in opts]
    step = start_step

    def stage_at(s):
        """Repo-level stage from repo_start on."""
        return REPO_STAGE if s >= run.repo_start else FILE_STAGE

    # A checkpoint holds the context its weights trained in; stage_rows counts that stage's consumed rows
    stage = stage_at(start_step - 1)
    model.set_stage(stage)
    if start_step == run.num_iterations:  # finished: the final checkpoint's model
        return model
    compiled = torch.compile(model, dynamic=False, fullgraph=True)

    # A stage's packed rows (train/stages.py), each corpus permuted by the data seed (RandomState's stream is frozen
    # across numpy versions, so a resume sees the same order); FIM pieces read as FIM examples
    def stage_batches(stage, start):
        mixture = mixture_weights(stage.mixture, tokenizer, stage.seq_len)
        readable = readable_rows(tokenizer, stage=stage, run=run)
        orders = {corpus: np.random.RandomState(data_seed).permutation(readable[corpus]) for corpus in mixture}
        batches = PretrainingBatches(mixture, "train", tokenizer, orders, seq_len=stage.seq_len,
                                     batch_rows=stage.batch_rows, device=device, reading="fim", start=start)
        return mixture, iter(batches)

    # First step's micro-batches: continue from stage_rows if it stays in the checkpoint's stage
    first = stage_at(start_step)
    mixture, train_iter = stage_batches(first, stage_rows if first is stage else 0)
    batches = [next(train_iter) for _ in range(run.grad_accum_steps)]
    # lr and weight decay as the optimizers hold them (a resumed run's come from its checkpoint)
    group = opts[0].param_groups[0]
    attempt = metrics.attempt(tag)
    metrics.append(tag, {"kind": "run", "step": start_step, "plan": asdict(run), "seed": seed, "data_seed": data_seed,
                         "model": asdict(model.config), "variant": variant,
                         "max_norm": None if math.isinf(max_norm) else max_norm,
                         "lr": group["initial_lr"], "weight_decay": group["weight_decay"],
                         "stage": asdict(first) | {"mixture": mixture}}, attempt=attempt)
    params = [p for p in model.parameters() if p.requires_grad]

    while True:
        last_step = step == run.num_iterations
        # the snapshot first: a kill while writing it leaves the previous checkpoint, so the resumed run writes it again
        if step != start_step and not last_step and (step % snapshot_every == 0 or step == run.repo_start):
            save_checkpoint(checkpoint_path(tag, step=step), {"step": step, "model": model.state_dict(),
                                                              "model_config": asdict(model.config)})
        # state after `step` steps at the stage they trained in
        if step != start_step and (last_step or step % save_every == 0 or step == run.repo_start):
            save_checkpoint(path, {"step": step, "model": model.state_dict(),
                                   "optimizers": [opt.state_dict() for opt in opts],
                                   "model_config": asdict(model.config), "stage_rows": stage_rows})
            gc.collect()
        if last_step:
            return model
        if stage_at(step) is not stage:
            # repo-level stage: new context and compiled graph; its batches were read the step before
            stage, stage_rows = REPO_STAGE, 0
            model.set_stage(stage)
            compiled = torch.compile(model, dynamic=False, fullgraph=True)
            metrics.append(tag, {"kind": "stage", "step": step, "stage": asdict(stage) | {"mixture": mixture}},
                           attempt=attempt)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # next step's micro-batches are read during this step (the repo-level stage's if the next step starts it)
        if step + 1 < run.num_iterations and stage_at(step + 1) is not stage:
            mixture, train_iter = stage_batches(REPO_STAGE, 0)
        lr = schedulers[0].get_last_lr()[0]
        loss, grad_norm, upcoming = optimizer_step(compiled, batches, params, opts,
                                                   train_iter if step + 1 < run.num_iterations else None,
                                                   max_norm=max_norm)
        for scheduler in schedulers:
            scheduler.step()
        stage_rows += len(batches) * stage.batch_rows
        batches = upcoming
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        # one host synchronization for loss and grad norm
        loss_value, grad_norm_value = torch.stack([loss, grad_norm.double()]).tolist()
        if not (math.isfinite(loss_value) and math.isfinite(grad_norm_value)):
            metrics.append(tag, {"kind": "failure", "step": step, "loss": None, "grad_norm": None}, attempt=attempt)
            raise FloatingPointError(f"{tag} step {step}: loss {loss_value}, grad_norm {grad_norm_value}")
        metrics.append(tag, {"kind": "train", "step": step, "tokens": (step + 1) * run.total_batch_size,
                             "loss": loss_value, "lr": lr, "grad_norm": grad_norm_value,
                             "tok_per_s": run.total_batch_size / dt}, attempt=attempt)
        if step == start_step:
            # no cyclic garbage per step: keep the collector off after setup (scans cost ~500 ms) except after saves
            gc.collect()
            gc.freeze()
            gc.disable()
        step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--target-param-data-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--save-every", type=int, required=True)
    parser.add_argument("--snapshot-every", type=int, required=True)
    args = parser.parse_args()
    with torch.device("meta"):
        model = GPT(gpt_config(depth=args.depth, stage=FILE_STAGE))
    run = plan(model=model, ratio=args.target_param_data_ratio)
    # no clipping (docs/h100-pretraining.md Regime): an infinite threshold only measures the norm
    train(model=model, run=run,
          weight_decay=tau_weight_decay(lr=run.lr, steps=run.num_iterations, tokens_per_param=run.tokens_per_param),
          lr_schedule=partial(lr_multiplier, steps=run.num_iterations), max_norm=math.inf, variant={},
          seed=args.seed, data_seed=args.seed, save_every=args.save_every, snapshot_every=args.snapshot_every,
          tag=run_tag(depth=args.depth, ratio=args.target_param_data_ratio, seed=args.seed, data_seed=args.seed,
                      variant={}, code=CODE))


if __name__ == "__main__":
    main()
