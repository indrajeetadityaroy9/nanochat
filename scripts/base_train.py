"""
Pretrain a base model:

    python -m scripts.base_train --depth=<depth> --target-param-data-ratio=<tokens per scaling parameter> --seed=<seed>
        --save-every=<steps>

Everything else is derived (config.py, train/schedule.py, train/optim.py). The file-level stage, then from repo_start
the repo-level stage; next-token loss with the packs' FIM pieces read as FIM examples (DeepSeek-Coder, arXiv 2401.14196
§3.1; data/pack.py). Every start resumes from the tag's newest checkpoint (weights, optimizer state, data position).
Metrics: train/metrics.py.
"""

import argparse
import gc
import math
import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields, replace
from functools import partial
from typing import Literal

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from nanochat.config import FILE_STAGE, REPO_STAGE, run_tag
from nanochat.data.loader import PretrainingBatches, mixture_weights
from nanochat.data.tokenizer import get_tokenizer
from nanochat.model.gpt import GPT, Mechanism, gpt_config
from nanochat.train import metrics
from nanochat.train.checkpoint import checkpoint_path, save_checkpoint
from nanochat.train.optim import GRAD_CLIP, WEIGHT_DECAY, optimizers, tau_weight_decay
from nanochat.train.schedule import RunPlan, lr_multiplier, plan
from nanochat.train.stages import readable_rows


@dataclass(kw_only=True)
class Variant(Mechanism):
    """A study run's changes from the baseline (experiments/studies.py), baseline by default: the model's Mechanism;
    weight decay (optim.tau_weight_decay, or optim.WEIGHT_DECAY); clip (none, or global norm optim.GRAD_CLIP); the
    repository stage's lr schedule (schedule.lr_multiplier); a horizon in tokens (file-level stage only). The baseline
    regime carries no hand-set value the studies do not test: data-normalized weight decay, no clip."""
    weight_decay: Literal["tau", "fixed"] = "tau"
    clip: Literal["none", "global_norm"] = "none"
    repo_schedule: Literal["continue", "rewarm"] = "continue"
    tokens: float | None = None


def optimizer_settings(run: RunPlan, variant: Variant) -> tuple[float, float, float]:
    """(peak lr, weight decay, clip threshold): the plan's lr, the variant's weight decay at it, and GRAD_CLIP or inf
    (clip_grad_norm_ then only measures)."""
    weight_decay = (tau_weight_decay(lr=run.lr, steps=run.num_iterations, tokens_per_param=run.tokens_per_param)
                    if variant.weight_decay == "tau" else WEIGHT_DECAY)
    return run.lr, weight_decay, GRAD_CLIP if variant.clip == "global_norm" else math.inf


def optimizer_step(model: torch.nn.Module, batches: list[tuple[torch.Tensor, ...]], params: list[torch.Tensor],
                   optimizers: tuple[torch.optim.Optimizer, ...], reads: Iterator | None, *,
                   max_norm: float) -> tuple[torch.Tensor, torch.Tensor, list]:
    """One optimizer step on its micro-batches (inputs, targets, cu_seqlens): each micro-batch's mean loss weighted by
    its share of the step's valid targets (the step loss is the mean over all of them); the next step's micro-batches
    read from `reads` during the backward (None on the last step); clip, step, zero grads. Returns (step loss fp64,
    gradient norm before clipping, next micro-batches) on the device, without a host synchronization."""
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


def train(*, depth: int, ratio: float, seed: int, data_seed: int, save_every: int, tag: str, variant: Variant) -> GPT:
    """Train (or resume) the run `tag` and return its model: init by `seed`, data order by `data_seed`, the baseline
    model's plan at this depth and ratio (every run of a study reads the same tokens in the same steps). The checkpoint
    at repo_start, the model before the repo-level stage, is kept."""
    device = torch.device("cuda")
    tokenizer = get_tokenizer()
    run = plan(depth=depth, ratio=ratio, tokens=variant.tokens)
    lr, weight_decay, max_norm = optimizer_settings(run, variant)
    switches = {f.name: getattr(variant, f.name) for f in fields(Mechanism)}

    with torch.device("meta"):
        orig_model = GPT(replace(gpt_config(depth=depth, stage=FILE_STAGE), **switches))
    orig_model.to_empty(device=device)
    orig_model.init_weights(seed=seed)
    opts = optimizers(orig_model, lr=lr, weight_decay=weight_decay)
    path = checkpoint_path(tag, step=None)
    start_step = stage_rows = 0
    if os.path.exists(path):  # resume: weights into the parameters the optimizers hold, then the optimizer state
        checkpoint = torch.load(path, map_location=device)
        orig_model.load_state_dict(checkpoint["model"])
        for opt, state in zip(opts, checkpoint["optimizers"], strict=True):
            opt.load_state_dict(state)
        start_step, stage_rows = checkpoint["step"], checkpoint["stage_rows"]
        del checkpoint
    # each group's lr = its initial lr x lr_multiplier(step); a resumed scheduler restarts at start_step
    schedulers = [LambdaLR(opt, partial(lr_multiplier, steps=run.num_iterations, repo_start=run.repo_start,
                                        repo_schedule=variant.repo_schedule),
                           last_epoch=start_step - 1) for opt in opts]
    step = start_step

    def stage_at(s):
        """The repo-level stage from repo_start on (a run cut to a horizon has none)."""
        return REPO_STAGE if s >= run.repo_start else FILE_STAGE

    # the stage of the steps taken: a checkpoint holds the context its weights trained in, stage_rows the rows that
    # stage consumed
    stage = stage_at(start_step - 1)
    orig_model.set_stage(stage)
    if start_step == run.num_iterations:  # finished: the final checkpoint's model
        return orig_model
    model = torch.compile(orig_model, dynamic=False, fullgraph=True)

    # a stage's packed rows (train/stages.py), each corpus permuted by the data seed (RandomState: its stream is frozen
    # across numpy versions, so a resumed run sees the same order), FIM pieces read as FIM examples
    def stage_batches(stage, start):
        mixture = mixture_weights(stage.mixture, tokenizer, stage.seq_len)
        readable = readable_rows(tokenizer, stage=stage, run=run)
        orders = {corpus: np.random.RandomState(data_seed).permutation(readable[corpus]) for corpus in mixture}
        batches = PretrainingBatches(mixture, "train", tokenizer, orders, seq_len=stage.seq_len,
                                     batch_rows=stage.batch_rows, device=device, reading="fim", start=start)
        return mixture, iter(batches)

    # the first step's micro-batches: from the rows consumed when it continues the checkpoint's stage
    first = stage_at(start_step)
    mixture, train_iter = stage_batches(first, stage_rows if first is stage else 0)
    batches = [next(train_iter) for _ in range(run.grad_accum_steps)]
    # the lr and weight decay the optimizers hold (a resumed run's are its checkpoint's)
    group = opts[0].param_groups[0]
    metrics.append(tag, {"kind": "run", "step": start_step, "plan": asdict(run), "seed": seed, "data_seed": data_seed,
                         "variant": asdict(variant), "lr": group["initial_lr"], "weight_decay": group["weight_decay"],
                         "stage": asdict(stage) | {"mixture": mixture}})
    params = list(orig_model.parameters())

    while True:
        last_step = step == run.num_iterations
        # the state after `step` steps, at the stage they trained in; hard-linked at repo_start
        if step != start_step and (last_step or step % save_every == 0 or step == run.repo_start):
            save_checkpoint(path, {"step": step, "model": orig_model.state_dict(),
                                   "optimizers": [opt.state_dict() for opt in opts],
                                   "model_config": asdict(orig_model.config), "stage_rows": stage_rows})
            if step == run.repo_start:
                os.link(path, checkpoint_path(tag, step=step))
            gc.collect()
        if last_step:
            return orig_model
        if stage_at(step) is not stage:
            # repo-level stage: its context, a new compiled graph; its batches were read the step before
            stage, stage_rows = REPO_STAGE, 0
            orig_model.set_stage(stage)
            model = torch.compile(orig_model, dynamic=False, fullgraph=True)
            metrics.append(tag, {"kind": "stage", "step": step, "stage": asdict(stage) | {"mixture": mixture}})

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # the next step's micro-batches are read during this step (the repo-level stage's when the next step starts it)
        if step + 1 < run.num_iterations and stage_at(step + 1) is not stage:
            mixture, train_iter = stage_batches(REPO_STAGE, 0)
        lr = schedulers[0].get_last_lr()[0]
        loss, grad_norm, upcoming = optimizer_step(model, batches, params, opts,
                                                   train_iter if step + 1 < run.num_iterations else None,
                                                   max_norm=max_norm)
        for scheduler in schedulers:
            scheduler.step()
        stage_rows += len(batches) * stage.batch_rows
        batches = upcoming
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        # loss and grad norm in one host synchronization
        loss_value, grad_norm_value = torch.stack([loss, grad_norm.double()]).tolist()
        metrics.append(tag, {"kind": "train", "step": step, "tokens": (step + 1) * run.total_batch_size,
                             "loss": loss_value, "lr": lr, "grad_norm": grad_norm_value,
                             "tok_per_s": run.total_batch_size / dt})
        if step == start_step:
            # no cyclic garbage per step: the collector stays off after setup (its scans cost ~500 ms) except after saves
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
    args = parser.parse_args()
    train(depth=args.depth, ratio=args.target_param_data_ratio, seed=args.seed, data_seed=args.seed,
          save_every=args.save_every, tag=run_tag(depth=args.depth, ratio=args.target_param_data_ratio, seed=args.seed,
                                                  data_seed=args.seed, changes={}),
          variant=Variant())


if __name__ == "__main__":
    main()
