"""
The training loop. train() pretrains (or resumes) a run: the file-level stage, then the repo-level stage from
repo_start; next-token loss with the packs' FIM pieces read as FIM examples (DeepSeek-Coder, arXiv 2401.14196 §3.1;
corpus/pack.py). Each start resumes from the tag's newest checkpoint (weights, optimizer state, data position). What a
step computes is this package's: the optimizers, weight decay and gradient clip (optim.py), each step's stage and lr
(schedule.py), the stages' micro-batches in the run's data order (stages.py), the micro-batches' loss weights
(ops/cross_entropy.py step_shares). Caller: nanochat/run.py. Metrics: metrics.py.
"""

import gc
import math
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

from nanochat.corpus.tokenizer import get_tokenizer
from nanochat.model.gpt import GPT, non_embedding_flops_per_token
from nanochat.ops.cross_entropy import step_shares
from nanochat.train import metrics
from nanochat.train.checkpoint import checkpoint_path, save_checkpoint
from nanochat.train.optim import MAX_NORM, optimizers, tau_weight_decay, update
from nanochat.train.schedule import RunPlan, lr_multiplier
from nanochat.train.stages import stage_batches


def train(*, model: GPT, run: RunPlan, vocabulary: Path, seed: int, muon_tau_scale: float, adamw_lr_scale: float,
          checkpoints: frozenset[int], tag: str, config: dict) -> None:
    """Train (or resume) run `tag` of `model` (shapes on the meta device, at FILE_STAGE's context) by the plan `run`
    (schedule.plan at the caller's peak lr) on the packed data of the vocabulary in directory `vocabulary`
    (corpus/tokenizer.py get_tokenizer): initialization and data order by `seed`, decoupled weight decay by the tau
    law at the run's lr (optim.tau_weight_decay; Muon's and AdamW's by optim.optimizers' scales), each update clipped
    (optim.update), the stages by run.stage_at; `config` (nanochat/run.py RunConfig as a dict) and the vocabulary's
    data key recorded with the run. A checkpoint after each step count in `checkpoints` (run.num_iterations among
    them: the final model); a run resumes from its newest."""
    device = torch.device("cuda")
    tokenizer = get_tokenizer(vocabulary)
    model.to_empty(device=device)
    model.init_weights(seed=seed)
    opts = optimizers(model, lr=run.lr, weight_decay=tau_weight_decay(lr=run.lr, steps=run.num_iterations,
                                                                      tokens_per_param=run.tokens_per_param),
                      muon_tau_scale=muon_tau_scale, adamw_lr_scale=adamw_lr_scale)
    saved = [s for s in sorted(checkpoints) if checkpoint_path(tag, s).exists()]
    start_step = stage_rows = 0
    if saved:  # resume: weights into the optimizers' parameters, then optimizer state
        start_step = saved[-1]
        checkpoint = torch.load(checkpoint_path(tag, start_step), map_location=device)
        model.load_state_dict(checkpoint["model"])
        for opt, state in zip(opts, checkpoint["optimizers"], strict=True):
            opt.load_state_dict(state)
        stage_rows = checkpoint["stage_rows"]
        del checkpoint
    # group lr = initial lr x lr_multiplier(step); a resumed scheduler restarts at start_step
    schedulers = [LambdaLR(opt, partial(lr_multiplier, steps=run.num_iterations), last_epoch=start_step - 1)
                  for opt in opts]
    step = start_step

    # A checkpoint holds the context its weights trained in; stage_rows counts that stage's consumed rows
    stage = run.stage_at(start_step - 1)
    model.set_stage(stage)
    if start_step == run.num_iterations:  # finished
        return
    compiled = torch.compile(model, dynamic=False, fullgraph=True)
    read = partial(stage_batches, tokenizer, run=run, seed=seed, device=device)

    # First step's micro-batches: continue from stage_rows if it stays in the checkpoint's stage
    first = run.stage_at(start_step)
    mixture, train_iter = read(stage=first, start=stage_rows if first is stage else 0)
    batches = [next(train_iter) for _ in range(run.grad_accum_steps)]
    # the attempt this process is (the run's process starts before it); lr and weight decay as the optimizers hold them
    # (a resumed run's come from its checkpoint)
    attempt = sum(r["kind"] == "run" for r in metrics.records(tag))
    metrics.append(tag, {"kind": "run", "step": start_step, "plan": asdict(run), "seed": seed, "config": config,
                         "data_key": tokenizer.data_key, "model": asdict(model.config),
                         "params": sum(p.numel() for p in model.parameters()),
                         "flops_per_token": non_embedding_flops_per_token(model),
                         "max_norm": MAX_NORM,
                         "lr": {name: [g["initial_lr"] for g in opt.param_groups]
                                for name, opt in zip(("adamw", "muon"), opts, strict=True)},
                         "weight_decay": {name: [g["weight_decay"] for g in opt.param_groups]
                                          for name, opt in zip(("adamw", "muon"), opts, strict=True)},
                         "stage": asdict(first) | {"mixture": mixture}}, attempt=attempt)

    while True:
        # state after `step` steps at the stage they trained in
        if step != start_step and step in checkpoints:
            save_checkpoint(checkpoint_path(tag, step), {"model": model.state_dict(),
                                                         "optimizers": [opt.state_dict() for opt in opts],
                                                         "model_config": asdict(model.config),
                                                         "stage_rows": stage_rows})
            gc.collect()
        if step == run.num_iterations:
            metrics.append(tag, {"kind": "end", "step": step, "peak_memory_bytes": torch.cuda.max_memory_allocated()},
                           attempt=attempt)
            return
        if run.stage_at(step) is not stage:
            # repo-level stage: new context and compiled graph; its batches were read the step before
            stage, stage_rows = run.stage_at(step), 0
            model.set_stage(stage)
            compiled = torch.compile(model, dynamic=False, fullgraph=True)
            metrics.append(tag, {"kind": "stage", "step": step, "stage": asdict(stage) | {"mixture": mixture}},
                           attempt=attempt)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # the next step's micro-batches are read during this step's backward (the repo-level stage's if the next step
        # starts it), none after the last step
        reading = step + 1 < run.num_iterations
        if reading and run.stage_at(step + 1) is not stage:
            mixture, train_iter = read(stage=run.stage_at(step + 1), start=0)
        lr = schedulers[0].get_last_lr()[0]
        losses, upcoming = [], []
        for (x, y, cu_seqlens, segments), share in zip(batches, step_shares([b[1] for b in batches]), strict=True):
            loss = compiled(x, y, cu_seqlens, segments) * share
            losses.append(loss.detach())
            loss.backward()
            if reading:
                upcoming.append(next(train_iter))
        grad_norm = update(model, opts)
        for scheduler in schedulers:
            scheduler.step()
        stage_rows += len(batches) * stage.batch_rows
        batches = upcoming
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        # one host synchronization for loss and grad norm
        loss_value, grad_norm_value = torch.stack([torch.stack(losses).double().sum(), grad_norm.double()]).tolist()
        if not (math.isfinite(loss_value) and math.isfinite(grad_norm_value)):
            metrics.append(tag, {"kind": "failure", "step": step, "loss": None, "grad_norm": None}, attempt=attempt)
            raise FloatingPointError(f"{tag} step {step}: loss {loss_value}, grad_norm {grad_norm_value}")
        metrics.append(tag, {"kind": "train", "step": step, "tokens": (step + 1) * run.total_batch_size,
                             "loss": loss_value, "lr": lr, "grad_norm": grad_norm_value,
                             "tok_per_s": run.total_batch_size / dt}, attempt=attempt)
        if step == start_step:
            # no cyclic garbage per step: keep the collector off after setup (its scans are slow) except after saves
            gc.collect()
            gc.freeze()
            gc.disable()
        step += 1
