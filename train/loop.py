"""
The training loop. train() trains (or resumes) a run: the file-level stage, then the repo-level stage from
repo_level_start; next-token loss with the packs' FIM pieces read as FIM examples (DeepSeek-Coder, arXiv 2401.14196
§3.1; corpus/pack.py). Each start resumes from the run's newest checkpoint (weights, optimizer state, data position).
What a step computes is this package's: the optimizers, weight decay and gradient clip (optim.py), each step's stage
and learning rate (schedule.py), the stages' micro-batches in the run's data order (stages.py), the micro-batches'
loss weights (model/ops/cross_entropy.py step_shares). Caller: run.py. Metrics: metrics.py.
"""

import gc
import math
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

import metrics
from corpus.tokenizer import get_tokenizer
from model.modeling import CausalLM, non_embedding_flops_per_token
from model.ops.cross_entropy import step_shares
from train.checkpoint import checkpoint_path, save_checkpoint
from train.optim import MAX_GRAD_NORM, optimizers, tau_weight_decay, update
from train.schedule import RunPlan, lr_lambda
from train.stages import stage_batches


def train(*, model: CausalLM, run: RunPlan, tokenizer_path: Path, seed: int, muon_tau_scale: float,
          adamw_learning_rate_scale: float, checkpoints: frozenset[int], run_name: str, config: dict) -> None:
    """Train (or resume) run `run_name` of `model` (shapes on the meta device, at FILE_LEVEL_STAGE's context) by the
    plan `run` (schedule.plan at the caller's peak learning rate) on the packed data of the tokenizer in directory
    `tokenizer_path` (corpus/tokenizer.py get_tokenizer): initialization and data order by `seed`, decoupled weight
    decay by the tau law at the run's learning rate (optim.tau_weight_decay; Muon's and AdamW's by optim.optimizers'
    scales), each update clipped (optim.update), the stages by run.stage_at; `config` (run.py RunConfig as a dict)
    and the tokenizer's data key recorded with the run. A checkpoint after each step count in `checkpoints`
    (run.max_steps among them: the final model); a run resumes from its newest."""
    device = torch.device("cuda")
    tokenizer = get_tokenizer(tokenizer_path)
    model.to_empty(device=device)
    model.init_weights(seed=seed)
    opts = optimizers(model, lr=run.learning_rate,
                      weight_decay=tau_weight_decay(lr=run.learning_rate, steps=run.max_steps,
                                                    tokens_per_param=run.tokens_per_param),
                      muon_tau_scale=muon_tau_scale, adamw_learning_rate_scale=adamw_learning_rate_scale)
    saved = [s for s in sorted(checkpoints) if checkpoint_path(run_name, s).exists()]
    start_step = stage_rows = 0
    if saved:  # resume: weights into the optimizers' parameters, then optimizer state
        start_step = saved[-1]
        checkpoint = torch.load(checkpoint_path(run_name, start_step), map_location=device)
        model.load_state_dict(checkpoint["model"])
        for opt, state in zip(opts, checkpoint["optimizers"], strict=True):
            opt.load_state_dict(state)
        stage_rows = checkpoint["stage_rows"]
        del checkpoint
    # group lr = initial lr x lr_lambda(step); a resumed scheduler restarts at start_step
    schedulers = [LambdaLR(opt, partial(lr_lambda, num_training_steps=run.max_steps), last_epoch=start_step - 1)
                  for opt in opts]
    step = start_step

    # A checkpoint holds the context its weights trained in; stage_rows counts that stage's consumed rows
    stage = run.stage_at(start_step - 1)
    model.set_stage(stage)
    if start_step == run.max_steps:  # finished
        return
    compiled = torch.compile(model, dynamic=False, fullgraph=True)
    read = partial(stage_batches, tokenizer, run=run, seed=seed, device=device)

    # First step's micro-batches: continue from stage_rows if it stays in the checkpoint's stage
    first = run.stage_at(start_step)
    mixture, train_iter = read(stage=first, start=stage_rows if first is stage else 0)
    batches = [next(train_iter) for _ in range(run.gradient_accumulation_steps)]
    # the attempt this process is (the run's process starts before it); learning rate and weight decay as the
    # optimizers hold them (a resumed run's come from its checkpoint)
    attempt = sum(r["kind"] == "run" for r in metrics.records(run_name))
    metrics.append(run_name, {"kind": "run", "step": start_step, "plan": asdict(run), "seed": seed, "config": config,
                              "data_key": tokenizer.data_key, "model": asdict(model.config),
                              "num_parameters": sum(p.numel() for p in model.parameters()),
                              "flops_per_token": non_embedding_flops_per_token(model),
                              "max_grad_norm": MAX_GRAD_NORM,
                              "learning_rate": {name: [g["initial_lr"] for g in opt.param_groups]
                                                for name, opt in zip(("adamw", "muon"), opts, strict=True)},
                              "weight_decay": {name: [g["weight_decay"] for g in opt.param_groups]
                                               for name, opt in zip(("adamw", "muon"), opts, strict=True)},
                              "stage": asdict(first) | {"mixture": mixture}}, attempt=attempt)

    while True:
        # state after `step` steps at the stage they trained in
        if step != start_step and step in checkpoints:
            save_checkpoint(checkpoint_path(run_name, step), {"model": model.state_dict(),
                                                              "optimizers": [opt.state_dict() for opt in opts],
                                                              "config": asdict(model.config),
                                                              "stage_rows": stage_rows})
            gc.collect()
        if step == run.max_steps:
            metrics.append(run_name, {"kind": "end", "step": step,
                                      "peak_memory_bytes": torch.cuda.max_memory_allocated()}, attempt=attempt)
            return
        if run.stage_at(step) is not stage:
            # repo-level stage: new context and compiled graph; its batches were read the step before
            stage, stage_rows = run.stage_at(step), 0
            model.set_stage(stage)
            compiled = torch.compile(model, dynamic=False, fullgraph=True)
            metrics.append(run_name, {"kind": "stage", "step": step, "stage": asdict(stage) | {"mixture": mixture}},
                           attempt=attempt)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # the next step's micro-batches are read during this step's backward (the repo-level stage's if the next step
        # starts it), none after the last step
        reading = step + 1 < run.max_steps
        if reading and run.stage_at(step + 1) is not stage:
            mixture, train_iter = read(stage=run.stage_at(step + 1), start=0)
        learning_rate = schedulers[0].get_last_lr()[0]
        losses, upcoming = [], []
        for (input_ids, labels, cu_seqlens, segments), share in zip(batches, step_shares([b[1] for b in batches]),
                                                                     strict=True):
            loss = compiled(input_ids, labels, cu_seqlens, segments) * share
            losses.append(loss.detach())
            loss.backward()
            if reading:
                upcoming.append(next(train_iter))
        grad_norm = update(model, opts)
        for scheduler in schedulers:
            scheduler.step()
        stage_rows += len(batches) * stage.micro_batch_size
        batches = upcoming
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        # one host synchronization for loss and grad norm
        loss_value, grad_norm_value = torch.stack([torch.stack(losses).double().sum(), grad_norm.double()]).tolist()
        if not (math.isfinite(loss_value) and math.isfinite(grad_norm_value)):
            metrics.append(run_name, {"kind": "failure", "step": step, "loss": None, "grad_norm": None},
                           attempt=attempt)
            raise FloatingPointError(f"{run_name} step {step}: loss {loss_value}, grad_norm {grad_norm_value}")
        metrics.append(run_name, {"kind": "train", "step": step, "num_input_tokens_seen": (step + 1) * run.batch_size,
                                  "loss": loss_value, "learning_rate": learning_rate, "grad_norm": grad_norm_value,
                                  "tokens_per_second": run.batch_size / dt}, attempt=attempt)
        if step == start_step:
            # no cyclic garbage per step: keep the collector off after setup (its scans are slow) except after saves
            gc.collect()
            gc.freeze()
            gc.disable()
        step += 1
