"""
Training loop: next-token loss over the file-level, then the repo-level stage, the packs' FIM pieces read as FIM
examples (DeepSeek-Coder, arXiv 2401.14196 §3.1); each start resumes from the run's newest checkpoint.
"""

import gc
import math
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

import metrics
from corpus.tokenizer import get_tokenizer
from model.modeling import CausalLM, accumulated_weights, non_embedding_flops_per_token, num_matmul_params
from model.ops.cross_entropy import step_shares
from train.checkpoint import checkpoint_path, save_checkpoint
from train.diagnostics import Diagnostics
from train.optim import MAX_GRAD_NORM, optimizers, tau_weight_decay, update
from train.schedule import RunPlan, lr_lambda
from train.stages import stage_batches


def train(*, model: CausalLM, run: RunPlan, tokenizer_path: Path, seed: int, checkpoints: frozenset[int],
          run_name: str, config: dict) -> None:
    """Train (or resume) run `run_name` of `model` (meta-device shapes at the file-level context) by plan `run` on CUDA.
    Checkpoints after each step count in `checkpoints` (run.max_steps among them); `config` (RunConfig as a dict) and
    the tokenizer's data key are recorded with the run."""
    device = torch.device("cuda")
    tokenizer = get_tokenizer(tokenizer_path)
    model.to_empty(device=device)
    model.init_weights(seed=seed)
    # fp32 weight gradients the GEMMs accumulate into across micro-batches (model/ops/linear.py); update zeroes them
    accumulated = accumulated_weights(model)
    for p in accumulated:
        p.grad = torch.zeros_like(p)
    opts = optimizers(model, lr=run.learning_rate,
                      weight_decay=tau_weight_decay(lr=run.learning_rate, steps=run.max_steps,
                                                    tokens_per_param=run.tokens_per_param))
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
    if start_step == run.max_steps:
        return
    compiled = torch.compile(model, dynamic=False, fullgraph=True)
    read = partial(stage_batches, tokenizer, run=run, seed=seed, device=device)

    # First step's micro-batches: continue from stage_rows if it stays in the checkpoint's stage
    first = run.stage_at(start_step)
    mixture, train_iter = read(stage=first, start=stage_rows if first is stage else 0)
    batches = [next(train_iter) for _ in range(run.gradient_accumulation_steps)]
    diagnostics = Diagnostics(model, opts, gradient_accumulation_steps=run.gradient_accumulation_steps)
    diagnostics.set_stage(first, mixture)
    # this process's attempt index; lr and weight decay as the optimizers hold them (a resumed run's: its checkpoint's)
    attempt = sum(r["kind"] == "run" for r in metrics.records(run_name))
    matmul_params, hidden_size = num_matmul_params(model), model.config.hidden_size

    def model_flops_per_token() -> int:
        """PaLM's nominal model FLOPs per token at the model's stage (metrics.model_flops_per_token)."""
        return metrics.model_flops_per_token(matmul_params=matmul_params, hidden_size=hidden_size,
                                             windows=model.window_sizes, context=model.config.max_position_embeddings)

    metrics.append(run_name, {"kind": "run", "step": start_step, "plan": asdict(run), "seed": seed, "config": config,
                              "data_key": tokenizer.data_key, "model": asdict(model.config),
                              "num_parameters": sum(p.numel() for p in model.parameters()),
                              "flops_per_token": non_embedding_flops_per_token(model),
                              "model_flops_per_token": model_flops_per_token(),
                              "max_grad_norm": MAX_GRAD_NORM,
                              "learning_rate": {name: [g["initial_lr"] for g in opt.param_groups]
                                                for name, opt in zip(("adamw", "muon"), opts, strict=True)},
                              "weight_decay": {name: [g["weight_decay"] for g in opt.param_groups]
                                               for name, opt in zip(("adamw", "muon"), opts, strict=True)},
                              "stage": asdict(first) | {"mixture": mixture}}, attempt=attempt)

    # A step's values reach the host without blocking and are logged after the next step launches (no per-step drain);
    # pending: the last step's log() arguments
    pending, previous_end = None, torch.cuda.Event(enable_timing=True)
    previous_end.record()

    def log(step: int, learning_rate: float, model_flops: int, values: torch.Tensor, layout: tuple,
            start: torch.cuda.Event, end: torch.cuda.Event) -> None:
        """A step's train record once its values reach the host, or its failure record (raising)."""
        end.synchronize()
        values = values.tolist()
        loss_value, grad_norm_value = values[:2]
        if not (math.isfinite(loss_value) and math.isfinite(grad_norm_value)):
            metrics.append(run_name, {"kind": "failure", "step": step}, attempt=attempt)
            raise FloatingPointError(f"{run_name} step {step}: loss {loss_value}, grad_norm {grad_norm_value}")
        dt = start.elapsed_time(end) / 1000
        metrics.append(run_name, {"kind": "train", "step": step, "num_input_tokens_seen": (step + 1) * run.batch_size,
                                  "loss": loss_value, "learning_rate": learning_rate, "grad_norm": grad_norm_value,
                                  "step_seconds": dt, "tokens_per_second": run.batch_size / dt,
                                  "model_flops": model_flops, "mfu": metrics.mfu(model_flops, dt)}
                       | diagnostics.record(values, layout), attempt=attempt)

    while True:
        if pending is not None and (step in checkpoints or step == run.max_steps):
            # the last step's values checked finite before a checkpoint saves its weights or the run ends
            log(*pending)
            pending = None
        # state after `step` steps at the stage they trained in
        if step != start_step and step in checkpoints:
            save_checkpoint(checkpoint_path(run_name, step), {"model": model.state_dict(),
                                                              "optimizers": [opt.state_dict() for opt in opts],
                                                              "config": asdict(model.config),
                                                              "stage_rows": stage_rows})
            gc.collect()
        if step == run.max_steps:
            metrics.append(run_name, {"kind": "end", "step": step,
                                      "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                                      "peak_reserved_bytes": torch.cuda.max_memory_reserved()}, attempt=attempt)
            return
        if run.stage_at(step) is not stage:
            # repo-level stage: new context and compiled graph; its batches were read the step before
            stage, stage_rows = run.stage_at(step), 0
            model.set_stage(stage)
            compiled = torch.compile(model, dynamic=False, fullgraph=True)
            diagnostics.set_stage(stage, mixture)
            metrics.append(run_name, {"kind": "stage", "step": step, "stage": asdict(stage) | {"mixture": mixture},
                                      "model_flops_per_token": model_flops_per_token(),
                                      "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                                      "peak_reserved_bytes": torch.cuda.max_memory_reserved()}, attempt=attempt)
            torch.cuda.reset_peak_memory_stats()

        # the next step's micro-batches are read during this step's backward (the repo-level stage's if it starts it)
        reading = step + 1 < run.max_steps
        if reading and run.stage_at(step + 1) is not stage:
            mixture, train_iter = read(stage=run.stage_at(step + 1), start=0)
        learning_rate = schedulers[0].get_last_lr()[0]
        diagnostics.begin_step(step, stage_rows)
        losses, upcoming, model_flops = [], [], 0
        for (input_ids, labels, cu_seqlens, segments, lengths), share in zip(
                batches, step_shares([b[1] for b in batches]), strict=True):
            loss, token_losses, lse = compiled(input_ids, labels, cu_seqlens, segments)
            loss = loss * share
            losses.append(loss.detach())
            diagnostics.micro_batch(labels, token_losses, lse)
            loss.backward()
            model_flops += metrics.executed_model_flops(lengths, matmul_params=matmul_params, hidden_size=hidden_size,
                                                        windows=model.window_sizes)
            if reading:
                upcoming.append(next(train_iter))
        grad_norm = update(model, opts, accumulated)
        model.refresh_gemm_weights()
        values, layout = diagnostics.end_step(step, torch.stack(losses).sum(), grad_norm)
        for scheduler in schedulers:
            scheduler.step()
        stage_rows += len(batches) * stage.micro_batch_size
        batches = upcoming
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        if pending is not None:
            log(*pending)
        pending, previous_end = (step, learning_rate, model_flops, values, layout, previous_end, end), end
        if step == start_step:
            # collector frozen and off after the first step, collecting only after saves (karpathy/nanochat 07c4dd4)
            gc.collect()
            gc.freeze()
            gc.disable()
        step += 1
