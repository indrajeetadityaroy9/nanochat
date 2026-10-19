"""
Pretrain a base model on the GPU. From the root directory of the project:

python -m scripts.base_train --depth=<depth> --target-param-data-ratio=<tokens per scaling parameter> --seed=<seed>
    --save-every=<steps>

Every other setting is derived: nanochat/config.py (the stages' contexts, RoPE bases and mixtures, micro-batch),
nanochat/train/schedule.py (batch, horizon, the repo-level stage's first step, peak lr, lr schedule) and
nanochat/train/optim.py (the optimizer groups, weight decay, gradient clip). The run trains the file-level
stage, then from the plan's repo_start on the repo-level stage (config.REPO_STAGE: the model continues at its context
and RoPE base), on the next token with the packs' fill-in-the-middle pieces read as FIM examples (DeepSeek-Coder, arXiv
2401.14196 §3.1; data/pack.py). The run's tag is config.run_tag. --save-every writes full checkpoints (only the newest
is kept); every start resumes from the tag's newest checkpoint, and continues bitwise as the uninterrupted run would
(the gradients are deterministic, model/gpt.py; the data order is the seed's). Metrics:
<base_dir>/metrics/<tag>.jsonl (nanochat/train/metrics.py).
"""

import argparse
import gc
import os
import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from typing import Literal

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from nanochat.config import FILE_STAGE, REPO_STAGE, run_tag
from nanochat.data.loader import PretrainingBatches
from nanochat.data.tokenizer import get_tokenizer
from nanochat.model.gpt import ASPECT_RATIO, GPT, gpt_config
from nanochat.train import metrics, stages
from nanochat.train.checkpoint import checkpoint_path, save_checkpoint
from nanochat.train.optim import GRAD_CLIP, WEIGHT_DECAY, optimizers, tau_weight_decay
from nanochat.train.schedule import lr_multiplier, plan


@dataclass(frozen=True, kw_only=True)
class Variant:
    """The ablation study's departures from the canonical mechanism (experiments/), canonical by default: the model's
    switches (model/gpt.py GPTConfig), the lr schedule shape and the repo-level stage's schedule
    (schedule.lr_multiplier), the weight decay (optim.py WEIGHT_DECAY, or the EMA-timescale rule tau_weight_decay), the
    gradient clip (optim.py GRAD_CLIP; inf: none), a factor on the plan's peak lr and on its batch (schedule.plan), and
    the RoPE base: "switch" the stages' own (config.Stage), "single" the repo-level stage's for both (Men et al., arXiv
    2405.14591: the base's lower bound at the final context)."""
    value_embeds: bool = True
    x0: bool = True
    resid: bool = True
    qk_gain: bool = True
    windows: bool = True
    schedule: Literal["wsd", "d2z"] = "wsd"
    weight_decay: Literal["fixed", "tau"] = "fixed"
    grad_clip: float = GRAD_CLIP
    lr_scale: float = 1.0
    batch_scale: float = 1.0
    repo_schedule: Literal["continue", "rewarm"] = "continue"
    rope: Literal["switch", "single"] = "switch"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--target-param-data-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--save-every", type=int, required=True)
    args = parser.parse_args()
    train(depth=args.depth, ratio=args.target_param_data_ratio, seed=args.seed, data_seed=args.seed,
          save_every=args.save_every, tag=run_tag(args.depth, args.target_param_data_ratio, args.seed),
          variant=Variant())


def optimizer_step(model, batches, params, grad_clip, optimizers, reads):
    """One optimizer step on its micro-batches (inputs, targets, cu_seqlens), the one step of every training run: each
    micro-batch's mean next-token loss weighted by its share n_i / n of the step's valid targets (so the step loss is
    the mean over all of them) and backpropagated; after each backward the next step's micro-batch read from `reads`
    (None on the last step), overlapping the loader with the GPU; the gradient clipped to L2 norm grad_clip; every
    optimizer stepped and the gradients cleared. Returns (the step loss summed in fp64 on the device, the gradient norm
    before clipping, the next step's micro-batches) without a host synchronization."""
    valid = torch.stack([(y != PretrainingBatches.PAD).sum() for _, y, _ in batches]).double()
    shares = (valid / valid.sum()).float()
    losses, upcoming = [], []
    for micro_step, (x, y, cu_seqlens) in enumerate(batches):
        loss = model(x, y, cu_seqlens) * shares[micro_step]
        losses.append(loss.detach())
        loss.backward()
        if reads is not None:
            upcoming.append(next(reads))
    grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
    for opt in optimizers:
        opt.step()
    model.zero_grad()
    return torch.stack(losses).double().sum(), grad_norm, upcoming


def train(*, depth, ratio, seed, data_seed, save_every, tag, variant):
    """Train (or resume) the run `tag` and return its model: initialization by `seed`, data order by `data_seed`, the
    plan of the canonical model at this depth and ratio (so every variant of an ablation but a batch-scaled one reads
    the same tokens in the same steps)."""
    device = torch.device("cuda")
    tokenizer = get_tokenizer()
    run = plan(depth, ratio, tokenizer.vocab_size, variant.batch_scale)
    lr = run.lr * variant.lr_scale
    weight_decay = (WEIGHT_DECAY if variant.weight_decay == "fixed"
                    else tau_weight_decay(lr, run.num_iterations, run.tokens_per_param))
    switches = {k: getattr(variant, k) for k in ("value_embeds", "x0", "resid", "qk_gain", "windows")}

    with torch.device("meta"):
        orig_model = GPT(replace(gpt_config(depth, tokenizer.vocab_size, FILE_STAGE, aspect_ratio=ASPECT_RATIO),
                                 **switches))
    orig_model.to_empty(device=device)
    orig_model.init_weights(seed)
    adamw, muon = optimizers(orig_model, lr, weight_decay)
    path = checkpoint_path(tag)
    start_step = stage_rows = 0
    if os.path.exists(path):  # resume: weights copied into the parameters the optimizers hold, then their state
        checkpoint = torch.load(path, map_location=device)
        orig_model.load_state_dict(checkpoint["model"])
        for opt, state in zip((adamw, muon), checkpoint["optimizers"], strict=True):
            opt.load_state_dict(state)
        start_step, stage_rows = checkpoint["step"], checkpoint["stage_rows"]
        del checkpoint
    # every group's lr: its initial lr x lr_multiplier(step); a resumed scheduler starts at start_step from the
    # initial lr the loaded optimizer state holds
    schedulers = [LambdaLR(opt, partial(lr_multiplier, n=run.num_iterations, shape=variant.schedule,
                                        repo_start=run.repo_start, repo=variant.repo_schedule),
                           last_epoch=start_step - 1) for opt in (adamw, muon)]
    step = start_step
    stage = REPO_STAGE if start_step >= run.repo_start else FILE_STAGE

    def rotary(stage):
        """The stage as the model runs it: its own RoPE base, or the repo-level stage's (variant.rope "single")."""
        return replace(stage, rope_base=REPO_STAGE.rope_base) if variant.rope == "single" else stage

    orig_model.set_stage(rotary(stage))
    model = torch.compile(orig_model, dynamic=False, fullgraph=True)

    # A stage's mixture of packed rows (train/stages.py: the rows each stage may read, no document read twice), each
    # corpus's in one permutation by the seed (RandomState: its streams are frozen across numpy versions, so a resumed run
    # sees the same order), the FIM pieces read as FIM examples. The data state is the stage's rows consumed.
    def stage_batches(stage, start):
        mixture = stages.stage_mixture(stage, tokenizer)
        readable = stages.readable_rows(stage, run, tokenizer)
        orders = {corpus: np.random.RandomState(data_seed).permutation(readable[corpus]) for corpus in mixture}
        batches = PretrainingBatches(mixture, "train", tokenizer, orders, seq_len=stage.seq_len,
                                     batch_rows=stage.batch_rows, device=device, reading="fim",
                                     start=start)
        return mixture, iter(batches)

    mixture, train_iter = stage_batches(stage, stage_rows)
    batches = [next(train_iter) for _ in range(run.grad_accum_steps)] if start_step < run.num_iterations else []
    metrics.append(tag, {"kind": "run", "step": start_step, "plan": asdict(run), "seed": seed, "data_seed": data_seed,
                         "variant": asdict(variant), "weight_decay": weight_decay,
                         "stage": asdict(rotary(stage)) | {"mixture": mixture}})
    params = list(orig_model.parameters())

    while True:
        last_step = step == run.num_iterations
        if step == run.repo_start and stage is FILE_STAGE:
            # the repo-level stage: the model continues at its context and RoPE base (a new compiled graph for its
            # shapes); its batches were read during the step before
            stage, stage_rows = REPO_STAGE, 0
            orig_model.set_stage(rotary(stage))
            model = torch.compile(orig_model, dynamic=False, fullgraph=True)
            metrics.append(tag, {"kind": "stage", "step": step, "stage": asdict(rotary(stage)) | {"mixture": mixture}})
        # the state after `step` steps, with the stage of the step after it: a full checkpoint
        if step != start_step and (last_step or step % save_every == 0):
            save_checkpoint(path, {"step": step, "model": orig_model.state_dict(),
                                   "optimizers": [adamw.state_dict(), muon.state_dict()],
                                   "model_config": asdict(orig_model.config), "stage_rows": stage_rows})
            gc.collect()
        if last_step:
            return orig_model

        torch.cuda.synchronize()
        t0 = time.time()
        # The next step's micro-batches are read while the GPU runs this step's (from the repo-level stage's rows when
        # the next step is its first); the data state stays the rows of the steps taken.
        if step + 1 == run.repo_start and stage is FILE_STAGE:
            mixture, train_iter = stage_batches(REPO_STAGE, 0)
        lr = schedulers[0].get_last_lr()[0]
        loss, grad_norm, upcoming = optimizer_step(model, batches, params, variant.grad_clip, (adamw, muon),
                                                   train_iter if step + 1 < run.num_iterations else None)
        for scheduler in schedulers:
            scheduler.step()
        stage_rows += len(batches) * stage.batch_rows
        batches = upcoming
        torch.cuda.synchronize()
        dt = time.time() - t0

        # the step loss and the grad norm read in one host synchronization
        loss_value, grad_norm_value = torch.stack([loss, grad_norm.double()]).tolist()
        metrics.append(tag, {"kind": "train", "step": step, "tokens": (step + 1) * run.total_batch_size,
                             "loss": loss_value, "lr": lr, "grad_norm": grad_norm_value,
                             "tok_per_s": run.total_batch_size / dt})
        if step == start_step:
            # A training step leaves no cyclic garbage: after the setup's is collected, the collector stays off (its
            # scans cost ~500 ms each) except after each checkpoint save
            gc.collect()
            gc.freeze()
            gc.disable()
        step += 1


if __name__ == "__main__":
    main()
