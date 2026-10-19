"""
Pretrain a base model on the GPU. From the root directory of the project:

python -m scripts.base_train --depth=<depth> --target-param-data-ratio=<tokens per scaling parameter> --seed=<seed>
    --save-every=<steps>

Every other setting is derived: nanochat/config.py (the stages' contexts, RoPE bases and mixtures, micro-batch),
nanochat/train/schedule.py (batch, horizon, the repo-level stage's first step, peak lr, lr schedule) and
nanochat/train/optim.py (the optimizer groups, gradient clipping). The run trains the file-level stage, then from the
plan's repo_start on the repo-level stage (config.REPO_STAGE: the model continues at its context and RoPE base), on the
next token with the packs' fill-in-the-middle pieces read as FIM examples (DeepSeek-Coder, arXiv 2401.14196 §3.1;
data/pack.py). The run's tag is config.run_tag. --save-every writes full checkpoints (only the newest is kept); every
start resumes from the tag's newest checkpoint, and continues bitwise as the uninterrupted run would (the gradients are
deterministic, model/gpt.py; the data order is the seed's). Metrics: <base_dir>/metrics/<tag>.jsonl
(nanochat/train/metrics.py).
"""

import argparse
import gc
import os
import time
from dataclasses import asdict
from functools import partial

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from nanochat.config import FILE_STAGE, REPO_STAGE, run_tag
from nanochat.data.loader import PretrainingBatches
from nanochat.data.tokenizer import get_tokenizer
from nanochat.model.gpt import ASPECT_RATIO, GPT, gpt_config
from nanochat.train import metrics, stages
from nanochat.train.checkpoint import checkpoint_path, save_checkpoint
from nanochat.train.optim import GRAD_CLIP, optimizers
from nanochat.train.schedule import lr_multiplier, plan


def main():
    parser = argparse.ArgumentParser(description="Pretrain base model")
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--target-param-data-ratio", type=float, required=True, help="training tokens per scaling parameter")
    parser.add_argument("--seed", type=int, required=True, help="seed of the model initialization and of the data order")
    parser.add_argument("--save-every", type=int, required=True, help="full checkpoint every N steps besides the final one")
    args = parser.parse_args()
    model_tag = run_tag(args.depth, args.target_param_data_ratio, args.seed)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    tokenizer = get_tokenizer()
    run = plan(args.depth, args.target_param_data_ratio, tokenizer.vocab_size)

    with torch.device("meta"):
        orig_model = GPT(gpt_config(args.depth, tokenizer.vocab_size, FILE_STAGE, aspect_ratio=ASPECT_RATIO))
    orig_model.to_empty(device=device)
    orig_model.init_weights()
    adamw, muon = optimizers(orig_model, run.lr)
    path = checkpoint_path(model_tag)
    start_step = stage_rows = 0
    if os.path.exists(path):  # resume: weights copied into the parameters the optimizers hold, then their state
        checkpoint = torch.load(path, map_location=device)
        orig_model.load_state_dict(checkpoint["model"])
        for opt, state in zip((adamw, muon), checkpoint["optimizers"], strict=True):
            opt.load_state_dict(state)
        start_step, stage_rows = checkpoint["step"], checkpoint["stage_rows"]
        del checkpoint
    # every group's lr: its initial lr (run.lr) x lr_multiplier(step); a resumed scheduler starts at start_step from the
    # initial lr the loaded optimizer state holds
    schedulers = [LambdaLR(opt, partial(lr_multiplier, n=run.num_iterations), last_epoch=start_step - 1)
                  for opt in (adamw, muon)]
    step = start_step
    stage = REPO_STAGE if start_step >= run.repo_start else FILE_STAGE
    orig_model.set_stage(stage)
    model = torch.compile(orig_model, dynamic=False, fullgraph=True)

    # A stage's mixture of packed rows (train/stages.py: the rows each stage may read, no document read twice), each
    # corpus's in one permutation by the seed (RandomState: its streams are frozen across numpy versions, so a resumed run
    # sees the same order), the FIM pieces read as FIM examples. The data state is the stage's rows consumed.
    def stage_batches(stage, start):
        mixture = stages.stage_mixture(stage, tokenizer)
        readable = stages.readable_rows(stage, run, tokenizer)
        orders = {corpus: np.random.RandomState(args.seed).permutation(readable[corpus]) for corpus in mixture}
        batches = PretrainingBatches(mixture, "train", tokenizer, orders, seq_len=stage.seq_len,
                                     batch_rows=stage.batch_rows, device=device, reading="fim",
                                     start=start)
        return mixture, iter(batches)

    mixture, train_iter = stage_batches(stage, stage_rows)
    batches = [next(train_iter) for _ in range(run.grad_accum_steps)] if start_step < run.num_iterations else []
    metrics.append(model_tag, {"kind": "run", "step": start_step, "args": vars(args), "plan": asdict(run),
                               "stage": asdict(stage) | {"mixture": mixture}})
    params = list(orig_model.parameters())

    while True:
        last_step = step == run.num_iterations
        if step == run.repo_start and stage is FILE_STAGE:
            # the repo-level stage: the model continues at its context and RoPE base (a new compiled graph for its
            # shapes); its batches were read during the step before
            stage, stage_rows = REPO_STAGE, 0
            orig_model.set_stage(stage)
            model = torch.compile(orig_model, dynamic=False, fullgraph=True)
            metrics.append(model_tag, {"kind": "stage", "step": step, "stage": asdict(stage) | {"mixture": mixture}})
        # the state after `step` steps, with the stage of the step after it: a full checkpoint
        if step != start_step and (last_step or step % args.save_every == 0):
            save_checkpoint(path, {"step": step, "model": orig_model.state_dict(),
                                   "optimizers": [adamw.state_dict(), muon.state_dict()],
                                   "model_config": asdict(orig_model.config), "stage_rows": stage_rows})
            gc.collect()
        if last_step:
            break

        torch.cuda.synchronize()
        t0 = time.time()
        # The next step's micro-batches are read while the GPU runs this step's (from the repo-level stage's rows when
        # the next step is its first); the data state stays the rows of the steps taken.
        if step + 1 == run.repo_start and stage is FILE_STAGE:
            mixture, train_iter = stage_batches(REPO_STAGE, 0)
        upcoming = []
        # The step loss: the mean next-token loss over the step's valid targets. The model's loss is its micro-batch's
        # mean, so micro-batch i weighs its share n_i / n of the step's valid targets.
        valid = torch.stack([(y != PretrainingBatches.PAD).sum() for _, y, _ in batches]).double()
        shares = (valid / valid.sum()).float()
        losses = []
        for micro_step, (x, y, cu_seqlens) in enumerate(batches):
            loss = model(x, y, cu_seqlens) * shares[micro_step]
            losses.append(loss.detach())
            loss.backward()
            if step + 1 < run.num_iterations:
                upcoming.append(next(train_iter))
        grad_norm = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)  # the norm before clipping
        lr = schedulers[0].get_last_lr()[0]
        for opt, scheduler in zip((adamw, muon), schedulers, strict=True):
            opt.step()
            scheduler.step()
        model.zero_grad()
        stage_rows += len(batches) * stage.batch_rows
        batches = upcoming
        torch.cuda.synchronize()
        dt = time.time() - t0

        # the step loss summed in fp64 on the device, read with the grad norm in one host synchronization
        loss_value, grad_norm_value = torch.stack([torch.stack(losses).double().sum(), grad_norm.double()]).tolist()
        metrics.append(model_tag, {"kind": "train", "step": step, "tokens": (step + 1) * run.total_batch_size,
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
