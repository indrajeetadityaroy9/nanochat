"""
Pretrain a base model on the GPU. From the root directory of the project:

python -m scripts.base_train --depth=<depth> --target-param-data-ratio=<tokens per scaling parameter> --seed=<seed>
    --objective=<spec|ntp> --save-every=<steps>

Every other setting is derived: nanochat/config.py (context, mixture, micro-batch), nanochat/train/schedule.py (batch,
horizon, schedules) and nanochat/train/optim.py (the optimizer groups). The run's tag is d{depth}-r{ratio}-{objective}-
s{seed}. --save-every writes full checkpoints (only the newest is kept); every start resumes from the tag's newest
checkpoint with the plan's hyperparameters. The final checkpoint holds the raw weights and, in its optimizer file, the
tail averages (train/checkpoint.py load_model). Metrics: <base_dir>/metrics/<tag>.jsonl (nanochat/train/metrics.py).
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import gc
import itertools
import json
import time
from dataclasses import asdict

import numpy as np
import torch
from torchmetrics.aggregation import SumMetric

from nanochat.config import MICRO_BATCH_ROWS, MIXTURE, SEQ_LEN, run_tag
from nanochat.data.loader import PretrainingBatches, mixture_weights, packed_rows
from nanochat.data.tokenizer import get_tokenizer, prefix_targets
from nanochat.model.config import gpt_config, padded_vocab
from nanochat.model.gpt import GPT
from nanochat.train import metrics
from nanochat.train.checkpoint import (
    checkpoint_dir,
    checkpoint_steps,
    load_checkpoint,
    save_checkpoint,
)
from nanochat.train.optim import AnvilAdam, param_groups
from nanochat.train.schedule import (
    MTP_WEIGHTS,
    OBJECTIVES,
    anvil_rails,
    flops_per_token,
    loss_columns,
    loss_weights,
    lr_multiplier,
    plan,
)
from nanochat.train.tail_average import TailAverage


def main():
    parser = argparse.ArgumentParser(description="Pretrain base model")
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--target-param-data-ratio", type=float, required=True, help="training tokens per scaling parameter")
    parser.add_argument("--seed", type=int, required=True, help="seed of the model initialization and of the data order")
    parser.add_argument("--objective", type=str, required=True, choices=OBJECTIVES)
    parser.add_argument("--save-every", type=int, required=True, help="full checkpoint every N steps besides the final one")
    args = parser.parse_args()
    spec = args.objective == "spec"
    model_tag = run_tag(args.depth, args.target_param_data_ratio, args.objective, args.seed)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    tokenizer = get_tokenizer()

    run = plan(args.depth, args.target_param_data_ratio, args.objective, tokenizer.vocab_size)
    num_iterations = run.num_iterations
    print(f"Run {model_tag}: {json.dumps(asdict(run))}")
    model_config = gpt_config(args.depth, tokenizer.vocab_size, SEQ_LEN, run.engram_rows)
    with torch.device("meta"):
        orig_model = GPT(model_config)
    orig_model.to_empty(device=device)
    orig_model.init_weights()
    directory = checkpoint_dir(model_tag)
    steps = checkpoint_steps(directory)
    resume_step = steps[-1] if steps else None
    if resume_step is not None:
        model_data, optimizer_data, meta_data = load_checkpoint(directory, resume_step, device, load_optimizer=True)
        orig_model.load_state_dict(model_data, strict=True, assign=True)
        del model_data

    columns = loss_columns(args.objective)
    if spec:
        orig_model.set_loss_targets(len(MTP_WEIGHTS), tokenizer.bos_token_id, prefix_targets(tokenizer, rows=padded_vocab(tokenizer.vocab_size)).to(device))
    else:
        orig_model.set_loss_targets(1, tokenizer.bos_token_id, None)
    model = torch.compile(orig_model, dynamic=False, fullgraph=True)

    # torch's fused AdamW for the AdamW groups (nanochat's and DeepSeek's update: decay p (1 - lr wd), bias-corrected
    # moments, denominator sqrt(v) / sqrt(1 - beta2^t) + eps), AnvilAdam for the others
    groups = param_groups(orig_model, run)
    optimizers = {"anvil": AnvilAdam([g for g in groups if g["kind"] != "adamw"]),
                  "adamw": torch.optim.AdamW([g for g in groups if g["kind"] == "adamw"], fused=True)}
    for optimizer in optimizers.values():
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    hyperparameters = [{k: v for k, v in g.items() if k != "params"} for o in optimizers.values() for g in o.param_groups]
    tail = TailAverage(orig_model, num_iterations)
    smooth_loss = 0.0
    if resume_step is not None:
        tail.load_state_dict(optimizer_data["tail_average"])
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(optimizer_data[name])
        smooth_loss = meta_data["smooth_loss"]
        del optimizer_data
    # the plan's hyperparameters, also over a resumed optimizer's (load_state_dict restores the saved groups' values)
    groups = [g for o in optimizers.values() for g in o.param_groups]
    for group, values in zip(groups, hyperparameters, strict=False):
        group.update(values)

    # The mixture's packed rows, each corpus's in one permutation by the seed (RandomState: its streams are frozen across
    # numpy versions, so a resumed run sees the same order): spec assembles the FIM pieces as FIM examples, ntp reads
    # them left to right. The data state is the rows consumed.
    mixture = mixture_weights(MIXTURE, tokenizer)
    orders = {corpus: np.random.RandomState(args.seed).permutation(packed_rows(corpus, "train", tokenizer)) for corpus in mixture}
    run_rows = meta_data["run_rows"] if resume_step is not None else 0
    start_step = step = resume_step or 0
    train_iter = iter(PretrainingBatches(mixture, "train", tokenizer, orders, batch_rows=MICRO_BATCH_ROWS, device=device,
                                         reading="fim" if spec else "left_to_right", start=run_rows))

    metrics.append(model_tag, {"kind": "run", "step": start_step, "args": vars(args), "plan": asdict(run),
                               "optimizer": hyperparameters, "mixture": mixture})
    # the step's sums (fp64, torchmetrics; nan_strategy "disable": no host synchronization, a NaN shows in the record):
    # its loss, and per column its mean loss over its own valid targets
    loss_sum = SumMetric(nan_strategy="disable").set_dtype(torch.float64).to(device)
    column_sums = [SumMetric(nan_strategy="disable").set_dtype(torch.float64).to(device) for _ in columns]
    params = [p for group in groups for p in group["params"]]
    # per group: its weights' RMS, its step's update RMS (weights before the step minus after, in `before`)
    before = [torch.empty_like(p) for p in params]
    bounds = list(itertools.accumulate(len(g["params"]) for g in groups))
    group_numel = torch.tensor([sum(p.numel() for p in g["params"]) for g in groups], dtype=torch.float64, device=device)

    def group_rms(norms):
        return (torch.stack([torch.stack(norms[a:b]).double().square().sum() for a, b in zip([0, *bounds], bounds, strict=False)]) / group_numel).sqrt()

    ema_beta = 0.9  # the step loss's debiased EMA: nanochat 92d63d4 scripts/base_train.py:548-550
    engram_layers = model_config.engram.layers
    flops = flops_per_token(orig_model)

    while True:
        last_step = step == num_iterations
        # the state after `step` steps: a full checkpoint (the raw weights; the tail averages in the optimizer file)
        if step != start_step and (last_step or step % args.save_every == 0):
            optimizer_data = {name: o.state_dict() for name, o in optimizers.items()} | {"tail_average": tail.state_dict()}
            meta = {"step": step, "model_config": asdict(model_config), "user_config": vars(args), "objective": args.objective,
                    "run_rows": run_rows, "smooth_loss": smooth_loss}
            save_checkpoint(directory, step, orig_model.state_dict(), optimizer_data, meta)
            del optimizer_data
            gc.collect()
        if last_step:
            break

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        weights = loss_weights(step, args.objective, run.stage_ends)
        batches = [next(train_iter) for _ in range(run.grad_accum_steps)]
        # The step loss (ops/softcap_ce.py): every valid target of column k weighs w_k, the sum divided by the step's
        # valid next-token targets n_0; the loss op divides micro-batch i's sums by its own n_i0, so its weights are
        # w_k n_i0 / n_0. Column monitors: each column's mean over its own valid targets (weights n_ik / n_k).
        valid = torch.stack([(orig_model.loss_targets(y) != -1).sum(0) for _, y, _ in batches]).float()
        shares = valid / valid.sum(0)
        micro_weights = torch.tensor(weights, device=device) * shares[:, :1]
        gates = []
        for micro_step, (x, y, cu_seqlens) in enumerate(batches):
            loss, column_losses, engram_gate = model(x, y, loss_weights=micro_weights[micro_step], cu_seqlens=cu_seqlens)
            loss_sum.update(loss.detach())
            for column_sum, column_loss, share in zip(column_sums, column_losses, shares[micro_step], strict=False):
                column_sum.update(column_loss * share)
            gates.append(engram_gate)  # micro-batches of equal token counts: the step's mean is their mean
            loss.backward()
        grad_norm = torch.nn.utils.get_total_norm([p.grad for p in params])
        lrm = lr_multiplier(step, num_iterations, 0)  # the schedule's shape; groups with a warmup ramp on top
        rails = anvil_rails(step, num_iterations)
        for group in groups:
            group["lr"] = group["initial_lr"] * lr_multiplier(step, num_iterations, group["warmup"])
            if group["kind"] == "anvil":
                group["momentum"], group["fast_beta"], group["fast_weight"] = rails
        with torch.no_grad():
            torch._foreach_copy_(before, params)
            for optimizer in optimizers.values():
                optimizer.step()
            torch._foreach_sub_(before, params)
            weight_norms, update_norms = torch._foreach_norm(params), torch._foreach_norm(before)
        tail.update(step)
        model.zero_grad(set_to_none=True)
        run_rows += len(batches) * MICRO_BATCH_ROWS
        torch.cuda.synchronize()
        dt = time.time() - t0

        values = torch.cat([torch.stack([loss_sum.compute(), grad_norm.double(), *(c.compute() for c in column_sums)]),
                            torch.stack(gates).mean(0).double(), group_rms(weight_norms), group_rms(update_norms)]).tolist()
        for metric in (loss_sum, *column_sums):
            metric.reset()
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * values[0]
        gate_values = values[2 + len(columns):2 + len(columns) + len(engram_layers)]
        rms_values = values[2 + len(columns) + len(engram_layers):]
        record = {"kind": "train", "step": step, "tokens": (step + 1) * run.total_batch_size, "lrm": lrm, "loss": values[0],
                  "smooth_loss": smooth_loss / (1 - ema_beta ** (step + 1)),
                  "loss_columns": {c: values[2 + j] for j, c in enumerate(columns)}, "grad_norm": values[1],
                  "engram_gate": {str(layer): v for layer, v in zip(engram_layers, gate_values, strict=False)},
                  "weight_rms": {g["name"]: v for g, v in zip(groups, rms_values[:len(groups)], strict=False)},
                  "update_rms": {g["name"]: v for g, v in zip(groups, rms_values[len(groups):], strict=False)},
                  "step_ms": dt * 1000, "tok_per_s": run.total_batch_size / dt, "tflops": flops * run.total_batch_size / dt / 1e12,
                  "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
        metrics.append(model_tag, record)
        print(json.dumps(record))
        if step == start_step:
            # A training step leaves no cyclic garbage: after the setup's is collected, the collector stays off (its
            # scans cost ~500 ms each) except after each checkpoint save. The host cache holds pinned copies Inductor's
            # autotuning made at compile time.
            gc.collect()
            gc.freeze()
            gc.disable()
            torch.accelerator.empty_host_cache()
        step += 1


if __name__ == "__main__":
    main()
