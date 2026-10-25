"""
Per-step training diagnostics beyond the loss and gradient norm (train/loop.py), computed on the device and read with
the step's values one step late, as the loss is (no host synchronization):
- per-corpus training loss: the step's rows by the corpus each was drawn from (corpus/mixture.py mixture_rows, the
  loader's assignment);
- the output logits' logsumexp over the valid positions: its maximum and mean square (PaLM's z-loss quantity,
  arXiv 2204.02311 §5), from the cross-entropy kernel's per-row outputs (model/ops/cross_entropy.py lm_loss);
- the QK-norm gains' largest magnitude (logit growth inside attention; Kimi K2's QK-clip trigger, arXiv 2507.20534
  §2.1, bounds the same quantity);
- every UPDATE_STATS_EVERY steps, each optimizer group's update RMS (the step's update without its decoupled weight
  decay: AdamW's lr m_hat / (sqrt(v_hat) + eps) from its state, torch/optim/adamw.py; Muon's adjusted-lr orthogonalized
  update, optim.py Muon.update_square) and parameter norm;
- host-side flags: a loss or gradient norm above SPIKE times the median of the previous SPIKE_WINDOW steps, and a step
  whose gradient norm exceeded MAX_GRAD_NORM (clipped).
"""

import statistics
from collections import deque

import numpy as np
import torch
from torch import Tensor

from corpus.loader import IGNORE_INDEX
from corpus.mixture import Stage, mixture_rows
from model.modeling import CausalLM
from train.optim import MAX_GRAD_NORM

UPDATE_STATS_EVERY = 10       # steps between update-RMS and parameter-norm measurements
SPIKE_WINDOW = 100            # steps of the running median
SPIKE = {"loss": 1.1, "grad_norm": 3.0}


class Diagnostics:
    def __init__(self, model: CausalLM, opts: tuple[torch.optim.Optimizer, ...], *,
                 gradient_accumulation_steps: int) -> None:
        self.adamw, self.muon = opts
        self.gains = [p for layer in model.layers for p in (layer.self_attn.q_norm_weight,
                                                             layer.self_attn.k_norm_weight)]
        # optimizer groups in step order: (name, parameters)
        self.groups = [(f"{name}[{i}]", group["params"]) for name, opt in zip(("adamw", "muon"), opts, strict=True)
                       for i, group in enumerate(opt.param_groups)]
        self.numel = [sum(p.numel() for p in params) for _, params in self.groups]
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.history = {name: deque(maxlen=SPIKE_WINDOW) for name in SPIKE}

    def set_stage(self, stage: Stage, mixture: dict[str, float]) -> None:
        """The stage's corpora (mixture order, as the loader's) and the pinned host buffers of a step's values."""
        self.names, self.weights = list(mixture), np.array(list(mixture.values()))
        self.micro_batch_size = stage.micro_batch_size
        # end_step's layout: 5 scalars, the update statistics, the rows' loss sums and counts
        size = 5 + 2 * (len(self.groups) + self.gradient_accumulation_steps * self.micro_batch_size)
        self.host = [torch.empty(size, dtype=torch.float64, pin_memory=True) for _ in range(2)]

    def begin_step(self, step: int, stage_rows: int) -> None:
        """A step's accumulators."""
        self.rows_start, self.row_stats, self.lse_max, self.lse_square = stage_rows, [], [], []
        self.measured = step % UPDATE_STATS_EVERY == 0

    def micro_batch(self, labels: Tensor, losses: Tensor, lse: Tensor) -> None:
        """One micro-batch's per-row loss sums and valid counts, and its logsumexp statistics (lm_loss outputs)."""
        valid = labels != IGNORE_INDEX
        self.row_stats.append(torch.stack([losses.view(labels.shape).sum(1), valid.sum(1).float()]))
        valid = valid.flatten()
        self.lse_max.append(torch.where(valid, lse, -torch.inf).amax())
        self.lse_square.append((lse.square() * valid).sum())

    def after_update(self) -> None:
        """The update statistics of a measured step, after the optimizers stepped (the groups' lr still the step's)."""
        if not self.measured:
            return
        update = []
        for group in self.adamw.param_groups:
            (beta1, beta2), total = group["betas"], 0
            for p in group["params"]:
                state = self.adamw.state[p]
                step = state["step"]
                total = total + (state["exp_avg"] / (1 - beta1**step) / (
                    state["exp_avg_sq"].sqrt() / (1 - beta2**step).sqrt() + group["eps"])).square().sum()
            update.append(total * group["lr"] ** 2)
        update.append(self.muon.update_square)
        norm = [torch.stack(torch._foreach_norm(params)).square().sum() for _, params in self.groups]
        self.update_stats = torch.stack(update + norm)

    def end_step(self, step: int, loss: Tensor, grad_norm: Tensor) -> tuple[Tensor, dict]:
        """The step's values copied to a pinned host buffer without blocking, and the layout record() reads them by."""
        parts = [loss.double().view(1), grad_norm.double().view(1), torch.stack(self.lse_max).amax().double().view(1),
                 torch.stack(self.lse_square).sum().double().view(1),
                 torch.stack(torch._foreach_norm(self.gains, float("inf"))).amax().double().view(1)]
        layout = {"groups": [name for name, _ in self.groups] if self.measured else [],
                  "rows_start": self.rows_start, "micro_batches": len(self.row_stats), "names": self.names,
                  "weights": self.weights, "micro_batch_size": self.micro_batch_size, "numel": self.numel}
        if self.measured:
            parts.append(self.update_stats.double())
        parts.append(torch.stack(self.row_stats).double().flatten())
        values = torch.cat(parts)
        host = self.host[step % 2][:values.numel()]
        host.copy_(values, non_blocking=True)
        return host, layout

    def record(self, values: list[float], layout: dict) -> dict:
        """The train record's diagnostics from a step's host values (after its end event)."""
        loss, grad_norm, lse_max, lse_square, gain = values[:5]
        at = 5
        out: dict = {"max_logsumexp": lse_max, "qk_norm_gain_max": gain}
        if layout["groups"]:
            n = len(layout["groups"])
            update, norm = values[at:at + n], values[at + n:at + 2 * n]
            at += 2 * n
            out["update_rms"] = {g: (u / k) ** 0.5 for g, u, k in zip(layout["groups"], update, layout["numel"],
                                                                     strict=True)}
            out["param_norm"] = {g: v ** 0.5 for g, v in zip(layout["groups"], norm, strict=True)}
        mbs, k = layout["micro_batch_size"], layout["micro_batches"]
        rows = np.asarray(values[at:at + 2 * k * mbs]).reshape(k, 2, mbs)
        corpora = np.concatenate([mixture_rows(layout["weights"], start=layout["rows_start"] + j * mbs, count=mbs)
                                  for j in range(k)])
        sums = np.bincount(corpora, rows[:, 0].flatten(), minlength=len(layout["names"]))
        counts = np.bincount(corpora, rows[:, 1].flatten(), minlength=len(layout["names"]))
        out["corpus_loss"] = {name: float(s / c) for name, s, c in zip(layout["names"], sums, counts, strict=True)
                              if c}
        out["mean_square_logsumexp"] = lse_square / max(float(counts.sum()), 1.0)
        for name, value in (("loss", loss), ("grad_norm", grad_norm)):
            history = self.history[name]
            out[f"{name}_spike"] = len(history) == SPIKE_WINDOW and value > SPIKE[name] * statistics.median(history)
            history.append(value)
        out["clipped"] = grad_norm > MAX_GRAD_NORM
        return out
