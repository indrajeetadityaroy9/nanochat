"""
Per-step training diagnostics, computed on the device and read one step late with the loss: per-corpus loss,
logsumexp max and mean square (PaLM's z-loss quantity, arXiv 2204.02311 §5), the largest QK-norm gain and, every
UPDATE_STATS_EVERY steps, each optimizer group's update RMS (without weight decay) and parameter norm.
"""

import numpy as np
import torch
from torch import Tensor

from corpus.loader import IGNORE_INDEX
from corpus.mixture import Stage, mixture_rows
from model.modeling import CausalLM

# every 10th step's cost is within throughput noise (GH200)
UPDATE_STATS_EVERY = 10


class Diagnostics:
    def __init__(self, model: CausalLM, opts: tuple[torch.optim.Optimizer, ...], *,
                 gradient_accumulation_steps: int) -> None:
        self.adamw, self.muon = opts
        self.gains = [p for layer in model.layers for p in (layer.self_attn.q_norm_weight,
                                                             layer.self_attn.k_norm_weight)]
        self.groups = [(f"{name}[{i}]", group["params"]) for name, opt in zip(("adamw", "muon"), opts, strict=True)
                       for i, group in enumerate(opt.param_groups)]
        self.numel = [sum(p.numel() for p in params) for _, params in self.groups]
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def set_stage(self, stage: Stage, mixture: dict[str, float]) -> None:
        """The stage's corpora (mixture order, as the loader's) and the pinned host buffers of a step's values."""
        self.names, self.weights = list(mixture), np.array(list(mixture.values()))
        self.micro_batch_size = stage.micro_batch_size
        # end_step's layout: 5 scalars, each group's update and parameter squared norms, the rows' loss sums and counts
        size = 5 + 2 * (len(self.groups) + self.gradient_accumulation_steps * self.micro_batch_size)
        self.host = [torch.empty(size, dtype=torch.float64, pin_memory=True) for _ in range(2)]

    def begin_step(self, step: int, stage_rows: int) -> None:
        """A step's accumulators; its rows start at the stage's row stage_rows."""
        self.rows_start, self.row_stats, self.lse_max, self.lse_square = stage_rows, [], [], []
        self.measured = step % UPDATE_STATS_EVERY == 0

    @torch.no_grad()
    def micro_batch(self, labels: Tensor, losses: Tensor, lse: Tensor) -> None:
        """One micro-batch's per-row loss sums, valid counts and logsumexp statistics. No autograd: a graph copied into
        the pinned buffers would chain every step's onto the last, never freed."""
        valid = labels != IGNORE_INDEX
        self.row_stats.append(torch.stack([losses.view(labels.shape).sum(1), valid.sum(1).float()]))
        valid = valid.flatten()
        self.lse_max.append(torch.where(valid, lse, -torch.inf).amax())
        self.lse_square.append((lse.square() * valid).sum())

    @torch.no_grad()
    def end_step(self, step: int, loss: Tensor, grad_norm: Tensor) -> tuple[Tensor, tuple]:
        """After the optimizers stepped: the step's values copied to a pinned host buffer without blocking, and their
        layout."""
        scalars = [loss, grad_norm, torch.stack(self.lse_max).amax(), torch.stack(self.lse_square).sum(),
                   torch.stack(torch._foreach_norm(self.gains, float("inf"))).amax()]
        if self.measured:
            for group in self.adamw.param_groups:
                (beta1, beta2), total = group["betas"], 0
                for p in group["params"]:
                    state = self.adamw.state[p]
                    step_count = state["step"]
                    total = total + (state["exp_avg"] / (1 - beta1**step_count) / (
                        state["exp_avg_sq"].sqrt() / (1 - beta2**step_count).sqrt() + group["eps"])).square().sum()
                scalars.append(total * group["lr"] ** 2)
            scalars.append(self.muon.update_square)
            scalars += [torch.stack(torch._foreach_norm(params)).square().sum() for _, params in self.groups]
        values = torch.cat([torch.stack(scalars).double(), torch.stack(self.row_stats).double().flatten()])
        host = self.host[step % 2][:values.numel()]
        host.copy_(values, non_blocking=True)
        return host, (self.names, self.weights, self.micro_batch_size, self.rows_start, self.measured)

    def record(self, values: list[float], layout: tuple) -> dict:
        """The train record's diagnostics from a step's host values (after its end event)."""
        names, weights, micro_batch_size, rows_start, measured = layout
        lse_max, lse_square, gain = values[2:5]
        n = len(self.groups) if measured else 0
        update, norm = values[5:5 + n], values[5 + n:5 + 2 * n]
        rows = np.asarray(values[5 + 2 * n:]).reshape(self.gradient_accumulation_steps, 2, micro_batch_size)
        corpora = mixture_rows(weights, start=rows_start, count=self.gradient_accumulation_steps * micro_batch_size)
        sums = np.bincount(corpora, rows[:, 0].flatten(), minlength=len(names))
        counts = np.bincount(corpora, rows[:, 1].flatten(), minlength=len(names))
        # a corpus whose weight is below one row per step may draw no rows in it
        out = {"max_logsumexp": lse_max, "mean_square_logsumexp": lse_square / counts.sum(), "qk_norm_gain_max": gain,
               "corpus_loss": {name: float(s / c) for name, s, c in zip(names, sums, counts, strict=True) if c}}
        if measured:
            out["update_rms"] = {g: (u / k) ** 0.5 for (g, _), u, k in zip(self.groups, update, self.numel,
                                                                            strict=True)}
            out["param_norm"] = {g: v ** 0.5 for (g, _), v in zip(self.groups, norm, strict=True)}
        return out
