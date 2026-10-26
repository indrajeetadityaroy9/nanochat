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
  decay: AdamW's lr m_hat / (sqrt(v_hat) + eps) from its state, torch/optim/adamw.py; Muon's adjusted-lr update,
  optim.py Muon.update_square) and parameter norm.
"""

import numpy as np
import torch
from torch import Tensor

from corpus.loader import IGNORE_INDEX
from corpus.mixture import Stage, mixture_rows
from model.modeling import CausalLM

# Steps between update-RMS and parameter-norm measurements, by their cost: every step costs 0.50% of d24 throughput
# (GH200, 25-step windows, 8 warm-up, alternated: 351.4K / 351.8K tok/s without, 349.9K / 349.8K with), every 10th
# step is within the windows' spread (7e77833: 341,290-341,471 tok/s with or without)
UPDATE_STATS_EVERY = 10


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
        """One micro-batch's per-row loss sums and valid counts, and its logsumexp statistics (lm_loss outputs). No
        autograd: lm_loss's per-position outputs require grad, and a graph from them copied into the pinned buffers
        would chain every step's onto the last (~12.7 MiB per step at d24, never freed)."""
        valid = labels != IGNORE_INDEX
        self.row_stats.append(torch.stack([losses.view(labels.shape).sum(1), valid.sum(1).float()]))
        valid = valid.flatten()
        self.lse_max.append(torch.where(valid, lse, -torch.inf).amax())
        self.lse_square.append((lse.square() * valid).sum())

    @torch.no_grad()
    def end_step(self, step: int, loss: Tensor, grad_norm: Tensor) -> tuple[Tensor, tuple]:
        """After the optimizers stepped (the groups' lr still the step's): the step's values copied to a pinned host
        buffer without blocking, and the layout record() reads them by."""
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
