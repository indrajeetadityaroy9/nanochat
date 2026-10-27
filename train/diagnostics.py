"""
Per-step training diagnostics, computed on the device and read one step late with the loss. The train record gets
per-corpus loss, logsumexp max and mean square (PaLM's z-loss quantity, arXiv 2204.02311 §5) and the largest QK-norm
gain; the run's float32 sidecar (RUNS/<run>/COLUMNS_FILE names its columns) gets, every step, each parameter tensor's
gradient norm before clipping, each QK-norm gain's max norm, resid_lambdas and x0_lambdas, and every
UPDATE_STATS_EVERY steps each tensor's parameter norm and update RMS (without weight decay); those steps' train records
also get the GPU's NVML telemetry (power, SM clock, temperature, used memory), read on the host.
"""

import json
from pathlib import Path

import numpy as np
import pynvml
import torch
from torch import Tensor

from corpus.loader import IGNORE_INDEX
from corpus.mixture import Stage, mixture_rows
from model.modeling import CausalLM
from paths import RUNS

# every 10th step's cost is within throughput noise (GH200)
UPDATE_STATS_EVERY = 10
# the sidecar under RUNS/<run>/: .npy float32 arrays, NaN until written, row s the step s (STEP_FILE) or the step
# s x UPDATE_STATS_EVERY (UPDATE_FILE) of the latest attempt to log it; COLUMNS_FILE: their columns and the tensors
STEP_FILE, UPDATE_FILE, COLUMNS_FILE = "diagnostics-step.npy", "diagnostics-update.npy", "diagnostics.json"
# end_step's leading scalars: loss, grad_norm, logsumexp max and square sum
SCALARS = 4


def sidecar(path: Path, shape: tuple[int, int]) -> np.memmap:
    """A sidecar array: the existing file (a resumed attempt overwrites the rows of the steps it repeats), else a new
    one of NaN."""
    if path.exists():
        return np.lib.format.open_memmap(path, mode="r+")
    rows = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    rows[:] = np.nan
    return rows


class Diagnostics:
    def __init__(self, model: CausalLM, opts: tuple[torch.optim.Optimizer, ...], *, gradient_accumulation_steps: int,
                 run_name: str, max_steps: int) -> None:
        self.adamw, self.muon = opts
        named = list(model.named_parameters())
        self.params = [p for _, p in named]
        gains = [(name, p) for name, p in named if name.endswith(("q_norm_weight", "k_norm_weight"))]
        self.gains = [p for _, p in gains]
        self.lambdas = (model.resid_lambdas, model.x0_lambdas)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        # NVML (the image's pynvml) initialized once; the handle of the model's GPU, found by its UUID
        pynvml.nvmlInit()
        uuid = torch.cuda.get_device_properties(model.lm_head.weight.device).uuid
        self.gpu = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}")
        groups = {p: f"{name}[{i}]" for name, opt in zip(("adamw", "muon"), opts, strict=True)
                  for i, group in enumerate(opt.param_groups) for p in group["params"]}
        layers = range(model.config.num_hidden_layers)
        step_columns = ([f"grad_norm/{name}" for name, _ in named] + [f"gain_max/{name}" for name, _ in gains]
                        + [f"resid_lambdas/{i}" for i in layers] + [f"x0_lambdas/{i}" for i in layers])
        update_columns = [f"{kind}/{name}" for kind in ("param_norm", "update_rms") for name, _ in named]
        self.gain_columns = slice(len(named), len(named) + len(gains))
        directory = RUNS / run_name
        (directory / COLUMNS_FILE).write_text(json.dumps({
            "update_stats_every": UPDATE_STATS_EVERY,
            # layer: the decoder layer of layers.<i>.* and value_embeds.<i>.*, null for the model-wide tensors
            "tensors": [{"name": name, "shape": list(p.shape), "group": groups[p],
                         "layer": int(name.split(".")[1]) if name.startswith(("layers.", "value_embeds.")) else None}
                        for name, p in named],
            STEP_FILE: step_columns, UPDATE_FILE: update_columns}, indent=1), encoding="utf-8")
        self.step_rows = sidecar(directory / STEP_FILE, (max_steps, len(step_columns)))
        self.update_rows = sidecar(directory / UPDATE_FILE,
                                   (-(-max_steps // UPDATE_STATS_EVERY), len(update_columns)))

    def set_stage(self, stage: Stage, mixture: dict[str, float]) -> None:
        """The stage's corpora (mixture order, as the loader's) and the pinned host buffers of a step's values."""
        self.names, self.weights = list(mixture), np.array(list(mixture.values()))
        self.micro_batch_size = stage.micro_batch_size
        # end_step's layout: the scalars, a step row, an update row (measured steps), the rows' loss sums and counts
        size = (SCALARS + self.step_rows.shape[1] + self.update_rows.shape[1]
                + 2 * self.gradient_accumulation_steps * self.micro_batch_size)
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

    def update_rms(self) -> Tensor:
        """Each tensor's update RMS without weight decay, model order: AdamW's lr m̂ / (√v̂ + eps) from its state,
        Muon's from Muon.update_norms."""
        rms = {}
        for group in self.adamw.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                state = self.adamw.state[p]
                step_count = state["step"]
                rms[p] = (state["exp_avg"] / (1 - beta1**step_count) / (
                    state["exp_avg_sq"].sqrt() / (1 - beta2**step_count).sqrt() + group["eps"])
                ).square().mean().sqrt() * group["lr"]
        for params, norms in self.muon.update_norms:
            for p, norm in zip(params, norms.unbind(0), strict=True):
                rms[p] = norm / p.numel() ** 0.5
        return torch.stack([rms[p] for p in self.params])

    @torch.no_grad()
    def end_step(self, step: int, loss: Tensor, grad_norm: Tensor, grad_norms: Tensor) -> tuple[Tensor, tuple]:
        """After the optimizers stepped: the step's values (grad_norms: optim.update's per-tensor norms) copied to a
        pinned host buffer without blocking, and their layout."""
        parts = [torch.stack([loss, grad_norm, torch.stack(self.lse_max).amax(), torch.stack(self.lse_square).sum()]),
                 grad_norms, torch.stack(torch._foreach_norm(self.gains, float("inf"))), *self.lambdas]
        if self.measured:
            parts += [torch.stack(torch._foreach_norm(self.params)), self.update_rms()]
        parts.append(torch.stack(self.row_stats).flatten())
        values = torch.cat(parts).double()
        host = self.host[step % 2][:values.numel()]
        host.copy_(values, non_blocking=True)
        return host, (step, self.names, self.weights, self.micro_batch_size, self.rows_start, self.measured)

    def record(self, values: np.ndarray, layout: tuple) -> dict:
        """From a step's host values (after its end event): its sidecar rows written, and its train record's
        diagnostics."""
        step, names, weights, micro_batch_size, rows_start, measured = layout
        lse_max, lse_square = values[2:SCALARS]
        end = SCALARS + self.step_rows.shape[1]
        self.step_rows[step] = values[SCALARS:end]
        if measured:
            self.update_rows[step // UPDATE_STATS_EVERY] = values[end:end + self.update_rows.shape[1]]
            end += self.update_rows.shape[1]
        rows = values[end:].reshape(self.gradient_accumulation_steps, 2, micro_batch_size)
        corpora = mixture_rows(weights, start=rows_start, count=self.gradient_accumulation_steps * micro_batch_size)
        sums = np.bincount(corpora, rows[:, 0].flatten(), minlength=len(names))
        counts = np.bincount(corpora, rows[:, 1].flatten(), minlength=len(names))
        # a corpus whose weight is below one row per step may draw no rows in it
        out = {"max_logsumexp": float(lse_max), "mean_square_logsumexp": float(lse_square / counts.sum()),
               "qk_norm_gain_max": float(values[SCALARS:][self.gain_columns].max()),
               "corpus_loss": {name: float(s / c) for name, s, c in zip(names, sums, counts, strict=True) if c}}
        if measured:  # NVML's readings now, while the next step runs; no device sync
            out |= {"gpu_power_watts": pynvml.nvmlDeviceGetPowerUsage(self.gpu) / 1000,
                    "gpu_sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(self.gpu, pynvml.NVML_CLOCK_SM),
                    "gpu_temperature_c": pynvml.nvmlDeviceGetTemperature(self.gpu, pynvml.NVML_TEMPERATURE_GPU),
                    "gpu_memory_used_bytes": pynvml.nvmlDeviceGetMemoryInfo(self.gpu).used}
        return out

    def flush(self) -> None:
        """The sidecar's written rows to disk."""
        self.step_rows.flush()
        self.update_rows.flush()
