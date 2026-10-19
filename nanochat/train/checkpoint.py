"""
Checkpoints under <base_dir>/base_checkpoints/<tag>/: checkpoint.pt, the newest ({"step", "model", "optimizers",
"model_config", "stage_rows"}), written to a temporary file and renamed (os.replace is atomic), so a run killed
mid-save resumes from a complete file; step_<step:06d>.pt, hard links of the newest at the steps the run keeps.
"""

import os

import torch

from nanochat.model.gpt import GPT, GPTConfig
from nanochat.paths import base_dir


def checkpoint_path(tag: str, *, step: int | None) -> str:
    """The newest checkpoint (step None) or the one kept at `step`."""
    name = "checkpoint.pt" if step is None else f"step_{step:06d}.pt"
    return os.path.join(base_dir(), "base_checkpoints", tag, name)


def save_checkpoint(path: str, checkpoint: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_model(tag: str, *, step: int | None, device: torch.device) -> tuple[GPT, int]:
    """(model, step) of a checkpoint, at the stage (context, RoPE base) its weights trained in."""
    checkpoint = torch.load(checkpoint_path(tag, step=step), map_location=device)
    with torch.device("meta"):
        model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], assign=True)
    model.init_buffers()
    return model, checkpoint["step"]
