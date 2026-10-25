"""
Checkpoints: RUNS/<tag>/step<step>.pt (tag: the run's key, run.py), a run's full state after `step` steps
({"model", "optimizers", "model_config", "stage_rows"}), each written to a temp file then renamed, so a run killed
mid-save never leaves a partial one. Every checkpoint is both a resume point (a run resumes from its newest) and a
model to evaluate.
"""

from pathlib import Path

import torch

from model.gpt import GPT, GPTConfig
from paths import RUNS


def checkpoint_path(tag: str, step: int) -> Path:
    return RUNS / tag / f"step{step}.pt"


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)


def load_model(tag: str, step: int, *, device: torch.device) -> GPT:
    """A checkpoint's model, at the context its weights trained in (its GPTConfig, the output-logit cap included)."""
    checkpoint = torch.load(checkpoint_path(tag, step), map_location=device)
    with torch.device("meta"):
        model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], assign=True)
    model.init_buffers()
    return model
