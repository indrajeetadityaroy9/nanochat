"""
Checkpoints: CHECKPOINTS/<tag>-step<step>.pt, a run's full state after `step` steps ({"model", "optimizers",
"model_config", "stage_rows"}), each written to a temp file then renamed, so a run killed mid-save never leaves a
partial one. Every checkpoint is both a resume point (a run resumes from its newest) and a model to evaluate.
"""

from pathlib import Path

import torch

from nanochat.model.gpt import GPT, GPTConfig
from nanochat.paths import CHECKPOINTS


def checkpoint_name(tag: str, step: int) -> str:
    """<tag>-step<step>; a checkpoint's evaluation (eval/suite.py evaluate) is named the same."""
    return f"{tag}-step{step}"


def checkpoint_path(tag: str, step: int) -> Path:
    return CHECKPOINTS / f"{checkpoint_name(tag, step)}.pt"


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)


def load_model(tag: str, step: int, *, device: torch.device) -> GPT:
    """A checkpoint's model, at the context its weights trained in."""
    checkpoint = torch.load(checkpoint_path(tag, step), map_location=device)
    with torch.device("meta"):
        model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], assign=True)
    model.init_buffers()
    return model
