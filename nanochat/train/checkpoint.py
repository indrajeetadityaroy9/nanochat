"""
Checkpoints are flat files in CHECKPOINTS, each written to a temp file then replaced (atomic), so a run killed mid-save
keeps the previous file. <tag>.pt is the newest, for resuming ({"step", "model", "optimizers", "model_config",
"stage_rows"}); <tag>-step<step:06d>.pt are model-only snapshots ({"step", "model", "model_config"}, no optimizer
state) for evaluating intermediate models.
"""

from pathlib import Path

import torch

from nanochat.model.gpt import GPT, GPTConfig
from nanochat.paths import CHECKPOINTS


def checkpoint_name(tag: str, *, step: int | None) -> str:
    """<tag> for the newest checkpoint (step None), <tag>-step<step:06d> for the snapshot at `step`; base_eval names its
    output for a checkpoint the same."""
    return tag if step is None else f"{tag}-step{step:06d}"


def checkpoint_path(tag: str, *, step: int | None) -> Path:
    return CHECKPOINTS / f"{checkpoint_name(tag, step=step)}.pt"


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)


def load_model(tag: str, *, step: int | None, device: torch.device) -> tuple[GPT, int]:
    """(model, step) of a checkpoint, at the stage (context, RoPE base) its weights trained in."""
    checkpoint = torch.load(checkpoint_path(tag, step=step), map_location=device)
    with torch.device("meta"):
        model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], assign=True)
    model.init_buffers()
    return model, checkpoint["step"]
