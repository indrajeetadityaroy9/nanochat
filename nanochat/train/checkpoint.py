"""
A base_train run's checkpoints under <base_dir>/base_checkpoints/<tag>/: checkpoint.pt, the newest, the one a run resumes
from, one torch.save of {"step", "model" (the raw weights), "optimizers" (their state dicts), "model_config",
"stage_rows"} written to a temporary file and moved over the previous one (os.replace, atomic on POSIX), so that the
path always holds one complete checkpoint; and step_<step, 6 digits>.pt, the same at each step the run keeps
(train --keep: production milestones, the lr-horizon study's branch points), hard links of the newest at that step.
"""

import os

import torch

from nanochat.data.tokenizer import Tokenizer, get_tokenizer
from nanochat.model.gpt import GPT, GPTConfig
from nanochat.paths import base_dir


def checkpoint_path(model_tag: str, step: int | None = None) -> str:
    """The run's newest checkpoint, or (step) the one it kept at that step."""
    name = "checkpoint.pt" if step is None else f"step_{step:06d}.pt"
    return os.path.join(base_dir(), "base_checkpoints", model_tag, name)


def save_checkpoint(path: str, checkpoint: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_model(device: torch.device, model_tag: str, step: int | None = None) -> tuple[GPT, Tokenizer, int]:
    """(model, tokenizer, step) of a run's checkpoint (checkpoint_path): its raw weights, at the stage (context and
    rotary base) the run was in (model_config)."""
    checkpoint = torch.load(checkpoint_path(model_tag, step), map_location=device)
    with torch.device("meta"):
        model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], assign=True)  # every parameter, so the meta model needs no storage
    model.init_buffers()
    return model, get_tokenizer(), checkpoint["step"]
