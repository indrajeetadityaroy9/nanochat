"""
A base_train run's checkpoint: <base_dir>/base_checkpoints/<tag>/checkpoint.pt, one torch.save of {"step", "model" (the
raw weights), "optimizers" (their state dicts), "model_config", "stage_rows"}, written to a temporary file and moved over
the previous one (os.replace, atomic on POSIX), so that the path always holds one complete checkpoint, the newest.
"""

import os

import torch

from nanochat.data.tokenizer import get_tokenizer
from nanochat.model.gpt import GPT, GPTConfig, rotary_tables
from nanochat.paths import base_dir


def checkpoint_path(model_tag):
    return os.path.join(base_dir(), "base_checkpoints", model_tag, "checkpoint.pt")


def save_checkpoint(path, checkpoint):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_model(device, model_tag):
    """(model, tokenizer, step) of a run's checkpoint: its raw weights, at the stage (context and rotary base) the run was
    in (model_config)."""
    checkpoint = torch.load(checkpoint_path(model_tag), map_location=device)
    with torch.device("meta"):
        model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], assign=True)  # every parameter, so the meta model needs no storage
    model.cos, model.sin = rotary_tables(model.config, device)
    return model, get_tokenizer(), checkpoint["step"]
