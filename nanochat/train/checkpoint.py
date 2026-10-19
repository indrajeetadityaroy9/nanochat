"""
base_train checkpoints in <base_dir>/base_checkpoints/<tag>/: model_<step>.pt (the raw weights), optim_<step>.pt (the
optimizers' states and the tail averages) and meta_<step>.json, written in that order, so a checkpoint is complete
exactly when its meta file exists.
"""

import json
import os
import re

import torch

from nanochat.data.tokenizer import get_tokenizer
from nanochat.model.config import GPTConfig
from nanochat.model.gpt import GPT
from nanochat.paths import base_dir, write_json
from nanochat.train.tail_average import TailAverage


def checkpoint_dir(model_tag):
    return os.path.join(base_dir(), "base_checkpoints", model_tag)


def checkpoint_file(directory, kind, step):
    """The checkpoint's file of `kind`: model, optim (.pt), meta (.json)."""
    extension = {"model": "pt", "optim": "pt", "meta": "json"}[kind]
    return os.path.join(directory, f"{kind}_{step:06d}.{extension}")


def save_checkpoint(directory, step, model_data, optimizer_data, meta_data):
    """A full checkpoint. Deletes the older checkpoints (meta file first)."""
    os.makedirs(directory, exist_ok=True)
    torch.save(model_data, checkpoint_file(directory, "model", step))
    torch.save(optimizer_data, checkpoint_file(directory, "optim", step))
    write_json(checkpoint_file(directory, "meta", step), meta_data)
    for old in checkpoint_steps(directory)[:-1]:
        for kind in ("meta", "model", "optim"):
            os.remove(checkpoint_file(directory, kind, old))


def checkpoint_steps(directory):
    """The steps of the complete checkpoints in the directory, ascending."""
    if not os.path.isdir(directory):
        return []
    return sorted(int(m[1]) for f in os.listdir(directory) if (m := re.fullmatch(r"meta_(\d+)\.json", f)))


def load_checkpoint(directory, step, device, load_optimizer=False):
    """(model state dict, optimizer state dict or None, metadata) of the checkpoint at `step`, tensors on device."""
    model_data = torch.load(checkpoint_file(directory, "model", step), map_location=device)
    optimizer_data = torch.load(checkpoint_file(directory, "optim", step), map_location=device) if load_optimizer else None
    with open(checkpoint_file(directory, "meta", step), encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def load_model(device, model_tag, estimator):
    """(model in eval mode, tokenizer, metadata) of a base_train run's newest checkpoint. estimator 'raw': its weights;
    'tail': its weights with the tail averages of its optimizer file written in (train/tail_average.py; the run's final
    estimator, defined at the final checkpoint)."""
    directory = checkpoint_dir(model_tag)
    step = checkpoint_steps(directory)[-1]
    model_data, _, meta_data = load_checkpoint(directory, step, device)
    with torch.device("meta"):
        model = GPT(GPTConfig(**meta_data["model_config"]))
    model.load_state_dict(model_data, strict=True, assign=True)  # every parameter, so the meta model needs no storage
    if estimator == "tail":
        tail = TailAverage(model, step)
        # memory-mapped: only the tail averages are read, not the optimizer state
        tail.load_state_dict(torch.load(checkpoint_file(directory, "optim", step), map_location="cpu", mmap=True)["tail_average"])
        tail.finalize()
    model.init_rotary()
    model.eval()
    return model, get_tokenizer(), meta_data
