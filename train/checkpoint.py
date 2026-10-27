"""
Checkpoints, each written to a temp file then renamed (atomic): RUNS/<run_name>/checkpoint-<step>.pt, a run's full
state (resumed from; evaluated if among run.checkpoint_steps); weights-<step>.pt, its fp32 weights and model config
only (for analysis; never resumed from or evaluated).
"""

from pathlib import Path

import torch

from model.modeling import CausalLM, ModelConfig
from paths import RUNS


def checkpoint_path(run_name: str, step: int) -> Path:
    return RUNS / run_name / f"checkpoint-{step}.pt"


def weights_path(run_name: str, step: int) -> Path:
    return RUNS / run_name / f"weights-{step}.pt"


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)


def load_model(run_name: str, step: int, *, device: torch.device) -> CausalLM:
    """A checkpoint's model at the context and ModelConfig (output-logit cap included) its weights trained in."""
    checkpoint = torch.load(checkpoint_path(run_name, step), map_location=device)
    with torch.device("meta"):
        model = CausalLM(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"], assign=True)
    model.init_buffers()
    return model
