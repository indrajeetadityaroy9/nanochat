"""
Checkpoints: RUNS/<run_name>/checkpoint-<step>.pt (run_name: run.py run_name), a run's full state after `step` steps
({"model", "optimizers", "config", "stage_rows"}), each written to a temp file then renamed, so a run killed
mid-save never leaves a partial one. Every checkpoint is both a resume point (a run resumes from its newest) and a
model to evaluate.
"""

from pathlib import Path

import torch

from model.modeling import CausalLM, ModelConfig
from paths import RUNS


def checkpoint_path(run_name: str, step: int) -> Path:
    return RUNS / run_name / f"checkpoint-{step}.pt"


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)


def load_model(run_name: str, step: int, *, device: torch.device) -> CausalLM:
    """A checkpoint's model, at the context its weights trained in (its ModelConfig, the output-logit cap
    included)."""
    checkpoint = torch.load(checkpoint_path(run_name, step), map_location=device)
    with torch.device("meta"):
        model = CausalLM(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"], assign=True)
    model.init_buffers()
    return model
