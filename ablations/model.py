"""
Production's GPT (nanochat/model/gpt.py) with the value-embedding experiment's alternative (harness.Run): value_embeds
"none", no value-embedding tables and gates; with "alternating" it is GPT.
"""
from typing import Literal

from torch import nn

from nanochat.model.gpt import GPT, GPTConfig


class Candidate(GPT):
    def __init__(self, config: GPTConfig, *, value_embeds: Literal["alternating", "none"]) -> None:
        """Shapes only (meta device)."""
        super().__init__(config)
        if value_embeds == "none":
            self.value_embeds = nn.ModuleDict()
            for block in self.blocks:
                block.attn.ve_gate = None
