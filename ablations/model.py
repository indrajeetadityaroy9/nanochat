"""
Production's GPT (nanochat/model/gpt.py) with the architecture experiments' alternatives (harness.Run), each off in
production; with all off every method is GPT's:
- mlp "swiglu": SwiGLU, hidden 8C/3 (ablations/swiglu.py) in place of ReLU^2.
- value_embeds "none": no value-embedding tables and gates.
- rope "staged": RoPE base 10^4 in the file-level stage, config.ROPE_BASE (10^6) in the repository stage, as Code
  Llama's long-context stage raises it (arXiv 2308.12950 §2.4) (production: ROPE_BASE in both).
"""
from torch import nn

from ablations.swiglu import SwiGLU
from nanochat.config import REPO_STAGE, ROPE_BASE
from nanochat.model.gpt import GPT


class Candidate(GPT):
    def __init__(self, config, *, mlp: str, value_embeds: str, rope: str):
        """Shapes only (meta device)."""
        super().__init__(config)
        self.rope = rope
        if mlp == "swiglu":
            for block in self.blocks:
                block.mlp = SwiGLU(config)
        if value_embeds == "none":
            self.value_embeds = nn.ModuleDict()
            for block in self.blocks:
                block.attn.ve_gate = None

    def set_stage(self, stage):
        if self.rope == "staged":
            self.config.rope_base = ROPE_BASE if stage is REPO_STAGE else 10_000  # Llama 2's base, Code Llama's start
        super().set_stage(stage)
