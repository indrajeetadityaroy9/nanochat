"""
Run plan (batch, steps, repo-level start, peak learning rate) from DeepSeek LLM's compute-optimal laws, each step's
stage, and the per-step learning-rate multiplier: one schedule across both stages, the linear decay to zero
continuing through the repo-level stage (arXiv 2502.15938 §4.2).
"""
from dataclasses import dataclass

from corpus.mixture import FILE_LEVEL_STAGE, MICRO_BATCH_TOKENS, REPO_LEVEL_SHARE, REPO_LEVEL_STAGE, Stage
from model.modeling import CausalLM, non_embedding_flops_per_token, num_non_embedding_params
from train.optim import BETAS

# Linear warmup over 2 / (1 - beta2) steps (arXiv 1910.04209 §4.3 Eq. 14): 40 at beta2 = 0.95
WARMUP_STEPS = round(2 / (1 - BETAS[1]))


@dataclass(frozen=True, kw_only=True)
class RunPlan:
    batch_size: int                   # tokens per optimizer step
    gradient_accumulation_steps: int  # micro-batches per step
    max_steps: int
    repo_level_start: int             # first step of the repo-level stage
    num_non_embedding_params: int     # model.modeling.num_non_embedding_params
    tokens_per_param: float           # max_steps x batch_size / num_non_embedding_params
    learning_rate: float              # peak learning rate of every optimizer group

    def stage_at(self, step: int) -> Stage:
        """The stage step `step` trains in: the repo-level stage from repo_level_start on (Qwen2.5-Coder,
        arXiv 2409.12186 §3.2)."""
        return REPO_LEVEL_STAGE if step >= self.repo_level_start else FILE_LEVEL_STAGE


def plan(*, model: CausalLM, tokens_per_param: float) -> RunPlan:
    """Plan of `model` (any device, its file-level stage's context) at `tokens_per_param` tokens per non-embedding
    parameter. With C = M D (M: DeepSeek LLM's Eq. 2 FLOPs per token at the file-level stage's context,
    model/modeling.py), B_opt = 0.2920 C^0.3271 tokens, rounded to whole micro-batches, at least one, and
    eta_opt = 0.3118 C^-0.1250 (DeepSeek LLM, arXiv 2401.02954 §3.1 Eq. 1). Each run uses its own compute's batch and
    learning rate for both stages."""
    params = num_non_embedding_params(model)
    tokens = round(tokens_per_param * params)
    compute = non_embedding_flops_per_token(model) * tokens
    batch_size = max(1, round(0.2920 * compute ** 0.3271 / MICRO_BATCH_TOKENS)) * MICRO_BATCH_TOKENS
    max_steps = tokens // batch_size
    return RunPlan(
        batch_size=batch_size, gradient_accumulation_steps=batch_size // MICRO_BATCH_TOKENS,
        max_steps=max_steps,
        repo_level_start=max_steps - round(REPO_LEVEL_SHARE * max_steps),
        num_non_embedding_params=params, tokens_per_param=max_steps * batch_size / params,
        learning_rate=0.3118 * compute ** -0.1250,
    )


def lr_lambda(current_step: int, *, num_training_steps: int) -> float:
    """LambdaLR multiplier at step `current_step`: linear warmup over WARMUP_STEPS steps, then linear decay to 0
    through the repo-level stage (arXiv 2502.15938 §4.2)."""
    if current_step < WARMUP_STEPS:
        return (current_step + 1) / WARMUP_STEPS
    return (num_training_steps - current_step) / (num_training_steps - WARMUP_STEPS)
