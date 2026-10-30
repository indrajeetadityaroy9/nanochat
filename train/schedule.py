"""
Run plan from DeepSeek LLM's compute-optimal laws (an SFT run's from its data), each step's stage, and the
learning-rate multiplier: warmup, then linear decay to zero through the last stage (arXiv 2502.15938 §4.2).
"""
from dataclasses import dataclass

from corpus.mixture import BATCH_SIZE, FILE_LEVEL_STAGE, MICRO_BATCH_TOKENS, REPO_LEVEL_SHARE, REPO_LEVEL_STAGE, Stage
from model.modeling import CausalLM, non_embedding_flops_per_token, num_non_embedding_params
from posttrain.sft import SFT_STAGE
from train.optim import BETAS

# Linear warmup over 2 / (1 - beta2) steps (arXiv 1910.04209 §4.3 Eq. 14): 40 at beta2 = 0.95
WARMUP_STEPS = round(2 / (1 - BETAS[1]))


@dataclass(frozen=True, kw_only=True)
class RunPlan:
    batch_size: int                   # tokens per optimizer step
    gradient_accumulation_steps: int
    max_steps: int
    repo_level_start: int             # the last stage's first step
    tokens_per_param: float           # max_steps x batch_size / num_non_embedding_params
    learning_rate: float              # peak learning rate of every optimizer group
    weight_decay: float | None = None  # None: the tau law's at learning_rate (train/optim.py tau_weight_decay)
    last_stage: Stage = REPO_LEVEL_STAGE

    def stage_at(self, step: int) -> Stage:
        """The stage step `step` trains in (repo-level stage as Qwen2.5-Coder, arXiv 2409.12186 §3.2)."""
        return self.last_stage if step >= self.repo_level_start else FILE_LEVEL_STAGE


def plan(*, model: CausalLM, tokens_per_param: float) -> RunPlan:
    """Plan of `model` at `tokens_per_param`: C = M D (M: Eq. 2 FLOPs per token at the file-level context),
    B_opt = 0.2920 C^0.3271 tokens in whole micro-batches, eta_opt = 0.3118 C^-0.1250 (DeepSeek LLM,
    arXiv 2401.02954 §3.1 Eq. 1), for both stages."""
    params = num_non_embedding_params(model)
    tokens = round(tokens_per_param * params)
    compute = non_embedding_flops_per_token(model) * tokens
    batch_size = round(0.2920 * compute ** 0.3271 / MICRO_BATCH_TOKENS) * MICRO_BATCH_TOKENS
    max_steps = tokens // batch_size
    return RunPlan(
        batch_size=batch_size, gradient_accumulation_steps=batch_size // MICRO_BATCH_TOKENS,
        max_steps=max_steps,
        repo_level_start=max_steps - round(REPO_LEVEL_SHARE * max_steps),
        tokens_per_param=max_steps * batch_size / params, learning_rate=0.3118 * compute ** -0.1250,
    )


def sft_plan(*, model: CausalLM, rows: int, learning_rate: float) -> RunPlan:
    """SFT's plan (docs/post-training.md §11.2): SFT_STAGE from the first step, the base plan's 15 micro-batches per
    step (45 rows of 32,768 tokens), whole steps over `rows` packed rows, at peak `learning_rate`, no weight decay."""
    gradient_accumulation_steps = BATCH_SIZE // MICRO_BATCH_TOKENS
    max_steps = rows // (gradient_accumulation_steps * SFT_STAGE.micro_batch_size)
    return RunPlan(batch_size=BATCH_SIZE, gradient_accumulation_steps=gradient_accumulation_steps,
                   max_steps=max_steps, repo_level_start=0,
                   tokens_per_param=max_steps * BATCH_SIZE / num_non_embedding_params(model),
                   learning_rate=learning_rate, weight_decay=0.0, last_stage=SFT_STAGE)


def lr_lambda(current_step: int, *, num_training_steps: int) -> float:
    """LambdaLR multiplier at step `current_step` (module docstring)."""
    if current_step < WARMUP_STEPS:
        return (current_step + 1) / WARMUP_STEPS
    return (num_training_steps - current_step) / (num_training_steps - WARMUP_STEPS)
