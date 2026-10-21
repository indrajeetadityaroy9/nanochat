"""
Run plan (batch, steps, repo-level start, peak lr) from DeepSeek LLM's compute-optimal laws, each step's stage, and the
per-step lr multiplier: one schedule across both stages (DeepSeek-Coder keeps its final pre-training lr for long
context, arXiv 2401.14196 §3.6).
"""
from dataclasses import dataclass

from nanochat.config import FILE_STAGE, MICRO_BATCH_TOKENS, REPO_SHARE, REPO_STAGE, Stage
from nanochat.model.gpt import GPT, non_embedding_flops_per_token, scaling_params
from nanochat.train.optim import BETAS

# Linear warmup over 2 / (1 - beta2) steps (Ma & Yarats, arXiv 1910.04209): 40 at beta2 = 0.95
WARMUP = round(2 / (1 - BETAS[1]))


@dataclass(frozen=True, kw_only=True)
class RunPlan:
    total_batch_size: int    # tokens per optimizer step
    grad_accum_steps: int    # micro-batches per step
    num_iterations: int
    repo_start: int          # first step of the repo-level stage
    scaling_params: int      # model.gpt.scaling_params
    tokens_per_param: float  # num_iterations x total_batch_size / scaling_params
    lr: float                # peak lr of every optimizer group

    def stage_at(self, step: int) -> Stage:
        """The stage step `step` trains in: the repo-level stage from repo_start on (Qwen2.5-Coder, arXiv 2409.12186
        §3.2)."""
        return REPO_STAGE if step >= self.repo_start else FILE_STAGE


def plan(*, model: GPT, ratio: float) -> RunPlan:
    """Plan of `model` (any device, its file-level stage's context) at `ratio` tokens per scaling parameter. With C = M D
    (M: non-embedding training FLOPs per token at the file-level stage), B_opt = 0.2920 C^0.3271 tokens, rounded to
    whole micro-batches, at least one (d8 at 5 tokens per parameter: 0.46), and eta_opt = 0.3118 C^-0.1250 (DeepSeek
    LLM, arXiv 2401.02954 §3.1 Eq. 1). Every run, proxy or production, uses its own compute's batch and lr for both
    stages."""
    params = scaling_params(model)
    compute = non_embedding_flops_per_token(model) * int(ratio * params)
    total_batch_size = max(1, round(0.2920 * compute ** 0.3271 / MICRO_BATCH_TOKENS)) * MICRO_BATCH_TOKENS
    num_iterations = int(ratio * params) // total_batch_size
    return RunPlan(
        total_batch_size=total_batch_size, grad_accum_steps=total_batch_size // MICRO_BATCH_TOKENS,
        num_iterations=num_iterations,
        repo_start=num_iterations - round(REPO_SHARE * num_iterations),
        scaling_params=params, tokens_per_param=num_iterations * total_batch_size / params,
        lr=0.3118 * compute ** -0.1250,
    )


def lr_multiplier(it: int, *, steps: int) -> float:
    """LambdaLR multiplier at step `it`: linear warmup over WARMUP steps, then linear decay to 0 through the repo-level
    stage (Bergsma et al., arXiv 2502.15938: beats WSD and cosine, more so at more tokens per parameter)."""
    if it < WARMUP:
        return (it + 1) / WARMUP
    return (steps - it) / (steps - WARMUP)
