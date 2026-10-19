"""
Run plan (batch, steps, repo-level start, peak lr) from DeepSeek LLM's compute-optimal laws, and the per-step lr
multiplier: one schedule over both stages (DeepSeek-Coder keeps its final pre-training lr for long context, arXiv
2401.14196 §3.6).

python -m nanochat.train.schedule <depth> <ratio> <seed> prints the run's tag, steps and repo_start.
"""

import sys
from dataclasses import dataclass
from typing import Literal

import torch

from nanochat.config import FILE_STAGE, MICRO_BATCH_TOKENS, REPO_SHARE, run_tag
from nanochat.model.gpt import GPT, gpt_config, non_embedding_flops_per_token, scaling_params
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


def plan(*, depth: int, ratio: float, tokens: float | None) -> RunPlan:
    """The plan at `ratio` tokens per scaling parameter. Compute C = M D (M: non-embedding training FLOPs per token at
    the file-level stage) gives B_opt = 0.2920 C^0.3271 tokens, rounded to whole micro-batches, and eta_opt =
    0.3118 C^-0.1250 (DeepSeek LLM, arXiv 2401.02954 §3.1 Eq. 1). `tokens`: the run cut to that horizon at the same
    batch and lr, file-level stage only."""
    with torch.device("meta"):
        model = GPT(gpt_config(depth=depth, stage=FILE_STAGE))
    params = scaling_params(model)
    compute = non_embedding_flops_per_token(model) * int(ratio * params)
    total_batch_size = round(0.2920 * compute ** 0.3271 / MICRO_BATCH_TOKENS) * MICRO_BATCH_TOKENS
    num_iterations = int(tokens or ratio * params) // total_batch_size
    return RunPlan(
        total_batch_size=total_batch_size, grad_accum_steps=total_batch_size // MICRO_BATCH_TOKENS,
        num_iterations=num_iterations,
        repo_start=num_iterations if tokens else num_iterations - round(REPO_SHARE * num_iterations),
        scaling_params=params, tokens_per_param=num_iterations * total_batch_size / params,
        lr=0.3118 * compute ** -0.1250,
    )


def lr_multiplier(it: int, *, steps: int, repo_start: int, repo_schedule: Literal["continue", "rewarm"]) -> float:
    """The lr multiplier at step `it` (LambdaLR's argument): linear warmup over WARMUP steps, then linear decay to 0
    (Bergsma et al., arXiv 2502.15938: beats WSD and cosine, more so at more tokens per parameter). repo_schedule
    "rewarm" restarts at repo_start: warmup to the peak, then linear decay to 0 (SmolLM3's separate long-context
    schedule, Hugging Face 2025)."""
    if repo_schedule == "rewarm" and it >= repo_start:
        k = it - repo_start
        return (k + 1) / WARMUP if k < WARMUP else (steps - it) / (steps - repo_start - WARMUP)
    if it < WARMUP:
        return (it + 1) / WARMUP
    return (steps - it) / (steps - WARMUP)


if __name__ == "__main__":
    depth, ratio, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    run = plan(depth=depth, ratio=ratio, tokens=None)
    print(run_tag(depth=depth, ratio=ratio, seed=seed, data_seed=seed, changes={}), run.num_iterations, run.repo_start)
