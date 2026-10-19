"""
The run plan of a training: batch, steps, the first step of the repo-level stage and the peak lr, derived from the depth
and the tokens per scaling parameter by DeepSeek LLM's compute-optimal hyperparameter laws; the lr multiplier of each
step, one schedule over both stages (the repo-level stage continues the run's learning rate, as DeepSeek-Coder's
long-context training keeps its final pre-training rate, arXiv 2401.14196 §3.6).

python -m nanochat.train.schedule <depth> <ratio> <seed> prints the run's tag, steps and repo_start (the run scripts'
one source of them).
"""

import sys
from dataclasses import dataclass
from typing import Literal

import torch

from nanochat.config import FILE_STAGE, MICRO_BATCH_TOKENS, REPO_SHARE
from nanochat.model.gpt import ASPECT_RATIO, GPT, gpt_config, non_embedding_flops_per_token, scaling_params
from nanochat.train.optim import BETAS

# Linear warmup over 2 / (1 - beta2) steps (Ma & Yarats, arXiv 1910.04209, abstract: "As a sensible untuned default, we
# recommend linear warmup over 2 · (1 − β2)^−1 iterations"): 40 at beta2 = 0.95
WARMUP = round(2 / (1 - BETAS[1]))


@dataclass(frozen=True, kw_only=True)
class RunPlan:
    """The derived settings of one run (plan); the run record holds them, all that reproduces its hyperparameters with
    train/optim.py's fixed constants."""
    total_batch_size: int    # tokens per optimizer step
    grad_accum_steps: int    # micro-batches per step
    num_iterations: int
    repo_start: int          # the first step of the repo-level stage (config.REPO_STAGE): its share REPO_SHARE of steps
    scaling_params: int      # the model's scaling parameters (model.gpt.scaling_params)
    tokens_per_param: float  # num_iterations x total_batch_size / scaling_params
    lr: float                # the peak lr of every optimizer group


def plan(depth: int, ratio: float, vocab_size: int, batch_scale: float = 1) -> RunPlan:
    """The run plan of the model of this depth at `ratio` training tokens D per scaling parameter. Its compute
    C = M D, M the model's non-embedding training FLOPs per token at the file-level stage, gives the batch and lr by
    DeepSeek LLM's laws (arXiv 2401.02954 §3.1 Eq. 1): B_opt = 0.2920 C^0.3271 tokens, rounded to whole micro-batches
    (at least one), eta_opt = 0.3118 C^-0.1250. batch_scale: the batch study's factor on B_opt (1 in the regime)."""
    with torch.device("meta"):
        model = GPT(gpt_config(depth, vocab_size, FILE_STAGE, aspect_ratio=ASPECT_RATIO))
    params = scaling_params(model)
    tokens = int(ratio * params)
    compute = non_embedding_flops_per_token(model) * tokens
    total_batch_size = max(1, round(batch_scale * 0.2920 * compute ** 0.3271 / MICRO_BATCH_TOKENS)) * MICRO_BATCH_TOKENS
    num_iterations = tokens // total_batch_size
    return RunPlan(
        total_batch_size=total_batch_size, grad_accum_steps=total_batch_size // MICRO_BATCH_TOKENS,
        num_iterations=num_iterations, repo_start=num_iterations - round(REPO_SHARE * num_iterations),
        scaling_params=params, tokens_per_param=num_iterations * total_batch_size / params,
        lr=0.3118 * compute ** -0.1250,
    )


def lr_multiplier(it: int, n: int, shape: Literal["wsd", "d2z"], repo_start: int,
                  repo: Literal["continue", "rewarm"]) -> float:
    """The lr multiplier at step it of an n-step run: linear warmup over WARMUP steps, then
    wsd: constant, then the (1 - sqrt) cooldown to 0 over the last 20% of the steps (Hagele et al., arXiv 2405.18392
        §3.2: constant lr with a 20% cooldown matches cosine, the gains of a longer cooldown plateau at ~20% (Fig. 5),
        and (1 - sqrt), f = 1 - sqrt((it - (n - n_decay)) / n_decay), outperforms the linear cooldown (Eq. (1-sqrt)));
    d2z: linear decay to 0 at step n (Bergsma et al., arXiv 2502.15938: beats WSD and cosine, more so at more tokens
        per parameter).
    From repo_start on, repo "continue" keeps that schedule (DeepSeek-Coder's long-context stage keeps its final
    pre-training rate, arXiv 2401.14196 §3.6); "rewarm" gives the repo-level stage its own: linear warmup over WARMUP
    steps back to the peak, then linear decay to 0 at step n (SmolLM3's separate long-context schedules, Hugging Face
    2025)."""
    if repo == "rewarm" and it >= repo_start:
        k = it - repo_start
        return (k + 1) / WARMUP if k < WARMUP else (n - it) / (n - repo_start - WARMUP)
    if it < WARMUP:
        return (it + 1) / WARMUP
    if shape == "d2z":
        return (n - it) / (n - WARMUP)
    cooldown = round(0.2 * n)
    if it < n - cooldown:
        return 1.0
    return 1 - ((it - n + cooldown) / cooldown) ** 0.5


if __name__ == "__main__":
    from nanochat.config import run_tag
    from nanochat.data.tokenizer import get_tokenizer

    depth, ratio, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    run = plan(depth, ratio, get_tokenizer().vocab_size)
    print(run_tag(depth, ratio, seed), run.num_iterations, run.repo_start)
