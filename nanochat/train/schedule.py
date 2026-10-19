"""
The run plan of a training (dev/D20_SPEC.md "Run settings", "Optimizer transfer"): the quantities derived from the
depth, the tokens per scaling parameter and the objective; the step-indexed schedules (lr multiplier, ANVIL rails, loss
weights), modded-nanogpt track_1_short's with its step counts as fractions of its 1194-step run; the optimizer recipes'
own runs (modded's, nanochat's d12, DeepSeek's Engram-27B: lr schedule, tokens, parameters), the anchors of the
learning rates and weight decays (train/optim.py); the model's training FLOPs per token.
"""

import bisect
import itertools
from dataclasses import dataclass

import torch

from nanochat.config import MICRO_BATCH_ROWS, SEQ_LEN
from nanochat.model.attention import layer_windows
from nanochat.model.config import gpt_config
from nanochat.model.engram import engram_rows
from nanochat.model.gpt import GPT, scaling_params

# base_train --objective: spec = next token + multi-token prediction + prefix-token curricula, FIM pieces as FIM
# examples; ntp = the next token only, FIM pieces read left to right
OBJECTIVES = ("spec", "ntp")
TOKENS_PER_MICRO_BATCH = MICRO_BATCH_ROWS * SEQ_LEN

# Modded's run (config.py:26-27, 59; schedule.py:28, 33): 1122 scheduled + 52 growth steps = the main stages, + 20
# extension steps; rail engage at 514 (anvil.py:48), rail beta warmup 240 and cooldown 50 steps (schedule.py:78)
MODDED_SCHEDULED_STEPS, MODDED_GROWTH_STEPS, MODDED_EXTENSION_STEPS = 1122, 52, 20
MODDED_RUN_STEPS = MODDED_SCHEDULED_STEPS + MODDED_GROWTH_STEPS + MODDED_EXTENSION_STEPS
ANVIL_ENGAGE_STEP, ANVIL_WARMUP_STEPS, ANVIL_COOLDOWN_STEPS = 514, 240, 50
# schedule.py:78 get_rail_beta: Nesterov lookahead from 0.85 to 0.93 and back
ANVIL_BETA_MIN, ANVIL_BETA_MAX = 0.85, 0.93
# anvil.py:48 RAIL_FAST_BETA, RAIL_FAST_WEIGHT: the fast rail's beta and blend weight after the engage point
ANVIL_FAST_BETA, ANVIL_FAST_WEIGHT = 0.85, 0.4385
# config.py:52-59 STAGE_DURATIONS (fractions of the scheduled steps), TAPER_DURATION (the batch-20 taper's share of the
# last), SCHEDULE_GROWTH_STAGE (the stage the growth steps extend); config.py:76-88 the stages' lr_mul and batch_size
# (units of 2048 x 8 tokens): stages 0, 1, 2, the taper (1.73 sqrt(20/24)), the extension
MODDED_STAGE_DURATIONS = (0.2854, 0.3209, 0.3931)
MODDED_TAPER_DURATION, MODDED_GROWTH_STAGE = 0.06, 2
MODDED_STAGE_LR_MULS = (1.0, 1.52, 1.73, 1.73 * (20 / 24) ** 0.5, 1.0)
MODDED_STAGE_BATCHES = tuple(units * 2048 * 8 for units in (8, 16, 24, 20, 8))
# modded's model (train_gpt.py:150-159: vocab 50257, 6 heads of 128, width 768; model/gpt.py:156 the vocabulary padded
# to a multiple of 128): its attention layers (model/gpt.py:57-58 ATTN_BANK_ORDER, 7 layers: q/k/v/o each width x
# heads x head_dim) and MLPs (model/gpt.py:103-107: 12 bank slots, slot 7 dead; hidden 2816)
MODDED_VOCAB, MODDED_HEADS, MODDED_HEAD_DIM, MODDED_WIDTH = 50257, 6, 128, 768
MODDED_ATTN_LAYERS, MODDED_MLPS, MODDED_MLP_HIDDEN = 7, 11, 2816
# config.py:68 LR_COOLDOWN_FRAC: the lr decays linearly over this last share of the run; modded's floor (schedule.py:9
# LR_FLOOR), held over its extension
LR_COOLDOWN_FRAC, MODDED_LR_FLOOR = 0.80, 0.30
# config.py:54-88 TRAINING_STAGES, schedule.py:37-65: MTP weights [1, 0.5, 0.25 -> 0] over stage 0, [1, 0.5 -> 0] over
# stage 1, [1] after (config.py:77); prefix weight 0.25 over stage 0, 0.20/0.15/0.10/0.05 over the quarters of stage 1,
# 0 after
MTP_WEIGHTS = (1.0, 0.5, 0.25)
PREFIX_WEIGHTS = ((0.25,), (0.20, 0.15, 0.10, 0.05))
# nanochat fork HEAD 773a0a9 scripts/base_train.py:258-299: B_REF = 2^19 "measured empirically" at d12; ratio 12 from
# its default --target-param-data-ratio (:56)
NANOCHAT_REF_DEPTH, NANOCHAT_REF_RATIO, B_REF = 12, 12, 2**19
# nanochat 92d63d4 scripts/base_train.py:67-69, 360-369: linear warmup over 40 steps, constant, linear warmdown over
# the last 0.65 of the run to 0.05
NANOCHAT_WARMUP_STEPS, NANOCHAT_WARMDOWN_RATIO, NANOCHAT_FINAL_LR_FRAC = 40, 0.65, 0.05
# Power Lines (Bergsma et al. arXiv 2505.13738): Table 5's tuned-weight-decay row B_opt ~ D^0.383, independent of N;
# Fig. 1 the optimal weight-decay timescale as a fraction of the run tau / T ~ TPP^-0.527 (tokens per parameter)
POWER_LINES_EXPONENT, TAU_EXPONENT = 0.383, 0.527
# DeepSeek's Engram-27B run (arXiv 2601.07372 v2 Table 5: "Training Steps 50000", "Lr Scheduler Step Decay (Bi et al.,
# 2024)"); the step decay (arXiv 2401.02954 §2.3): the maximum after 2000 warmup steps, 31.6% of it after 80% of the
# training tokens, 10% after 90%
DEEPSEEK_STEPS, DEEPSEEK_WARMUP_STEPS = 50_000, 2000
DEEPSEEK_STEP_DECAY = ((0.8, 0.316), (0.9, 0.1))


def depth_scaling_params(depth: int, vocab_size: int) -> int:
    """The scaling parameters (model.gpt.scaling_params) of the model of this depth, built on the meta device. The
    Engram tables are not scaling parameters; their rows bound is the vocabulary size (the demo's table-size base,
    engram_demo_v1.py:246)."""
    with torch.device("meta"):
        return scaling_params(GPT(gpt_config(depth, vocab_size, SEQ_LEN, vocab_size)))


def power_lines_batch(tokens: int, d_ref: int) -> int:
    """The batch for a horizon of `tokens`: B_REF (tokens / d_ref)^0.383 rounded to whole micro-batches, at least one."""
    return max(1, round(B_REF * (tokens / d_ref) ** POWER_LINES_EXPONENT / TOKENS_PER_MICRO_BATCH)) * TOKENS_PER_MICRO_BATCH


def lr_multiplier(it: int, n: int, warmup: int) -> float:
    """The lr multiplier at step it of an n-step run: a linear warmup over `warmup` steps (nanochat 92d63d4
    scripts/base_train.py:363-364), modded's constant-then-linear shape (config.py:68, schedule.py:67-75) decayed to
    zero at the end of the run (linear D2Z for a known horizon: Bergsma et al., ICLR 2025, arXiv 2502.15938)."""
    cool = int(n * (1 - LR_COOLDOWN_FRAC))
    ramp = 1.0 if it >= warmup else (it + 1) / warmup
    return ramp * (1.0 if it < cool else (n - it) / (n - cool))


def modded_run() -> tuple[list[float], list[int]]:
    """modded's lr multiplier and batch (tokens) at every step of its run (schedule.py:67-75 get_lr, config.py:76-88):
    the stage's lr_mul and batch_size, the stage ends as TrainingSchedule.__init__ (schedule.py:28-38) places them, the
    lr linear over the last LR_COOLDOWN_FRAC of the main stages to MODDED_LR_FLOOR, held over the extension."""
    total = sum(MODDED_STAGE_DURATIONS)
    durations = [d / total for d in MODDED_STAGE_DURATIONS]
    durations[-1] -= MODDED_TAPER_DURATION
    ends = [round(c * MODDED_SCHEDULED_STEPS) for c in itertools.accumulate(durations + [MODDED_TAPER_DURATION])]
    ends[MODDED_GROWTH_STAGE:] = [e + MODDED_GROWTH_STEPS for e in ends[MODDED_GROWTH_STAGE:]]
    main = ends[-1]
    cool = int(main * (1 - LR_COOLDOWN_FRAC))
    multipliers, batches = [], []
    for step in range(MODDED_RUN_STEPS):
        stage = bisect.bisect_right(ends, step)
        lr = MODDED_STAGE_LR_MULS[stage]
        if step >= cool:
            t = min(1.0, (step - cool) / (main - cool))
            lr = lr * (1 - t) + MODDED_LR_FLOOR * t
        multipliers.append(lr)
        batches.append(MODDED_STAGE_BATCHES[stage])
    return multipliers, batches


def modded_scaling_params() -> int:
    """modded's model's scaling parameters as model.gpt.scaling_params counts ours: the attention and MLP matrices +
    lm_head (the padded vocabulary x width)."""
    attention = 4 * MODDED_WIDTH * MODDED_HEADS * MODDED_HEAD_DIM
    vocab = -(-MODDED_VOCAB // 128) * 128
    return MODDED_ATTN_LAYERS * attention + MODDED_MLPS * 2 * MODDED_MLP_HIDDEN * MODDED_WIDTH + vocab * MODDED_WIDTH


def flops_per_token(model) -> int:
    """Training FLOPs per token (forward + backward): 6 per weight of every linear map (karpathy/nanochat 92d63d4
    gpt.py:319-349), plus per layer attention's score and value matmuls, 12 x width x the mean number of keys a query
    attends (causal, within the layer's window, over a context of sequence_len tokens): DeepSeek LLM's 6 l_seq d_model
    per layer at full context (arXiv 2401.02954 §3.1)."""
    config = model.config
    t = config.sequence_len
    linear = sum(m.weight.numel() for m in model.modules() if isinstance(m, torch.nn.Linear))
    attention = sum(12 * config.n_embd * (w * (w + 1) / 2 + (t - w) * w) / t for w in layer_windows(config)[0])
    return round(6 * linear + attention)


def nanochat_lr_multiplier(it: int, n: int) -> float:
    """nanochat's lr multiplier at step it of an n-step run (92d63d4 scripts/base_train.py:360-369)."""
    warmdown = round(NANOCHAT_WARMDOWN_RATIO * n)
    if it < NANOCHAT_WARMUP_STEPS:
        return (it + 1) / NANOCHAT_WARMUP_STEPS
    if it <= n - warmdown:
        return 1.0
    progress = (n - it) / warmdown
    return progress + (1 - progress) * NANOCHAT_FINAL_LR_FRAC


def deepseek_lr_multiplier(it: int) -> float:
    """DeepSeek's step decay at step it of its run; the warmup linear (its shape is not stated)."""
    if it < DEEPSEEK_WARMUP_STEPS:
        return (it + 1) / DEEPSEEK_WARMUP_STEPS
    multiplier = 1.0
    for fraction, value in DEEPSEEK_STEP_DECAY:
        if it >= fraction * DEEPSEEK_STEPS:
            multiplier = value
    return multiplier


def anvil_rails(step: int, n: int) -> tuple[float, float, float]:
    """(momentum, fast_beta, fast_weight) of ANVIL at step `step` of an n-step run: modded's get_rail_beta +
    training.py:178-183 rail switch, its step counts scaled by n / 1194."""
    warmup = round(ANVIL_WARMUP_STEPS / MODDED_RUN_STEPS * n)
    cooldown_start = n - round(ANVIL_COOLDOWN_STEPS / MODDED_RUN_STEPS * n)
    if step < warmup:
        beta = ANVIL_BETA_MIN + step / warmup * (ANVIL_BETA_MAX - ANVIL_BETA_MIN)
    elif step > cooldown_start:
        beta = ANVIL_BETA_MAX - (step - cooldown_start) / (n - cooldown_start) * (ANVIL_BETA_MAX - ANVIL_BETA_MIN)
    else:
        beta = ANVIL_BETA_MAX
    if step >= round(ANVIL_ENGAGE_STEP / MODDED_RUN_STEPS * n):
        return beta, ANVIL_FAST_BETA, ANVIL_FAST_WEIGHT
    return beta, beta, 1.0


def stage_ends(n: int) -> list[int]:
    """The ends of modded's stages 0 and 1 in an n-step run (schedule.py:37-38: steps 320 and 681 of 1194, scaled)."""
    total = sum(MODDED_STAGE_DURATIONS)
    ends = [round(sum(MODDED_STAGE_DURATIONS[:k + 1]) / total * MODDED_SCHEDULED_STEPS) for k in range(len(MODDED_STAGE_DURATIONS) - 1)]
    return [round(end / MODDED_RUN_STEPS * n) for end in ends]


def loss_columns(objective: str) -> list[str]:
    """The names of the training loss's target columns (GPT.set_loss_targets): spec the next len(MTP_WEIGHTS) tokens,
    then the next token's prefix token; ntp the next token."""
    if objective == "spec":
        return ["next"] + [f"mtp_t{j + 1}" for j in range(1, len(MTP_WEIGHTS))] + ["prefix"]
    return ["next"]


def loss_weights(it: int, objective: str, ends: list[int]) -> list[float]:
    """Weights of the loss columns at step it: the next token's constant; MTP column j >= 1 keeps its weight until stage
    2 - j, decays linearly to 0 over that stage, then 0; the prefix column modded's stepped weights."""
    if objective != "spec":
        return [1.0]
    stage, start = len(ends), 0
    for k, end in enumerate(ends):
        if it < end:
            stage = k
            break
        start = end
    t = (it - start) / (ends[stage] - start) if stage < len(ends) else 0.0

    def decay(j):
        return 1.0 if stage < len(ends) - j else 1 - t if stage == len(ends) - j else 0.0

    steps = PREFIX_WEIGHTS[stage] if stage < len(ends) else (0.0,)
    return [MTP_WEIGHTS[0]] + [w * decay(j) for j, w in enumerate(MTP_WEIGHTS) if j > 0] + [steps[min(len(steps) - 1, int(t * len(steps)))]]


@dataclass(frozen=True)
class RunPlan:
    """The derived settings of one run (plan); the run record holds them."""
    objective: str
    total_batch_size: int    # tokens per optimizer step
    grad_accum_steps: int    # micro-batches per step
    num_iterations: int
    scaling_params: int      # the model's scaling parameters (model.gpt.scaling_params)
    nanochat_ref_steps: int  # steps of nanochat's d12 run (D_ref tokens at B_REF), its AdamW recipe's run
    engram_rows: int         # rows per Engram sub-table
    stage_ends: list[int]


def plan(depth: int, ratio: float, objective: str, vocab_size: int) -> RunPlan:
    """The run plan of the model of this depth at `ratio` training tokens per scaling parameter."""
    params = depth_scaling_params(depth, vocab_size)
    target_tokens = int(ratio * params)
    d_ref = NANOCHAT_REF_RATIO * depth_scaling_params(NANOCHAT_REF_DEPTH, vocab_size)
    total_batch_size = power_lines_batch(target_tokens, d_ref)
    num_iterations = target_tokens // total_batch_size
    return RunPlan(
        objective=objective, total_batch_size=total_batch_size,
        grad_accum_steps=total_batch_size // TOKENS_PER_MICRO_BATCH, num_iterations=num_iterations,
        scaling_params=params, nanochat_ref_steps=d_ref // B_REF, engram_rows=engram_rows(target_tokens),
        stage_ends=stage_ends(num_iterations),
    )
