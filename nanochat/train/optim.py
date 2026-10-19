"""
The optimizer: ANVIL for the matrices, modded-nanogpt's Adam for the gates and MUDD, a row Adam for the Engram tables
(AnvilAdam), torch's AdamW for everything else; the per-group recipes and their transfer to a run (param_groups). ANVIL
is modded-nanogpt's matrix update (track_1_short/optim/anvil.py, schedule.py, training.py; record #92, ANVIL2, PR #360).

Every group comes from one recipe and its own run (train/schedule.py): nanochat's AdamW (its d12 run at B_REF tokens
per step), modded's ANVIL (lr at 131,072 tokens) and Adam (one update per 262,144 tokens), DeepSeek's Engram-27B. Betas
and eps are the recipe's own (nanochat 92d63d4 gpt.py:437-442 keeps them at every batch). The lr follows the recipe's
sqrt(B / recipe batch) rule; modded's lrs also its horizon, x (D / D_modded)^-0.32 (Bjorck et al., ICLR 2025, arXiv
2409.19913 Eq. 1: the optimal lr at a fixed batch over D training tokens); the Engram groups' DeepSeek LLM's
compute-optimal law (deepseek_lr). The weight decay sets the run's integrated decay by Power Lines (weight_decay).
"""

import inspect
import math

import torch
from torch import Tensor

from nanochat.train.schedule import (
    B_REF,
    DEEPSEEK_STEPS,
    NANOCHAT_REF_RATIO,
    NANOCHAT_WARMUP_STEPS,
    TAU_EXPONENT,
    deepseek_lr_multiplier,
    flops_per_token,
    lr_multiplier,
    modded_run,
    modded_scaling_params,
    nanochat_lr_multiplier,
)

# Engram (arXiv 2601.07372 v2): Table 5 "Base Learning Rate 4e-4", "Weight Decay 0.1", "Engram Lr Multipler x5", batch
# 1280 sequences x 4096 tokens; Table 1 "# Activated (w/o token embed) 3.8B", the parameters its run's tokens scale with
ENGRAM_BASE_LR, ENGRAM_WEIGHT_DECAY, ENGRAM_LR_MULTIPLIER = 4e-4, 0.1, 5
ENGRAM_REF_BATCH_TOKENS, DEEPSEEK_ACTIVE_PARAMS = 1280 * 4096, 3.8e9
# DeepSeek-V3's AdamW betas (arXiv 2412.19437 §4.2) for the Engram norms and convolution
DEEPSEEK_BETAS = (0.9, 0.95)
# DeepSeek LLM's compute-optimal hyperparameters (arXiv 2401.02954 §3.1 Eq. 1): eta_opt = 0.3118 C^-0.1250, B_opt =
# 0.2920 C^0.3271 tokens, at compute C = M D (M: non-embedding FLOPs per token, D: training tokens)
DEEPSEEK_LR_LAW, DEEPSEEK_BATCH_LAW = (0.3118, -0.1250), (0.2920, 0.3271)
# Row Adam beta2, eps: modded-nanogpt's hashed n-gram table (track_1_short/ngram_table.py:74 NGRAM_ADAM_BETA2 = 0.95,
# beta1 = 0 per its docstring:28; eps = its Adam default, track_1_short/training.py:111; @4ea6b93)
ENGRAM_BETA2, ENGRAM_EPS = 0.95, 1e-10
# Bjorck et al. (arXiv 2409.19913 Eq. 1): lr_opt ~ D^-0.32 at a fixed batch
HORIZON_LR_EXPONENT = 0.32


def deepseek_lr(model, run) -> float:
    """DeepSeek LLM's compute-optimal lr at this run's compute C = M D (M: the model's training FLOPs per token without
    the head's vocabulary GEMMs), carried from its optimal batch B_opt(C) to the run's batch B by sqrt(B / B_opt)."""
    tokens = run.num_iterations * run.total_batch_size
    compute = (flops_per_token(model) - 6 * model.lm_head.weight.numel()) * tokens
    lr_opt = DEEPSEEK_LR_LAW[0] * compute ** DEEPSEEK_LR_LAW[1]
    batch_opt = DEEPSEEK_BATCH_LAW[0] * compute ** DEEPSEEK_BATCH_LAW[1]
    return lr_opt * (run.total_batch_size / batch_opt) ** 0.5


def weight_decay(group: dict, n: int, recipe_lr: float, recipe_wd: float, recipe_run: tuple[list[float], float],
                 tpp: float) -> float:
    """The group's weight decay over an n-step run of tpp tokens per scaling parameter: the integrated decay sum_t wd
    lr_t^p (p = 1 for AdamW's decoupled lr wd, p = 2 for the lr^2 wd of ANVIL and modded's Adam) is the recipe run's x
    (tpp / its TPP)^0.527, Power Lines' law for the number of decay timescales in a run (Bergsma et al., arXiv
    2505.13738 Fig. 1: tau_opt / T ~ TPP^-0.527). recipe_run: the recipe's lr multiplier at each of its updates (its lr
    recipe_lr x them) and its TPP; the group's lr_t is its lr x schedule.lr_multiplier."""
    p = 1 if group["kind"] == "adamw" else 2
    multipliers, recipe_tpp = recipe_run
    decay = recipe_wd * math.fsum((recipe_lr * m) ** p for m in multipliers) * (tpp / recipe_tpp) ** TAU_EXPONENT
    return decay / math.fsum((group["lr"] * lr_multiplier(it, n, group["warmup"])) ** p for it in range(n))


# anvil.py:36-43 ANVIL_MAPS: (a, b, c) of X <- a*X + (b*A + c*A^2) X with A the Gram of X
ANVIL_MAPS = [
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
]
# anvil.py:52: the cascade input is divided by FROBENIUS_MARGIN * ||X||_F + FROBENIUS_EPS
ANVIL_FROBENIUS_MARGIN, ANVIL_FROBENIUS_EPS = 1.05, 1e-6
# anvil.py:48 RAIL_SLOW_BETA: the slow rail's beta (per step; the ANVIL betas are not batch-transferred)
ANVIL_SLOW_BETA = 0.98
# training.py:115-120 anvil_defaults: base learning rate, the equalizer's EMA decay (beta2), weight decay
ANVIL_LR, ANVIL_BETA2, ANVIL_WEIGHT_DECAY = 0.023, 0.9, 2.25
# training.py:76 (config.py TRAINING_STAGES[0]): the batch ANVIL_LR belongs to, 8 x 2048 x 8 tokens; modded scales its
# lr with the batch's square root (config.py:80)
ANVIL_REF_BATCH_TOKENS = 8 * 2048 * 8
# training.py:110-112 adam_defaults: modded's Adam lr, eps and weight decay. Modded steps Adam on odd steps only, on the
# gradient summed over the two steps (training.py:151-153; optim/anvil.py:679-681): MODDED_ADAM_LR belongs to
# MODDED_ADAM_PERIOD x ANVIL_REF_BATCH_TOKENS tokens per update.
MODDED_ADAM_LR, MODDED_ADAM_EPS, MODDED_ADAM_WEIGHT_DECAY, MODDED_ADAM_PERIOD = 0.008, 1e-10, 0.005, 2
# anvil.py:331-342: MLP c_proj at 2x the learning rate
ANVIL_MLP_C_PROJ_LR_MUL = 2.0
# modded model/training.py:73-76: MUDD's coefficient MLPs (mudd_w1, mudd_w2, mudd_b2) Adam lr_mul (b2: wd_mul 0)
MUDD_LR_MUL = 0.25
# nanochat's AdamW recipe (karpathy/nanochat 92d63d4 gpt.py:428-442; fork HEAD 773a0a9 scripts/base_train.py:60-64),
# tuned at width NANOCHAT_REF_WIDTH, its width-scaled lrs x 1/sqrt(width / NANOCHAT_REF_WIDTH); eps 1e-10 for every
# group
NANOCHAT_REF_WIDTH = 768
EMBEDDING_LR, UNEMBEDDING_LR, SCALAR_LR, SMEAR_LR, NANOCHAT_EPS = 0.3, 0.008, 0.5, 0.2, 1e-10


@torch.compile(dynamic=False, fullgraph=True)
def modded_adam_step_fused(p: Tensor, grad: Tensor, exp_avg: Tensor, exp_avg_sq: Tensor, beta1_t: Tensor,
                           beta2_t: Tensor, eps_t: Tensor, step_size_t: Tensor, eff_wd_t: Tensor) -> None:
    """modded-nanogpt's Adam (optim/anvil.py:700-731 _adam_update / _adam_update_step): the bias correction folded into
    the step size lr sqrt(1 - beta2^t) / (1 - beta1^t), eps added to sqrt(v), cautious weight decay lr^2 * weight_decay
    where the update and the parameter agree in sign."""
    exp_avg.mul_(beta1_t).add_(grad * (1 - beta1_t))
    exp_avg_sq.mul_(beta2_t).add_(grad.square() * (1 - beta2_t))
    update = exp_avg / (exp_avg_sq.sqrt() + eps_t) * step_size_t
    mask = (update * p) > 0
    p.sub_(update + p * mask * eff_wd_t)


@torch.compile(dynamic=False, fullgraph=True)
def engram_step_fused(p: Tensor, grad: Tensor, exp_avg_sq: Tensor, beta2_t: Tensor, eps_t: Tensor,
                      step_size_t: Tensor) -> None:
    """The Engram tables' row Adam (beta1 = 0, no weight decay): one second moment per row, the mean of the row's squared
    gradient; p -= g step_size / (sqrt(v) + eps), the bias correction folded into step_size = lr sqrt(1 - beta2^t)
    (modded's form, as the modded Adam above). Dense over every row: a row without gradient keeps its values and its
    moment decays by beta2, the same as a lazy row-sparse update that replays the missed decays."""
    exp_avg_sq.mul_(beta2_t).add_(grad.square().mean(dim=-1) * (1 - beta2_t))
    p.sub_(grad * (step_size_t / (exp_avg_sq.sqrt() + eps_t)).unsqueeze(-1))


def anvil_shape_lr_mul(rows: int, cols: int) -> float:
    """anvil.py:328: a (rows, cols) matrix's lr multiplier sqrt(max(1, rows / cols)), the matrix as modded's banks store
    it."""
    return max(1.0, rows / cols) ** 0.5


def param_groups(model, run) -> list[dict]:
    """The optimizer groups of a GPT for a run (schedule.RunPlan): nanochat's AdamW for the embeddings, lm_head, scalars
    and smear gate; DeepSeek's AdamW for the Engram norms and convolution; the row Adam for the Engram tables; modded's
    Adam for the attention/value-embedding gates and MUDD; ANVIL for the matrices; each recipe transferred to the run
    (module docstring). 'lr' is the peak; the trainer sets each step's lr (x schedule.lr_multiplier with the group's
    'warmup': nanochat's 40 steps on its AdamW groups (its recipe), none on the others) and the ANVIL rails (momentum,
    fast_beta, fast_weight)."""
    model_dim, n_head = model.config.n_embd, model.config.n_head
    blocks = model.transformer.h
    sites = (model.mudd_in, model.mudd_out)
    hidden = blocks[0].mlp.c_proj.weight.size(1)
    batch, n = run.total_batch_size, run.num_iterations
    tpp = n * batch / run.scaling_params
    # each recipe's run: its lr multiplier at each of its updates and its tokens per scaling parameter (modded's Adam
    # steps on odd steps only, on the gradient summed over the two steps: training.py:151-153, optim/anvil.py:679-681)
    modded, modded_batches = modded_run()
    modded_tpp = sum(modded_batches) / modded_scaling_params()
    nanochat_run = ([nanochat_lr_multiplier(it, run.nanochat_ref_steps) for it in range(run.nanochat_ref_steps)], NANOCHAT_REF_RATIO)
    anvil_run = (modded, modded_tpp)
    modded_adam_run = (modded[MODDED_ADAM_PERIOD - 1::MODDED_ADAM_PERIOD], modded_tpp)
    deepseek_run = ([deepseek_lr_multiplier(it) for it in range(DEEPSEEK_STEPS)], DEEPSEEK_STEPS * ENGRAM_REF_BATCH_TOKENS / DEEPSEEK_ACTIVE_PARAMS)
    # the lr multipliers of the recipes' lrs: nanochat's sqrt batch rule and width scale; modded's sqrt batch rule
    # (config.py:80) and horizon factor; DeepSeek LLM's law relative to Engram's base lr
    horizon = (n * batch / sum(modded_batches)) ** -HORIZON_LR_EXPONENT
    nanochat_scale = (batch / B_REF) ** 0.5
    width = (model_dim / NANOCHAT_REF_WIDTH) ** -0.5
    modded_adam_scale = (batch / (MODDED_ADAM_PERIOD * ANVIL_REF_BATCH_TOKENS)) ** 0.5 * horizon
    engram_scale = deepseek_lr(model, run) / ENGRAM_BASE_LR
    torch_eps = inspect.signature(torch.optim.AdamW).parameters["eps"].default
    modded_betas = (0.9, 0.99)
    # (name, kind, params, recipe lr, its multiplier, betas, eps, warmup steps, recipe run, recipe weight decay)
    adam_groups = [
        # nanochat 92d63d4 gpt.py:437-442 (nanochat leaves its smear lr out of its batch rule; here every group follows it)
        ('lm_head', 'adamw', list(model.lm_head.parameters()), UNEMBEDDING_LR, width * nanochat_scale, (0.8, 0.96), NANOCHAT_EPS, NANOCHAT_WARMUP_STEPS, nanochat_run, 0.01),
        ('wte', 'adamw', list(model.transformer.wte.parameters()), EMBEDDING_LR, width * nanochat_scale, (0.8, 0.995), NANOCHAT_EPS, NANOCHAT_WARMUP_STEPS, nanochat_run, 0.001),
        ('value_embeds', 'adamw', list(model.value_embeds.parameters()), EMBEDDING_LR * 0.5, width * nanochat_scale, (0.8, 0.995), NANOCHAT_EPS, NANOCHAT_WARMUP_STEPS, nanochat_run, 0.01),
        ('resid_lambdas', 'adamw', [model.resid_lambdas], SCALAR_LR * 0.01, nanochat_scale, (0.8, 0.95), NANOCHAT_EPS, NANOCHAT_WARMUP_STEPS, nanochat_run, 0.05),
        ('x0_lambdas', 'adamw', [model.x0_lambdas], SCALAR_LR, nanochat_scale, (0.96, 0.95), NANOCHAT_EPS, NANOCHAT_WARMUP_STEPS, nanochat_run, 0.0),
        ('gates', 'adamw', [model.smear_gate.weight, model.smear_lambda], SMEAR_LR, nanochat_scale, (0.8, 0.95), NANOCHAT_EPS, NANOCHAT_WARMUP_STEPS, nanochat_run, 0.0),
        # Engram RMSNorm gains (no weight decay: a gain's neutral value is 1) and convolution: DeepSeek's weight decay,
        # DeepSeek-V3's betas, torch's default eps
        ('engram_norms', 'adamw', [p for m in model.engram.values() for p in (m.norm_h, m.norm_k, m.norm_v)], ENGRAM_BASE_LR, engram_scale, DEEPSEEK_BETAS, torch_eps, 0, deepseek_run, 0.0),
        ('engram_conv', 'adamw', [m.conv_w for m in model.engram.values()], ENGRAM_BASE_LR, engram_scale, DEEPSEEK_BETAS, torch_eps, 0, deepseek_run, ENGRAM_WEIGHT_DECAY),
        # attention-output gates: modded model/training.py:63 gate recipe, Adam (0.9, 0.99), lr_mul 1, wd_mul 1
        ('attn_gates', 'modded_adam', [b.attn.attn_gate.weight for b in blocks], MODDED_ADAM_LR, modded_adam_scale, modded_betas, MODDED_ADAM_EPS, 0, modded_adam_run, MODDED_ADAM_WEIGHT_DECAY),
        # MUDD coefficient MLPs: modded model/training.py:73-76 (mudd_w1/w2 wd_mul 1, mudd_b2 wd_mul 0)
        ('mudd', 'modded_adam', [p for m in sites for p in (m.w1, m.w2)], MODDED_ADAM_LR * MUDD_LR_MUL, modded_adam_scale, modded_betas, MODDED_ADAM_EPS, 0, modded_adam_run, MODDED_ADAM_WEIGHT_DECAY),
        ('mudd_prior', 'modded_adam', [m.b for m in sites], MODDED_ADAM_LR * MUDD_LR_MUL, modded_adam_scale, modded_betas, MODDED_ADAM_EPS, 0, modded_adam_run, 0.0),
        # value-embedding gates: record #82 ve_gate_bank (R82:1725, 1747-1750)
        ('ve_gates', 'modded_adam', [b.attn.ve_gate.weight for b in blocks if b.attn.ve_gate is not None], MODDED_ADAM_LR, modded_adam_scale, modded_betas, MODDED_ADAM_EPS, 0, modded_adam_run, MODDED_ADAM_WEIGHT_DECAY),
    ]
    groups = []
    for name, kind, params, lr, scale, betas, eps, warmup, recipe_run, wd in adam_groups:
        group = dict(name=name, kind=kind, params=params, lr=lr * scale, betas=betas, eps=eps, warmup=warmup)
        group["weight_decay"] = weight_decay(group, n, lr, wd, recipe_run, tpp)
        groups.append(group)
    # Engram tables: the row Adam (engram_step_fused) at the Engram multiplier on the norms' lr, no weight decay (Table 5
    # "Engram Weight Decay 0.0"); beta2 and eps those of modded's table
    groups.append(dict(name='engram', kind='engram', params=[m.table for m in model.engram.values()],
                       lr=ENGRAM_LR_MULTIPLIER * ENGRAM_BASE_LR * engram_scale, betas=(0.0, ENGRAM_BETA2), eps=ENGRAM_EPS,
                       weight_decay=0.0, warmup=0))
    # ANVIL groups as modded's banks cut the matrices (model/gpt.py:235-252, 283-288): c_q/c_k one matrix per head
    # (record #80's per-head groups), c_v/attn.c_proj whole, c_fc as its [gate; up] matrices, mlp.c_proj at 2x lr with
    # the (hidden, C) shape multiplier, Engram w_kv as its [W_k; W_v] matrices; lr by modded's sqrt batch rule
    # (config.py:80) and horizon factor, beta2 and the rails per step
    for name, params, row_blocks, lr_mul in [
        ('anvil_qk', [w for b in blocks for w in (b.attn.c_q.weight, b.attn.c_k.weight)], n_head, anvil_shape_lr_mul(model_dim // n_head, model_dim)),
        ('anvil_vo', [w for b in blocks for w in (b.attn.c_v.weight, b.attn.c_proj.weight)], 1, anvil_shape_lr_mul(model_dim, model_dim)),
        ('anvil_mlp_fc', [b.mlp.c_fc.weight for b in blocks], 2, anvil_shape_lr_mul(hidden, model_dim)),
        ('anvil_mlp_proj', [b.mlp.c_proj.weight for b in blocks], 1, ANVIL_MLP_C_PROJ_LR_MUL * anvil_shape_lr_mul(hidden, model_dim)),
        ('anvil_engram_kv', [m.w_kv.weight for m in model.engram.values()], 2, anvil_shape_lr_mul(model_dim, model_dim // 2)),
    ]:
        group = dict(name=name, kind='anvil', params=params, lr=ANVIL_LR * (batch / ANVIL_REF_BATCH_TOKENS) ** 0.5 * horizon,
                     lr_mul=lr_mul, beta2=ANVIL_BETA2, row_blocks=row_blocks, warmup=0)
        group["weight_decay"] = weight_decay(group, n, ANVIL_LR, ANVIL_WEIGHT_DECAY, anvil_run, tpp)
        groups.append(group)
    return groups


def anvil_cascade(X: Tensor) -> Tensor:
    """bf16 (n, rows, cols) X -> Frobenius normalization and the six ANVIL maps (bf16), on the smaller Gram. As modded,
    b*A + c*A@A and a*X + X@B are single baddbmm calls (fp32 accumulation, one bf16 rounding each)."""
    X = (X.float() / (X.float().norm(dim=(-2, -1), keepdim=True) * ANVIL_FROBENIUS_MARGIN + ANVIL_FROBENIUS_EPS)).bfloat16()
    if X.size(-2) > X.size(-1): # Tall matrix
        for a, b, c in ANVIL_MAPS:
            A = X.mT @ X
            B = torch.baddbmm(A, A, A, beta=b, alpha=c)
            X = torch.baddbmm(X, X, B, beta=a)
    else: # Wide matrix
        for a, b, c in ANVIL_MAPS:
            A = X @ X.mT
            B = torch.baddbmm(A, A, A, beta=b, alpha=c)
            X = torch.baddbmm(X, B, X, beta=a)
    return X


def lane_equalize(g: Tensor, lane_energy: Tensor, beta2_t: Tensor, red_dim: int) -> Tensor:
    """anvil.py:771-784 _rail_equalizer: rescale each lane (row of a tall/square matrix, column of a wide one) by the
    inverse root of an EMA of its mean squared update, then the whole matrix back to its pre-equalization norm."""
    lane_power = g.square().mean(dim=red_dim, keepdim=True)
    lane_len = g.size(red_dim)
    pre_norm = (lane_power.sum(dim=(-2, -1), keepdim=True) * lane_len).sqrt()
    lane_energy.lerp_(lane_power, 1 - beta2_t)
    lane_gain = lane_energy.clamp_min(1e-10).rsqrt()
    post_norm = ((lane_power * lane_len) * lane_gain.square()).sum(dim=(-2, -1), keepdim=True).sqrt()
    return g * (lane_gain * (pre_norm / post_norm.clamp_min(1e-10)))


@torch.compile(dynamic=False, fullgraph=True)
def anvil_step_fused(stacked_grads: Tensor, stacked_params: Tensor, velocity: Tensor, lane_energy: Tensor,
                     momentum_t: Tensor, fast_beta_t: Tensor, fast_weight_t: Tensor, lr_t: Tensor, decay_t: Tensor,
                     beta2_t: Tensor, red_dim: int) -> None:
    """Fused ANVIL step on (n, rows, cols) stacks: twin-rail momentum (velocity[0] fast, velocity[1] slow) -> Nesterov
    lookahead -> ANVIL cascade -> lane equalization -> update lr_eff * g plus weight decay (weight_decay * lr * lr_eff)
    where the parameter and the slow rail agree in sign. The 0-D CPU tensors keep one compiled graph."""
    g = stacked_grads
    velocity[0].lerp_(g, 1 - fast_beta_t)
    velocity[1].lerp_(g, 1 - ANVIL_SLOW_BETA)
    blend = fast_weight_t * velocity[0] + (1 - fast_weight_t) * velocity[1]
    g = g.lerp(blend, momentum_t)
    g = anvil_cascade(g.bfloat16()).float()
    g = lane_equalize(g, lane_energy, beta2_t, red_dim)
    mask = (velocity[1] * stacked_params) >= 0
    stacked_params.sub_(lr_t * g + decay_t * stacked_params * mask)


class AnvilAdam(torch.optim.Optimizer):
    """ANVIL for the matrices, the row Adam for the Engram tables, modded's Adam for the gates and MUDD (the AdamW groups
    run on torch.optim.AdamW). Groups (param_groups): 'kind' 'modded_adam' ('lr', 'betas', 'eps', 'weight_decay') |
    'engram' ('lr', 'betas' (0, beta2), 'eps') | 'anvil' ('lr' the scheduled base lr, 'lr_mul', 'weight_decay', 'beta2',
    'row_blocks': each param stacks this many equal matrices along its rows, each updated as its own matrix; 'momentum',
    'fast_beta', 'fast_weight' set by the trainer). Parameters are fp32; their state follows their dtype."""

    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # one compiled graph per input signature (shapes, strides) of each fused step: Dynamo's recompile limit must
        # hold them all, or it falls back to eager
        self._recompile_limits = {
            kind: len({(p.shape, p.stride()) for g in self.param_groups if g['kind'] == kind for p in g['params']})
            for kind in ("modded_adam", "engram")
        } | {"anvil": len({self._matrices(g) for g in self.param_groups if g['kind'] == "anvil"})}
        # 0-D CPU tensors: changing values do not recompile
        self._lr_t, self._beta1_t, self._beta2_t, self._eps_t, self._wd_t, self._momentum_t, self._fast_beta_t, \
            self._fast_weight_t = (torch.tensor(0.0) for _ in range(8))

    def _step_modded_adam(self, group: dict) -> None:
        beta1, beta2 = group['betas']
        self._beta1_t.fill_(beta1)
        self._beta2_t.fill_(beta2)
        self._eps_t.fill_(group['eps'])
        self._wd_t.fill_(group['lr'] ** 2 * group['weight_decay'])  # anvil.py:704 eff_wd
        for p in group['params']:
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            t = state['step']
            self._lr_t.fill_(group['lr'] * (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t))  # anvil.py:703 step size
            modded_adam_step_fused(p, p.grad, state['exp_avg'], state['exp_avg_sq'], self._beta1_t, self._beta2_t,
                                   self._eps_t, self._lr_t, self._wd_t)

    def _step_engram(self, group: dict) -> None:
        beta2 = group['betas'][1]
        self._beta2_t.fill_(beta2)
        self._eps_t.fill_(group['eps'])
        for p in group['params']:
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg_sq'] = torch.zeros(p.size(0), device=p.device)
            state['step'] += 1
            self._lr_t.fill_(group['lr'] * (1 - beta2 ** state['step']) ** 0.5)
            engram_step_fused(p, p.grad, state['exp_avg_sq'], self._beta2_t, self._eps_t, self._lr_t)

    @staticmethod
    def _matrices(group: dict) -> tuple[int, int, int]:
        """(n, rows, cols) of an ANVIL group's update: each param's row_blocks row blocks, stacked"""
        p, row_blocks = group['params'][0], group['row_blocks']
        return len(group['params']) * row_blocks, p.size(-2) // row_blocks, p.size(-1)

    def _step_anvil(self, group: dict) -> None:
        params = group['params']
        n, rows, cols = self._matrices(group)
        # the equalizer's lanes: rows of a tall (or square) matrix, columns of a wide one (anvil.py:386-389, 427)
        red_dim = -1 if rows >= cols else -2
        state = self.state[params[0]]  # group-level state, kept on the first param
        if not state:
            state["velocity"] = torch.zeros(2, n, rows, cols, device=params[0].device)
            state["lane_energy"] = torch.zeros((n, rows, 1) if red_dim == -1 else (n, 1, cols), device=params[0].device)
        stacked_params = torch.stack(params)
        lr_eff = group["lr"] * group["lr_mul"]
        self._momentum_t.fill_(group["momentum"])
        self._fast_beta_t.fill_(group["fast_beta"])
        self._fast_weight_t.fill_(group["fast_weight"])
        self._lr_t.fill_(lr_eff)
        self._wd_t.fill_(group["weight_decay"] * group["lr"] * lr_eff)  # anvil.py:445,765 eff_wd * eff_lr
        self._beta2_t.fill_(group["beta2"])
        anvil_step_fused(torch.stack([p.grad for p in params]).view(n, rows, cols), stacked_params.view(n, rows, cols),
                         state["velocity"], state["lane_energy"], self._momentum_t, self._fast_beta_t,
                         self._fast_weight_t, self._lr_t, self._wd_t, self._beta2_t, red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for kind, step in (("modded_adam", self._step_modded_adam), ("engram", self._step_engram),
                           ("anvil", self._step_anvil)):
            with torch._dynamo.config.patch(recompile_limit=self._recompile_limits[kind]):
                for group in self.param_groups:
                    if group['kind'] == kind:
                        step(group)
