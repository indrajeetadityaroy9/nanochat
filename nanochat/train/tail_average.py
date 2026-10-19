"""
Tail weight averaging, modded-nanogpt's terminal ships (track_1_short/tail_average.py, ANVIL2 record 2026-08-30): the
run ends on blends of late-training fp32 weight EMAs instead of the raw final iterate, then each weight group is
rescaled to its pre-blend Frobenius norm (modded's "decontraction": averaging shrinks norms, and the model was trained
at the un-averaged scale).

  accumulator  nanochat weights (modded label)                        window  EMA span  written back as
  tail_ema     wte (embed), lm_head (lm_head)                          298     298       lerp(final, ema, 0.65)
  tail_avg     c_v + attn.c_proj (vo_bank), c_fc + mlp.c_proj           250     101.95    replaced by the ema
               (mlp_bank), value_embeds (value_embeds)
  bank_blend   c_q + c_k (qk_bank), vo, mlp                             298     298       lerp(current, ema, 0.55);
                                                                                          vo/mlp 0.55 - 0.2007

Windows and spans are modded's step counts as fractions of its 1194-step run (1122 scheduled + 52 growth + 20
extension; config.py, schedule.py total_steps, the run length TailAverage receives), so they scale with ours. Modded
ticks some accumulators only every 4th step or on odd steps (an optimizer-time saving on 8 GPUs); here every
accumulator ticks every step at the per-step rate of the same EMA, rate = 2 / (span + 1), with the span of a rate r
per p-step tick 2 / (1 - (1 - r)^(1/p)) - 1 (tick_span):
- tail_ema: per-step rate 2 / (298 + 1) (tail_ema_rate folds two of them into one odd-step tick): span 298.
- bank_blend: 2 / (298 // 4 + 1) per 4-step tick (BANK_BLEND_RATE, BANK_BLEND_PERIOD): span 295.97.
- tail_avg / value_embeds: tuned rate 4/53 per 4-step tick (TAIL_AVG_RATE, TAIL_AVG_PERIOD): span 101.95.
bank_blend is written back last, so on vo/mlp it blends into the tail average; its 0.2007 is modded's Richardson
correction for the lag of a seeded EMA ((1-B)A + BX + w(A - X) = lerp(A, X, B - w)). Norms are restored per modded
label (one Frobenius norm over all of a group's tensors) for every group except value_embeds, as in modded.
"""

import torch

from nanochat.train.schedule import MODDED_RUN_STEPS


def tick_span(rate: float, period: int) -> float:
    """The per-step EMA span of an EMA ticked at `rate` every `period` steps (the same decay per period)."""
    return 2 / (1 - (1 - rate) ** (1 / period)) - 1


# tail_average.py:30-47 (TAIL_EMA_*, TAIL_AVG_*, VALUE_EMBED_AVG_WINDOW, BANK_BLEND_*, RICHARDSON_W)
TAIL_EMA_BLEND = 0.65
BANK_BLEND = 0.55
RICHARDSON_W = 0.2007
TICK_PERIOD = 4  # TAIL_AVG_PERIOD = BANK_BLEND_PERIOD
TAIL_AVG_SPAN = tick_span(4 / 53, TICK_PERIOD)
BANK_BLEND_SPAN = tick_span(2 / (298 // TICK_PERIOD + 1), TICK_PERIOD)
# (name, window steps, EMA span steps in the reference run, {weight group: blend, None = replace}); ship order
ACCUMULATORS = [
    ("tail_ema", 298, 298, {"embed": TAIL_EMA_BLEND, "lm_head": TAIL_EMA_BLEND}),
    ("tail_avg", 250, TAIL_AVG_SPAN, {"vo": None, "mlp": None, "value_embeds": None}),
    ("bank_blend", 298, BANK_BLEND_SPAN, {"qk": BANK_BLEND, "vo": BANK_BLEND - RICHARDSON_W, "mlp": BANK_BLEND - RICHARDSON_W}),
]
NORM_RESTORED = ("embed", "lm_head", "qk", "vo", "mlp")


class TailAverage:
    """update(step) after optimizer step `step` (0-based) of a num_steps-step run; finalize() once after the last step
    writes the blends into the model's (fp32) weights."""

    def __init__(self, model, num_steps: int):
        blocks = model.transformer.h
        self.groups = {
            "embed": [model.transformer.wte.weight],
            "lm_head": [model.lm_head.weight],
            "qk": [w for b in blocks for w in (b.attn.c_q.weight, b.attn.c_k.weight)],
            "vo": [w for b in blocks for w in (b.attn.c_v.weight, b.attn.c_proj.weight)],
            "mlp": [w for b in blocks for w in (b.mlp.c_fc.weight, b.mlp.c_proj.weight)],
            "value_embeds": [ve.weight for ve in model.value_embeds.values()],
        }
        scale = num_steps / MODDED_RUN_STEPS
        self.accumulators = []
        for name, window, span, blends in ACCUMULATORS:
            params = [p for g in blends for p in self.groups[g]]
            self.accumulators.append(dict(name=name, start=num_steps - round(window * scale), rate=2 / (span * scale + 1),
                                          blends=blends, params=params, bufs=[torch.empty_like(p) for p in params]))

    @torch.no_grad()
    def update(self, step: int) -> None:
        for acc in self.accumulators:
            if step == acc["start"]:  # the EMA starts at its window's first weights
                torch._foreach_copy_(acc["bufs"], acc["params"])
            elif step > acc["start"]:
                torch._foreach_lerp_(acc["bufs"], acc["params"], acc["rate"])

    def _norm(self, group: str) -> float:
        return torch.stack([p.norm() for p in self.groups[group]]).norm().item()

    @torch.no_grad()
    def finalize(self) -> None:
        norms_before = {g: self._norm(g) for g in NORM_RESTORED}
        for acc in self.accumulators:
            bufs = iter(acc["bufs"])
            for group, blend in acc["blends"].items():
                for p in self.groups[group]:
                    avg = next(bufs)
                    p.copy_(avg if blend is None else torch.lerp(p, avg, blend))
        for g in NORM_RESTORED:  # disjoint groups: each rescaled to its pre-blend norm
            torch._foreach_mul_(self.groups[g], norms_before[g] / self._norm(g))

    def state_dict(self) -> dict:
        return {acc["name"]: acc["bufs"] for acc in self.accumulators}

    def load_state_dict(self, state: dict) -> None:
        for acc in self.accumulators:
            torch._foreach_copy_(acc["bufs"], state[acc["name"]])
