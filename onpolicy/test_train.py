"""onpolicy/train.py's held-out monitor split and stochastic rounding, and its DPO loss and completion
log-probabilities, against hand-computed values. The stages themselves need a run's data (the guard reads
eval-model/val beside the output directory), so they are smoke-run on it, not here.

    python -m pytest onpolicy/test_train.py
"""
import math
from types import SimpleNamespace

import pytest
import torch

from onpolicy import train


def test_monitor_split_whole_prompts():
    examples = [{"prompt_id": f"p{i % 7}", "n": i} for i in range(30)]
    kept, held = train.monitor_split(examples, 10, 0)
    assert {e["prompt_id"] for e in kept}.isdisjoint({e["prompt_id"] for e in held})
    assert len(kept) + len(held) == 30 and len(held) >= 10
    assert train.monitor_split(examples, 10, 0) == (kept, held)  # deterministic


def test_stochastic_rounding_unbiased_and_exact_on_grid():
    g = torch.Generator().manual_seed(0)
    grid = torch.tensor([0.0277, -0.0049, 1.5]).to(torch.bfloat16).float()
    assert torch.equal(train.bf16(grid, g).float(), grid)
    shift = grid.abs() * 2.0**-10  # 1/8 to 1/4 of the bf16 spacing, which nearest rounding erases
    assert torch.equal((grid + shift).to(torch.bfloat16).float(), grid)
    mean = torch.stack([train.bf16(grid + shift, g).float() for _ in range(4000)]).mean(0)
    assert torch.allclose(mean - grid, shift, rtol=0.2)


def test_dpo_loss_eq7():
    # arXiv 2305.18290 Eq. 7 on summed completion log-probs: -log sigmoid(beta ((c - ref_c) - (r - ref_r)))
    c, r, ref_c, ref_r, beta = -40.0, -35.0, -44.0, -30.0, 0.05
    z = beta * ((c - ref_c) - (r - ref_r))  # 0.05 x (4 - (-5)) = 0.45
    loss, stats = train.dpo_loss(torch.tensor(c), torch.tensor(r), ref_c, ref_r, beta)
    assert float(loss) == pytest.approx(-math.log(1 / (1 + math.exp(-z))), rel=1e-6)
    assert stats["reward_chosen"] == pytest.approx(0.2) and stats["reward_rejected"] == pytest.approx(-0.25)
    assert stats["reward_margin"] == pytest.approx(0.45) and stats["reward_accuracy"] == 1.0
    # at the reference the loss is log 2
    assert float(train.dpo_loss(torch.tensor(c), torch.tensor(r), c, r, beta)[0]) == pytest.approx(math.log(2))


def test_dpo_gradient_raises_chosen_lowers_rejected():
    c, r = torch.tensor(-10.0, requires_grad=True), torch.tensor(-12.0, requires_grad=True)
    train.dpo_loss(c, r, -10.0, -12.0, 0.05)[0].backward()
    assert c.grad < 0 < r.grad


def test_completion_logp_capped_and_chunked(monkeypatch):
    torch.manual_seed(0)
    V, D, cap = 11, 4, 3.0
    embed, head = torch.randn(V, D), torch.randn(V, D, requires_grad=True)
    model = SimpleNamespace(
        model=lambda input_ids, use_cache: SimpleNamespace(last_hidden_state=embed[input_ids]),
        lm_head=SimpleNamespace(weight=head), config=SimpleNamespace(final_logit_softcapping=cap))
    prompt, completion = torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7, 8])
    monkeypatch.setattr(train, "HEAD_CHUNK", 2)
    got = train.completion_logp(model, prompt, completion)
    h = embed[torch.cat([prompt, completion])][2:-1]
    y = h @ head.T
    want = (cap * torch.tanh(y / cap)).log_softmax(-1).gather(-1, completion[:, None]).sum()
    assert float(got) == pytest.approx(float(want), rel=1e-6)
    (g,) = torch.autograd.grad(got, head)
    (w,) = torch.autograd.grad(want, head)
    assert torch.allclose(g, w, atol=1e-6)
