"""onpolicy/train.py's FTPO loss, held-out monitor split, chosen-token rule and stochastic rounding against
hand-computed values.

    python -m pytest onpolicy/test_train.py
"""
import math

import pytest
import torch

from onpolicy import train


def softplus(x: float) -> float:
    return math.log1p(math.exp(x))


def test_loss_by_hand():
    # V = 5, rejected 0, chosen {2, 3}; MARGIN 2, TAU_TARGET 0.5, LAMBDA_TARGET 0.05, LAMBDA_NONTARGET 0.4
    y = torch.tensor([2.0, 0.0, 1.0, 3.0, -0.5])
    ref = torch.tensor([1.5, 0.0, 0.0, 0.5, -1.0])
    loss, stats = train.ftpo_loss(y, ref, torch.tensor([2, 3]), 0)
    # Delta = -1 (w 1) and 1 (w 0.5), k = 2: (softplus(3) + 0.5 softplus(1)) / 2
    pref = (softplus(3.0) + 0.5 * softplus(1.0)) / 2
    # |d| over T = {0, 2, 3}: 0.5, 1, 2.5 -> excess 0, 0.5, 2 -> (0 + 0.25 + 4) / 3
    target = 4.25 / 3
    # d over N = {1, 4}: 0, 0.5 -> 0.25 / 2
    nontarget = 0.125
    assert stats["pref"] == pytest.approx(pref, rel=1e-6)
    assert stats["target"] == pytest.approx(target, rel=1e-6)
    assert stats["nontarget"] == pytest.approx(nontarget, rel=1e-6)
    assert float(loss) == pytest.approx(pref + 0.05 * target + 0.4 * nontarget, rel=1e-6)
    assert stats["chosen_win"] == 0.5 and stats["margin_win"] == 0.0 and stats["delta"] == pytest.approx(0.0)


def test_loss_off_past_margin():
    y = torch.tensor([0.0, 2.5, 3.0, 1.0])
    loss, stats = train.ftpo_loss(y, y.clone(), torch.tensor([1, 2]), 0)
    assert stats["pref"] == 0.0 and float(loss) == 0.0 and stats["margin_win"] == 1.0


def test_pref_official_three_rows():
    # reference_ftpo_trainer.py:282-283 on a batch of 3 rows: ((softplus(m - Delta) w).sum(-1) / k).mean()
    # row a: V 4, rejected 0 (y 1), chosen {1, 2} (y 0, 2.5): Delta -1 (w 1), 1.5 (w 0.25)
    # row b: V 4, rejected 3 (y 2), chosen {0} (y -1): Delta -3 (w clamp 2.5 -> 1)
    # row c: V 4, rejected 1 (y 0), chosen {0, 2, 3} (y 3, 0.5, 1): Delta 3 (w 0), 0.5 (w 0.75), 1 (w 0.5)
    rows = [(torch.tensor([1.0, 0.0, 2.5, 0.0]), [1, 2], 0), (torch.tensor([-1.0, 0.0, 0.0, 2.0]), [0], 3),
            (torch.tensor([3.0, 0.0, 0.5, 1.0]), [0, 2, 3], 1)]
    hand = [(softplus(3.0) + 0.25 * softplus(0.5)) / 2, softplus(5.0) / 1,
            (0 * softplus(-1.0) + 0.75 * softplus(1.5) + 0.5 * softplus(1.0)) / 3]
    ours = [train.ftpo_loss(y, y.clone(), torch.tensor(c), r)[1]["pref"] for y, c, r in rows]
    assert ours == pytest.approx(hand, rel=1e-6)
    assert sum(ours) / 3 == pytest.approx(sum(hand) / 3, rel=1e-6)
    # the official code's batched lines, chosen ids padded with a mask
    logits = torch.stack([y for y, _, _ in rows])
    ch_ids = torch.tensor([[1, 2, 0], [0, 0, 0], [0, 2, 3]])
    ch_mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=torch.float)
    rejected = torch.tensor([0, 3, 1])
    delta = logits[torch.arange(3)[:, None], ch_ids] - logits.gather(-1, rejected[:, None])
    weights = torch.clamp((2.0 - delta) / 2.0, 0.0, 1.0) * ch_mask
    official = ((torch.nn.functional.softplus(2.0 - delta) * weights).sum(-1) / ch_mask.sum(-1).clamp(min=1)).mean()
    assert float(official) == pytest.approx(sum(hand) / 3, rel=1e-6)


def test_monitor_split_whole_prompts():
    examples = [{"prompt_id": f"p{i % 7}", "n": i} for i in range(30)]
    kept, held = train.monitor_split(examples, 10, 0)
    assert {e["prompt_id"] for e in kept}.isdisjoint({e["prompt_id"] for e in held})
    assert len(kept) + len(held) == 30 and len(held) >= 10
    assert train.monitor_split(examples, 10, 0) == (kept, held)  # deterministic


def test_loss_gradient_flows():
    y = torch.tensor([2.0, 0.0, 1.0], requires_grad=True)
    loss, _ = train.ftpo_loss(y, y.detach().clone(), torch.tensor([2]), 0)
    loss.backward()
    assert y.grad[2] < 0 < y.grad[0]  # the chosen logit is pushed up, the rejected one down


def test_candidates():
    surfaces = {0: " the", 1: " The", 2: "\n", 3: " x", 4: " y", 5: " z"}
    ref = torch.full((train.TOP_LOGPROBS + 6,), -50.0)
    ref[:6] = torch.tensor([6.0, 5.5, 5.4, 5.0, 2.0, 3.0])
    # rejected 0; " The" is its lower-case surface; at T 0.6 the floor is the best remaining logit ("\n", 5.4)
    # + 0.6 ln 0.01 = 2.637: " y" (2.0) is out, " z" (3.0) in; "\n" strips to nothing; least likely first
    decode = lambda i: surfaces.get(i, f"<{i}>")
    assert train.candidates(ref, 0, 0.6, decode) == [5, 3]


def test_regularise_dedupes_and_keeps_small_sets():
    surfaces = {1: " Foo", 2: "foo,", 3: " bar"}
    decode = surfaces.__getitem__
    assert train.regularise([[1, 2, 3], [3]], decode) == [[1, 3], [3]]


def test_regularise_prunes_over_represented():
    # 62 keys: "a" in 200 rows (100 with "b"), "b" in 100, 60 keys once; the 95th-percentile count is 1, so ref = 50;
    # targets round(200 (50/200)^0.5) = 100 and round(100 (50/100)^0.5) = 71. "a" leaves the 100 rows it shares with
    # "b"; "b" can then only empty rows holding it alone: 29 of them, lowest index first
    decode = {0: "a", 1: "b", **{10 + k: f"k{k}" for k in range(60)}}.__getitem__
    rows = [[0, 1] for _ in range(100)] + [[0] for _ in range(100)] + [[10 + k] for k in range(60)]
    out = train.regularise(rows, decode)
    assert out[:29] == [[]] * 29 and out[29:100] == [[1]] * 71 and out[100:200] == [[0]] * 100


def test_stochastic_rounding_unbiased_and_exact_on_grid():
    g = torch.Generator().manual_seed(0)
    grid = torch.tensor([0.0277, -0.0049, 1.5]).to(torch.bfloat16).float()
    assert torch.equal(train.bf16(grid, g).float(), grid)
    shift = grid.abs() * 2.0**-10  # 1/8 to 1/4 of the bf16 spacing, which nearest rounding erases
    assert torch.equal((grid + shift).to(torch.bfloat16).float(), grid)
    mean = torch.stack([train.bf16(grid + shift, g).float() for _ in range(4000)]).mean(0)
    assert torch.allclose(mean - grid, shift, rtol=0.2)
