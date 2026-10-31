"""onpolicy/selection.py's DPO pairs on synthetic rollouts.

    python -m pytest onpolicy/test_selection.py
"""
from onpolicy import selection

BOS = 1


def row(pid: str, sample: int, ids: list[int], success, loop: tuple[int, int] | None = None) -> dict:
    return {"prompt_id": pid, "sample": sample, "ids": ids, "tokens": len(ids), "success": success,
            "loop": loop is not None, "loop_onset": loop and loop[0], "loop_period": loop and loop[1]}


def test_dpo_pairs():
    prompts = {p: {"ids": [BOS, 7, 8]} for p in ("a", "b", "c", "d")}
    cycle = [20, 21, 22, 23] * 4  # onset 2, period 4 after [10, 11]
    looping = [10, 11, *cycle]
    rows = [
        # a: two successes (the shorter, sample 1, is chosen; its BOS kept), three exact loops (two pairs, sample
        # order), a loop without an exact cycle (skipped)
        row("a", 0, [30, 31, 32, 33, BOS], True), row("a", 1, [40, 41], True),
        row("a", 4, looping, False, (2, 4)), row("a", 2, looping, False, (2, 4)), row("a", 3, looping, False, (2, 4)),
        {**row("a", 5, looping, False), "loop": True},
        # b: loops, never solved: no pair
        row("b", 0, looping, False, (2, 4)),
        # c: unverified (success None): no pair
        row("c", 0, [40, BOS], None), row("c", 1, looping, None, (2, 4)),
        # d: a pair longer than max_length: dropped
        row("d", 0, [50] * 20, True), row("d", 1, looping, False, (2, 4)),
    ]
    pairs, too_long = selection.dpo_pairs(prompts, rows, max_length=20, bos=BOS)
    assert too_long == 1
    assert [(p["prompt_id"], p["chosen_sample"], p["rejected_sample"]) for p in pairs] == [("a", 1, 2), ("a", 1, 3)]
    assert pairs[0]["prompt_ids"] == [BOS, 7, 8] and pairs[0]["chosen_ids"] == [40, 41, BOS]
    assert pairs[0]["rejected_ids"] == looping[:2 + 4 + 4]  # through the first repeat, no BOS


def test_rejected_side_window_cap():
    r = {"ids": list(range(2000)), "loop_onset": 100, "loop_period": 600}
    assert selection.rejected_side(r) == list(range(100 + 600 + selection.WINDOW_CAP))
