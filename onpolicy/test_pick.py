"""onpolicy/pick.py's paired statistics and pre-registered rule on synthetic evaluations.

    python -m pytest onpolicy/test_pick.py
"""
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from onpolicy import pick


def evaluation(directory, closed: list[bool], passes: list[int], nll: float) -> None:
    """eval-<name>/: a probe (half evalplus, half val; a never-closed response loops), 4 val samples per prompt with
    passes[i] of them passing, and nll.json."""
    directory.mkdir(parents=True)
    half = len(closed) // 2
    rows = [{"id": str(i), "closed": c, "repetition": 0.0 if c else 1.0} for i, c in enumerate(closed)]
    (directory / "termination.json").write_text(json.dumps({"responses": {"evalplus": rows[:half],
                                                                           "val": rows[half:]}}))
    val = directory / "val"
    val.mkdir()
    (val / "prompts.jsonl").write_text("".join(json.dumps({"id": f"q{i}", "ids": [0]}) + "\n"
                                               for i in range(len(passes))))
    records, verdicts = [], []
    for i, n in enumerate(passes):
        for s in range(4):
            records.append({"prompt_id": f"q{i}", "sample": s, "closed": True, "ended": True})
            verdicts.append({"prompt_id": f"q{i}", "sample": s, "verdict": "pass" if s < n else "fail"})
    pq.write_table(pa.Table.from_pylist(records), val / "rollouts.parquet")
    (val / "verdicts.jsonl").write_text("".join(json.dumps(v) + "\n" for v in verdicts))
    (directory / "nll.json").write_text(json.dumps({"nll": nll}))


def candidate(root, name: str) -> None:
    (root / "ftpo" / name).mkdir(parents=True)
    (root / "ftpo" / name / "model.safetensors").write_bytes(b"")


def test_mcnemar_exact():
    assert pick.mcnemar(0, 0) == 1.0
    assert pick.mcnemar(0, 5) == pytest.approx(2 / 32)
    assert pick.mcnemar(26, 10) == pytest.approx(0.011, abs=5e-4)  # docs/results.md: step 1,000 vs 2,000


def test_rule(tmp_path):
    export_closed = [True] * 10 + [False] * 10
    passes = [2] * 40
    evaluation(tmp_path / "eval-model", export_closed, passes, 1.0)
    # ftpo-0.15: closes 14, pass@1 unchanged, NLL +5%: eligible
    candidate(tmp_path, "ftpo-0.15")
    evaluation(tmp_path / "eval-ftpo-0.15", [True] * 14 + [False] * 6, passes, 1.05)
    # ftpo-0.25: closes 14 too, eligible: the tie goes to 0.15
    candidate(tmp_path, "ftpo-0.25")
    evaluation(tmp_path / "eval-ftpo-0.25", [True] * 14 + [False] * 6, passes, 1.08)
    # ftpo-0.35: closes 18 but NLL +12%: not eligible
    candidate(tmp_path, "ftpo-0.35")
    evaluation(tmp_path / "eval-ftpo-0.35", [True] * 18 + [False] * 2, passes, 1.12)
    out = pick.pick(tmp_path)
    assert [n for n, s in out["candidates"].items() if s["eligible"]] == ["ftpo-0.15", "ftpo-0.25"]
    assert out["picked"] == "ftpo-0.15"
    s = out["candidates"]["ftpo-0.15"]
    assert s["probe"]["all"]["closure"]["diff"] == pytest.approx(0.2)
    assert s["probe"]["all"]["loop"]["diff"] == pytest.approx(-0.2)
    assert s["probe"]["all"]["closure"]["gains"] == 4 and s["probe"]["all"]["closure"]["losses"] == 0
    assert s["val_pass@1"]["diff"] == 0 and s["val_pass@1"]["ci95"] == [0, 0]
    assert s["nll_drift_percent"] == pytest.approx(5.0)


def test_rule_pass_drop_and_none(tmp_path):
    evaluation(tmp_path / "eval-model", [True] * 10 + [False] * 10, [3] * 40, 1.0)
    # every val prompt loses a pass: the interval lies below 0, not eligible
    candidate(tmp_path, "ftpo-0.15")
    evaluation(tmp_path / "eval-ftpo-0.15", [True] * 20, [2] * 40, 1.0)
    candidate(tmp_path, "ftpo-final")  # never evaluated
    out = pick.pick(tmp_path)
    assert out["candidates"]["ftpo-0.15"]["val_pass@1"]["ci95"][1] < 0
    assert "error" in out["candidates"]["ftpo-final"]
    assert out["picked"] is None and out["picked_dir"] is None
