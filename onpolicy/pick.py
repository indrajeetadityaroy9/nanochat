"""
The FTPO candidate to keep (docs/post-training.md §8.6), by a rule fixed before any candidate is evaluated. Each
candidate (onpolicy/train.py: <dir>/ftpo/ftpo-<threshold>/ and ftpo-final/) is compared with the export it was
trained from (<dir>/model), paired on identical inputs (scripts/onpolicy.sh `evaluate`, eval-<name>/):

- closure and loop share per probe set (evalplus, val) and over both (`paired`): the termination probe's responses
  (eval/official_termination.py, the same prompts and seeds for every model) matched by set and prompt id; closure a
  response's `closed` (natural termination), a loop a never-closed response with repetition >= LOOP; the difference
  candidate - export with the exact McNemar p over the discordant prompts (`mcnemar`);
- KodCode val pass@1 (`pass_at_1`): the held-out val rollouts (the same prompts and seeds), per prompt its passing
  share of the samples (a success: closed, ended, verified pass; onpolicy/selection.py `load`), the difference of the
  means with a paired bootstrap 95% interval over prompts (BOOTSTRAP resamples, default_rng(SEED));
- NLL drift: onpolicy/train.py `nll` of the candidate over that of the export on the export's own passing val
  responses, minus 1, in percent.

Rule: a candidate is eligible when its val pass@1 interval's upper bound is >= 0 and its NLL drift <= MAX_DRIFT %;
of the eligible, the highest combined closure is picked, a tie going to the lower threshold (less training). A
candidate with an evaluation missing is not eligible. Writes <dir>/pick.json {export, candidates: their statistics,
picked: the name or null, picked_dir, reason}.

    python -m onpolicy.pick <onpolicy dir>
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from eval.loop import LOOP
from onpolicy.selection import load

MAX_DRIFT = 10.0  # NLL drift, percent: FTPO's measured collateral was +3.7% to +59% (onpolicy/train.py LR)
BOOTSTRAP, SEED = 10000, 0
SETS = ("evalplus", "val")


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p from the discordant counts b and c: 2 P(X <= min(b, c)), X ~ Binomial(b + c, 1/2)."""
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def probe(path: Path) -> dict[tuple[str, str], dict]:
    report = json.loads(path.read_text())
    return {(s, str(r["id"])): r for s in SETS for r in report["responses"][s]}


def paired(a: list[bool], b: list[bool]) -> dict:
    """Export rate a, candidate rate b, b - a and the exact McNemar p."""
    gains = sum(y and not x for x, y in zip(a, b, strict=True))
    losses = sum(x and not y for x, y in zip(a, b, strict=True))
    return {"n": len(a), "export": sum(a) / len(a), "candidate": sum(b) / len(b), "diff": (sum(b) - sum(a)) / len(a),
            "gains": gains, "losses": losses, "p": mcnemar(gains, losses)}


def probe_stats(export: dict, candidate: dict) -> dict:
    if set(export) != set(candidate):
        raise ValueError("the probes hold different prompts")
    out = {}
    for name, keys in [(s, sorted(k for k in export if k[0] == s)) for s in SETS] + [("all", sorted(export))]:
        out[name] = {
            "closure": paired([export[k]["closed"] for k in keys], [candidate[k]["closed"] for k in keys]),
            "loop": paired([not export[k]["closed"] and export[k]["repetition"] >= LOOP for k in keys],
                           [not candidate[k]["closed"] and candidate[k]["repetition"] >= LOOP for k in keys])}
    return out


def pass_rates(rollouts: Path) -> dict[str, float]:
    _, rows = load([rollouts])
    per = {}
    for r in rows:
        per.setdefault(r["prompt_id"], []).append(bool(r["success"]))
    return {p: sum(v) / len(v) for p, v in per.items()}


def pass_at_1(export: dict[str, float], candidate: dict[str, float]) -> dict:
    if set(export) != set(candidate):
        raise ValueError("the val evaluations hold different prompts")
    keys = sorted(export)
    a, b = np.array([export[k] for k in keys]), np.array([candidate[k] for k in keys])
    d = b - a
    draws = np.random.default_rng(SEED).integers(0, len(d), size=(BOOTSTRAP, len(d)))
    lo, hi = np.percentile(d[draws].mean(1), [2.5, 97.5]).tolist()
    return {"prompts": len(keys), "export": float(a.mean()), "candidate": float(b.mean()), "diff": float(d.mean()),
            "ci95": [lo, hi]}


def order(name: str) -> float:
    """A candidate's threshold (ftpo-final last)."""
    tail = name.removeprefix("ftpo-")
    return math.inf if tail == "final" else float(tail)


def pick(directory: Path) -> dict:
    nll_export = json.loads((directory / "eval-model" / "nll.json").read_text())["nll"]
    export_probe = probe(directory / "eval-model" / "termination.json")
    export_val = pass_rates(directory / "eval-model" / "val")
    names = sorted((p.name for p in (directory / "ftpo").glob("ftpo-*") if (p / "model.safetensors").exists()),
                   key=order)
    stats, eligible = {}, []
    for name in names:
        ev = directory / f"eval-{name}"
        try:
            nll = json.loads((ev / "nll.json").read_text())["nll"]
            s = {"probe": probe_stats(export_probe, probe(ev / "termination.json")),
                 "val_pass@1": pass_at_1(export_val, pass_rates(ev / "val")),
                 "nll": nll, "nll_drift_percent": 100 * (nll / nll_export - 1)}
        except (FileNotFoundError, KeyError, ValueError) as e:
            stats[name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        s["eligible"] = s["val_pass@1"]["ci95"][1] >= 0 and s["nll_drift_percent"] <= MAX_DRIFT
        stats[name] = s
        if s["eligible"]:
            eligible.append(name)
    if eligible:
        picked = max(eligible, key=lambda n: (stats[n]["probe"]["all"]["closure"]["candidate"], -order(n)))
        reason = (f"highest combined closure among eligible {eligible}: "
                  f"{stats[picked]['probe']['all']['closure']['candidate']:.4f}")
    else:
        picked = None
        reason = f"no eligible candidate of {names}" if names else "no candidate"
    return {"rule": f"eligible: val pass@1 diff 95% CI upper >= 0 and NLL drift <= {MAX_DRIFT}%; pick the highest "
                    "combined closure; ties to the lower threshold",
            "export": {"nll": nll_export, "val_pass@1": float(np.mean(list(export_val.values())))},
            "candidates": stats, "picked": picked,
            "picked_dir": str(directory / "ftpo" / picked) if picked else None, "reason": reason}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="the on-policy stage's directory (RUNS/<run>/onpolicy-<step>)")
    args = parser.parse_args()
    result = pick(args.directory)
    (args.directory / "pick.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("picked", "reason")}), flush=True)
