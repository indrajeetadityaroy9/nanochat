"""The smoke's metrics record (<base_dir>/metrics/<tag>.jsonl), split into launches at each "run" record: the stage
switches, per launch and context the median tok/s, and the losses of the steps both launches took (the relaunch
resumes at the switch checkpoint). Training is bitwise deterministic (model/gpt.py), so those losses must be equal:
exits 1 otherwise. Usage: python -m tests.smoke_check TAG"""
import json
import os
import statistics
import sys

from nanochat.paths import base_dir

launches = []
for line in open(os.path.join(base_dir(), "metrics", f"{sys.argv[1]}.jsonl")):
    r = json.loads(line)
    if r["kind"] == "run":
        launches.append({"start": r["step"], "stage": r["stage"]["seq_len"], "train": {}, "switch": []})
    elif r["kind"] == "stage":
        launches[-1]["switch"].append((r["step"], r["stage"]["seq_len"], r["stage"]["rope_base"]))
    elif r["kind"] == "train":
        launches[-1]["train"][r["step"]] = r
for i, launch in enumerate(launches):
    steps = sorted(launch["train"])
    print(f"launch {i}: from step {launch['start']} at seq_len {launch['stage']}, steps {steps[0]}..{steps[-1]}, "
          f"switches {launch['switch']}")
    # the launch's contexts: its starting one, then each switch's from the switch's step on
    bounds = [(launch["start"], launch["stage"])] + [(s, seq_len) for s, seq_len, _ in launch["switch"]]
    for (first, seq_len), (end, _) in zip(bounds, bounds[1:] + [(steps[-1] + 1, None)], strict=True):
        sel = [s for s in steps if first <= s < end][1:]  # the launch's first step at a context includes compilation
        if sel:
            tok_per_s = statistics.median(launch["train"][s]["tok_per_s"] for s in sel)
            print(f"  seq_len {seq_len}: median tok/s {tok_per_s:,.0f}")
common = sorted(set(launches[0]["train"]) & set(launches[1]["train"]))
print("steps in both launches (loss first, relaunch, rel diff):")
differ = 0
for s in common:
    a, b = launches[0]["train"][s]["loss"], launches[1]["train"][s]["loss"]
    differ += a != b
    print(f"  {s}: {a:.6f} {b:.6f} {abs(a - b) / a:.1e}")
last = launches[-1]["train"][max(launches[-1]["train"])]
print(f"final step {last['step']}: loss {last['loss']:.4f}; steps of both launches with different losses "
      f"{differ}/{len(common)}")
sys.exit(1 if differ or not common else 0)
