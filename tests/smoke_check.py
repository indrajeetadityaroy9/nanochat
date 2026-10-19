"""The smoke's metrics record (<base_dir>/metrics/<tag>.jsonl), split into launches at each "run" record: the stage
switches, per launch and context the median tok/s, and the losses of the steps both launches took. The relaunch must
resume at the switch checkpoint's step and train to the end: exits 1 otherwise. Usage: python -m tests.smoke_check TAG
SWITCH STEPS"""
import json
import os
import statistics
import sys

from nanochat.paths import base_dir

switch, steps = int(sys.argv[2]), int(sys.argv[3])
launches = []
with open(os.path.join(base_dir(), "metrics", f"{sys.argv[1]}.jsonl"), encoding="utf-8") as f:
    for r in map(json.loads, f):
        if r["kind"] == "run":
            launches.append({"start": r["step"], "stage": r["stage"]["seq_len"], "train": {}, "switch": []})
        elif r["kind"] == "stage":
            launches[-1]["switch"].append((r["step"], r["stage"]["seq_len"]))
        elif r["kind"] == "train":
            launches[-1]["train"][r["step"]] = r
for i, launch in enumerate(launches):
    trained = sorted(launch["train"])
    print(f"launch {i}: from step {launch['start']} at seq_len {launch['stage']}, steps {trained[0]}..{trained[-1]}, "
          f"switches {launch['switch']}")
    # the launch's contexts: its starting one, then each switch's from the switch's step on
    bounds = [(launch["start"], launch["stage"]), *launch["switch"]]
    for (first, seq_len), (end, _) in zip(bounds, [*bounds[1:], (trained[-1] + 1, None)], strict=True):
        sel = [s for s in trained if first <= s < end][1:]  # the launch's first step at a context includes compilation
        if sel:
            print(f"  seq_len {seq_len}: median tok/s {statistics.median(launch['train'][s]['tok_per_s'] for s in sel):,.0f}")
common = sorted(set(launches[0]["train"]) & set(launches[1]["train"]))
print("steps in both launches (loss first, relaunch): " + ", ".join(
    f"{s}: {launches[0]['train'][s]['loss']:.6f} {launches[1]['train'][s]['loss']:.6f}" for s in common))
relaunch = sorted(launches[1]["train"])
sys.exit(0 if relaunch[0] == switch and relaunch[-1] == steps - 1 else 1)
