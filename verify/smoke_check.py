"""The smoke's metrics record (<base_dir>/metrics/<tag>.jsonl), split into launches at each "run" record: the stage
switches, the losses of the steps both launches took (the relaunch resumes at the switch checkpoint), and per launch
and stage the median tok/s. Usage: python -m verify.smoke_check TAG"""
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
    switch = launch["switch"][0][0] if launch["switch"] else None
    for name, sel in (("file", [s for s in steps if switch is None or s < switch]),
                      ("repo", [s for s in steps if switch is not None and s >= switch])):
        sel = sel[1:]  # the launch's first step of a stage includes its compilation
        if sel:
            print(f"  {name}: median tok/s {statistics.median(launch['train'][s]['tok_per_s'] for s in sel):,.0f}")
common = sorted(set(launches[0]["train"]) & set(launches[1]["train"]))
print("steps in both launches (loss first, relaunch, rel diff):")
for s in common:
    a, b = launches[0]["train"][s]["loss"], launches[1]["train"][s]["loss"]
    print(f"  {s}: {a:.6f} {b:.6f} {abs(a - b) / a:.1e}")
last = launches[-1]["train"][max(launches[-1]["train"])]
print(f"final step {last['step']}: loss {last['loss']:.4f}")
