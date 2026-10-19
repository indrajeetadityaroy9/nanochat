"""
The results of one or more base_eval runs (<base_dir>/base_eval/<tag>.json, scripts/base_eval.py) side by side as
Markdown tables, written to --out: bits per byte of each mixture corpus's val split and their mean, reference-solution
bits per byte, and pass@1 of the code-generation and infilling benchmarks. results/ holds the tables the paper reports,
each written by the run script that produced it.

    python -m scripts.report --out=<file.md> <tag> [<tag> ...]
"""
import argparse
import json
import os
import statistics

from nanochat.paths import base_dir


def table(title, tags, rows, fmt):
    """A Markdown table: one row per (label, {tag: value}), one column per tag."""
    lines = [f"### {title}", "", "| | " + " | ".join(f"`{t}`" for t in tags) + " |", "|---" * (len(tags) + 1) + "|"]
    lines += [f"| {label} | " + " | ".join(format(values[t], fmt) for t in tags) + " |" for label, values in rows]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("tags", nargs="+")
    args = parser.parse_args()
    tags = args.tags
    results = {}
    for tag in tags:
        with open(os.path.join(base_dir(), "base_eval", f"{tag}.json"), encoding="utf-8") as f:
            results[tag] = json.load(f)
    first = results[tags[0]]
    bpb = [(c, {t: results[t]["bpb"][c] for t in tags}) for c in first["bpb"]]
    bpb.append(("mean", {t: statistics.fmean(results[t]["bpb"].values()) for t in tags}))
    solution = [(b, {t: results[t]["solution_bpb"][b] for t in tags}) for b in first["solution_bpb"]]
    passes = []
    for bench, r in first["codegen"].items():
        for key in [k for k in r if k.endswith("pass@1")]:
            passes.append((f"{bench} {key}", {t: results[t]["codegen"][bench][key] for t in tags}))
    passes += [(f"{name} pass@1", {t: results[t]["fim"][name]["pass@1"] for t in tags}) for name in first["fim"]]
    runs = "Runs (steps): " + ", ".join(f"`{t}` ({results[t]['step']})" for t in tags) + "\n"
    tables = [table("Validation bits per byte", tags, bpb, ".5f"),
              table("Reference-solution bits per byte", tags, solution, ".5f"),
              table("pass@1 (greedy)", tags, passes, ".4f")]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join([runs, *tables]))


if __name__ == "__main__":
    main()
