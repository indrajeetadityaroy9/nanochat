"""
The results of one or more base_eval runs (<base_dir>/base_eval/<tag>.json, scripts/base_eval.py) side by side as
Markdown tables (pandas to_markdown), written to --out: bits per byte of each mixture corpus's val split and the
records' mixture bits per byte per group (eval/loss_eval.py mixture_bpb), of each repository corpus at 32K per context
bucket, of the reference solutions, and pass@1 of the code-generation and infilling benchmarks. results/ holds the
tables the paper reports, each written by the run script that produced it.

    python -m scripts.report --out=<file.md> <tag> [<tag> ...]
"""
import argparse
import json
import os

import pandas as pd

from nanochat.eval.loss_eval import CONTEXTS, bpb
from nanochat.paths import base_dir


def table(title: str, rows: dict[str, dict[str, float]], fmt: str) -> str:
    """A titled Markdown table of {row label: {tag: value}}, one column per tag."""
    return f"### {title}\n\n{pd.DataFrame(rows).T.to_markdown(floatfmt=fmt)}\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("tags", nargs="+")
    args = parser.parse_args()
    results = {}
    for tag in args.tags:
        with open(os.path.join(base_dir(), "base_eval", f"{tag}.json"), encoding="utf-8") as f:
            results[tag] = json.load(f)
    first = results[args.tags[0]]
    val = {c: {t: bpb(r["8k"]["corpora"][c]) for t, r in results.items()} for c in first["8k"]["corpora"]}
    val |= {f"mixture: {g}": {t: r["8k"]["bpb"][g] for t, r in results.items()} for g in first["8k"]["bpb"]}
    tables = [table("Validation bits per byte (8K rows)", val, ".5f")]
    if "32k" in first:
        edges = (0, *CONTEXTS, "")
        long = {f"{c}, context {edges[i] + 1}-{edges[i + 1]}": {
            t: r["32k"]["corpora"][c]["bits"][i] / r["32k"]["corpora"][c]["bytes"][i] for t, r in results.items()}
            for c in first["32k"]["corpora"] for i in range(len(CONTEXTS) + 1)}
        long |= {f"mixture: {g}": {t: r["32k"]["bpb"][g] for t, r in results.items()} for g in first["32k"]["bpb"]}
        tables.append(table("Validation bits per byte (32K rows)", long, ".5f"))
    solution = {b: {t: bpb(r["solution"][b]) for t, r in results.items()} for b in first["solution"]}
    tables.append(table("Reference-solution bits per byte", solution, ".5f"))
    passes = {f"{bench} {key}": {t: r["codegen"][bench][key] for t, r in results.items()}
              for bench, scores in first["codegen"].items() for key in scores if key.endswith("pass@1")}
    passes |= {f"{name} pass@1": {t: r["fim"][name]["pass@1"] for t, r in results.items()} for name in first["fim"]}
    tables.append(table("pass@1 (greedy)", passes, ".4f"))
    runs = "Runs (steps): " + ", ".join(f"`{t}` ({r['step']})" for t, r in results.items()) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join([runs, *tables]))


if __name__ == "__main__":
    main()
