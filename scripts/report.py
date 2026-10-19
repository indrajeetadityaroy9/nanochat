"""
base_eval results (<base_dir>/base_eval/<tag>.json) side by side as Markdown tables written to --out: val bits per byte
per corpus and per mixture group (eval/loss_eval.py mixture_bpb) at 8K, per corpus and context bucket at 32K, reference
solutions, and pass@1 of the generation and infilling benchmarks. results/ holds the tables the paper reports.

    python -m scripts.report --out=<file.md> <tag> [<tag> ...]
"""
import argparse
import json
import os

import pandas as pd

from nanochat.eval.loss_eval import CONTEXTS, bpb
from nanochat.paths import base_dir


def table(title: str, rows: dict[str, dict[str, float]]) -> str:
    """A titled Markdown table of {row label: {tag: value}}, one column per tag (to_markdown's default floatfmt)."""
    return f"### {title}\n\n{pd.DataFrame(rows).T.to_markdown()}\n"


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
    tables = [table("Validation bits per byte (8K rows)", val)]
    if "32k" in first:
        edges = (0, *CONTEXTS, "")
        long = {f"{c}, context {edges[i] + 1}-{edges[i + 1]}": {
            t: r["32k"]["corpora"][c]["bits"][i] / r["32k"]["corpora"][c]["bytes"][i] for t, r in results.items()}
            for c in first["32k"]["corpora"] for i in range(len(CONTEXTS) + 1)}
        long |= {f"mixture: {g}": {t: r["32k"]["bpb"][g] for t, r in results.items()} for g in first["32k"]["bpb"]}
        tables.append(table("Validation bits per byte (32K rows)", long))
    solution = {b: {t: bpb(r["solution"][b]) for t, r in results.items()} for b in first["solution"]}
    tables.append(table("Reference-solution bits per byte", solution))
    passes = {f"{bench} {key}": {t: r["codegen"][bench][key] for t, r in results.items()}
              for bench, scores in first["codegen"].items() for key in scores if key.endswith("pass@1")}
    passes |= {f"{name} pass@1": {t: r["fim"][name]["pass@1"] for t, r in results.items()} for name in first["fim"]}
    tables.append(table("pass@1 (greedy)", passes))
    runs = "Runs (steps): " + ", ".join(f"`{t}` ({r['step']})" for t, r in results.items()) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join([runs, *tables]))


if __name__ == "__main__":
    main()
