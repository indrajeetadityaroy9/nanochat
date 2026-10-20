"""
base_eval results (BASE_EVAL/<tag>.json) side by side as Markdown tables written to RESULTS/<name>.md: val bits per byte
per corpus and per mixture group (eval/loss_eval.py mixture_bpb) at 8K, per corpus and context bucket at 32K, reference
solutions, pass@1 of the generation and infilling benchmarks by generation mode (greedy, system) and the system
mode's search statistics. RESULTS holds the tables the paper reports.

    python -m scripts.report --name=<name> <tag> [<tag> ...]   (a tag: base_eval's run name, <tag>[-step<step>])
"""
import argparse
import json

import pandas as pd

from nanochat.eval.loss_eval import CONTEXTS, bpb
from nanochat.paths import BASE_EVAL, RESULTS


def table(frame: pd.DataFrame, *, title: str, floatfmt: str = "g") -> str:
    """A titled Markdown table."""
    return f"### {title}\n\n{frame.to_markdown(floatfmt=floatfmt)}\n"


def tag_table(title: str, rows: dict[str, dict[str, float]]) -> str:
    """table of {row label: {tag: value}}, one column per tag."""
    return table(pd.DataFrame(rows).T, title=title)


def search_rows(bench: str, modes: dict) -> dict[str, float]:
    """A benchmark's generation-mode statistics (eval/code_eval.py checked, searched): the shares of tasks whose greedy
    completion has diagnostics (code benchmarks), whose system candidate 0 (healed, syntax-constrained greedy) has
    them, whose search accepted a candidate, whose chosen candidate has diagnostics; the mean attempts."""
    system, n = modes["system"], len(modes["system"]["attempts"])
    rows = {}
    if "diagnostics" in modes["greedy"]:
        rows["greedy with diagnostics"] = sum(d > 0 for d in modes["greedy"]["diagnostics"]) / n
    rows["system candidate 0 with diagnostics"] = sum(d[0] > 0 for d in system["diagnostics"]) / n
    rows["system accepted"] = sum(system["accepted"]) / n
    # verify.chosen: the accepted candidate (no diagnostics), else candidate 0
    rows["system chosen with diagnostics"] = sum(not a and d[0] > 0 for a, d in
                                                 zip(system["accepted"], system["diagnostics"], strict=True)) / n
    rows["system mean attempts"] = sum(system["attempts"]) / n
    return {f"{bench}: {label}": value for label, value in rows.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="writes RESULTS/<name>.md")
    parser.add_argument("tags", nargs="+")
    args = parser.parse_args()
    results = {}
    for tag in args.tags:
        results[tag] = json.loads((BASE_EVAL / f"{tag}.json").read_text(encoding="utf-8"))
    first = results[args.tags[0]]
    val = {c: {t: bpb(r["8k"]["corpora"][c]) for t, r in results.items()} for c in first["8k"]["corpora"]}
    val |= {f"mixture: {g}": {t: r["8k"]["bpb"][g] for t, r in results.items()} for g in first["8k"]["bpb"]}
    tables = [tag_table("Validation bits per byte (8K rows)", val)]
    if first["32k"] is not None:
        edges = (0, *CONTEXTS, "")
        long = {f"{c}, context {edges[i] + 1}-{edges[i + 1]}": {
            t: r["32k"]["corpora"][c]["bits"][i] / r["32k"]["corpora"][c]["bytes"][i] for t, r in results.items()}
            for c in first["32k"]["corpora"] for i in range(len(CONTEXTS) + 1)}
        long |= {f"mixture: {g}": {t: r["32k"]["bpb"][g] for t, r in results.items()} for g in first["32k"]["bpb"]}
        tables.append(tag_table("Validation bits per byte (32K rows)", long))
    solution = {b: {t: bpb(r["solution"][b]) for t, r in results.items()} for b in first["solution"]}
    tables.append(tag_table("Reference-solution bits per byte", solution))
    passes = {f"{bench} {key} ({mode})": {t: r["codegen"][bench][mode][key] for t, r in results.items()}
              for bench, modes in first["codegen"].items() for mode, scores in modes.items() for key in scores
              if key.endswith("pass@1")}
    passes |= {f"{name} pass@1 ({mode})": {t: r["fim"][name][mode]["pass@1"] for t, r in results.items()}
               for name, modes in first["fim"].items() for mode in modes}
    tables.append(tag_table("pass@1 (greedy: plain greedy decoding; system: eval/verify.py search)", passes))
    stats = {}
    for t, r in results.items():
        # FIM's system mode runs in SPM order only (eval/fim_eval.py SYSTEM_MODE)
        for bench, modes in (*r["codegen"].items(), *((b, m) for b, m in r["fim"].items() if "system" in m)):
            for label, value in search_rows(bench, modes).items():
                stats.setdefault(label, {})[t] = value
    tables.append(tag_table("Generation modes: shares of tasks with diagnostics, accepted, mean attempts", stats))
    runs = "Runs (steps): " + ", ".join(f"`{t}` ({r['step']})" for t, r in results.items()) + "\n"
    (RESULTS / f"{args.name}.md").write_text("\n".join([runs, *tables]), encoding="utf-8")


if __name__ == "__main__":
    main()
