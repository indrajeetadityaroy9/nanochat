"""The official tools' outputs for one checkpoint (scripts/official_eval.sh, RUNS/<run>/official-<step>/) as one record
in eval/suite.py's codegen and fim layout, so that metrics.compare pairs it with another checkpoint's record or with
the harness's eval-<step>.json; per-problem pass from each tool's own verdict:

| Record | Tool output | Pass |
| codegen.{humaneval,mbpp}_plus.greedy | evalplus.evaluate's `<samples>.eval_results.json` | base_status, plus_status |
| codegen.{humaneval,mbpp}_cpp.{greedy,sampled} | MultiPL-E main.py's `<problem>.results.json.gz` | pass_k.py's: "OK", exit 0 |
| codegen.mbpp_3shot.greedy | lm-evaluation-harness `--log_samples`: `samples_mbpp_*.jsonl` | its `pass_at_1` |
| fim.santacoder_fim/psm.greedy | bigcode-evaluation-harness generations and references (eval/official_fim.py) | its exact match (`py` rows) |
| fim.<benchmark>/<order>.{greedy,sampled} | human-eval-infilling's `<samples>_results.jsonl.gz` (eval/official_infilling.py) | its `passed` |

Cells as eval/suite.py writes them: metrics.pass_cell (mean, se, Wilson for one sample a problem), the fim records'
metrics.summary clustered by HumanEval problem (metrics.problem_clusters), evalplus_average of the greedy records.
Infilling records join the summary once their verdicts exist.

    python -m eval.official <official dir> [<eval-<step>.json to compare against>]
"""

import gzip
import json
import re
import sys
from pathlib import Path

import metrics

FIM = "santacoder_fim/psm"


def evalplus(out: Path, dataset: str) -> dict:
    (results,) = (out / "evalplus" / dataset).glob("*.eval_results.json")
    tasks = json.loads(results.read_text())["eval"]
    return {"ids": list(tasks), "base_status": [[r["base_status"] for r in rs] for rs in tasks.values()],
            "plus_status": [[r["plus_status"] for r in rs] for rs in tasks.values()]}


def multipl_e(directory: Path) -> dict:
    """pass_k.py for_file's count: status "OK" and exit code 0; problems in the dataset's order (their index)."""
    files = sorted(directory.glob("*.results.json.gz"), key=lambda p: int(p.name.split("_")[1]))
    problems = [json.loads(gzip.decompress(p.read_bytes())) for p in files]
    return {"ids": [p["name"] for p in problems],
            "status": [["OK" if r["status"] == "OK" and r["exit_code"] == 0 else r["status"] for r in p["results"]]
                       for p in problems]}


def mbpp_3shot(out: Path) -> dict:
    (samples,) = (out / "lm-eval").rglob("samples_mbpp_*.jsonl")
    rows = sorted((json.loads(line) for line in samples.read_text().splitlines()), key=lambda r: r["doc"]["task_id"])
    return {"ids": [str(r["doc"]["task_id"]) for r in rows],
            "status": [["passed" if r["pass_at_1"] == 1 else "failed"] for r in rows]}


def santacoder(out: Path) -> tuple[dict, dict]:
    """The Python rows' exact match as the task's process_results counts it, and every language's rate."""
    from datasets import load_dataset
    rows = load_dataset("bigcode/santacoder-fim-task", split="train")
    generations = json.loads((out / "bigcode" / "generations_nanochat_fim.json").read_text())
    references = json.loads((out / "bigcode" / "references_nanochat_fim.json").read_text())
    hits = [(row["language"], row["name"], g.strip() == ref.strip())
            for row, (g,), ref in zip(rows, generations, references, strict=True)]
    rates = {lang: sum(h for lg, _, h in hits if lg == lang) / sum(lg == lang for lg, _, _ in hits)
             for lang in sorted({lg for lg, _, _ in hits})}
    py = [(name, hit) for lang, name, hit in hits if lang == "py"]
    return {"ids": [n for n, _ in py], "exact_match": [h for _, h in py]}, rates


def infilling(path: Path) -> dict:
    """One judged sample file: greedy {"ids", "result"} (eval/suite.py's layout), sampled {"ids", "status"} with each
    task's samples' `passed`; tasks in file order."""
    tasks: dict[str, list[dict]] = {}
    for line in gzip.decompress(path.read_bytes()).decode().splitlines():
        sample = json.loads(line)
        tasks.setdefault(sample["task_id"], []).append(sample)
    if all(len(s) == 1 for s in tasks.values()):
        return {"ids": list(tasks), "result": [s[0]["result"] for s in tasks.values()]}
    return {"ids": list(tasks), "status": [[x["passed"] for x in s] for s in tasks.values()]}


def record(out: Path) -> dict:
    codegen = {"humaneval_plus": {"greedy": evalplus(out, "humaneval")},
               "mbpp_plus": {"greedy": evalplus(out, "mbpp")},
               **{f"{d}_cpp": {mode: multipl_e(out / "multipl-e" / mode / f"{d}-cpp") for mode in ("greedy", "sampled")}
                  for d in ("humaneval", "mbpp")},
               "mbpp_3shot": {"greedy": mbpp_3shot(out)}}
    for benchmark, modes in codegen.items():
        for r in modes.values():  # eval/suite.py's cells: EvalPlus {column: {"pass@1": cell}}, else {"pass@1": cell}
            columns = metrics.pass_columns(benchmark, r)
            r["scores"] = ({c: {"pass@1": metrics.pass_cell(p)} for c, p in columns.items()} if "base_status" in r
                           else {"pass@1": metrics.pass_cell(columns["pass@1"])})
    fims = {}
    if (out / "bigcode" / "evaluation_results.json").exists():
        fim, rates = santacoder(out)
        fim["scores"] = {"exact_match": metrics.summary(fim["exact_match"], metrics.problem_clusters(fim))}
        fim["languages"] = rates
        fims[FIM] = {"greedy": fim}
    for mode in ("greedy", "sampled"):
        for path in sorted((out / "infilling" / mode).glob("*.jsonl_results.jsonl.gz")):
            name, order = path.name.removesuffix(".jsonl_results.jsonl.gz").rsplit("-", 1)
            r = fims.setdefault(f"{name}/{order}", {})[mode] = infilling(path)
            clusters = ["/".join(i.split("/")[1:3]) for i in r["ids"]]  # the HumanEval problem
            passed = metrics.pass_columns(f"{name}/{order}", r)["pass@1"]
            r["scores"] = {"pass@1": metrics.summary([sum(p) / len(p) for p in passed], clusters)}
    return {"step": int(re.fullmatch(r"official-(\d+)", out.name)[1]), "codegen": codegen, "fim": fims,
            "evalplus_average": {"greedy": metrics.evalplus_average(codegen["humaneval_plus"]["greedy"],
                                                                   codegen["mbpp_plus"]["greedy"])},
            "tools": json.loads((out / "tools.json").read_text())}


def aligned(a: dict, b: dict) -> dict:
    """b's codegen and fim records in a's problem order, on the problems a's records hold (KeyError if b lacks one)."""
    out = {"codegen": {}, "fim": {}}
    for section in out:
        for benchmark, modes in b[section].items():
            for mode, r in modes.items():
                if benchmark not in a.get(section, {}) or mode not in a[section][benchmark]:
                    continue
                where = {i: k for k, i in enumerate(r["ids"])}
                order = [where[i] for i in a[section][benchmark][mode]["ids"]]
                out[section].setdefault(benchmark, {})[mode] = {
                    k: [v[j] for j in order] for k, v in r.items() if isinstance(v, list) and len(v) == len(r["ids"])}
    return out


def agreement(a: dict, b: dict) -> dict:
    """Per single-sample row: problems both pass, a only, b only, neither (metrics.compare's McNemar table)."""
    out = {}
    for section in ("codegen", "fim"):
        for benchmark, modes in b[section].items():
            for mode, y in modes.items():
                columns_b = metrics.pass_columns(benchmark, y)
                for column, pa in metrics.pass_columns(benchmark, a[section][benchmark][mode]).items():
                    pb = columns_b[column]
                    if all(len(p) == 1 for p in (*pa, *pb)):
                        pairs = [(x[0], z[0]) for x, z in zip(pa, pb, strict=True)]
                        out[f"{section}/{benchmark}/{mode}/{column}"] = {
                            "both": sum(x and z for x, z in pairs), "a_only": sum(x and not z for x, z in pairs),
                            "b_only": sum(z and not x for x, z in pairs), "neither": sum(not (x or z) for x, z in pairs)}
    return out


if __name__ == "__main__":
    out = Path(sys.argv[1])
    summary = record(out)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    if len(sys.argv) > 2:
        harness = json.loads(Path(sys.argv[2]).read_text())
        official = aligned(harness, summary)
        (out / f"compare-{Path(sys.argv[2]).stem}.json").write_text(json.dumps(
            {"a": str(sys.argv[2]), "b": "official", "rows": metrics.compare(harness, official),
             "agreement": agreement(harness, official)}, indent=1) + "\n")
