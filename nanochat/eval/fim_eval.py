"""
Fill-in-the-middle evaluation of a base model trained with FIM sentinels (data/pack.py): openai/human-eval-infilling
(Bavarian et al., arXiv 2207.14255; the package and its data installed in the image, docker/Dockerfile): single-line,
multi-line and random-span infilling of the 164 HumanEval solutions. Each problem's prompt (prefix) and suffix form a
PSM or SPM prompt (inference.decoder.fim_prompt); the model writes the middle greedily until it writes BOS (its
end-of-middle target in training), bounded by the benchmark's longest reference middle in tokens. Scored by the
package's own entry point (`python -m human_eval_infilling.evaluate_functional_correctness`, its 3 s timeout and 4
workers) in its own process; pass@1 by its estimate_pass_at_k over the per-problem verdicts.
"""

import json
import os
import subprocess
import sys
import tempfile

from human_eval_infilling.data import read_problems
from human_eval_infilling.evaluation import estimate_pass_at_k

from nanochat.inference.decoder import NativeDecoder, fim_prompt

BENCHMARKS = ("single-line", "multi-line", "random-span")
FIM_MODES = ("psm", "spm")  # the two orders the packer trains (Bavarian et al. App. D)


def infilling(name, problems, middles):
    """The official evaluation of one middle per problem of benchmark `name`: {"pass@1", "ids", "completions",
    "result"} ("passed", "failed: <exception>" or "timed out" per problem, human_eval_infilling/execution.py)."""
    with tempfile.TemporaryDirectory() as tmp:
        samples = os.path.join(tmp, "samples.jsonl")
        with open(samples, "w", encoding="utf-8") as f:
            for p, middle in zip(problems, middles, strict=True):
                f.write(json.dumps({"task_id": p["task_id"], "completion": middle}) + "\n")
        subprocess.run([sys.executable, "-I", "-m", "human_eval_infilling.evaluate_functional_correctness", name,
                        samples], cwd=tmp, check=True)
        with open(samples + "_results.jsonl", encoding="utf-8") as f:
            scored = [json.loads(line) for line in f]
    return {"pass@1": float(estimate_pass_at_k(1, [s["passed"] for s in scored], 1).mean()),
            "ids": [s["task_id"] for s in scored], "completions": middles, "result": [s["result"] for s in scored]}


def fim_generation(model, tokenizer):
    """{"<benchmark>/<mode>": infilling(...)} of greedy middles in each FIM_MODES order."""
    decoder, results = NativeDecoder(model, bos_token_id=tokenizer.bos_token_id), {}
    for name in BENCHMARKS:
        problems = list(read_problems(name).values())
        max_tokens = max(len(tokenizer.enc.encode_ordinary(p["canonical_solution"])) for p in problems)
        for mode in FIM_MODES:
            middles = [None] * len(problems)
            requests = ((fim_prompt(tokenizer, p["prompt"], p["suffix"], mode), max_tokens, None) for p in problems)
            for i, ids in decoder.run(requests):
                middles[i] = tokenizer.enc.decode(ids)
            results[f"{name}/{mode}"] = infilling(name, problems, middles)
    return results
