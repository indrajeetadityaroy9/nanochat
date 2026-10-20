"""
Fill-in-the-middle evaluation on openai/human-eval-infilling (Bavarian et al., arXiv 2207.14255; installed in the
image, docker/Dockerfile): single-line, multi-line and random-span infilling of the 164 HumanEval solutions. Each
problem's prefix and suffix form a PSM or SPM prompt (inference.decoder.fim_prompt); the model writes the middle
until BOS (its end-of-middle target in training), at most the benchmark's longest reference middle in tokens.
Two generation modes: "greedy", in PSM and SPM order; "system" in SPM order only (SYSTEM_MODE), verify.search:
candidates under the complete Python 3 syntax constraint given the prefix and suffix (inference/syntax_constraint.py:
BOS iff prefix + middle + suffix parses), the greedy one then seeded samples until prefix + middle + suffix has no
diagnostics and passes the examples of the problem's prompt (the prefix). FIM prompts are not healed: they end with a
sentinel (PSM) or with the prefix encoded on its own (SPM), the training format, whose character-level spans train
mid-token boundaries (data/bpe/pack.rs random_span). Scored by the package's entry point
(`python -m human_eval_infilling.evaluate_functional_correctness`: 3 s timeout, 4 workers) in its own process; pass@1
by its estimate_pass_at_k.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from human_eval_infilling.data import read_problems
from human_eval_infilling.evaluation import estimate_pass_at_k

from nanochat.eval.code_eval import searched
from nanochat.eval.verify import chosen, search, syntax_constraint
from nanochat.inference.decoder import NativeDecoder, Request, fim_prompt

BENCHMARKS = ("single-line", "multi-line", "random-span")
FIM_MODES = ("psm", "spm")  # the two orders the packer trains (Bavarian et al. App. D)
# the system mode's order: at d8 SPM beat PSM on all three benchmarks (greedy pass@1 single-line 0.036 vs 0.003,
# multi-line 0.013 vs 0.001, random-span 0.008 vs 0)
SYSTEM_MODE = "spm"


def infilling(name, problems, middles):
    """Official evaluation of one middle per problem: {"pass@1", "ids", "completions", "result"} (per problem "passed",
    "failed: <exception>" or "timed out", human_eval_infilling/execution.py)."""
    with tempfile.TemporaryDirectory() as tmp:
        samples = Path(tmp, "samples.jsonl")
        with samples.open("w", encoding="utf-8") as f:
            for p, middle in zip(problems, middles, strict=True):
                f.write(json.dumps({"task_id": p["task_id"], "completion": middle}) + "\n")
        subprocess.run([sys.executable, "-I", "-m", "human_eval_infilling.evaluate_functional_correctness", name,
                        samples], cwd=tmp, check=True)
        with Path(tmp, "samples.jsonl_results.jsonl").open(encoding="utf-8") as f:
            scored = [json.loads(line) for line in f]
    return {"pass@1": float(estimate_pass_at_k(1, [s["passed"] for s in scored], 1).mean()),
            "ids": [s["task_id"] for s in scored], "completions": middles, "result": [s["result"] for s in scored]}


def system_middles(decoder, tokenizer, problems, prompts, *, max_tokens):
    """The system mode's middles and verify.search's candidates per problem: candidates of its FIM prompt under the
    syntax constraint of its prefix and suffix (verify.syntax_constraint), checked (verify.verdicts) as prefix + middle +
    suffix against the examples of its prompt (the prefix); the middle of verify.chosen's candidate."""

    def request(i):
        p = problems[i]
        return Request(prompt=prompts[i], max_tokens=max_tokens,
                       constraint=syntax_constraint(p["prompt"], suffix=p["suffix"]))

    def check(i, ids):
        p = problems[i]
        return p["prompt"], p["prompt"] + tokenizer.enc.decode(ids) + p["suffix"]

    tried = search(decoder, request, check, language="python", count=len(problems))
    return [tokenizer.enc.decode(chosen(c).ids) for c in tried], tried


def fim_generation(model, tokenizer):
    """{"<benchmark>/<mode>": {"greedy": infilling(...) of greedy middles}} in each FIM_MODES order, the SYSTEM_MODE
    entry also {"system": that of system_middles and its code_eval.searched record}."""
    runs = {}
    for name in BENCHMARKS:
        problems = list(read_problems(name).values())
        max_tokens = max(len(tokenizer.enc.encode_ordinary(p["canonical_solution"])) for p in problems)
        prompts = {mode: [fim_prompt(tokenizer, p["prompt"], p["suffix"], mode=mode) for p in problems]
                   for mode in FIM_MODES}
        runs[name] = problems, max_tokens, prompts
    max_len = max(len(prompt) + max_tokens - 1 for _, max_tokens, prompts in runs.values()
                  for mode_prompts in prompts.values() for prompt in mode_prompts)
    decoder, results = NativeDecoder(model, bos_token_id=tokenizer.bos_token_id, max_len=max_len), {}
    for name, (problems, max_tokens, prompts) in runs.items():
        for mode in FIM_MODES:
            middles = [None] * len(problems)
            for i, ids in decoder.run(Request(prompt=prompt, max_tokens=max_tokens) for prompt in prompts[mode]):
                middles[i] = tokenizer.enc.decode(ids)
            results[f"{name}/{mode}"] = {"greedy": infilling(name, problems, middles)}
            if mode == SYSTEM_MODE:
                system, tried = system_middles(decoder, tokenizer, problems, prompts[mode], max_tokens=max_tokens)
                results[f"{name}/{mode}"]["system"] = {**infilling(name, problems, system), **searched(tried)}
    return results
