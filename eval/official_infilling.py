"""HumanEval infilling (arXiv 2207.14255) generation for scripts/official_eval.sh: openai/human-eval-infilling ships
the problems and the judge (evaluate_functional_correctness) but no generator, so vLLM generates the samples here from
the export, and the package's CLI judges them.

| Mode | Orders | Samples a task | Decoding |
|---|---|---|---|
| greedy | PSM, SPM | 1 | argmax |
| sampled | SPM | 100 single- and multi-line, 200 random-span | T 0.2 (arXiv 2207.14255 Fig. 6, Tables 1 and 4: SPM, T 0.2, 100 and 200 samples) |

Prompts are corpus.tokenizer.fim_prompt's token ids, the layout the packer trains (docs/corpus.md §8): prefix and suffix
encoded apart, BOS first. A middle ends at BOS (the model's end of text) or another special token, at most the
benchmark's longest reference middle in tokens (the paper names no budget). Samples are written as the CLI reads them,
{"task_id", "completion"} per line, a task's samples consecutive; sample k of task i seeded n i + k by vLLM's per-request
seed (n requests of one sample each).

    python -m eval.official_infilling <model dir> <out dir> {greedy,sampled} [<tasks per benchmark>]
"""

import json
import sys
from pathlib import Path

from human_eval_infilling.data import read_problems
from vllm import LLM, SamplingParams, TokensPrompt

from corpus.tokenizer import fim_prompt, get_tokenizer

BENCHMARKS = ("single-line", "multi-line", "random-span")
SAMPLES = {"single-line": 100, "multi-line": 100, "random-span": 200}
TEMPERATURE = 0.2
MODES = {"greedy": ("psm", "spm"), "sampled": ("spm",)}


def main(model: str, out: Path, mode: str, limit: int | None = None) -> None:
    tokenizer = get_tokenizer()
    stops = sorted(tokenizer.special_tokens.values())
    llm = LLM(model, trust_remote_code=True, dtype="bfloat16")
    out.mkdir(parents=True, exist_ok=True)
    for name in BENCHMARKS:
        problems = list(read_problems(name).values())[:limit]
        budget = max(len(tokenizer.enc.encode_ordinary(p["canonical_solution"])) for p in read_problems(name).values())
        n = 1 if mode == "greedy" else SAMPLES[name]
        for order in MODES[mode]:
            prompts = [TokensPrompt(prompt_token_ids=fim_prompt(tokenizer, p["prompt"], p["suffix"], mode=order))
                       for p in problems for _ in range(n)]
            params = [SamplingParams(temperature=0.0, max_tokens=budget, stop_token_ids=stops) if mode == "greedy"
                      else SamplingParams(temperature=TEMPERATURE, max_tokens=budget, stop_token_ids=stops,
                                          seed=n * i + k)
                      for i in range(len(problems)) for k in range(n)]
            outputs = llm.generate(prompts, params)
            samples = [{"task_id": p["task_id"], "completion": o.outputs[0].text}
                       for p, o in zip((p for p in problems for _ in range(n)), outputs, strict=True)]
            (out / f"{name}-{order}.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else None)
