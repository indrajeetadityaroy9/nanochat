"""Checks of the prompts scripts/official_reasoning.sh serves, before generation: every message a tool will send holds no
"\\r" (the corpus normalizes line endings, eval/line_endings.py, so the model never trained on one; docs/evaluation.md
§4), and vLLM's POST /tokenize renders the first problem's messages through the --chat export's template
(inference/export_hf.py CHAT_TEMPLATE) as BOS + encode_ordinary(prompt.strip() + THINK), prompt being the text the SFT
renderer saw (posttrain/sft_data.py render; docs/post-training.md §11.1):

- evalplus: EvalPlus's OpenAI backend's one user message per HumanEval+ and MBPP+ task (evalplus/provider/openai.py
  codegen: the instruction prefix of codegen.py:222, then the task prompt stripped in a python fence); prompt = that
  message.
- lcb: the messages eval/official_lcb.py sends for each problem of the release and contest-date window (lcb_runner's
  benchmarks/code_generation.py load_code_generation_dataset, as its main loads them; OpenAIChat's
  SYSTEM_MESSAGE_GENERIC and generic question template, line-ending-normalized); prompt = the system message, "\\n\\n",
  the question (lcb_runner's no-system-role style). Also counts the problems whose unnormalized messages hold "\\r".
  Run by /opt/lcb's Python in /opt/LiveCodeBench (code_generation.py reads its few-shot files relative to the working
  directory).

    python -m eval.official_tokenize evalplus <server URL> <out json>
    python -m eval.official_tokenize lcb <server URL> <out json> <release> <start date> <end date>
    python -m eval.official_tokenize check <json>...    (nanochat image; exit 1 on a "\\r" or an id mismatch)
"""

import json
import sys
import urllib.request
from pathlib import Path

MODEL = "nanochat"
INSTRUCTION_PREFIX = ("Please provide a self-contained Python script that solves the following problem in a markdown "
                      "code block:")


def evalplus_messages() -> list[list[dict]]:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    return [[{"role": "user", "content": INSTRUCTION_PREFIX + f"\n```python\n{task['prompt'].strip()}\n```"}]
            for load in (get_human_eval_plus, get_mbpp_plus) for task in load().values()]


def lcb_messages(release: str, start: str, end: str) -> tuple[list[list[dict]], int]:
    """eval/official_lcb.py's messages of every window problem, and how many problems' unnormalized ones hold "\\r"."""
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.prompts.code_generation import format_prompt_generation

    from eval.official_lcb import messages

    problems = sorted(load_code_generation_dataset(release, start_date=start, end_date=end),
                      key=lambda p: p.question_id)
    raw = sum(any("\r" in m["content"] for m in format_prompt_generation(p, LMStyle.OpenAIChat)) for p in problems)
    return [messages(p, LMStyle.OpenAIChat) for p in problems], raw


def served(url: str, messages: list[dict]) -> list[int]:
    body = json.dumps({"model": MODEL, "messages": messages, "add_generation_prompt": True}).encode()
    request = urllib.request.Request(f"{url}/tokenize", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())["tokens"]


def check(paths: list[str]) -> bool:
    from corpus.tokenizer import get_tokenizer
    from posttrain.sft import THINK

    tok = get_tokenizer()
    ok = True
    for path in paths:
        x = json.loads(Path(path).read_text())
        equal = x["ids"] == [tok.bos_token_id, *tok.enc.encode_ordinary(x["prompt"].strip() + THINK)]
        print(f"{path}: {x['prompts']} prompts, {x['with_cr']} with a carriage return "
              f"({x['with_cr_before_normalization']} before normalization); the first's {len(x['ids'])} "
              f"served ids {'equal' if equal else 'DIFFERENT'}")
        ok &= equal and x["with_cr"] == 0
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "check":
        sys.exit(0 if check(sys.argv[2:]) else 1)
    if sys.argv[1] == "evalplus":
        every, raw = evalplus_messages(), None
    else:
        every, raw = lcb_messages(*sys.argv[4:7])
    first = every[0]
    Path(sys.argv[3]).write_text(json.dumps({
        "prompts": len(every), "with_cr": sum(any("\r" in m["content"] for m in ms) for ms in every),
        "with_cr_before_normalization": raw, "messages": first, "prompt": "\n\n".join(m["content"] for m in first),
        "ids": served(sys.argv[2], first)}))
