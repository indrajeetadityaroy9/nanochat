"""Natural termination of a post-trained checkpoint's reasoning (docs/post-training.md §5.1): a fixed prompt set sampled
once each by vLLM offline from the --chat export (inference/export_hf.py; bf16, the head in fp32, eos = BOS, max new
tokens the context less the prompt), and how the responses end:

- evalplus: the first --prompts HumanEval+ tasks' messages as EvalPlus's OpenAI backend sends them
  (eval/official_tokenize.py evalplus_messages), rendered by the export's chat template as `vllm serve` renders them
  for scripts/official_reasoning.sh, each checked to equal BOS + encode_ordinary(prompt.strip() + THINK).
- val: --prompts SFT val documents drawn without replacement by default_rng(SEED) (corpus/posttrain/build.py's
  held-out problems), their context (the prompt and THINK, <split>.context) after a BOS, the ids the SFT trained on;
  each with its source group and style (val-labels.jsonl, build.py labels_path) and its teacher's reasoning tokens.

Sampling: T 0.6, top-p 0.95 (the reasoning protocol's), seed SEED + the prompt's index. A response closes when it holds
"</think>" (the reasoning parser's split, eval/official_reasoning_parser.py: the reasoning through the last one, the
answer after it) and ends when it emits BOS (vLLM's finish reason "stop"; else "length", the limit). Per set
(`summary`): the four exclusive outcomes (closed and ended, closed and at the limit, unclosed at the limit, unclosed
and ended); the closing-time curve P(closed with reasoning <= B) at BUDGETS; reasoning tokens p50/p90/p95 (an unclosed
response all reasoning) and answer tokens p50 given closure; the shares of closed answers holding a fenced code block
and ending with BOS; the evalplus responses' pass on EvalPlus's base and plus tests (evalplus.evaluate's
check_correctness on the answer as EvalPlus's codegen sanitizes it; an unclosed response's answer is empty), over all
and given closure; the val responses' outcomes by source and by teacher-length bucket (TEACHER_BUCKETS) and the rank
correlation of teacher length with closure; the never-closed responses' mean repetition (`repetition`). Each response
keeps its prompt's id and the last TAIL_CHARS characters of its text. Beside a training job (its metrics.jsonl and the
probe's start time given), the job's train tokens/s during the probe against its BEFORE steps before it and the
seconds it lost, as information. VLLM_GPU_MEMORY_UTILIZATION applies through eval/official_site/sitecustomize.py.

    python -m eval.official_termination <model dir> <out json> [<metrics.jsonl> <since, unix seconds>] [--prompts N]
"""

import argparse
import gc
import json
import os
import re
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from corpus.layout import TOKEN_DTYPE, split_path
from corpus.posttrain.sft import END_THINK, NAME, THINK
from corpus.tokenizer import get_tokenizer
from eval.official_tokenize import evalplus_tasks

PROMPTS, SEED, BEFORE = 64, 0, 50
SAMPLING = {"temperature": 0.6, "top_p": 0.95}
END = "</think>"
FENCE = re.compile(r"```[^\n]*\n.*?```", re.S)
BUDGETS = (1024, 2048, 4096, 8192, 16384, 32768)
TEACHER_BUCKETS = (2048, 4096, 8192, 16384)  # <2K, 2-4K, 4-8K, 8-16K, >=16K
OUTCOMES = ("closed_ended", "closed_limit", "unclosed_limit", "unclosed_ended")
TAIL_CHARS, REPEAT_TAIL, REPEAT_WINDOW = 2000, 4096, 32


def evalplus_prompts(template, n: int) -> list[dict]:
    tok = get_tokenizer()
    out = []
    for task_id, messages in evalplus_tasks()[:n]:
        ids = template.encode(template.apply_chat_template(messages, add_generation_prompt=True, tokenize=False),
                              add_special_tokens=False)
        text = "\n\n".join(m["content"] for m in messages)
        if ids != [tok.bos_token_id, *tok.enc.encode_ordinary(text.strip() + THINK)]:
            raise ValueError(f"the chat template renders an EvalPlus prompt differently: {text[:200]!r}")
        out.append({"id": task_id, "ids": ids})
    return out


def val_prompts(n: int) -> list[dict]:
    tok = get_tokenizer()
    path = split_path(NAME, "val", tok)
    tokens = np.memmap(path.with_suffix(".bin"), dtype=TOKEN_DTYPE, mode="r")
    offsets = np.memmap(path.with_suffix(".idx"), dtype=np.int64, mode="r")
    contexts = np.fromfile(path.with_suffix(".context"), dtype=np.int64)
    labels = [json.loads(line) for line in path.with_name("val-labels.jsonl").read_text().splitlines()]
    out = []
    for i in sorted(np.random.default_rng(SEED).choice(len(contexts), size=n, replace=False).tolist()):
        document = tokens[offsets[i]:offsets[i + 1]].tolist()
        context = document[:contexts[i]]
        assert tok.enc.decode(context).endswith(THINK)
        # the teacher's reasoning, encoded alone as build.py render encodes it
        reasoning = tok.enc.decode(document[contexts[i]:]).partition(END_THINK)[0]
        source = labels[i]["group"] + (f"/{labels[i]['style']}" if labels[i]["style"] else "")
        out.append({"id": i, "ids": [tok.bos_token_id, *context], "source": source,
                    "teacher_reasoning_tokens": len(tok.enc.encode_ordinary(reasoning))})
    return out


def repetition(ids: list[int]) -> float:
    """The share of the last REPEAT_TAIL tokens inside a REPEAT_WINDOW-token window that recurs within them."""
    tail = ids[-REPEAT_TAIL:]
    windows = [tuple(tail[i:i + REPEAT_WINDOW]) for i in range(len(tail) - REPEAT_WINDOW + 1)]
    counts = Counter(windows)
    covered = np.zeros(len(tail), bool)
    for i, w in enumerate(windows):
        if counts[w] > 1:
            covered[i:i + REPEAT_WINDOW] = True
    return float(covered.mean()) if len(tail) else 0.0


def response(ids: list[int], text: str, finish: str, bos: int, decode) -> dict:
    """One response's ending: tokens (the closing BOS not counted), closure, the reasoning through the last END."""
    ended = finish == "stop"  # the only stop condition: eos (BOS)
    n = len(ids) - (ids[-1:] == [bos])
    closures = text.count(END)
    lo, hi = 1, n  # the shortest prefix holding every END (decode(ids[:k]).count(END) only grows with k)
    while closures and lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (mid + 1, hi) if decode(ids[:mid]).count(END) < closures else (lo, mid)
    reasoning = lo if closures else n
    answer = text.rpartition(END)[2].strip() if closures else ""
    outcome = ("closed" if closures else "unclosed") + ("_ended" if ended else "_limit")
    return {"tokens": n, "finish": finish, "outcome": outcome, "closed": closures > 0, "closures": closures,
            "ended": ended, "reasoning_tokens": reasoning, "answer_tokens": n - reasoning,
            "fenced": bool(FENCE.search(answer)), "answer": answer, "tail": text[-TAIL_CHARS:],
            "repetition": None if closures else repetition(ids[:n])}


def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank)."""
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):
        r[x == v] = r[x == v].mean()
    return r


def outcomes(rows: list[dict]) -> dict:
    counts = Counter(r["outcome"] for r in rows)
    return {"responses": len(rows), "outcomes": {k: counts[k] for k in OUTCOMES},
            "closed": sum(r["closed"] for r in rows) / len(rows),
            "at_length_limit": sum(not r["ended"] for r in rows) / len(rows)}


def summary(rows: list[dict]) -> dict:
    closed = [r for r in rows if r["closed"]]
    unclosed = [r for r in rows if not r["closed"]]
    out = outcomes(rows) | {
        "closed_within": {str(b): sum(r["reasoning_tokens"] <= b for r in closed) / len(rows) for b in BUDGETS},
        "reasoning_tokens_p50_p90_p95": np.percentile([r["reasoning_tokens"] for r in rows], [50, 90, 95]).tolist(),
        "reasoning_tokens_max_closed": max((r["reasoning_tokens"] for r in closed), default=None),
        "answer_tokens_p50_closed": float(np.median([r["answer_tokens"] for r in closed])) if closed else None,
        "fenced_of_closed": sum(r["fenced"] for r in closed) / len(closed) if closed else None,
        "ended_of_closed": sum(r["ended"] for r in closed) / len(closed) if closed else None,
        "repetition_mean_unclosed": statistics.fmean(r["repetition"] for r in unclosed) if unclosed else None}
    if all("base" in r for r in rows):  # the evalplus set
        out["evalplus_pass"] = {k: sum(r[k] for r in rows) / len(rows) for k in ("base", "plus")}
        out["evalplus_pass_closed"] = {k: sum(r[k] for r in closed) / len(closed) if closed else None
                                       for k in ("base", "plus")}
    if all("teacher_reasoning_tokens" in r for r in rows):  # the val set
        by_source = {}
        for r in rows:
            by_source.setdefault(r["source"], []).append(r)
        out["by_source"] = {s: outcomes(rs) for s, rs in sorted(by_source.items())}
        edges = (0, *TEACHER_BUCKETS, None)
        out["by_teacher_reasoning"] = {}
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            rs = [r for r in rows if r["teacher_reasoning_tokens"] >= lo
                  and (hi is None or r["teacher_reasoning_tokens"] < hi)]
            if rs:
                out["by_teacher_reasoning"][f"{lo}-{hi or ''}"] = outcomes(rs)
        teacher = np.array([r["teacher_reasoning_tokens"] for r in rows], float)
        closure = np.array([r["closed"] for r in rows], float)
        out["teacher_length_closure_spearman"] = (float(np.corrcoef(ranks(teacher), ranks(closure))[0, 1])
                                                  if closure.std() and teacher.std() else None)
    return out


def evalplus_check(rows: list[dict]) -> None:
    """Each evalplus response's base and plus pass (EvalPlus's codegen sanitize, then evaluate's check_correctness)."""
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
    from evalplus.evaluate import check_correctness, get_groundtruth
    from evalplus.sanitize import sanitize

    problems = get_human_eval_plus()
    expected = get_groundtruth(problems, get_human_eval_plus_hash(), [])
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) // 2)) as pool:
        futures = [pool.submit(check_correctness, "humaneval", 0, problems[r["id"]],
                               sanitize(r["answer"], entrypoint=problems[r["id"]]["entry_point"]),
                               expected[r["id"]], False, True) for r in rows]
        for r, future in zip(rows, futures, strict=True):
            result = future.result()
            r["base"] = result["base"][0] == "pass"
            r["plus"] = r["base"] and result["plus"][0] == "pass"


def train_steps(metrics: Path) -> list[tuple[float, int, float, float]]:
    """The job's train steps: (time, step, tokens/s, seconds)."""
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    return [(r["time"], r["step"], r["tokens_per_second"], r["step_seconds"]) for r in rows if r.get("kind") == "train"]


def job_cost(metrics: Path, since: float, end: float) -> dict:
    """The job's median rate over its BEFORE steps before `since`, its steps in [since, end]: their rates and the
    seconds they took beyond the before rate."""
    steps = train_steps(metrics)
    before = statistics.median([rate for t, _, rate, _ in steps if t < since][-BEFORE:])
    during = [(step, rate, seconds) for t, step, rate, seconds in steps if since <= t <= end]
    if not during:
        return {"before_median": before, "steps": None}
    rates = [rate for _, rate, _ in during]
    return {"before_median": before, "steps": [during[0][0], during[-1][0]],
            "during_median": statistics.median(rates), "during_mean": statistics.fmean(rates),
            "slowdown_median": 1 - statistics.median(rates) / before,
            "seconds_lost": sum(seconds * (1 - rate / before) for _, rate, seconds in during)}


def main(model: Path, out: Path, metrics: Path | None, since: float | None, n: int) -> None:
    from vllm import LLM, SamplingParams

    start = time.time()
    context = json.loads((model / "config.json").read_text())["max_position_embeddings"]
    llm = LLM(str(model), trust_remote_code=True, dtype="bfloat16", max_model_len=context, seed=SEED)
    tok = get_tokenizer()
    sets = {"evalplus": evalplus_prompts(llm.get_tokenizer(), n), "val": val_prompts(n)}
    flat = [(name, p) for name, every in sets.items() for p in every]
    generating = time.time()
    outputs = llm.generate([{"prompt_token_ids": p["ids"]} for _, p in flat],
                           [SamplingParams(**SAMPLING, max_tokens=context - len(p["ids"]), seed=SEED + i)
                            for i, (_, p) in enumerate(flat)], use_tqdm=False)
    end = time.time()
    rows = {name: [] for name in sets}
    for (name, p), output in zip(flat, outputs, strict=True):
        (completion,) = output.outputs
        rows[name].append({k: v for k, v in p.items() if k != "ids"} | {"prompt_tokens": len(p["ids"])}
                          | response(list(completion.token_ids), completion.text, completion.finish_reason,
                                     tok.bos_token_id, tok.enc.decode))
    del llm, outputs  # the engine's GPU memory, released before the CPU-only checks
    gc.collect()
    evalplus_check(rows["evalplus"])
    report = {"model": str(model), "context": context, "sampling": SAMPLING | {"n": 1, "seed": SEED},
              "prompts_per_set": n,
              "sets": {name: summary(r) for name, r in rows.items()} | {"all": summary(sum(rows.values(), []))},
              "seconds": round(time.time() - start), "generation_seconds": round(end - generating),
              "responses": {name: [{k: v for k, v in r.items() if k != "answer"} for r in rs]
                            for name, rs in rows.items()}}
    if metrics is not None:
        report["job_tokens_per_second"] = job_cost(metrics, start if since is None else since, end)
    out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("metrics", type=Path, nargs="?")
    parser.add_argument("since", type=float, nargs="?")
    parser.add_argument("--prompts", type=int, default=PROMPTS)
    args = parser.parse_args()
    main(args.model, args.out, args.metrics, args.since, args.prompts)
