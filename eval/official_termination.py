"""Natural termination of a post-trained checkpoint's reasoning (docs/evaluation.md §10): a fixed prompt set sampled
once each by vLLM offline from the --chat export (inference/export_hf.py; bf16, the head in fp32, eos = BOS, max new
tokens the context less the prompt), and how the responses end:

- evalplus: the first --prompts HumanEval+ tasks' messages as EvalPlus's OpenAI backend sends them
  (eval/official_tokenize.py evalplus_messages), rendered by the export's chat template as `vllm serve` renders them
  for scripts/official_reasoning.sh, each checked to equal BOS + encode_ordinary(prompt.strip() + THINK).
- val: --prompts SFT val documents drawn without replacement by default_rng(SEED) (corpus/posttrain/build.py's
  held-out problems), their context (the prompt and THINK, <split>.context) after a BOS, the ids the SFT trained on;
  each with its source group and style (val-labels.jsonl, build.py labels_path) and its teacher's reasoning tokens.

Sampling: T 0.6, top-p 0.95 (the reasoning protocol's), seed SEED + the prompt's index, in CHUNK-token rounds with
prefix caching (`generate`; a round's seed offset by SEED_STRIDE per round, so only the first CHUNK tokens equal a
one-shot generation's). A response closes when it holds "</think>" (the reasoning parser's split,
eval/official_reasoning_parser.py: the reasoning through the last one, the answer after it) and ends when it emits BOS
(finish "stop"; else "length", the limit, or "loop": stopped after a round with its last REPEAT_TAIL tokens
`repetition` >= LOOP and no "</think>", a loop that no response has been seen to leave). Per set (`summary`): the five
exclusive outcomes (closed and ended, closed and at the limit, unclosed at the limit, unclosed and loop-stopped,
unclosed and ended; unclosed at the limit or loop-stopped, comparable with an unchunked probe's unclosed at the limit);
the closing-time curve P(closed with reasoning <= B) at BUDGETS; reasoning tokens p50/p90/p95 (an unclosed
response all reasoning) and answer tokens p50 given closure; the shares of closed answers holding a fenced code block
and ending with BOS; the evalplus responses' pass on EvalPlus's base and plus tests (evalplus.evaluate's
check_correctness on the answer as EvalPlus's codegen sanitizes it; an unclosed response's answer is empty), over all
and given closure; the val responses' outcomes by source and by teacher-length bucket (TEACHER_BUCKETS) and the rank
correlation of teacher length with closure; the never-closed responses' mean repetition (`repetition`). Each response
keeps its prompt's id, its full answer and the last TAIL_CHARS characters of its text, an evalplus one also its full
reasoning. Generation (`main`, with vLLM) writes the responses unscored; `score`, a separate CPU process (never one
forked from the engine's: EvalPlus's 4 GiB address-space guard fails large allocations there), adds the pass and the
summaries. Beside a training job (its metrics.jsonl and the probe's start time given), the job's train tokens/s during
the probe against its BEFORE steps before it and the seconds it lost, as information. VLLM_GPU_MEMORY_UTILIZATION
applies through eval/official_site/sitecustomize.py.

With --budget, s1's budget forcing (arXiv 2501.19393 §3.1), as a measure of what the reasoning holds when cut, and the
closure's sensitivity to sampling, in one engine session: phase 1 samples each prompt of both sets (same rendering and
seeds) under each of CONFIGS (the protocol's; T 0.8 and T 1.0; Qwen3's anti-repetition top-k 20 with presence penalty
1.0), stopped at END_THINK (string stop, kept) or at max(FORCED_BUDGETS) new tokens; phase 2, for each evalplus prompt
and each budget B (FORCED_BUDGETS under the protocol's sampling, max(FORCED_BUDGETS) under the others), appends to the
prompt the reasoning (a response closed within B tokens: its reasoning, encoded alone as corpus/posttrain/build.py
render encodes it; else its first B generated tokens) and END_THINK, and samples the answer (the config's sampling,
seed SEED + the prompt's index, at most ANSWER_TOKENS, eos = BOS; the answer after the last "</think>"). Per config
and set (`sampled_summary`): closure by max(FORCED_BUDGETS), the closing-time curve, the never-closed responses that
loop (`repetition` >= LOOP); per budget (`forced_summary`): naturally closed and forced counts, the base and plus pass
(`evalplus_check`) over all, the naturally closed and the forced, and the shares with a fenced code block and ending
with BOS. Each response keeps its full reasoning and answers.

With --onsets, each never-closed response's loop onset (`loop_onset`) in a probe's json, from the stored reasoning
(evalplus responses, and every --budget response) re-encoded as ordinary text; written to a sidecar json. With
--data-repetition, `repetition` of every SFT document's reasoning (train and val splits), the teacher's analogue of a
loop, counted per threshold and per reasoning-length bucket.

    python -m eval.official_termination <model dir> <out json> [<metrics.jsonl> <since, unix seconds>] [--prompts N]
        [--budget]
    python -m eval.official_termination --score <generation json> <out json>
    python -m eval.official_termination --onsets <probe json> <out json>
    python -m eval.official_termination --data-repetition <workers> <out json>
"""

import argparse
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
from eval.loop import CHUNK, END, LOOP, REPEAT_TAIL, REPEAT_WINDOW, repetition
from eval.official_tokenize import evalplus_tasks

PROMPTS, SEED, BEFORE = 64, 0, 50
SAMPLING = {"temperature": 0.6, "top_p": 0.95}
FENCE = re.compile(r"```[^\n]*\n.*?```", re.S)
BUDGETS = (1024, 2048, 4096, 8192, 16384, 32768)
TEACHER_BUCKETS = (2048, 4096, 8192, 16384)  # <2K, 2-4K, 4-8K, 8-16K, >=16K
OUTCOMES = ("closed_ended", "closed_limit", "unclosed_limit", "unclosed_loop_stopped", "unclosed_ended")
SEED_STRIDE = 1_000_003
TAIL_CHARS = 2000
FORCED_BUDGETS, ANSWER_TOKENS = (4096, 8192, 16384), 4096
CONFIGS = {"protocol": SAMPLING, "t0.8": {"temperature": 0.8, "top_p": 0.95},
           "t1.0": {"temperature": 1.0, "top_p": 0.95},
           "qwen3": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.0},
           # DRY (eval/dry.py) on the reasoning, the protocol's sampling
           "dry": SAMPLING | {"extra_args": {"dry": {}}}}
SENSITIVITY = ("protocol", "t0.8", "t1.0", "qwen3")


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


def loop_onset(ids: list[int]) -> int:
    """The first token of the repeating tail: one past the last token not inside a REPEAT_WINDOW-token window that
    recurs anywhere in `ids` (0 if every token is; len(ids) if the last token is not)."""
    windows = [tuple(ids[i:i + REPEAT_WINDOW]) for i in range(len(ids) - REPEAT_WINDOW + 1)]
    counts = Counter(windows)
    covered = np.zeros(len(ids), bool)
    for i, w in enumerate(windows):
        if counts[w] > 1:
            covered[i:i + REPEAT_WINDOW] = True
    free = np.flatnonzero(~covered)
    return int(free[-1]) + 1 if len(free) else 0


def onsets(record: Path, out: Path) -> None:
    """Loop onsets of a probe's never-closed responses whose reasoning the json stores; per set, the median."""
    report = json.loads(record.read_text())
    rows = report["responses"]
    configs = rows if report.get("budget") else {"": rows}
    enc = get_tokenizer().enc
    result = {}
    for config, sets in configs.items():
        for name, responses in sets.items():
            found = []
            for r in responses:
                if r["closed"] or not r.get("reasoning"):
                    continue
                ids = enc.encode_ordinary(r["reasoning"])
                found.append({"id": r["id"], "tokens": len(ids), "repetition": r["repetition"],
                              "onset": loop_onset(ids)})
            loops = [f["onset"] for f in found if (f["repetition"] or 0) >= LOOP]
            result[f"{config}/{name}" if config else name] = {
                "unclosed_with_text": len(found), "loops": len(loops),
                "onset_median": statistics.median(loops) if loops else None, "responses": found}
    out.write_text(json.dumps({"record": record.name, "window": REPEAT_WINDOW, "loop": LOOP, "sets": result},
                              indent=1) + "\n")


def document_repetition(task: tuple[str, int, int]) -> list[tuple[int, float]]:
    """(reasoning tokens, `repetition`) of documents [start, end) of an SFT split."""
    split, start, end = task
    tok = get_tokenizer()
    path = split_path(NAME, split, tok)
    tokens = np.memmap(path.with_suffix(".bin"), dtype=TOKEN_DTYPE, mode="r")
    offsets = np.memmap(path.with_suffix(".idx"), dtype=np.int64, mode="r")
    contexts = np.fromfile(path.with_suffix(".context"), dtype=np.int64)
    out = []
    for i in range(start, end):
        # the reasoning, encoded alone as build.py render encodes it
        text = tok.enc.decode(tokens[offsets[i] + contexts[i]:offsets[i + 1]].tolist()).partition(END_THINK)[0]
        ids = tok.enc.encode_ordinary(text)
        out.append((len(ids), repetition(ids)))
    return out


def data_repetition(workers: int, out: Path) -> None:
    """Counts of SFT documents by reasoning `repetition` (thresholds) and by reasoning-length bucket."""
    tok = get_tokenizer()
    tasks = []
    for split in ("train", "val"):
        n = len(np.fromfile(split_path(NAME, split, tok).with_suffix(".context"), dtype=np.int64))
        step = -(-n // (workers * 8))
        tasks += [(split, s, min(s + step, n)) for s in range(0, n, step)]
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(document_repetition, tasks))
    thresholds = (0.5, 0.8, 0.9, 0.99)
    report = {"tail": REPEAT_TAIL, "window": REPEAT_WINDOW, "splits": {}}
    for split in ("train", "val", "all"):
        rows = [r for (s, _, _), p in zip(tasks, parts) if split in (s, "all") for r in p]
        lengths = np.array([n for n, _ in rows])
        reps = np.array([r for _, r in rows])
        buckets = np.digitize(lengths, TEACHER_BUCKETS)
        report["splits"][split] = {
            "documents": len(rows),
            "at_least": {str(t): int((reps >= t).sum()) for t in thresholds},
            "by_length": [{"bucket": b, "documents": int((buckets == b).sum()),
                           "at_least": {str(t): int(((buckets == b) & (reps >= t)).sum()) for t in thresholds}}
                          for b in range(len(TEACHER_BUCKETS) + 1)]}
    out.write_text(json.dumps(report, indent=1) + "\n")


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
    outcome = ("closed" if closures else "unclosed") + (
        "_ended" if ended else "_loop_stopped" if finish == "loop" and not closures else "_limit")
    return {"tokens": n, "finish": finish, "outcome": outcome, "closed": closures > 0, "closures": closures,
            "ended": ended, "reasoning_tokens": reasoning, "answer_tokens": n - reasoning,
            "fenced": bool(FENCE.search(answer)), "answer": answer, "tail": text[-TAIL_CHARS:],
            "loop_stopped_at": n if finish == "loop" else None,
            "repetition": None if closures else repetition(ids[:n])}


def generate(llm, prompts: list[list[int]], sampling: list[dict], seeds: list[int], limits: list[int],
             stop: str | None = None) -> list[tuple[list[int], str, str]]:
    """Each prompt's (generated ids, their text, finish), sampled in CHUNK-token rounds, each round's prompt the
    prompt and every token so far (prefix caching reuses its KV): finish "stop" at eos (BOS) or at `stop` (text cut
    after its first occurrence), "length" at the prompt's limit, "loop" for a sequence holding neither `stop` nor END
    whose last REPEAT_TAIL tokens have `repetition` >= LOOP after a round. Round r is seeded seed + r * SEED_STRIDE,
    so the first CHUNK tokens equal a one-shot generation's and the rest differ from it (each request restarts its
    seeded generator): chunked and one-shot generations are equal in distribution, not token for token."""
    from vllm import SamplingParams

    tok = get_tokenizer()
    generated, finish = [[] for _ in prompts], [None] * len(prompts)
    active, chunk = list(range(len(prompts))), 0
    while active:
        outputs = llm.generate(
            [{"prompt_token_ids": prompts[k] + generated[k]} for k in active],
            [SamplingParams(**sampling[k], max_tokens=min(CHUNK, limits[k] - len(generated[k])),
                            seed=seeds[k] + chunk * SEED_STRIDE,
                            **({"stop": [stop], "include_stop_str_in_output": True} if stop else {}))
             for k in active], use_tqdm=False)
        remaining = []
        for k, output in zip(active, outputs, strict=True):
            (completion,) = output.outputs
            generated[k] += completion.token_ids
            text = tok.enc.decode([t for t in generated[k] if t != tok.bos_token_id])
            if completion.finish_reason == "stop" or (stop and stop in text):  # `stop` across a round's boundary
                finish[k] = "stop"
            elif len(generated[k]) >= limits[k]:
                finish[k] = "length"
            elif (len(generated[k]) >= REPEAT_TAIL and END not in text and (not stop or stop not in text)
                  and repetition(generated[k]) >= LOOP):
                finish[k] = "loop"
            else:
                remaining.append(k)
        active, chunk = remaining, chunk + 1
    out = []
    for ids, f in zip(generated, finish, strict=True):
        text = tok.enc.decode([t for t in ids if t != tok.bos_token_id])
        if stop and stop in text:
            text = text[:text.index(stop) + len(stop)]
        out.append((ids, text, f))
    return out


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
            # comparable with a one-shot probe's unclosed_limit
            "unclosed_limit_or_loop_stopped": counts["unclosed_limit"] + counts["unclosed_loop_stopped"],
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


def forced(llm, sets: dict[str, list[dict]], configs: list[str]) -> dict[str, dict[str, list[dict]]]:
    """Phase 1 (each config's reasoning, up to max(FORCED_BUDGETS) tokens) and phase 2 (the evalplus answers per
    budget) of --budget: {config: {set: rows}}."""
    from vllm import SamplingParams

    tok = get_tokenizer()
    flat = [(name, p) for name, every in sets.items() for p in every]
    runs = [(config, i, name, p) for config in configs for i, (name, p) in enumerate(flat)]
    thoughts = generate(llm, [p["ids"] for *_, p in runs], [CONFIGS[config] for config, *_ in runs],
                        [SEED + i for _, i, *_ in runs], [max(FORCED_BUDGETS)] * len(runs), stop=END_THINK)
    end_think = tok.enc.encode_ordinary(END_THINK)
    out = {config: {name: [] for name in sets} for config in configs}
    requests = []
    for (config, i, name, p), (generated, text, finish) in zip(runs, thoughts, strict=True):
        ids = [t for t in generated if t != tok.bos_token_id]
        closed = text.endswith(END_THINK)
        reasoning = tok.enc.encode_ordinary(text[:-len(END_THINK)]) if closed else ids
        row = {k: v for k, v in p.items() if k != "ids"} | {
            "prompt_tokens": len(p["ids"]), "closed": closed, "finish": finish, "reasoning_tokens": len(reasoning),
            "loop_stopped_at": len(ids) if finish == "loop" else None,
            "repetition": None if closed else repetition(ids), "reasoning": text, "budgets": {}}
        out[config][name].append(row)
        if name == "evalplus":
            for b in FORCED_BUDGETS if config == "protocol" else FORCED_BUDGETS[-1:]:
                natural = closed and len(reasoning) <= b
                requests.append((config, i, row, b, natural,
                                 [*p["ids"], *(reasoning if natural else ids[:b]), *end_think]))
    answers = llm.generate([{"prompt_token_ids": r[-1]} for r in requests],
                           [SamplingParams(**CONFIGS[config], max_tokens=ANSWER_TOKENS, seed=SEED + i)
                            for config, i, *_ in requests], use_tqdm=False)
    for (_, _, row, b, natural, _), output in zip(requests, answers, strict=True):
        (completion,) = output.outputs
        answer = completion.text.rpartition(END)[2].strip()
        row["budgets"][str(b)] = {"id": row["id"], "natural": natural, "ended": completion.finish_reason == "stop",
                                  "answer_tokens": len(completion.token_ids), "fenced": bool(FENCE.search(answer)),
                                  "answer": answer}
    return out


def forced_summary(rows: list[dict], b: int) -> dict:
    answers = [r["budgets"][str(b)] for r in rows]
    out = {"responses": len(answers)}
    for name, rs in (("all", answers), ("natural", [a for a in answers if a["natural"]]),
                     ("forced", [a for a in answers if not a["natural"]])):
        out[name] = {"responses": len(rs), "base": sum(a["base"] for a in rs), "plus": sum(a["plus"] for a in rs),
                     "fenced": sum(a["fenced"] for a in rs), "ended": sum(a["ended"] for a in rs)}
    return out


def sampled_summary(rows: list[dict]) -> dict:
    unclosed = [r for r in rows if not r["closed"]]
    out = {"responses": len(rows), "closed": sum(r["closed"] for r in rows),
           "closed_within": {str(b): sum(r["closed"] and r["reasoning_tokens"] <= b for r in rows)
                             for b in BUDGETS if b <= max(FORCED_BUDGETS)},
           "unclosed_ended": sum(r["finish"] == "stop" for r in unclosed),
           "unclosed_loops": sum(r["repetition"] >= LOOP for r in unclosed),
           "unclosed_loop_stopped": sum(r["finish"] == "loop" for r in unclosed)}
    if rows and rows[0]["budgets"]:
        out["budgets"] = {b: forced_summary(rows, int(b)) for b in rows[0]["budgets"]}
    return out


def main(model: Path, out: Path, metrics: Path | None, since: float | None, n: int, budget: bool,
         configs: list[str]) -> None:
    """Generation (the GPU process): the responses, unscored, with their full answers (and the evalplus set's full
    reasoning) to `out`; `score` scores them in a separate process."""
    from vllm import LLM, SamplingParams

    start = time.time()
    context = json.loads((model / "config.json").read_text())["max_position_embeddings"]
    processors = []
    if budget and "dry" in configs:
        from eval.dry import DryLogitsProcessor
        processors = [DryLogitsProcessor]
    llm = LLM(str(model), trust_remote_code=True, dtype="bfloat16", max_model_len=context, seed=SEED,
              enable_prefix_caching=True, logits_processors=processors)
    tok = get_tokenizer()
    report = {"model": str(model), "context": context, "sampling": SAMPLING | {"n": 1, "seed": SEED},
              "prompts_per_set": n, "budget": budget, "chunk": CHUNK, "loop_stop": LOOP}
    if budget:
        sets = {"evalplus": evalplus_prompts(llm.get_tokenizer(), n), "val": val_prompts(n)}
        generating = time.time()
        responses = forced(llm, sets, configs)
        end = time.time()
        report |= {"configs": {c: CONFIGS[c] for c in configs}, "reasoning_max_tokens": max(FORCED_BUDGETS),
                   "answer_max_tokens": ANSWER_TOKENS}
    else:
        sets = {"evalplus": evalplus_prompts(llm.get_tokenizer(), n), "val": val_prompts(n)}
        flat = [(name, p) for name, every in sets.items() for p in every]
        generating = time.time()
        outputs = generate(llm, [p["ids"] for _, p in flat], [SAMPLING] * len(flat),
                           [SEED + i for i in range(len(flat))], [context - len(p["ids"]) for _, p in flat])
        end = time.time()
        responses = {name: [] for name in sets}
        for (name, p), (ids, text, finish) in zip(flat, outputs, strict=True):
            responses[name].append(
                {k: v for k, v in p.items() if k != "ids"} | {"prompt_tokens": len(p["ids"])}
                | response(ids, text, finish, tok.bos_token_id, tok.enc.decode)
                | ({"reasoning": text.rpartition(END)[0] or text} if name == "evalplus" else {}))
    report |= {"seconds": round(time.time() - start), "generation_seconds": round(end - generating),
               "responses": responses}
    if metrics is not None:
        report["job_tokens_per_second"] = job_cost(metrics, start if since is None else since, end)
    out.write_text(json.dumps(report, indent=1) + "\n")


def score(generated: Path, out: Path) -> None:
    """Scoring (a CPU process apart from the engine's, whose CUDA state a forked check_correctness would inherit
    against EvalPlus's address-space limit): the evalplus answers' pass, then the summaries."""
    report = json.loads(generated.read_text())
    rows = report["responses"]
    if report["budget"]:
        evalplus_check([a for sets in rows.values() for r in sets["evalplus"] for a in r["budgets"].values()])
        report["summary"] = {config: {name: sampled_summary(r) for name, r in sets.items()}
                             for config, sets in rows.items()}
    else:
        evalplus_check(rows["evalplus"])
        report["sets"] = {name: summary(r) for name, r in rows.items()} | {"all": summary(sum(rows.values(), []))}
    report["responses"] = report.pop("responses")  # last, after the summaries
    out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path,
                        help="the model dir; with --score or --onsets, the probe's json; with --data-repetition, the "
                             "worker count")
    parser.add_argument("out", type=Path)
    parser.add_argument("metrics", type=Path, nargs="?")
    parser.add_argument("since", type=float, nargs="?")
    parser.add_argument("--prompts", type=int, default=PROMPTS)
    parser.add_argument("--budget", action="store_true")
    parser.add_argument("--configs", default=",".join(SENSITIVITY), help="with --budget, the samplings (CONFIGS)")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--onsets", action="store_true")
    parser.add_argument("--data-repetition", action="store_true")
    args = parser.parse_args()
    if args.score:
        score(args.model, args.out)
    elif args.onsets:
        onsets(args.model, args.out)
    elif args.data_repetition:
        data_repetition(int(str(args.model)), args.out)
    else:
        main(args.model, args.out, args.metrics, args.since, args.prompts, args.budget, args.configs.split(","))
