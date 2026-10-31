"""
On-policy rollouts (docs/post-training.md §8): --prompts prompts of a pool (onpolicy/prompts.py `sample`, seeded by
--seed), each sampled under every (temperature, count) of --samples at top-p TOP_P, by vLLM offline from a --chat
export (inference/export_hf.py; bf16, the head in fp32, eos = BOS) through eval/official_termination.py `generate`:
CHUNK-token rounds with prefix caching, a response stopped once its last REPEAT_TAIL tokens have repetition >= LOOP
with no "</think>" (finish "loop"), else at BOS ("stop") or at min(--max-tokens, context - prompt) ("length"). Request
k (prompts in id order, each prompt's samples in --samples order) is seeded --seed + k. Each response's ending is
eval/official_termination.py `response`'s (closure, the reasoning through the last "</think>", the answer after it);
a never-closed response with repetition >= LOOP is a loop, and `cycle` gives its onset and period when its tail is
exactly periodic (the loops seen so far are: per-token log-probability ≈ -0.001 once established).

Writes <out dir>/prompts.jsonl (the sampled pool rows with their ids), rollouts.parquet (one row per response, its
generated ids included) and meta.json (settings, generation time and tokens, and beside a training job its tokens/s
during generation, eval/official_termination.py `job_cost`). VLLM_GPU_MEMORY_UTILIZATION applies through
eval/official_site/sitecustomize.py.

    python -m onpolicy.rollouts <model dir> <out dir> [--split train] [--prompts N] [--samples 0.6:4,1.0:4]
        [--seed S] [--max-tokens 16384] [<metrics.jsonl> <since, unix seconds>]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from corpus.tokenizer import get_tokenizer
from eval.official_termination import CHUNK, LOOP, generate, job_cost, response
from onpolicy.prompts import context_ids, read_pool, sample

TOP_P = 0.95  # the reasoning protocol's (eval/official_termination.py SAMPLING)
# Rollout length: closed reasoning reached 11,757 tokens at most in the step-2000 probe (128 responses), and the
# thinking-budget probe samples to 16,384
MAX_TOKENS = 16384
# `cycle`: periods tried, and the tail that must repeat exactly at the period
MAX_PERIOD, CYCLE_TAIL = 2048, 1024


def cycle(ids: list[int]) -> tuple[int, int] | None:
    """(onset, period) of a response ending in an exact cycle: the smallest period p <= MAX_PERIOD at which its last
    CYCLE_TAIL tokens repeat (ids[i] == ids[i - p]), and the first index from which every token equals the one p
    later (the cycle's first copy starts there; its first repeat at onset + p). None if no period fits."""
    a = np.asarray(ids)
    n = len(a)
    for p in range(1, MAX_PERIOD + 1):
        if n < CYCLE_TAIL + p:
            return None
        if np.array_equal(a[n - CYCLE_TAIL:], a[n - CYCLE_TAIL - p:n - p]):
            differ = np.flatnonzero(a[p:] != a[:-p])
            return (int(differ[-1]) + 1 if len(differ) else 0), p
    return None


def parse_samples(text: str) -> list[tuple[float, int]]:
    return [(float(t), int(k)) for t, k in (item.split(":") for item in text.split(","))]


def main(model: Path, out: Path, split: str, n: int, samples: list[tuple[float, int]], seed: int, max_tokens: int,
         metrics: Path | None, since: float | None) -> None:
    from vllm import LLM

    start = time.time()
    tok = get_tokenizer()
    out.mkdir(parents=True, exist_ok=True)
    rows = sample(read_pool(split), n, seed)
    for row in rows:
        row["ids"] = context_ids(row)
    with open(out / "prompts.jsonl", "w") as f:
        f.writelines(json.dumps({k: v for k, v in row.items() if k != "r1_answer"}) + "\n" for row in rows)
    context = json.loads((model / "config.json").read_text())["max_position_embeddings"]
    llm = LLM(str(model), trust_remote_code=True, dtype="bfloat16", max_model_len=context, seed=seed,
              enable_prefix_caching=True)
    requests = [(row, j, t) for row in rows for j, t in enumerate(t for t, k in samples for _ in range(k))]
    generating = time.time()
    outputs = generate(llm, [row["ids"] for row, _, _ in requests],
                       [{"temperature": t, "top_p": TOP_P} for _, _, t in requests],
                       [seed + k for k in range(len(requests))],
                       [min(max_tokens, context - len(row["ids"])) for row, _, _ in requests])
    end = time.time()
    records = []
    for k, ((row, j, t), (ids, text, finish)) in enumerate(zip(requests, outputs, strict=True)):
        r = response(ids, text, finish, tok.bos_token_id, tok.enc.decode)
        loop = not r["closed"] and r["repetition"] >= LOOP
        onset, period = (cycle(ids) if loop else None) or (None, None)
        records.append({"prompt_id": row["id"], "sample": j, "temperature": t, "seed": seed + k, "ids": list(ids),
                        "loop": loop, "loop_onset": onset, "loop_period": period}
                       | {key: r[key] for key in ("finish", "outcome", "closed", "ended", "tokens", "reasoning_tokens",
                                                  "answer_tokens", "answer", "repetition")})
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(records), out / "rollouts.parquet")
    generated = sum(len(r["ids"]) for r in records)
    meta = {"model": str(model), "split": split, "prompts": len(rows), "samples": samples, "top_p": TOP_P,
            "max_tokens": max_tokens, "seed": seed, "chunk": CHUNK, "loop_stop": LOOP, "responses": len(records),
            "generated_tokens": generated, "generation_seconds": round(end - generating),
            "generated_tokens_per_second": generated / (end - generating), "seconds": round(time.time() - start)}
    if metrics is not None:
        meta["job_tokens_per_second"] = job_cost(metrics, start if since is None else since, end)
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("metrics", type=Path, nargs="?")
    parser.add_argument("since", type=float, nargs="?")
    parser.add_argument("--split", default="train")
    parser.add_argument("--prompts", type=int, default=500)
    parser.add_argument("--samples", type=parse_samples, default=parse_samples("0.6:4,1.0:4"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()
    main(args.model, args.out, args.split, args.prompts, args.samples, args.seed, args.max_tokens, args.metrics,
         args.since)
