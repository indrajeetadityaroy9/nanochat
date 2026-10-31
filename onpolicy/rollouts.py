"""
The KodCode val evaluation's rollouts (docs/post-training.md §5): --prompts prompts of the val pool held out from the
train pool (onpolicy/prompts.py `held_out`, `sample`, seeded by --seed; only prompts whose context and --max-tokens
fit the model's context), each sampled under every (temperature, count) of --samples at top-p TOP_P, by vLLM offline
from a --chat export (inference/export_hf.py; bf16, the head in fp32, eos = BOS), through
eval/official_termination.py `generate`, comparable across models: CHUNK-token rounds with prefix caching, a response
stopped once its last REPEAT_TAIL tokens have repetition >= LOOP with no "</think>" (finish "loop"), else at BOS
("stop") or at min(--max-tokens, context - prompt) ("length"). Request k (prompts in id order, each prompt's samples
in --samples order) is seeded --seed + k. Each response's ending is eval/official_termination.py `response`'s
(closure, the reasoning through the last "</think>", the answer after it); a never-closed response stopped as a loop
or with repetition >= LOOP is a loop, and `cycle` gives its onset and period when its tail is exactly periodic (the
loops seen so far are: per-token log-probability ≈ -0.001 once established). `cycle` and `stopped` are also the RL
sampler's loop stop (onpolicy/sampler.py), checked every CHECK generated tokens.

Writes <out dir>/prompts.jsonl (the sampled pool rows with their ids and `source`), rollouts.parquet (one row per
response, its generated ids included) and meta.json (settings, the held-out counts, generation time and tokens).
VLLM_GPU_MEMORY_UTILIZATION applies through eval/official_site/sitecustomize.py.

    python -m onpolicy.rollouts <model dir> <out dir> [--prompts 512] [--samples 0.6:4] [--seed S]
        [--max-tokens 16384]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from vllm import LLM

from corpus.tokenizer import get_tokenizer
from eval.loop import END, looping
from eval.official_termination import CHUNK, LOOP, generate, response
from onpolicy.prompts import context_ids, held_out, read_pool, sample, usable

TOP_P = 0.95  # the reasoning protocol's (eval/official_termination.py SAMPLING)
# Rollout length: closed reasoning reached 11,757 tokens at most in the step-2000 probe (128 responses), and the
# thinking-budget probe samples to 16,384
MAX_TOKENS = 16384
# `cycle`: periods tried, and the tail that must repeat exactly at the period
MAX_PERIOD, CYCLE_TAIL = 2048, 1024
# The RL sampler's check interval in generated tokens (a divisor of CHUNK, so the protocol's checks fall on it): a
# stopped loop runs at most CHECK tokens past the point `cycle` finds it, against up to CHUNK between rounds
CHECK = 128
# The val evaluation: 4 samples per prompt at the reasoning protocol's T 0.6
SAMPLES = "0.6:4"


def cycle(ids: list[int]) -> tuple[int, int] | None:
    """(onset, period) of a response ending in an exact cycle: the smallest period p <= MAX_PERIOD at which its last
    CYCLE_TAIL tokens repeat (ids[i] == ids[i - p]), and the first index from which every token equals the one p
    later (the cycle's first copy starts there; its first repeat at onset + p). None if no period fits. Only the
    periods at which the last token recurs can fit, so only those are tried, in increasing order."""
    a = np.asarray(ids)
    n = len(a)
    if n < CYCLE_TAIL + 1:
        return None
    lo = max(n - 1 - MAX_PERIOD, 0)
    for p in (n - 1 - lo - np.flatnonzero(a[lo:n - 1] == a[n - 1]))[::-1].tolist():
        if n < CYCLE_TAIL + p:
            return None
        if np.array_equal(a[n - CYCLE_TAIL:], a[n - CYCLE_TAIL - p:n - p]):
            differ = np.flatnonzero(a[p:] != a[:-p])
            return (int(differ[-1]) + 1 if len(differ) else 0), p
    return None


def stopped(ids: list[int], decode) -> bool:
    """Whether the loop stop ends a response at these ids: an exact cycle or the protocol's rule at its own checks
    (eval/loop.py `looping`), and no END."""
    return (cycle(ids) is not None or looping(ids, decode) is not None) and END not in decode(ids)


def parse_samples(text: str) -> list[tuple[float, int]]:
    return [(float(t), int(k)) for t, k in (item.split(":") for item in text.split(","))]


def main(model: Path, out: Path, n: int, samples: list[tuple[float, int]], seed: int, max_tokens: int) -> None:
    start = time.time()
    tok = get_tokenizer()
    out.mkdir(parents=True, exist_ok=True)
    context = json.loads((model / "config.json").read_text())["max_position_embeddings"]
    pool, held = held_out(read_pool("val"), read_pool("train"))
    pool = [row for row in pool if 1 + row["context_tokens"] + max_tokens <= context]
    rows = sample(pool, min(n, sum(map(usable, pool))), seed)
    for row in rows:
        row["ids"] = context_ids(row)
        row["source"] = f"kodcode/{row['style']}"
    with open(out / "prompts.jsonl", "w") as f:
        f.writelines(json.dumps({k: v for k, v in row.items() if k != "r1_answer"}) + "\n" for row in rows)
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
    for k, ((row, j, t), (ids, _, finish)) in enumerate(zip(requests, outputs, strict=True)):
        r = response(ids, tok.enc.decode([i for i in ids if i != tok.bos_token_id]), finish, tok.bos_token_id,
                     tok.enc.decode)
        loop = not r["closed"] and (finish == "loop" or r["repetition"] >= LOOP)
        onset, period = (cycle(ids) if loop else None) or (None, None)
        records.append({"prompt_id": row["id"], "sample": j, "temperature": t, "seed": seed + k, "ids": list(ids),
                        "loop": loop, "loop_onset": onset, "loop_period": period}
                       | {key: r[key] for key in ("finish", "outcome", "closed", "ended", "tokens",
                                                  "reasoning_tokens", "answer_tokens", "answer", "repetition")})
    pq.write_table(pa.Table.from_pylist(records), out / "rollouts.parquet")
    generated = sum(len(r["ids"]) for r in records)
    meta = {"model": str(model), "held_out": held, "prompts": len(rows), "samples": samples, "top_p": TOP_P,
            "max_tokens": max_tokens, "seed": seed, "loop_stop": f"rounds of {CHUNK}, repetition >= {LOOP}",
            "responses": len(records), "generated_tokens": generated, "generation_seconds": round(end - generating),
            "generated_tokens_per_second": generated / (end - generating), "seconds": round(time.time() - start)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--prompts", type=int, default=512)
    parser.add_argument("--samples", type=parse_samples, default=parse_samples(SAMPLES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()
    main(args.model, args.out, args.prompts, args.samples, args.seed, args.max_tokens)
