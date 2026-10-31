"""
On-policy rollouts (docs/post-training.md §8): --prompts prompts of a pool (onpolicy/prompts.py; --split mix, the
mining pool of every source's SFT train contexts, `sample_mix`; train or val, KodCode's pools, `sample`, for the val
split only its rows held out from the train pool, prompts.py `held_out`; seeded by --seed; without the prompts of
each --exclude directory's prompts.jsonl, an earlier wave's, and without the prompts whose context and --max-tokens
exceed the model's context; at most the usable rows left), each sampled under every
(temperature, count) of --samples at top-p TOP_P, by vLLM offline from a --chat export (inference/export_hf.py; bf16,
the head in fp32, eos = BOS). Val rollouts go through eval/official_termination.py `generate`, comparable across
models: CHUNK-token rounds with prefix caching, a response stopped once its last REPEAT_TAIL tokens have repetition >=
LOOP with no "</think>" (finish "loop"), else at BOS ("stop") or at min(--max-tokens, context - prompt) ("length").
Mining rollouts (splits mix and train) are sampled in one engine call with the loop stop inside the engine
(`MiningLoopStop`, the served stop's mechanism, eval/official_loop_stop.py): every CHECK generated tokens, a request
whose output holds no "</think>" and ends in an exact cycle (`cycle`), or meets the protocol's rule at its checks, gets
BOS as its next token, read back as finish "loop" without it (`mined`). Their rows need a loop only through its first
repeat: the protocol's rule runs a loop ≈4,100 tokens past four repeats (29% of the step-7676 val rollouts' tokens),
and rounds re-prefill every sequence's context once the requests outnumber the KV cache. Request k (prompts in id
order, each prompt's samples in --samples order) is seeded --seed + k. Each response's ending is
eval/official_termination.py `response`'s (closure, the reasoning through the last "</think>", the answer after it);
a never-closed response stopped as a loop or with repetition >= LOOP is a loop, and `cycle` gives its onset and
period when its tail is exactly periodic (the loops seen so far are: per-token log-probability ≈ -0.001 once
established).

Writes <out dir>/prompts.jsonl (the sampled pool rows with their ids and `source`), rollouts.parquet (one row per
response, its generated ids included) and meta.json (settings, the prompts per source, the val split's held-out
counts, generation time and tokens, and beside a training job its tokens/s during generation,
eval/official_termination.py `job_cost`).
VLLM_GPU_MEMORY_UTILIZATION applies through eval/official_site/sitecustomize.py.

    python -m onpolicy.rollouts <model dir> <out dir> [--split mix|train|val] [--prompts N] [--samples 0.6:1]
        [--seed S] [--max-tokens 16384] [--exclude <dir> ...] [<metrics.jsonl> <since, unix seconds>]
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsContext

from corpus.posttrain.sft import END_THINK
from corpus.tokenizer import get_tokenizer
from eval.loop import END, looping
from eval.official_loop_stop import LoopStopLogitsProcessor
from eval.official_termination import ANSWER_TOKENS, CHUNK, LOOP, generate, job_cost, response
from onpolicy.prompts import context_ids, held_out, read_pool, sample, sample_mix, usable
from onpolicy.selection import cut, newline_ids

TOP_P = 0.95  # the reasoning protocol's (eval/official_termination.py SAMPLING)
# Rollout length: closed reasoning reached 11,757 tokens at most in the step-2000 probe (128 responses), and the
# thinking-budget probe samples to 16,384
MAX_TOKENS = 16384
# `cycle`: periods tried, and the tail that must repeat exactly at the period
MAX_PERIOD, CYCLE_TAIL = 2048, 1024
# MiningLoopStop's check interval in generated tokens (a divisor of CHUNK, so the protocol's checks fall on it): a
# stopped loop runs at most CHECK tokens past the point `cycle` finds it, against up to CHUNK between rounds
CHECK = 128
# Samples per prompt: one at the evaluation protocol's T 0.6, where the loops are (in the step-3000 smoke, 47 of 160
# responses looped at T 0.6 against 7 of 160 at T 1.0, docs/post-training.md §8.2); one completion per prompt as
# Antidoom generates (github.com/Liquid4All/antidoom @ bd6a126 src/antidoom/generate.py:85-86), so a wave's loops
# come from as many distinct prompts as it has rows
SAMPLES = "0.6:1"
# DPO waves (split train): DeLoop's chosen action is the first later action of the same rollout that breaks the loop
# (arXiv 2610.19376 §4.1). The export's loops never break within their rollout, nor when resampled from a loop's
# history (0 of 88 continuations escaped), but a commit does: END_THINK forced at the loop's cut (onpolicy/selection.py
# `cut`), as s1's budget forcing ends thinking (arXiv 2501.19393 §3.1), then the model's own answer passes KodCode's
# tests in 24.8% of 802 val loops cut at o + 2p. COMMITS commits per exact-cycle loop, as samples RESUMED +
# sample * COMMITS + m, seeded from RESUME_SEED (below SEED_STRIDE: no sample's seed in any round), at most
# ANSWER_TOKENS each (the forced answers' cap of eval/official_termination.py)
COMMITS, RESUMED, RESUME_SEED = 2, 100, 500_000


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
    """Whether the mining stop ends a response at these ids: an exact cycle or the protocol's rule at its own checks
    (eval/loop.py `looping`), and no END."""
    return (cycle(ids) is not None or looping(ids, decode) is not None) and END not in decode(ids)


class MiningLoopStop(LoopStopLogitsProcessor):
    """The mining rollouts' loop stop inside the engine (vLLM 0.32 model runner V2): at every CHECK generated tokens
    past CYCLE_TAIL, BOS as the next token of a request whose output `stopped` stops."""

    def apply(self, logits, ctx: LogitsContext):
        slots = ctx.idx_mapping_np
        seq_lens = ctx.seq_lens_upper_bound_np[:len(slots)]
        generated = seq_lens - self.req_states.prompt_len.np[slots]
        refilled = seq_lens >= self.req_states.prefill_len.np[slots]
        for i in ((generated > CYCLE_TAIL) & (generated % CHECK == 0) & refilled).nonzero()[0].tolist():
            slot, n = int(slots[i]), int(generated[i])
            if stopped(self.output(slot, n, ctx), self.decode):
                rows = (ctx.expanded_idx_mapping == slot).nonzero().squeeze(1)
                logits[rows] = float("-inf")
                logits[rows, self.bos] = 0.0
        return logits


def mined(llm, prompts: list[list[int]], sampling: list[dict], seeds: list[int], limits: list[int], decode,
          bos: int) -> list[tuple[list[int], str]]:
    """Mining rollouts in one engine call (MiningLoopStop active): each prompt's (generated ids, finish), a stopped
    response's forced BOS read back as finish "loop" and dropped."""
    from vllm import SamplingParams

    outputs = llm.generate([{"prompt_token_ids": p} for p in prompts],
                           [SamplingParams(**s, max_tokens=n, seed=k)
                            for s, k, n in zip(sampling, seeds, limits, strict=True)], use_tqdm=False)
    out = []
    for output in outputs:
        (completion,) = output.outputs
        ids = list(completion.token_ids)
        if (ids[-1:] == [bos] and len(ids) - 1 > CYCLE_TAIL and (len(ids) - 1) % CHECK == 0
                and stopped(ids[:-1], decode)):
            out.append((ids[:-1], "loop"))
        else:
            out.append((ids, "stop" if completion.finish_reason == "stop" else "length"))
    return out


def parse_samples(text: str) -> list[tuple[float, int]]:
    return [(float(t), int(k)) for t, k in (item.split(":") for item in text.split(","))]


def main(model: Path, out: Path, split: str, n: int, samples: list[tuple[float, int]], seed: int, max_tokens: int,
         exclude: list[Path], metrics: Path | None, since: float | None) -> None:
    from vllm import LLM

    start = time.time()
    tok = get_tokenizer()
    out.mkdir(parents=True, exist_ok=True)
    context = json.loads((model / "config.json").read_text())["max_position_embeddings"]
    pool, held = read_pool(split), None
    if split == "val":
        pool, held = held_out(pool, read_pool("train"))
    seen = {json.loads(line)["id"] for d in exclude for line in (d / "prompts.jsonl").read_text().splitlines()}
    pool = [row for row in pool if row["id"] not in seen and 1 + row["context_tokens"] + max_tokens <= context]
    if split == "mix":
        rows = sample_mix(pool, min(n, len(pool)), seed)
    else:
        rows = sample(pool, min(n, sum(map(usable, pool))), seed)
    for row in rows:
        row["ids"] = context_ids(row)
        row["source"] = row.get("source") or f"kodcode/{row['style']}"
    with open(out / "prompts.jsonl", "w") as f:
        f.writelines(json.dumps({k: v for k, v in row.items() if k != "r1_answer"}) + "\n" for row in rows)
    llm = LLM(str(model), trust_remote_code=True, dtype="bfloat16", max_model_len=context, seed=seed,
              enable_prefix_caching=True, logits_processors=None if split == "val" else [MiningLoopStop])
    requests = [(row, j, t) for row in rows for j, t in enumerate(t for t, k in samples for _ in range(k))]
    generating = time.time()
    prompts = [row["ids"] for row, _, _ in requests]
    sampling = [{"temperature": t, "top_p": TOP_P} for _, _, t in requests]
    seeds = [seed + k for k in range(len(requests))]
    limits = [min(max_tokens, context - len(row["ids"])) for row, _, _ in requests]
    if split == "val":
        outputs = [(ids, finish) for ids, _, finish in generate(llm, prompts, sampling, seeds, limits)]
    else:
        outputs = mined(llm, prompts, sampling, seeds, limits, tok.enc.decode, tok.bos_token_id)

    def record(row: dict, j: int, t: float, s: int, ids: list[int], finish: str, resume_of: int | None = None,
               resume_from: int | None = None) -> dict:
        r = response(ids, tok.enc.decode([i for i in ids if i != tok.bos_token_id]), finish, tok.bos_token_id,
                     tok.enc.decode)
        loop = not r["closed"] and (finish == "loop" or r["repetition"] >= LOOP)
        onset, period = (cycle(ids) if loop else None) or (None, None)
        return ({"prompt_id": row["id"], "sample": j, "temperature": t, "seed": s, "ids": list(ids), "loop": loop,
                 "loop_onset": onset, "loop_period": period, "resume_of": resume_of, "resume_from": resume_from}
                | {key: r[key] for key in ("finish", "outcome", "closed", "ended", "tokens", "reasoning_tokens",
                                           "answer_tokens", "answer", "repetition")})

    records = [record(row, j, t, seed + k, ids, finish)
               for k, ((row, j, t), (ids, finish)) in enumerate(zip(requests, outputs, strict=True))]
    if split == "train":
        by_id = {row["id"]: row for row in rows}
        end_think, newline = tok.enc.encode_ordinary(END_THINK), newline_ids()
        cuts = [cut(r["ids"], r["loop_onset"], r["loop_period"], newline) if r["loop_period"] is not None else None
                for r in records]
        loops = [(r, m) for r, c in zip(records, cuts, strict=True) if c is not None for m in range(COMMITS)]
        at = {(r["prompt_id"], r["sample"]): c for r, c in zip(records, cuts, strict=True) if c is not None}
        prefixes = [r["ids"][:at[(r["prompt_id"], r["sample"])]] + end_think for r, _ in loops]
        committed = mined(llm, [by_id[r["prompt_id"]]["ids"] + p for (r, _), p in zip(loops, prefixes, strict=True)],
                          [{"temperature": r["temperature"], "top_p": TOP_P} for r, _ in loops],
                          [seed + RESUME_SEED + n for n in range(len(loops))],
                          [min(ANSWER_TOKENS, context - len(by_id[r["prompt_id"]]["ids"]) - len(p))
                           for (r, _), p in zip(loops, prefixes, strict=True)], tok.enc.decode, tok.bos_token_id)
        records += [record(by_id[r["prompt_id"]], RESUMED + r["sample"] * COMMITS + m, r["temperature"],
                           seed + RESUME_SEED + n, p + list(ids), finish, r["sample"], len(p) - len(end_think))
                    for n, (((r, m), p), (ids, finish))
                    in enumerate(zip(zip(loops, prefixes, strict=True), committed, strict=True))]
    end = time.time()
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(records), out / "rollouts.parquet")
    generated = sum(len(r["ids"]) - (r["resume_from"] or 0) for r in records)
    meta = {"model": str(model), "split": split, "held_out": held, "excluded": len(seen), "prompts": len(rows),
            "sources": dict(sorted(Counter(row["source"] for row in rows).items())), "samples": samples,
            "top_p": TOP_P, "max_tokens": max_tokens, "seed": seed,
            "loop_stop": (f"rounds of {CHUNK}, repetition >= {LOOP}" if split == "val"
                          else f"in-engine every {CHECK}: exact cycle or repetition >= {LOOP}"),
            "responses": len(records),
            "commits": sum(r["resume_of"] is not None for r in records),
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
    parser.add_argument("--split", default="mix", choices=("mix", "train", "val"))
    parser.add_argument("--prompts", type=int, default=500)
    parser.add_argument("--samples", type=parse_samples, default=parse_samples(SAMPLES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    main(args.model, args.out, args.split, args.prompts, args.samples, args.seed, args.max_tokens, args.exclude,
         args.metrics, args.since)
