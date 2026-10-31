"""
The RL stage's sampler (docs/post-training.md §6), in the vLLM image beside the trainer (onpolicy/rl.py, the training
image) on the same GPU: for each policy the trainer publishes, its evaluation if it is the first or the trainer's last,
then its batch of rollouts, judged.

Rollouts: PROMPTS train-pool prompts (onpolicy/prompts.py, usable rows, in a SEED permutation of the pool, batch k
the k-th slice) x GROUP samples, Lite PPO's batch (arXiv 2508.08221 §3: 128 prompts x 8, 1,024 responses), at T 1.0
and top-p 1.0, sampling the trainer's distribution exactly (DAPO samples at 1.0, arXiv 2503.14476 §4.1; ProRL's
rollouts at T 0.6 destabilised training against 1.2, arXiv 2507.12507 §5.1), at most MAX_TOKENS each: the SFT
model's first T 1.0 batch averaged 7,582 tokens a response and 3.6% (37 of 1,024) reached the limit, where an 8,192
limit cut 42% (truncation above 10-15% destabilizes training, arXiv 2510.13786 App. A.15); the trainer masks the cut
ones (onpolicy/rl.py).

Engines: ENGINES vLLM engines (one: with a batch's 1,024 requests in flight it holds the GPU at 100% utilization and
660 W), each driven by a thread through LLMEngine.step with token deltas, a group's samples on one engine (its
prompt's KV shared by prefix caching). Every CHECK generated tokens past CYCLE_TAIL, a response the mining stop catches
(onpolicy/rollouts.py `stopped`: an exact cycle, or the protocol's rule at its checks; no END) is aborted there,
finish "loop", in the driver thread rather than inside the engine (no device read in the engine's loop). Loops hold
68.9% of the SFT val rollouts' tokens (eval-model/val: 7.07M of 10.25M), and stopping each exact cycle at its first
detectable point removes 23.0% of the tokens and 38.6% of the KV reads; no verdict changes, as no loop escapes (0 of 88
resampled from loop histories). Each finished response is judged at once by a process pool (onpolicy/verify.py
`judge`; a prompt whose stub passes is trivial, its group unused), so the tests run while generation continues.

Evaluation: the 512 held-out KodCode val prompts of eval-model/val (the SFT model's pass@1 0.325 there), 4 samples
each at the protocol's T 0.6, top-p 0.95 (eval/official_termination.py SAMPLING), at most MAX_TOKENS as there, seeds
as onpolicy/rollouts.py gives them (0 + the request's index), the same loop stop: pass@1 over the 2,048, a trivial
prompt's passes not counted. Every batch's prompts are new, so its reward already estimates the policy on held-out
problems; the evaluation is for comparing with the SFT model and choosing the checkpoint.

Exchange with the trainer under X (each written under a temporary name, then renamed):
  X/policy-<k>/       the policy after k updates in the export's format (the trainer's); loaded by every engine
  X/batch-<k>.pkl     the rollouts of policy k: {"step", "groups": [{"prompt_id", "prompt" (ids), "trivial",
                      "samples": [{"ids" (generated, a loop's through its check), "finish", "verdict"}]}], "stats"}
  X/final             the step of the trainer's last policy, written before that policy is published
Evaluations append to <out>/evals.jsonl (a step already there is not evaluated again), each evaluated policy copied to
<out>/step-<k>/. The sampler starts at the first policy published and ends after the last one's evaluation.

    python -m onpolicy.sampler <model dir> <exchange dir> <out dir>
"""
import argparse
import json
import os
import pickle
import shutil
import threading
import time
from collections import Counter
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from corpus.tokenizer import get_tokenizer
from eval.official_termination import SAMPLING
from onpolicy.prompts import context_ids, read_pool, usable
from onpolicy.rollouts import CHECK, CYCLE_TAIL, MAX_TOKENS, stopped
from onpolicy.verify import check, judge, stub

PROMPTS, GROUP, TEMPERATURE, TOP_P, SEED = 128, 8, 1.0, 1.0, 0
VAL_SAMPLES = 4
# a batch's request seeds: BATCH_SEED + k * PROMPTS * GROUP + its index, clear of the evaluation's 0..2047
BATCH_SEED = 10_000_000
# one engine draws 660 W at 100% utilization on batch 0 (1,024 requests), so a second adds no GPU work; half the
# memory leaves the trainer its packed 32K-token rows
ENGINES, UTILIZATION, MAX_SEQS = 1, 0.5, 1024


class Engines:
    """`n` vLLM engines sharing `utilization` of the GPU's memory, each running at most `max_seqs` requests."""

    def __init__(self, model: Path, n: int = ENGINES, utilization: float = UTILIZATION, max_seqs: int = MAX_SEQS):
        from vllm import LLM

        self.context = json.loads((model / "config.json").read_text())["max_position_embeddings"]
        self.llms = [LLM(str(model), trust_remote_code=True, dtype="bfloat16", max_model_len=self.context, seed=SEED,
                         enable_prefix_caching=True, gpu_memory_utilization=utilization / n, max_num_seqs=max_seqs)
                     for _ in range(n)]
        self.decode = get_tokenizer().enc.decode

    def load(self, weights: Path) -> None:
        """Every engine's weights in place from a directory in the export's format; their prefix caches dropped."""
        for llm in self.llms:
            llm.collective_rpc("reload_weights", kwargs={"weights_path": str(weights)})
            llm.reset_prefix_cache()

    def sample(self, prompts: list[list[int]], sampling: list[dict], groups: list[int], done) -> list[tuple]:
        """Each request's (generated ids, finish "stop" | "length" | "loop"), group g's requests on engine g mod n;
        done(k, ids, finish) called from a driver thread as request k finishes."""
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        out: list = [None] * len(prompts)
        failed: list[BaseException] = []

        def finish(k: int, ids: list[int], how: str) -> None:
            out[k] = (ids, how)
            done(k, ids, how)

        def drive(llm, ks: list[int]) -> None:
            try:
                engine, ids, due = llm.llm_engine, {k: [] for k in ks}, dict.fromkeys(ks, CYCLE_TAIL + CHECK)
                for k in ks:
                    engine.add_request(str(k), {"prompt_token_ids": prompts[k]},
                                       SamplingParams(**sampling[k], output_kind=RequestOutputKind.DELTA,
                                                      detokenize=False))
                while engine.has_unfinished_requests():
                    for o in engine.step():
                        k = int(o.request_id)
                        if out[k] is not None:  # aborted
                            continue
                        ids[k] += o.outputs[0].token_ids
                        if o.finished:
                            finish(k, ids[k], "stop" if o.outputs[0].finish_reason == "stop" else "length")
                        elif len(ids[k]) >= due[k]:
                            m = len(ids[k]) // CHECK * CHECK
                            due[k] = m + CHECK
                            if stopped(ids[k][:m], self.decode):
                                engine.abort_request([str(k)])
                                finish(k, ids[k][:m], "loop")
            except BaseException as e:  # re-raised by the caller
                failed.append(e)

        threads = [threading.Thread(target=drive, args=(llm, [k for k, g in enumerate(groups) if g % len(self.llms)
                                                               == e]), daemon=True)
                   for e, llm in enumerate(self.llms)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if failed:
            raise failed[0]
        return out


def rollouts(engines: Engines, judges: ProcessPoolExecutor, rows: list[dict], samples: int, sampling: dict,
             seed: int, max_tokens: int) -> tuple[list[dict], dict]:
    """rows' groups of `samples` judged responses (the batch file's "groups") and the generation's stats; a row's
    prompt its "ids" (eval-model/val's rows) or its SFT context."""
    prompts = [row.get("ids") or context_ids(row) for row in rows]
    context = engines.context
    stubs = [judges.submit(check, (stub(row), row["test"], row["kind"])) for row in rows]
    requests = [(g, j) for g in range(len(rows)) for j in range(samples)]
    verdicts: dict[int, Future] = {}

    def done(k: int, ids: list[int], how: str) -> None:
        row = rows[requests[k][0]]
        verdicts[k] = judges.submit(judge, (ids, row["names"], row["test"], row["kind"]))

    start = time.time()
    outputs = engines.sample([prompts[g] for g, _ in requests],
                             [sampling | {"seed": seed + k, "max_tokens": min(max_tokens, context - len(prompts[g]))}
                              for k, (g, _) in enumerate(requests)], [g for g, _ in requests], done)
    generated = time.time()
    groups = [{"prompt_id": row["id"], "prompt": prompts[g], "trivial": stubs[g].result() == "pass", "samples": []}
              for g, row in enumerate(rows)]
    for k, ((g, _), (ids, how)) in enumerate(zip(requests, outputs, strict=True)):
        groups[g]["samples"].append({"ids": ids, "finish": how, "verdict": verdicts[k].result()})
    tokens = sum(len(ids) for ids, _ in outputs)
    return groups, {"responses": len(outputs), "generated_tokens": tokens, "generation_seconds": generated - start,
                    "judging_seconds": time.time() - generated, "tokens_per_second": tokens / (generated - start),
                    "finishes": dict(Counter(how for _, how in outputs))}


def evaluate(engines: Engines, judges: ProcessPoolExecutor, val: list[dict], step: int) -> dict:
    groups, stats = rollouts(engines, judges, val, VAL_SAMPLES, SAMPLING, 0, MAX_TOKENS)
    verdicts = Counter("pass_trivial" if g["trivial"] and s["verdict"] == "pass" else s["verdict"]
                       for g in groups for s in g["samples"])
    return {"step": step, "pass@1": verdicts["pass"] / stats["responses"], "verdicts": dict(verdicts),
            "loops": stats["finishes"].get("loop", 0),
            "mean_tokens": stats["generated_tokens"] / stats["responses"]} | stats


def wait(path: Path) -> None:
    while not path.exists():
        time.sleep(0.2)


def main(model: Path, exchange: Path, out: Path) -> None:
    context = json.loads((model / "config.json").read_text())["max_position_embeddings"]
    pool = [row for row in read_pool("train") if usable(row) and 1 + row["context_tokens"] + MAX_TOKENS <= context]
    order = np.random.RandomState(SEED).permutation(len(pool))
    val = [json.loads(line) for line in (out.parent / "eval-model" / "val" / "prompts.jsonl").read_text().splitlines()]
    engines = Engines(model)
    while not (present := [int(p.name[7:]) for p in exchange.glob("policy-*") if p.name[7:].isdigit()]):
        time.sleep(0.2)
    k = min(present)
    workers = len(os.sched_getaffinity(0)) - ENGINES - 4
    log = out / "evals.jsonl"
    evaluated = {json.loads(line)["step"] for line in log.read_text().splitlines()} if log.exists() else set()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as judges:
        while True:
            wait(exchange / f"policy-{k}")
            engines.load(exchange / f"policy-{k}")
            last = (exchange / "final").exists() and int((exchange / "final").read_text()) == k
            if (k == 0 or last) and k not in evaluated:
                with open(log, "a") as f:
                    f.write(json.dumps(evaluate(engines, judges, val, k)) + "\n")
                shutil.copytree(exchange / f"policy-{k}", out / f"step-{k}", dirs_exist_ok=True)
            if last:
                return
            rows = [pool[i] for i in order[k * PROMPTS:(k + 1) * PROMPTS]]
            groups, stats = rollouts(engines, judges, rows, GROUP, {"temperature": TEMPERATURE, "top_p": TOP_P},
                                     BATCH_SEED + k * PROMPTS * GROUP, MAX_TOKENS)
            tmp = exchange / f"batch-{k}.pkl.tmp"
            tmp.write_bytes(pickle.dumps({"step": k, "groups": groups, "stats": stats}, protocol=5))
            tmp.rename(exchange / f"batch-{k}.pkl")
            k += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("exchange", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    main(args.model, args.exchange, args.out)
