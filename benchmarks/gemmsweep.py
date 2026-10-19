"""Time every GEMM role of ops/quack_gemm.py on real operands: each role's first call in one compiled forward + backward
of a DEPTH model on the file stage's first fim micro-batch (M = config.MICRO_BATCH_TOKENS), operands copied at the call
(the compiled graph reuses its buffers). The model: init_weights(seed=0), then two optimizer steps on that micro-batch at
profile_step.py's settings (the first moves lm_head, the second the zero-init c_o and c_proj off zero, so every operand
of the third backward is nonzero).
Candidates: quack.gemm_config.get_all_configs() at this compute capability (the autotuner's list), minus configs the call
rejects; cuBLAS torch.mm for roles with a scalar alpha; for the fp32-out weight-gradient roles every tile x serial
split-K in SPLITS, with rel L2 error vs fp64 on the same bf16 inputs.
Timing: median over ROUNDS interleaved, rotated rounds; each a CUDA graph of >= TARGET_MS of calls.
gemmsweep_sm90.jsonl: ROUNDS 5, TARGET_MS 30, C 256 and 768, random bf16 operands (an earlier version of this script).
Usage: python -m benchmarks.gemmsweep OUT.jsonl ROUNDS TARGET_MS DEPTH [DEPTH ...]
Output lines: {"M", "C", "role", "config" (GemmConfig fields or "cublas"), "ms" (per round), "err" (dW roles)}."""
import concurrent.futures
import dataclasses
import json
import math
import statistics
import sys
import time

import numpy as np
import torch
from quack.cache.async_compile import CompilePending, pool_scope
from quack.gemm_config import get_all_configs
from quack.gemm_interface import gemm_tuned, prune_invalid_gemm_configs

from nanochat.config import FILE_STAGE
from nanochat.data.loader import PretrainingBatches, mixture_weights, packed_rows
from nanochat.data.tokenizer import get_tokenizer
from nanochat.model.gpt import GPT, gpt_config
from nanochat.ops import cross_entropy, linear
from nanochat.train.optim import optimizers
from nanochat.train.schedule import plan
from scripts.base_train import Variant, optimizer_settings, optimizer_step

SPLITS = (1, 2, 4, 8, 16)  # 1 and quack.gemm_interface._expand_split_k_configs's (2, 4, 8, 16)
# QuACK's errors for a config the call does not support
REJECTED = (RuntimeError, AssertionError, NotImplementedError, ValueError, TypeError)


def plain(A, B, *, out_dtype, alpha):
    """(run(config), candidates) of quack_gemm.gemm(A, B): gemm_tuned.fn at each config; cuBLAS unless alpha is a
    tensor."""
    out = torch.empty(A.shape[0], B.shape[1], dtype=out_dtype, device=A.device)
    cands = [c.kwargs["config"] for c in prune_invalid_gemm_configs(gemm_tuned.configs, dict(A=A, B=B, out=out),
                                                                    alpha=alpha, split_k=1)]
    cands += [] if isinstance(alpha, torch.Tensor) else ["cublas"]

    def run(cfg):
        if cfg == "cublas":
            return torch.mm(A, B, out_dtype=out.dtype)
        gemm_tuned.fn(A, B, out, alpha=alpha, config=cfg, split_k=cfg.split_k)
        return out
    return run, cands


def weight_grad(A, B, *, alpha):
    """plain at every split-K, and the fp64 reference alpha A B summed over K in chunks of min(m, n) (each temporary at
    most the accumulator's size)."""
    run, cands = plain(A, B, out_dtype=torch.float32, alpha=alpha)
    tiles = [c for c in cands if c != "cublas"]
    cands = [dataclasses.replace(c, split_k=s) for c in tiles for s in SPLITS] + cands[len(tiles):]  # then cuBLAS
    chunk = min(A.shape[0], B.shape[1])

    def ref():
        acc = torch.zeros(A.shape[0], B.shape[1], dtype=torch.float64, device=A.device)
        for k in range(0, A.shape[1], chunk):
            acc += A[:, k:k + chunk].double() @ B[k:k + chunk].double()
        return acc * float(alpha)
    return run, cands, ref


def roles(depth, device_configs):
    """(C, M, {role: (run(config), candidates, fp64 reference or None)}) from the model's first call of each role."""
    device = torch.device("cuda")
    tokenizer = get_tokenizer()
    T = FILE_STAGE.seq_len
    mixture = mixture_weights(FILE_STAGE.mixture, tokenizer, T)
    orders = {c: np.random.RandomState(0).permutation(packed_rows(c, "train", tokenizer, T))  # fixed seed
              for c in mixture}
    batch = next(iter(PretrainingBatches(mixture, "train", tokenizer, orders, seq_len=T,
                                         batch_rows=FILE_STAGE.batch_rows, device=device, reading="fim", start=0)))
    with torch.device("meta"):
        orig = GPT(gpt_config(depth=depth, stage=FILE_STAGE))
    orig.to_empty(device=device)
    orig.init_weights(seed=0)
    model = torch.compile(orig, dynamic=False, fullgraph=True)
    lr, weight_decay, max_norm = optimizer_settings(plan(depth=depth, ratio=20, tokens=None), Variant())
    opts = optimizers(orig, lr=lr, weight_decay=weight_decay)
    for _ in range(2):
        optimizer_step(model, [batch], list(orig.parameters()), opts, None, max_norm=max_norm)
    del opts
    calls = {}  # role -> (function, args, kwargs) of its first call
    epilogue_roles = {"gemm_act": "mlp_up", "gemm_dact": "mlp_dact"}
    originals = [(module, name, getattr(module, name)) for module, name in
                 ((linear, "gemm"), (linear, "gemm_act"), (linear, "gemm_dact"), (cross_entropy, "gemm"))]

    def copy(a):
        return a.clone() if isinstance(a, torch.Tensor) else a
    for module, name, fn in originals:
        def call(*args, fn=fn, name=name, **kwargs):
            role = kwargs["role"] if name == "gemm" else epilogue_roles[name]
            if role not in calls:
                calls[role] = (fn, [copy(a) for a in args], {k: copy(v) for k, v in kwargs.items()})
            return fn(*args, **kwargs)
        setattr(module, name, call)
    model(*batch).backward()
    for module, name, fn in originals:
        setattr(module, name, fn)
    rs = {}
    for role, (fn, args, kwargs) in calls.items():
        if role in epilogue_roles.values():
            rs[role] = (lambda cfg, fn=fn, args=args, kwargs=kwargs: fn(*args, **kwargs | {"config": cfg}),
                        device_configs, None)
        else:
            A, B = args
            rs[role] = (weight_grad(A, B, alpha=kwargs["alpha"]) if kwargs["out_dtype"] == torch.float32
                        else (*plain(A, B, out_dtype=kwargs["out_dtype"], alpha=kwargs["alpha"]), None))
    return orig.config.n_embd, batch[0].numel(), rs


def compile_all(rs):
    """Compile every QuACK candidate in QuACK's pool; rejected configs are reported when timed."""
    t = time.time()
    with pool_scope() as pool:
        for f, cands, _ in rs.values():
            for c in cands:
                if c != "cublas":
                    try:
                        f(c)
                    except (CompilePending, *REJECTED):
                        pass
        concurrent.futures.wait(pool._futures.values())
    torch.cuda.synchronize()
    print(f"compile/load {time.time() - t:.1f} s", flush=True)


def graph_ms(f, cfg, *, calls):
    """ms per call: one replay of a CUDA graph of `calls` calls of f(cfg)."""
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            f(cfg)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / calls


def main():
    out_path, rounds, target_ms, depths = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4:]
    capacity = torch.cuda.get_device_capability()[0]
    device_configs = [c for c in get_all_configs() if c.device_capacity == capacity]
    print(f"device capacity {capacity}: {len(device_configs)} configs", flush=True)
    with open(out_path, "a", encoding="utf-8") as fh:
        for depth in depths:
            C, M, rs = roles(int(depth), device_configs)
            compile_all(rs)
            for role, (f, cands, ref) in rs.items():
                t0 = time.time()
                exact = ref() if ref is not None else None
                ok, err = [], {}
                for c in cands:
                    try:
                        o = f(c)
                        torch.cuda.synchronize()
                        if exact is not None:
                            err[str(c)] = ((o.double() - exact).norm() / exact.norm()).item()
                        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        s.record()
                        f(c)
                        e.record()
                        torch.cuda.synchronize()
                        ok.append((c, math.ceil(target_ms / s.elapsed_time(e))))
                    except REJECTED as error:
                        print(f"C {C} {role} {c}: {type(error).__name__}: {error}", flush=True)
                del exact
                times = {str(c): [] for c, _ in ok}
                for rnd in range(rounds):
                    k = rnd * len(ok) // rounds
                    for c, n in ok[k:] + ok[:k]:
                        times[str(c)].append(graph_ms(f, c, calls=n))
                fields = {str(c): c if c == "cublas" else dataclasses.asdict(c) for c, _ in ok}
                for c, ms in times.items():
                    fh.write(json.dumps({"M": M, "C": C, "role": role, "config": fields[c], "ms": ms}
                                        | ({"err": err[c]} if err else {})) + "\n")
                fh.flush()
                best = min(times, key=lambda c: statistics.median(times[c]))
                print(f"C {C} M {M} {role}: {len(ok)} configs, {time.time() - t0:.1f} s, best "
                      f"{statistics.median(times[best]):.4f} ms {best}", flush=True)
            del rs
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
