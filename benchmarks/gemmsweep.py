"""Every GEMM role of the model (ops/quack_gemm.py: attention projections, relu^2 MLP with h = 4C, lm head) at width
C and M tokens on the current GPU, exactly as nanochat calls them: QuACK's candidates for this device's compute
capability (quack.gemm_config.get_all_configs() filtered to it: the autotuner's list; configs a call rejects are
dropped); the weight-gradient roles (proj_dw, mlp_dw_proj, mlp_dw_fc, lm_dw) at every tile config the call accepts x
serial split-K 1/2/4/8/16, plus cuBLAS (torch.mm(out_dtype), the production path) for those without alpha, each with its
rel L2 error vs fp64 (same bf16 inputs, fp64 sum chunked over M). Timing: median of ROUNDS interleaved rounds, a CUDA
graph of >= TARGET_MS of calls per replay, config order rotated each round; kernels compiled first in QuACK's pool.
Usage: python -m benchmarks.gemmsweep OUT.jsonl M C [C ...]; each line {"M", "C", "role", "config" (GemmConfig's
fields, or "cublas"), "ms": [per round], "err" (dW)}."""
import dataclasses
import json
import math
import statistics
import sys
import time

import torch
from quack.cache.async_compile import CompilePending, pool_scope
from quack.gemm_config import get_all_configs
from quack.gemm_interface import gemm_act, gemm_dact, gemm_tuned, prune_invalid_gemm_configs

from nanochat.config import VOCAB_SIZE

ROUNDS, TARGET_MS, SPLITS, CHUNK, V = 5, 30.0, (1, 2, 4, 8, 16), 4096, VOCAB_SIZE
CAPACITY = torch.cuda.get_device_capability()[0]
DEVICE_CONFIGS = [c for c in get_all_configs() if c.device_capacity == CAPACITY]
out_path, M, widths = sys.argv[1], int(sys.argv[2]), [int(c) for c in sys.argv[3:]]


def plain(A, B, out_dtype=None, alpha=1.0):
    """(callable(config), candidates) for quack_gemm.gemm(A, B): gemm_tuned.fn with the config's split-K."""
    out = torch.empty(A.shape[0], B.shape[1], dtype=out_dtype or A.dtype, device=A.device)
    cands = [c.kwargs["config"] for c in prune_invalid_gemm_configs(gemm_tuned.configs, dict(A=A, B=B, out=out),
                                                                    alpha=alpha, split_k=1)]

    def run(cfg):
        if cfg == "cublas":
            return torch.mm(A, B, out_dtype=out.dtype)
        gemm_tuned.fn(A, B, out, alpha=alpha, config=cfg, split_k=cfg.split_k)
        return out
    return run, cands


def weight_grad(A, B, alpha=None):
    run, tiles = plain(A, B, torch.float32, 1.0 if alpha is None else alpha)
    cands = sorted({dataclasses.replace(c, split_k=s) for c in tiles for s in SPLITS}, key=str)
    cands += ["cublas"] if alpha is None else []
    scale = 1.0 if alpha is None else alpha.item()

    def ref():
        acc = torch.zeros(A.shape[0], B.shape[1], dtype=torch.float64, device="cuda")
        for k in range(0, A.shape[1], CHUNK):
            acc += A[:, k:k + CHUNK].double() @ B[k:k + CHUNK].double()
        return acc * scale
    return run, cands, ref


def roles(C):
    h = 4 * C
    g = torch.Generator(device="cuda").manual_seed(0)
    r = lambda *s: torch.randn(*s, device="cuda", dtype=torch.bfloat16, generator=g)
    x, dy, W, w_fc, w_proj = r(M, C), r(M, C), r(C, C), r(h, C), r(C, h)
    preact, postact, dpreact = r(M, h), r(M, h), r(M, h)
    logits, w_lm = r(M, V), r(V, C)
    a = torch.full((1,), 1e-3, device="cuda")
    pre, post = torch.empty(M, h, device="cuda", dtype=torch.bfloat16), torch.empty(M, h, device="cuda", dtype=torch.bfloat16)
    dx, post2 = torch.empty(M, h, device="cuda", dtype=torch.bfloat16), torch.empty(M, h, device="cuda", dtype=torch.bfloat16)
    w_fc_t = w_fc.t()
    return {
        "proj_fwd": (*plain(x, W.t()), None),
        "proj_dx": (*plain(dy, W), None),
        "proj_dw": weight_grad(dy.t(), x),
        "mlp_up": (lambda cfg: gemm_act(x, w_fc_t, activation="relu_sq", preact_out=pre, postact_out=post, config=cfg),
                   DEVICE_CONFIGS, None),
        "mlp_down": (*plain(postact, w_proj.t()), None),
        "mlp_dact": (lambda cfg: gemm_dact(dy, w_proj, preact, activation="relu_sq", dx_out=dx, postact_out=post2,
                                           dynamic_scheduler=True, config=cfg), DEVICE_CONFIGS, None),
        "mlp_dx": (*plain(dpreact, w_fc), None),
        "mlp_dw_proj": weight_grad(dy.t(), postact),
        "mlp_dw_fc": weight_grad(dpreact.t(), x),
        "lm_dx": (*plain(logits, w_lm, alpha=a), None),
        "lm_dw": weight_grad(logits.t(), x, a),
    }


def compile_all(rs):
    pending = [(f, c) for f, cands, _ in rs.values() for c in cands if c != "cublas"]
    t = time.time()
    with pool_scope():
        while pending:
            waiting = []
            for f, c in pending:
                try:
                    f(c)
                except CompilePending:
                    waiting.append((f, c))
                except (RuntimeError, AssertionError, NotImplementedError, ValueError, TypeError) as err:
                    print(f"compile {c}: {type(err).__name__}: {str(err)[:160]}", flush=True)
            pending = waiting
            if pending:
                time.sleep(0.5)
    torch.cuda.synchronize()
    print(f"compile/load {time.time() - t:.1f} s", flush=True)


def graph_ms(f, cfg, n):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            f(cfg)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    g.replay()
    e.record()
    torch.cuda.synchronize()
    del g
    return s.elapsed_time(e) / n


print(f"device capacity {CAPACITY}: {len(DEVICE_CONFIGS)} configs", flush=True)
with open(out_path, "a") as fh:
    for C in widths:
        rs = roles(C)
        compile_all(rs)
        for role, (f, cands, ref) in rs.items():
            t0 = time.time()
            exact = ref() if ref is not None else None
            ok, err = [], {}
            for c in cands:
                try:
                    f(c)
                    o = f(c)
                    torch.cuda.synchronize()
                    if exact is not None:
                        err[str(c)] = ((o.double() - exact).norm() / exact.norm()).item()
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record(); f(c); e.record(); torch.cuda.synchronize()
                    ok.append((c, max(3, math.ceil(TARGET_MS / s.elapsed_time(e)))))
                except (RuntimeError, AssertionError, NotImplementedError, ValueError, TypeError) as error:
                    print(f"C {C} {role} {c}: {type(error).__name__}: {str(error)[:160]}", flush=True)
            del exact
            times = {str(c): [] for c, _ in ok}
            for rnd in range(ROUNDS):
                k = rnd * len(ok) // ROUNDS
                for c, n in ok[k:] + ok[:k]:
                    times[str(c)].append(graph_ms(f, c, n))
            fields = {str(c): c if c == "cublas" else dataclasses.asdict(c) for c, _ in ok}
            for c, ms in times.items():
                fh.write(json.dumps({"M": M, "C": C, "role": role, "config": fields[c], "ms": ms}
                                    | ({"err": err[c]} if err else {})) + "\n")
            fh.flush()
            best = min(times, key=lambda c: statistics.median(times[c]))
            print(f"C {C} M {M} {role}: {len(ok)} configs, {time.time() - t0:.1f} s, best {statistics.median(times[best]):.4f} ms {best}", flush=True)
        del rs
        torch.cuda.empty_cache()
