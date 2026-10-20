"""ops/quack_gemm.py TABLE entries from gemmsweep.py records. Forward and dX roles: lowest median time. Weight-gradient
roles: among configs whose median is within the fastest's median absolute deviation over rounds of the fastest median,
lowest error vs fp64, then fastest. Prints each pick against the fastest, then the TABLE entries.
Usage: python -m benchmarks.gemmtable benchmarks/gemmsweep_sm90.jsonl"""
import collections
import json
import statistics
import sys


def entry(cfg):
    """A record's config in TABLE syntax (split_k omitted at 1)."""
    if cfg == "cublas":
        return '"cublas"'
    split = f", split_k={cfg['split_k']}" if cfg["split_k"] != 1 else ""
    return (f"SM90(tile_m={cfg['tile_m']}, tile_n={cfg['tile_n']}, pingpong={cfg['pingpong']}, swap_ab={cfg['swap_ab']}, "
            f"cluster_m={cfg['cluster_m']}, cluster_n={cfg['cluster_n']}{split})")


def main():
    rows = collections.defaultdict(dict)  # (C, role) -> entry -> (ms per round, err)
    with open(sys.argv[1], encoding="utf-8") as f:
        for r in map(json.loads, f):
            rows[(r["C"], r["role"])][entry(r["config"])] = (r["ms"], r.get("err"))
    table = collections.defaultdict(dict)
    for (C, role), cs in rows.items():
        median = {c: statistics.median(ms) for c, (ms, _) in cs.items()}
        fastest = min(median, key=median.get)
        pick = fastest
        if cs[fastest][1] is not None:
            noise = statistics.median(abs(t - median[fastest]) for t in cs[fastest][0])
            pick = min((c for c in cs if median[c] <= median[fastest] + noise), key=lambda c: (cs[c][1], median[c]))
        table[C][role] = pick
        print(f"C {C} {role} {pick}: {median[pick]:.4f} ms (+{median[pick] / median[fastest] - 1:.2%} vs fastest)")
    for C, picks in table.items():
        print(f"    {C}: dict(\n" + ",\n".join(f"        {r}={p}" for r, p in picks.items()) + "),")


if __name__ == "__main__":
    main()
