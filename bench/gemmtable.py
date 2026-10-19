"""Pick a device's GEMM configs from gemmsweep.py's records: forward and dX roles the lowest median time; weight-gradient
roles (proj_dw, mlp_dw_proj, mlp_dw_fc, lm_dw) among the configs within 1% of the fastest median the lowest error vs
fp64, then the fastest. Prints each pick against the fastest, then the TABLE entries of ops/quack_gemm.py.
Usage: python -m bench.gemmtable gemmsweep.jsonl"""
import collections
import json
import statistics
import sys

ORDER = ("proj_fwd", "proj_dx", "proj_dw", "mlp_up", "mlp_down", "mlp_dact", "mlp_dw_proj", "mlp_dx", "mlp_dw_fc",
         "lm_dx", "lm_dw")


def entry(cfg):
    """A record's config as ops/quack_gemm.py's TABLE writes it."""
    if cfg == "cublas":
        return '"cublas"'
    split = f", split_k={cfg['split_k']}" if cfg["split_k"] != 1 else ""
    return (f"SM90(tile_m={cfg['tile_m']}, tile_n={cfg['tile_n']}, pingpong={cfg['pingpong']}, swap_ab={cfg['swap_ab']}, "
            f"cluster_m={cfg['cluster_m']}, cluster_n={cfg['cluster_n']}{split})")


rows = collections.defaultdict(dict)  # (C, role) -> entry -> (median ms, err)
with open(sys.argv[1]) as f:
    for line in f:
        r = json.loads(line)
        rows[(r["C"], r["role"])][entry(r["config"])] = (statistics.median(r["ms"]), r.get("err"))
table = collections.defaultdict(dict)
for (C, role), cs in sorted(rows.items()):
    fastest = min(cs, key=lambda c: cs[c][0])
    near = [c for c in cs if cs[c][0] <= 1.01 * cs[fastest][0]]
    pick = fastest if cs[fastest][1] is None else min(near, key=lambda c: (cs[c][1], cs[c][0]))
    table[C][role] = pick
    print(f"C {C} {role} {pick}: {cs[pick][0]:.4f} ms (+{100 * (cs[pick][0] / cs[fastest][0] - 1):.2f}% vs fastest)")
for C, roles in table.items():
    print(f"    {C}: dict(\n" + ",\n".join(f"        {r}={roles[r]}" for r in ORDER) + "),")
