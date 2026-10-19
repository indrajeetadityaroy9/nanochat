"""GPT as trained (torch.compile dynamic=False fullgraph, bf16 compute, fused ops) against the fp64 reference
(reference.py) on the data pipeline's tokens: the first micro-batch of a stage (reference.first_micro_batch: packed
pieces, FIM examples with their sentinels, document-masking segments, padding targets), a freshly initialized model at
DEPTH perturbed by reference.perturb (every path non-zero).
- The whole micro-batch through the compiled model: loss, gradient norm, step time, peak memory.
- ROWS rows alone (B = 1, T = the stage's context) through reference.compare (fp64 reference, bf16 eager baseline,
  negative controls) for the canonical model, then the first row for each GPTConfig switch off (its other parameters
  the canonical model's).
- Deletion equivalence, on the first row, for each switch but windows (which no weight pins): the switch-off model's
  init_weights(0) equals the canonical model's on every shared parameter (common random numbers), and the switch-off
  model and the canonical model with the component pinned at its identity (value-embedding tables 0, so v + gate * 0 =
  v; x0_lambdas 0; resid_lambdas 1; q / k gains 1) compute bitwise-equal per-position losses and training loss, and
  shared-parameter gradients equal within REORDER_BOUND. Eager (op by op, each op rounded as written): Inductor fuses
  the two graphs differently and keeps different fp32 intermediates.
- The confirm study's candidate additions (experiments/studies.py CANDIDATES), each on the canonical model, perturbed,
  on the first row through reference.compare.
Usage: python -m tests.model_check DEPTH file|repo ROWS"""
import dataclasses
import sys
import time

import torch

from experiments.studies import CANDIDATES
from nanochat.config import FILE_STAGE, REPO_STAGE
from nanochat.data.loader import PretrainingBatches
from nanochat.data.tokenizer import get_tokenizer
from nanochat.model.gpt import GPT, GPTConfig, gpt_config
from tests.reference import compare, first_micro_batch, perturb, rel

# each switch's parameters (by name substring) and their identity value; windows has none
IDENTITY = {"value_embeds": ("value_embeds.", 0.0), "x0": ("x0_lambdas", 0.0), "resid": ("resid_lambdas", 1.0),
            "qk_gain": ("_gain", 1.0), "windows": None}
# bf16's unit roundoff (half its machine epsilon): autograd sums the gradients of a tensor's uses in the tensor's dtype,
# and the gradient reductions are not in a fixed order (FA4's and the embedding's backward add with atomics), so a
# deleted use or a different summation order moves a bf16 sum by about one rounding
REORDER_BOUND = torch.finfo(torch.bfloat16).eps / 2

depth, stage, n_rows = int(sys.argv[1]), {"file": FILE_STAGE, "repo": REPO_STAGE}[sys.argv[2]], int(sys.argv[3])
torch.manual_seed(0)
device = torch.device("cuda")
tokenizer = get_tokenizer()
T = stage.seq_len


def build(config: GPTConfig) -> GPT:
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights(0)
    return model


def compile_alone(model: GPT):
    """The model compiled as base_train does, after clearing Dynamo's caches: each model variant is its own graph, and
    the variants together exceed Dynamo's recompile limit on GPT.forward's code object."""
    torch._dynamo.reset()
    return torch.compile(model, dynamic=False, fullgraph=True)


def step(model: GPT, idx: torch.Tensor, targets: torch.Tensor, cu: torch.Tensor):
    """Eager per-position losses, training loss and each parameter's gradient."""
    with torch.no_grad():
        losses = model.losses(idx, targets, cu)
    model.zero_grad(set_to_none=True)
    loss = model(idx, targets, cu_seqlens=cu)
    loss.backward()
    return losses, loss.detach(), {n: p.grad for n, p in model.named_parameters()}


config = gpt_config(depth, stage)
full = build(config)
init = {n: p.detach().clone() for n, p in full.named_parameters()}
perturb(full)
weights = {n: p.detach().clone() for n, p in full.named_parameters()}
x, y, cu = first_micro_batch(stage, tokenizer, device)
print(f"d{depth} width {config.n_embd} heads {config.n_head} context {T} rope {config.rope_base} "
      f"windows {full.window_sizes}")
compiled = torch.compile(full, dynamic=False, fullgraph=True)
for it in range(3):
    full.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss = compiled(x, y, cu_seqlens=cu)
    loss.backward()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    grad_norm = torch.cat([p.grad.flatten() for p in full.parameters()]).norm().item()
    print(f"micro-batch {tuple(x.shape)} it {it}: loss {loss.item():.4f} grad norm {grad_norm:.4e} "
          f"{dt:.3f} s ({x.numel() / dt:,.0f} tok/s fwd+bwd) peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


def row(r: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Row r of the micro-batch alone: its inputs, targets and segments."""
    bounds = cu[(cu >= r * T) & (cu < (r + 1) * T)] - r * T
    return x[r:r + 1], y[r:r + 1], torch.cat([bounds, torch.tensor([T], dtype=torch.int32, device=device)]).unique_consecutive()


sentinels = torch.tensor(sorted(tokenizer.special_tokens.values()), device=device)
failures = {}
for r in range(n_rows):
    idx, targets, row_cu = row(r)
    print(f"canonical, row {r}: {len(row_cu) - 1} segments, {(targets != PretrainingBatches.PAD).sum().item()} valid "
          f"targets, {torch.isin(idx, sentinels).sum().item()} special-token inputs")
    failures[f"canonical row {r}"] = compare(full, compiled, idx, targets, row_cu, indent="  ")
    torch.cuda.empty_cache()
idx, targets, row_cu = row(0)
not_equivalent = []
for switch, identity in IDENTITY.items():
    off = build(dataclasses.replace(config, **{switch: False}))
    same_init = all(torch.equal(p, init[n]) for n, p in off.named_parameters())
    with torch.no_grad():
        for n, p in off.named_parameters():
            p.copy_(weights[n])
    print(f"{switch} off, row 0: windows {off.window_sizes}")
    failures[f"{switch} off"] = compare(off, compile_alone(off), idx, targets, row_cu, indent="  ")
    if identity is None:
        continue
    part, value = identity
    with torch.no_grad():
        for n, p in full.named_parameters():
            if part in n:
                p.fill_(value)
            else:
                p.copy_(weights[n])
    losses_full, loss_full, grads_full = step(full, idx, targets, row_cu)
    losses_off, loss_off, grads_off = step(off, idx, targets, row_cu)
    forward = torch.equal(losses_off, losses_full) and torch.equal(loss_off, loss_full)
    reordered = {n: rel(grads_off[n], grads_full[n]) for n in grads_off if not torch.equal(grads_off[n], grads_full[n])}
    if not (same_init and forward) or any(e > REORDER_BOUND for e in reordered.values()):
        not_equivalent.append(switch)
    print(f"deletion equivalence {switch}: init shared parameters equal {same_init}; losses bitwise equal {forward} "
          f"({loss_off.item():.9f} vs pinned canonical {loss_full.item():.9f}); shared-parameter gradients bitwise "
          f"{len(grads_off) - len(reordered)}/{len(grads_off)}, the others rel diff "
          f"{ {n: f'{e:.1e}' for n, e in reordered.items()} } (bound {REORDER_BOUND:.1e})")
    torch.cuda.empty_cache()
# the confirm study's candidate additions (experiments/studies.py CANDIDATES) on the canonical model, perturbed, on the
# first row
for name, fields in CANDIDATES.items():
    added = build(dataclasses.replace(config, **fields))
    perturb(added)
    print(f"{name} added, row 0: windows {added.window_sizes}")
    failures[f"{name} added"] = compare(added, compile_alone(added), idx, targets, row_cu, indent="  ")
    torch.cuda.empty_cache()
failures = {k: v for k, v in failures.items() if v}
print(f"compare failures (ours above bf16 eager): {failures}; deletions not equivalent: {not_equivalent}")
sys.exit(1 if failures or not_equivalent else 0)
