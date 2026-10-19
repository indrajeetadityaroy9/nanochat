"""Bitwise run-to-run determinism of base_train's compiled forward + backward at d24 (the production model, perturbed
init: tests/reference.perturb) on the first file-level micro-batch of the packed train rows: the step repeated 6 times
from the same weights and batch, each parameter's gradient compared bit for bit with the first repeat's. Prints the
median fwd+bwd time and the parameters whose gradients differ; exits 1 if any does."""
import statistics
import sys
import time

import numpy as np
import torch

from nanochat.config import FILE_STAGE
from nanochat.data.loader import PretrainingBatches, mixture_weights, packed_rows
from nanochat.data.tokenizer import get_tokenizer
from nanochat.model.gpt import ASPECT_RATIO, GPT, gpt_config

from tests.reference import perturb

torch.manual_seed(0)
device = torch.device("cuda")
tokenizer = get_tokenizer()
stage = FILE_STAGE
with torch.device("meta"):
    model = GPT(gpt_config(24, tokenizer.vocab_size, stage, aspect_ratio=ASPECT_RATIO))
model.to_empty(device=device)
model.init_weights(0)
perturb(model)
mixture = mixture_weights(stage.mixture, tokenizer, stage.seq_len)
orders = {c: np.random.RandomState(42).permutation(packed_rows(c, "train", tokenizer, stage.seq_len)) for c in mixture}
x, y, cu = next(iter(PretrainingBatches(mixture, "train", tokenizer, orders, seq_len=stage.seq_len,
                                        batch_rows=stage.batch_rows, device=device, reading="fim", start=0)))
compiled = torch.compile(model, dynamic=False, fullgraph=True)


def step():
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    compiled(x, y, cu_seqlens=cu).backward()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


step()  # compile / warm up
times = [step()]
ref = {n: p.grad.clone() for n, p in model.named_parameters()}
differ = set()
for _ in range(5):
    times.append(step())
    differ |= {n for n, p in model.named_parameters() if not torch.equal(p.grad, ref[n])}
print(f"d24 fwd+bwd median {statistics.median(times) * 1e3:.1f} ms; parameters with run-to-run gradient differences "
      f"{len(differ)}/{len(ref)}: {sorted(differ)}")
sys.exit(1 if differ else 0)
