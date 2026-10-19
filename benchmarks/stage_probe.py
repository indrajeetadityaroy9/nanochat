"""Time and peak memory of one micro-batch (forward + backward, compiled as base_train) of each training stage at DEPTH
and ASPECT_RATIO on this GPU: MICRO_TOKENS // seq_len rows of random token ids (timing does not depend on them), each
row 4 equal segments at the file-level stage and 1 segment (one long repository sample) at the repo-level stage.
Usage: python -m benchmarks.stage_probe DEPTH ASPECT_RATIO [MICRO_TOKENS (default MICRO_BATCH_TOKENS)]"""
import sys
import time

import torch

from nanochat.config import FILE_STAGE, MICRO_BATCH_TOKENS, REPO_STAGE, VOCAB_SIZE
from nanochat.model.gpt import GPT, gpt_config

depth, aspect = int(sys.argv[1]), int(sys.argv[2])
micro = int(sys.argv[3]) if len(sys.argv) > 3 else MICRO_BATCH_TOKENS
with torch.device("meta"):
    orig = GPT(gpt_config(depth, VOCAB_SIZE, FILE_STAGE, aspect_ratio=aspect))
orig.to_empty(device="cuda")
orig.init_weights(0)
for stage, segments in ((FILE_STAGE, 4), (REPO_STAGE, 1)):
    orig.set_stage(stage)
    model = torch.compile(orig, dynamic=False, fullgraph=True)
    rows, T = micro // stage.seq_len, stage.seq_len
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randint(0, VOCAB_SIZE, (rows, T), device="cuda", generator=g, dtype=torch.int32)
    y = torch.randint(0, VOCAB_SIZE, (rows, T), device="cuda", generator=g)
    cu = torch.arange(0, rows * T + 1, T // segments, device="cuda", dtype=torch.int32)
    for _ in range(3):
        model(x, y, cu_seqlens=cu).backward()
    orig.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(5):
        model(x, y, cu_seqlens=cu).backward()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / 5 * 1e3
    orig.zero_grad(set_to_none=True)
    print(f"d{depth} x {orig.config.n_embd} stage seq_len {T} rope {stage.rope_base:g}, {micro} tokens: {ms:.1f} ms / "
          f"micro-batch ({micro / ms * 1e3:,.0f} tok/s), peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
