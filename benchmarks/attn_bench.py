"""Causal varlen attention, forward + backward, at a model's layer shape on this GPU, on the data pipeline's segments:
FA4 (nanochat::flash_attn, the model's op) and cuDNN (Transformer Engine DotProductAttention, qkv_format thd,
fused-attention backend only). Heads and head_dim of gpt_config(DEPTH, stage); cu_seqlens of the stage's first packed
micro-batch (data/loader.py PretrainingBatches, fim reading: real documents and padding), which alone sets the
kernels' work (q, k, v values do not); the model's windows (gpt.layer_windows). Median ms of
torch.utils.benchmark.Timer.blocked_autorange's measurements; outputs and gradients compared to FA4's
(max |diff| / max |FA4|).
Usage: python -m benchmarks.attn_bench DEPTH [file|repo stage]"""
import os
import sys

import numpy as np
import torch
import transformer_engine.pytorch as te
from torch.utils import benchmark

from nanochat.config import FILE_STAGE, REPO_STAGE
from nanochat.data.loader import PretrainingBatches, mixture_weights, packed_rows
from nanochat.data.tokenizer import get_tokenizer
from nanochat.model.gpt import gpt_config, layer_windows
from nanochat.ops.flash_attn import flash_attn

# read by Transformer Engine when it selects a backend (at the first forward), not at import
os.environ["NVTE_FLASH_ATTN"], os.environ["NVTE_FUSED_ATTN"] = "0", "1"
stage = {"file": FILE_STAGE, "repo": REPO_STAGE}[sys.argv[2] if len(sys.argv) > 2 else "file"]
config = gpt_config(int(sys.argv[1]), stage)
B, T = stage.batch_rows, stage.seq_len
H, D = config.n_head, config.n_embd // config.n_head
dev = torch.device("cuda")
tokenizer = get_tokenizer()
mixture = mixture_weights(stage.mixture, tokenizer, T)
orders = {c: np.random.RandomState(0).permutation(packed_rows(c, "train", tokenizer, T)) for c in mixture}
_, _, cu = next(iter(PretrainingBatches(mixture, "train", tokenizer, orders, seq_len=T, batch_rows=B, device=dev,
                                        reading="fim", start=0)))
g = torch.Generator(device=dev).manual_seed(0)
q, k, v, dout = (torch.randn(B * T, H, D, device=dev, dtype=torch.bfloat16, generator=g) for _ in range(4))


def leaves():
    return [t.detach().clone().requires_grad_() for t in (q, k, v)]


def fa4(window):
    def run(x):
        o = flash_attn(x[0].view(B, T, H, D), x[1].view(B, T, H, D), x[2].view(B, T, H, D), window, cu)[0]
        return o.view(B * T, H, D)
    return run


def cudnn(window):
    attn = te.DotProductAttention(H, D, qkv_format="thd", attn_mask_type="padding_causal", window_size=(window, 0),
                                  attention_dropout=0.0)

    def run(x):
        return attn(x[0], x[1], x[2], cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=T, max_seqlen_kv=T).view(B * T, H, D)
    return run


print(f"{torch.cuda.get_device_name()}: B {B} x T {T}, H {H} x D {D}, {int((cu.diff() > 0).sum())} segments")
for window in sorted(set(layer_windows(config))):
    ref = None
    for name, make in (("FA4", fa4), ("cuDNN", cudnn)):
        run = make(window)
        x = leaves()
        o = run(x)
        o.backward(dout)
        got = [o.detach(), *(t.grad.clone() for t in x)]
        if ref is None:
            ref, diff = got, ""
        else:
            diff = " max|diff|/max|FA4| " + " ".join(f"{n} {((a.float() - b.float()).abs().max() / b.float().abs().max()).item():.1e}"
                                                     for n, a, b in zip(("o", "dq", "dk", "dv"), got, ref, strict=True))
        fwd, both = (benchmark.Timer("f()", globals={"f": f}).blocked_autorange().median * 1e3
                     for f in (lambda run=run, x=x: run(x), lambda run=run, x=x: run(x).backward(dout)))
        print(f"  window {window:5d} {name:6s} fwd {fwd:6.3f} ms  fwd+bwd {both:6.3f} ms{diff}")
