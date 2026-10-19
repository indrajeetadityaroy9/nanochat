"""Causal varlen attention, forward + backward, at the d32 layer shape on this GPU: FA4 (nanochat::flash_attn, the
model's op) and cuDNN (Transformer Engine DotProductAttention, qkv_format thd, fused-attention backend only). H = 8
heads of 128, rows of 8192 tokens x 4 (32,768 tokens), each row SEGMENTS equal documents (cu_seqlens padded with 8
empty trailing segments as the loader pads), window 512 (S layers) and full (L layers). Median ms of 20 timed
iterations after 5 warmup; outputs and gradients compared to FA4's (max |diff| / max |FA4|).
Usage: python -m bench.attn_bench SEGMENTS"""
import os
import statistics
import sys

os.environ["NVTE_FLASH_ATTN"], os.environ["NVTE_FUSED_ATTN"] = "0", "1"
import torch

from nanochat.ops.flash_attn import flash_attn

segments = int(sys.argv[1])
B, T, H, D = 4, 8192, 8, 128
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)
q, k, v, dout = (torch.randn(B * T, H, D, device=dev, dtype=torch.bfloat16, generator=g) for _ in range(4))
cu = torch.arange(0, B * T + 1, T // segments, device=dev, dtype=torch.int32)
cu = torch.cat([cu, cu[-1:].repeat(8)])


def timed(f):
    for _ in range(5):
        f()
    torch.cuda.synchronize()
    out = []
    for _ in range(20):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); f(); e.record(); torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out)


def leaves():
    return [t.detach().clone().requires_grad_() for t in (q, k, v)]


def fa4(window):
    def run(x):
        o = flash_attn(x[0].view(B, T, H, D), x[1].view(B, T, H, D), x[2].view(B, T, H, D), window, cu)[0]
        return o.view(B * T, H, D)
    return run


def cudnn(window):
    import transformer_engine.pytorch as te
    attn = te.DotProductAttention(H, D, qkv_format="thd", attn_mask_type="padding_causal", window_size=(window, 0),
                                  attention_dropout=0.0)

    def run(x):
        return attn(x[0], x[1], x[2], cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=T, max_seqlen_kv=T).view(B * T, H, D)
    return run


print(f"{torch.cuda.get_device_name()}: B {B} x T {T}, H {H} x D {D}, {segments} segments per row")
for window in (512, T):
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
        fwd = timed(lambda: run(x))
        both = timed(lambda: run(x).backward(dout))
        print(f"  window {window:5d} {name:6s} fwd {fwd:6.3f} ms  fwd+bwd {both:6.3f} ms{diff}")
