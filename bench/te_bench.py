"""The block's GEMM work, forward + backward, at the d24 width C (gpt_config(24, VOCAB_SIZE, FILE_STAGE,
aspect_ratio=ASPECT_RATIO)) and M = MICRO_BATCH_TOKENS tokens on this GPU: ours (bf16: nanochat::linear for q, k, v, o
and nanochat::relu2_mlp after the RMS norm) vs Transformer Engine (te.Linear for qkv (C -> 3C) and o, and
te.LayerNormMLP with RMSNorm and squared ReLU) in bf16 and under FP8 recipes (DelayedScaling, Float8CurrentScaling,
Float8BlockScaling; all three run on H100, bench/h100.txt). Median ms of torch.utils.benchmark.Timer.blocked_autorange's
measurements. Usage: python -m bench.te_bench"""
import torch
import torch.nn.functional as F
import transformer_engine.pytorch as te
from torch.utils import benchmark
from transformer_engine.common import recipe

from nanochat.config import FILE_STAGE, MICRO_BATCH_TOKENS, VOCAB_SIZE
from nanochat.model.gpt import ASPECT_RATIO, gpt_config
from nanochat.ops.linear import linear, relu2_mlp

C, M = gpt_config(24, VOCAB_SIZE, FILE_STAGE, aspect_ratio=ASPECT_RATIO).n_embd, MICRO_BATCH_TOKENS
dev = "cuda"
torch.manual_seed(0)
x = torch.randn(M, C, device=dev, dtype=torch.bfloat16, requires_grad=True)
dy = torch.randn(M, C, device=dev, dtype=torch.bfloat16)
s = 3 ** 0.5 * C ** -0.5
wq, wk, wv, wo = (torch.empty(C, C, device=dev).uniform_(-s, s).requires_grad_() for _ in range(4))
wfc = torch.empty(4 * C, C, device=dev).uniform_(-s, s).requires_grad_()
wproj = torch.empty(C, 4 * C, device=dev).uniform_(-s / 2, s / 2).requires_grad_()


def ms(f):
    return benchmark.Timer("f()", globals={"f": f}).blocked_autorange().median * 1e3


def ours():
    xn = F.rms_norm(x, (C,))
    q, k, v = linear(xn, wq), linear(xn, wk), linear(xn, wv)
    a = linear(q + k + v, wo)  # stands in for attention: same shapes into the o projection
    y = a + relu2_mlp(F.rms_norm(a, (C,)), wfc, wproj)[0]
    y.backward(dy)


qkv = te.Linear(C, 3 * C, bias=False, params_dtype=torch.float32).to(dev)
o = te.Linear(C, C, bias=False, params_dtype=torch.float32).to(dev)
mlp = te.LayerNormMLP(C, 4 * C, bias=False, normalization="RMSNorm", activation="srelu", zero_centered_gamma=True,
                      params_dtype=torch.float32).to(dev)
xt = x.detach().clone().requires_grad_()


def te_block(fp8_recipe):
    def run():
        with te.fp8_autocast(enabled=fp8_recipe is not None, fp8_recipe=fp8_recipe):
            xn = F.rms_norm(xt, (C,))
            q, k, v = qkv(xn).chunk(3, dim=-1)
            a = o(q + k + v)
            y = a + mlp(a)
        y.backward(dy)
    return run


print(f"C {C}, M {M}, {torch.cuda.get_device_name()}")
print(f"ours bf16 (QuACK)          {ms(ours):8.2f} ms")
with torch.autocast("cuda", dtype=torch.bfloat16):
    print(f"TE bf16                    {ms(te_block(None)):8.2f} ms")
    for name, r in (("DelayedScaling", recipe.DelayedScaling()), ("Float8CurrentScaling", recipe.Float8CurrentScaling()),
                    ("Float8BlockScaling", recipe.Float8BlockScaling())):
        print(f"TE fp8 {name:20s}{ms(te_block(r)):8.2f} ms")
