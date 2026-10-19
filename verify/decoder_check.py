"""decode.cu (NativeDecoder) against the PyTorch model's eager greedy decoding (full forward per token, fp32 logits from
the bf16 hidden state and bf16 lm_head) on prompts of real documents (the first documents of stack_edu_python's compiled
val split, each BOS + its tokens cut to leave room for the new tokens in the context); and batch invariance (each
request decoded alone vs all at once). At a divergence, the eager top-2 logit gap (a near tie flips on fp32 sum-order
noise). Along the native sequences, the eager logit deficit of each native token (eager max logit minus the token's):
0 where both pick the same token, at most a near tie's gap otherwise.
MODEL: a base_train model tag (at its checkpoint's stage), or a depth: a fresh model at that depth and STAGE (file or
repo: config.FILE_STAGE, REPO_STAGE) with every matrix and table uniform with std C^-0.5 and every vector (lambdas, q/k
gains) uniform in [0.5, 1.5] (init_weights zeroes the output projections, lm_head and gates and fixes the vectors, under
which greedy decoding is trivial).
Usage: python -m verify.decoder_check MODEL N_PROMPTS NEW_TOKENS [STAGE]"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

from nanochat.config import FILE_STAGE, REPO_STAGE
from nanochat.data.loader import TOKEN_DTYPE, split_path
from nanochat.data.tokenizer import get_tokenizer
from nanochat.inference.decoder import NativeDecoder
from nanochat.model.gpt import ASPECT_RATIO, GPT, gpt_config
from nanochat.train.checkpoint import load_model

source, n_prompts, new = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
device = torch.device("cuda")
if source.isdigit():
    tokenizer = get_tokenizer()
    with torch.device("meta"):
        model = GPT(gpt_config(int(source), tokenizer.vocab_size, {"file": FILE_STAGE, "repo": REPO_STAGE}[sys.argv[4]],
                               aspect_ratio=ASPECT_RATIO))
    model.to_empty(device=device)
    model.init_weights()
    bound = (3 / model.config.n_embd) ** 0.5
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim == 2:
                p.uniform_(-bound, bound)
            else:
                p.uniform_(0.5, 1.5)
    model.eval()
else:
    model, tokenizer, _ = load_model(device, source)
print(f"model {source}: {model.config}")

path = split_path("stack_edu_python", "val", tokenizer)
tokens, offsets = np.memmap(path + ".bin", dtype=TOKEN_DTYPE, mode="r"), np.memmap(path + ".idx", dtype=np.int64, mode="r")
room = model.config.sequence_len - new
prompts = [[tokenizer.bos_token_id, *tokens[offsets[i]:offsets[i + 1]][:room - 1].tolist()] for i in range(n_prompts)]
print(f"{len(prompts)} prompts of {sorted(map(len, prompts))} tokens, {new} new tokens, windows {model.window_sizes}")


@torch.no_grad()
def last_logits(ids):
    """fp32 logits of the last position from the model's trunk (GPT.hidden, one segment) and the bf16 lm_head."""
    idx = torch.tensor([ids], dtype=torch.int32, device=device)
    h = model.hidden(idx, torch.tensor([0, len(ids)], dtype=torch.int32, device=device), None)
    return F.linear(h[-1].float(), model.lm_head.weight.to(torch.bfloat16).float())


def eager(ids):
    out, gaps = [], []
    for _ in range(new):
        z = last_logits(ids + out)
        top = z.topk(2)
        gaps.append((top.values[0] - top.values[1]).item())
        t = top.indices[0].item()
        if t == tokenizer.bos_token_id:
            break
        out.append(t)
    return out, gaps


ref = [eager(p) for p in prompts]
decoder = NativeDecoder(model)
print(f"decoder batch {decoder.batch}")
together = dict(decoder.run((p, new, None) for p in prompts))
alone = [next(iter(decoder.run([(p, new, None)])))[1] for p in prompts]
same = sum(together[i] == ref[i][0] for i in range(len(prompts)))
print(f"native (batched) == eager: {same}/{len(prompts)}")
print(f"native alone == native batched: {sum(alone[i] == together[i] for i in range(len(prompts)))}/{len(prompts)}")
for i, (r, gaps) in enumerate(ref):
    n = together[i]
    if n != r:
        j = next((k for k, (a, b) in enumerate(zip(r, n)) if a != b), min(len(r), len(n)))
        print(f"  prompt {i} ({len(prompts[i])} tokens): diverge at {j} of {len(r)}/{len(n)}; eager top-2 gap there "
              f"{gaps[j] if j < len(gaps) else float('nan'):.3e}; median gap {sorted(gaps)[len(gaps) // 2]:.3e}")
deficits = []
for i, p in enumerate(prompts):
    out = together[i]
    deficits += [(z.max() - z[t]).item() for j, t in enumerate(out) for z in [last_logits(p + out[:j])]]
print(f"eager deficit along the native tokens: {sum(d > 0 for d in deficits)}/{len(deficits)} steps nonzero, "
      f"max {max(deficits):.3e}")
