"""decode.cu (NativeDecoder) against the PyTorch model's eager greedy decoding (full forward per token, fp32 logits from
the bf16 hidden state and bf16 lm_head) for a base_train run's checkpoint (at its checkpoint's stage), on prompts of
real documents (the first documents of stack_edu_python's compiled val split, each BOS + its tokens cut to leave room
for the new tokens in the context); batch invariance (each request decoded alone vs all at once); and a stop rule
(code_eval.StopAt, the evaluators' rule) over more requests than the decoder has slots (the prompts in turn, so
finished slots take new requests within the run), each against the eager tokens up to and including the one at which
the rule first holds. At a divergence, the eager top-2 logit gap (a near tie flips on fp32 sum-order noise). Along the
native sequences, the eager logit deficit of each native token (eager max logit minus the token's): 0 where both pick
the same token, at most a near tie's gap otherwise.
Usage: python -m tests.decoder_check TAG N_PROMPTS NEW_TOKENS"""
import statistics
import sys

import numpy as np
import torch
import torch.nn.functional as F

from nanochat.data.loader import TOKEN_DTYPE, split_path
from nanochat.eval.code_eval import StopAt
from nanochat.inference.decoder import NativeDecoder
from nanochat.train.checkpoint import load_model

device = torch.device("cuda")
model, tokenizer, _ = load_model(device, sys.argv[1])
n_prompts, new = int(sys.argv[2]), int(sys.argv[3])
print(f"model {sys.argv[1]}: {model.config}")
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


def eager(ids, stop=None):
    out, gaps = [], []
    for _ in range(new):
        z = last_logits(ids + out)
        top = z.topk(2)
        gaps.append((top.values[0] - top.values[1]).item())
        t = top.indices[0].item()
        if t == tokenizer.bos_token_id:
            break
        out.append(t)
        if stop is not None and stop(out):
            break
    return out, gaps


ref = [eager(p) for p in prompts]
decoder = NativeDecoder(model, bos_token_id=tokenizer.bos_token_id)
print(f"decoder batch {decoder.batch}")
together = dict(decoder.run((p, new, None) for p in prompts))
alone = [next(iter(decoder.run([(p, new, None)])))[1] for p in prompts]
same = sum(together[i] == ref[i][0] for i in range(len(prompts)))
print(f"native (batched) == eager: {same}/{len(prompts)}")
print(f"native alone == native batched: {sum(alone[i] == together[i] for i in range(len(prompts)))}/{len(prompts)}")
STOPS = ["\n\n", "\ndef "]  # Python's blank line and next function: hold within a few tokens on most documents
cut = [eager(p, StopAt(tokenizer, STOPS))[0] for p in prompts]
requests = decoder.batch + len(prompts)
stopped = dict(decoder.run((prompts[j % len(prompts)], new, StopAt(tokenizer, STOPS)) for j in range(requests)))
print(f"native with stop rule {STOPS} == eager cut there: "
      f"{sum(stopped[j] == cut[j % len(prompts)] for j in range(requests))}/{requests} requests (batch "
      f"{decoder.batch}), {sum(len(c) < new for c in cut)}/{len(prompts)} prompts end before {new} tokens")
for i, (r, gaps) in enumerate(ref):
    n = together[i]
    if n != r:
        j = next((k for k, (a, b) in enumerate(zip(r, n, strict=False)) if a != b), min(len(r), len(n)))
        print(f"  prompt {i} ({len(prompts[i])} tokens): diverge at {j} of {len(r)}/{len(n)}; eager top-2 gap there "
              f"{gaps[j] if j < len(gaps) else float('nan'):.3e}; median gap {statistics.median_high(gaps):.3e}")
deficits = []
for i, p in enumerate(prompts):
    out = together[i]
    deficits += [(z.max() - z[t]).item() for j, t in enumerate(out) for z in [last_logits(p + out[:j])]]
print(f"eager deficit along the native tokens: {sum(d > 0 for d in deficits)}/{len(deficits)} steps nonzero, "
      f"max {max(deficits):.3e}")
