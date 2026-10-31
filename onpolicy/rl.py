"""
RL with verifiable rewards from the SFT model (docs/post-training.md §6): the trainer, in the training image with the
native model (model/modeling.py, the fused kernels SFT trained with), beside onpolicy/sampler.py in the vLLM image on
the same GPU (scripts/rl.sh), exchanging the policies and their batches through files (the sampler's docstring), for
BUDGET seconds of training.

Objective: one update per batch, so on policy (2 or 4 updates per batch "led to rapid entropy collapse around 100
steps", arXiv 2505.16400 §3.2; Nemotron-Cascade's code RL, arXiv 2512.13607 §4.1.2): the ratio is 1, and the loss is
the token-level policy gradient (DAPO, arXiv 2503.14476 §3.3, Eq. 12)
    -sum_i A_i sum_t log pi(o_i,t) / sum_i |o_i|,
A_i = (r_i - its group's mean reward) / the std of the batch's rewards (Lite PPO, arXiv 2508.08221 §4.1: the group's
mean, the batch's std), r_i 1 when the response's answer passes its problem's tests and 0 otherwise (an unclosed or
failing answer, or a loop: an exact cycle never ends, so it is a failure rather than an answer cut short), plus DAPO's
soft overlong punishment (`overlong`, arXiv 2503.14476 §3.4, Eq. 13): 0 up to L_MAX - L_CACHE tokens, then falling
linearly to -1 at L_MAX, the sampler's limit, so a cut response scores -1. Masking cut responses instead (overlong
filtering, DAPO §3.4) left nothing against length: from the SFT model the cut share rose from 3.7% to 19.5% in six
batches and the pass rate over all responses fell from 0.420 to 0.391 (rl/attempt-16k-masked-train.jsonl), where
10-15% truncation "typically destabilized training" (ScaleRL, arXiv 2510.13786 App. A.15). The trained groups are
those whose rewards differ (a group of equal rewards has zero advantage and no gradient, DAPO §3.2); a trivial prompt's
group is unused; no KL term (DAPO §2.3). pi is the distribution the sampler draws from at T 1.0: the capped logits
c tanh(h W / c) of the final hidden states, the head in fp32 as vLLM computes it (the export's "head_dtype"). The head
and its loss are `Head`, its gradients computed in its forward HEAD_CHUNK rows at a time and only at the trained
positions (no prompt position); the backbone runs over the responses packed into rows of the context (`pack`: first
fit decreasing, each response its own attention segment).

Optimizer (docs/post-training.md §6): AdamW at a constant LR, beta (0.9, 0.95), no weight decay, the gradient's norm
clipped at 1.0; eps 1e-15, as an eps near the gradients' scale clips them (ScaleRL, arXiv 2510.13786 App. A.3). The
weights are fp32; the forward's bf16 GEMM copies are rounded to nearest and the published policy
(inference/export_hf.py `hf_weights`) holds the same rounding, so the sampler samples the policy the trainer
differentiates, up to the two engines' kernels.

Per batch k: X/policy-<k> published (the export's format, its other files from the export directory); X/batch-<k>.pkl
awaited and read; the update; policy k + 1 published before anything is written to disk. The batch whose end would
let the next one overrun BUDGET (this attempt's time, the last batch's duration ahead) is the last: X/final names its
policy before it is published. Every SAVE_EVERY batches and after the last, <out>/state.pt (weights, optimizer state,
step), resumed from. <out>/train.jsonl: a line per batch.

    python -m onpolicy.rl <checkpoint.pt> <export dir> <exchange dir> <out dir>
"""
import argparse
import json
import pickle
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from corpus.tokenizer import get_tokenizer
from inference.export_hf import hf_weights
from model.modeling import CausalLM, ModelConfig, accumulated_weights
from model.ops.cross_entropy import CHUNK as HEAD_CHUNK

LR, BETAS, EPS, MAX_GRAD_NORM = 2e-6, (0.9, 0.95), 1e-15, 1.0
BUDGET = 2 * 3600
# the sampler's and the evaluation's response limit (onpolicy/rollouts.py MAX_TOKENS; rollouts imports vLLM, which the
# training image lacks) and DAPO's punishment interval below it ("4,096 tokens as the soft punish cache", §4.1)
L_MAX, L_CACHE = 16384, 4096
# responses per packed row at most (attention segments, the padding's included): the row's cu_seqlens has a fixed
# MAX_SEGMENTS + 1 entries, so the compiled forward sees one shape
MAX_SEGMENTS = 256
SAVE_EVERY = 5


@torch.compile(dynamic=False, fullgraph=True)
def head_chunk(x: torch.Tensor, weight: torch.Tensor, target: torch.Tensor,
               w: torch.Tensor, cap: float) -> tuple[torch.Tensor, ...]:
    """Rows x (n, C) bf16, weight (V, C) fp32 -> (sum of -w log pi(target), dx in x's dtype, dW fp32, log pi(target),
    entropy), pi the softmax of c tanh(x W^T / c) in fp32."""
    y = x.float() @ weight.t()
    t = torch.tanh(y / cap)
    z = cap * t
    lse = torch.logsumexp(z, 1)
    logp = z.gather(1, target[:, None]).squeeze(1) - lse
    p = torch.exp(z - lse[:, None])
    entropy = lse - (p * z).sum(1)
    hot = torch.arange(z.size(1), device=z.device) == target[:, None]
    dy = (p - hot.to(p.dtype)) * (1 - t * t) * w[:, None]
    return -(w * logp).sum(), (dy @ weight).to(x.dtype), dy.t() @ x.float(), logp, entropy


class Head(torch.autograd.Function):
    """sum over rows of -w log pi(target), gradients computed in the forward (as model/ops/cross_entropy.py lm_loss
    does), HEAD_CHUNK rows at a time, so one chunk's logits are alive; also each row's log pi(target) and entropy."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, target: torch.Tensor, w: torch.Tensor, cap: float):
        n, pad = len(x), -len(x) % HEAD_CHUNK
        x, target, w = F.pad(x, (0, 0, 0, pad)), F.pad(target, (0, pad)), F.pad(w, (0, pad))
        loss, dw = torch.zeros((), device=x.device), torch.zeros_like(weight)
        dx, logp, entropy = [], [], []
        for a in range(0, len(x), HEAD_CHUNK):
            part, d, g, lp, e = head_chunk(x[a:a + HEAD_CHUNK], weight, target[a:a + HEAD_CHUNK],
                                           w[a:a + HEAD_CHUNK], cap)
            loss += part
            dw += g
            dx.append(d)
            logp.append(lp)
            entropy.append(e)
        ctx.save_for_backward(torch.cat(dx)[:n], dw)
        logp, entropy = torch.cat(logp)[:n], torch.cat(entropy)[:n]
        ctx.mark_non_differentiable(logp, entropy)
        return loss, logp, entropy

    @staticmethod
    def backward(ctx, dloss: torch.Tensor, _logp: torch.Tensor, _entropy: torch.Tensor):
        dx, dw = ctx.saved_tensors
        return (dx.float() * dloss).to(dx.dtype), dw * dloss, None, None, None


def pack(lengths: list[int], row: int) -> list[list[int]]:
    """Indices of `lengths` in rows of at most `row` tokens and MAX_SEGMENTS - 1 sequences, first fit decreasing."""
    rows, room = [], []
    for i in sorted(range(len(lengths)), key=lambda i: -lengths[i]):
        for r, free in enumerate(room):
            if lengths[i] <= free and len(rows[r]) < MAX_SEGMENTS - 1:
                rows[r].append(i)
                room[r] -= lengths[i]
                break
        else:
            rows.append([i])
            room.append(row - lengths[i])
    return rows


def overlong(n: int) -> float:
    """DAPO's soft overlong punishment of an n-token response (arXiv 2503.14476 §3.4, Eq. 13): 0 up to
    L_MAX - L_CACHE, then ((L_MAX - L_CACHE) - n) / L_CACHE, -1 at L_MAX and beyond."""
    return max(-1.0, min(0.0, (L_MAX - L_CACHE - n) / L_CACHE))


class Policy:
    def __init__(self, checkpoint: Path, out: Path):
        self.device = torch.device("cuda")
        state = torch.load(checkpoint, map_location="cpu", mmap=True)
        with torch.device("meta"):
            self.model = CausalLM(ModelConfig(**state["config"]))
        self.model.load_state_dict(state["model"], assign=True)
        del state
        self.model.to(self.device)
        self.model.init_buffers()
        self.model.refresh_gemm_weights()
        # fp32 gradients the GEMMs accumulate into (model/ops/linear.py); zeroed, never freed, after each update
        for p in accumulated_weights(self.model):
            p.grad = torch.zeros_like(p)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=0.0,
                                     fused=True)
        self.step = 0
        if (out / "state.pt").exists():
            saved = torch.load(out / "state.pt", map_location=self.device)
            self.model.load_state_dict(saved["model"])
            self.opt.load_state_dict(saved["optimizer"])
            self.step = saved["step"]
        self.hidden = torch.compile(self.model.hidden, dynamic=False, fullgraph=True)
        self.row = self.model.config.max_position_embeddings
        self.cap = self.model.config.final_logit_softcapping
        self.bos = get_tokenizer().bos_token_id

    def packed(self, members: list[tuple[list[int], int, float]], tokens: int) -> tuple[torch.Tensor, ...]:
        """One row's (input_ids (1, row), cu_seqlens, segments, trained positions, their targets and weights)."""
        ids, bounds = np.full(self.row, self.bos, np.int32), []
        index, target, weight = [], [], []
        o = 0
        for s, p, a in members:
            ids[o:o + len(s)] = s
            bounds.append(o)
            index.append(np.arange(o + p - 1, o + len(s) - 1))
            target.append(np.asarray(s[p:], np.int64))
            weight.append(np.full(len(s) - p, a / tokens, np.float32))
            o += len(s)
        if o < self.row:  # the padding, a segment of its own
            bounds.append(o)
        cu = np.full(MAX_SEGMENTS + 1, self.row, np.int32)
        cu[:len(bounds)] = bounds
        put = lambda a: torch.from_numpy(a).to(self.device, non_blocking=True)
        return (put(ids[None]), put(cu), torch.tensor(len(bounds), dtype=torch.int32), put(np.concatenate(index)),
                put(np.concatenate(target)), put(np.concatenate(weight)))

    def update(self, batch: dict) -> dict:
        # every response of each group, its reward its pass plus the length's punishment
        kept = [(g["samples"], g["prompt"]) for g in batch["groups"] if not g["trivial"]]
        rewards = [np.array([(s["verdict"] == "pass") + overlong(len(s["ids"])) for s in samples], np.float64)
                   for samples, _ in kept]
        flat = np.concatenate(rewards)
        mixed = [(samples, prompt, r) for (samples, prompt), r in zip(kept, rewards, strict=True)
                 if len(r) > 1 and r.min() < r.max()]
        seqs = [(prompt + s["ids"], len(prompt), float(a))
                for samples, prompt, r in mixed for s, a in zip(samples, (r - r.mean()) / flat.std(), strict=True)]
        samples = [s for g in batch["groups"] for s in g["samples"]]
        stats = {"groups": len(batch["groups"]), "trivial": sum(g["trivial"] for g in batch["groups"]),
                 "cut": sum(s["finish"] == "length" for s in samples),
                 "pass_rate": sum(s["verdict"] == "pass" for s in samples) / len(samples),
                 "reward": float(flat.mean()), "mixed_groups": len(mixed), "trained_responses": len(seqs),
                 "verdicts": dict(Counter(s["verdict"] for s in samples))}
        self.step += 1
        if not seqs:
            return stats | {"updated": False}
        tokens = sum(len(s) - p for s, p, _ in seqs)
        loss = logp = entropy = torch.zeros((), device=self.device, dtype=torch.float64)
        rows = pack([len(s) for s, _, _ in seqs], self.row)
        for members in rows:
            input_ids, cu_seqlens, segments, index, target, weight = self.packed([seqs[i] for i in members], tokens)
            h = self.hidden(input_ids, cu_seqlens, segments)
            part, lp, e = Head.apply(h[index], self.model.lm_head.weight, target, weight, self.cap)
            part.backward()
            loss, logp, entropy = loss + part.detach(), logp + lp.sum(), entropy + e.sum()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
        self.opt.step()
        self.opt.zero_grad(set_to_none=False)
        self.model.refresh_gemm_weights()
        return stats | {"updated": True, "trained_tokens": tokens, "rows": len(rows), "loss": loss.item(),
                        "grad_norm": norm.item(), "logp": logp.item() / tokens, "entropy": entropy.item() / tokens}

    def publish(self, exchange: Path, source: Path) -> None:
        """X/policy-<step> in the export's format, written under a temporary name then renamed."""
        tmp, final = exchange / f"policy-{self.step}.tmp", exchange / f"policy-{self.step}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        save_file(hf_weights(self.model.state_dict()), tmp / "model.safetensors", metadata={"format": "pt"})
        for f in source.iterdir():
            if f.is_file() and f.name != "model.safetensors":
                shutil.copy(f, tmp / f.name)
        tmp.rename(final)

    def save(self, out: Path) -> None:
        tmp = out / "state.pt.tmp"
        torch.save({"model": self.model.state_dict(), "optimizer": self.opt.state_dict(), "step": self.step}, tmp)
        tmp.replace(out / "state.pt")


def main(checkpoint: Path, source: Path, exchange: Path, out: Path) -> None:
    torch.backends.cuda.matmul.allow_tf32 = False  # the head in fp32, as vLLM computes it
    begin = time.time()
    out.mkdir(parents=True, exist_ok=True)
    exchange.mkdir(parents=True, exist_ok=True)
    policy = Policy(checkpoint, out)
    policy.publish(exchange, source)
    while True:
        k, start = policy.step, time.time()
        path = exchange / f"batch-{k}.pkl"
        while not path.exists():
            time.sleep(0.2)
        waited = time.time()
        batch = pickle.loads(path.read_bytes())
        path.unlink()
        shutil.rmtree(exchange / f"policy-{k}")  # loaded (and evaluated) by the sampler before it sampled batch k
        stats = policy.update(batch)
        trained = time.time()
        last = trained - begin + (trained - start) > BUDGET
        if last:
            (exchange / "final").write_text(str(policy.step))
        policy.publish(exchange, source)
        published = time.time()
        if last or policy.step % SAVE_EVERY == 0:
            policy.save(out)
        with open(out / "train.jsonl", "a") as f:
            f.write(json.dumps({"step": k} | stats | {"sampler": batch["stats"], "seconds": {
                "waited": waited - start, "trained": trained - waited, "published": published - trained,
                "saved": time.time() - published, "elapsed": time.time() - begin}}) + "\n")
        if last:
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("exchange", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    main(args.checkpoint, args.source, args.exchange, args.out)
