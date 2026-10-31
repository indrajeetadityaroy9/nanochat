"""
The on-policy stage's training (docs/post-training.md §8.5): loop-commit training (`commit`) of a --chat export at its
own loops, then DPO (`dpo`; arXiv 2305.18290) of the commit model (the export when commit training aborts) on its own
verified KodCode rollouts.

Why commit: the export's loops fit the large-error profile of arXiv 2512.12895 (risk aversion, §3: a progress-making
step too hard for the model beside an easy cyclic one; temperature only a stopgap when the errors are large, §6;
harder problems loop more, §2): at the protocol's T 0.6 the loop token leads its runner-up by a median 5.5 logits
(p(loop token) at T 0.6 median 1.00), 545 of 802 KodCode val loops are on prompts the model never solved in 4
samples, and continuations resampled from a loop's history escaped in 0 of 88. Final-token preference optimization
(FTPO, arXiv 2510.15061; Antidoom, github.com/Liquid4All/antidoom) toward the model's own alternatives at the loop
token moved normal text with the loop (66-98% of the loop lead's change; KodCode val pass@1 0.360 -> 0.000 at full
strength). Committing is an action the model can take: its loops cut after one verbatim repeat (at o + 2p) and
closed by END_THINK pass KodCode's tests in 24.8% of 802 val loops (14.1% on prompts it never solved, 82.4% on prompts
it solved 3-4 times in 4), while its probability of committing there is about e^-20. The forced close is s1's budget
forcing (arXiv 2501.19393 §3.1: the end-of-thinking delimiter appended "to early exit the thinking stage and make the
model provide its current best answer", there a length cap), applied here at a loop; on a 0.6B student's unfinished
(length-limited) samples it recovers answers in 0.364 where the starting model solved 3-4 of 4 and 0.049 where it
never did (arXiv 2609.37326 App. E, Table 18), whose analysis places small students' missing ability in stopping,
"along the responses that run on" (§7).

A row (onpolicy/selection.py commit.jsonl) is a loop with an exact cycle cut after one verbatim repeat (prefix_ids:
the prompt and the response through its cut, selection.py `cut`: the first line start from onset + 2 period on, where
the commit opens as the model's own ends of reasoning do), the loop established as Word Salad Chopper establishes
it before chopping (arXiv 2511.00536 §3.2: two consecutive reasoning chunks flagged as near-repeats of earlier ones;
its probe on the chunks' trailing hidden states finds trapped states linearly separable, §2.4; forcing an
end-of-think there is one of the follow-ups it names, §5). The loss (`commit_loss`) on the model's capped logits y in
fp32 (the head and cap outside autocast, as vLLM computes them): L_commit, the NLL of END_THINK's tokens after the
prefix; plus, at the row's anchors (`anchored`: line starts, where a commit would begin, of its own response well
before the loop and of the reasoning and the answer of a loop-free response from the same source), RETAIN x
KL(reference || model), the normal-text term DITTO mixes in beside its repetition update (arXiv 2206.02369 §3: an
equal mix with the normal likelihood update; here the model's own normal text is kept against the reference rather
than relearned), and LAMBDA_TETHER x the mean max(y[</] - y_ref[</] - TAU, 0)^2 of END_THINK's decisive token "</":
FTPO's target-tether form (arXiv 2510.15061 §5.2: a zero-penalty window TAU, squared beyond it), one-sided, weighted
as FTPO weights its non-target tether, since "</" is not a target at normal positions (KL alone barely sees mass
added where the reference has e^-24).

Training (transformers' Trainer): every parameter in fp32 (master weights and AdamW state), bf16 autocast, each decoder
layer recomputed in the backward pass (`load`), SDPA attention, one epoch at most, seed 0; batch BATCH rows
(Antidoom's 4 x 4 accumulation, as 1 x 16), AdamW (betas the library's), weight decay WEIGHT_DECAY, linear decay after
a WARMUP share of the steps, gradient norm clipped at MAX_GRAD_NORM (src/antidoom/ftpo_train.py:119-141); the
reference logits at the anchors computed once, before the first update, by the same model.

Held-out monitor, gate and stop (`monitor_split`, `monitor`, `point`): before training, the rows of whole prompts,
drawn in a default_rng(SEED) permutation of the prompt ids until MONITOR_ROWS rows, leave the training set; at step 0,
every EARLY_EVERY optimizer steps through GATE_STEP, then every MONITOR_EVERY and at the last step, without gradient:
at their cuts commit_logp (END_THINK's log-probability), commit_win (the share where every END_THINK token is the
argmax: greedy decoding commits), commit_p06 (its probability at T 0.6) and lt_cut (log p of "</" after the commit's
newline); at their anchors lt_normal (log p of "</") and retain_kl; the hazard (`hazard_points`: the whole commit
appended at the reasoning anchors of the first HAZARD_ROWS rows, its own before the loop and its loop-free response's
in the first REASONING_SHARE of the reasoning, away from the natural end where the model commits at about 0.9; its p
at T 0.6, the rise over step 0 times the mean line starts of a loop-free reasoning: premature commits per response
added by training, at line starts) and commit_normal (its mean log p there; the specificity, commit_normal's gain over
commit_logp's, is logged); and the guard (`guard`): the NLL of
the SFT model's own passing KodCode val answers. A point fails when the NLL exceeds its step-0 value by NLL_RISE, or
from GATE_STEP on when commit_logp has not gained or the hazard exceeds HAZARD; the specificity is the hazard's
absolute bound, not a ratio of log-probability gains (normal positions start about e^18 below the cuts: in a 40-row
smoke lt_normal rose as much as lt_cut while the hazard stayed below 1e-7). A failure writes the last passing point's
weights (one from GATE_STEP on) or nothing (commit training aborted). Training stops at the first passing point from
GATE_STEP whose commit_win reaches STOP, writing the model to <out dir>; an epoch that ends passing writes it at its
end. The training batches' statistics are logged every LOG_STEPS steps.

dpo (DeLoop's preference stage, arXiv 2610.19376 §4.2, App. G.1; the pairs: onpolicy/selection.py dpo.jsonl): the
sigmoid loss -log sigmoid(beta ((log pi(y+|x) - log pi_ref(y+|x)) - (log pi(y-|x) - log pi_ref(y-|x)))) on the
completions' summed log-probabilities under the capped logits (`completion_logp`, the head and cap in fp32) and
nothing else: no likelihood term on the chosen side (DeLoop's NLL anchor raised loops 6.1% -> 27.0%, App. G.13).
The reference is the starting model, frozen: its log-probabilities computed once before the first update by the same
model. beta DPO_BETA (DeLoop's, chosen from {0.02, 0.05, 0.1, 0.2, 0.5} on a screening set, App. G.1, the lowest loop
rate of those values, App. G.6); AdamW at DPO_LR (Phi-4-mini-reasoning's rollout DPO with full parameters, one epoch
at 16K, arXiv 2504.21233 §5.3; DeLoop's LoRA lr 8e-6 is 16x that, against LoRA's optimum of about 10x the
full-parameter lr, docs/post-training.md §2.2), betas (0.9, 0.95), no weight decay, gradient norm clipped at 1.0;
batch DPO_BATCH pairs (DeLoop's 16), warmup DPO_WARMUP of the steps, linear decay; one epoch, seed 0; the model
loaded and written as commit's (`load`, `write`). Every GUARD_EVERY steps the guard: training stops once its NLL
exceeds its step-0 value by GUARD_RISE.

Each model is written as the export is (inference/export_hf.py: fp32 the parameters the forward reads in fp32,
bf16 the rest; config, generation config, tokenizer and model file copied), so vLLM serves it as it serves the
export. The bf16 tensors are rounded stochastically (`bf16`: E[rounded] = the fp32 value, a value on the bf16 grid
unchanged): an update smaller than half an element's bf16 spacing survives in expectation, where rounding to nearest
erases it. <out dir>/train-log.json holds the written model's step and `retention` (the share of the fp32 update the
written weights keep along its direction), the log history and the peak GPU memory allocated; for commit also the
rows' statistics, the monitor points and the reason the model was written (or commit training aborted).

    python -m onpolicy.train commit <model dir> <commit.jsonl> <out dir> [--max-steps N] [--batch N] [--cpu]
        [--monitor-rows N] [--monitor-every N] [--stop 0.4]
    python -m onpolicy.train dpo <model dir> <dpo.jsonl> <out dir> [--max-steps N] [--batch N] [--cpu]
"""
import argparse
import functools
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

# training: Antidoom's configs/default.yaml train.batch_size x gradient_accumulation_steps, warmup_ratio, weight_decay;
# src/antidoom/ftpo_train.py:126,140
BATCH, WARMUP, WEIGHT_DECAY = 16, 0.1, 0.01
LOG_STEPS, MAX_GRAD_NORM = 5, 2.5
# lr: Antidoom's LoRA 1.5e-5 (configs/default.yaml) / 10, LoRA's optimum being about 10x the full-parameter lr
# (docs/post-training.md §2.2); every parameter
LR = 1.5e-6
# retention: RETAIN carried from the gated FTPO run (chosen there over 50 on a 40-row smoke; the guard +4.2% by its
# step 20); the tether: FTPO's tau_mse_target and its non-target weight lambda_mse (configs/default.yaml); ANCHORS
# line starts per stratum, the loop's own ANCHOR_GAP tokens or more before its onset
# REASONING_SHARE of a reasoning holds its normal anchors (the end, where the model commits, left out); the hazard over
# HAZARD_ROWS held-out rows' reasoning anchors: a "</" alone mostly measured code's closing tags and the natural ends
# (0.58 per response at step 0)
RETAIN, LAMBDA_TETHER, TAU, ANCHORS, ANCHOR_GAP = 200.0, 0.4, 0.5, 4, 1024
REASONING_SHARE, HAZARD_ROWS = 0.9, 64
# the gate: the NLL of the SFT model's passing answers within 2% (+24% cost 13 pass@1 points); at most HAZARD
# premature commits per response (each trades a non-looping response's pass, about 0.6, for a commit's, about 0.3: 5%
# costs at most a tenth of the +0.097 pass@1 that committing at every val loop gains); checks every EARLY_EVERY steps
# through GATE_STEP
NLL_RISE, HAZARD, GATE_STEP, EARLY_EVERY = 0.02, 0.05, 20, 5
# capability guard: the NLL of the SFT model's first GUARD_RESPONSES passing KodCode val answers (eval-model/val
# beside the output directory); DPO stops once it exceeds its step-0 value by GUARD_RISE, checked every GUARD_EVERY
GUARD_RESPONSES, GUARD_RISE, GUARD_EVERY = 32, 0.10, 20
# held-out monitor and its stop: the Qwen3.5-4B Antidoom run's early stop at chosen_win 0.4 (arXiv 2606.13705 App. K.3;
# Antidoom configs/default.yaml early_stopping_chosen_win), here the share of cuts where greedy decoding commits;
# 256 rows put its standard error at sqrt(0.4 x 0.6 / 256) ≈ 0.03
MONITOR_ROWS, MONITOR_EVERY, STOP, SEED = 256, 10, 0.4, 0
# dpo: DeLoop's beta and batch (arXiv 2610.19376 App. G.1), Phi-4-mini-reasoning's rollout-DPO lr (arXiv 2504.21233
# §5.3), warmup ratio 0.1; head positions per chunk of `completion_logp`
DPO_LR, DPO_BATCH, DPO_WARMUP, DPO_BETA, HEAD_CHUNK = 5e-7, 16, 0.1, 0.05, 1024


def load(model_dir: Path, train: bool = True):
    """The model in fp32; to train, each decoder layer recomputed in the backward pass (torch's non-reentrant
    checkpoint, autocast state kept): 0.68 MB of saved activations per token instead of 2.04, with identical gradients
    (a 4,096-token sequence, CPU, bf16 autocast). The export's model file declares no gradient checkpointing for
    transformers to enable."""
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.float32,
                                                 attn_implementation="sdpa")
    if train:
        for layer in model.model.layers:
            layer.forward = functools.partial(checkpoint, layer.forward, use_reentrant=False)
    return model


def logits_at(model, ids: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """The capped logits (n, V) at positions index (n,) of ids (1, L): the backbone under the caller's autocast, the
    head and the cap in fp32 outside it (inference/modeling_nanochat.py NanochatForCausalLM.forward)."""
    h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, index]
    with torch.autocast(h.device.type, enabled=False):
        y = F.linear(h.float(), model.lm_head.weight.float())
    c = model.config.final_logit_softcapping
    return c * torch.tanh(y / c)


def kl(y: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """The mean over rows of KL(softmax(ref) || softmax(y)), y and ref (n, V) capped logits."""
    lp_ref, lp = ref.float().log_softmax(-1), y.log_softmax(-1)
    return (lp_ref.exp() * (lp_ref - lp)).sum(-1).mean()


def anchored(rows: list[dict], waves: Path, newline: np.ndarray, seed: int) -> tuple[list[dict], float]:
    """Each row's anchors, logit positions (`logits_at`) at line starts (the position's token is one of `newline`, the
    tokens ending in a newline): own, ANCHORS of its response's line starts at least ANCHOR_GAP tokens before the
    loop's onset (none when the onset is nearer the start), in its prefix; normal, ANCHORS of the line starts of the
    first REASONING_SHARE of the reasoning (`reasoning` of them) and ANCHORS of the answer of a loop-free response (no
    loop, the thinking closed, ended) to a prompt of
    the row's source in the waves, each source's responses in a default_rng(seed) permutation, reused in that order
    once spent; normal_ids, that response's context cut after its last anchor. And the mean line starts of a loop-free
    response's reasoning."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    prompts, pool = {}, defaultdict(list)
    for wave in sorted(p.parent for p in waves.glob("wave-*/rollouts.parquet")):
        prompts |= {p["id"]: p for p in map(json.loads, (wave / "prompts.jsonl").read_text().splitlines())}
        columns = ["prompt_id", "sample", "ids", "loop", "closed", "ended", "reasoning_tokens", "answer_tokens"]
        table = pq.read_table(wave / "rollouts.parquet", columns=columns)
        complete = pc.and_(pc.and_(pc.invert(table["loop"]), table["closed"]), table["ended"])
        for r in table.filter(complete).to_pylist():
            pool[prompts[r["prompt_id"]]["source"]].append(r)
    rng = np.random.default_rng(seed)
    order = {}
    for s, rs in sorted(pool.items()):
        rs.sort(key=lambda r: (r["prompt_id"], r["sample"]))
        order[s] = [rs[i] for i in rng.permutation(len(rs))]
    used = Counter()

    def pick(ids: np.ndarray, lo: int, hi: int) -> list[int]:
        """ANCHORS line starts in [lo, hi) of ids, sorted."""
        starts = lo + np.flatnonzero(np.isin(ids[lo:max(hi, lo)], newline))
        return sorted(rng.choice(starts, size=min(ANCHORS, len(starts)), replace=False).tolist()) if len(starts) else []

    out = []
    for r in rows:
        prefix = np.asarray(r["prefix_ids"])
        start = len(prefix) - r["cut"]
        source = prompts[r["prompt_id"]]["source"]
        n = order[source][used[source] % len(order[source])]
        used[source] += 1
        full = np.asarray(prompts[n["prompt_id"]]["ids"] + n["ids"])
        k, end = len(prompts[n["prompt_id"]]["ids"]), len(full) - 1
        normal = pick(full, k, k + int(REASONING_SHARE * n["reasoning_tokens"]))
        kept = len(normal)
        normal += pick(full, end - n["answer_tokens"], end)
        out.append({"own": pick(prefix, start, start + r["onset"] - ANCHOR_GAP), "normal": normal,
                    "reasoning": kept, "normal_ids": full[:max(normal, default=k) + 1].tolist()})
    reasoning = [r["ids"][:r["reasoning_tokens"]] for rs in pool.values() for r in rs]
    lines = float(np.mean([np.isin(np.asarray(ids), newline).sum() for ids in reasoning])) if reasoning else 0.0
    return out, lines


def commit_loss(y: torch.Tensor, commit: torch.Tensor, anchors: torch.Tensor, ref: torch.Tensor,
                lt: int) -> tuple[torch.Tensor, dict]:
    """A row's loss (module docstring): y (k, V) at the k positions predicting the commit tokens `commit` (k,),
    anchors and ref (n, V) the model's and the reference's logits at its anchors, lt the decisive token's id."""
    nll = -y.log_softmax(-1).gather(-1, commit[:, None]).sum()
    retain = kl(anchors, ref) if len(anchors) else y.new_zeros(())
    tether = ((anchors[:, lt] - ref[:, lt] - TAU).clamp(min=0) ** 2).mean() if len(anchors) else y.new_zeros(())
    loss = nll + RETAIN * retain + LAMBDA_TETHER * tether
    return loss, {"commit_nll": float(nll.detach()), "retain_kl": float(retain.detach()),
                  "tether": float(tether.detach())}


def bf16(v: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """v rounded to bf16 stochastically: uniform noise below the 16 bits bf16 drops, added to the fp32 bits (sign and
    magnitude, so the magnitude rounds up with probability its remainder / spacing), then truncated."""
    bits = v.detach().float().cpu().contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32, generator=generator)
    return ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)


def served(model) -> dict[str, torch.Tensor]:
    """The state dict as inference/export_hf.py writes it: fp32 the parameters the forward reads in fp32, the rest
    rounded to bf16 stochastically (one generator, seed 0)."""
    from inference.modeling_nanochat import NanochatPreTrainedModel

    fp32, generator = set(NanochatPreTrainedModel._keep_in_fp32_modules_strict), torch.Generator().manual_seed(0)
    return {k: v.detach().float().cpu().contiguous() if fp32 & set(k.split(".")) else bf16(v, generator)
            for k, v in model.state_dict().items()}


def retention(start: dict[str, torch.Tensor], model, saved: dict[str, torch.Tensor]) -> dict:
    """<saved - start, final - start> / |final - start|^2 over every tensor, and the share of elements written
    changed."""
    num = den = 0.0
    changed = total = 0
    for k, v in model.state_dict().items():
        s = start[k].float()
        d, ds = v.detach().float().cpu() - s, saved[k].float() - s
        num, den = num + float((ds * d).sum()), den + float((d * d).sum())
        changed, total = changed + int((ds != 0).sum()), total + ds.numel()
    return {"retained": num / den if den else None, "elements_changed": changed / total}


def write(model, source: Path, out: Path) -> dict:
    """The trained model in the export's format beside the source's other files (its train-log.json aside); its
    retention."""
    saved = served(model)
    kept = retention(load_file(source / "model.safetensors"), model, saved)
    out.mkdir(parents=True, exist_ok=True)
    save_file(saved, out / "model.safetensors", metadata={"format": "pt"})
    for f in source.iterdir():
        if f.is_file() and f.name not in ("model.safetensors", "train-log.json"):
            shutil.copy(f, out / f.name)
    return kept


def quantiles(values: list[float]) -> dict | None:
    return {"n": len(values), "p10_p50_p90": np.percentile(values, [10, 50, 90]).tolist()} if values else None


def guard_set(out: Path, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The guard's (prompt ids, answer ids): the SFT model's first GUARD_RESPONSES passing KodCode val answers in
    (prompt, sample) order, from scripts/onpolicy.sh's eval-model/val beside the output directory."""
    import pyarrow.parquet as pq

    val = out.parent / "eval-model" / "val"
    verdicts = {(v["prompt_id"], v["sample"]): v["verdict"]
                for v in map(json.loads, (val / "verdicts.jsonl").read_text().splitlines())}
    prompts = {p["id"]: p["ids"] for p in map(json.loads, (val / "prompts.jsonl").read_text().splitlines())}
    rows = sorted((r for r in pq.read_table(val / "rollouts.parquet").to_pylist()
                   if r["closed"] and r["ended"] and verdicts.get((r["prompt_id"], r["sample"])) == "pass"),
                  key=lambda r: (r["prompt_id"], r["sample"]))[:GUARD_RESPONSES]
    return [(torch.tensor(prompts[r["prompt_id"]], device=device), torch.tensor(r["ids"], device=device))
            for r in rows]


def guard(model, items: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    """The mean per-token NLL of the guard's answers under the model (capped logits, `completion_logp`)."""
    with torch.no_grad(), torch.autocast(items[0][0].device.type, dtype=torch.bfloat16):
        total = sum(float(completion_logp(model, prompt, answer)) for prompt, answer in items)
    return -total / sum(len(answer) for _, answer in items)


def monitor_split(examples: list[dict], n: int, seed: int) -> tuple[list[dict], list[dict]]:
    """(train, monitor): the monitor the rows of whole prompts, the prompt ids in a default_rng(seed) permutation of
    their sorted order taken until it holds n rows (or every prompt but one has been taken)."""
    ids = sorted({e["prompt_id"] for e in examples})
    counts = Counter(e["prompt_id"] for e in examples)
    held, rows = set(), 0
    for i in np.random.default_rng(seed).permutation(len(ids)).tolist()[:max(len(ids) - 1, 0)]:
        if rows >= n:
            break
        held.add(ids[i])
        rows += counts[ids[i]]
    return [e for e in examples if e["prompt_id"] not in held], [e for e in examples if e["prompt_id"] in held]


def forward(model, e: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """A row's logits at its commit positions (k, V) and at its anchors (n, V), under the caller's autocast."""
    own = len(e["own"])
    y = logits_at(model, e["input_ids"].to(device), torch.cat([e["own"], e["commit_at"]]).to(device))
    n = logits_at(model, e["normal_ids"].to(device), e["normal"].to(device))
    return y[own:], torch.cat([y[:own], n])


def monitor(model, examples: list[dict], device: torch.device, lt: int) -> dict:
    """The held-out rows' commit_logp, commit_win, commit_p06, lt_cut; at their anchors lt_normal and retain_kl (module
    docstring); without gradient."""
    stats = defaultdict(list)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        for e in examples:
            y, anchors = forward(model, e, device)
            commit = e["commit"].to(device)
            stats["commit_logp"].append(float(y.log_softmax(-1).gather(-1, commit[:, None]).sum()))
            stats["commit_win"].append(float((y.argmax(-1) == commit).all()))
            stats["commit_p06"].append(float((y / 0.6).log_softmax(-1).gather(-1, commit[:, None]).sum().exp()))
            stats["lt_cut"].append(float(y[1].log_softmax(-1)[lt]))
            if len(anchors):
                stats["lt_normal"].extend(anchors.log_softmax(-1)[:, lt].tolist())
                stats["retain_kl"].append(float(kl(anchors, e["anchor_ref"].to(device))))
    return {k: float(np.mean(v)) for k, v in stats.items()}


def segment(model, ids: torch.Tensor, at: list[int], commit: torch.Tensor) -> list[tuple[float, float]]:
    """(log p, p at T 0.6) of the commit tokens appended after ids[0, :a + 1], for each anchor a in at."""
    out = []
    for a in at:
        x = torch.cat([ids[0, :a + 1], commit])[None]
        y = logits_at(model, x, torch.arange(a, a + len(commit), device=x.device))
        lp = y.log_softmax(-1).gather(-1, commit[:, None]).sum()
        out.append((float(lp), float((y / 0.6).log_softmax(-1).gather(-1, commit[:, None]).sum().exp())))
    return out


def hazard_points(model, examples: list[dict], device: torch.device) -> list[tuple[float, float]]:
    """The commit's (log p, p at T 0.6) at every reasoning anchor of the first HAZARD_ROWS held-out rows: their own
    anchors before the loop and their loop-free responses' reasoning anchors; without gradient."""
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        return [s for e in examples[:HAZARD_ROWS]
                for s in segment(model, e["input_ids"].to(device), e["own"].tolist(), e["commit"].to(device))
                + segment(model, e["normal_ids"].to(device), e["normal"][:e["reasoning"]].tolist(),
                          e["commit"].to(device))]


def commit(model_dir: Path, data: Path, out: Path, max_steps: int, batch: int | None, cpu: bool, monitor_rows: int,
           monitor_every: int, stop: float) -> dict:
    from transformers import Trainer, TrainerCallback, TrainingArguments

    from corpus.posttrain.sft import END_THINK
    from corpus.tokenizer import get_tokenizer
    from onpolicy.selection import newline_ids

    tok = get_tokenizer()
    end = tok.enc.encode_ordinary(END_THINK)
    lt = end[1]  # "</": END_THINK opens with a newline, the commit's decisive token follows it
    newline = newline_ids()
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    model = load(model_dir).to(device).eval()
    marks, lines = anchored(rows, data.parent, newline, SEED)
    examples = []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        for r, m in zip(rows, marks, strict=True):
            p = len(r["prefix_ids"])
            e = {"input_ids": torch.tensor([r["prefix_ids"] + end]), "commit": torch.tensor(end),
                 "commit_at": torch.arange(p - 1, p - 1 + len(end)), "own": torch.tensor(m["own"], dtype=torch.long),
                 "normal_ids": torch.tensor([m["normal_ids"]]), "normal": torch.tensor(m["normal"], dtype=torch.long),
                 "reasoning": m["reasoning"], "prompt_id": r["prompt_id"]}
            e["anchor_ref"] = forward(model, e, device)[1].cpu()
            examples.append(e)
    train, held = monitor_split(examples, monitor_rows, SEED)
    sets = {"rows": len(rows), "train": len(train), "monitor": len(held),
            "monitor_prompts": len({e["prompt_id"] for e in held}), "line_starts_per_reasoning": lines,
            "anchors": quantiles([len(e["own"]) + len(e["normal"]) for e in examples]),
            "prefix_tokens": quantiles([len(r["prefix_ids"]) for r in rows])}
    items = guard_set(out, device)
    points, written, good, base = [], {}, {}, []

    class CommitTrainer(Trainer):
        stored = defaultdict(list)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # the backbone is a submodule: Accelerate's autocast wraps only the top-level forward
            with torch.autocast(inputs["input_ids"].device.type, dtype=torch.bfloat16):
                y, anchors = forward(model, inputs, inputs["input_ids"].device)
            loss, stats = commit_loss(y, inputs["commit"], anchors, inputs["anchor_ref"], lt)
            for k, v in stats.items():
                self.stored[k].append(v)
            return loss

        def log(self, logs, *args, **kwargs):
            logs = logs | {k: sum(v) / len(v) for k, v in self.stored.items() if v}
            self.stored.clear()
            return super().log(logs, *args, **kwargs)

    def point(model, step: int) -> dict:
        p = {"step": step} | monitor(model, held, device, lt) | {"nll": guard(model, items)}
        seg = hazard_points(model, held, device)
        if not step:
            base.extend(seg)
        p["commit_normal"] = float(np.mean([s[0] for s in seg]))
        p["hazard"] = (float(np.mean([s[1] for s in seg])) - float(np.mean([s[1] for s in base]))) * lines
        if step:
            p["nll_rise"] = p["nll"] / points[0]["nll"] - 1
            gain = p["commit_logp"] - points[0]["commit_logp"]
            p["specificity"] = (p["commit_normal"] - points[0]["commit_normal"]) / gain if gain > 0 else None
            fail = "nll" if p["nll_rise"] > NLL_RISE else None
            if not fail and step >= GATE_STEP:
                fail = ("no commit gain" if p["commit_logp"] <= points[0]["commit_logp"]
                        else "hazard" if p["hazard"] > HAZARD else None)
            if fail and good:
                model.load_state_dict(good["state"])
                written.update(step=good["step"], reason=fail, retention=write(model, model_dir, out))
            elif fail:
                written.update(step=None, reason=f"aborted: {fail}")
            elif step >= GATE_STEP and p["commit_win"] >= stop:
                written.update(step=step, reason="commit_win", retention=write(model, model_dir, out))
            elif step >= GATE_STEP:
                good.update(step=step, state={k: v.detach().to("cpu", copy=True)
                                              for k, v in model.state_dict().items()})
        points.append(p)
        print(json.dumps(p), flush=True)
        return p

    class Monitor(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            point(model, 0)

        def on_step_end(self, args, state, control, model=None, **kwargs):
            every = EARLY_EVERY if state.global_step <= GATE_STEP else monitor_every
            if state.global_step % every == 0 or state.global_step >= state.max_steps:
                point(model, state.global_step)
                if written:
                    control.should_training_stop = True

    args = TrainingArguments(
        output_dir=str(out / "trainer"), per_device_train_batch_size=1, gradient_accumulation_steps=batch or BATCH,
        learning_rate=LR, warmup_steps=WARMUP, lr_scheduler_type="linear", weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM, num_train_epochs=1, max_steps=max_steps, bf16=True, use_cpu=device.type == "cpu",
        logging_steps=LOG_STEPS, save_strategy="no", report_to="none", seed=0, data_seed=0,
        dataloader_num_workers=0, remove_unused_columns=False, gradient_checkpointing=False)
    model.train()
    trainer = CommitTrainer(model=model, args=args, train_dataset=train, data_collator=lambda features: features[0],
                            callbacks=[Monitor()])
    trainer.loss_is_scaled_for_ga = False  # a row's loss: the step averages its rows
    trainer.train()
    if not written:
        written.update(step=trainer.state.global_step, reason="epoch",
                       retention=write(trainer.model, model_dir, out))
    return {"sets": sets, "lr": LR, "retain": RETAIN, "tether": [LAMBDA_TETHER, TAU],
            "nll_rise_max": NLL_RISE, "hazard_max": HAZARD, "gate_step": GATE_STEP, "monitor_every": monitor_every,
            "stop": stop, "stopped": trainer.state.global_step < trainer.state.max_steps, "monitor": points,
            "written": written, "steps": trainer.state.global_step, "max_steps": trainer.state.max_steps,
            "max_memory_allocated": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            "log_history": trainer.state.log_history}


def completion_logp(model, prompt: torch.Tensor, completion: torch.Tensor) -> torch.Tensor:
    """The summed log-probability of completion (C,) after prompt (P,) under the capped logits: the backbone under
    the caller's autocast, the head and cap in fp32 outside it (`logits_at`), HEAD_CHUNK positions at a time, each
    chunk's logits recomputed in the backward pass rather than kept."""
    from torch.utils.checkpoint import checkpoint

    ids = torch.cat([prompt, completion])[None]
    h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, len(prompt) - 1:-1]
    head, c = model.lm_head.weight, model.config.final_logit_softcapping

    def chunk(h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.autocast(h.device.type, enabled=False):
            y = F.linear(h.float(), head.float())
            return (c * torch.tanh(y / c)).log_softmax(-1).gather(-1, target[:, None]).sum()

    run = functools.partial(checkpoint, chunk, use_reentrant=False) if torch.is_grad_enabled() else chunk
    return sum(run(h[a:a + HEAD_CHUNK], completion[a:a + HEAD_CHUNK]) for a in range(0, len(completion), HEAD_CHUNK))


def dpo_loss(chosen: torch.Tensor, rejected: torch.Tensor, ref_chosen: float, ref_rejected: float,
             beta: float) -> tuple[torch.Tensor, dict]:
    """DPO's sigmoid loss for one pair from its sides' summed log-probabilities under the model and the reference;
    the implicit rewards beta (log pi - log pi_ref) of each side."""
    r_chosen, r_rejected = beta * (chosen - ref_chosen), beta * (rejected - ref_rejected)
    loss = -F.logsigmoid(r_chosen - r_rejected)
    stats = {"reward_chosen": r_chosen, "reward_rejected": r_rejected, "reward_margin": r_chosen - r_rejected,
             "reward_accuracy": (r_chosen > r_rejected).float(), "logp_chosen": chosen, "logp_rejected": rejected}
    return loss, {k: float(v.detach()) for k, v in stats.items()}


def dpo(model_dir: Path, data: Path, out: Path, max_steps: int, batch: int | None, cpu: bool) -> dict:
    from transformers import Trainer, TrainerCallback, TrainingArguments

    rows = [json.loads(line) for line in data.read_text().splitlines()]
    if not rows:
        raise ValueError(f"no pairs in {data}")
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    model = load(model_dir).to(device).eval()
    examples = []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        for r in rows:
            e = {k: torch.tensor(r[f"{k}_ids"]) for k in ("prompt", "chosen", "rejected")}
            for side in ("chosen", "rejected"):
                e[f"ref_{side}"] = float(completion_logp(model, e["prompt"].to(device), e[side].to(device)))
            examples.append(e)
    items = guard_set(out, device)
    checks = [{"step": 0, "nll": guard(model, items)}]

    class Guard(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step % GUARD_EVERY == 0:
                checks.append({"step": state.global_step, "nll": guard(model, items)})
                print(json.dumps(checks[-1]), flush=True)
                if checks[-1]["nll"] > (1 + GUARD_RISE) * checks[0]["nll"]:
                    control.should_training_stop = True

    class DPOTrainer(Trainer):
        stored = defaultdict(list)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # the reference's autocast, explicit: on the CPU the Trainer's own pass otherwise ran without it (a
            # learning-rate-0 step's rewards were off by about 1 nat per pair)
            prompt = inputs["prompt"]
            with torch.autocast(prompt.device.type, dtype=torch.bfloat16):
                chosen = completion_logp(model, prompt, inputs["chosen"])
                rejected = completion_logp(model, prompt, inputs["rejected"])
            loss, stats = dpo_loss(chosen, rejected, inputs["ref_chosen"], inputs["ref_rejected"], DPO_BETA)
            for k, v in stats.items():
                self.stored[k].append(v)
            return loss

        def log(self, logs, *args, **kwargs):
            logs = logs | {k: sum(v) / len(v) for k, v in self.stored.items() if v}
            self.stored.clear()
            return super().log(logs, *args, **kwargs)

    args = TrainingArguments(
        output_dir=str(out / "trainer"), per_device_train_batch_size=1, gradient_accumulation_steps=batch or DPO_BATCH,
        learning_rate=DPO_LR, warmup_steps=DPO_WARMUP, lr_scheduler_type="linear", adam_beta1=0.9, adam_beta2=0.95,
        weight_decay=0.0, max_grad_norm=1.0, num_train_epochs=1, max_steps=max_steps, bf16=True,
        use_cpu=device.type == "cpu", logging_steps=1, save_strategy="no", report_to="none", seed=0, data_seed=0,
        dataloader_num_workers=0, remove_unused_columns=False, gradient_checkpointing=False)
    model.train()
    trainer = DPOTrainer(model=model, args=args, train_dataset=examples, data_collator=lambda features: features[0],
                         callbacks=[Guard()])
    trainer.loss_is_scaled_for_ga = False  # a pair's loss: the step averages its pairs
    trainer.train()
    return {"pairs": len(rows), "beta": DPO_BETA, "lr": DPO_LR,
            "guard_rise": GUARD_RISE, "guard": checks,
            "written": {"step": trainer.state.global_step, "retention": write(trainer.model, model_dir, out)},
            "steps": trainer.state.global_step, "max_steps": trainer.state.max_steps,
            "max_memory_allocated": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            "log_history": trainer.state.log_history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("commit", "dpo"))
    parser.add_argument("model", type=Path)
    parser.add_argument("data", type=Path, help="commit.jsonl or dpo.jsonl (onpolicy/selection.py)")
    parser.add_argument("out", type=Path)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch", type=int, help="rows (commit) or pairs (dpo) per step; a smoke's")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--monitor-rows", type=int, default=MONITOR_ROWS)
    parser.add_argument("--monitor-every", type=int, default=MONITOR_EVERY)
    parser.add_argument("--stop", type=float, default=STOP, help="commit: the held-out commit_win to stop at")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "commit":
        record = commit(args.model, args.data, args.out, args.max_steps, args.batch, args.cpu, args.monitor_rows,
                        args.monitor_every, args.stop)
    else:
        record = dpo(args.model, args.data, args.out, args.max_steps, args.batch, args.cpu)
    (args.out / "train-log.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("steps", "written", "max_memory_allocated")}), flush=True)
