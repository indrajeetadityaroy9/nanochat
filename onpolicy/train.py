"""
Final-token preference optimization (FTPO, arXiv 2510.15061 §5, App. J) of a --chat export against its own loops, as
Antidoom applies it to reasoning loops (github.com/Liquid4All/antidoom, commit bd6a126; docs/post-training.md §8.5).

A row (onpolicy/selection.py ftpo.jsonl) is a loop's prefix through its first repeat's first token's position
(prefix_ids, BOS first) and that token (rejected_id). Its chosen tokens are the frozen reference's (the starting
model's) alternatives at the final position, picked as Antidoom picks them from the rollout's top log-probabilities
(src/antidoom/sampling.py:19-66 with configs/default.yaml): the TOP_LOGPROBS most likely tokens, less the rejected
token and any token whose decoded text equals the rejected token's in lower case, less those whose probability at the
rollout's temperature is below MIN_P of the most likely remaining one, less those whose decoded text is shorter than
MIN_DECODED_CHARS once stripped, at most CHOSEN_MAX. Over the set, chosen tokens are deduplicated per row by their
normalized word and the over-represented ones pruned (`regularise`, src/antidoom/ftpo_data.py:99-102, 256-357:
counts above max(CHOSEN_MIN_REF, their CHOSEN_REF_PERCENTILE-th percentile) cut to count (ref / count)^CHOSEN_STRENGTH,
first from rows holding other chosen tokens, then by dropping single-token rows); rows left with none are dropped.

The loss at the final position, on the model's capped logits y in fp32 (the head and cap outside autocast, as vLLM
computes them), against the reference's y_ref (`ftpo_loss`, App. J with the official trainer's normalization): with
Delta_c = y[c] - y[r] and w_c = clamp((MARGIN - Delta_c) / MARGIN, 0, 1), L_pref = sum_c w_c softplus(MARGIN -
Delta_c) / k over the row's k chosen tokens (src/antidoom/reference_ftpo_trainer.py:282-283; App. J divides by
sum_c w_c instead, equal while every w_c is 1); L_target = mean over T = C + {r} of max(|y - y_ref| - TAU_TARGET,
0)^2; L_nontarget = mean over the rest of the vocabulary of (y - y_ref)^2; L = L_pref + LAMBDA_TARGET L_target +
LAMBDA_NONTARGET L_nontarget. MARGIN, LAMBDA_TARGET, TAU_TARGET, LAMBDA_NONTARGET: configs/default.yaml
(clip_epsilon_logits, lambda_mse_target, tau_mse_target, lambda_mse).

Training (transformers' Trainer): full parameters loaded in fp32 (master weights and AdamW state), bf16 autocast,
each decoder layer recomputed in the backward pass (`load`), SDPA attention, one epoch at most, seed 0; batch BATCH
rows (Antidoom's 4 x 4 accumulation, as 1 x 16), AdamW (betas the library's), weight decay WEIGHT_DECAY, linear decay
after a WARMUP share of the steps, gradient norm clipped at MAX_GRAD_NORM (src/antidoom/ftpo_train.py:119-141); the
reference logits computed once, before the first update, by the same model. LR: see its constant. --freeze head
(or FREEZE=head) trains without the output head, the token embedding and every value embedding (`frozen`); none,
the default, trains every parameter.

Held-out monitor (`monitor_split`, `monitor`): before training, the rows of whole prompts, drawn in a
default_rng(SEED) permutation of the prompt ids until MONITOR_ROWS rows, leave the training set; at step 0, every
MONITOR_EVERY optimizer steps and at the last step their statistics are computed without gradient: the mean per row
of chosen_win (the share of the row's chosen tokens whose logit beats the rejected one), margin_win (the share ahead
by MARGIN), Delta (the mean Delta_c) and nontarget (L_nontarget). At the first monitor point (after step 0) whose
chosen_win reaches each of THRESHOLDS, the model is written as ftpo-<threshold>/ (Antidoom's README: "Around
0.15-0.3 ... strong reduction ... past 0.5 may be overtraining"); training stops after the last threshold's save; an
epoch that ends below it writes ftpo-final/. The training batches' statistics are logged every LOG_STEPS steps.

Each candidate is written as the export is (inference/export_hf.py: fp32 the parameters the forward reads in fp32,
bf16 the rest; config, generation config, tokenizer and model file copied), so vLLM serves it as it serves the
export. The bf16 tensors are rounded stochastically (`bf16`: E[rounded] = the fp32 value, a value on the bf16 grid
unchanged): an update smaller than half an element's bf16 spacing survives in expectation, where rounding to nearest
erases it. <out dir>/train-log.json holds the chosen sets' statistics, the monitor points ({step, chosen_win,
margin_win, delta, nontarget}), the candidates written (their step, monitor point and `retention`, the share of the
fp32 update the written weights keep along its direction), the log history (loss, its three terms, chosen_win,
margin_win, Delta of the training batches) and the peak GPU memory allocated.

`nll`: the mean per-token negative log-likelihood (capped logits at T 1, the model's forward, `nll`) of a fixed set
of responses given their prompts: a rollouts directory's passing responses (verified, closed and ended,
onpolicy/selection.py `load`), at most NLL_RESPONSES of them (a default_rng(SEED) draw in (prompt, sample) order); the
same directory for every model gives the same set. Writes {model, rollouts, responses, tokens, nll}.

    python -m onpolicy.train ftpo <model dir> <ftpo.jsonl> <out dir> [--max-steps N] [--batch N] [--cpu]
        [--freeze none|head] [--monitor-rows N] [--monitor-every N] [--thresholds 0.15,0.25,0.35]
    python -m onpolicy.train nll <model dir> <rollouts dir> <out json> [--cpu]
"""
import argparse
import functools
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

# chosen tokens: configs/default.yaml generation.top_logprobs, chosen_max_tokens, min_p (chosen_min_p null),
# chosen_min_decoded_chars; train.chosen_regularisation_strength; src/antidoom/ftpo_data.py:71-72
TOP_LOGPROBS, CHOSEN_MAX, MIN_P, MIN_DECODED_CHARS = 20, 20, 0.01, 1
CHOSEN_STRENGTH, CHOSEN_MIN_REF, CHOSEN_REF_PERCENTILE = 0.5, 50, 95
# the loss: configs/default.yaml train.clip_epsilon_logits, lambda_mse_target, tau_mse_target, lambda_mse
MARGIN, LAMBDA_TARGET, TAU_TARGET, LAMBDA_NONTARGET = 2.0, 0.05, 0.5, 0.4
# training: configs/default.yaml train.batch_size x gradient_accumulation_steps, warmup_ratio, weight_decay;
# src/antidoom/ftpo_train.py:126,140
BATCH, WARMUP, WEIGHT_DECAY = 16, 0.1, 0.01
LOG_STEPS, MAX_GRAD_NORM = 5, 2.5
# Full-parameter lr: the most selective measured (step-3000 export, CPU, 20 trained rows, constant lr, batch 4): the
# NLL of the export's own passing responses rose 3.7% / 11% / 59%, 0.012 / 0.016 / 0.042 NLL per logit of held-out
# Delta gained, at 5e-7 / 1.5e-6 / 6e-6 (docs/post-training.md §8.5)
LR = 5e-7
# held-out monitor and candidates: Antidoom's README chosen_win band (0.15-0.3 "strong reduction", past 0.5
# "overtraining")
MONITOR_ROWS, MONITOR_EVERY, THRESHOLDS, SEED = 512, 10, (0.15, 0.25, 0.35), 0
NLL_RESPONSES, NLL_CHUNK = 256, 1024  # nll: responses at most; positions per head chunk


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


def last_logits(model, ids: torch.Tensor) -> torch.Tensor:
    """The capped logits (V,) at the last position of ids (1, L): the backbone under the caller's autocast, the head
    and the cap in fp32 outside it (inference/modeling_nanochat.py NanochatForCausalLM.forward)."""
    h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, -1]
    with torch.autocast(h.device.type, enabled=False):
        y = F.linear(h.float(), model.lm_head.weight.float())
    c = model.config.final_logit_softcapping
    return c * torch.tanh(y / c)


def ftpo_loss(y: torch.Tensor, ref: torch.Tensor, chosen: torch.Tensor, rejected: int) -> tuple[torch.Tensor, dict]:
    """App. J's loss for one row (y, ref: (V,) fp32; chosen: (k,) ids), its preference term divided by the k chosen
    tokens as the official trainer divides it (reference_ftpo_trainer.py:282-283), and its terms and statistics."""
    delta = y[chosen] - y[rejected]
    w = ((MARGIN - delta) / MARGIN).clamp(0, 1)
    pref = (w * F.softplus(MARGIN - delta)).sum() / max(len(chosen), 1)
    targets = torch.zeros_like(y, dtype=torch.bool)
    targets[chosen] = True
    targets[rejected] = True
    d = y - ref
    target = ((d[targets].abs() - TAU_TARGET).clamp(min=0) ** 2).mean()
    nontarget = (d[~targets] ** 2).mean()
    loss = pref + LAMBDA_TARGET * target + LAMBDA_NONTARGET * nontarget
    with torch.no_grad():
        stats = {"pref": pref, "target": target, "nontarget": nontarget, "chosen_win": (delta > 0).float().mean(),
                 "margin_win": (delta >= MARGIN).float().mean(), "delta": delta.mean(), "active_weight": w.mean()}
    return loss, {k: float(v.detach()) for k, v in stats.items()}


def candidates(ref: torch.Tensor, rejected: int, temperature: float, decode) -> list[int]:
    """The chosen tokens of a row from its reference logits (src/antidoom/sampling.py:19-66): the TOP_LOGPROBS most
    likely, less the rejected token and its lower-case surface; p^(1/T) >= MIN_P max p^(1/T), i.e. a logit within
    T ln(1/MIN_P) of the best remaining; a stripped surface of MIN_DECODED_CHARS; at most CHOSEN_MAX, least likely
    first as the repository orders them."""
    surface = decode(rejected).lower()
    top = [int(i) for i in torch.topk(ref, TOP_LOGPROBS).indices]
    kept = [i for i in top if i != rejected and decode(i).lower() != surface]
    if not kept:
        return []
    floor = float(ref[kept[0]]) + max(temperature, 1e-6) * math.log(MIN_P)
    kept = [i for i in kept if float(ref[i]) >= floor]
    return [i for i in reversed(kept) if len(decode(i).strip()) >= MIN_DECODED_CHARS][:CHOSEN_MAX]


def key(text: str) -> str:
    """src/antidoom/ftpo_data.py:99-102 normalise_chosen_key."""
    stripped = text.strip().lower()
    return re.sub(r"^[^\w]+|[^\w]+$", "", stripped, flags=re.UNICODE) or stripped or text


def regularise(chosen: list[list[int]], decode) -> list[list[int]]:
    """src/antidoom/ftpo_data.py:256-357 _regularise_chosen_examples at CHOSEN_STRENGTH: each row's chosen tokens
    deduplicated by `key`; a key whose count exceeds ref = max(CHOSEN_MIN_REF, the CHOSEN_REF_PERCENTILE-th percentile
    count) is cut to round(count (ref / count)^CHOSEN_STRENGTH), first from the rows with the most chosen tokens
    (more than one), then by emptying rows holding only it (lowest index first). Returns the rows' chosen ids."""
    rows = []
    for ids in chosen:
        seen, row = set(), []
        for i in ids:
            if (k := key(decode(i))) not in seen:
                seen.add(k)
                row.append((i, k))
        rows.append(row)
    counts = Counter(k for row in rows for _, k in row)
    values = sorted(counts.values())
    ref = max(CHOSEN_MIN_REF, max(1, values[int((len(values) - 1) * CHOSEN_REF_PERCENTILE / 100)]) if values else 1)
    target = {k: max(1, round(c * (ref / c) ** CHOSEN_STRENGTH)) if CHOSEN_STRENGTH > 0 and c > ref else c
              for k, c in counts.items()}
    holders = defaultdict(set)
    for n, row in enumerate(rows):
        for _, k in row:
            holders[k].add(n)
    order = sorted(target, key=lambda k: counts[k] - target[k], reverse=True)
    for k in order:
        while counts[k] > target[k]:
            options = [n for n in holders[k] if any(x == k for _, x in rows[n]) and len(rows[n]) > 1]
            if not options:
                break
            n = max(options, key=lambda n: (len(rows[n]), -n))
            rows[n] = [(i, x) for i, x in rows[n] if x != k]
            counts[k] -= 1
    for k in sorted(target, key=lambda k: counts[k] - target[k], reverse=True):
        while counts[k] > target[k]:
            options = [n for n in holders[k] if [x for _, x in rows[n]] == [k]]
            if not options:
                break
            rows[min(options)] = []
            counts[k] -= 1
    return [[i for i, _ in row] for row in rows]


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
    """The trained model in the export's format beside the export's other files; its retention."""
    saved = served(model)
    kept = retention(load_file(source / "model.safetensors"), model, saved)
    out.mkdir(parents=True, exist_ok=True)
    save_file(saved, out / "model.safetensors", metadata={"format": "pt"})
    for f in source.iterdir():
        if f.is_file() and f.name != "model.safetensors":
            shutil.copy(f, out / f.name)
    return kept


def quantiles(values: list[float]) -> dict | None:
    return {"n": len(values), "p10_p50_p90": np.percentile(values, [10, 50, 90]).tolist()} if values else None


def frozen(model, freeze: str) -> list[str]:
    """--freeze head: the output head, the token embedding and every value embedding set requires_grad False (the
    Trainer's AdamW takes only the parameters that require it); none: nothing. The frozen parameters' names."""
    if freeze == "none":
        return []
    names = [n for n, p in model.named_parameters()
             if n == "lm_head.weight" or n.startswith(("model.embed_tokens.", "model.value_embeds."))]
    parameters = dict(model.named_parameters())
    for n in names:
        parameters[n].requires_grad_(False)
    return names


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


def monitor(model, examples: list[dict], device: torch.device) -> dict:
    """The held-out rows' mean per-row chosen_win, margin_win, Delta and nontarget, without gradient."""
    stats = defaultdict(list)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        for e in examples:
            y = last_logits(model, e["input_ids"].to(device))
            _, s = ftpo_loss(y, e["ref"].to(device), e["chosen"].to(device), e["rejected"])
            for k in ("chosen_win", "margin_win", "delta", "nontarget"):
                stats[k].append(s[k])
    return {k: float(np.mean(v)) for k, v in stats.items()}


def ftpo(model_dir: Path, data: Path, out: Path, max_steps: int, batch: int | None, cpu: bool, freeze: str,
         monitor_rows: int, monitor_every: int, thresholds: tuple[float, ...]) -> dict:
    from transformers import Trainer, TrainerCallback, TrainingArguments

    from corpus.tokenizer import get_tokenizer

    decode = functools.lru_cache(maxsize=None)(lambda i: get_tokenizer().enc.decode([i]))
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    model = load(model_dir).to(device).eval()
    refs = []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        for r in rows:
            refs.append(last_logits(model, torch.tensor([r["prefix_ids"]], device=device)).cpu())
    picked = [candidates(ref, r["rejected_id"], r["temperature"], decode) for r, ref in zip(rows, refs, strict=True)]
    chosen = regularise(picked, decode)
    examples = [{"input_ids": torch.tensor([r["prefix_ids"]]), "rejected": r["rejected_id"], "chosen": torch.tensor(c),
                 "ref": ref, "prompt_id": r["prompt_id"]} for r, c, ref in zip(rows, chosen, refs, strict=True) if c]
    train, held = monitor_split(examples, monitor_rows, SEED)
    sets = {"rows": len(rows), "no_candidates": sum(not c for c in picked), "kept": len(examples),
            "candidates": quantiles([len(c) for c in picked if c]), "chosen": quantiles([len(c) for c in chosen if c]),
            "pruned_by_regularisation": sum(map(len, picked)) - sum(map(len, chosen)), "train": len(train),
            "monitor": len(held), "monitor_prompts": len({e["prompt_id"] for e in held})}
    names = frozen(model, freeze)
    points, written = [], {}

    class FTPOTrainer(Trainer):
        stored = defaultdict(list)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, stats = ftpo_loss(last_logits(model, inputs["input_ids"]), inputs["ref"], inputs["chosen"],
                                    int(inputs["rejected"]))
            for k, v in stats.items():
                self.stored[k].append(v)
            return loss

        def log(self, logs, *args, **kwargs):
            logs = logs | {k: sum(v) / len(v) for k, v in self.stored.items() if v}
            self.stored.clear()
            return super().log(logs, *args, **kwargs)

    def point(model, step: int) -> dict:
        p = {"step": step} | monitor(model, held, device)
        p["saved"] = [] if step == 0 else [t for t in thresholds if t not in written and p["chosen_win"] >= t]
        for t in p["saved"]:
            written[t] = {"step": step, "retention": write(model, model_dir, out / f"ftpo-{t:g}")}
        points.append(p)
        print(json.dumps(p), flush=True)
        return p

    class Monitor(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            point(model, 0)

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step % monitor_every == 0 or state.global_step >= state.max_steps:
                point(model, state.global_step)
                if max(thresholds) in written:
                    control.should_training_stop = True

    args = TrainingArguments(
        output_dir=str(out / "trainer"), per_device_train_batch_size=1, gradient_accumulation_steps=batch or BATCH,
        learning_rate=LR, warmup_steps=WARMUP, lr_scheduler_type="linear", weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM, num_train_epochs=1, max_steps=max_steps, bf16=True, use_cpu=device.type == "cpu",
        logging_steps=LOG_STEPS, save_strategy="no", report_to="none", seed=0, data_seed=0,
        dataloader_num_workers=0, remove_unused_columns=False, gradient_checkpointing=False)
    model.train()
    trainer = FTPOTrainer(model=model, args=args, train_dataset=train, data_collator=lambda features: features[0],
                          callbacks=[Monitor()])
    trainer.loss_is_scaled_for_ga = False  # a row's loss is a mean: the step averages its rows
    trainer.train()
    candidates_written = {f"ftpo-{t:g}": v for t, v in written.items()}
    if max(thresholds) not in written:
        candidates_written["ftpo-final"] = {"step": trainer.state.global_step,
                                            "retention": write(trainer.model, model_dir, out / "ftpo-final")}
    return {"sets": sets, "lr": LR, "freeze": freeze, "frozen": names, "monitor_every": monitor_every,
            "thresholds": list(thresholds), "monitor": points, "candidates": candidates_written,
            "steps": trainer.state.global_step, "max_steps": trainer.state.max_steps,
            "max_memory_allocated": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            "log_history": trainer.state.log_history}


def nll(model_dir: Path, rollouts: Path, cpu: bool) -> dict:
    """The mean per-token NLL (capped logits, T 1) of the rollouts' passing responses (at most NLL_RESPONSES) given
    their prompts: the backbone under bf16 autocast, the head and cap in fp32 (`last_logits`), NLL_CHUNK positions at
    a time."""
    from onpolicy.selection import load as load_rollouts

    prompts, rows = load_rollouts([rollouts])
    passing = sorted((r for r in rows if r["success"]), key=lambda r: (r["prompt_id"], r["sample"]))
    if len(passing) > NLL_RESPONSES:
        keep = np.random.default_rng(SEED).choice(len(passing), size=NLL_RESPONSES, replace=False)
        passing = [passing[i] for i in sorted(keep.tolist())]
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    model = load(model_dir, train=False).to(device).eval()
    c, head = model.config.final_logit_softcapping, model.lm_head.weight.float()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for r in passing:
            context = prompts[r["prompt_id"]]["ids"]
            ids = torch.tensor([context + list(r["ids"])], device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
            h, target = h[len(context) - 1:-1].float(), ids[0, len(context):]
            for a in range(0, len(target), NLL_CHUNK):
                y = F.linear(h[a:a + NLL_CHUNK], head)
                total += float(F.cross_entropy(c * torch.tanh(y / c), target[a:a + NLL_CHUNK], reduction="sum"))
            tokens += len(target)
    return {"model": str(model_dir), "rollouts": str(rollouts), "responses": len(passing), "tokens": tokens,
            "nll": total / tokens if tokens else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("ftpo", "nll"))
    parser.add_argument("model", type=Path)
    parser.add_argument("data", type=Path, help="ftpo: ftpo.jsonl; nll: a rollouts directory with verdicts")
    parser.add_argument("out", type=Path, help="ftpo: the out directory; nll: the out json")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch", type=int, help="rows per step; a smoke's")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--freeze", choices=("none", "head"), default=os.environ.get("FREEZE") or "none")
    parser.add_argument("--monitor-rows", type=int, default=MONITOR_ROWS)
    parser.add_argument("--monitor-every", type=int, default=MONITOR_EVERY)
    parser.add_argument("--thresholds", type=lambda s: tuple(sorted(map(float, s.split(",")))), default=THRESHOLDS)
    args = parser.parse_args()
    if args.stage == "nll":
        result = nll(args.model, args.data, args.cpu)
        args.out.write_text(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        record = ftpo(args.model, args.data, args.out, args.max_steps, args.batch, args.cpu, args.freeze,
                      args.monitor_rows, args.monitor_every, args.thresholds)
        (args.out / "train-log.json").write_text(json.dumps(record, indent=1) + "\n")
        print(json.dumps({k: record[k] for k in ("sets", "steps", "candidates", "max_memory_allocated")}), flush=True)
