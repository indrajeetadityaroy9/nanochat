"""Faithfulness of the serving path at long context: the --chat export (inference/export_hf.py) in vLLM (the termination
probe's engine: bf16, the head in fp32, max_model_len the context, seed 0) against the training model
(train.checkpoint.load_model, bf16) of a post-trained checkpoint, on SFT val documents.

- Prefill: per-token NLL of the responses of one val document per target length in TARGETS (context + response, the
  nearest of corpus/posttrain/build.py val_spans), its sequence BOS + document + closing BOS (vLLM's cut to the context
  less one), scored on its response labels ([start + context, end) of its row): the training model's model.losses on
  the val rows as train/stages.py val_batches feeds validation (each document under its own cu_seqlens) against vLLM's
  prompt logprobs of the same ids. Per position bucket (BUCKETS, the token's index in the sequence): mean NLL by each
  path, their mean and max absolute difference; per document and pooled.
- Decode (incremental, the paged KV cache): the DECODE val prompts with the longest teacher responses generated greedily
  by vLLM (T 0, at most DECODE_TOKENS, its logprob of each chosen token), then the training model's teacher-forced
  forward on prompt + generation (`forward_logprobs`: model.hidden, then the head as model.losses computes it). Per
  bucket of the generated token's index (DECODE_BUCKETS): the share of tokens equal to the training model's argmax; at
  disagreements the training model's log-prob margin of its argmax over vLLM's token (mean, max, share above
  NEAR_TIE); the mean log-prob of the chosen tokens by vLLM and by the training model.

The export's 18 sliding-window layers (513) and 6 full ones are honoured only if the differences stay flat past 512.

    python -m eval.serving_check train <run name> <step> <out json>       (the nanochat image)
    python -m eval.serving_check vllm <model dir> <train json> <out json>  (the nanochat-eval image)
    python -m eval.serving_check decode <run name> <step> <vllm json> <out json>  (the nanochat image)
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np

TARGETS = (1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768)
BUCKETS = (0, 512, 2048, 8192, 16384, 32768)
DECODE, DECODE_TOKENS, NEAR_TIE = 4, 8192, 0.05
DECODE_BUCKETS = (0, 512, 2048, 4096, 8192)


def train(run_name: str, step: int, out: Path) -> None:
    import torch

    from corpus.posttrain.build import labels_path, val_spans
    from corpus.posttrain.sft import SFT_STAGE
    from corpus.tokenizer import get_tokenizer
    from train.checkpoint import load_model
    from train.stages import val_batches

    spans = val_spans()
    labels = [json.loads(line) for line in labels_path().read_text(encoding="utf-8").splitlines()]
    lengths = np.array([end - start for _, start, end, _ in spans])
    chosen = []
    for target in TARGETS:
        order = np.argsort(np.abs(lengths - target), kind="stable")
        chosen.append(int(next(d for d in order if d not in chosen)))
    # the longest teacher responses (the hardest problems, the longest generations) among the documents whose prompt
    # (context + 1 tokens) leaves room for DECODE_TOKENS in the context
    fits = [d for d in np.argsort([start + c - end for _, start, end, c in spans], kind="stable")
            if spans[d][3] + 1 + DECODE_TOKENS <= SFT_STAGE.seq_len]
    decoded = [int(d) for d in fits[:DECODE]]
    rows = {}
    for d in {*chosen, *decoded}:
        rows.setdefault(spans[d][0], []).append(d)
    device = torch.device("cuda")
    model = load_model(run_name, step, device=device)
    losses = torch.compile(model.losses, dynamic=False, fullgraph=True)
    documents, micro = {}, SFT_STAGE.micro_batch_size
    with torch.no_grad():
        for b, (input_ids, label_ids, cu_seqlens, segments, _) in enumerate(
                val_batches(get_tokenizer(), stage=SFT_STAGE, device=device)):
            wanted = [r for r in rows if r // micro == b]
            if not wanted:
                if b * micro > max(rows):
                    break
                continue
            loss = losses(input_ids, label_ids, cu_seqlens, segments).view(micro, -1).cpu().numpy()
            inputs, targets = input_ids.cpu().numpy(), label_ids.cpu().numpy()
            for r in wanted:
                for d in rows[r]:
                    _, start, end, context = spans[d]
                    i = r % micro
                    sequence = [*inputs[i, start:end].tolist(), int(targets[i, end - 1])]
                    assert (targets[i, start + context:end - 1] == inputs[i, start + context + 1:end]).all()
                    documents[d] = {"document": d, "group": labels[d]["group"], "style": labels[d]["style"],
                                    "tokens": len(sequence), "context": context, "ids": sequence,
                                    # the NLL of sequence[j] for j in [context + 1, len): row position start + j - 1
                                    "train_nll": loss[i, start + context:end].tolist()}
    out.write_text(json.dumps({"run": run_name, "step": step, "documents": [documents[d] for d in chosen],
                               "decode": [{"document": d, "prompt": documents[d]["ids"][:spans[d][3] + 1]}
                                          for d in decoded]}) + "\n")


def stats(position: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    out = {}
    for lo, hi in itertools.pairwise(BUCKETS):
        m = (position >= lo) & (position < hi)
        if m.any():
            out[f"{lo}-{hi}"] = {"tokens": int(m.sum()), "train": float(a[m].mean()), "vllm": float(b[m].mean()),
                                 "difference": float((b[m] - a[m]).mean()),
                                 "max_abs_difference": float(np.abs(b[m] - a[m]).max())}
    return out


def vllm(model: Path, inp: Path, out: Path) -> None:
    from vllm import LLM, SamplingParams

    data = json.loads(inp.read_text())
    context = json.loads((model / "config.json").read_text())["max_position_embeddings"]
    llm = LLM(str(model), trust_remote_code=True, dtype="bfloat16", max_model_len=context, seed=0)
    prompts = [d["ids"][:context - 1] for d in data["documents"]]
    outputs = llm.generate([{"prompt_token_ids": p} for p in prompts],
                           SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0), use_tqdm=False)
    pooled, report = [], []
    for d, p, output in zip(data["documents"], prompts, outputs, strict=True):
        position = np.arange(d["context"] + 1, len(p))
        served = np.array([-output.prompt_logprobs[j][p[j]].logprob for j in position])
        trained = np.array(d["train_nll"][:len(position)])
        pooled.append((position, trained, served))
        report.append({k: d[k] for k in ("document", "group", "style", "tokens", "context")}
                      | {"buckets": stats(position, trained, served)})
    position, trained, served = (np.concatenate(x) for x in zip(*pooled, strict=True))
    generated = llm.generate([{"prompt_token_ids": d["prompt"]} for d in data["decode"]],
                             SamplingParams(max_tokens=DECODE_TOKENS, temperature=0, logprobs=1), use_tqdm=False)
    decode = []
    for d, output in zip(data["decode"], generated, strict=True):
        (completion,) = output.outputs
        ids = list(completion.token_ids)
        decode.append(d | {"generated": ids, "finish": completion.finish_reason,
                           "vllm_logprob": [lp[t].logprob for lp, t in zip(completion.logprobs, ids, strict=True)]})
    out.write_text(json.dumps({"run": data["run"], "step": data["step"], "model": str(model), "context": context,
                               "prefill": {"pooled": stats(position, trained, served), "documents": report},
                               "decode": decode}) + "\n")


def forward_logprobs(model, ids: list[int], targets: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The training model's teacher-forced next-token log-probs after each of ids' last len(targets) positions: (its
    argmax, the argmax's log-prob, the target's log-prob); the head as model.losses computes it (bf16 operands, fp32
    product, then the cap), the sequence alone in a row of the SFT context."""
    import torch

    from corpus.posttrain.sft import SFT_STAGE
    from corpus.tokenizer import get_tokenizer
    from model.ops.cross_entropy import CHUNK

    device, length, rows = torch.device("cuda"), SFT_STAGE.seq_len, len(ids)
    x = torch.full((1, length), get_tokenizer().bos_token_id, dtype=torch.int32, device=device)
    x[0, :rows] = torch.tensor(ids, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, rows, length], dtype=torch.int32, device=device)
    hidden = model.hidden(x, cu_seqlens, torch.tensor(2, dtype=torch.int32))[rows - 1 - len(targets):rows - 1]
    w, c = model.lm_head.weight.to(torch.bfloat16).t(), model.config.final_logit_softcapping
    t = torch.tensor(targets, device=device)
    best, best_lp, target_lp = [], [], []
    for h, tc in zip(hidden.split(CHUNK), t.split(CHUNK), strict=True):
        z = torch.mm(h, w, out_dtype=torch.float32)
        lp = torch.log_softmax(c * torch.tanh(z / c), dim=1)
        top = lp.max(dim=1)
        best.append(top.indices)
        best_lp.append(top.values)
        target_lp.append(lp.gather(1, tc[:, None]).squeeze(1))
    return tuple(torch.cat(v).cpu().numpy() for v in (best, best_lp, target_lp))


def decode_stats(position: np.ndarray, agree: np.ndarray, margin: np.ndarray, served: np.ndarray,
                 trained: np.ndarray) -> dict:
    out = {}
    for lo, hi in itertools.pairwise(DECODE_BUCKETS):
        m = (position >= lo) & (position < hi)
        if m.any():
            off = margin[m & ~agree]
            out[f"{lo}-{hi}"] = {"tokens": int(m.sum()), "argmax_agreement": float(agree[m].mean()),
                                 "disagreements": len(off), "margin_mean": float(off.mean()) if len(off) else None,
                                 "margin_max": float(off.max()) if len(off) else None,
                                 "margin_above_near_tie": int((off > NEAR_TIE).sum()),
                                 "vllm_logprob_mean": float(served[m].mean()),
                                 "train_logprob_mean": float(trained[m].mean()),
                                 "logprob_max_abs_difference": float(np.abs(served[m] - trained[m]).max())}
    return out


def decode(run_name: str, step: int, inp: Path, out: Path) -> None:
    import torch

    from train.checkpoint import load_model

    data = json.loads(inp.read_text())
    model = load_model(run_name, step, device=torch.device("cuda"))
    pooled, report = [], []
    with torch.no_grad():
        for d in data.pop("decode"):
            generated = d["generated"]
            best, best_lp, target_lp = forward_logprobs(model, d["prompt"] + generated, generated)
            position, agree = np.arange(len(generated)), best == np.array(generated)
            row = (position, agree, best_lp - target_lp, np.array(d["vllm_logprob"]), target_lp)
            pooled.append(row)
            report.append({"document": d["document"], "prompt_tokens": len(d["prompt"]),
                           "generated_tokens": len(generated), "finish": d["finish"], "buckets": decode_stats(*row)})
    data["decode"] = {"pooled": decode_stats(*(np.concatenate(x) for x in zip(*pooled, strict=True))),
                      "documents": report}
    out.write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]))
    elif sys.argv[1] == "vllm":
        vllm(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        decode(sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
