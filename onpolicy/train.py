"""
RSFT and DPO of a --chat export with TRL (docs/post-training.md §8.5), full parameters: the model loaded in fp32 (master
weights and AdamW state), bf16 autocast, the LM head and its cap in fp32 as the evaluation computes them
(inference/modeling_nanochat.py; TRL's fused head with cast_lm_head_to_fp32), SDPA attention, each decoder layer
recomputed in the backward pass (`load`); one epoch, no dropout
(the model has none), seed 0. The sets are onpolicy/selection.py's token ids, used as sampled (TRL's tokenization is
skipped: it would append an eos to the rejected side).

- rsft: SFTTrainer on rsft.jsonl, the loss on the completion (completion_only_loss): its tokens' mean over a batch of
  RSFT_BATCH sequences; AdamW at RSFT_LR, warmup RSFT_WARMUP of the steps, cosine decay. Phi-4-mini-reasoning's
  distillation stages: batch 128, lr 1e-5, warmup ratio 0.1 (arXiv 2504.21233 §5.3).
- dpo: DPOTrainer on pairs.jsonl, the sigmoid loss (DPO, arXiv 2305.18290) only: no likelihood term on the chosen
  side (DeLoop's NLL anchor raised loops 6.1% -> 27.0%, arXiv 2610.19376 App. G.13), against the starting model as
  the frozen reference; beta DPO_BETA (DeLoop's, chosen from {0.02, 0.05, 0.1, 0.2, 0.5} on a screening set, App.
  G.1, the lowest loop rate of those values, App. G.6); AdamW at DPO_LR (Phi-4-mini-reasoning's rollout DPO with full
  parameters, one epoch at 16K, §5.3; DeLoop's LoRA lr 8e-6 is 16x that, against LoRA's optimum of about 10x the
  full-parameter lr, docs/post-training.md §2.2); batch DPO_BATCH pairs (DeLoop's 16), warmup DPO_WARMUP, linear
  decay.

The result is written as the export is (inference/export_hf.py: fp32 the parameters the forward reads in fp32, bf16
the rest; config, generation config, tokenizer and model file copied), so vLLM serves it as it serves the export. The
bf16 tensors are rounded stochastically (`bf16`: E[rounded] = the fp32 value, a value on the bf16 grid unchanged): an
update smaller than half an element's bf16 spacing, which most of DPO's are, survives in expectation, where rounding
to nearest erases it. train-log.json holds TRL's log history, `retention`: the share of the fp32 update (final -
start) the written weights keep along its direction, and the peak GPU memory allocated.

    python -m onpolicy.train rsft|dpo <model dir> <set jsonl> <out dir> [--max-steps N] [--batch N] [--cpu]
"""
import argparse
import functools
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

RSFT_LR, RSFT_BATCH, RSFT_WARMUP = 1e-5, 128, 0.1
DPO_LR, DPO_BATCH, DPO_WARMUP, DPO_BETA = 5e-7, 16, 0.1, 0.05


def load(model_dir: Path, train: bool = True):
    """The model in fp32; to train, each decoder layer recomputed in the backward pass (torch's non-reentrant
    checkpoint, autocast state kept): 0.68 MB of saved activations per token instead of 2.04, with identical gradients
    (a 4,096-token sequence, CPU, bf16 autocast); on the GPU 0.235 MB per token of each sequence of a DPO pair
    (scripts/onpolicy.sh `pair_tokens`). The export's model file declares no gradient checkpointing for transformers
    to enable."""
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.float32,
                                                 attn_implementation="sdpa")
    if train:
        for layer in model.model.layers:
            layer.forward = functools.partial(checkpoint, layer.forward, use_reentrant=False)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token  # padding is masked out of attention and loss
    return model, tokenizer


def fp32_heads() -> None:
    """TRL's fused LM head with the projection and the cap in fp32, in both trainers."""
    from trl.trainer import dpo_trainer, sft_trainer, utils

    for module in (dpo_trainer, sft_trainer):
        module.add_fused_lm_head = functools.partial(utils.add_fused_lm_head, cast_lm_head_to_fp32=True)


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


def write(trainer, source: Path, out: Path) -> dict:
    """The trained model in the export's format beside the export's other files; its retention."""
    saved = served(trainer.model)
    kept = retention(load_file(source / "model.safetensors"), trainer.model, saved)
    out.mkdir(parents=True, exist_ok=True)
    save_file(saved, out / "model.safetensors", metadata={"format": "pt"})
    for f in source.iterdir():
        if f.is_file() and f.name != "model.safetensors":
            shutil.copy(f, out / f.name)
    return {"retention": kept, "max_memory_allocated": torch.cuda.max_memory_allocated() if torch.cuda.is_available()
            else None, "log_history": trainer.state.log_history}


def common(out: Path, batch: int, lr: float, warmup: float, max_steps: int, cpu: bool) -> dict:
    return {"output_dir": str(out / "trainer"), "per_device_train_batch_size": 1, "gradient_accumulation_steps": batch,
            "learning_rate": lr, "warmup_steps": warmup, "num_train_epochs": 1, "max_steps": max_steps,
            "adam_beta1": 0.9, "adam_beta2": 0.95, "weight_decay": 0.0, "max_grad_norm": 1.0, "bf16": True,
            "use_cpu": cpu, "logging_steps": 1, "save_strategy": "no", "report_to": "none", "seed": 0,
            "data_seed": 0, "dataloader_num_workers": 0, "remove_unused_columns": False,
            "gradient_checkpointing": False}


def rsft(model_dir: Path, data: Path, out: Path, max_steps: int, batch: int | None, cpu: bool) -> dict:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    fp32_heads()
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    dataset = Dataset.from_list([{"input_ids": r["prompt_ids"] + r["completion_ids"],
                                  "completion_mask": [0] * len(r["prompt_ids"]) + [1] * len(r["completion_ids"])}
                                 for r in rows])
    model, tokenizer = load(model_dir)
    args = SFTConfig(**common(out, batch or RSFT_BATCH, RSFT_LR, RSFT_WARMUP, max_steps, cpu),
                     lr_scheduler_type="cosine", completion_only_loss=True, max_length=None, packing=False)
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
    trainer.train()
    return write(trainer, model_dir, out)


def dpo(model_dir: Path, data: Path, out: Path, max_steps: int, batch: int | None, cpu: bool) -> dict:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    class TokenizedDPOTrainer(DPOTrainer):
        def _prepare_dataset(self, dataset, *args, **kwargs):
            return dataset  # pairs.jsonl's ids as sampled

    fp32_heads()
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    dataset = Dataset.from_list([{k: r[k] for k in ("prompt_ids", "chosen_ids", "rejected_ids")} for r in rows])
    model, tokenizer = load(model_dir)
    reference, _ = load(model_dir, train=False)
    args = DPOConfig(**common(out, batch or DPO_BATCH, DPO_LR, DPO_WARMUP, max_steps, cpu),
                     lr_scheduler_type="linear", beta=DPO_BETA, loss_type=["sigmoid"], max_length=None,
                     precompute_ref_log_probs=False)
    trainer = TokenizedDPOTrainer(model=model, ref_model=reference, args=args, train_dataset=dataset,
                                  processing_class=tokenizer)
    trainer.train()
    return write(trainer, model_dir, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("rsft", "dpo"))
    parser.add_argument("model", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch", type=int, help="sequences (rsft) or pairs (dpo) per step; a smoke's")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    record = {"rsft": rsft, "dpo": dpo}[args.stage](args.model, args.data, args.out, args.max_steps, args.batch,
                                                    args.cpu)
    (args.out / "train-log.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("retention", "max_memory_allocated")}), flush=True)
