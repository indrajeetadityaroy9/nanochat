"""
Evaluation of a base_train run's final checkpoint (its raw weights, train/checkpoint.load_model):
- loss_evaluation: the val bits and bytes of every training-mixture corpus in rows of the file-level stage, of every
  repository corpus in rows of the repo-level stage per context bucket (a model trained at its context), and of the
  reference solutions of HumanEval+, MBPP+ and HumanEval-X C++ (eval/loss_eval.py, eval/code_eval.py);
- greedy pass@1 of HumanEval+ / MBPP+ (EvalPlus base and plus) and HumanEval-cpp / MBPP-cpp (MultiPL-E)
  (eval/code_eval.py);
- fill-in-the-middle pass@1 of HumanEval single-line, multi-line and random-span infilling, PSM and SPM prompts
  (eval/fim_eval.py).
Written to <base_dir>/base_eval/<tag>.json.

    python -m scripts.base_eval --model-tag=d24-r490.805-s42
"""
import argparse
import os

import torch

from nanochat.config import FILE_STAGE, MIXTURE, REPO_STAGE
from nanochat.data.tokenizer import Tokenizer
from nanochat.eval.code_eval import generation, solution_bits, tasks
from nanochat.eval.fim_eval import fim_generation
from nanochat.eval.loss_eval import val_bits
from nanochat.model.gpt import GPT
from nanochat.paths import base_dir, write_json
from nanochat.train.checkpoint import load_model


def loss_evaluation(model: GPT, tokenizer: Tokenizer, benchmarks: dict, device: torch.device) -> dict:
    """{"8k": val_bits of config.MIXTURE in file-level rows, "32k": val_bits of the repo-level stage's mixture in its
    rows (only for a model at that context: one trained in the repo-level stage), "solution": code_eval.solution_bits}
    by the per-position losses of the compiled training forward (GPT.losses): one graph per row shape (cu_seqlens padded
    to one length per stage; a segment per row for the solutions)."""
    losses = torch.compile(model.losses, dynamic=False, fullgraph=True)
    results = {"8k": val_bits(losses, tokenizer, MIXTURE, stage=FILE_STAGE, device=device),
               "solution": solution_bits(losses, tokenizer, device, FILE_STAGE.batch_rows, FILE_STAGE.seq_len,
                                         benchmarks)}
    if model.config.sequence_len == REPO_STAGE.seq_len:
        results["32k"] = val_bits(losses, tokenizer, REPO_STAGE.mixture, stage=REPO_STAGE, device=device)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", required=True)
    tag = parser.parse_args().model_tag
    device = torch.device("cuda")
    benchmarks = tasks()
    model, tokenizer, step = load_model(device, tag)
    results = {"run": tag, "step": step, **loss_evaluation(model, tokenizer, benchmarks, device),
               "codegen": generation(model, tokenizer, benchmarks), "fim": fim_generation(model, tokenizer)}
    os.makedirs(os.path.join(base_dir(), "base_eval"), exist_ok=True)
    write_json(os.path.join(base_dir(), "base_eval", f"{tag}.json"), results)


if __name__ == "__main__":
    main()
