"""
Evaluation of a base_train run's final checkpoint (its raw weights, train/checkpoint.load_model):
- bits per byte of each training-mixture corpus's val split (eval/loss_eval.py);
- reference-solution bits per byte of HumanEval+, MBPP+ and HumanEval-X C++ (eval/code_eval.py);
- greedy pass@1 of HumanEval+ / MBPP+ (EvalPlus base and plus) and HumanEval-cpp / MBPP-cpp (MultiPL-E)
  (eval/code_eval.py);
- fill-in-the-middle pass@1 of HumanEval single-line, multi-line and random-span infilling, PSM and SPM prompts
  (eval/fim_eval.py).
Written to <base_dir>/base_eval/<tag>.json.

    python -m scripts.base_eval --model-tag=d24-r490.805-s42-g1
"""
import argparse
import os

import torch

from nanochat.config import FILE_STAGE, MIXTURE
from nanochat.data.loader import mixture_corpora
from nanochat.eval.code_eval import generation, solution_bpb, tasks
from nanochat.eval.fim_eval import fim_generation
from nanochat.eval.loss_eval import val_bpb
from nanochat.paths import base_dir, write_json
from nanochat.train.checkpoint import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", required=True)
    tag = parser.parse_args().model_tag
    device = torch.device("cuda")
    benchmarks = tasks()
    model, tokenizer, step = load_model(device, tag)
    # per-position losses from the compiled training forward (GPT.losses), every batch FILE_STAGE.batch_rows x
    # FILE_STAGE.seq_len: one graph scores every corpus's bpb (cu_seqlens padded to one length), one more the solution
    # bpb (a segment per row)
    losses = torch.compile(model.losses, dynamic=False, fullgraph=True)
    results = {"run": tag, "step": step,
               "bpb": val_bpb(losses, tokenizer, mixture_corpora(MIXTURE), stage=FILE_STAGE, device=device),
               "solution_bpb": solution_bpb(losses, tokenizer, device, FILE_STAGE.batch_rows, FILE_STAGE.seq_len,
                                            benchmarks),
               "codegen": generation(model, tokenizer, benchmarks),
               "fim": fim_generation(model, tokenizer)}
    os.makedirs(os.path.join(base_dir(), "base_eval"), exist_ok=True)
    write_json(os.path.join(base_dir(), "base_eval", f"{tag}.json"), results)


if __name__ == "__main__":
    main()
