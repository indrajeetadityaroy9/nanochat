"""
Evaluation of a base_train run's newest checkpoint, written to <base_dir>/base_eval/<tag>.json:
- loss_evaluation: val bits and bytes of the training mixture in file-level rows, of the repo-level mixture per context
  bucket in repo-level rows (models at that context), and of the HumanEval+, MBPP+ and HumanEval-X C++ reference
  solutions (eval/loss_eval.py, eval/code_eval.py);
- greedy pass@1 of HumanEval+ / MBPP+ (EvalPlus) and HumanEval-cpp / MBPP-cpp (MultiPL-E) (eval/code_eval.py);
- FIM pass@1 of HumanEval single-line, multi-line and random-span infilling, PSM and SPM (eval/fim_eval.py).

    python -m scripts.base_eval --model-tag=<tag>   (config.run_tag; runs/h100.sh logs it)
"""
import argparse
import os

import torch

from nanochat.config import FILE_STAGE, MIXTURE, REPO_STAGE
from nanochat.data.tokenizer import Tokenizer, get_tokenizer
from nanochat.eval.code_eval import generation, solution_bits, tasks
from nanochat.eval.fim_eval import fim_generation
from nanochat.eval.loss_eval import val_bits
from nanochat.model.gpt import GPT
from nanochat.paths import base_dir, write_json
from nanochat.train.checkpoint import load_model


def loss_evaluation(model: GPT, tokenizer: Tokenizer, benchmarks: dict, *, device: torch.device) -> dict:
    """{"8k": val_bits of config.MIXTURE in file-level rows, "32k": val_bits of REPO_STAGE's mixture (models at its
    context), "solution": code_eval.solution_bits} by the compiled GPT.losses, one graph per row shape."""
    losses = torch.compile(model.losses, dynamic=False, fullgraph=True)
    results = {"8k": val_bits(losses, tokenizer, MIXTURE, stage=FILE_STAGE, device=device),
               "solution": solution_bits(losses, tokenizer, benchmarks, batch_rows=FILE_STAGE.batch_rows,
                                         seq_len=FILE_STAGE.seq_len, device=device)}
    if model.config.sequence_len == REPO_STAGE.seq_len:
        results["32k"] = val_bits(losses, tokenizer, REPO_STAGE.mixture, stage=REPO_STAGE, device=device)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", required=True)
    tag = parser.parse_args().model_tag
    device = torch.device("cuda")
    benchmarks, tokenizer = tasks(), get_tokenizer()
    model, step = load_model(tag, step=None, device=device)
    results = {"run": tag, "step": step, **loss_evaluation(model, tokenizer, benchmarks, device=device),
               "codegen": generation(model, tokenizer, benchmarks), "fim": fim_generation(model, tokenizer)}
    os.makedirs(os.path.join(base_dir(), "base_eval"), exist_ok=True)
    write_json(os.path.join(base_dir(), "base_eval", f"{tag}.json"), results)


if __name__ == "__main__":
    main()
