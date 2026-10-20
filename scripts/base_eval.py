"""
Evaluation of a base_train run's newest checkpoint, written to BASE_EVAL/<tag>.json, or of its model-only
snapshot at --step (CHECKPOINTS/<tag>-step<step:06d>.pt), written to BASE_EVAL/<tag>-step<step:06d>.json (that
name is the result's "run" and scripts/report.py's tag):
- loss_evaluation: val bits and bytes of the training mixture in file-level rows, of the repo-level mixture per context
  bucket in repo-level rows (models at that context), and of the HumanEval+, MBPP+ and HumanEval-X C++ reference
  solutions (eval/loss_eval.py, eval/code_eval.py);
- pass@1 of HumanEval+ / MBPP+ (EvalPlus) and HumanEval-cpp / MBPP-cpp (MultiPL-E) (eval/code_eval.py) and FIM pass@1
  of HumanEval single-line, multi-line and random-span infilling, PSM and SPM (eval/fim_eval.py), each by two
  generation modes (FIM's system mode in SPM order only): "greedy" and "system" (healing, the Python syntax constraint,
  then seeded resampling until a candidate passes the static checks and its prompt's examples; eval/verify.py), with
  per-task diagnostics counts and the system mode's attempts.
A result is one JSON object: schema (SCHEMA), run, step, then those; a measurement not taken is null (the ablations'
evaluations, ablations/study.py, take loss_evaluation only).

    python -m scripts.base_eval --model-tag=<tag> [--step=<step>]   (config.run_tag; runs/h100.sh logs it)
"""
import argparse
import json

import torch

from nanochat.config import FILE_STAGE, MIXTURE, REPO_STAGE
from nanochat.data.tokenizer import Tokenizer, get_tokenizer
from nanochat.eval.code_eval import generation, solution_bits, tasks
from nanochat.eval.fim_eval import fim_generation
from nanochat.eval.loss_eval import val_bits
from nanochat.model.gpt import GPT
from nanochat.paths import BASE_EVAL
from nanochat.train.checkpoint import checkpoint_name, load_model

SCHEMA = 1


def save(run: str, step: int, results: dict) -> None:
    """BASE_EVAL/<run>.json; NaN and infinities are not JSON and raise."""
    (BASE_EVAL / f"{run}.json").write_text(json.dumps({"schema": SCHEMA, "run": run, "step": step} | results,
                                                      allow_nan=False), encoding="utf-8")


def loss_evaluation(model: GPT, tokenizer: Tokenizer, benchmarks: dict, *, device: torch.device) -> dict:
    """{"8k": val_bits of config.MIXTURE in file-level rows, "32k": val_bits of REPO_STAGE's mixture (null for a model
    not at its context), "solution": code_eval.solution_bits} by the compiled GPT.losses, one graph per row shape."""
    losses = torch.compile(model.losses, dynamic=False, fullgraph=True)
    return {"8k": val_bits(losses, tokenizer, MIXTURE, stage=FILE_STAGE, device=device),
            "32k": (val_bits(losses, tokenizer, REPO_STAGE.mixture, stage=REPO_STAGE, device=device)
                    if model.config.sequence_len == REPO_STAGE.seq_len else None),
            "solution": solution_bits(losses, tokenizer, benchmarks, batch_rows=FILE_STAGE.batch_rows,
                                      seq_len=FILE_STAGE.seq_len, device=device)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--step", type=int, default=None, help="a model-only snapshot <tag>-step<step>.pt; default: newest")
    args = parser.parse_args()
    run = checkpoint_name(args.model_tag, step=args.step)
    device = torch.device("cuda")
    benchmarks, tokenizer = tasks(), get_tokenizer()
    model, step = load_model(args.model_tag, step=args.step, device=device)
    save(run, step, loss_evaluation(model, tokenizer, benchmarks, device=device) |
         {"codegen": generation(model, tokenizer, benchmarks), "fim": fim_generation(model, tokenizer)})


if __name__ == "__main__":
    main()
