"""
Evaluation of a base_train run's newest checkpoint, written to BASE_EVAL/<tag>.json, or of its model-only
snapshot at --step (CHECKPOINTS/<tag>-step<step:06d>.pt), written to BASE_EVAL/<tag>-step<step:06d>.json (that
name is the result's "run"):
- train/metrics.loss_evaluation: val bits and bytes of the training mixture in file-level rows, of the repo-level
  mixture per context bucket in repo-level rows (models at that context), and of the HumanEval+, MBPP+ and HumanEval-X
  C++ reference solutions;
- pass@1 of HumanEval+ / MBPP+ (EvalPlus) and HumanEval-cpp / MBPP-cpp (MultiPL-E) and FIM pass@1 of HumanEval
  single-line, multi-line and random-span infilling, PSM and SPM (eval/code_eval.py), each by two generation modes
  (FIM's filtered mode in SPM order only): "greedy" and "filtered" (filtered pass@1: the Python syntax constraint, of
  the prefix and suffix for FIM, left-to-right prompts also healed, then seeded resampling until a candidate passes the
  static checks and its prompt's examples; eval/verify.py), with per-task diagnostics counts and the filtered mode's
  attempts; FIM middles cut to the reference's lines (eval/code_eval.py fim_lines).
Saved by train/metrics.save_evaluation (the ablations' evaluations, ablations/harness.py, take loss_evaluation only).

    python -m scripts.base_eval --model-tag=<tag> [--step=<step>]   (config.run_tag; scripts/production.py)
"""
import argparse
import warnings

import torch

from nanochat.data.tokenizer import get_tokenizer
from nanochat.eval.code_eval import fim_generation, generation, tasks
from nanochat.train.checkpoint import checkpoint_name, load_model
from nanochat.train.metrics import loss_evaluation, save_evaluation


def main() -> None:
    warnings.filterwarnings("ignore", category=SyntaxWarning)  # ast.parse of generated code (eval/verify.py, inference/)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--step", type=int, default=None, help="a model-only snapshot <tag>-step<step>.pt; default: newest")
    args = parser.parse_args()
    device = torch.device("cuda")
    benchmarks, tokenizer = tasks(), get_tokenizer()
    model, step = load_model(args.model_tag, step=args.step, device=device)
    save_evaluation(checkpoint_name(args.model_tag, step=args.step), step,
                    loss_evaluation(model, tokenizer, benchmarks, device=device) |
                    {"codegen": generation(model, tokenizer, benchmarks), "fim": fim_generation(model, tokenizer)})


if __name__ == "__main__":
    main()
