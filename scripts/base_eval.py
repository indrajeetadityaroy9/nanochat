"""
Evaluation of a run's checkpoint: evaluate(tag, step=None) of its newest checkpoint, written to BASE_EVAL/<tag>.json,
or evaluate(tag, step=s) of its model-only snapshot at step s (CHECKPOINTS/<tag>-step<s:06d>.pt), written to
BASE_EVAL/<tag>-step<s:06d>.json (that name is the result's "run"):
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
Caller: scripts/production.py, one process per checkpoint.
"""
import warnings

import torch

from nanochat.data.tokenizer import get_tokenizer
from nanochat.eval.code_eval import fim_generation, generation, tasks
from nanochat.train.checkpoint import checkpoint_name, load_model
from nanochat.train.metrics import loss_evaluation, save_evaluation


def evaluate(tag: str, *, step: int | None) -> None:
    """BASE_EVAL/<checkpoint_name>.json of run `tag`'s checkpoint: its newest (step None) or its snapshot at `step`."""
    warnings.filterwarnings("ignore", category=SyntaxWarning)  # ast.parse of generated code (eval/verify.py, inference/)
    device = torch.device("cuda")
    benchmarks, tokenizer = tasks(), get_tokenizer()
    model, trained_steps = load_model(tag, step=step, device=device)
    save_evaluation(checkpoint_name(tag, step=step), trained_steps,
                    loss_evaluation(model, tokenizer, benchmarks, device=device) |
                    {"codegen": generation(model, tokenizer, benchmarks), "fim": fim_generation(model, tokenizer)})
