"""
Evaluation of a base_train run's final checkpoint, for both its raw weights and its tail average
(train/checkpoint.load_model's estimators):
- bits per byte of each training-mixture corpus's val split (eval/loss_eval.py), both estimators;
- reference-solution bits per byte of HumanEval+, MBPP+ and HumanEval-X C++ (eval/code_eval.py), both estimators;
- tail average only (the estimator S1 compares): greedy pass@1 of HumanEval+ / MBPP+ (EvalPlus base and plus) and
  HumanEval-cpp / MBPP-cpp (MultiPL-E) (eval/code_eval.py); fill-in-the-middle pass@1 of HumanEval single-line,
  multi-line and random-span infilling, PSM and SPM prompts (eval/fim_eval.py), --objective spec runs only (an ntp
  model trains no sentinels).
Written to <base_dir>/base_eval/<tag>.json: "bpb" and "solution_bpb" keyed by estimator ("raw", "tail").

    python -m scripts.base_eval --model-tag=d10-r8-spec-s42
"""
import argparse
import os

import torch

from nanochat.config import MICRO_BATCH_ROWS, MIXTURE, SEQ_LEN
from nanochat.data.loader import mixture_corpora
from nanochat.eval.code_eval import generation, solution_bpb, tasks
from nanochat.eval.fim_eval import fim_generation
from nanochat.eval.loss_eval import val_bpb
from nanochat.paths import base_dir, write_json
from nanochat.train.checkpoint import load_model


def main():
    parser = argparse.ArgumentParser(description="Base model evaluation: the whole suite on a run's final checkpoint")
    parser.add_argument('--model-tag', type=str, required=True, help='base_train run tag (base_checkpoints/<tag>)')
    tag = parser.parse_args().model_tag
    device = torch.device("cuda")
    benchmarks = tasks()
    results = {"run": tag, "bpb": {}, "solution_bpb": {}}
    for estimator in ("raw", "tail"):  # tail last: generation reuses its model
        model, tokenizer, meta = load_model(device, tag, estimator)
        # losses from the compiled training forward, every batch MICRO_BATCH_ROWS x SEQ_LEN: one graph scores every
        # corpus's bpb (cu_seqlens padded to one length), one more the solution bpb (no attention segments)
        compiled = torch.compile(model, dynamic=False, fullgraph=True)
        results["bpb"][estimator] = val_bpb(compiled, tokenizer, mixture_corpora(MIXTURE), batch_rows=MICRO_BATCH_ROWS, device=device)
        results["solution_bpb"][estimator] = solution_bpb(compiled, tokenizer, device, MICRO_BATCH_ROWS, SEQ_LEN,
                                                          benchmarks)
        if estimator == "raw":
            del model, compiled
            torch._dynamo.reset()
            torch.cuda.empty_cache()
    results.update(step=meta["step"], objective=meta["objective"], codegen=generation(model, tokenizer, benchmarks))
    if meta["objective"] == "spec":
        results["fim"] = fim_generation(model, tokenizer)
    os.makedirs(os.path.join(base_dir(), "base_eval"), exist_ok=True)
    write_json(os.path.join(base_dir(), "base_eval", f"{tag}.json"), results)


if __name__ == "__main__":
    main()
