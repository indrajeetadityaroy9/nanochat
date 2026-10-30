"""
LiveCodeBench code generation (arXiv 2403.07974) in the reasoning protocol of a post-trained model: the problems of
release_v6 in WINDOW, the official prompt, answer extraction and evaluation of lcb_runner (LiveCodeBench/LiveCodeBench
at the commit docker/Dockerfile pins), the evaluation run in a fresh isolated process on the timed-check cores.
lcb_runner is imported by the functions that use it, so importing this module does not need it.
"""

import contextlib
import dataclasses
import functools
import importlib.util
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from types import ModuleType

from huggingface_hub import hf_hub_download

from eval.verify import run_group, timed, timed_slots, timed_workers

NAME = "livecodebench"
# livecodebench/code_generation_lite at a fixed revision and the files of its release_v6 (code_generation_lite.py
# ALLOWED_FILES at that revision); the dataset's loading script needs datasets < 4 (trust_remote_code)
DATASET = {"repo_id": "livecodebench/code_generation_lite", "repo_type": "dataset",
           "revision": "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"}
RELEASE_V6 = ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl")
# Contest dates [start, end): LiveCodeBench v6 (08/24-05/25), Nemotron-Cascade's (arXiv 2512.13607 Table 1)
WINDOW = (datetime(2024, 8, 1), datetime(2025, 5, 1))
# Samples a problem: MobileLLM-R1's average over 16 runs (arXiv 2509.24945 Table 7)
N_SAMPLE = 16
# lcb_runner's LMStyle of a model with no system turn: SYSTEM_MESSAGE_GENERIC, "\n\n" and the generic question
# template in the one user turn (lcb_runner/prompts/code_generation.py:231-240); its extract_code takes the last
# fenced block (lcb_runner/utils/extraction_utils.py:13-17)
STYLE = "OpenAIReasonPreview"
# Per-test time limit of LiveCodeBench's evaluation, its CLI default (lcb_runner/runner/parser.py:93 --timeout,
# passed to codegen_metrics by lcb_runner/runner/scenario_router.py:192-197)
TIMEOUT = 6


@functools.cache
def prompt_module() -> ModuleType:
    """lcb_runner/prompts/code_generation.py loaded on its own, in the checkout's directory: the package's __init__
    imports self_repair, which needs anthropic, and the module opens its few-shot files by paths relative to the
    working directory (code_generation.py:169-173). Loaded once, by problems() (the working directory is the
    process's: load it before other threads run)."""
    import lcb_runner.lm_styles

    root = Path(lcb_runner.lm_styles.__file__).parents[1]
    spec = importlib.util.spec_from_file_location("lcb_runner.prompts.code_generation",
                                                  root / "lcb_runner" / "prompts" / "code_generation.py")
    module = importlib.util.module_from_spec(spec)
    with contextlib.chdir(root):
        spec.loader.exec_module(module)
    return module


def problems() -> list:
    """release_v6's problems with contest_date in WINDOW, in file order, each parsed by lcb_runner's
    CodeGenerationProblem (its fields of the JSON line, as the loading script's features keep)."""
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

    prompt_module()
    fields = [f.name for f in dataclasses.fields(CodeGenerationProblem)]
    out = []
    for name in RELEASE_V6:
        with open(hf_hub_download(filename=name, **DATASET), encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                if WINDOW[0] <= datetime.fromisoformat(p["contest_date"]) < WINDOW[1]:
                    out.append(CodeGenerationProblem(**{k: p[k] for k in fields}))
    return out


def prompt_text(problem) -> str:
    """The prompt text of a problem (BOS and the opening of the reasoning not included): lcb_runner's
    format_prompt_generation user turn of STYLE, stripped."""
    from lcb_runner.lm_styles import LMStyle

    return prompt_module().format_prompt_generation(problem, LMStyle(STYLE))[0]["content"].strip()


def answer(text: str) -> str:
    """lcb_runner's extract_code (STYLE) of the text after a response's last "</think>" (eval/code_eval.py
    reasoning_answer passes it): the lines between its last two lines holding ```, "" with fewer than two."""
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code

    return extract_code(text, LMStyle(STYLE))


# lcb_runner's codegen_metrics (argv[4] s a test, argv[3] workers) on the problems' evaluation samples and the answers;
# a sample passes iff every test result is > 0 (lcb_runner/evaluation/pass_k_utils.py:61); its error_code from the
# per-sample metadata (null when every test passed)
LCB_EVALUATE = """
import json, sys
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
with open(sys.argv[1], encoding="utf-8") as f:
    samples, answers = json.load(f)
_, results, metadata = codegen_metrics(samples, answers, k_list=[1], num_process_evaluate=int(sys.argv[3]),
                                       timeout=int(sys.argv[4]))
out = [[[all(g > 0 for g in r) for r in results[i]], [json.loads(m).get("error_code") for m in metadata[i]]]
       for i in range(len(answers))]
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(out, f)
"""


def score(items: list, answers: list[list[str]], *, ks: tuple[int, ...] = (1,)) -> dict:
    """LiveCodeBench's evaluation of each problem's answers (the same count each; the evaluation samples of
    lcb_runner/runner/scenario_router.py:189): per-sample pass booleans and lcb_runner error codes, and
    eval.code_eval.pass_scores of each k in ks; pass@1 is the mean over samples, then over problems (avg@k, arXiv
    2501.12948 §3)."""
    from eval.code_eval import pass_scores  # here: eval.code_eval imports this module

    with tempfile.TemporaryDirectory() as tmp:
        path, scored = Path(tmp, "samples.json"), Path(tmp, "results.json")
        path.write_text(json.dumps([[p.get_evaluation_sample() for p in items], answers]), encoding="utf-8")
        with timed_slots(timed_workers()):
            run_group(timed([sys.executable, "-I", "-c", LCB_EVALUATE, path, scored, str(timed_workers()),
                             str(TIMEOUT)]), cwd=tmp).check_returncode()
        result = json.loads(scored.read_text(encoding="utf-8"))
    status = [passed for passed, _ in result]
    return {"ids": [p.question_id for p in items], "status": status, "error_code": [codes for _, codes in result],
            "scores": pass_scores(status, ks)}
