"""lcb_runner's main (LiveCodeBench 28fef95, docker/Dockerfile.eval's /opt/lcb venv) with the post-trained model
registered as "nanochat": an OpenAIChat-style model (lcb_runner/lm_styles.py), so its OpenAIRunner sends the system
message and the question as two messages with the CLI's temperature, top_p, n and max_tokens to the OpenAI-compatible
server at $OPENAI_BASE_URL (`vllm serve` of the --chat export, inference/export_hf.py, whose chat template joins the
two as lcb_runner's no-system-role style writes them, then THINK). Every other argument is main.py's.

    OPENAI_BASE_URL=http://<host>:<port>/v1 OPENAI_KEY=none python -m eval.official_lcb <main.py arguments>
"""

import runpy

from lcb_runner import lm_styles

MODEL = "nanochat"

if __name__ == "__main__":
    lm_styles.LanguageModelStore[MODEL] = lm_styles.LanguageModel(MODEL, MODEL, lm_styles.LMStyle.OpenAIChat, None)
    runpy.run_module("lcb_runner.runner.main", run_name="__main__")
