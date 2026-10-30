"""On PYTHONPATH in the official tools' vLLM containers (scripts/official_eval.sh, scripts/official_reasoning.sh):

- With VLLM_GPU_MEMORY_UTILIZATION set, every vllm.LLM takes that gpu_memory_utilization (MultiPL-E's
  automodel_vllm.py fixes 0.95, EvalPlus's provider leaves vLLM's 0.92), so the tools fit beside another process on the
  GPU. It sizes the KV cache only.
- Every vllm.LLM.generate refuses a prompt holding "\\r" (ValueError): the corpus normalizes line endings
  (corpus/pretrain/prepare.py), so the model never trained on one; a tool whose prompts carry one gets a normalizing
  task or hook (docs/evaluation.md §4) instead of a silently corrupted measurement. Token-id prompts are decoded to
  check.
- With NANOCHAT_REASONING_PARSER set, every process registers the "nanochat" reasoning parser
  (eval/official_reasoning_parser.py): `vllm serve --reasoning-parser-plugin` imports it in the API server only, and
  the spawned engine core looks the parser up too.
"""
import os

if os.environ.get("NANOCHAT_REASONING_PARSER"):
    import eval.official_reasoning_parser  # noqa: F401 (registers the parser)

try:
    from vllm import LLM
except ImportError:
    LLM = None

if LLM is not None:
    import functools

    init, generate = LLM.__init__, LLM.generate

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        if share := os.environ.get("VLLM_GPU_MEMORY_UTILIZATION"):
            kwargs = kwargs | {"gpu_memory_utilization": float(share)}
        init(self, *args, **kwargs)

    @functools.wraps(generate)
    def checked_generate(self, prompts, *args, **kwargs):
        for p in prompts if isinstance(prompts, list | tuple) else [prompts]:
            if isinstance(p, dict) and "prompt" not in p:
                text = self.get_tokenizer().decode(p["prompt_token_ids"])
            else:
                text = p["prompt"] if isinstance(p, dict) else p
            if "\r" in text:
                raise ValueError(f"a prompt holds a carriage return (docs/evaluation.md §4): {text[:200]!r}")
        return generate(self, prompts, *args, **kwargs)

    LLM.__init__, LLM.generate = __init__, checked_generate
