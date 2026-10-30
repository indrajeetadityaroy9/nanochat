"""On PYTHONPATH in scripts/official_eval.sh's generation container: with VLLM_GPU_MEMORY_UTILIZATION set, every
vllm.LLM takes that gpu_memory_utilization (MultiPL-E's automodel_vllm.py fixes 0.95, EvalPlus's provider leaves vLLM's
0.92), so the tools fit beside another process on the GPU. It sizes the KV cache only."""
import os

if share := os.environ.get("VLLM_GPU_MEMORY_UTILIZATION"):
    import functools

    from vllm import LLM

    init = LLM.__init__

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        init(self, *args, **(kwargs | {"gpu_memory_utilization": float(share)}))

    LLM.__init__ = __init__
