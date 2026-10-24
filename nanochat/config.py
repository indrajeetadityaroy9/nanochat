"""Run settings shared by training and evaluation: the published dataset, vocabulary, head width, micro-batch, the
repo-level share of steps, RoPE base, run names. The stages and their mixture are the corpus's (corpus/mixture.py)."""
import hashlib
import json

# The published corpus (corpus/publish.py)
DATASET = "iaroy/nanochat-data"

# Ids including special tokens: nanochat's 2^15 (karpathy/nanochat 92d63d4 scripts/tok_train.py:19)
VOCAB_SIZE = 2**15
# Attention head width: nanochat's 128; compiled into csrc's kernels (native.py), whose register arrays need it at
# compile time
HEAD_DIM = 128
# Most whole rows of both contexts (12 x 8K, 3 x 32K) whose d24 step fits 80 GB
MICRO_BATCH_TOKENS = 98_304
# Repo-level share of steps: Qwen2.5-Coder's 300B of 5.2T + 300B (arXiv 2409.12186 §3.2.1-3.2.2)
REPO_SHARE = 300e9 / (5.2e12 + 300e9)
# One RoPE base for both stages: 10^6, above the 32K context's lower bound of arXiv 2405.14591 (Table 2: 6.4e5), large
# from the start as Llama 3 (500,000, arXiv 2407.21783 §3.2); Qwen2.5-Coder instead raises 10^4 to 10^6 at its
# repo-level stage (arXiv 2409.12186 §3.2.2)
ROPE_BASE = 1_000_000


def run_tag(*, depth: int, ratio: float, seed: int, variant: dict) -> str:
    """Name for a run's checkpoints, metrics and evaluations: depth, ratio and seed, then a hash of those and `variant`
    (an ablation run's fields, ablations/harness.py; the base model's lr scale, scripts/pretrain.py), so runs of
    different settings never share artifacts."""
    # float(ratio): JSON writes int 20 as "20" but float 20.0 as "20.0", so one run would get two names
    identity = {"depth": depth, "ratio": float(ratio), "seed": seed, "variant": variant}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return f"d{depth}-r{ratio:g}-s{seed}-{digest}"
