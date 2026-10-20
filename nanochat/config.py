"""Run settings shared by training and evaluation: vocabulary, mixture, micro-batch, stages, run names."""
import hashlib
import json
from dataclasses import dataclass

from nanochat.data.schema import REPOSITORY_FILE
from nanochat.data.sources import SOURCES

# The prepared data (tokenizer, compiled and packed rows) at its Hub revision: `python -m nanochat.paths` downloads it,
# and every run records it (train/metrics.py)
DATASET, DATASET_REVISION = "iaroy/nanochat-data", "6744e4068949e32b7e335bde01dcd3be746da0f8"

# Ids including special tokens: nanochat's 2^15 (karpathy/nanochat 92d63d4 scripts/tok_train.py:19)
VOCAB_SIZE = 2**15
# {group: (share, corpora)}; corpora at their group's natural row share (data.loader.mixture_weights). Code 87% as
# DeepSeek-Coder (arXiv 2401.14196 §2); math 3% as StarCoder2's OpenWebMath share (arXiv 2402.19173 Table 4); text the
# rest. Qwen2.5-Coder's 7B mixture ablation (arXiv 2409.12186 Table 3) is not monotone in the code share: 100:0:0
# scored highest on its coding benchmarks, 70:20:10 next, 85:10:5 lowest (ablations/ablations.md: why not tested).
MIXTURE = {
    "code": (0.87, ["stack_edu_python", "ultradata_code_l2_python", "ultradata_stack_v3_l2_python", "kaggle_notebooks",
                    "codeforces_cots", "stack_edu_cpp", "ultradata_code_l2_cpp", "ultradata_stack_v3_l2_cpp"]),
    "text": (0.10, ["climbmix"]),
    "math": (0.03, ["opc_fineweb_math", "ultradata_math_l2"]),
}
# Most whole rows of both contexts (12 x 8K, 3 x 32K) whose d24 step fits 80 GB: peak 69.6 GiB; 131,072 tokens
# does not fit (ablations/ablations.md)
MICRO_BATCH_TOKENS = 98_304


@dataclass(frozen=True, kw_only=True)
class Stage:
    """A training stage: context and mixture {group: (share, corpora)}."""
    seq_len: int
    mixture: dict

    @property
    def batch_rows(self):
        return MICRO_BATCH_TOKENS // self.seq_len


# Qwen2.5-Coder's stages (arXiv 2409.12186 §3.2): file-level at 8,192 tokens, then repo-level at 32,768 on the
# repository corpora (data/schema.py REPOSITORY_FILE)
FILE_STAGE = Stage(seq_len=8192, mixture=MIXTURE)
REPO_STAGE = Stage(seq_len=32_768,
                   mixture={"code": (1.0, [c for c in MIXTURE["code"][1] if SOURCES[c].schema is REPOSITORY_FILE])})
# Repo-level share of tokens: Qwen2.5-Coder's 300B of 5.2T + 300B (§3.2.1-2)
REPO_SHARE = 300e9 / (5.2e12 + 300e9)
# One RoPE base for both stages: Qwen2.5-Coder's repo-level 10^6 (§3.2.2), above Men et al.'s lower bound for a 32K
# context (arXiv 2405.14591); Llama 3 trains at a large base (5 x 10^5) from the start (arXiv 2407.21783 §3.2)
ROPE_BASE = 1_000_000


def run_tag(*, depth: int, ratio: float, seed: int, data_seed: int, variant: dict) -> str:
    """Name for a run's checkpoints, metrics and evaluations: depth, ratio and seeds, then a hash of those and `variant`
    (an ablation run's fields, ablations/harness.py; production's: none), so runs of different settings never share
    artifacts."""
    # float(ratio): JSON writes int 20 as "20" but float 20.0 as "20.0", so one run would get two names
    identity = {"depth": depth, "ratio": float(ratio), "seed": seed, "data_seed": data_seed, "variant": variant}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return f"d{depth}-r{ratio:g}-s{seed}-d{data_seed}-{digest[:12]}"
