"""
The run settings every training and evaluation shares: context, vocabulary, mixture, micro-batch, training stages.
"""
from dataclasses import dataclass

from nanochat.data.schema import REPOSITORY_FILE
from nanochat.data.sources import SOURCES

# The context: Qwen2.5-Coder's file-level pretraining sequence length, 8,192 tokens (arXiv 2409.12186 §3.2.1)
SEQ_LEN = 8192
# The vocabulary, special tokens included: nanochat's 2^15 ids (karpathy/nanochat 92d63d4 scripts/tok_train.py:19
# --vocab-size default 32768; DeepSeek-Coder's BPE vocabulary has 32,000, arXiv 2401.14196 §3.2)
VOCAB_SIZE = 2**15
# {group: (share, corpora)}: DeepSeek-Coder's 87% code, 10% + 3% natural language (arXiv 2401.14196 §2: source code,
# English code-related text, Chinese general text), the 3% here math web text as StarCoder2's OpenWebMath (14.42B of the
# 7B model's 658.58B unique tokens, arXiv 2402.19173 Table 4); code = the Python and C++ corpora, each corpus at its
# group's natural row share (data.loader.mixture_weights). On coding benchmarks more code scores higher (Qwen2.5-Coder
# ablation, arXiv 2409.12186 Table 3: code:text:math 100:0:0 49.8 / 40.3 Common / BigCodeBench vs 70:20:10 48.3 / 38.3)
MIXTURE = {
    "code": (0.87, ["stack_edu_python", "ultradata_code_l2_python", "ultradata_stack_v3_l2_python", "kaggle_notebooks",
                    "codeforces_cots", "stack_edu_cpp", "ultradata_code_l2_cpp", "ultradata_stack_v3_l2_cpp"]),
    "text": (0.10, ["climbmix"]),
    "math": (0.03, ["opc_fineweb_math", "ultradata_math_l2"]),
}
# Micro-batches of 98,304 tokens, 12 rows of the file-level stage's 8,192 and 3 of the repo-level stage's 32,768: the
# most whole rows of both contexts whose d24 training step fits the H100's 80 GB, measured on the packed train rows
# (profile_step.py: peak 69.6 GiB with the optimizer state, 48.9 GiB at 65,536 tokens, so +20.7 GiB per 32,768 and over
# 80 GB at 131,072), 1% faster per token than 65,536 (benchmarks/h100.txt)
MICRO_BATCH_TOKENS = 98_304


@dataclass(frozen=True, kw_only=True)
class Stage:
    """A training stage: its context (rows of seq_len tokens), RoPE base and mixture {group: (share, corpora)}."""
    seq_len: int
    rope_base: float
    mixture: dict

    @property
    def batch_rows(self):
        """Rows per micro-batch: MICRO_BATCH_TOKENS // seq_len."""
        return MICRO_BATCH_TOKENS // self.seq_len


# Qwen2.5-Coder's two pretraining stages (arXiv 2409.12186 §3.2): file-level at 8,192 tokens, then repo-level, the
# context extended to 32,768 tokens and "RoPE's base frequency is adjusted from 10,000 to 1,000,000" (§3.2.2), on
# long-context code: here the code corpora whose documents are repository samples (data/schema.py REPOSITORY_FILE), at
# their natural row shares.
FILE_STAGE = Stage(seq_len=SEQ_LEN, rope_base=10_000, mixture=MIXTURE)
REPO_STAGE = Stage(seq_len=32_768, rope_base=1_000_000,
                   mixture={"code": (1.0, [c for c in MIXTURE["code"][1] if SOURCES[c].schema is REPOSITORY_FILE])})
# The repo-level stage's share of a run's tokens: Qwen2.5-Coder's ~300B of its 5.2T file-level + 300B tokens (§3.2.1-2)
REPO_SHARE = 300e9 / (5.2e12 + 300e9)


def run_tag(depth, ratio, seed, grad_clip):
    """A run's tag (its checkpoint directory and metrics file); grad_clip distinguishes the clipping ablation's arms
    (runs/ablation_clip.sh)."""
    return f"d{depth}-r{ratio:g}-s{seed}-g{grad_clip:g}"
