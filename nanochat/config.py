"""
The run settings every training and evaluation shares (dev/D20_SPEC.md "Run settings").
"""

SEQ_LEN = 8192
# The vocabulary, special tokens included: DeepSeek-Coder's 32K BPE vocabulary (arXiv 2401.14196 §3.2) as 2^15 ids
VOCAB_SIZE = 2**15
# {group: (share, corpora)}: Qwen2.5-Coder's code:text:math = 7:2:1 (arXiv 2409.12186 §3.1.2, Table 3), code = the
# Python and C++ corpora, each corpus at its group's natural row share (data.loader.mixture_weights)
MIXTURE = {
    "code": (0.7, ["stack_edu_python", "ultradata_code_l2_python", "ultradata_stack_v3_l2_python", "kaggle_notebooks", "codeforces_cots",
                   "stack_edu_cpp", "ultradata_code_l2_cpp", "ultradata_stack_v3_l2_cpp"]),
    "text": (0.2, ["climbmix"]),
    "math": (0.1, ["opc_fineweb_math", "ultradata_math_l2"]),
}
# Micro-batches of CALIBRATION_M tokens: the M at which the pinned QuACK GEMM table (ops/quack_gemm.py) was measured
CALIBRATION_M = 65_536
MICRO_BATCH_ROWS = CALIBRATION_M // SEQ_LEN


def run_tag(depth, ratio, objective, seed):
    """A run's tag (its checkpoint directory and metrics file)."""
    return f"d{depth}-r{ratio:g}-{objective}-s{seed}"
