"""
The training mixture: two stages, each a set of source groups with a fixed, "supply" or "rest" share; a run's weights
per stage (weights) and its row-by-row assignment to packed corpora (mixture_rows).
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

# Whole rows of both contexts: 12 x 8,192 and 3 x 32,768
MICRO_BATCH_TOKENS = 98_304
# Repo-level share of steps: Qwen2.5-Coder's 300B of 5.2T + 300B (arXiv 2409.12186 §3.2.1-3.2.2)
REPO_LEVEL_SHARE = 300e9 / (5.2e12 + 300e9)

# Weyl sequence frac(n * GOLDEN): low-discrepancy prefixes (Knuth, TAOCP Vol. 3 §6.4)
GOLDEN = (math.sqrt(5) - 1) / 2


@dataclass(frozen=True)
class Group:
    share: float | Literal["supply", "rest"]
    sources: tuple[str, ...]


@dataclass(frozen=True)
class Stage:
    seq_len: int
    tokens: int
    groups: dict[str, Group]

    @property
    def rows(self) -> int:
        """The base model's training rows of seq_len tokens in this stage."""
        return self.tokens // self.seq_len

    @property
    def micro_batch_size(self) -> int:
        """Rows of a micro-batch."""
        return MICRO_BATCH_TOKENS // self.seq_len

    @property
    def sources(self) -> list[str]:
        return [s for group in self.groups.values() for s in group.sources]

    @property
    def code(self) -> list[str]:
        """The sources of the stage's code groups (every group but NON_CODE)."""
        return [s for name, group in self.groups.items() if name not in NON_CODE for s in group.sources]

    @property
    def supply(self) -> list[str]:
        """The sources of the stage's "supply" groups, each read whole."""
        return [s for group in self.groups.values() if group.share == "supply" for s in group.sources]

    def shares(self, amounts: Mapping[str, float], total: float) -> dict[str, float]:
        """{source: its share of the stage}, given each source's amount (tokens, or packed rows) and the stage's total
        of that amount."""
        groups = {name: group.share for name, group in self.groups.items() if isinstance(group.share, float)}
        groups |= {name: sum(amounts[s] for s in group.sources) / total
                   for name, group in self.groups.items() if group.share == "supply"}
        groups |= {name: CODE_SHARE - sum(v for g, v in groups.items() if g not in NON_CODE)
                   for name, group in self.groups.items() if group.share == "rest"}
        return {s: groups[name] * amounts[s] / sum(amounts[t] for t in group.sources)
                for name, group in self.groups.items() for s in group.sources}


# The d24 plan at b065ad7 (schedule.plan): 69,580 steps of 15 micro-batches; they size the packs. The last
# REPO_LEVEL_SHARE of steps at 32K
MAX_STEPS, BATCH_SIZE = 69_580, 15 * MICRO_BATCH_TOKENS
REPO_LEVEL_STEPS = round(REPO_LEVEL_SHARE * MAX_STEPS)

# DeepSeek-Coder's 87% source code (arXiv 2401.14196 §2)
CODE_SHARE = 0.87
NON_CODE = ("text", "math")
REPOSITORY_SOURCES = ("stack_edu_python", "ultradata_stack_v3_l2_python", "ultradata_code_l2_python",
                      "stack_edu_cpp", "ultradata_stack_v3_l2_cpp", "ultradata_code_l2_cpp")
MATH_SOURCES = ("ultradata_math_l3_qa", "ultradata_math_l3_conversation", "ultradata_math_l3_multi_style",
                "ultradata_math_l3_textbook_exercise")
# 30% of code: rephrased synthetic share "converging to ~30%" (arXiv 2510.01631 §1)
SYNTHETIC_SOURCES = ("ultradata_code_l3_python", "ultradata_code_l3_cpp", "nemotron_rewriting_python",
                     "nemotron_transpilation_cpp")
# One corpus per language: a pack call parses one grammar
REPOSITORIES = {"Python": "repositories_python", "C++": "repositories_cpp"}
# Contexts 8,192 and 32,768 (arXiv 2409.12186 §3.2.1, §3.2.2)
FILE_LEVEL_STAGE = Stage(8192, (MAX_STEPS - REPO_LEVEL_STEPS) * BATCH_SIZE, {
    "repository": Group("rest", REPOSITORY_SOURCES),
    "synthetic": Group(0.30 * CODE_SHARE, SYNTHETIC_SOURCES),
    "notebooks": Group("supply", ("kaggle_notebooks",)),
    "functions": Group("supply", ("codexglue_python",)),
    "text": Group(0.10, ("climbmix",)),
    "math": Group(0.03, MATH_SOURCES),
})
REPO_LEVEL_STAGE = Stage(32_768, REPO_LEVEL_STEPS * BATCH_SIZE, {
    "repositories": Group("rest", tuple(REPOSITORIES.values())),
    "traces": Group("supply", ("codeforces_cots",)),
    "text": Group(0.10, ("climbmix",)),
    "math": Group(0.03, MATH_SOURCES),
})

IN_PART = ("synthetic", "text", "math")
PART = {source: group for group in IN_PART for source in FILE_LEVEL_STAGE.groups[group].sources}
# Bytes per token of the reference tokenizer, for token counts made before the corpus's tokenizer exists
BYTES_PER_TOKEN = {"stack_edu_python": 3.460, "ultradata_stack_v3_l2_python": 3.638,
                   "ultradata_code_l2_python": 3.709, "stack_edu_cpp": 3.065, "ultradata_stack_v3_l2_cpp": 3.015,
                   "ultradata_code_l2_cpp": 2.935, "kaggle_notebooks": 3.459, "codeforces_cots": 3.536,
                   "ultradata_code_l3_python": 3.826, "ultradata_code_l3_cpp": 3.593, "climbmix": 4.275,
                   "nemotron_rewriting_python": 3.909, "nemotron_transpilation_cpp": 3.651, "codexglue_python": 4.028,
                   "ultradata_math_l3_qa": 2.917, "ultradata_math_l3_conversation": 3.457,
                   "ultradata_math_l3_multi_style": 3.332, "ultradata_math_l3_textbook_exercise": 3.032}
# Relative widening of a byte target for a retrained tokenizer's compression: below the 6.4% (English) to 6.9% (code)
# relative NSL spread across training mixes of arXiv 2402.01035 Fig. 2 [arithmetic]; it held for every tokenizer arm
# (results/tokenizer_split_merge-intrinsic.json train_tokens_over_draw >= 1.017)
SPREAD = 0.06


def draw(group: str) -> float:
    """The tokens both stages draw of a group read in part."""
    return sum(float(stage.groups[group].share) * stage.tokens for stage in (FILE_LEVEL_STAGE, REPO_LEVEL_STAGE)
               if group in stage.groups)


def weights(stage: Stage, rows: dict[str, int], count: int
            ) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
    """A run's ({corpus: weight}, {corpus: rows drawn}, {corpus: rows drawn beyond its packed rows}) in a stage of
    `count` rows. A "supply" corpus is read whole (its rows over `count`), reweighted until it never overdraws; when
    that leaves a share negative (a run too short to read it whole), every supply share is the base run's (its rows
    over stage.rows)."""
    supply, weighed = set(stage.supply), dict(rows)
    while True:
        shares = stage.shares(weighed, count)
        if min(shares.values()) < 0:
            shares = stage.shares(weighed, stage.rows)
        names = list(shares)
        counts = np.bincount(mixture_rows(np.array([shares[n] for n in names]), start=0, count=count),
                             minlength=len(names))
        draws = dict(zip(names, counts.tolist(), strict=True))
        over = {n: d - rows[n] for n, d in draws.items() if d > rows[n]}
        if not over.keys() & supply:
            return shares, draws, over
        for n in over.keys() & supply:
            weighed[n] -= over[n]


def mixture_rows(weights: np.ndarray, *, start: int, count: int) -> np.ndarray:
    """Corpus index (into weights) of each run row start..start+count-1, by the Weyl point frac((n + 1) * GOLDEN)."""
    bounds = np.cumsum(weights)
    bounds /= bounds[-1]
    points = np.modf(np.arange(start + 1, start + count + 1) * GOLDEN)[0]
    return np.searchsorted(bounds, points, side="right")
