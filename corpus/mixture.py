"""
The training mixture: the run's tokens, its two stages, each stage's groups of sources. A group's
share is a fixed fraction of the stage's tokens, "supply" (read whole once), or "rest" (the code group taking what the
other code groups leave of CODE_SHARE). Within a group, sources weigh by their amounts: tokens before packing, packed
rows after. Groups read in part (IN_PART) read the same fraction of each source (prepare.extent fetches only that).
A run's weights in a stage (weights) and its mixture of packed corpora row by row (mixture_rows).
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

# Tokens per micro-batch: whole rows of both contexts, 12 x 8,192 and 3 x 32,768
MICRO_BATCH_TOKENS = 98_304
# Repo-level share of steps: Qwen2.5-Coder's 300B of 5.2T + 300B (arXiv 2409.12186 §3.2.1-3.2.2)
REPO_LEVEL_SHARE = 300e9 / (5.2e12 + 300e9)

# the Weyl sequence frac(n * GOLDEN) spreads every prefix of itself most evenly over [0, 1) (Knuth, TAOCP Vol. 3 §6.4)
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
        of that amount: a fixed group share; a "supply" group's amount over the total; the "rest" group, CODE_SHARE less
        the stage's other code groups; each group's share split among its sources by their amounts."""
        groups = {name: group.share for name, group in self.groups.items() if isinstance(group.share, float)}
        groups |= {name: sum(amounts[s] for s in group.sources) / total
                   for name, group in self.groups.items() if group.share == "supply"}
        groups |= {name: CODE_SHARE - sum(v for g, v in groups.items() if g not in NON_CODE)
                   for name, group in self.groups.items() if group.share == "rest"}
        return {s: groups[name] * amounts[s] / sum(amounts[t] for t in group.sources)
                for name, group in self.groups.items() for s in group.sources}


# The stages' sizes: schedule.plan's 69,580 steps of 15 micro-batches (102.6B tokens), the last REPO_LEVEL_SHARE of
# steps at 32K, rounded as schedule.plan rounds them. A Stage's tokens are this run's; any other run draws by the
# stage's shares at its own size (weights)
MAX_STEPS, BATCH_SIZE = 69_580, 15 * MICRO_BATCH_TOKENS
REPO_LEVEL_STEPS = round(REPO_LEVEL_SHARE * MAX_STEPS)

# Code share: DeepSeek-Coder's 87% source code, 10% English and 3% Chinese natural language (arXiv 2401.14196 §2), its
# natural language taken as web text and mathematics
CODE_SHARE = 0.87
NON_CODE = ("text", "math")
REPOSITORY_SOURCES = ("stack_edu_python", "ultradata_stack_v3_l2_python", "ultradata_code_l2_python",
                      "stack_edu_cpp", "ultradata_stack_v3_l2_cpp", "ultradata_code_l2_cpp")
MATH_SOURCES = ("ultradata_math_l3_qa", "ultradata_math_l3_conversation", "ultradata_math_l3_multi_style",
                "ultradata_math_l3_textbook_exercise")
# Synthetic code, 30% of code: good mixtures' share of rephrased synthetic data, "converging to ~30%" (Kang et al.,
# arXiv 2510.01631 §1), measured on web text
SYNTHETIC_SOURCES = ("ultradata_code_l3_python", "ultradata_code_l3_cpp", "nemotron_rewriting_python",
                     "nemotron_transpilation_cpp")
# The repository stage's documents per (repository, language) across REPOSITORY_SOURCES (tokenize.py), a corpus per
# language, as a pack call parses one grammar
REPOSITORIES = {"Python": "repositories_python", "C++": "repositories_cpp"}
# Each stage's context: Qwen2.5-Coder's 8,192 and 32,768 (arXiv 2409.12186 §3.2.1, §3.2.2)
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
# {source: its group} of every source of a group read in part (such a group has the same sources in both stages)
PART = {source: group for group in IN_PART for source in FILE_LEVEL_STAGE.groups[group].sources}
# Bytes per token of the reference tokenizer (SA-BPE within SPLIT_PATTERN, trained before this corpus) on each source,
# for token counts that precede the corpus's tokenizer (extents, the tokenizer sample)
BYTES_PER_TOKEN = {"stack_edu_python": 3.460, "ultradata_stack_v3_l2_python": 3.638,
                   "ultradata_code_l2_python": 3.709, "stack_edu_cpp": 3.065, "ultradata_stack_v3_l2_cpp": 3.015,
                   "ultradata_code_l2_cpp": 2.935, "kaggle_notebooks": 3.459, "codeforces_cots": 3.536,
                   "ultradata_code_l3_python": 3.826, "ultradata_code_l3_cpp": 3.593, "climbmix": 4.275,
                   "nemotron_rewriting_python": 3.909, "nemotron_transpilation_cpp": 3.651, "codexglue_python": 4.028,
                   "ultradata_math_l3_qa": 2.917, "ultradata_math_l3_conversation": 3.457,
                   "ultradata_math_l3_multi_style": 3.332, "ultradata_math_l3_textbook_exercise": 3.032}
# How much better a retrained tokenizer may compress a source than the reference tokenizer: NSL "only varies by 5 to 6
# percentage points" across tokenizer training mixes (Dagan et al., arXiv 2402.01035 §2.3);
# ablations/tokenizer_split_merge.py intrinsic measures each arm's margin (train_tokens_over_draw)
SPREAD = 0.06


def draw(group: str) -> float:
    """The tokens both stages draw of a group read in part."""
    return sum(float(stage.groups[group].share) * stage.tokens for stage in (FILE_LEVEL_STAGE, REPO_LEVEL_STAGE)
               if group in stage.groups)


def weights(stage: Stage, rows: dict[str, int], count: int
            ) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
    """A run's {corpus: weight} in a stage of `count` rows given each corpus's packed rows, {corpus: the rows it draws}
    and {corpus: the rows it draws beyond its packed rows} (empty when the run fits): the stage's shares (Stage.shares)
    of the packed rows, replayed by mixture_rows; a "supply" corpus that draws beyond its rows weighs as many rows less,
    so it is read whole once and never overdraws."""
    supply, weighed = set(stage.supply), dict(rows)
    while True:
        shares = stage.shares(weighed, count)
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
    """Corpus index (into weights) of each run row start..start+count-1: row n goes to the corpus whose share of the
    cumulative weights holds the Weyl point frac((n + 1) * GOLDEN), so every prefix of the run holds each corpus nearly
    in proportion to its weight (the sequence's low discrepancy), and any stretch of the run is computed without the
    rows before it."""
    bounds = np.cumsum(weights)
    bounds /= bounds[-1]
    points = np.modf(np.arange(start + 1, start + count + 1) * GOLDEN)[0]
    return np.searchsorted(bounds, points, side="right")
