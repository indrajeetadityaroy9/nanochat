"""
The training mixture (normalization_doc.md §5): the run's tokens, its two stages, and each stage's groups of sources.
A group's share is its fraction of the stage's tokens, "supply" for a group read whole once (its share is its tokens
over the stage's), or "rest" for the code group that takes what the stage's other code groups leave of CODE_SHARE.
Within a group, sources weigh in by their tokens.

The groups read in part (IN_PART) read the same fraction of each of their sources; prepare fetches only the Hub files
that hold it (prepare.extent).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from nanochat.config import MICRO_BATCH_TOKENS, REPO_SHARE


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
        """The stage's training rows of seq_len tokens."""
        return self.tokens // self.seq_len

    @property
    def sources(self) -> list[str]:
        return [s for group in self.groups.values() for s in group.sources]

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


# Production's d24 run (docs/h100-pretraining.md): 86,975 steps of 12 micro-batches; its last REPO_SHARE of steps,
# Qwen2.5-Coder's 300B of 5.2T + 300B (config.py), at 32K, placed as train/schedule.plan places it
STEPS, STEP_TOKENS = 86_975, 12 * MICRO_BATCH_TOKENS
REPO_STEPS = round(REPO_SHARE * STEPS)

CODE_SHARE = 0.87
NON_CODE = ("text", "math")
REPOSITORY_SOURCES = ("stack_edu_python", "ultradata_stack_v3_l2_python", "ultradata_code_l2_python",
                      "stack_edu_cpp", "ultradata_stack_v3_l2_cpp", "ultradata_code_l2_cpp")
MATH_SOURCES = ("ultradata_math_l3_qa", "ultradata_math_l3_conversation", "ultradata_math_l3_multi_style",
                "ultradata_math_l3_textbook_exercise")
# Synthetic code: UltraData-Code L3's exercises and Nemotron's rewritten Python and Python transpiled into C++, 30% of
# code: the share of rephrased data that good mixtures converge to from scratch (Kang et al., arXiv 2510.01631)
SYNTHETIC_SOURCES = ("ultradata_code_l3_python", "ultradata_code_l3_cpp", "nemotron_rewriting_python",
                     "nemotron_transpilation_cpp")
# The repository stage's documents per (repository, language) across REPOSITORY_SOURCES (compile.py), a corpus per
# language, as a pack call parses one grammar
REPOSITORIES = {"Python": "repositories_python", "C++": "repositories_cpp"}
# Each stage's context: Qwen2.5-Coder's 8,192 and 32,768 (arXiv 2409.12186 §3.2)
FILE_STAGE = Stage(8192, (STEPS - REPO_STEPS) * STEP_TOKENS, {
    "repository": Group("rest", REPOSITORY_SOURCES),
    "synthetic": Group(0.30 * CODE_SHARE, SYNTHETIC_SOURCES),
    "notebooks": Group("supply", ("kaggle_notebooks",)),
    "functions": Group("supply", ("codexglue_python",)),
    "text": Group(0.10, ("climbmix",)),
    "math": Group(0.03, MATH_SOURCES),
})
REPO_STAGE = Stage(32_768, REPO_STEPS * STEP_TOKENS, {
    "repositories": Group("rest", tuple(REPOSITORIES.values())),
    "traces": Group("supply", ("codeforces_cots",)),
    "text": Group(0.10, ("climbmix",)),
    "math": Group(0.03, MATH_SOURCES),
})

IN_PART = ("synthetic", "text", "math")
# {source: its group} of every source of a group read in part (such a group has the same sources in both stages)
PART = {source: group for group in IN_PART for source in FILE_STAGE.groups[group].sources}
# Production's tokenizer's bytes per token on each source (measured on its compiled train splits and on samples of the
# sources it did not compile: normalization_doc.md §5.1), for token counts before this corpus's tokenizer exists
BYTES_PER_TOKEN = {"stack_edu_python": 3.460, "ultradata_stack_v3_l2_python": 3.638,
                   "ultradata_code_l2_python": 3.709, "stack_edu_cpp": 3.065, "ultradata_stack_v3_l2_cpp": 3.015,
                   "ultradata_code_l2_cpp": 2.935, "kaggle_notebooks": 3.459, "codeforces_cots": 3.536,
                   "ultradata_code_l3_python": 3.826, "ultradata_code_l3_cpp": 3.593, "climbmix": 4.275,
                   "nemotron_rewriting_python": 3.909, "nemotron_transpilation_cpp": 3.651, "codexglue_python": 4.028,
                   "ultradata_math_l3_qa": 2.917, "ultradata_math_l3_conversation": 3.457,
                   "ultradata_math_l3_multi_style": 3.332, "ultradata_math_l3_textbook_exercise": 3.032}
# How much better a retrained tokenizer may compress a source than production's: "the NSL on any given subset only
# varies by 5 to 6 percentage points" across tokenizer training mixes, "about 5%" in its example (Dagan et al.,
# arXiv 2402.01035 §2.3); ablations/tokenizer.py intrinsic measures each arm's margin (train_tokens_over_draw)
SPREAD = 0.06


def draw(group: str) -> float:
    """The tokens both stages draw of a group read in part."""
    return sum(float(stage.groups[group].share) * stage.tokens for stage in (FILE_STAGE, REPO_STAGE)
               if group in stage.groups)
