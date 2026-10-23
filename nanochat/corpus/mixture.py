"""
The corpus's training mixture (the design document, normalization_doc.md §5): the run's tokens, its two stages, and
each stage's groups {group: (share, sources)}. A share is the group's fraction of its stage's tokens, "supply" for a
group read whole once (its share is its tokens over the stage's), or "rest" for the group that takes what the others
leave of the code share. Within a group, sources weigh in by their tokens.

The groups read in part (IN_PART) read the same fraction of each of their sources' tokens; prepare fetches only that
fraction of each source's Hub files (prepare.extent).
"""

from typing import Literal, cast

from nanochat.corpus.sources import SOURCES

# Production's d24 run (docs/h100-pretraining.md): 86,975 steps of 12 micro-batches of 98,304 tokens
STEPS, STEP_TOKENS = 86_975, 12 * 98_304
# Repository-stage share of steps: Qwen2.5-Coder's 300B of 5.2T + 300B (arXiv 2409.12186 §3.2.1-3.2.2), placed as
# train/schedule.plan places it
REPO_SHARE = 300e9 / (5.2e12 + 300e9)
REPO_STEPS = round(REPO_SHARE * STEPS)
FILE_TOKENS, REPO_TOKENS = (STEPS - REPO_STEPS) * STEP_TOKENS, REPO_STEPS * STEP_TOKENS

CODE_SHARE = 0.87
REPOSITORY_SOURCES = ["stack_edu_python", "ultradata_stack_v3_l2_python", "ultradata_code_l2_python",
                      "stack_edu_cpp", "ultradata_stack_v3_l2_cpp", "ultradata_code_l2_cpp"]
MATH_SOURCES = ["ultradata_math_l3_qa", "ultradata_math_l3_conversation", "ultradata_math_l3_multi_style",
                "ultradata_math_l3_textbook_exercise"]

Share = float | Literal["supply", "rest"]
FILE_STAGE: dict[str, tuple[Share, list[str]]] = {
    "repository": ("rest", REPOSITORY_SOURCES),
    "exercises": (0.10, ["ultradata_code_l3_python", "ultradata_code_l3_cpp"]),
    "notebooks": ("supply", ["kaggle_notebooks"]),
    "text": (0.10, ["climbmix"]),
    "math": (0.03, MATH_SOURCES),
}
# "repositories": the stage's documents per (repository, language) across REPOSITORY_SOURCES (compile.py)
REPO_STAGE: dict[str, tuple[Share, list[str]]] = {
    "code": (CODE_SHARE, ["repositories", "codeforces_cots"]),
    "text": (0.10, ["climbmix"]),
    "math": (0.03, MATH_SOURCES),
}
# The sources the stages read, in sources.py order (deduplication's keep-first order)
READ = [source for source in SOURCES
        if any(source in sources for stage in (FILE_STAGE, REPO_STAGE) for _, sources in stage.values())]

IN_PART = ("exercises", "text", "math")
# Production's tokenizer's bytes per token on each source (measured on its compiled train splits and on samples of the
# sources it did not compile: normalization_doc.md §5.1, §9), for token counts before this corpus's tokenizer exists,
# and how far a retrained tokenizer's compression may move from it: "the NSL on any given subset only varies by 5 to 6
# percentage points" across tokenizer training mixes (Dagan et al., arXiv 2402.01035 §2.3)
BYTES_PER_TOKEN = {"stack_edu_python": 3.460, "ultradata_stack_v3_l2_python": 3.638,
                   "ultradata_code_l2_python": 3.709, "stack_edu_cpp": 3.065, "ultradata_stack_v3_l2_cpp": 3.015,
                   "ultradata_code_l2_cpp": 2.935, "kaggle_notebooks": 3.459, "codeforces_cots": 3.536,
                   "ultradata_code_l3_python": 3.826, "ultradata_code_l3_cpp": 3.593, "climbmix": 4.275,
                   "ultradata_math_l3_qa": 2.917, "ultradata_math_l3_conversation": 3.457,
                   "ultradata_math_l3_multi_style": 3.332, "ultradata_math_l3_textbook_exercise": 3.032}
SPREAD = 0.06
# {source: its group} for every source of a group read in part (each such group has the same sources in both stages)
PART = {source: group for group in IN_PART for source in FILE_STAGE[group][1]}


def draw(group: str) -> float:
    """The tokens both stages draw of a group read in part (its share is a fraction in each stage that has it)."""
    return sum(cast(float, stage[group][0]) * tokens
               for stage, tokens in ((FILE_STAGE, FILE_TOKENS), (REPO_STAGE, REPO_TOKENS)) if group in stage)


def source_shares(stage: dict[str, tuple[Share, list[str]]], stage_tokens: int, tokens: dict[str, float]
                  ) -> dict[str, float]:
    """{source: its share of a stage's tokens}, given each source's tokens: a fixed group share; a "supply" group's
    tokens over the stage's; the "rest" group, the code share less the stage's other code groups (every group of the
    stage but text and mathematics); each group's share split among its sources by their tokens."""
    groups = {group: share for group, (share, _) in stage.items() if isinstance(share, float)}
    groups |= {group: sum(tokens[s] for s in sources) / stage_tokens
               for group, (share, sources) in stage.items() if share == "supply"}
    groups |= {group: CODE_SHARE - sum(v for g, v in groups.items() if g not in ("text", "math"))
               for group, (share, _) in stage.items() if share == "rest"}
    return {s: groups[group] * tokens[s] / sum(tokens[t] for t in sources)
            for group, (_, sources) in stage.items() for s in sources}
