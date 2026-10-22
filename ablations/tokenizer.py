"""
The tokenizer: the decision production's tokenizer waits on. Four vocabularies of 32,768 ids (32,764 mergeable tokens,
BOS and the 3 FIM sentinels), trained on the same sample (data/tokenizer.py sample) by the same crate and encoded by
tiktoken, each trained on, so that every candidate production keeps or dismisses has a run of its own:
- sa-bpe: production's, SA-BPE's priority F·log R (arXiv 2604.14053 §4.1: F the pair's frequency, R the repositories
  holding it in the current segmentation, a pair of one repository scoring 0; Table 3's F·log R row: compression 3.643
  and 72 of 646 tested tokens verified under-trained, against BPE's 3.623 and 640, at 100M parameters) within
  SPLIT_PATTERN. R counts the authorship units of data/tokenizer.py sample: repositories as in the paper, web hosts and
  documents as §6 suggests for natural language ("documents, books, or document collections"), which its Limitations
  leave untested. Not the paper's default F·log R·L (Table 4), L the languages holding the pair, "prioritizing
  language-universal merges" among 18 programming languages of natural shares, where it supports the lower-resource
  ones (§6, App. G): on Python and C++, this sample's languages, it compresses less (Table 10: 3.651 against 3.652,
  3.239 against 3.240) and uses more of the vocabulary (Table 12: 16,663 against 16,289 tokens, 16,625 against
  16,314); its under-trained tokens are 44 against 72 (Table 3; one tokenizer each, no seeds); §6 says the language
  aspect "must be avoided" for natural language, a third of this sample's sources; and 31,604 of production's 32,508
  merged tokens occur in the Python, the C++ and the English val documents alike, so L would be 3 for all but the
  vocabulary's tail [inference: val presence standing for the training sample's];
- sa-bpe-fused: within FUSED_PATTERN (karpathy/nanochat's, the first tokenizer's), whose punctuation clause takes the
  line breaks after it, as cl100k's does; also the positive control: arXiv 2608.18062's companion post deletes the
  trailing [\r\n/]* of GPT-4o's punctuation clause with the tokenizer corpus held fixed and moves MBPP pass@1 from
  0.086 to 0.230 (seeds 0.167, 0.072, 0.018 against 0.224, 0.246, 0.220; 1.27B parameters, ~20B math and code tokens,
  ~128K ids, healed prompts: the paper's §4.1-4.3), while code bits per byte shows no association with MBPP over its
  tokenizers (Spearman +0.07, -0.12, -0.04, every p above 0.5); a measure that misses it at d16 cannot decide the
  others;
- hybrid: SA-BPE within HYBRID_PATTERN, whose clauses resolve the limitations of the others' (at HYBRID_PATTERN);
- bpe: merges by frequency alone (Table 3's F row, SA-BPE's baseline; the crate's score "frequency") within
  SPLIT_PATTERN, 0.55% from F·log R in compression, so a tie in the runs is likely [inference], which the documents'
  statistics break.
Not run: tiktoken encodes by merge rank alone, which has no form for PickyBPE's token removals (its encoder replays
merges and removals in training order, arXiv 2409.04599 Alg. 2; its code compression 3.61 against BPE's 3.62 in
arXiv 2604.14053 Table 4), BoundlessBPE's supermerges (rule dictionaries of their own, arXiv 2504.00178 §3.2) or
UnigramLM; SuperBPE's one code result is lower, not significantly (arXiv 2503.13423 Table 1: HumanEval pass@10 15.9
to 13.4 at 8B parameters and 200K ids).

Runs: every arm at d16, SEEDS seeds each (the seed's spread is the measures' largest: the post's fusing tokenizer
scored MBPP 0.167, 0.072, 0.018), at equal bytes: each arm reads the document set once, as production reads its
data (train/stages.py max_ratio), so its tokens are the set's bytes over its bytes per token (arXiv 2402.01035 compares
at equal tokens, its Appendix J at equal characters); each run's plan is its own (schedule.plan at its tokens), its
context production's 8,192 and 32,768 tokens (eval/bpb.py's byte-offset buckets show the bytes of context a better
compressor sees; arXiv 2503.13423 §3.1 fixes context in bytes). The set holds 20 tokens per scaling parameter of d16
for sa-bpe (Chinchilla, arXiv 2203.15556 Table 3: 20.2B tokens for 1B parameters): of each code corpus, the leading
curated train documents whose sa-bpe tokens fill its file-level draw
(rows from production's packs: the same vocabulary, files and order), so the code group binds every arm's max_ratio;
of text and math the leading files holding MARGIN times their draw, and of each repository corpus the trailing files
holding MARGIN times the repository stage's draw (the same count for every corpus, as data/pack.py repo_stage), so
they never bind; of the val split a systematic sample of documents whose sa-bpe tokens fill at most 1 / MARGIN of
bpb's val sample at each row length, so every arm scores all of them (eval/bpb.py val_bits): the same documents.

RESULTS/tokenizer-data.json holds each arm's intrinsic record (describe), on every corpus: bytes per token and Rényi
efficiency; its vocabulary tokens holding a line break (the count correlating most strongly with MBPP in the companion
post, Spearman -0.719 over 58 tokenizers), fusing punctuation with one, and ending in a line break and indentation
(the post's MinGram pair: MBPP 0.016 against 0.216, 86.4% of the first's completions IndentationError or TabError);
the tokens its train documents hold never, under 10 and under 100 times; TokEval's AST boundary alignment, identifier
fragmentation and Python indentation consistency on the code corpora, its operator isolation on the code and math
corpora (ablations/tokeval.py). Not taken from TokEval: its multilingual metrics (fertility, Gini, MorphScore) and its
digit metrics (every arm's pattern cuts a digit run into the same chunks of at most two digits).

Decisions, pre-registered (RESULTS/tokenizer.json "decision"), each challenger (sa-bpe-fused, hybrid, bpe) against
sa-bpe on the runs of the same seeds:
- parsing: the HumanEval+ and MBPP+ problems whose healed greedy completion compiles under only one of the two
  (ablations/harness.py evaluation), the problems of every seed pooled, McNemar's exact test (the companion post's
  test of single runs: 104 problems against 4 for its MinGram pair); better or worse at p < 0.05, else equal. The test
  covers the problems' sampling; the seeds, training's only in part [inference];
- code bits per byte: the difference of the arms' means, better or worse beyond its 1 - ALPHA two-sided normal
  bound, 1.96 √(2 / SEEDS) CODE_SIGMA, else equal;
- a challenger qualifies iff it is worse on neither and better on one. Pattern: of sa-bpe-fused and the hybrid, the
  qualifier with the lower unparsed share is adopted, else production's pattern; beside it, whether the control is
  detected (sa-bpe-fused parses worse). Score: bpe is adopted iff it qualifies, or is equal on both and its documents
  leave fewer vocabulary tokens never read than sa-bpe's (SA-BPE's claimed benefit, fewer under-trained tokens: 640
  for F against 72 for F·log R, Table 3), carried to the adopted pattern [inference]; otherwise SA-BPE.

python -m ablations.tokenizer vocabularies   (the arms into TOKENIZER_ABLATION/<arm>)
python -m ablations.tokenizer data           (every arm's document set, its intrinsic record, then the checks)
python -m ablations.tokenizer runs           (ablations/harness.py, then the decision)
"""
import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import regex
from scipy.stats import binomtest, norm

from ablations import tokeval
from ablations.harness import ALPHA, Setting, main, production_plan
from nanochat.config import FILE_STAGE, REPO_STAGE
from nanochat.data.compile import compile_split
from nanochat.data.loader import (
    TOKEN_DTYPE,
    mixture_corpora,
    packed_rows,
    row_shards,
    shard_documents,
    split_path,
)
from nanochat.data.pack import in_order, pack_split, write
from nanochat.data.sources import curated_files
from nanochat.data.tokenizer import SPLIT_PATTERN, Tokenizer, get_tokenizer, merges, sample, save
from nanochat.eval.bpb import sample_rows
from nanochat.native import data_crate
from nanochat.paths import RESULTS, TOKENIZER_ABLATION
from nanochat.train.stages import drawn_rows, max_ratio, readable_rows

# karpathy/nanochat's split pattern (92d63d4), the first tokenizer's: SPLIT_PATTERN with cl100k's punctuation clause,
# whose [\r\n]* takes the line breaks after a run of punctuation ("):\n" one chunk)
FUSED_PATTERN = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]"""
                 r"""|\s+(?!\S)|\s+""")
# The hybrid's pattern. A pattern cuts a text into chunks, its matches from left to right; BPE merges, trained and
# applied within chunks (tiktoken encodes chunk by chunk), never cross a chunk's end, so every chunk end is a token end
# under every vocabulary trained with the pattern: a property of the chunks holds whatever the merges. No clause looks
# behind, so the chunks after a chunk end j are the chunks of the text from j alone. Its clauses give:
# (1) a line break shares a chunk only with line breaks of its kind (\n+, \r+: the post's Claude-approximate [\n]+,
#     [\r]+; no other clause matches one), the forced break at line breaks of the post's MinGram cb_nl. So no token
#     joins a line break to the punctuation before it, where FUSED_PATTERN's [\r\n]* makes "):\n" one chunk and a
#     prompt ending "):" ends inside the token training saw (the token-boundary bias DeepSeek-V3 names for multi-line
#     prompts without a terminal line break, arXiv 2412.19437 §4.1); production's pattern, the post's gpt4onl edit
#     (the punctuation clause without its trailing line breaks), shares this. And no token joins a line break to
#     whitespace, where production's \s*[\r\n] takes the spaces before a break and a whitespace-only line
#     ("\n    \n"), whose merges make tokens of a line break and indentation: the 187 such entries of the post's MinGram
#     cb, whose completions failed 86.4% by IndentationError or TabError (MBPP 0.016 against cb_nl's 0.216);
# (2) a run of spaces, of tabs or of other whitespace is one chunk ( +, \t+: the Claude-approximate [ ]+, [\t]+), and a
#     letter run takes no tab or line break before it: an indentation of d >= 2 spaces or d >= 1 tabs ends at a chunk
#     end, so, no clause looking behind, the statement s after it is cut as at depth 0: chunks(indent + s) =
#     (indent,) + chunks(s), one indentation chunk per depth and the statement's chunks the same at every depth.
#     Production's \s+(?!\S) gives the indentation's last space to the statement's first chunk (" if", " (") or to a
#     chunk of its own (before a digit): chunks = (" " * (d - 1), " " + w, ...), the depth carried by d - 1 spaces and
#     a space inside a statement chunk that differs from its depth-0 form ("if" against " if"); its letter clause
#     [^\r\n\p{L}\p{N}]?+ admits a tab, so a tab-indented statement starts "\tif". A single space before a letter run
#     stays its prefix (" foo"), as in prose. The Claude-approximate pattern, an early reconstruction in the post,
#     scored MBPP 0.250, 0.234, 0.240 over 3 seeds (gpt4onl 0.224, 0.246, 0.220), 0.3% of its completions
#     IndentationErrors;
# (3) no contraction clause (2402.01035's Punct drops cl100k's, its App. E and Fig. 3; o200k folds contractions into
#     the letter run): the letter clause takes a whole run of letters, which '(?i:[sdmt]|ll|ve|re) cuts where an
#     apostrophe starts a chunk ("'s" "elf" for a line's 'self'); English contractions chunk as before ("doesn" "'t").
# Kept from cl100k: one leading punctuation character or space on a letter run (".append", "(self", " foo"), which Punct
# drops (2402.01035 Table 1, code: 0.81 against Punct's 0.86 tokens per Llama token at 32K, "an additional 5%
# compression", App. E; pass@1 higher for Punct at 1.5B and 32K, lower at 7B and 80K, untested for significance);
# production's digit pairs (karpathy/nanochat: "2 is the sweet spot for vocab size of 32K") and punctuation runs. And
# [^\S\r\n\t ]+ takes every other whitespace character, so every character falls in some chunk: tiktoken drops what no
# clause matches.
HYBRID_PATTERN = r"""(?:[^\s\p{L}\p{N}]| )?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++|\n+|\r+|\t+| +|[^\S\r\n\t ]+"""
# arm: (pattern, merge score); every arm is trained on, sa-bpe as the harness's production arm
ARMS = {"sa-bpe": (SPLIT_PATTERN, "sa-bpe"), "sa-bpe-fused": (FUSED_PATTERN, "sa-bpe"),
        "hybrid": (HYBRID_PATTERN, "sa-bpe"), "bpe": (SPLIT_PATTERN, "frequency")}
CHALLENGERS = ("sa-bpe-fused", "hybrid", "bpe")
DEPTH, SEEDS = 16, 2
# The spread of a run's 8K code bits per byte over seeds (pooled within each configuration): the six runs at d8 at 20 of
# the standard-regime comparison (60138de's baseline and c00d2f6's default, data seeds 0 to 2), carried to d16
# [inference]
CODE_SIGMA = 0.0034
# Text, math and repository-stage rows kept per drawn row: an arm compressing a corpus better than sa-bpe packs it into
# fewer rows (sa-bpe takes 9.21% more tokens than the first tokenizer, sa-bpe-fused's construction, on stack_edu_cpp's
# whole val split, the largest gap measured), so only the code group binds
MARGIN = 1.25


def vocabularies() -> None:
    """Each arm's tokenizer.tiktoken and tokenizer.pattern in TOKENIZER_ABLATION/<arm>, on one load of the sample."""
    docs, sources = sample()
    for arm, (pattern, score) in ARMS.items():
        (TOKENIZER_ABLATION / arm).mkdir(parents=True)
        save(TOKENIZER_ABLATION / arm, merges(docs, sources, pattern=pattern, score=score), pattern)


def keep_documents(corpus: str, split: str, tokenizer: Tokenizer, keep: np.ndarray) -> None:
    """Rewrite a compiled split (data/compile.py) with the documents keep marks: its .bin, .idx and .json (each curated
    file's kept documents, so its packing shards stay the curated files)."""
    path = split_path(corpus, split, tokenizer)
    offsets = np.fromfile(path.with_suffix(".idx"), dtype=np.int64)
    lengths = np.diff(offsets)
    bounds = shard_documents(corpus, split, tokenizer)
    np.fromfile(path.with_suffix(".bin"), dtype=TOKEN_DTYPE)[np.repeat(keep, lengths)].tofile(path.with_suffix(".bin"))
    np.concatenate([[0], np.cumsum(lengths[keep])]).astype(np.int64).tofile(path.with_suffix(".idx"))
    path.with_suffix(".json").write_text(json.dumps({"files": [{"documents": int(keep[a:b].sum())}
                                                               for a, b in itertools.pairwise(bounds)]}))


@dataclass(frozen=True, kw_only=True)
class Documents:
    """A corpus's part of the document set: its leading curated train files (`train`), of which the first `kept`
    documents are kept; its trailing train files the repository stage reads (`repo`: a repository corpus's, else none);
    every `val_step`-th document of its val split."""
    train: list[Path]
    kept: int
    repo: list[Path]
    val_step: int


def documents() -> dict[str, Documents]:
    """The document set of every corpus (module docstring), from production's compiled data and packs, which count
    sa-bpe's tokens: sa-bpe is production's vocabulary, retrained."""
    production = get_tokenizer()
    assert get_tokenizer("sa-bpe").data_key == production.data_key
    run = production_plan(depth=DEPTH, ratio=20)
    # the repository stage: the fewest trailing files, as many of every repository corpus, holding MARGIN times its draw
    repo = drawn_rows(production, stage=REPO_STAGE, run=run)
    repo_rows = {corpus: np.bincount(row_shards(corpus, "train", production, REPO_STAGE.seq_len)) for corpus in repo}
    trailing = 1
    while sum(int(rows[-trailing:].sum()) for rows in repo_rows.values()) < MARGIN * sum(repo.values()):
        trailing += 1
    chosen = {}
    for corpus, drawn in drawn_rows(production, stage=FILE_STAGE, run=run).items():
        code = corpus in FILE_STAGE.mixture["code"][1]
        rows = np.bincount(row_shards(corpus, "train", production, FILE_STAGE.seq_len))
        # the fewest leading files whose rows hold the draw (code) or MARGIN times it
        files = int(np.searchsorted(np.cumsum(rows), math.ceil(drawn if code else MARGIN * drawn))) + 1
        bounds = shard_documents(corpus, "train", production)
        offsets = np.memmap(split_path(corpus, "train", production).with_suffix(".idx"), dtype=np.int64, mode="r")
        tokens_per_row = offsets[bounds[files]] / rows[:files].sum()
        limit = min(sample_rows(stage) * stage.seq_len / MARGIN for stage in (FILE_STAGE, REPO_STAGE)
                    if corpus in mixture_corpora(stage.mixture))
        val = np.memmap(split_path(corpus, "val", production).with_suffix(".idx"), dtype=np.int64, mode="r")
        train = curated_files(corpus, "train")
        chosen[corpus] = Documents(
            train=train[:files],
            kept=int(np.searchsorted(offsets, drawn * tokens_per_row)) if code else int(bounds[files]),
            repo=train[-trailing:] if corpus in repo else [],
            val_step=math.ceil(val[-1] / limit))
    return chosen


def describe(tokenizer: Tokenizer, chosen: dict[str, Documents]) -> dict:
    """An arm's intrinsic record on its document set: bytes per token of each corpus's train and val documents; the
    Rényi efficiency of each corpus's val documents and of all of them (tokeval.py); its vocabulary tokens
    holding a line break, fusing punctuation with one and ending in a line break and indentation; how many mergeable
    tokens its train documents hold never, under 10 and under 100 times."""
    held = {split: np.zeros(tokenizer.bos_token_id, dtype=np.int64) for split in ("train", "val")}
    lengths = tokenizer.token_lengths[:tokenizer.bos_token_id]
    bytes_per_token, renyi = {}, {}
    for corpus in chosen:
        bytes_per_token[corpus] = {}
        for split, total in held.items():
            tokens = np.memmap(split_path(corpus, split, tokenizer).with_suffix(".bin"), dtype=TOKEN_DTYPE, mode="r")
            counts = np.bincount(tokens, minlength=tokenizer.bos_token_id)
            total += counts
            bytes_per_token[corpus][split] = float(counts @ lengths / len(tokens))
            if split == "val":
                renyi[corpus] = tokeval.renyi_efficiency(counts, tokenizer.vocab_size)
    renyi["all"] = tokeval.renyi_efficiency(held["val"], tokenizer.vocab_size)
    merged = [b.decode("utf-8", "ignore") for b in tokenizer.token_bytes[256:tokenizer.bos_token_id]]
    # a line break and the indentation after it ending the token (\Z: $ also matches before a final line break)
    fused, indented = regex.compile(r"[^\s\p{L}\p{N}][\r\n]"), regex.compile(r"[\r\n][ \t]+\Z")
    never = held["train"]
    return {"data_key": tokenizer.data_key, "bytes_per_token": bytes_per_token, "renyi_efficiency": renyi,
            "line_break_tokens": sum("\n" in t or "\r" in t for t in merged),
            "punctuation_line_break_tokens": sum(bool(fused.search(t)) for t in merged),
            "line_break_indentation_tokens": sum(bool(indented.search(t)) for t in merged),
            "unread_tokens": {"never": int((never == 0).sum()), "under 10": int((never < 10).sum()),
                              "under 100": int((never < 100).sum())}}


def data() -> None:
    """Every arm's compiled document set, checked to hold the same bytes per document in every arm, and
    RESULTS/tokenizer-data.json (describe, and TokEval's metrics: tokeval.code of every code corpus,
    tokeval.operator_isolation of every code and math corpus); every arm's packs and their checks: max_ratio, every
    draw within its readable rows, every val split within bpb's sample."""
    chosen = documents()
    tokenizers = [get_tokenizer(arm) for arm in ARMS]
    for tokenizer in tokenizers:
        for corpus, part in chosen.items():
            split_path(corpus, "val", tokenizer).parent.mkdir(parents=True)
            compile_split(corpus, "val", tokenizer, files=curated_files(corpus, "val"))
            compile_split(corpus, "train", tokenizer, files=part.train + part.repo)
            val = np.arange(int(shard_documents(corpus, "val", tokenizer)[-1]))
            keep_documents(corpus, "val", tokenizer, val % part.val_step == 0)
            bounds = shard_documents(corpus, "train", tokenizer)
            train = np.arange(int(bounds[-1]))
            # the leading files' first kept documents and every document of the trailing (repository-stage) files
            keep_documents(corpus, "train", tokenizer, (train < part.kept) | (train >= bounds[len(part.train)]))
        print(f"{tokenizer.name}: data key {tokenizer.data_key}")
    for corpus, split in itertools.product(chosen, ("train", "val")):
        sizes = []
        for tokenizer in tokenizers:
            path = split_path(corpus, split, tokenizer)
            tokens = np.fromfile(path.with_suffix(".bin"), dtype=TOKEN_DTYPE)
            ends = np.concatenate([[0], np.cumsum(tokenizer.token_lengths[tokens])])
            sizes.append(np.diff(ends[np.fromfile(path.with_suffix(".idx"), dtype=np.int64)]))
        assert all(np.array_equal(sizes[0], other) for other in sizes[1:]), (corpus, split)
    record = {tokenizer.name: describe(tokenizer, chosen) | {"code": {}, "operator_isolation": {}}
              for tokenizer in tokenizers}
    code, math_ = FILE_STAGE.mixture["code"][1], FILE_STAGE.mixture["math"][1]
    for corpus in code + math_:
        tokens, docs = tokeval.val_split(corpus, tokenizers)
        operators = tokeval.operator_isolation(tokens, docs, tokenizers)
        metrics = tokeval.code(corpus, tokens, docs, tokenizers) if corpus in code else None
        for arm in ARMS:
            record[arm]["operator_isolation"][corpus] = operators[arm]
            if metrics is not None:
                record[arm]["code"][corpus] = metrics[arm]
    (RESULTS / "tokenizer-data.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    data_crate()  # built once, before the packer's threads call it
    for tokenizer in tokenizers:
        arm = tokenizer.name
        for corpus, part in chosen.items():
            first, trailing = len(part.train), len(part.repo)
            stages = [(FILE_STAGE, range(first))]
            if trailing:  # a repository corpus
                stages.append((REPO_STAGE, range(first, first + trailing)))
            for stage, shards in stages:
                pack_split(corpus, "val", tokenizer, seq_len=stage.seq_len)
                rows = write(corpus, "train", tokenizer, in_order(corpus, "train", tokenizer, shards,
                                                                  seq_len=stage.seq_len), seq_len=stage.seq_len)
                print(f"{arm} {corpus} T{stage.seq_len}: {len(shards)} train files, {rows:,} rows")
        ratio = max_ratio(tokenizer, depth=DEPTH)
        run = production_plan(depth=DEPTH, ratio=ratio)
        for stage in (FILE_STAGE, REPO_STAGE):
            readable = readable_rows(tokenizer, stage=stage, run=run)
            slack = {corpus: len(readable[corpus]) - drawn
                     for corpus, drawn in drawn_rows(tokenizer, stage=stage, run=run).items()}
            tight = min(slack, key=slack.get)
            fill = {corpus: packed_rows(corpus, "val", tokenizer, stage.seq_len) / sample_rows(stage)
                    for corpus in mixture_corpora(stage.mixture)}
            fullest = max(fill, key=fill.get)
            print(f"{arm} d{DEPTH} at {ratio:.2f} T{stage.seq_len}: least slack {slack[tight]:,} rows ({tight}); "
                  f"val at most {fill[fullest]:.3f} of bpb's sample ({fullest})")


def runs() -> None:
    """Every arm at d16, each at its max_ratio (ablations/harness.py main); then RESULTS/tokenizer.json gains each
    challenger's contrast with sa-bpe and the decision (module docstring)."""
    ratios = {arm: max_ratio(get_tokenizer(arm), depth=DEPTH) for arm in ARMS}
    main(experiment="tokenizer",
         settings=[Setting(depth=DEPTH, ratio=ratios["sa-bpe"], seeds=range(SEEDS),
                           alternatives={arm: {"tokenizer": arm, "ratio": ratios[arm]} for arm in CHALLENGERS})],
         production={"tokenizer": "sa-bpe"})
    parses = {}  # (arm, seed): per problem, in the benchmarks' order, whether its completion compiles
    for line in (RESULTS / "tokenizer.jsonl").read_text(encoding="utf-8").splitlines():
        run = json.loads(line)
        outcomes = run["evaluations"][-1]["parse"].values()
        parses[run["arm"], run["fields"]["seed"]] = [o == "parses" for b in outcomes for o in b["outcomes"]]
    summary = json.loads((RESULTS / "tokenizer.json").read_text(encoding="utf-8"))
    compared = summary["settings"][0]["alternatives"]
    margin = norm.ppf(1 - ALPHA / 2) * math.sqrt(2 / SEEDS) * CODE_SIGMA
    contrasts = {}
    for arm in CHALLENGERS:
        pairs = [(a, b) for seed in range(SEEDS)
                 for a, b in zip(parses[arm, seed], parses["production", seed], strict=True)]
        here, there = sum(a and not b for a, b in pairs), sum(b and not a for a, b in pairs)
        p = binomtest(here, here + there).pvalue if here + there else 1.0
        code = compared[arm]["difference"]["code"]
        contrasts[arm] = {"parses_only_here": here, "parses_only_in_sa-bpe": there, "p": p,
                          "parsing": "equal" if p >= ALPHA else "better" if here > there else "worse",
                          "code": "better" if code < -margin else "worse" if code > margin else "equal"}
    verdicts = {arm: (c["parsing"], c["code"]) for arm, c in contrasts.items()}
    qualified = {arm for arm, v in verdicts.items() if "worse" not in v and "better" in v}
    never = {arm: r["unread_tokens"]["never"]
             for arm, r in json.loads((RESULTS / "tokenizer-data.json").read_text(encoding="utf-8")).items()}
    frequency = "bpe" in qualified or (verdicts["bpe"] == ("equal", "equal") and never["bpe"] < never["sa-bpe"])
    decision = {"pattern": min(qualified & {"sa-bpe-fused", "hybrid"},
                               key=lambda arm: compared[arm]["difference"]["unparsed"], default="sa-bpe"),
                "score": "frequency" if frequency else "sa-bpe",
                "control_detected": verdicts["sa-bpe-fused"][0] == "worse"}
    (RESULTS / "tokenizer.json").write_text(
        json.dumps(summary | {"contrasts": contrasts, "decision": decision}, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m ablations.tokenizer",
                                     description="The tokenizer experiment's steps")
    parser.add_argument("step", choices=("vocabularies", "data", "runs"))
    {"vocabularies": vocabularies, "data": data, "runs": runs}[parser.parse_args().step]()
